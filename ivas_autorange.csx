// ivas_autorange.csx — IVAS Script Console workflow.
//
// Re-ranges the SELECTED Mass Spectrum Analysis using the peak_detection model:
//   1. seed the expected ions from the top-level range file,
//   2. choose an optional RunConfig YAML to preload parameters from,
//   3. review/override the parameters in a property grid,
//   4. export the ions to APT (into the temp output dir; the model re-histograms via apav),
//   5. run detect_peaks_headless.py in the project's venv,
//   6. load the generated .rrng back onto the selected node.
//
// Load it via the console's "Load…" button (or "Run script…"), then Run (or Ctrl+Enter).
// When loaded from disk, ScriptDirectory is this file's folder, so extRoot and the default
// RunConfig resolve against it automatically.
//
// NOTE: updated for the Phase-2 contract reshape (Cameca.Scripting.Contracts) — the selected node and
// expected elements are read via async methods, POCO settings go through Settings.ReviewAsync, and
// ScriptDirectory / RunProcessAsync are globals-level members. See scripting-host-plan.md.

using System.ComponentModel;
using System.Globalization;
using System.Collections.Generic;
using System.IO;

// ---- Stage 1: choose the RunConfig YAML to preload from ----
class ConfigChoice
{
	[Category("Run config"), DisplayName("RunConfig YAML"),
	 Description("Path to a RunConfig YAML to preload parameters from. Relative paths resolve " +
	             "against the project root. If the file does not exist, the script defaults below " +
	             "are used.  (passed to the tool as --config)")]
	public string RunConfigPath { get; set; } = "defaultRunConfig.yaml";
}

// ---- Stage 2: settings surfaced in a property grid before the run (edit, then Resume) ----
class RangingSettings
{
	[Category("Identification"), DisplayName("Range molecules"),
	 Description("Identify molecular ions (e.g. ZrO), not just elements.  (--include-molecules)")]
	public bool RangeMolecules { get; set; } = false;

	[Category("Identification"), DisplayName("Flag unknowns"),
	 Description("Mark low-confidence peaks as Unknown.  (--flag-unknowns)")]
	public bool FlagUnknowns { get; set; } = true;

	[Category("Identification"), DisplayName("MC threshold"),
	 Description("Confidence threshold below which a peak is flagged unknown.  (--mc-threshold)")]
	public double McThreshold { get; set; } = 0.2;

	[Category("Molecule RF"), DisplayName("Unknown molecule RF"),
	 Description("Run the random-forest pass for unknown molecules.  (--unknown-molecule-rf)")]
	public bool UnknownMoleculeRf { get; set; } = true;

	[Category("Molecule RF"), DisplayName("Unknown molecule RF threshold"),
	 Description("Probability threshold for the molecule RF.  (--unknown-molecule-rf-threshold)")]
	public double UnknownMoleculeRfThreshold { get; set; } = 0.8;

	[Category("Molecule RF"), DisplayName("Molecule RF rescue elements"),
	 Description("Run a molecule-only RF pass on element-labeled peaks and allow molecule overrides / mixed top-2.  (--molecule-rf-rescue-elements)")]
	public bool MoleculeRfRescueElements { get; set; } = true;

	[Category("Identification"), DisplayName("Context rescore"),
	 Description("Use nearby peak labels to rescore ambiguous RF candidates after initial classification.  (--context-rescore)")]
	public bool ContextRescore { get; set; } = true;

	[Category("Model"), DisplayName("YOLO weights"),
	 Description("Weights filename in peak_detection/RangingNN/modelweights.  (--yolo-weights)")]
	public string YoloWeights { get; set; } = "best_v0_2026-06-23.pt";

	[Category("Model"), DisplayName("YOLO iterations"),
	 Description("Number of YOLO recursive detection iterations. 0 = single pass.  (--n-iter)")]
	public int NIter { get; set; } = 0;

	[Category("Model"), DisplayName("Augment molecule training charge ratios"),
	 Description("Augment RF training with extra molecule charge-state ratios.  (--augment-molecule-training-charge-ratios)")]
	public bool AugmentMoleculeTrainingChargeRatios { get; set; } = true;

	[Category("Model"), DisplayName("Training num files"),
	 Description("Number of classifier training files to load.  (--training-num-files)")]
	public int TrainingNumFiles { get; set; } = 5000;

	[Category("Model"), DisplayName("Training path"),
	 Description("Classifier training-data directory, relative to the project root.  (--training-path)")]
	public string TrainingPath { get; set; } =
		"peak_detection/Ionclassifier/training_data/NewData_truthcoverage_lightmol1p_C3_BO_C2O_2p_2026-06-10/Data0001";

	[Category("Output"), DisplayName("Progress update fraction"),
	 Description("Throttle training-data progress bars to ~one update per this fraction of progress " +
	             "(0.2 = every 20%). 0 = continuous (tool default).  (--progress-min-fraction)")]
	public double ProgressUpdateFraction { get; set; } = 0.2;

	[Category("Output"), DisplayName("Save artifacts"),
	 Description("Write per-dataset diagnostic CSVs (detailed results, unknown report).  (--save-artifacts)")]
	public bool SaveArtifacts { get; set; } = false;

	[Category("Output"), DisplayName("Save peak ranges txt"),
	 Description("Also write a plain-text peak_ranges.txt next to the result.  (--save-peak-ranges-txt)")]
	public bool SavePeakRangesTxt { get; set; } = false;
}

// argparse BooleanOptionalAction => pass --flag or --no-flag explicitly (don't rely on the py default).
string Flag(string name, bool on) => on ? $"--{name}" : $"--no-{name}";
string Num(double v) => v.ToString(CultureInfo.InvariantCulture);

// Minimal parser for the flat scalar `key: value` RunConfig YAML written by the tool.
// (sort_keys=True, no nesting; multi-line/indented continuation lines like `command:` are skipped.)
Dictionary<string, string> ParseRunConfigYaml(string path)
{
	var d = new Dictionary<string, string>(StringComparer.Ordinal);
	foreach (var raw in File.ReadAllLines(path))
	{
		if (raw.Length == 0 || raw[0] == ' ' || raw[0] == '\t' || raw[0] == '#') continue; // skip blanks/continuations/comments
		int colon = raw.IndexOf(':');
		if (colon <= 0) continue;
		var key = raw.Substring(0, colon).Trim();
		var val = raw.Substring(colon + 1).Trim();
		if (val.Length >= 2 &&
		    ((val[0] == '\'' && val[val.Length - 1] == '\'') || (val[0] == '"' && val[val.Length - 1] == '"')))
			val = val.Substring(1, val.Length - 2);
		d[key] = val;
	}
	return d;
}

bool YBool(string v) { var t = v.Trim().ToLowerInvariant(); return t == "true" || t == "yes" || t == "1"; }
double YDouble(string v) => double.Parse(v, CultureInfo.InvariantCulture);
int YInt(string v) => int.Parse(v, CultureInfo.InvariantCulture);

// Overlay values parsed from a RunConfig YAML onto the settings shown in the property grid.
// Only keys the grid exposes are mapped here; every other key in the YAML is still applied by
// the Python tool via --config (the grid edits below simply override the matching ones).
void ApplyRunConfig(RangingSettings s, Dictionary<string, string> c)
{
	string v;
	if (c.TryGetValue("include_molecules", out v)) s.RangeMolecules = YBool(v);
	if (c.TryGetValue("flag_unknowns", out v)) s.FlagUnknowns = YBool(v);
	if (c.TryGetValue("mc_threshold", out v)) s.McThreshold = YDouble(v);
	if (c.TryGetValue("unknown_molecule_rf", out v)) s.UnknownMoleculeRf = YBool(v);
	if (c.TryGetValue("unknown_molecule_rf_threshold", out v)) s.UnknownMoleculeRfThreshold = YDouble(v);
	if (c.TryGetValue("molecule_rf_rescue_elements", out v)) s.MoleculeRfRescueElements = YBool(v);
	if (c.TryGetValue("context_rescore", out v)) s.ContextRescore = YBool(v);
	if (c.TryGetValue("yolo_weights", out v)) s.YoloWeights = v;
	if (c.TryGetValue("n_iter", out v)) s.NIter = YInt(v);
	if (c.TryGetValue("augment_molecule_training_charge_ratios", out v)) s.AugmentMoleculeTrainingChargeRatios = YBool(v);
	if (c.TryGetValue("training_num_files", out v)) s.TrainingNumFiles = YInt(v);
	if (c.TryGetValue("training_path", out v)) s.TrainingPath = v;
	if (c.TryGetValue("progress_min_fraction", out v) && v.ToLowerInvariant() != "null") s.ProgressUpdateFraction = YDouble(v);
	if (c.TryGetValue("save_artifacts", out v)) s.SaveArtifacts = YBool(v);
	if (c.TryGetValue("save_peak_ranges_txt", out v)) s.SavePeakRangesTxt = YBool(v);
}

// ---- 1. Require a selected Mass Spectrum Analysis ----
if (await Api.GetSelectedMassSpectrumAsync() is not {} ms) { Print("Select a Mass Spectrum Analysis first."); return; }

// ---- Paths (cwd = project root so the tool's relative weights/training paths resolve) ----
// When the script is loaded from disk, extRoot is its own folder; the fallback covers REPL runs.
var extRoot    = ScriptDirectory ?? @"C:\workspace\extensions\peak_detection";
var venvPython = Path.Combine(extRoot, @".venv\Scripts\python.exe");
var pyScript   = Path.Combine(extRoot, "detect_peaks_headless.py");

// Temp output directory: the APT export, the output .rrng, and the run-config snapshot the
// tool writes all live here (the tool writes run_config_*.yaml next to --output-rrng).
var outputDir  = @"C:\temp\ranging";
Directory.CreateDirectory(outputDir);
var aptPath    = Path.Combine(outputDir, "spectrum.apt");
var rngPath    = Path.Combine(outputDir, "result.rrng");

// ---- 2. Expected ions come from the TOP-LEVEL range file, supplied as a list ----
string[] elements = await ms.GetRootExpectedElementsAsync();
if (elements.Length == 0) { Print("The top-level range file has no ion definitions to seed from."); return; }
Print($"Expected ions ({elements.Length}): {string.Join(",", elements)}");

// ---- 3. Choose the RunConfig YAML, then preload its parameters ----
var cc = await Settings.ReviewAsync(new ConfigChoice(), "Choose RunConfig YAML");
var configPath = (cc.RunConfigPath ?? "").Trim();
if (configPath.Length > 0 && !Path.IsPathRooted(configPath))
	configPath = Path.Combine(extRoot, configPath);
bool configExists = configPath.Length > 0 && File.Exists(configPath);

var settings = new RangingSettings();
if (configExists)
{
	Print($"Loading RunConfig: {configPath}");
	try
	{
		var loaded = ParseRunConfigYaml(configPath);
		ApplyRunConfig(settings, loaded);
		Print($"  Loaded {loaded.Count} parameters from RunConfig (review/override below).");
	}
	catch (Exception ex)
	{
		Print($"  [Warn] Failed to parse RunConfig ({ex.Message}); using script defaults.");
		configExists = false;
	}
}
else
{
	Print(configPath.Length == 0
		? "No RunConfig specified; using script defaults."
		: $"RunConfig not found ({configPath}); using script defaults.");
}

// ---- 4. Review/adjust the (possibly preloaded) settings ----
var s = await Settings.ReviewAsync(settings,
	configExists ? "Review parameters (loaded from RunConfig)" : "Review parameters (script defaults)");

// ---- 5. Export APT for the model to re-histogram ----
var apt = await ms.ExportAptAsync(aptPath);
if (!apt.Ok) { Print($"APT export failed: {apt.Message}"); return; }
Print($"Exported {aptPath}");

// ---- 6. Run detect_peaks_headless.py in the project venv ----
// When a RunConfig was loaded it is passed via --config (so parameters NOT shown in the grid
// still apply); the explicit args below override the matching keys per CLI > config precedence.
var args = new List<string>
{
	"-u",            // unbuffered stdout so progress streams live instead of arriving at the end
	pyScript,
};
if (configExists)
{
	args.Add("--config");
	args.Add(configPath);
}
args.AddRange(new[]
{
	"--input", aptPath,
	"--elements", string.Join(",", elements),
	"--output-rrng", rngPath,
	"--yolo-weights", s.YoloWeights,
	"--n-iter", s.NIter.ToString(CultureInfo.InvariantCulture),
	"--training-path", s.TrainingPath,
	"--training-num-files", s.TrainingNumFiles.ToString(CultureInfo.InvariantCulture),
	"--mc-threshold", Num(s.McThreshold),
	"--unknown-molecule-rf-threshold", Num(s.UnknownMoleculeRfThreshold),
	Flag("include-molecules", s.RangeMolecules),
	Flag("flag-unknowns", s.FlagUnknowns),
	Flag("unknown-molecule-rf", s.UnknownMoleculeRf),
	Flag("augment-molecule-training-charge-ratios", s.AugmentMoleculeTrainingChargeRatios),
	Flag("molecule-rf-rescue-elements", s.MoleculeRfRescueElements),
	Flag("context-rescore", s.ContextRescore),
	Flag("save-artifacts", s.SaveArtifacts),
	Flag("save-peak-ranges-txt", s.SavePeakRangesTxt),
});
if (s.ProgressUpdateFraction > 0)   // 0 => omit, tool keeps its continuous default
{
	args.Add("--progress-min-fraction");
	args.Add(Num(s.ProgressUpdateFraction));
}

Print("Running peak detection… (model load + inference can take a while)");
var result = await RunProcessAsync(venvPython, args, extRoot,
	onOutput: line => Print(line),
	onError:  line => Print($"[err] {line}"));   // stderr streamed live too
if (!result.Ok) { Print($"Peak detection failed (exit {result.ExitCode})."); return; }

// ---- 7. Load the generated ranges back onto the selected node ----
await ms.LoadRangeFileAsync(rngPath);
Print($"Loaded {rngPath} into the selected Mass Spectrum Analysis.");
