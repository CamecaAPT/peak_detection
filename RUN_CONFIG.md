# Run configuration & output directories

Both peak-detection entry points share one parameter system:

- **`detect_peaks_refactor.py`** — evaluation / benchmarking (range file is *ground truth*; emits metrics, plots, batch summaries).
- **`detect_peaks_headless.py`** — production inference (writes a `.rrng`; expected species supplied, no plots).

The ~39 detection / RF / unknown-flagging / context-rescoring tunables they have in common are defined once in [`peak_detection/run_config.py`](peak_detection/run_config.py) (`SHARED_PARAMS`). That single source of truth generates the CLI arguments for both scripts and the YAML run-config described below.

---

## 1. The run-config YAML

Every run **writes** a `run_config_<timestamp>.yaml` capturing the *effective* parameters, and any run can **load** one with `--config` to reuse those parameters.

It contains:

- every shared parameter (the value actually used),
- the script-specific tunables for that script (see [§4](#4-script-specific-tunables)),
- a `command:` line (the full command, so the I/O paths are recorded), and
- a `timestamp:`.

Keys are sorted alphabetically so two configs diff cleanly. Example (abridged):

```yaml
command: detect_peaks_headless.py --input APT_test/R7001_304617.apt ...
conf: 0.1
context_rescore: false
flag_unknowns: true
include_molecules: false
iou: 0.01
n_iter: 0
save_peak_ranges_txt: true        # headless-specific tunable
separate_molecule_rf: true        # headless-specific tunable
timestamp: '20260625_153901'
training_path: peak_detection/Ionclassifier/training_data/NewData_truthcoverage_lightmol1p_C3_BO_C2O_2p_2026-06-10/Data0001
yolo_weights: best_v0_2026-06-23.pt
```

---

## 2. Loading a config — `--config` and precedence

Resolution order, highest wins:

**CLI flag  >  `--config` YAML value  >  built-in default**

A YAML supplies *defaults*; an explicitly-typed CLI flag still overrides it. Partial configs are fine — pin a few keys and let the rest fall back.

```powershell
# YAML sets conf: 0.1 and training_num_files: 30; the CLI overrides conf only.
.venv\Scripts\python.exe detect_peaks_headless.py `
    --config run_config_20260625_153901.yaml `
    --input "APT_test\R5100_228062 W.apt" --expected-rrng "RRNG_test\R5100_228062 W.RRNG" `
    --output-rrng "out\R5100.rrng" `
    --conf 0.25
# -> conf = 0.25 (CLI), training_num_files = 30 (YAML), everything else = YAML/defaults
```

Notes / guardrails:

- **Required I/O still must be on the CLI.** A YAML can't satisfy required arguments (e.g. headless `--input` / `--output-rrng`) — those are per-run, not part of a reusable config.
- **Unknown keys are ignored** with a notice, so a stale or cross-script config never crashes:
  ```
  [config] ignoring unknown keys from run_config_*.yaml: ['progress_min_fraction', 'save_artifacts', ...]
  ```
- **Booleans** work both ways: a YAML `flag_unknowns: false` is overridable with `--flag-unknowns` / `--no-flag-unknowns`.

---

## 3. Hand-editing a preset

You don't need to start from a generated file — write a partial YAML by hand:

```yaml
# my_preset.yaml
conf: 0.1
include_molecules: true
n_iter: 2
save_plots: false        # refactor tunable
```

```powershell
.venv\Scripts\python.exe detect_peaks_refactor.py --config my_preset.yaml `
    --apt_path "APT_test\R7001_304617.apt" --rrng_path "RRNG_test\R7001_304617.RRNG"
```

---

## 4. Script-specific tunables

Beyond the shared set, each script persists its own **behavior** flags to the YAML (declared as `SCRIPT_CONFIG_KEYS` in each script). Per-run I/O paths are deliberately excluded.

| Script | Persisted tunables |
|--------|--------------------|
| `detect_peaks_refactor.py` | `save_plots`, `save_csv`, `save_rrng_output` |
| `detect_peaks_headless.py` | `save_artifacts`, `save_peak_ranges_txt`, `separate_molecule_rf`, `progress_min_fraction` |

These round-trip like any other key. A config written by one script can be fed to the other — the keys the other script doesn't recognize are skipped with the notice shown above; the shared keys still apply.

```powershell
# Set a headless tunable on the CLI once; it is written to the config...
.venv\Scripts\python.exe detect_peaks_headless.py `
    --input "APT_test\R7001_304617.apt" --expected-rrng "RRNG_test\R7001_304617.RRNG" `
    --output-rrng "out\R7001.rrng" --save-peak-ranges-txt --separate-molecule-rf

# ...then reuse it without re-typing the flags (the .txt is still written).
.venv\Scripts\python.exe detect_peaks_headless.py --config out\run_config_*.yaml `
    --input "APT_test\R5100_228062 W.apt" --expected-rrng "RRNG_test\R5100_228062 W.RRNG" `
    --output-rrng "out\R5100.rrng"
```

---

## 5. Directories

There are three distinct kinds of path. They resolve differently — this is the part most worth understanding.

### a. Script-relative resources (model & default training data)

Resolved relative to the **package location** (the repo root containing `peak_detection/`), so they work no matter which directory you launch from:

- **`--yolo_weights`** — just a filename; loaded from `peak_detection/RangingNN/modelweights/<name>` (default `best_v0_2026-06-23.pt`).
- The model's internal `prediction_args.yaml` under `peak_detection/RangingNN/cfg/`.
- **`--training_path`** — a *relative* value (e.g. the default `peak_detection/Ionclassifier/training_data/…`) is resolved against the package root, so it works from any working directory. Pass an **absolute** path to point elsewhere.

### b. Input paths (you supply these; CWD-relative or absolute)

- refactor: `--apt_path`, `--rrng_path` (a file → single mode; a directory → batch mode).
- headless: `--input`, and expected species via `--elements` or `--expected-rrng`.

### c. Output directory

**`detect_peaks_refactor.py` already has `--output_dir`** — it is the explicit output location for the run, and the run-config YAML is written into it in both modes:

- **Single mode** (`--apt_path` is one file): `--output_dir` *is* the dataset folder. It receives the per-dataset artifacts (detailed-results CSV, the predicted `.rrng` when `--save_rrng_output`, the `*_yolo_args.yaml` call snapshot, etc.) **and** `run_config_<ts>.yaml`. Default when omitted: a folder derived from the APT filename.
- **Batch mode** (`--apt_path` is a directory): `--output_dir` is the **parent** that holds one folder per dataset *plus* the global results — `peak_detection_summary.csv`, `yolo_identifications.csv`, the summary plots, and `run_config_<ts>.yaml`. Default when omitted: the current directory.

```powershell
# Single: all results for this dataset land in results\R7001\
.venv\Scripts\python.exe detect_peaks_refactor.py `
    --apt_path "APT_test\R7001_304617.apt" --rrng_path "RRNG_test\R7001_304617.RRNG" `
    --output_dir "results\R7001" --save_rrng_output

# Batch: per-dataset folders + summary CSV + plots + run_config.yaml all under results\bench1\
.venv\Scripts\python.exe detect_peaks_refactor.py `
    --apt_path "APT_test" --rrng_path "RRNG_test" --output_dir "results\bench1"
```

**`detect_peaks_headless.py` does not use a single output-directory flag** — by design it has:

- **`--output-rrng`** — the explicit path of the result range file (the one guaranteed output). The `run_config_<ts>.yaml` is written into **this file's directory**, so the parameters sit next to the result.
- **`--artifacts-dir`** — optional folder for *diagnostic* CSVs (only when `--save-artifacts`) and the per-call `*_yolo_args.yaml` snapshot. **When omitted it defaults to the `--output-rrng` directory**, so those files sit next to the result rather than in a stray folder.

```powershell
# out\R7001.rrng + out\run_config_<ts>.yaml are written together.
.venv\Scripts\python.exe detect_peaks_headless.py `
    --input "APT_test\R7001_304617.apt" --expected-rrng "RRNG_test\R7001_304617.RRNG" `
    --output-rrng "out\R7001.rrng"
```

### Summary — where each output lands

| Output | refactor | headless |
|--------|----------|----------|
| Result range file | per-dataset folder (`--save_rrng_output`) | `--output-rrng` (always) |
| Run-config YAML | `--output_dir` (single = dataset folder; batch = parent) | directory of `--output-rrng` |
| Batch summary CSV + plots + identifications | `--output_dir` (batch only) | n/a |
| Per-call YOLO args snapshot (`*_yolo_args.yaml`) | per-dataset folder | `--artifacts-dir`, else the `--output-rrng` directory |
| Diagnostic CSVs | per-dataset folder | `--artifacts-dir` (with `--save-artifacts`), else the `--output-rrng` directory |
