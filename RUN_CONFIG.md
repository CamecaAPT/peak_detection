# Run configuration & output directories

Both peak-detection entry points share one parameter system:

- **`detect_peaks_refactor.py`** — evaluation / benchmarking (range file is *ground truth*; emits metrics, plots, batch summaries).
- **`detect_peaks_headless.py`** — production inference (writes a `.rrng`; expected species supplied, no plots).

The ~39 detection / RF / unknown-flagging / context-rescoring tunables they have in common live in the **`configs/`** folder, not as CLI flags:

- [`configs/models/<model>.yaml`](configs/models/rf.yaml) — a fully self-contained per-model config (ranging params + RF tunables + every guardrail this model uses).

`peak_detection/classifiers/config.py::load_merged_config()` performs the merge; `peak_detection/IonIdentificationModels/RF/rf_pipeline.py::flat_rf_kwargs()` flattens the merged dict into the flat keyword names each entry point script's own function (`process_dataset` in `detect_peaks_refactor.py`, `detect_peaks_headless` in `detect_peaks_headless.py`) expects. **There are no per-model CLI flags** (no `--yolo_weights`, `--iou`, etc.) — to change a tunable, edit the relevant yaml or pass `--config path/to/override.yaml`.

---

## 1. The effective-config YAML

Every run **writes** an `effective_config_<timestamp>.yaml` capturing the merged parameters actually used, for provenance/reproducibility (not for re-tuning via CLI overrides — see below).

It contains:

- the fully-merged config (nested: `ranging`, `training`, `guardrails.*`),
- an `output_control` block with that script's output-control flags (see [§4](#4-script-specific-tunables)),
- a `command:` line (the full command, so the I/O paths are recorded), and
- a `timestamp:`.

---

## 2. Overriding tunables — `--config`

Resolution order, highest wins:

**`--config` YAML  >  `configs/models/<model>.yaml`**

There's no CLI-flag layer above the config files anymore — pass `--config` pointing at a YAML with just the keys you want to change (partial overrides are fine; only the keys present are merged in):

```yaml
# my_override.yaml
ranging:
  conf: 0.25
training:
  training_num_files: 30
```

```powershell
.venv\Scripts\python.exe detect_peaks_headless.py `
    --config my_override.yaml `
    --input "APT_test\R5100_228062 W.apt" --expected-rrng "RRNG_test\R5100_228062 W.RRNG" `
    --output-rrng "out\R5100.rrng"
# -> conf = 0.25 and training_num_files = 30 from my_override.yaml, everything else from
#    configs/models/rf.yaml
```

To change a default for every future run instead of a one-off override, edit `configs/models/rf.yaml` directly.

---

## 3. Hand-editing a preset

Same idea as an override file, just checked in / reused across runs:

```yaml
# my_preset.yaml
ranging:
  conf: 0.1
training:
  include_molecules: true
```

```powershell
.venv\Scripts\python.exe detect_peaks_refactor.py --config my_preset.yaml `
    --apt_path "APT_test\R7001_304617.apt" --rrng_path "RRNG_test\R7001_304617.RRNG"
```

---

## 4. Script-specific tunables

Beyond the shared config, each script persists its own **output-control** flags into the effective-config YAML's `output_control` block (declared as `SCRIPT_CONFIG_KEYS` in each script). Per-run I/O paths are deliberately excluded. These remain ordinary CLI flags (they control what gets written, not model behavior):

| Script | Output-control flags |
|--------|--------------------|
| `detect_peaks_refactor.py` | `--save_plots`, `--save_csv`, `--save_rrng_output` |
| `detect_peaks_headless.py` | `--save-artifacts`, `--save-peak-ranges-txt`, `--progress-min-fraction` |

---

## 5. Directories

There are three distinct kinds of path. They resolve differently — this is the part most worth understanding.

### a. Script-relative resources (model & default training data)

Resolved relative to the **package location** (the repo root containing `peak_detection/`), so they work no matter which directory you launch from:

- **`yolo_weights`** (in `configs/models/<model>.yaml`) — just a filename; loaded from `peak_detection/RangingModels/RangingNN/modelweights/<name>` (default `best_v0_2026-06-23.pt`).
- The model's internal `prediction_args.yaml` under `peak_detection/RangingModels/RangingNN/cfg/`.
- **`training_path`** (in `configs/models/rf.yaml`) — a *relative* value is resolved against the package root, so it works from any working directory. Use an **absolute** path to point elsewhere.

### b. Input paths (you supply these; CWD-relative or absolute)

- refactor: `--apt_path`, `--rrng_path` (a file → single mode; a directory → batch mode).
- headless: `--input`, and expected species via `--elements` or `--expected-rrng`.

### c. Output directory

**`detect_peaks_refactor.py` already has `--output_dir`** — it is the explicit output location for the run, and the effective-config YAML is written into it in both modes:

- **Single mode** (`--apt_path` is one file): `--output_dir` *is* the dataset folder. It receives the per-dataset artifacts (detailed-results CSV, the predicted `.rrng` when `--save_rrng_output`, the `*_yolo_args.yaml` call snapshot, etc.) **and** `effective_config_<ts>.yaml`. Default when omitted: a folder derived from the APT filename.
- **Batch mode** (`--apt_path` is a directory): `--output_dir` is the **parent** that holds one folder per dataset *plus* the global results — `peak_detection_summary.csv`, `yolo_identifications.csv`, the summary plots, and `effective_config_<ts>.yaml`. Default when omitted: the current directory.

```powershell
# Single: all results for this dataset land in results\R7001\
.venv\Scripts\python.exe detect_peaks_refactor.py `
    --apt_path "APT_test\R7001_304617.apt" --rrng_path "RRNG_test\R7001_304617.RRNG" `
    --output_dir "results\R7001" --save_rrng_output

# Batch: per-dataset folders + summary CSV + plots + effective_config.yaml all under results\bench1\
.venv\Scripts\python.exe detect_peaks_refactor.py `
    --apt_path "APT_test" --rrng_path "RRNG_test" --output_dir "results\bench1"
```

**`detect_peaks_headless.py` does not use a single output-directory flag** — by design it has:

- **`--output-rrng`** — the explicit path of the result range file (the one guaranteed output). The `effective_config_<ts>.yaml` is written into **this file's directory**, so the parameters sit next to the result.
- **`--artifacts-dir`** — optional folder for *diagnostic* CSVs (only when `--save-artifacts`) and the per-call `*_yolo_args.yaml` snapshot. **When omitted it defaults to the `--output-rrng` directory**, so those files sit next to the result rather than in a stray folder.

```powershell
# out\R7001.rrng + out\effective_config_<ts>.yaml are written together.
.venv\Scripts\python.exe detect_peaks_headless.py `
    --input "APT_test\R7001_304617.apt" --expected-rrng "RRNG_test\R7001_304617.RRNG" `
    --output-rrng "out\R7001.rrng"
```

### Summary — where each output lands

| Output | refactor | headless |
|--------|----------|----------|
| Result range file | per-dataset folder (`--save_rrng_output`) | `--output-rrng` (always) |
| Effective-config YAML | `--output_dir` (single = dataset folder; batch = parent) | directory of `--output-rrng` |
| Batch summary CSV + plots + identifications | `--output_dir` (batch only) | n/a |
| Per-call YOLO args snapshot (`*_yolo_args.yaml`) | per-dataset folder | `--artifacts-dir`, else the `--output-rrng` directory |
| Diagnostic CSVs | per-dataset folder | `--artifacts-dir` (with `--save-artifacts`), else the `--output-rrng` directory |
