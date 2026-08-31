# detect_peaks_refactor.py

Evaluation/benchmarking entry point — the range file is treated as ground truth. Runs YOLO peak detection + RF (or another registered model) classification, then emits accuracy metrics, comparison plots, and batch summaries.

## Usage

```powershell
# Single dataset
.venv\Scripts\python.exe detect_peaks_refactor.py `
    --apt_path "data\APT_test\R13_40310Zr Unsaved - Top Level ROI.csv" `
    --rrng_path "data\RRNG_test\R13_40310Zr Top Level ROI.RRNG" `
    --output_dir "results\R13"

# Batch mode (point at directories instead of files)
.venv\Scripts\python.exe detect_peaks_refactor.py `
    --apt_path "data\APT_test" --rrng_path "data\RRNG_test" --output_dir "results\bench1"
```

Callable from Python:

```python
from detect_peaks_refactor import process_dataset
stats = process_dataset('data.csv', 'data.RRNG')
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--config CONFIG` | No | none | Path to a YAML overriding values from `configs\models\<model>.yaml`. |
| `--model {rf}` | No | `rf` | Classification model to use (see `configs/models/`). |
| `--apt_path APT_PATH` | No | `ALL_APT_processedCSV` | Path to a `.apt`/`.csv` file (single-dataset mode) or a directory (batch mode). |
| `--rrng_path RRNG_PATH` | No | `ALL_RRNG_NEW` | Path to a `.rrng` file (single mode) or a directory (batch mode). |
| `--output_dir OUTPUT_DIR` | No | derived from `--apt_path` (single) / current directory (batch) | Single mode: the dataset folder. Batch mode: the parent folder holding per-dataset folders plus the global summary CSV, identifications, and plots. |
| `--save_plots` / `--no-save_plots` | No | `--save_plots` (on) | Write comparison plots. |
| `--save_rrng_output` / `--no-save_rrng_output` | No | `--no-save_rrng_output` (off) | Write the predicted ranges as a `.RRNG` file. |
| `--save-rrng-with-uncertainty` / `--no-save-rrng-with-uncertainty` | No | `--no-save-rrng-with-uncertainty` (off) | Use the top-two identification format (`Name:{el1}:{conf1}%-{el2}:{conf2}%`) for the predicted `.RRNG` file. Requires `--save_rrng_output`. |
| `--save_csv` / `--no-save_csv` | No | `--save_csv` (on) | Write per-peak/per-dataset result CSVs (detailed results, summary, identifications). |

No per-model tunables (`--iou`, `--conf`, etc.) exist as CLI flags — model behavior comes entirely from `configs\models\<model>.yaml`, overridable per run via `--config`. See [`RUN_CONFIG.md`](../RUN_CONFIG.md) for the full config-resolution and output-directory reference.

## Notes

- In batch mode, datasets under `--apt_path`/`--rrng_path` are matched by filename (`match_datasets`); unmatched files are logged and skipped.
- Batch mode also writes `peak_detection_summary.csv`, `<model>_diagnostics.csv` (when the pipeline populates diagnostics), `yolo_identifications.csv`, per-dataset peak summaries (via `write_dataset_peak_summaries.py`), and a set of summary plots into `--output_dir`.
- The effective config is written as `effective_config_<timestamp>.yaml` into `--output_dir`.
