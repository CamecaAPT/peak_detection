# peak_detection
For APT spectrum peak ranging and identification

Module RangingNN contains a supervised YOLO-based model for ranging the APT M/C specturm. It was trained on expert labeled datasets. 

Module Ionclassifier contains a supervised Recurrent CNN model for identify the ion species of peaks. It was trained on synthetic datasets. 

### Project structure

---
```
peak_detection/
├── detect_peaks_headless.py       # CLI: production inference (writes .rrng, no plots)
├── detect_peaks_refactor.py       # CLI: evaluation/benchmarking (range file is ground truth)
├── RUN_CONFIG.md                  # Full config/CLI reference (flags, --config overrides, output paths)
├── docs/TOC.md                    # Per-script docs index (every run script + its args)
├── scripts/                       # One-off/maintenance CLI scripts (synthetic data, migrations, review tools)
├── configs/
│   └── models/
│       └── rf.yaml                # Self-contained RF model config (ranging + training + guardrails)
├── data/                          # Sample APT/RRNG test data
└── peak_detection/                # Main package
    ├── yolo_detection.py          # Model-agnostic YOLO1D peak ranging (run_yolo_ranging)
    ├── models.py                  # PeakRange, DetailedId, DatasetStats dataclasses
    ├── data_io.py                 # APT/RRNG file I/O
    ├── training.py                # Classifier training-data loading
    ├── utils/                     # Shared helpers (label parsing, IoU, config coercion)
    ├── guardrail.py                # Shared, model-agnostic guardrails (unknown-flagging, context-rescore, ...)
    ├── registry/                   # Pluggable classifier registry, selected via --model
    │   ├── base.py                 # ClassifierContext, ClassifierPipeline ABC
    │   └── config.py                # Per-model YAML config loader
    ├── IonIdentificationModels/
    │   ├── RF/                     # Random-forest species classifier (registers "rf")
    │   │   ├── rf_model.py          # Underlying RF train/infer implementation
    │   │   ├── rf_pipeline.py       # RFClassifierPipeline orchestrator
    │   │   └── molecule_rescue.py   # RF-specific molecule-rescue guardrails
    │   └── Ionclassifier/           # RNN-based species classifier (not currently wired into --model)
    └── RangingModels/
        └── RangingNN/               # YOLO1D ranging model, weights, and predictor
```

### Command-line interface

---
Two entry points share one parameter system — see [`docs/TOC.md`](docs/TOC.md) for a per-script doc of every run script in the repo (usage + full argument reference), including the maintenance/data-generation scripts under `scripts/`.

**`detect_peaks_refactor.py`** — evaluation/benchmarking (the range file is treated as ground truth; emits metrics, plots, batch summaries):
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

**`detect_peaks_headless.py`** — production inference (writes a `.rrng`; expected species supplied directly or via a range file; no plots):
```powershell
.venv\Scripts\python.exe detect_peaks_headless.py `
    --input "data\APT_test\R13_40310Zr Unsaved - Top Level ROI.csv" `
    --elements "Zr,O,Ti,ZrO,ZrH" `
    --output-rrng "results\R13_predicted.rrng"
```

No per-model tunables (`--iou`, `--conf`, etc.) exist as flags on either script — model behavior comes entirely from `configs\models\<model>.yaml`, overridable per run via `--config`.

#### `detect_peaks_refactor.py` parameters

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

#### `detect_peaks_headless.py` parameters

| Flag | Required | Default | Description |
|---|---|---|---|
| `--config CONFIG` | No | none | Path to a YAML overriding values from `configs\models\<model>.yaml`. Required I/O flags below must still be supplied on the command line. |
| `--model {rf}` | No | `rf` | Classification model to use (see `configs/models/`). |
| `--input INPUT` | **Yes** | — | Path to the input `.apt` or `.csv` file. |
| `--output-rrng OUTPUT_RRNG` | **Yes** | — | Path for the output range (`.rrng`) file — the one guaranteed output. |
| `--elements ELEMENTS` | Exactly one of `--elements` / `--expected-rrng` | — | Comma-separated expected species, elements and/or molecules (e.g. `'Zr,O,Ti,ZrO,ZrH'`). |
| `--expected-rrng EXPECTED_RRNG` | Exactly one of `--elements` / `--expected-rrng` | — | A range file to parse expected species/elements from instead. |
| `--artifacts-dir ARTIFACTS_DIR` | No | directory of `--output-rrng` | Directory for optional diagnostic artifacts. |
| `--save-artifacts` / `--no-save-artifacts` | No | `--no-save-artifacts` (off) | Write per-dataset diagnostic CSVs (detailed results, unknown-peak error report). |
| `--save-peak-ranges-txt` / `--no-save-peak-ranges-txt` | No | `--no-save-peak-ranges-txt` (off) | Also write a plain-text `peak_ranges.txt` next to the result. |
| `--save-rrng-with-uncertainty` / `--no-save-rrng-with-uncertainty` | No | `--no-save-rrng-with-uncertainty` (off) | Use the top-two identification format (`Name:{el1}:{conf1}%-{el2}:{conf2}%`) in the output `.rrng` file. |
| `--progress-min-fraction PROGRESS_MIN_FRACTION` | No | continuous updates | Throttle training-data progress bars to ~one update per this fraction of progress (e.g. `0.2` = every 20%). |

### Installation 

---
At the in-development stage, please install the package from the github source code:

pip install git+https://github.com/wdwzyyg/peak_detection.git

Python version 3.10.0 is recommended. Older python version may not support the pytorch version used here, and newer version has not been tested. 
You can create an independent python environment by running the command:
```
conda create -n name_of_environment python=3.10.0
conda activate name_of_environment
pip install git+https://github.com/wdwzyyg/peak_detection.git
...
```

### Usage 

---
#### Using the RangingNN and IonClassifier models

Use ML models to predict APT peak ranges and ion types:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor.ipynb)

Dev notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2025-10-14.ipynb)

Dev notebook 11/5/25 (example of using per-spectrum RF):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2025_11_5.ipynb)

Dev notebook 11/18/25 (example of using updated ranging model with iterative peak finding):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2025_11_18.ipynb)

Dev notebook 3/16/26 (example of using new CLI-based code detect_peaks.py):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2026_03_16.ipynb)

Ranging model data creation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/DataCreation.ipynb)
