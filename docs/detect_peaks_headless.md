# detect_peaks_headless.py

Non-interactive peak detection + identification for production use. Detects and classifies peaks in an APT/CSV spectrum and always writes a predicted `.rrng`; every other output is optional. No plots are produced.

Mirrors the model behavior of `detect_peaks_refactor.py`'s `process_dataset`, minus the evaluation/plotting machinery.

## Usage

```powershell
.venv\Scripts\python.exe detect_peaks_headless.py `
    --input "data\APT_test\R13_40310Zr Unsaved - Top Level ROI.csv" `
    --elements "Zr,O,Ti,ZrO,ZrH" `
    --output-rrng "out\R13_predicted.rrng"

# Expected species parsed from an existing range file instead of a list:
.venv\Scripts\python.exe detect_peaks_headless.py `
    --input "R13.csv" --expected-rrng "R13.RRNG" `
    --output-rrng "out\R13_predicted.rrng"
```

Callable from Python:

```python
from detect_peaks_headless import detect_peaks_headless
ranges = detect_peaks_headless('R13.apt', output_rrng='out.rrng', elements=['Zr', 'O', 'ZrO'])
```

## Arguments

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

No per-model tunables (`--iou`, `--conf`, etc.) exist as CLI flags — model behavior comes entirely from `configs\models\<model>.yaml`, overridable per run via `--config`. See [`RUN_CONFIG.md`](../RUN_CONFIG.md) for the full config-resolution and output-directory reference.

## Notes

- `--elements` and `--expected-rrng` are mutually exclusive and one is required.
- The effective config (merged YAML + this script's `--save-*`/`--progress-min-fraction` flags) is written as `effective_config_<timestamp>.yaml` into the `--output-rrng` file's directory.
