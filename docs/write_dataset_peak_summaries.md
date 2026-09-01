# write_dataset_peak_summaries.py

Writes a per-dataset, human-readable text summary (`<dataset>_peak_summary.txt`) of peak-detection and classification results for every dataset folder in a completed `detect_peaks_refactor.py` batch run. Called automatically at the end of batch mode, but can be re-run standalone against an existing results directory.

## Usage

```powershell
.venv\Scripts\python.exe write_dataset_peak_summaries.py --results_dir "results\bench1"
```

Callable from Python:

```python
from pathlib import Path
from write_dataset_peak_summaries import write_dataset_peak_summaries
written = write_dataset_peak_summaries(Path("results/bench1"))
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--results_dir RESULTS_DIR` | **Yes** | — | Batch-run output folder containing `peak_detection_summary.csv` and per-dataset subfolders. |

## Notes

- Requires `peak_detection_summary.csv` in `--results_dir`; raises `FileNotFoundError` if missing.
- For each dataset row, reads `<dataset>/<dataset>_detailed_results.csv`; datasets missing that file are silently skipped.
- Depends on `report_metrics.py` for shared formatting helpers — kept dependency-light (pandas only) so it doesn't need to import the full `peak_detection` ML stack.
