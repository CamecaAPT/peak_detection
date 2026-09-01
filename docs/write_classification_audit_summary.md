# write_classification_audit_summary.py

Writes an aggregate element/molecule classification audit summary (`element_molecule_prediction_type_audit_summary.txt`) across an entire `detect_peaks_refactor.py` batch run: RF accuracy before/after molecule rescue, context-rescoring corrections, unknown-prediction breakdowns, and pure-vs-mixed element/molecule prediction correctness.

## Usage

```powershell
.venv\Scripts\python.exe write_classification_audit_summary.py --results_dir "results\bench1"
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--results_dir RESULTS_DIR` | **Yes** | — | Batch-run output folder (contains `peak_detection_summary.csv`, `*_diagnostics.csv`, per-dataset `*_detailed_results.csv`, `*_context_rescore_overrides.csv`, `*_molecule_rescue_candidates.csv`). |
| `--output OUTPUT` | No | `<results_dir>/element_molecule_prediction_type_audit_summary.txt` | Output path for the summary text file. |
| `--rf_accuracy_top_n RF_ACCURACY_TOP_N` | No | read from the newest `effective_config_*.yaml` in `--results_dir` | Number of top-N RF candidates used when judging final correctness. |

## Notes

- Gracefully degrades: missing `peak_detection_summary.csv` / `*_diagnostics.csv` / rescue or context-override CSVs produce explanatory "not found" lines in the summary rather than failing.
- Depends on `report_metrics.py` for shared formatting/truth-splitting helpers — kept dependency-light (pandas only), no `peak_detection` package import required.
