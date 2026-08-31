# scripts/build_truth_molecule_summary.py

Builds a small, repo-committable truth summary from a full `detect_peaks_refactor.py` batch-run output folder: `truth_molecules_canonical_summary.csv` (distinct truth molecules + instance counts) and `merged_true_species.txt` (union of every dataset's `*_true_species.txt`). Lets `generate_synthetic_data.py --results_dir` point at this small summary instead of the full external batch-run directory.

## Usage

```bash
python scripts/build_truth_molecule_summary.py --results_dir "C:\path\to\results-headless-dev"
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--results_dir RESULTS_DIR` | **Yes** | — | Batch-run output folder (contains per-dataset `*/*_detailed_results.csv` and `*/*_true_species.txt`). |
| `--output_dir OUTPUT_DIR` | No | `scripts/synthetic_resources` | Where to write the consolidated truth summary. |

## Notes

- Only molecules (multi-atom labels) are collected into `truth_molecules_canonical_summary.csv`; pure elements are not included there (they come from `merged_true_species.txt` instead, consumed by `generate_synthetic_data.py`'s `load_truth_elements`).
- Imports `generate_synthetic_data.py` directly for its `canonical_formula`/`is_molecule` helpers, so it must be run with `scripts/` importable (e.g. `python scripts/build_truth_molecule_summary.py`, not from another working directory without adjusting `sys.path`).
