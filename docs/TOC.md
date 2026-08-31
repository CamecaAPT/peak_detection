# Script documentation index

One page per run script — usage, arguments, and notes. See [`RUN_CONFIG.md`](../RUN_CONFIG.md) for the shared config/tunable system used by the two main entry points.

## Main entry points

| Script | Purpose | Docs |
|---|---|---|
| `detect_peaks_headless.py` | Production inference: detect + identify peaks, write a `.rrng` | [docs](detect_peaks_headless.md) |
| `detect_peaks_refactor.py` | Evaluation/benchmarking against a ground-truth range file | [docs](detect_peaks_refactor.md) |

## Report/summary writers

| Script | Purpose | Docs |
|---|---|---|
| `write_dataset_peak_summaries.py` | Per-dataset peak-detection/classification text summary from a completed run | [docs](write_dataset_peak_summaries.md) |
| `write_classification_audit_summary.py` | Aggregate element/molecule classification audit summary across a batch run | [docs](write_classification_audit_summary.md) |

## Model training

| Script | Purpose | Docs |
|---|---|---|
| `peak_detection/RangingModels/RangingNN/run_train.py` | Augment/split raw APT+RRNG data and train the RangingNN (YOLO1D) ranging model | [docs](run_train.md) |

## Data / training-set generation (`scripts/`)

| Script | Purpose | Docs |
|---|---|---|
| `scripts/generate_synthetic_data.py` | Generate synthetic ion-classifier training CSVs with truth coverage | [docs](generate_synthetic_data.md) |
| `scripts/build_truth_molecule_summary.py` | Build a consolidated truth-molecule summary from a batch-run folder | [docs](build_truth_molecule_summary.md) |
| `scripts/generate_periodic_table.py` | Regenerate `data/periodic_table.json` from `data/mass.txt` | [docs](generate_periodic_table.md) |

## Maintenance / migration (`scripts/`)

| Script | Purpose | Docs |
|---|---|---|
| `scripts/migrate_ionclassifier_checkpoint.py` | One-time re-pickle of the Ionclassifier checkpoint under its current module path | [docs](migrate_ionclassifier_checkpoint.md) |
| `scripts/migrate_rangingnn_checkpoints.py` | One-time re-pickle of RangingNN `.pt` checkpoints under their current module path | [docs](migrate_rangingnn_checkpoints.md) |
| `scripts/regenerate_mixed_label_plots.py` | Rebuild saved YOLO comparison PNGs from existing detailed-results CSVs | [docs](regenerate_mixed_label_plots.md) |

## Interactive / review tools (`scripts/`)

| Script | Purpose | Docs |
|---|---|---|
| `scripts/review_ranges.py` | Interactive matplotlib viewer to visually verify RRNG ranges against spectra | [docs](review_ranges.md) |
