# scripts/regenerate_mixed_label_plots.py

Rebuilds saved YOLO comparison PNGs (full-range + zoom slices) from an existing `detect_peaks_refactor.py` batch run's saved `*_detailed_results.csv` files, plus the original APT/CSV and RRNG files. Does **not** rerun YOLO detection or RF classification — it only re-renders plots using current plotting code, useful after a plotting-code change without redoing inference.

## Usage

```bash
python scripts/regenerate_mixed_label_plots.py --results_dir "results/bench1" --apt_dir "data/APT_test" --rrng_dir "data/RRNG_test"
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--results_dir RESULTS_DIR` | **Yes** | — | Batch-run output folder with per-dataset subfolders containing `*_detailed_results.csv`. |
| `--apt_dir APT_DIR` | No | `ALL_APT_processedCSV` | Directory of original APT/CSV files, matched to dataset folders by filename. |
| `--rrng_dir RRNG_DIR` | No | `ALL_RRNG_NEW` | Directory of original RRNG files, matched the same way. |

## Notes

- Datasets in `--results_dir` with no APT/RRNG match (via the same `match_datasets` logic as `detect_peaks_refactor.py`) are skipped with a `[skip]` message.
- Only regenerates the full-range plot plus any zoom-slice PNG (`_zoom_{lo}_{hi}.png`, for `[0,25], [25,50], [50,75], [75,100], [100,125]`) that already exists in the dataset folder — it will not create zoom slices that weren't originally saved.
- Reconstructs `PeakRange`/`DatasetStats` objects from the CSV columns rather than the original in-memory detection output, so plots reflect exactly what's on disk.
