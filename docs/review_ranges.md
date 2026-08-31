# scripts/review_ranges.py

Interactive matplotlib viewer for visually verifying that each RRNG range spans exactly one peak. Displays APT spectra in navigable chunks with range overlays; step through chunks/files with the arrow keys.

## Usage

```bash
python scripts/review_ranges.py <data_directory> <csv_file> [--max_window_size 10] [--buffer 0.5] [--show_all]
```

`<csv_file>` must have the APT/CSV filename in its first column and the RRNG filename in its second column (or `apt_path`/`rrng_path` columns as a fallback), each resolved relative to `<data_directory>`.

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `data_directory` (positional) | **Yes** | — | Folder containing the APT/CSV and RRNG files referenced by `csv_file`. |
| `csv_file` (positional) | **Yes** | — | CSV listing APT/RRNG file pairs to review. |
| `--max_window_size MAX_WINDOW_SIZE` | No | `10` | Max chunk width in Da before starting a new chunk. |
| `--buffer BUFFER` | No | `0.5` | Padding in Da added on each side of a chunk's view window. |
| `--show_all` | No | off (flag) | Step through the entire spectrum in fixed-width chunks, not just regions containing ranges. |

## Notes

- Uses the Qt matplotlib backend (`matplotlib.use("QtAgg")`) — requires a Qt binding (e.g. PyQt5/PySide) installed and a display available; won't run headless.
- Key bindings: `Right` = next chunk (or next file at the end of the last chunk), `Left` = previous chunk (or previous file), `Q`/`Escape` = quit.
- Files listed in `csv_file` that can't be found on disk are skipped with a warning, not a hard failure.
