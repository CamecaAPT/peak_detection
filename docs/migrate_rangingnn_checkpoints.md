# scripts/migrate_rangingnn_checkpoints.py

One-time migration: re-pickles every RangingNN `.pt` checkpoint under its current module path (`peak_detection.RangingModels.RangingNN.*`), so the `peak_detection.RangingNN` back-compat alias in `RangingModels/RangingNN/__init__.py` can eventually be removed. Takes no CLI arguments.

## Usage

```bash
python scripts/migrate_rangingnn_checkpoints.py
```

## Arguments

None. Migrates every `*.pt` file found under `peak_detection/RangingModels/RangingNN/modelweights/`.

## Notes

- Overwrites each checkpoint file in place — back up the `modelweights/` folder first if needed.
- Prints and exits cleanly if no `.pt` files are found under the weights directory.
