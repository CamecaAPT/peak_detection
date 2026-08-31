# scripts/migrate_ionclassifier_checkpoint.py

One-time migration: re-pickles the Ionclassifier checkpoint under its current module path (`peak_detection.IonIdentificationModels.Ionclassifier.*`), so the `peak_detection.Ionclassifier` back-compat alias in `Ionclassifier/__init__.py` can eventually be removed. Takes no CLI arguments.

## Usage

```bash
python scripts/migrate_ionclassifier_checkpoint.py
```

## Arguments

None. The checkpoint path is hardcoded: `peak_detection/IonIdentificationModels/Ionclassifier/modelweights/model_bestepoch.tar`.

## Notes

- Overwrites the checkpoint file in place — back up first if the original pickle format needs to be preserved.
- Only needs to be run once per checkpoint after a module-path rename; re-running on an already-migrated checkpoint is a no-op (loads and re-saves unchanged).
