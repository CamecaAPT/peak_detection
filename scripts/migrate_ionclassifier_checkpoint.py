"""
One-time migration: re-pickle the Ionclassifier checkpoint under its current module
path (peak_detection.IonIdentificationModels.Ionclassifier.*) so the
peak_detection.Ionclassifier back-compat alias in Ionclassifier/__init__.py can be removed.

Usage:
    python migrate_ionclassifier_checkpoint.py
"""
import torch

import peak_detection  # noqa: F401  (registers the peak_detection.Ionclassifier alias)

CHECKPOINT_PATH = 'peak_detection/IonIdentificationModels/Ionclassifier/modelweights/model_bestepoch.tar'


def main():
    print(f"Migrating {CHECKPOINT_PATH} ...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    torch.save(ckpt, CHECKPOINT_PATH)
    print(f"  Saved {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
