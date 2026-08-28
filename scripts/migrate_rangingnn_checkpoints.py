"""
One-time migration: re-pickle RangingNN .pt checkpoints under their current module
path (peak_detection.RangingModels.RangingNN.*) so the peak_detection.RangingNN
back-compat alias in RangingModels/RangingNN/__init__.py can be removed.

Usage:
    python migrate_rangingnn_checkpoints.py
"""
import os
import glob

import torch

import peak_detection  # noqa: F401  (registers the peak_detection.RangingNN alias)

WEIGHTS_DIR = os.path.join('peak_detection', 'RangingModels', 'RangingNN', 'modelweights')


def main():
    weight_files = sorted(glob.glob(os.path.join(WEIGHTS_DIR, '*.pt')))
    if not weight_files:
        print(f"No .pt files found under {WEIGHTS_DIR}")
        return

    for path in weight_files:
        print(f"Migrating {path} ...")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        torch.save(ckpt, path)
        print(f"  Saved {path}")

    print(f"\nDone. Migrated {len(weight_files)} checkpoint(s).")


if __name__ == '__main__':
    main()
