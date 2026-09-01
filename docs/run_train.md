# peak_detection/RangingModels/RangingNN/run_train.py

Trains the RangingNN (YOLO1D) peak-ranging model: optionally augments raw `.apt`/`.RRNG` pairs into `.h5` training files, splits them into train/val/test, then runs training via `BaseTrainer`.

## Usage

```powershell
# Full pipeline: augment raw data, split, and train
.venv\Scripts\python.exe peak_detection\RangingModels\RangingNN\run_train.py `
    --run_augmentation --apt_path "data\APT_raw" --rrng_path "data\RRNG_raw" `
    --data_dir "peak_detection\RangingModels\RangingNN\training_data\NewData" `
    --epochs 100 --batch_size 8 --device cuda

# Data is already augmented and split into train/val/test subfolders under --data_dir
.venv\Scripts\python.exe peak_detection\RangingModels\RangingNN\run_train.py `
    --data_dir "peak_detection\RangingModels\RangingNN\training_data\NewData" --skip_split

# Only prep data (augment + split), skip training
.venv\Scripts\python.exe peak_detection\RangingModels\RangingNN\run_train.py `
    --run_augmentation --apt_path "data\APT_raw" --rrng_path "data\RRNG_raw" `
    --data_dir "peak_detection\RangingModels\RangingNN\training_data\NewData" --skip_train
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--data_dir DATA_DIR` | No | `RangingNN/training_data` | Folder holding augmented `.h5` files — either already split into `train`/`val`/`test` subfolders, or flat (for `--run_augmentation` / the auto-split step). |
| `--run_augmentation` | No | off (flag) | Run raw `.apt`/`.RRNG` → `.h5` augmentation into `--data_dir` before splitting. |
| `--apt_path APT_PATH` | Required with `--run_augmentation` | none | Folder of raw `.apt`/`.pos` files. |
| `--rrng_path RRNG_PATH` | Required with `--run_augmentation` | none | Folder of raw `.RRNG` files, matched to `--apt_path` files by filename. |
| `--skip_split` | No | off (flag) | Skip the random train/val/test split (`--data_dir` is already split). |
| `--skip_train` | No | off (flag) | Stop after augmentation/split; don't launch training. |
| `--val_frac VAL_FRAC` | No | `0.1` | Fraction of `.h5` files held out for validation (not used during training itself — see Notes). |
| `--test_frac TEST_FRAC` | No | `0.1` | Fraction of `.h5` files held out for test. |
| `--seed SEED` | No | `0` | Random seed for the train/val/test split and augmentation peak-shift randomness. |
| `--deterministic` / `--no-deterministic` | No | `--deterministic` (on) | Use deterministic algorithms in the trainer (slower, but reproducible on the same device). |
| `--epochs EPOCHS` | No | `100` | Number of training epochs. |
| `--device DEVICE` | No | `cuda` if available, else `cpu` | Torch device to train on. |
| `--batch_size BATCH_SIZE` | No | `8` | Training batch size. |
| `--dropout DROPOUT` | No | `0.3` | Dropout rate passed to the model config. |
| `--model_save_dir MODEL_SAVE_DIR` | No | `RangingNN/modelweights` | Parent directory for this run's checkpoint folder (named `<timestamp>_dropout_<dropout>`). |
| `--base_cfg BASE_CFG` | No | `RangingNN/cfg/current.yaml` | Base trainer config YAML (model/save_dir/data/epochs/device/batch/dropout/seed/deterministic are overwritten from the CLI flags above before use). |
| `--yolo_cfg YOLO_CFG` | No | `RangingNN/cfg/yolov8.yaml` | YOLO1D model architecture config. |

## Notes

- `--data_dir` files are matched between `--apt_path` and `--rrng_path` by filename (same matching logic as `detect_peaks_refactor.py`'s `match_datasets`); unmatched `.apt`/`.pos` files are skipped with a `[Warning]`.
- The train/val/test split **moves** (not copies) `.h5` files directly under `--data_dir` into `train/`, `val/`, `test/` subfolders. `BaseTrainer` only reads `train/` and `test/` — `val/` is a separate held-out set for post-training analysis (`evaluate()`/`make_heatmap()` in this file), not used during training.
- If `--data_dir` has no loose `.h5` files at the top level (already split, or augmentation wasn't run), the split step is skipped with a message rather than failing.
- Each run's checkpoint is saved to a fresh timestamped folder (`<model_save_dir>/<YYYYMMDD_HHMMSS>_dropout_<dropout>/`), so repeated runs never overwrite a previous checkpoint.
- `evaluate()`/`predictionStats()`/`make_heatmap()` (confidence/IoU sweep + F1 heatmap over a test `.h5` file) are defined in this module but are not wired to a CLI flag — call them directly from Python if needed.

## Reproducibility

`--seed` and `--deterministic` are wired through (`main()` seeds `random`/`numpy` before augmentation runs; `init_seeds()` in `trainer.py` seeds `random`/`numpy`/`torch`/CUDA and toggles `torch.use_deterministic_algorithms`/cuDNN determinism for training), but this only guarantees reproducibility under one specific condition that's easy to violate:

- **`--data_dir` must be empty of loose `.h5` files before `--run_augmentation` runs.** `augment_data()` skips regenerating a file if it already exists (`if not os.path.exists(p): ... else: skip`) — so on a *partially*-populated `--data_dir`, the number of `np.random` draws consumed before reaching each not-yet-generated file depends on how many files were skipped, which differs between runs. That shifts the global RNG position, so the same `--seed` produces **different** peak-shift augmentation for those files than a from-scratch run would. True reproducibility of the augmentation step requires starting from an empty `--data_dir` (or a `--data_dir` in the exact same partial state) every time.
- The reproducibility guarantee is also device-scoped: matching `--seed`/`--deterministic` only reproduces results on the *same* hardware (same GPU/CPU, same single- vs multi-GPU topology) — not across different machines or device configurations.

That guarantee also does not extend backward: the currently-shipped RangingNN checkpoint (`best_v0_2026-06-23.pt`, referenced as `yolo_weights` in `configs/models/rf.yaml`) was trained **before** `--seed`/`--deterministic` existed as CLI arguments on this script. Loading that checkpoint directly confirms its embedded `train_args`:

```
seed = 0
deterministic = False
data = /ocean/projects/mat240020p/rjacobs/APT_Cameca/peak_detection/all_augmented_data_2026-06-17/
device = [0, 1, 2, 3]
epochs = 100
dropout = 0.3
batch = 16
```

So it can't be reproduced going forward even with matching hyperparameters, for reasons independent of this script's seeding fixes:

- It was trained with `deterministic=False` — the original run itself wasn't reproducible run-to-run, so there's no seed that recreates it even in principle.
- It used 4-GPU distributed training (`device=[0,1,2,3]`), which splits batches and reduces gradients differently than single-GPU/CPU training.
- Its training data (`all_augmented_data_2026-06-17/`) lived on an HPC filesystem path that no longer exists anywhere in this repo or workspace.
- Unlike the two main entry-point scripts (`detect_peaks_headless.py`/`detect_peaks_refactor.py`), which write a full `effective_config_<timestamp>.yaml` per run for provenance, `run_train.py` does not write an equivalent config/seed snapshot alongside older saved checkpoints — the checkpoint's own embedded `train_args` (as loaded above) is the only surviving record.

Practical implication: only *future* training runs (starting from an empty `--data_dir`, on consistent hardware) are reproducible under this script — the checkpoint currently in use cannot be regenerated from scratch and verified against the shipped weights.
