
"""This script handles data augmentation, train-test splitting, training, and evaluation for the RangingNN model."""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm

from peak_detection.RangingModels.RangingNN.data_generation import Augmentation
from peak_detection.RangingModels.RangingNN.metrics import box_iou
from peak_detection.RangingModels.RangingNN.model_utils import cw2lh
from peak_detection.RangingModels.RangingNN.predictor import DetectionPredictor
from peak_detection.RangingModels.RangingNN.trainer import BaseTrainer

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = THIS_DIR / "training_data"
DEFAULT_MODELWEIGHTS_DIR = THIS_DIR / "modelweights"
DEFAULT_BASE_CFG = THIS_DIR / "cfg" / "current.yaml"
DEFAULT_YOLO_CFG = THIS_DIR / "cfg" / "yolov8.yaml"
DEFAULT_PREDICTION_CFG = THIS_DIR / "cfg" / "prediction_args.yaml"


def get_all_data(apt_path, rrng_path):
    """
    Match .apt/.POS files to .RRNG files by filename (adapted from detect_peaks_refactor.py's
    match_datasets()) instead of pairing by raw sort-order position -- apt/rrng export folders
    commonly have different file counts and inconsistent naming, so index-based pairing risks
    silently matching the wrong ranging file to the wrong dataset.
    """
    apt_path = Path(apt_path)
    rrng_path = Path(rrng_path)
    apt_files = sorted(f.name for f in apt_path.iterdir() if f.suffix.lower() in ('.apt', '.pos'))
    rrng_files = sorted(f.name for f in rrng_path.iterdir() if f.suffix.lower() == '.rrng')

    def normalized_basename(filename):
        base = os.path.splitext(filename)[0]
        return re.sub(r'[^a-zA-Z0-9]', '', base).lower()

    def run_id(filename):
        match = re.search(r'R\d+_\d+', filename, flags=re.IGNORECASE)
        return match.group(0).lower() if match else None

    rrng_entries = [
        {'filename': f, 'normalized': normalized_basename(f), 'run_id': run_id(f)}
        for f in rrng_files
    ]

    matched_apt, matched_rrng = [], []
    for cf in apt_files:
        c_norm = normalized_basename(cf)
        c_run_id = run_id(cf)
        candidates = []
        for rrng in rrng_entries:
            r_norm = rrng['normalized']
            match_rank = None
            if c_norm == r_norm:
                match_rank = 0
            elif len(c_norm) > 5 and len(r_norm) > 5 and (c_norm.startswith(r_norm) or r_norm.startswith(c_norm)):
                match_rank = 1
            elif c_run_id and c_run_id == rrng['run_id']:
                match_rank = 2
            if match_rank is not None:
                candidates.append((match_rank, abs(len(c_norm) - len(r_norm)), rrng['filename']))

        best_match = min(candidates)[2] if candidates else None
        if best_match:
            matched_apt.append(cf)
            matched_rrng.append(best_match)
        else:
            print(f"  [Warning] No RRNG match found for {cf}")

    return matched_apt, matched_rrng

def augment_data(apt_path,
                  rrng_path,
                  apt_files,
                  rrng_files,
                  savepath):
    # Augmentation.file2h5() concatenates savepath + stem + '.h5' with no separator, so it
    # must already end in one.
    savepath = str(savepath)
    if not savepath.endswith(os.sep) and not savepath.endswith('/'):
        savepath += os.sep

    for i in range(len(apt_files)):
        p = os.path.join(savepath, Path(apt_files[i]).stem + '.h5')
        print(p)
        if not os.path.exists(p):
            print('Working on ', apt_files[i])
            try:
                aug = Augmentation(apt_file=os.path.join(apt_path, apt_files[i]),
                                  ranging_file=os.path.join(rrng_path, rrng_files[i]),
                                  savepath = savepath)
                aug.file2h5()
            except:
                print('Error with ', apt_files[i])
                continue

    return

def make_train_test_split(data_dir, val_frac: float, test_frac: float, seed: int = 0):
    """
    Randomly split every *.h5 file directly under data_dir into train/val/test subfolders
    (moved, not copied). BaseTrainer only reads <data_dir>/train and <data_dir>/test
    (see trainer.py's `self.data`); the val/ split is a separate held-out set for the
    evaluate()/make_heatmap() post-training analysis below, not used during training itself.
    """
    data_dir = Path(data_dir)
    h5_files = sorted(f.name for f in data_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found directly under {data_dir}; skipping split "
              f"(already split, or run with --run_augmentation first).")
        return

    rng = random.Random(seed)
    rng.shuffle(h5_files)
    n_val = round(len(h5_files) * val_frac)
    n_test = round(len(h5_files) * test_frac)
    splits = {
        "val": h5_files[:n_val],
        "test": h5_files[n_val:n_val + n_test],
        "train": h5_files[n_val + n_test:],
    }

    for split_name, files in splits.items():
        split_dir = data_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for f in files:
            shutil.move(str(data_dir / f), str(split_dir / f))

    print(f"Split {len(h5_files)} files -> "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

def train(save_path, data_path, epochs=10, device='cpu', batch_size=8, dropout=0.0,
          base_cfg=DEFAULT_BASE_CFG, yolo_cfg=DEFAULT_YOLO_CFG, seed=0, deterministic=True):
    # load configuration file
    yaml_dict = yaml.safe_load(Path(base_cfg).read_text())
    # use your file path here, check all the file path within the current.yaml file

    yaml_dict['model'] = str(yolo_cfg)
    yaml_dict['save_dir'] = str(save_path)
    yaml_dict['data'] = str(data_path)
    yaml_dict['epochs'] = epochs
    yaml_dict['device'] = device
    yaml_dict['batch'] = batch_size
    yaml_dict['dropout'] = dropout

    yaml_dict['seed'] = seed
    yaml_dict['deterministic'] = deterministic

    print(yaml_dict)

    # initiate trainer
    trainer = BaseTrainer(cfg=yaml_dict)

    # run training
    trainer.train()
    return

def predictionMetrics(gt_ranges, pred_ranges, iou_thres):
    """
    Args:
    pred_ranges (torch.Tensor): Tensor of shape [N, 2] representing detections.
        Each detection is of the format: low, high
    gt_ranges (torch.Tensor): Tensor of shape [M, 2] representing labels.
        Each label is of the format: low, high

    outputs:
    differences: Tensor of shape [#tp, 2] representing pred ranges - gt ranges in number of data points
    """
    correct = np.zeros(pred_ranges.shape[0]).astype(bool)

    iou = box_iou(gt_ranges, pred_ranges).squeeze()
    iou = iou.cpu().numpy()

    matches = np.nonzero(iou >= iou_thres) # IoU > threshold and classes match
    matches = np.array(matches).T
    if matches.shape[0]:
        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        differences = pred_ranges[matches[:, 1]] - gt_ranges[matches[:, 0]]
        correct[matches[:, 1].astype(int)] = True
    return (torch.as_tensor(correct, dtype=torch.bool),
            torch.as_tensor(differences, dtype=torch.float32),
            torch.as_tensor(pred_ranges[matches[:, 1]], dtype=torch.int),
            torch.as_tensor(gt_ranges[matches[:, 0]], dtype=torch.int),)

def predictionStats(file,
                    num,
                    iou_thres,
                    modelpath,
                    cfg=DEFAULT_PREDICTION_CFG,
                    save_dir=None):
    """
    Args:
    file: the h5 file path
    num: number of spectrums to be tested within the h5 file
    iou_thres: IOU threshold for the model
    returns recall, precision, low range absolute error, high range absolute error
    """
    with h5py.File(file, "r") as f:
        array = np.asarray(f['input'])[:num,None, :]
        array_raw = np.asarray(f['non_log_spectrums'])[:num,None, :]
        gt = np.asarray(f['label'])[:num]
        sp = torch.tensor(array, dtype=torch.float32)

    recall = np.zeros(sp.shape[0])
    precision = np.zeros(sp.shape[0])
    low_ae =[]
    high_ae = []
    sp_counts_ae = []

    for dataindex in range(sp.shape[0]):
        predictor = DetectionPredictor(modelpath, sp[dataindex], save_dir = save_dir, cfg = cfg)
        result = predictor()[0]
        pred_ranges = result[:,:2].cpu()
        gt_ranges = torch.as_tensor(cw2lh(gt[dataindex] * 30720))# target ranges denormalized, in pixels
        (tp, difference, pred_matched, gt_matched) = predictionMetrics(gt_ranges, pred_ranges, iou_thres)
        precision[dataindex] = tp.sum() / pred_ranges.shape[0]##########################
        recall[dataindex] = tp.sum() / gt_ranges.shape[0]##########################

        num_preds = pred_ranges.shape[0]
        num_trues = gt_ranges.shape[0]

        low_ae = low_ae + list(difference[:,0])
        high_ae = high_ae+ list(difference[:,1])
        for k in range(pred_matched.shape[0]):
            integrated_diff = array_raw[pred_matched[k][0]: pred_matched[k][1]].sum() - array_raw[gt_matched[k][0]: gt_matched[k][1]].sum()
            sp_counts_ae.append(integrated_diff)
    return num_trues, num_preds, recall, precision, torch.as_tensor(low_ae)*0.01, torch.as_tensor(high_ae)*0.01, torch.as_tensor(sp_counts_ae)

def evaluate(test_file, save_dir, model_path=None, cfgpath=None, num_points=25):
    model_path = model_path or (DEFAULT_MODELWEIGHTS_DIR / "best.pt")
    cfgpath = cfgpath or DEFAULT_PREDICTION_CFG

    cfg = yaml.safe_load(Path(cfgpath).read_text())

    confs = np.linspace(0.01, 0.8, num_points)
    ious = np.linspace(0.01, 0.8, num_points)
    preds_list = list()
    f1s_list = list()
    num_preds_list = list()
    for conf in confs:
        cfg['conf'] = conf
        for iou in ious:
            cfg['iou'] = iou
            print(cfg['iou'], cfg['conf'])
            num_trues, num_preds, recall, precision, low_ae, high_ae, sp_counts_ae = predictionStats(test_file,
                                                                          num = 1,
                                                                            modelpath=model_path,
                                                                          iou_thres=iou,
                                                                          save_dir=save_dir,
                                                                          cfg=cfg)
            f1 = 2 * (precision * recall) / (precision + recall)
            preds_list.append([iou, conf, num_preds, num_trues, f1])
            f1s_list.append(f1[0])
            num_preds_list.append(num_preds)
    return num_points, preds_list, f1s_list, num_preds_list, confs, ious

def make_heatmap(test_file, num_points, f1s_list, confs, ious):

    # Create 2D grid from x and y
    X, Y = np.meshgrid(confs, ious)
    Z = np.array(f1s_list).reshape(num_points, num_points)

    # Creating the heatmap
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 20, cmap=cm.viridis)
    plt.contour(X, Y, Z, 20, colors='black', linewidths=0.5, alpha=0.5)
    cbar = plt.colorbar(contour)
    cbar.set_label('Peak ranging F1 score', fontsize=14)
    plt.title(test_file.split('/')[-1], fontsize=14)
    plt.xlabel('Confidence score', fontsize=14)
    plt.ylabel('IoU in Non-max suppression', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    fname = test_file.split('/')[-1].split('.')[0]
    plt.savefig('heatmap_'+fname+'.png', dpi=300, bbox_inches='tight')
    return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Folder holding augmented .h5 files (either already split into "
                             "train/val/test subfolders, or flat for --run_augmentation / auto-split).")
    parser.add_argument("--run_augmentation", action="store_true",
                        help="Run raw .apt/.RRNG -> .h5 augmentation into --data_dir before splitting.")
    parser.add_argument("--apt_path", type=Path, default=None,
                        help="Folder of raw .apt files (required with --run_augmentation).")
    parser.add_argument("--rrng_path", type=Path, default=None,
                        help="Folder of raw .RRNG files (required with --run_augmentation).")
    parser.add_argument("--skip_split", action="store_true",
                        help="Skip the random train/val/test split (data_dir is already split).")
    parser.add_argument("--skip_train", action="store_true",
                        help="Stop after augmentation/split; don't launch training.")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Use deterministic algorithms in the trainer (slower, but reproducible on the same device).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--model_save_dir", type=Path, default=DEFAULT_MODELWEIGHTS_DIR)
    parser.add_argument("--base_cfg", type=Path, default=DEFAULT_BASE_CFG)
    parser.add_argument("--yolo_cfg", type=Path, default=DEFAULT_YOLO_CFG)
    args = parser.parse_args()

    print(torch.cuda.is_available(), torch.__version__)

    # Seed here too (not just inside BaseTrainer) so augment_data()'s peak-shift randomness
    # (plain np.random calls in Augmentation.file2h5) is reproducible given the same seed
    # and the same input .apt/.RRNG files.
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.run_augmentation:
        if args.apt_path is None or args.rrng_path is None:
            parser.error("--run_augmentation requires --apt_path and --rrng_path")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        apt_files, rrng_files = get_all_data(args.apt_path, args.rrng_path)
        augment_data(args.apt_path, args.rrng_path, apt_files, rrng_files, savepath=str(args.data_dir))

    if not args.skip_split:
        make_train_test_split(args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    if args.skip_train:
        print(f"--skip_train set; stopping before training. Data is ready under {args.data_dir}")
        return

    model_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_dropout_{args.dropout}"
    save_path = args.model_save_dir / model_name

    train(save_path=str(save_path),
          data_path=str(args.data_dir),
          epochs=args.epochs,
          device=args.device,
          batch_size=args.batch_size,
          dropout=args.dropout,
          base_cfg=args.base_cfg,
          yolo_cfg=args.yolo_cfg,
          seed=args.seed,
          deterministic=args.deterministic)


if __name__ == "__main__":
    main()
