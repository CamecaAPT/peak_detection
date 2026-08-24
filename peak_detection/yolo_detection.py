from __future__ import annotations

import os
import torch

from .models import PeakRange

import yaml
from peak_detection.RangingModels.RangingNN.predictor import DetectionPredictor


def run_yolo_ranging(spectrum_log, *, yolo_weights, iou, conf, max_det):
    """Run the YOLO1D ranging model and return detected peak ranges.

    Returns None (with a printed error) if the model weight/config files are missing.
    Returns a (possibly empty) list of PeakRange otherwise — each with only
    start/end/pos populated; label/id_score/method/detailed_id/is_unknown are left at
    their dataclass defaults for a classifier pipeline to fill in.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modelpath = os.path.join(base_dir, 'peak_detection', 'RangingModels', 'RangingNN', 'modelweights', yolo_weights)
    cfg_path = os.path.join(base_dir, 'peak_detection', 'RangingModels', 'RangingNN', 'cfg', 'prediction_args.yaml')

    if not os.path.exists(modelpath) or not os.path.exists(cfg_path):
        print(f"  [Error] YOLO model files not found at {modelpath}")
        return None

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['iou'], cfg['conf'], cfg['max_det'] = iou, conf, max_det

    if spectrum_log.shape[0] < 30720:
        sp_padded = torch.zeros(30720)
        sp_padded[:spectrum_log.shape[0]] = spectrum_log
    else:
        sp_padded = spectrum_log[:30720]

    predictor = DetectionPredictor(modelpath, sp_padded[None, None, ...], save_dir='test_results', cfg=cfg)
    result = predictor()[0]
    peak_range_pred = result[:, :2].cpu()

    multiplier = 0.01
    formatted_results = []
    for r in peak_range_pred.tolist():
        s, e = float(r[0]) * multiplier, float(r[1]) * multiplier
        formatted_results.append(PeakRange(start=s, end=e, pos=(s + e) / 2))
    return formatted_results
