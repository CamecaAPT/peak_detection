import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_detection.data_io import load_apt_from_file
from peak_detection.yolo_detection import run_yolo_ranging
from peak_detection.models import PeakRange

SAMPLE_APT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "APT_test", "R13_40310Zr Unsaved - Top Level ROI.csv",
)


def test_run_yolo_ranging_returns_peak_ranges():
    x, spectrum, spectrum_log = load_apt_from_file(SAMPLE_APT)
    result = run_yolo_ranging(
        spectrum_log, yolo_weights="best_v0_2026-06-23.pt", iou=0.01, conf=0.05, max_det=2000,
    )
    assert result is not None
    assert isinstance(result, list)
    assert all(isinstance(p, PeakRange) for p in result)
    assert len(result) > 0
    assert all(p.end > p.start for p in result)


def test_run_yolo_ranging_missing_weights_returns_none():
    x, spectrum, spectrum_log = load_apt_from_file(SAMPLE_APT)
    result = run_yolo_ranging(
        spectrum_log, yolo_weights="does_not_exist.pt", iou=0.01, conf=0.05, max_det=2000,
    )
    assert result is None
