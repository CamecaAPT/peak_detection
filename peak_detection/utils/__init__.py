import re
import numpy as np

def min_max_scale(ar):
    """Min-max normalize an array to [0, 1]."""
    return (ar - ar.min()) / (ar.max() - ar.min())

_RRNG_LABEL_RE = re.compile(r'([A-Z][a-z]?):(\d+)')
_SIMPLE_LABEL_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def _format_counts(counts):
    return "".join(sym + (str(c) if c > 1 else "") for sym, c in sorted(counts.items()))


def simplify_label(label):
    """
    Normalizes labels into a canonical composition format by
    alphabetically ordering element symbols.

    Examples:
        "H:2" -> "H2"
        "HH" -> "H2"
        "Si:1 O:1" -> "OSi"
        "COO" -> "CO2"
    """
    if not label or label == "Unknown":
        return label

    # Handle RRNG format: "Si:1 O:1", "H:2"
    if ':' in label:
        parts = _RRNG_LABEL_RE.findall(label)
        if parts:
            counts = {}
            for sym, count in parts:
                counts[sym] = counts.get(sym, 0) + int(count)
            return _format_counts(counts)

    # Handle Synthetic format or simple strings: "HH", "COO", "H2O"
    parts = _SIMPLE_LABEL_RE.findall(label)
    if parts:
        counts = {}
        for sym, count_str in parts:
            counts[sym] = counts.get(sym, 0) + (int(count_str) if count_str else 1)
        return _format_counts(counts)

    return label

def calculate_iou(range1, range2):
    """Calculates Intersection over Union for two PeakRange objects (or any object with .start/.end)."""
    return calculate_iou_1d((range1.start, range1.end), (range2.start, range2.end))


def calculate_iou_1d(interval1, interval2):
    """Calculates IoU for two [start, end] intervals."""
    s1, e1 = interval1
    s2, e2 = interval2
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0, inter_end - inter_start)
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0


# There's probably a more efficient way to do this... O(n*m)
# HARD CODED IOU THRESHOLD
def calculate_iou_metrics(truth, predicted, iou_threshold=0.1):
    """
    Calculates Precision, Recall, and F1 score based on IoU overlap.
    A true peak is 'found' if any predicted peak has IoU > threshold.
    """
    if not truth or not predicted:
        return 0, 0, 0

    tp = 0
    matched_truth = set()
    matched_pred = set()

    for i, t in enumerate(truth):
        for j, p in enumerate(predicted):
            if j in matched_pred:
                continue  # each predicted peak can only satisfy one truth peak
            if calculate_iou(t, p) > iou_threshold:
                tp += 1
                matched_truth.add(i)
                matched_pred.add(j)
                break  # Count each true peak at most once

    precision = len(matched_pred) / len(predicted) if len(predicted) > 0 else 0
    recall = len(matched_truth) / len(truth) if len(truth) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def is_molecule(label):
    """
    Returns True if the label represents a molecular species (multiple atoms or multiple types).
    e.g. "Fe" -> False, "Fe2" -> True, "FeO" -> True
    """
    if not label or label == "Unknown":
        return False

    # Normalize to simple chemical notation
    clean = simplify_label(label)

    # Regex to find atom symbols and their counts
    # Example: "Fe2O3" -> [('Fe', '2'), ('O', '3')]
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', clean)

    if len(parts) > 1 or any(count_str and int(count_str) > 1 for _, count_str in parts):
        return True

    return False
