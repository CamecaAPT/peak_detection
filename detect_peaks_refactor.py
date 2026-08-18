"""
detect_peaks_gg2.py — Clean CLI/notebook entry point for peak detection.

Usage:
    # Single dataset
    python detect_peaks_gg2.py --apt_path singletest --rrng_path RRNG_test

    # Callable from Python
    from detect_peaks_gg2 import process_dataset
    stats = process_dataset('data.csv', 'data.RRNG')
"""

import os
import sys
import re
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path for peak_detection package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from peak_detection.models import DatasetStats
from peak_detection.data_io import load_apt_from_file, parse_rrng, save_rrng
from peak_detection.utils import calculate_iou, calculate_iou_metrics
from peak_detection.yolo_detection import predict_peak_ranges_yolo, identify_peaks
from peak_detection.run_config import (
    add_shared_args,
    apply_config_defaults,
    config_from_namespace,
    write_run_config,
)

# Optional: per-dataset peak-summary writer. Guarded so a problem in that module doesn't
# prevent this script from running its core detection/evaluation.
try:
    from write_dataset_peak_summaries import write_dataset_peak_summaries
except Exception:
    write_dataset_peak_summaries = None

# Script-specific output-control tunables (beyond the shared RunConfig) that are persisted
# to / loadable from the run-config YAML. Per-run I/O paths are deliberately omitted.
SCRIPT_CONFIG_KEYS = ['save_plots', 'save_csv', 'save_rrng_output']


def _default_output_dir(apt_file):
    """Derive a filesystem-safe per-dataset folder name from an APT/CSV file path."""
    name = os.path.splitext(os.path.basename(apt_file))[0].lower()
    name = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_')
    return re.sub(r'_+', '_', name)


def plot_yolo_comparison(stats, xlim=None, save_path=None, facecolor=None):
    """
    Plot YOLO prediction vs truth ranges on top of the spectrum.

    Parameters
    ----------
    stats : DatasetStats
        The DatasetStats returned by `process_dataset`.
    xlim : tuple, optional
        (xmin, xmax) to zoom into a region.
    save_path : str, optional
        If provided, save the figure to this path. Otherwise call plt.show()
        for interactive viewing.
    facecolor : str, optional
        Background color of the plot. If None, uses the matplotlib default.
    """
    x = stats.x
    y_mapped = stats.spectrum
    truth = stats.truth
    detected = stats.detected_ranges
    dataset = stats.dataset

    def _is_molecule_label(label):
        label = str(label or '').strip()
        if not label or label == 'Unknown':
            return False
        return bool(re.search(r'\d', label)) or len(re.findall(r'[A-Z][a-z]?', label)) > 1

    def _predicted_plot_label(peak):
        if getattr(peak, 'is_unknown', False) and getattr(peak, 'label', None):
            return str(peak.label)
        det = getattr(peak, 'detailed_id', None)
        if det is None or not getattr(det, 'el1', None):
            return getattr(peak, 'label', '') or ''

        label1 = str(det.el1)
        label2 = str(getattr(det, 'el2', '') or '')
        conf1 = float(getattr(det, 'conf1', 0.0) or 0.0)
        conf2 = float(getattr(det, 'conf2', 0.0) or 0.0)

        show_second = (
            label2
            and label2 != 'Unknown'
            and label2 != label1
            and conf2 > 0.0
            and _is_molecule_label(label1) != _is_molecule_label(label2)
        )
        if show_second:
            return f"{label1}({conf1:.2f})/{label2}({conf2:.2f})"
        if getattr(peak, 'label', None):
            return str(peak.label)
        return f"{label1}({conf1:.2f})"

    # Compute default x-axis upper bound
    true_max = max(t.end for t in truth) if truth else 0
    pred_max = max(p.end for p in detected) if detected else 0
    plot_xmax = max(true_max, pred_max) + 5

    fig, ax = plt.subplots(figsize=(15, 8))
    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor(facecolor)
        # Override dark-theme text colors so labels are visible
        text_color = 'black' if facecolor in ('white', 'w', '#ffffff', '#fff') else 'white'
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
    plt.plot(x, y_mapped, color='black', alpha=0.3, label='Mapped Spectrum (min_max_scale)')

    # Plot true ranges (blue)
    for i, r in enumerate(truth):
        if xlim and (r.end < xlim[0] or r.start > xlim[1]):
            continue
        plt.axvspan(r.start, r.end, color='blue', alpha=0.15)
        if i == 0:
            plt.axvspan(r.start, r.end, color='blue', alpha=0.15, label='Real (RRNG)')
        if r.label:
            center = (r.start + r.end) / 2
            plt.text(center, 0.85, r.label, color='blue', fontsize=6,
                     ha='center', va='bottom', rotation=90, alpha=0.7)

    # Plot predicted ranges (red)
    for i, p in enumerate(detected):
        if xlim and (p.end < xlim[0] or p.start > xlim[1]):
            continue
        plt.axvspan(p.start, p.end, color='red', alpha=0.3, hatch='//')
        if i == 0:
            plt.axvspan(p.start, p.end, color='red', alpha=0.3, hatch='//', label='YOLO Prediction')
        pred_label = _predicted_plot_label(p)
        if pred_label:
            center = (p.start + p.end) / 2
            plt.text(center, 0.95, pred_label, color='darkred', fontsize=6,
                     ha='center', va='bottom', rotation=90, alpha=0.8)

    plt.xlabel('Mass/Charge Ratio (Da)')
    plt.ylabel('Mapped Intensity (0-1)')
    zoom_suffix = f" (Zoom {xlim[0]}-{xlim[1]})" if xlim else ""
    plt.title(f'YOLO Comparison: {dataset}{zoom_suffix}')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.2)

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0, plot_xmax)

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved comparison plot to {save_path}")
        plt.close('all')
    else:
        plt.show()


def _extract_rf_counts(counts: dict) -> dict:
    """Pull species/elemental/molecular total+correct counts out of an
    rf_accuracy_breakdown['counts']-shaped dict, for both the including- and
    excluding-unknowns scopes. Molecular counts are derived from species - elemental
    when not reported directly. Keys returned (no 'rf_' prefix, '_exc' suffix for the
    excluding-unknowns scope): species_total, species_correct, elemental_total,
    elemental_correct, molecular_total, molecular_correct (and their '_exc' variants).
    """
    def g(key):
        return int(counts.get(key, 0) or 0)

    result = {}
    for suffix, scope in (('', 'including_unknowns'), ('_exc', 'excluding_unknowns')):
        species_total = g(f'species_total_{scope}')
        species_correct = g(f'species_correct_{scope}')
        elemental_total = g(f'elemental_total_{scope}')
        elemental_correct = g(f'elemental_correct_{scope}')
        if f'molecular_total_{scope}' in counts:
            molecular_total = g(f'molecular_total_{scope}')
            molecular_correct = g(f'molecular_correct_{scope}')
        else:
            molecular_total = max(0, species_total - elemental_total)
            molecular_correct = max(0, species_correct - elemental_correct)

        result[f'species_total{suffix}'] = species_total
        result[f'species_correct{suffix}'] = species_correct
        result[f'elemental_total{suffix}'] = elemental_total
        result[f'elemental_correct{suffix}'] = elemental_correct
        result[f'molecular_total{suffix}'] = molecular_total
        result[f'molecular_correct{suffix}'] = molecular_correct
    return result


def _print_rf_accuracy_line(label, key, breakdown, counts):
    """Print one 'RF Accuracy (<label>): X% (c/t) including unknowns, Y% (c/t) excluding unknowns' line."""
    print(
        f"  RF Accuracy ({label}): "
        f"{breakdown.get(f'{key}_including_unknowns', 0.0):.1f}% "
        f"({counts.get(f'{key}_correct_including_unknowns', 0)}/{counts.get(f'{key}_total_including_unknowns', 0)}) including unknowns, "
        f"{breakdown.get(f'{key}_excluding_unknowns', 0.0):.1f}% "
        f"({counts.get(f'{key}_correct_excluding_unknowns', 0)}/{counts.get(f'{key}_total_excluding_unknowns', 0)}) excluding unknowns"
    )


def _print_rescue_impact_line(label, key, before, after, before_counts, after_counts, suffix=''):
    """Print one 'Molecule rescue impact (excluding unknowns): <label> X% (c/t) -> Y% (c/t)' line."""
    print(
        f"  Molecule rescue impact (excluding unknowns): "
        f"{label} {before.get(f'{key}_excluding_unknowns', 0.0):.1f}% "
        f"({before_counts.get(f'{key}_correct_excluding_unknowns', 0)}/{before_counts.get(f'{key}_total_excluding_unknowns', 0)}) -> "
        f"{after.get(f'{key}_excluding_unknowns', 0.0):.1f}% "
        f"({after_counts.get(f'{key}_correct_excluding_unknowns', 0)}/{after_counts.get(f'{key}_total_excluding_unknowns', 0)}){suffix}"
    )


def process_dataset(
    apt_file: str,
    rrng_file: str,
    output_dir: str = None,
    *,
    # YOLO parameters
    yolo_weights: str = 'best_v0_2026-06-23.pt',
    n_iter: int = 0,
    iou: float = 0.01,
    conf: float = 0.05,
    max_det: int = 2000,
    iter_min_intensity_quantile: float = 0.10,
    iter_min_intensity_fraction: float = 0.50,
    iter_intensity_stat_quantile: float = 0.90,
    mc_min: float = 0.0,
    mc_max: float = 307.2,
    # RF parameters
    # NOTE: keyword defaults below mirror peak_detection.run_config.SHARED_PARAMS (the single
    # source of truth used by the CLI in main()); keep them in sync if either one changes.
    training_path: str = 'peak_detection/IonIdentificationModels/training_data/NewData_truthcoverage_lightmol1p_C3_BO_C2O_2p_2026-06-10/Data0001',
    training_num_files: int = 10000,
    augment_molecule_training_charge_ratios: bool = False,
    molecule_rf_rescue_elements: bool = False,
    molecule_rf_rescue_threshold: float = 0.8,
    molecule_rf_rescue_margin: float = 0.15,
    molecule_rf_rescue_score_margin: float = 0.05,
    molecule_rf_rescue_dist_margin: float = 0.05,
    unknown_mixed_element_molecule_confidence_threshold: float = 0.95,
    include_molecules: bool = False,
    use_neighborhood: bool = False,
    neighbor_threshold: float = 2.0,
    use_signature: bool = False,
    unknown_molecule_rf: bool = False,
    unknown_molecule_rf_threshold: float = 0.8,
    followon_mc_vector_rf: bool = False,
    followon_mc_vector_round_decimals: int = 3,
    # Unknown flagging
    flag_unknowns: bool = True,
    mc_threshold: float = 0.2,
    unknown_confidence_threshold: float = 0.6,
    rf_accuracy_top_n: int = 1,
    # Context rescoring
    context_rescore: bool = False,
    context_window_da: float = 2.0,
    context_strength: float = 0.35,
    context_min_confidence: float = 0.75,
    context_min_candidate_confidence: float = 0.05,
    context_override_margin: float = 0.05,
    context_distance_sigma: float = 0.75,
    context_rescue_unknown_same_label: bool = True,
    context_rescue_unknown_min_score: float = 0.7,
    # Output control
    save_plots: bool = True,
    save_rrng_output: bool = False,
    save_csv: bool = True,
    xlim: tuple = None,
) -> DatasetStats:
    """
    Process a single APT dataset: detect peaks with YOLO, classify with RF, and evaluate.

    Returns a DatasetStats with metrics, detected_ranges, identifications, etc.
    """
    _t_file = time.perf_counter()
    rf_accuracy = 0.0
    rf_accuracy_ele = 0.0
    unknown_count = 0

    if output_dir is None:
        output_dir = _default_output_dir(apt_file)

    # output_dir is the directory to write into; prefix names the files within it, so a nested
    # path like results/R7001 yields results/R7001/R7001_*.csv rather than a doubled path.
    prefix = os.path.basename(os.path.normpath(output_dir)) or output_dir

    print(f"\nDetecting peaks for {prefix} (Zoom: {xlim})...")
    x, spectrum, spectrum_log = load_apt_from_file(apt_file)

    y_mapped = spectrum_log.numpy()

    truth, elements_for_molecules = parse_rrng(rrng_file)

    # Save true species and RF elements to files
    truth_species = sorted(list(set([t.label for t in truth if t.label and t.label != 'Unknown'])))
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{prefix}_rf_elements.txt"), 'w') as f:
        f.write("--- Suggested RF Classes (Species) ---\n")
        f.write("\n".join(truth_species))
        f.write("\n\n--- Base Elements for Permutations ---\n")
        f.write("\n".join(sorted(elements_for_molecules)))

    with open(os.path.join(output_dir, f"{prefix}_true_species.txt"), 'w') as f:
        f.write("\n".join(truth_species))

    print(f"  Metadata saved: {output_dir}/{prefix}_rf_elements.txt, {output_dir}/{prefix}_true_species.txt")

    # --- RF ELEMENT IDENTIFICATION ---
    all_predicted, _, rf_accuracy, rf_accuracy_ele, unknown_count, rf_accuracy_breakdown = predict_peak_ranges_yolo(
        apt_file, spectrum_log, x, rrng_file,
        n_iter=n_iter, prefix=prefix, artifacts_dir=output_dir,
        flag_unknowns=flag_unknowns,
        mc_threshold=mc_threshold,
        training_path=training_path, training_num_files=training_num_files, include_molecules=include_molecules,
        augment_molecule_training_charge_ratios=augment_molecule_training_charge_ratios,
        molecule_rf_rescue_elements=molecule_rf_rescue_elements,
        molecule_rf_rescue_threshold=molecule_rf_rescue_threshold,
        molecule_rf_rescue_margin=molecule_rf_rescue_margin,
        molecule_rf_rescue_score_margin=molecule_rf_rescue_score_margin,
        molecule_rf_rescue_dist_margin=molecule_rf_rescue_dist_margin,
        unknown_mixed_element_molecule_confidence_threshold=unknown_mixed_element_molecule_confidence_threshold,
        yolo_weights=yolo_weights, iou=iou, conf=conf, max_det=max_det,
        iter_min_intensity_quantile=iter_min_intensity_quantile,
        iter_min_intensity_fraction=iter_min_intensity_fraction,
        iter_intensity_stat_quantile=iter_intensity_stat_quantile,
        mc_min=mc_min, mc_max=mc_max,
        use_neighborhood=use_neighborhood, neighbor_threshold=neighbor_threshold,
        use_signature=use_signature,
        unknown_molecule_rf=unknown_molecule_rf,
        molecule_rf_threshold=unknown_molecule_rf_threshold,
        unknown_confidence_threshold=unknown_confidence_threshold,
        rf_accuracy_top_n=rf_accuracy_top_n,
        context_rescore=context_rescore,
        context_window_da=context_window_da,
        context_strength=context_strength,
        context_min_confidence=context_min_confidence,
        context_min_candidate_confidence=context_min_candidate_confidence,
        context_override_margin=context_override_margin,
        context_distance_sigma=context_distance_sigma,
        context_rescue_unknown_same_label=context_rescue_unknown_same_label,
        context_rescue_unknown_min_score=context_rescue_unknown_min_score,
        followon_mc_vector_rf=followon_mc_vector_rf,
        followon_mc_vector_round_decimals=followon_mc_vector_round_decimals,
        return_accuracy_breakdown=True,
    )

    # --- RF ACCURACY OUTPUT ---
    if rf_accuracy_breakdown and 'counts' in rf_accuracy_breakdown:
        c = rf_accuracy_breakdown['counts']
        _print_rf_accuracy_line('All species', 'species', rf_accuracy_breakdown, c)
        _print_rf_accuracy_line('Elemental only', 'elemental', rf_accuracy_breakdown, c)
        if 'molecular_excluding_unknowns' in rf_accuracy_breakdown:
            _print_rf_accuracy_line('Molecular only', 'molecular', rf_accuracy_breakdown, c)

        if 'before_rescue' in rf_accuracy_breakdown and 'after_rescue' in rf_accuracy_breakdown:
            b = rf_accuracy_breakdown['before_rescue']
            a = rf_accuracy_breakdown['after_rescue']
            bc = b.get('counts', {}) or {}
            ac = a.get('counts', {}) or {}
            rs = rf_accuracy_breakdown.get('rescue', {}) or {}
            _print_rescue_impact_line('species', 'species', b, a, bc, ac,
                                       suffix=f"; overrides {rs.get('overrides', 0)}/{rs.get('considered', 0)}")
            _print_rescue_impact_line('elements', 'elemental', b, a, bc, ac)
            _print_rescue_impact_line('molecules', 'molecular', b, a, bc, ac)
    else:
        print(f"  RF Accuracy (All species): {rf_accuracy:.1f}%")
        print(f"  RF Accuracy (Elemental only): {rf_accuracy_ele:.1f}%")

    rf_counts = {
        'rf_species_total': 0, 'rf_species_correct': 0,
        'rf_elemental_total': 0, 'rf_elemental_correct': 0,
        'rf_molecular_total': 0, 'rf_molecular_correct': 0,
        'rf_species_total_exc': 0, 'rf_species_correct_exc': 0,
        'rf_elemental_total_exc': 0, 'rf_elemental_correct_exc': 0,
        'rf_molecular_total_exc': 0, 'rf_molecular_correct_exc': 0,
        'rf_species_total_before': 0, 'rf_species_correct_before': 0,
        'rf_elemental_total_before': 0, 'rf_elemental_correct_before': 0,
        'rf_molecular_total_before': 0, 'rf_molecular_correct_before': 0,
        'rf_species_total_before_exc': 0, 'rf_species_correct_before_exc': 0,
        'rf_elemental_total_before_exc': 0, 'rf_elemental_correct_before_exc': 0,
        'rf_molecular_total_before_exc': 0, 'rf_molecular_correct_before_exc': 0,
    }
    molecule_rescue_considered = 0
    molecule_rescue_overrides = 0
    molecule_rescue_mixed_candidates = 0
    if rf_accuracy_breakdown and 'counts' in rf_accuracy_breakdown:
        c = rf_accuracy_breakdown.get('counts', {}) or {}
        rf_counts.update({f'rf_{k}': v for k, v in _extract_rf_counts(c).items()})

        if 'before_rescue' in rf_accuracy_breakdown and 'after_rescue' in rf_accuracy_breakdown:
            bc = (rf_accuracy_breakdown.get('before_rescue', {}) or {}).get('counts', {}) or {}
            for k, v in _extract_rf_counts(bc).items():
                key = f'rf_{k[:-4]}_before_exc' if k.endswith('_exc') else f'rf_{k}_before'
                rf_counts[key] = v

            rs = rf_accuracy_breakdown.get('rescue', {}) or {}
            molecule_rescue_considered = int(rs.get('considered', 0) or 0)
            molecule_rescue_overrides = int(rs.get('overrides', 0) or 0)
            molecule_rescue_mixed_candidates = int(rs.get('mixed_candidates', 0) or 0)

    detected1 = all_predicted

    # --- ACCURACY ASSESSMENT ---
    pc, rc, f1c = calculate_iou_metrics(truth, all_predicted)
    print(f"  Total Combined Metrics: Precision={pc:.3f}, Recall={rc:.3f}, F1={f1c:.3f}")

    # Calculate final found peaks (TP)
    tp_count = 0
    if len(truth) > 0 and len(all_predicted) > 0:
        matched_truth = set()
        for p in all_predicted:
            for i, t in enumerate(truth):
                if calculate_iou(p, t) > 0.1:
                    matched_truth.add(i)
        tp_count = len(matched_truth)

    # Split unknowns by whether the predicted range matches any truth range.
    pred_with_truth = 0
    pred_no_truth = 0
    unknown_with_truth = 0
    unknown_no_truth = 0
    if len(all_predicted) > 0:
        for p in all_predicted:
            best_iou = 0.0
            for t in truth:
                iou_val = calculate_iou(p, t)
                if iou_val > best_iou:
                    best_iou = iou_val
            has_truth = best_iou > 0.1
            if has_truth:
                pred_with_truth += 1
                if getattr(p, 'is_unknown', False):
                    unknown_with_truth += 1
            else:
                pred_no_truth += 1
                if getattr(p, 'is_unknown', False):
                    unknown_no_truth += 1

    # Calculate min/max mass ranges
    true_min = min([t.start for t in truth]) if truth else 0
    true_max = max([t.end for t in truth]) if truth else 0
    pred_min = min([p.start for p in all_predicted]) if all_predicted else 0
    pred_max = max([p.end for p in all_predicted]) if all_predicted else 0
    plot_xmax = max(true_max, pred_max) + 5

    # --- IDENTIFICATION ---
    identified_peaks = identify_peaks(all_predicted, x, spectrum_log, allowed_elements=elements_for_molecules)

    stats = DatasetStats(
        dataset=prefix,
        true_peaks_count=len(truth),
        predicted_peaks_count=len(all_predicted),
        found_peaks_count=tp_count,
        precision=pc,
        recall=rc,
        f1=f1c,
        true_min_mc=true_min,
        true_max_mc=true_max,
        pred_min_mc=pred_min,
        pred_max_mc=pred_max,
        rf_accuracy=round(rf_accuracy, 2),
        rf_accuracy_ele=round(rf_accuracy_ele, 2),
        **rf_counts,
        molecule_rescue_considered=molecule_rescue_considered,
        molecule_rescue_overrides=molecule_rescue_overrides,
        molecule_rescue_mixed_candidates=molecule_rescue_mixed_candidates,
        unknown_count=unknown_count,
        unknown_count_with_truth=unknown_with_truth,
        unknown_count_no_truth=unknown_no_truth,
        predicted_peaks_with_truth=pred_with_truth,
        predicted_peaks_no_truth=pred_no_truth,
        identifications=identified_peaks,
        detected_ranges=all_predicted,
        x=x,
        spectrum=y_mapped,
        truth=truth,
    )

    # --- SAVE PEAK RANGES ---
    if xlim is None:
        results_file = os.path.join(output_dir, f"{prefix}_peak_ranges.txt")
        with open(results_file, 'w') as f:
            f.write("peak_start, peak_end, round, peak_pos\n")
            for p in detected1:
                f.write(f"{p.start:.4f}, {p.end:.4f}, 1, {p.pos:.4f}\n")
        print(f"Ranges saved to {results_file}")

    # --- SAVE RRNG ---
    if save_rrng_output:
        rrng_out_path = os.path.join(output_dir, f"{prefix}_predicted.RRNG")
        save_rrng(rrng_out_path, all_predicted)
        print(f"Predicted RRNG saved to {rrng_out_path}")

    # --- PLOT ---
    if save_plots:
        if xlim is None:
            print(f"Manual RRNG ranges: {len(truth)}")
        zoom_str = f"_zoom_{xlim[0]}_{xlim[1]}" if xlim else ""
        comp_plot_path = os.path.join(output_dir, f"{prefix}_yolo_1d_model_comparison{zoom_str}.png")
        plot_yolo_comparison(stats, xlim=xlim, save_path=comp_plot_path)

        # Also save additional comparison plots sliced into fifths (0-25, 25-50, ...)
        # This runs only for the full-range plot to avoid generating slices of slices.
        if xlim is None:
            slice_edges = [0, 25, 50, 75, 100, 125]
            for lo, hi in zip(slice_edges, slice_edges[1:]):
                if float(lo) >= float(plot_xmax):
                    continue
                slice_path = os.path.join(output_dir, f"{prefix}_yolo_1d_model_comparison_zoom_{lo}_{hi}.png")
                plot_yolo_comparison(stats, xlim=(float(lo), float(hi)), save_path=slice_path)

    print(f"Total processing time for {prefix}: {time.perf_counter() - _t_file:.2f}s")

    return stats


def match_datasets(csv_dir, rrng_dir):
    """
    Match APT/POS/CSV files to RRNG files by filename.

    The ALL_APT_processedCSV and ALL_RRNG_NEW folders are expected to have
    effectively the same basenames, although some exports include extra suffix
    text. Prefer the closest normalized filename match, then fall back to a
    shared run ID such as R6012_266034.
    """
    input_files = sorted(f for f in os.listdir(csv_dir) if f.lower().endswith(('.csv', '.apt', '.pos')))
    rrng_files = sorted(f for f in os.listdir(rrng_dir) if f.lower().endswith('.rrng'))

    def normalized_basename(filename):
        base = os.path.splitext(filename)[0]
        return re.sub(r'[^a-zA-Z0-9]', '', base).lower()

    def run_id(filename):
        match = re.search(r'R\d+_\d+', filename, flags=re.IGNORECASE)
        return match.group(0).lower() if match else None

    rrng_entries = [
        {
            'filename': filename,
            'normalized': normalized_basename(filename),
            'run_id': run_id(filename),
        }
        for filename in rrng_files
    ]

    matches = []
    print(f"DEBUG: APT/POS/CSV files: {len(input_files)}, RRNG files: {len(rrng_files)}")
    for cf in input_files:
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
            prefix = re.sub(r'[^a-zA-Z0-9]', '_', cf.split('.')[0]).lower()
            prefix = re.sub(r'_+', '_', prefix).strip('_')
            matches.append((os.path.join(csv_dir, cf), os.path.join(rrng_dir, best_match), prefix))
        else:
            print(f"  [Warning] No RRNG match found for {cf}")

    return matches


def run_batch(csv_dir, rrng_dir, *, output_base='.', save_plots=True, save_csv=True, **kwargs):
    """
    Run process_dataset on all matched datasets in the given directories.
    Per-dataset output folders are created under ``output_base``.
    Returns list of stats dicts.
    """
    items_to_process = match_datasets(csv_dir, rrng_dir)
    print(f"Found {len(items_to_process)} matched datasets.")

    all_stats = []

    for apt_file, rrng_file, base_prefix in items_to_process:
        print(f"\n==================== DATASET: {base_prefix.upper()} ====================")
        try:
            stats = process_dataset(
                apt_file, rrng_file, os.path.join(output_base, base_prefix),
                save_plots=save_plots,
                save_csv=save_csv,
                **kwargs
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  [Error] Failed to process {base_prefix}: {e}")

    return all_stats


def _plot_summary_series(
    stats,
    output_path,
    title,
    primary_series,
    primary_ylabel,
    *,
    primary_ylim=None,
    primary_ylabel_color=None,
    secondary_series=None,
    secondary_ylabel=None,
    secondary_ylim=None,
    secondary_ylabel_color=None,
    annotate=None,
    grid_alpha=0.2,
    grid_linestyle='-',
    log_label='plot',
):
    """
    Shared per-dataset line-plot renderer used by the plot_rf_*/plot_yolo_metrics_summary
    functions below. ``stats`` must already be sorted by dataset name.

    ``primary_series``/``secondary_series`` are lists of dicts with keys ``values``, ``color``,
    ``label``, and optionally ``linestyle`` (default '-') and ``marker`` (default 'o'). Passing
    ``secondary_series`` adds a twin (right-hand) y-axis. ``annotate`` is an optional dict with
    ``correct``, ``total``, and ``color`` keys to draw correct/total percentage labels (via
    ``_annotate_count_percentages``) near each point on the primary axis.
    """
    if not stats:
        return
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in (s.dataset for s in stats)]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    for series in primary_series:
        ax1.plot(display_names, series['values'], marker=series.get('marker', 'o'),
                  color=series['color'], label=series['label'], linewidth=1.5,
                  linestyle=series.get('linestyle', '-'))

    if annotate is not None:
        _annotate_count_percentages(ax1, display_names, annotate['correct'], annotate['total'], annotate['color'])

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel(primary_ylabel, color=primary_ylabel_color or 'black')
    if primary_ylabel_color:
        ax1.tick_params(axis='y', labelcolor=primary_ylabel_color)
    plt.xticks(rotation=90, ha='center', fontsize=8)
    if primary_ylim is not None:
        ax1.set_ylim(*primary_ylim)
    ax1.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    plt.title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = [], []
    if secondary_series:
        ax2 = ax1.twinx()
        for series in secondary_series:
            ax2.plot(display_names, series['values'], marker=series.get('marker', 'o'),
                      color=series['color'], label=series['label'], linewidth=1.5,
                      linestyle=series.get('linestyle', '-'))
        if secondary_ylabel:
            ax2.set_ylabel(secondary_ylabel, color=secondary_ylabel_color or 'black')
        if secondary_ylabel_color:
            ax2.tick_params(axis='y', labelcolor=secondary_ylabel_color)
        if secondary_ylim is not None:
            ax2.set_ylim(*secondary_ylim)
        lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {log_label} to {output_path}")
    plt.close('all')


def plot_rf_accuracy_summary(all_stats, output_path="rf_accuracy_vs_dataset.png"):
    """Generates a summary plot for RF accuracy across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    overall_acc = [s.rf_accuracy for s in stats]
    elemental_acc = [s.rf_accuracy_ele for s in stats]
    unk_frac_truth = [(getattr(s, 'unknown_count_with_truth', 0) or 0) / (s.predicted_peaks_count or 1) for s in stats]
    unk_frac_no_truth = [(getattr(s, 'unknown_count_no_truth', 0) or 0) / (s.predicted_peaks_count or 1) for s in stats]

    _plot_summary_series(
        stats, output_path,
        title='RF Identification Accuracy and Unknown Peak Fraction (Truth-Matched vs Unmatched)',
        primary_series=[
            dict(values=overall_acc, color='black', label=f'RF Accuracy Overall (Avg: {np.mean(overall_acc):.1f}%)'),
            dict(values=elemental_acc, color='blue', label=f'RF Accuracy Elemental (Avg: {np.mean(elemental_acc):.1f}%)'),
        ],
        primary_ylabel='RF Accuracy (%)', primary_ylabel_color='black', primary_ylim=(-5, 105),
        secondary_series=[
            dict(values=unk_frac_truth, color='grey',
                 label=f'Unknown fraction (truth-matched / all predicted, Avg: {np.mean(unk_frac_truth):.3f})'),
            dict(values=unk_frac_no_truth, color='lightgrey', linestyle='--',
                 label=f'Unknown fraction (extra/no-truth / all predicted, Avg: {np.mean(unk_frac_no_truth):.3f})'),
        ],
        secondary_ylabel='Fraction of Unknowns', secondary_ylabel_color='grey', secondary_ylim=(-0.05, 1.05),
        log_label='RF accuracy summary plot',
    )


def plot_rf_counts_summary(all_stats, output_path="rf_counts_vs_dataset.png"):
    """Plot RF classification totals/correct counts across datasets (elemental-only and overall species)."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_ele = [int(getattr(s, 'rf_elemental_total', 0) or 0) for s in stats]
    corr_ele = [int(getattr(s, 'rf_elemental_correct', 0) or 0) for s in stats]
    true_all = [int(getattr(s, 'rf_species_total', 0) or 0) for s in stats]
    corr_all = [int(getattr(s, 'rf_species_correct', 0) or 0) for s in stats]

    _plot_summary_series(
        stats, output_path,
        title='RF Classification Counts (True vs Correct)',
        primary_series=[
            dict(values=true_ele, color='blue', label='True elemental (count)'),
            dict(values=corr_ele, color='navy', label='Correct elemental (count)'),
            dict(values=true_all, color='black', label='True overall (count)'),
            dict(values=corr_all, color='green', label='Correct overall (count)'),
        ],
        primary_ylabel='Count', primary_ylim=(-1, None),
        log_label='RF counts summary plot',
    )


def plot_rf_species_counts_with_unknowns_summary(all_stats, output_path="rf_species_counts_with_unknowns_vs_dataset.png"):
    """
    Plot overall (elements+molecules) RF evaluation counts across datasets:
      - total evaluated species (truth-matched peaks)
      - correct species
      - unknown species (truth-matched unknown predictions)
      - unknown extra predictions (unknown predictions with no truth match)
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    total_all = [int(getattr(s, 'rf_species_total', 0) or 0) for s in stats]
    correct_all = [int(getattr(s, 'rf_species_correct', 0) or 0) for s in stats]
    unknown_with_truth = [int(getattr(s, 'unknown_count_with_truth', 0) or 0) for s in stats]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]

    _plot_summary_series(
        stats, output_path,
        title='RF Overall Species Counts (Total vs Correct vs Unknown, Split by Truth Match)',
        primary_series=[
            dict(values=total_all, color='black', label='Total species (truth-matched count)'),
            dict(values=correct_all, color='green', label='Correct species (count)'),
            dict(values=unknown_with_truth, color='grey', linestyle='--', label='Unknown species (truth-matched count)'),
            dict(values=unknown_no_truth, color='lightgrey', linestyle=':', label='Unknown extra predictions (no truth match count)'),
        ],
        primary_ylabel='Count', primary_ylim=(-1, None),
        log_label='RF overall species+unknown counts plot',
    )


def _annotate_count_percentages(ax, display_names, correct_counts, total_counts, color):
    """Annotate correct/total percentages near each correct-count point."""
    offsets = [8, 18, 28, 38]
    for idx, (name, correct, total) in enumerate(zip(display_names, correct_counts, total_counts)):
        if total <= 0:
            continue
        pct = correct / total * 100.0
        ax.annotate(
            f"{pct:.0f}%",
            xy=(name, correct),
            xytext=(0, offsets[idx % len(offsets)]),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=6,
            color=color,
            annotation_clip=False,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
        )


def plot_rf_element_counts_summary(all_stats, output_path="rf_element_counts_vs_dataset.png"):
    """Plot RF elemental-only true/correct counts including unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_ele = [int(getattr(s, 'rf_elemental_total', 0) or 0) for s in stats]
    corr_ele = [int(getattr(s, 'rf_elemental_correct', 0) or 0) for s in stats]
    true_ele_exc = [int(getattr(s, 'rf_elemental_total_exc', 0) or 0) for s in stats]
    unknown_ele_truth = [max(0, ti - te) for ti, te in zip(true_ele, true_ele_exc)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_ele)
    total_correct = sum(corr_ele)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_ele + corr_ele + [1])

    _plot_summary_series(
        stats, output_path,
        title=f'RF Elemental Classification Counts Including Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        primary_series=[
            dict(values=true_ele, color='blue', label='True elemental incl. unknowns (count)'),
            dict(values=corr_ele, color='navy',
                 label=f'Correct elemental incl. unknowns (count): total {total_correct}/{total_true} ({total_acc:.1f}%)'),
        ],
        primary_ylabel='Element count', primary_ylim=(-1, main_ymax * 1.18 + 1),
        annotate=dict(correct=corr_ele, total=true_ele, color='navy'),
        secondary_series=[
            dict(values=unknown_ele_truth, color='grey', linestyle='--', label='Unknown elemental (truth-matched count)'),
            dict(values=unknown_no_truth, color='lightgrey', linestyle=':', label='Unknown extra predictions (no truth match count)'),
        ],
        secondary_ylabel='Unknown count', secondary_ylim=(-1, None),
        log_label='RF elemental counts summary plot',
    )


def plot_rf_molecule_counts_summary(all_stats, output_path="rf_molecule_counts_vs_dataset.png"):
    """Plot RF molecule-only true/correct counts including unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_mol = [int(getattr(s, 'rf_molecular_total', 0) or 0) for s in stats]
    corr_mol = [int(getattr(s, 'rf_molecular_correct', 0) or 0) for s in stats]
    true_mol_exc = [int(getattr(s, 'rf_molecular_total_exc', 0) or 0) for s in stats]
    unknown_mol_truth = [max(0, ti - te) for ti, te in zip(true_mol, true_mol_exc)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_mol)
    total_correct = sum(corr_mol)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_mol + corr_mol + [1])

    _plot_summary_series(
        stats, output_path,
        title=f'RF Molecular Classification Counts Including Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        primary_series=[
            dict(values=true_mol, color='black', label='True molecules incl. unknowns (count)'),
            dict(values=corr_mol, color='purple',
                 label=f'Correct molecules incl. unknowns (count): total {total_correct}/{total_true} ({total_acc:.1f}%)'),
        ],
        primary_ylabel='Molecule count', primary_ylim=(-1, main_ymax * 1.18 + 1),
        annotate=dict(correct=corr_mol, total=true_mol, color='purple'),
        secondary_series=[
            dict(values=unknown_mol_truth, color='grey', linestyle='--', label='Unknown molecules (truth-matched count)'),
            dict(values=unknown_no_truth, color='lightgrey', linestyle=':', label='Unknown extra predictions (no truth match count)'),
        ],
        secondary_ylabel='Unknown count', secondary_ylim=(-1, None),
        log_label='RF molecular counts summary plot',
    )


def plot_rf_element_counts_excluding_unknowns_summary(
    all_stats,
    output_path="rf_element_counts_excluding_unknowns_vs_dataset.png",
):
    """Plot RF elemental-only true/correct counts excluding unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_ele = [int(getattr(s, 'rf_elemental_total_exc', 0) or 0) for s in stats]
    corr_ele = [int(getattr(s, 'rf_elemental_correct_exc', 0) or 0) for s in stats]
    true_ele_inc = [int(getattr(s, 'rf_elemental_total', 0) or 0) for s in stats]
    unknown_ele_truth = [max(0, ti - te) for ti, te in zip(true_ele_inc, true_ele)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_ele)
    total_correct = sum(corr_ele)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_ele + corr_ele + [1])

    _plot_summary_series(
        stats, output_path,
        title=f'RF Elemental Classification Counts Excluding Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        primary_series=[
            dict(values=true_ele, color='blue', label='True elemental excl. unknowns (count)'),
            dict(values=corr_ele, color='navy',
                 label=f'Correct elemental excl. unknowns (count): total {total_correct}/{total_true} ({total_acc:.1f}%)'),
        ],
        primary_ylabel='Element count', primary_ylim=(-1, main_ymax * 1.18 + 1),
        annotate=dict(correct=corr_ele, total=true_ele, color='navy'),
        secondary_series=[
            dict(values=unknown_ele_truth, color='grey', linestyle='--', label='Unknown elemental (truth-matched count)'),
            dict(values=unknown_no_truth, color='lightgrey', linestyle=':', label='Unknown extra predictions (no truth match count)'),
        ],
        secondary_ylabel='Unknown count', secondary_ylim=(-1, None),
        log_label='RF elemental counts excluding unknowns summary plot',
    )


def plot_rf_molecule_counts_excluding_unknowns_summary(
    all_stats,
    output_path="rf_molecule_counts_excluding_unknowns_vs_dataset.png",
):
    """Plot RF molecule-only true/correct counts excluding unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_mol = [int(getattr(s, 'rf_molecular_total_exc', 0) or 0) for s in stats]
    corr_mol = [int(getattr(s, 'rf_molecular_correct_exc', 0) or 0) for s in stats]
    true_mol_inc = [int(getattr(s, 'rf_molecular_total', 0) or 0) for s in stats]
    unknown_mol_truth = [max(0, ti - te) for ti, te in zip(true_mol_inc, true_mol)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_mol)
    total_correct = sum(corr_mol)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_mol + corr_mol + [1])

    _plot_summary_series(
        stats, output_path,
        title=f'RF Molecular Classification Counts Excluding Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        primary_series=[
            dict(values=true_mol, color='black', label='True molecules excl. unknowns (count)'),
            dict(values=corr_mol, color='purple',
                 label=f'Correct molecules excl. unknowns (count): total {total_correct}/{total_true} ({total_acc:.1f}%)'),
        ],
        primary_ylabel='Molecule count', primary_ylim=(-1, main_ymax * 1.18 + 1),
        annotate=dict(correct=corr_mol, total=true_mol, color='purple'),
        secondary_series=[
            dict(values=unknown_mol_truth, color='grey', linestyle='--', label='Unknown molecules (truth-matched count)'),
            dict(values=unknown_no_truth, color='lightgrey', linestyle=':', label='Unknown extra predictions (no truth match count)'),
        ],
        secondary_ylabel='Unknown count', secondary_ylim=(-1, None),
        log_label='RF molecular counts excluding unknowns summary plot',
    )


def _mean_where_gated(values, gates):
    """Mean of `values` at positions where the matching `gates` entry is > 0; 0.0 if none."""
    filtered = [v for v, g in zip(values, gates) if g > 0]
    return float(np.mean(filtered)) if filtered else 0.0


def _rf_accuracy_pct_series(stats, total_key, correct_key, total_inc_key=None):
    """Shared data prep for the plot_rf_*_accuracy_pct_* functions below.

    Returns (pct, avg_pct, unk_frac_truth, avg_unk_truth, unk_frac_extra, avg_unk_extra) where
    `pct` is correct/total*100 using `correct_key`/`total_key`, and the unknown fractions compare
    `total_inc_key` (including-unknowns total, defaults to `total_key` when accuracy already
    counts unknowns as incorrect) against the excluding-unknowns total.
    """
    totals = [int(getattr(s, total_key, 0) or 0) for s in stats]
    corrects = [int(getattr(s, correct_key, 0) or 0) for s in stats]
    totals_inc = [int(getattr(s, total_inc_key or total_key, 0) or 0) for s in stats]
    totals_exc = totals if total_inc_key else [int(getattr(s, total_key + '_exc', 0) or 0) for s in stats]
    extra_unknown_counts = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    extra_totals = [int(getattr(s, 'predicted_peaks_no_truth', 0) or 0) for s in stats]

    pct = [(c / t * 100.0) if t > 0 else 0.0 for c, t in zip(corrects, totals)]
    unk_frac_truth = [((ti - te) / ti) if ti > 0 else 0.0 for ti, te in zip(totals_inc, totals_exc)]
    unk_frac_extra = [(u / t) if t > 0 else 0.0 for u, t in zip(extra_unknown_counts, extra_totals)]
    return (
        pct, _mean_where_gated(pct, totals),
        unk_frac_truth, _mean_where_gated(unk_frac_truth, totals_inc),
        unk_frac_extra, _mean_where_gated(unk_frac_extra, extra_totals),
    )


def _plot_rf_accuracy_pct_summary(stats, output_path, title, species_label, color, pct_label,
                                   total_key, correct_key, total_inc_key=None, log_label=''):
    """Shared renderer for the 4 plot_rf_*_accuracy_pct_* functions below."""
    pct, avg_pct, unk_frac_truth, avg_unk_truth, unk_frac_extra, avg_unk_extra = _rf_accuracy_pct_series(
        stats, total_key, correct_key, total_inc_key
    )
    _plot_summary_series(
        stats, output_path,
        title=title,
        primary_series=[dict(values=pct, color=color, label=f'{pct_label} (Avg: {avg_pct:.1f}%)')],
        primary_ylabel='Correct (%)', primary_ylim=(-5, 105),
        secondary_series=[
            dict(values=unk_frac_truth, color='grey', linestyle='--',
                 label=f'{species_label} unknown fraction (truth-matched, Avg: {avg_unk_truth:.3f})'),
            dict(values=unk_frac_extra, color='lightgrey', linestyle=':',
                 label=f'Extra unknown fraction (no truth match, Avg: {avg_unk_extra:.3f})'),
        ],
        secondary_ylabel='Unknown fraction', secondary_ylim=(-0.05, 1.05),
        log_label=log_label,
    )


def plot_rf_element_accuracy_pct_summary(all_stats, output_path="rf_element_accuracy_pct_vs_dataset.png"):
    """
    Plot RF elemental-only correctness percentage across datasets (correct/true * 100),
    excluding unknown predictions from the denominator. Also overlays unknown fraction.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    _plot_rf_accuracy_pct_summary(
        stats, output_path,
        title='RF Elemental Classification Accuracy (Excluding Unknowns)',
        species_label='Element', color='navy', pct_label='Elemental correct (%)',
        total_key='rf_elemental_total_exc', correct_key='rf_elemental_correct_exc',
        total_inc_key='rf_elemental_total',
        log_label='RF elemental accuracy percent plot',
    )


def plot_rf_molecule_accuracy_pct_summary(all_stats, output_path="rf_molecule_accuracy_pct_vs_dataset.png"):
    """
    Plot RF molecule-only correctness percentage across datasets (correct/true * 100),
    excluding unknown predictions from the denominator. Also overlays unknown fraction.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    _plot_rf_accuracy_pct_summary(
        stats, output_path,
        title='RF Molecular Classification Accuracy (Excluding Unknowns)',
        species_label='Molecule', color='purple', pct_label='Molecular correct (%)',
        total_key='rf_molecular_total_exc', correct_key='rf_molecular_correct_exc',
        total_inc_key='rf_molecular_total',
        log_label='RF molecular accuracy percent plot',
    )


def plot_rf_element_accuracy_pct_including_unknowns_summary(
    all_stats,
    output_path="rf_element_accuracy_pct_including_unknowns_vs_dataset.png",
):
    """
    Plot RF elemental-only correctness percentage across datasets (correct/true * 100),
    counting Unknown predictions as incorrect.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    _plot_rf_accuracy_pct_summary(
        stats, output_path,
        title='RF Elemental Classification Accuracy (Including Unknowns)',
        species_label='Element', color='navy', pct_label='Elemental correct incl. unknowns (%)',
        total_key='rf_elemental_total', correct_key='rf_elemental_correct',
        log_label='RF elemental accuracy percent including unknowns plot',
    )


def plot_rf_molecule_accuracy_pct_including_unknowns_summary(
    all_stats,
    output_path="rf_molecule_accuracy_pct_including_unknowns_vs_dataset.png",
):
    """
    Plot RF molecule-only correctness percentage across datasets (correct/true * 100),
    counting Unknown predictions as incorrect.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    _plot_rf_accuracy_pct_summary(
        stats, output_path,
        title='RF Molecular Classification Accuracy (Including Unknowns)',
        species_label='Molecule', color='purple', pct_label='Molecular correct incl. unknowns (%)',
        total_key='rf_molecular_total', correct_key='rf_molecular_correct',
        log_label='RF molecular accuracy percent including unknowns plot',
    )


def plot_yolo_metrics_summary(all_stats, output_path="yolo_metrics_vs_dataset.png"):
    """Generates a summary plot for YOLO metrics across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    precision = [s.precision for s in stats]
    recall = [s.recall for s in stats]
    f1 = [s.f1 for s in stats]

    _plot_summary_series(
        stats, output_path,
        title='YOLO Peak Detection Performance across Datasets',
        primary_series=[
            dict(values=precision, color='red', label=f'Precision (Avg: {np.mean(precision):.3f})'),
            dict(values=recall, color='green', label=f'Recall (Avg: {np.mean(recall):.3f})'),
            dict(values=f1, color='blue', label=f'F1 Score (Avg: {np.mean(f1):.3f})'),
        ],
        primary_ylabel='Score', grid_linestyle='--', grid_alpha=0.6,
        log_label='YOLO metrics summary plot',
    )


def main():
    parser = argparse.ArgumentParser(description="Peak detection for APT data (v2).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML run-config file. Its values become defaults that "
                             "explicit CLI flags still override.")
    parser.add_argument("--apt_path", type=str, default='ALL_APT_processedCSV',
                        help="Path to .apt/.csv file or directory for batch mode")
    parser.add_argument("--rrng_path", type=str, default='ALL_RRNG_NEW',
                        help="Path to .rrng file or directory for batch mode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for this run. Single mode: the dataset folder "
                             "(default: derived from the APT filename). Batch mode: parent folder "
                             "holding per-dataset folders plus the global summary CSV, "
                             "identifications, and plots (default: current directory). The "
                             "run-config YAML is written here in both modes.")

    # Shared YOLO / RF / unknown-flagging / context-rescoring parameters
    # (single source of truth: peak_detection/run_config.py).
    add_shared_args(parser)

    # Output control
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rrng_output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)

    # Apply --config YAML as defaults (explicit CLI flags still override), then parse.
    apply_config_defaults(parser)
    args = parser.parse_args()

    cfg = config_from_namespace(args)

    apt_path = args.apt_path
    rrng_path = args.rrng_path

    if not os.path.exists(apt_path) or not os.path.exists(rrng_path):
        print(f"Error: Path not found:\n  APT: {apt_path}\n  RRNG: {rrng_path}")
        sys.exit(1)

    # Detect single-file vs batch mode
    is_single = os.path.isfile(apt_path)

    # Resolve this run's output directory. In single mode it is the dataset folder (explicit
    # --output_dir, else derived from the APT filename); in batch mode it is the parent that
    # holds the per-dataset folders plus the global summary CSV, identifications, and plots.
    # The run-config YAML is written here as well.
    if is_single:
        out_base = args.output_dir or _default_output_dir(apt_path)
    else:
        out_base = args.output_dir or '.'

    # Script-specific tunables to persist in the run config (I/O paths are intentionally
    # excluded; the `command` header records those). These load back via --config too.
    write_run_config(cfg, extra={k: getattr(args, k) for k in SCRIPT_CONFIG_KEYS},
                     directory=out_base)

    # Shared params come from the RunConfig; output-control flags are script-specific.
    common_kwargs = {
        **cfg.to_kwargs(),
        'save_plots': args.save_plots,
        'save_rrng_output': args.save_rrng_output,
        'save_csv': args.save_csv,
    }

    if is_single:
        # Single file mode
        stats = process_dataset(apt_path, rrng_path, output_dir=out_base, **common_kwargs)
        print(f"\nDone. Results in {out_base}/")
    else:
        # Batch mode
        print(f"Scanning for datasets in {apt_path}...")
        all_stats = run_batch(apt_path, rrng_path, output_base=out_base, **common_kwargs)

        if all_stats:
            # Save global summary statistics to CSV (into the run output directory)
            summary_file = os.path.join(out_base, "peak_detection_summary.csv")
            fieldnames = [
                'dataset', 'config', 'true_peaks_count', 'predicted_peaks_count',
                'found_peaks_count', 'precision', 'recall', 'f1',
                'true_min_mc', 'true_max_mc', 'pred_min_mc', 'pred_max_mc',
                'rf_accuracy', 'rf_accuracy_ele',
                'rf_species_total', 'rf_species_correct',
                'rf_elemental_total', 'rf_elemental_correct',
                'rf_molecular_total', 'rf_molecular_correct',
                'rf_species_total_exc', 'rf_species_correct_exc',
                'rf_elemental_total_exc', 'rf_elemental_correct_exc',
                'rf_molecular_total_exc', 'rf_molecular_correct_exc',
                'rf_species_total_before', 'rf_species_correct_before',
                'rf_elemental_total_before', 'rf_elemental_correct_before',
                'rf_molecular_total_before', 'rf_molecular_correct_before',
                'rf_species_total_before_exc', 'rf_species_correct_before_exc',
                'rf_elemental_total_before_exc', 'rf_elemental_correct_before_exc',
                'rf_molecular_total_before_exc', 'rf_molecular_correct_before_exc',
                'molecule_rescue_considered', 'molecule_rescue_overrides', 'molecule_rescue_mixed_candidates',
                'unknown_count', 'unknown_count_with_truth', 'unknown_count_no_truth',
                'predicted_peaks_with_truth', 'predicted_peaks_no_truth',
            ]
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_stats:
                    csv_row = {k: getattr(row, k) for k in fieldnames}
                    writer.writerow(csv_row)

            if write_dataset_peak_summaries is not None:
                try:
                    written_peak_summaries = write_dataset_peak_summaries(Path(out_base))
                    print(f"Per-dataset peak summaries saved: {len(written_peak_summaries)} files")
                except Exception as e:
                    print(f"  [Warn] Failed writing per-dataset peak summaries ({e})")
            else:
                print("  [Warn] write_dataset_peak_summaries module not available; skipping per-dataset summaries")

            # If molecule rescue was enabled, print an overall before/after summary (excluding unknowns).
            if any(int(getattr(s, 'molecule_rescue_considered', 0) or 0) > 0 for s in all_stats):
                def _sum_before_after(total_after_key: str, correct_after_key: str, total_before_key: str, correct_before_key: str):
                    bt = bc = at = ac = 0
                    for s in all_stats:
                        at_i = int(getattr(s, total_after_key, 0) or 0)
                        ac_i = int(getattr(s, correct_after_key, 0) or 0)
                        bt_i = int(getattr(s, total_before_key, 0) or 0)
                        bc_i = int(getattr(s, correct_before_key, 0) or 0)
                        # If before wasn't populated (e.g. no rescue candidates), treat before == after.
                        if bt_i == 0 and at_i > 0:
                            bt_i, bc_i = at_i, ac_i
                        bt += bt_i
                        bc += bc_i
                        at += at_i
                        ac += ac_i
                    bp = (bc / bt * 100.0) if bt > 0 else 0.0
                    ap = (ac / at * 100.0) if at > 0 else 0.0
                    return (bc, bt, bp), (ac, at, ap)

                (bc_s, bt_s, bp_s), (ac_s, at_s, ap_s) = _sum_before_after(
                    'rf_species_total_exc', 'rf_species_correct_exc', 'rf_species_total_before_exc', 'rf_species_correct_before_exc'
                )
                (bc_e, bt_e, bp_e), (ac_e, at_e, ap_e) = _sum_before_after(
                    'rf_elemental_total_exc', 'rf_elemental_correct_exc', 'rf_elemental_total_before_exc', 'rf_elemental_correct_before_exc'
                )
                (bc_m, bt_m, bp_m), (ac_m, at_m, ap_m) = _sum_before_after(
                    'rf_molecular_total_exc', 'rf_molecular_correct_exc', 'rf_molecular_total_before_exc', 'rf_molecular_correct_before_exc'
                )
                overrides = sum(int(getattr(s, 'molecule_rescue_overrides', 0) or 0) for s in all_stats)
                mixed = sum(int(getattr(s, 'molecule_rescue_mixed_candidates', 0) or 0) for s in all_stats)
                considered = sum(int(getattr(s, 'molecule_rescue_considered', 0) or 0) for s in all_stats)
                print("\n==================== MOLECULE RESCUE SUMMARY (EXCLUDING UNKNOWNS) ====================")
                print(f"  Overall species: {bc_s}/{bt_s} ({bp_s:.1f}%) -> {ac_s}/{at_s} ({ap_s:.1f}%)")
                print(f"  Elemental only:  {bc_e}/{bt_e} ({bp_e:.1f}%) -> {ac_e}/{at_e} ({ap_e:.1f}%)")
                print(f"  Molecular only:  {bc_m}/{bt_m} ({bp_m:.1f}%) -> {ac_m}/{at_m} ({ap_m:.1f}%)")
                print(f"  Rescue accepted: {overrides} overrides, {mixed} mixed candidates / {considered} candidates\n")

            # Aggregate identifications for YOLO model
            yolo_export = []
            for s in all_stats:
                for p in s.identifications:
                    yolo_export.append({
                        'dataset': s.dataset,
                        'mass_center': p.pos,
                        'mass_start': p.start,
                        'mass_end': p.end,
                        'identified_label': p.label
                    })

            if yolo_export:
                id_file = os.path.join(out_base, "yolo_identifications.csv")
                with open(id_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['dataset', 'mass_center', 'mass_start', 'mass_end', 'identified_label'])
                    writer.writeheader()
                    for row in yolo_export:
                        writer.writerow(row)
                print(f"Global YOLO Identifications saved to {id_file}")

            # Generate summary plots (into the run output directory)
            summary_plots = [
                (plot_rf_accuracy_summary, "rf_accuracy_vs_dataset.png"),
                (plot_rf_counts_summary, "rf_counts_vs_dataset.png"),
                (plot_rf_species_counts_with_unknowns_summary, "rf_species_counts_with_unknowns_vs_dataset.png"),
                (plot_rf_element_counts_summary, "rf_element_counts_vs_dataset.png"),
                (plot_rf_molecule_counts_summary, "rf_molecule_counts_vs_dataset.png"),
                (plot_rf_element_counts_excluding_unknowns_summary, "rf_element_counts_excluding_unknowns_vs_dataset.png"),
                (plot_rf_molecule_counts_excluding_unknowns_summary, "rf_molecule_counts_excluding_unknowns_vs_dataset.png"),
                (plot_rf_element_accuracy_pct_summary, "rf_element_accuracy_pct_vs_dataset.png"),
                (plot_rf_molecule_accuracy_pct_summary, "rf_molecule_accuracy_pct_vs_dataset.png"),
                (plot_rf_element_accuracy_pct_including_unknowns_summary, "rf_element_accuracy_pct_including_unknowns_vs_dataset.png"),
                (plot_rf_molecule_accuracy_pct_including_unknowns_summary, "rf_molecule_accuracy_pct_including_unknowns_vs_dataset.png"),
                (plot_yolo_metrics_summary, "yolo_metrics_vs_dataset.png"),
            ]
            for plot_fn, plot_name in summary_plots:
                plot_fn(all_stats, output_path=os.path.join(out_base, plot_name))

            print(f"\nBatch Processing Complete. Summary saved to {summary_file}")
        else:
            print("\nNo datasets were successfully processed.")


if __name__ == "__main__":
    main()
