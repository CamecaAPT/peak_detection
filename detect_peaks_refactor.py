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

# Ensure project root is on path for peak_detection package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from peak_detection.models import DatasetStats
from peak_detection.data_io import (
    load_apt_from_file,
    parse_rrng,
    save_rrng,
        save_rrng_with_uncertainty as write_rrng_with_uncertainty,
)
from peak_detection.utils import calculate_iou, calculate_iou_metrics
from peak_detection.registry import get_pipeline, get_flattener, list_models
from peak_detection.registry.base import ClassifierContext
from peak_detection.registry.config import load_merged_config, write_effective_config

CONFIGS_DIR = os.path.join(current_dir, "configs")
from peak_detection.plotting import (
    plot_yolo_comparison,
    plot_accuracy_summary,
    plot_counts_summary,
    plot_species_counts_with_unknowns_summary,
    plot_element_counts_summary,
    plot_molecule_counts_summary,
    plot_element_counts_excluding_unknowns_summary,
    plot_molecule_counts_excluding_unknowns_summary,
    plot_element_accuracy_pct_summary,
    plot_molecule_accuracy_pct_summary,
    plot_element_accuracy_pct_including_unknowns_summary,
    plot_molecule_accuracy_pct_including_unknowns_summary,
    plot_yolo_metrics_summary,
)

# Optional: per-dataset peak-summary writer. Guarded so a problem in that module doesn't
# prevent this script from running its core detection/evaluation.
try:
    from write_dataset_peak_summaries import write_dataset_peak_summaries
except Exception:
    write_dataset_peak_summaries = None

# Script-specific output-control tunables (beyond the shared RunConfig) that are persisted
# to / loadable from the run-config YAML. Per-run I/O paths are deliberately omitted.
SCRIPT_CONFIG_KEYS = ['save_plots', 'save_csv', 'save_rrng_output', 'save_rrng_with_uncertainty']


def _default_output_dir(apt_file):
    """Derive a filesystem-safe per-dataset folder name from an APT/CSV file path."""
    name = os.path.splitext(os.path.basename(apt_file))[0].lower()
    name = re.sub(r'[^a-zA-Z0-9]', '_', name).strip('_')
    return re.sub(r'_+', '_', name)


def _extract_counts(counts: dict) -> dict:
    """Pull species/elemental/molecular total+correct counts out of an
    accuracy_breakdown['counts']-shaped dict (any registered model's breakdown, not
    RF-specific), for both the including- and excluding-unknowns scopes. Molecular counts
    are derived from species - elemental when not reported directly. Keys returned
    ('_exc' suffix for the excluding-unknowns scope): species_total, species_correct,
    elemental_total, elemental_correct, molecular_total, molecular_correct (and their
    '_exc' variants).
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


def process_dataset(
    apt_file: str,
    rrng_file: str,
    output_dir: str = None,
    *,
    # YOLO parameters
    yolo_weights: str = 'best_v0_2026-06-23.pt',
    iou: float = 0.01,
    conf: float = 0.05,
    max_det: int = 2000,
    mc_min: float = 0.0,
    mc_max: float = 307.2,
    # RF parameters
    # NOTE: keyword defaults below mirror configs/models/rf.yaml (the single source of
    # truth used by the CLI in main()); keep them in sync if it changes.
    training_path: str = 'peak_detection/IonIdentificationModels/training_data/NewData_truthcoverage_lightmol1p_C3_BO_C2O_2p_2026-06-10/Data0001',
    training_num_files: int = 5000,
    augment_molecule_training_charge_ratios: bool = True,
    molecule_rf_rescue_elements: bool = True,
    molecule_rf_rescue_threshold: float = 0.8,
    molecule_rf_rescue_margin: float = 0.15,
    molecule_rf_rescue_score_margin: float = 0.05,
    molecule_rf_rescue_dist_margin: float = 0.05,
    unknown_mixed_element_molecule_confidence_threshold: float = 0.95,
    include_molecules: bool = False,
    use_neighborhood: bool = False,
    neighbor_threshold: float = 2.0,
    use_signature: bool = False,
    unknown_molecule_rf: bool = True,
    unknown_molecule_rf_threshold: float = 0.8,
    # Unknown flagging
    flag_unknowns: bool = True,
    mc_threshold: float = 0.2,
    unknown_confidence_threshold: float = 0.6,
    rf_accuracy_top_n: int = 2,
    # Context rescoring
    context_rescore: bool = True,
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
    save_rrng_with_uncertainty: bool = False,
    save_csv: bool = True,
    xlim: tuple = None,
    model_name: str = "rf",
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
    if save_csv:
        with open(os.path.join(output_dir, f"{prefix}_rf_elements.txt"), 'w') as f:
            f.write("--- Suggested RF Classes (Species) ---\n")
            f.write("\n".join(truth_species))
            f.write("\n\n--- Base Elements for Permutations ---\n")
            f.write("\n".join(sorted(elements_for_molecules)))

        with open(os.path.join(output_dir, f"{prefix}_true_species.txt"), 'w') as f:
            f.write("\n".join(truth_species))

        print(f"  Metadata saved: {output_dir}/{prefix}_rf_elements.txt, {output_dir}/{prefix}_true_species.txt")

    # --- RF ELEMENT IDENTIFICATION ---
    cfg = dict(
        yolo_weights=yolo_weights, iou=iou, conf=conf, max_det=max_det, mc_min=mc_min, mc_max=mc_max,
        training_path=training_path, training_num_files=training_num_files,
        augment_molecule_training_charge_ratios=augment_molecule_training_charge_ratios,
        include_molecules=include_molecules, use_neighborhood=use_neighborhood,
        neighbor_threshold=neighbor_threshold, use_signature=use_signature,
        flag_unknowns=flag_unknowns, mc_threshold=mc_threshold,
        unknown_confidence_threshold=unknown_confidence_threshold, rf_accuracy_top_n=rf_accuracy_top_n,
        unknown_mixed_element_molecule_confidence_threshold=unknown_mixed_element_molecule_confidence_threshold,
        molecule_rf_rescue_elements=molecule_rf_rescue_elements,
        molecule_rf_rescue_threshold=molecule_rf_rescue_threshold,
        molecule_rf_rescue_margin=molecule_rf_rescue_margin,
        molecule_rf_rescue_score_margin=molecule_rf_rescue_score_margin,
        molecule_rf_rescue_dist_margin=molecule_rf_rescue_dist_margin,
        unknown_molecule_rf=unknown_molecule_rf, unknown_molecule_rf_threshold=unknown_molecule_rf_threshold,
        context_rescore=context_rescore, context_window_da=context_window_da,
        context_strength=context_strength, context_min_confidence=context_min_confidence,
        context_min_candidate_confidence=context_min_candidate_confidence,
        context_override_margin=context_override_margin, context_distance_sigma=context_distance_sigma,
        context_rescue_unknown_same_label=context_rescue_unknown_same_label,
        context_rescue_unknown_min_score=context_rescue_unknown_min_score,
    )
    ctx = ClassifierContext(
        apt_file=apt_file, rrng_file=rrng_file, x_exp=x, spectrum_log=spectrum_log,
        truth_data=truth, elements_for_molecules=elements_for_molecules,
        species_list=None, elements_list=None,
        prefix=prefix, artifacts_dir=output_dir, save_artifacts=save_csv, cfg=cfg,
    )
    rf_accuracy_breakdown = get_pipeline(model_name).run(ctx)
    all_predicted = ctx.peaks
    rf_accuracy = float(rf_accuracy_breakdown.get('species_excluding_unknowns', 0.0)) if rf_accuracy_breakdown else 0.0
    rf_accuracy_ele = float(rf_accuracy_breakdown.get('elemental_excluding_unknowns', 0.0)) if rf_accuracy_breakdown else 0.0
    unknown_count = sum(1 for p in all_predicted if getattr(p, 'is_unknown', False))

    accuracy_counts = {
        'species_total': 0, 'species_correct': 0,
        'elemental_total': 0, 'elemental_correct': 0,
        'molecular_total': 0, 'molecular_correct': 0,
        'species_total_exc': 0, 'species_correct_exc': 0,
        'elemental_total_exc': 0, 'elemental_correct_exc': 0,
        'molecular_total_exc': 0, 'molecular_correct_exc': 0,
    }
    if rf_accuracy_breakdown and 'counts' in rf_accuracy_breakdown:
        accuracy_counts.update(_extract_counts(rf_accuracy_breakdown.get('counts', {}) or {}))

    detected1 = all_predicted

    # --- ACCURACY ASSESSMENT ---
    pc, rc, f1c = calculate_iou_metrics(truth, all_predicted)

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
        species_accuracy=round(rf_accuracy, 2),
        elemental_accuracy=round(rf_accuracy_ele, 2),
        **accuracy_counts,
        unknown_count=unknown_count,
        unknown_count_with_truth=unknown_with_truth,
        unknown_count_no_truth=unknown_no_truth,
        predicted_peaks_with_truth=pred_with_truth,
        predicted_peaks_no_truth=pred_no_truth,
        identifications=all_predicted,
        detected_ranges=all_predicted,
        x=x,
        spectrum=y_mapped,
        truth=truth,
        extras=ctx.diagnostics,
    )

    # --- SAVE PEAK RANGES ---
    if save_csv and xlim is None:
        results_file = os.path.join(output_dir, f"{prefix}_peak_ranges.txt")
        with open(results_file, 'w') as f:
            f.write("peak_start, peak_end, round, peak_pos\n")
            for p in detected1:
                f.write(f"{p.start:.4f}, {p.end:.4f}, 1, {p.pos:.4f}\n")
        print(f"Ranges saved to {results_file}")

    # --- SAVE RRNG ---
    if save_rrng_output:
        rrng_out_path = os.path.join(output_dir, f"{prefix}_predicted.RRNG")
        if save_rrng_with_uncertainty:
            write_rrng_with_uncertainty(rrng_out_path, all_predicted)
        else:
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


def main():
    parser = argparse.ArgumentParser(description="Peak detection for APT data (v2).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML run-config file. Its values become defaults that "
                             "explicit CLI flags still override.")
    parser.add_argument("--model", type=str, default="rf", choices=list_models(),
                        help="Classification model to use (see configs/models/).")
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

    # Output control
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rrng_output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-rrng-with-uncertainty", action=argparse.BooleanOptionalAction, default=False,
                        help="Write predicted RRNG files using top-two identification candidates.")
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    # Model tunables come from configs/models/<model>.yaml <- --config override; no
    # per-model CLI flags (single source of truth: the configs/ folder).
    cfg = load_merged_config(args.model, configs_dir=CONFIGS_DIR, override_path=args.config)

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

    # Script-specific tunables to persist alongside the effective config (I/O paths are
    # intentionally excluded; the `command` header records those). These load back via --config too.
    write_effective_config(cfg, extra={k: getattr(args, k) for k in SCRIPT_CONFIG_KEYS},
                           directory=out_base)

    # Model params come from the merged yaml config; output-control flags are script-specific.
    common_kwargs = {
        **get_flattener(args.model)(cfg),
        'save_plots': args.save_plots,
        'save_rrng_output': args.save_rrng_output,
        'save_rrng_with_uncertainty': args.save_rrng_with_uncertainty,
        'save_csv': args.save_csv,
        'model_name': args.model,
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
            if args.save_csv:
                # Universal fields only: every registered model populates these the same
                # way, so this file is safe to compare across --model runs.
                fieldnames = [
                    'dataset', 'config', 'true_peaks_count', 'predicted_peaks_count',
                    'found_peaks_count', 'precision', 'recall', 'f1',
                    'true_min_mc', 'true_max_mc', 'pred_min_mc', 'pred_max_mc',
                    'species_accuracy', 'elemental_accuracy',
                    'species_total', 'species_correct',
                    'elemental_total', 'elemental_correct',
                    'molecular_total', 'molecular_correct',
                    'species_total_exc', 'species_correct_exc',
                    'elemental_total_exc', 'elemental_correct_exc',
                    'molecular_total_exc', 'molecular_correct_exc',
                    'unknown_count', 'unknown_count_with_truth', 'unknown_count_no_truth',
                    'predicted_peaks_with_truth', 'predicted_peaks_no_truth',
                ]
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in all_stats:
                        csv_row = {k: getattr(row, k) for k in fieldnames}
                        writer.writerow(csv_row)

                # Per-approach diagnostics: only written when at least one dataset's
                # pipeline populated ctx.diagnostics (today: RF's before-rescue snapshot).
                # Each row is the universal row plus that approach's own extra columns —
                # a standalone file, not requiring a join against peak_detection_summary.csv.
                diagnostics_rows = [s for s in all_stats if s.extras is not None]
                if diagnostics_rows:
                    diagnostics_file = os.path.join(out_base, f"{args.model}_diagnostics.csv")
                    extras_fieldnames = list(diagnostics_rows[0].extras.to_row().keys())
                    with open(diagnostics_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames + extras_fieldnames)
                        writer.writeheader()
                        for row in diagnostics_rows:
                            csv_row = {k: getattr(row, k) for k in fieldnames}
                            csv_row.update(row.extras.to_row())
                            writer.writerow(csv_row)
                    print(f"Per-approach diagnostics saved to {diagnostics_file}")

                if write_dataset_peak_summaries is not None:
                    try:
                        written_peak_summaries = write_dataset_peak_summaries(Path(out_base))
                        print(f"Per-dataset peak summaries saved: {len(written_peak_summaries)} files")
                    except Exception as e:
                        print(f"  [Warn] Failed writing per-dataset peak summaries ({e})")
                else:
                    print("  [Warn] write_dataset_peak_summaries module not available; skipping per-dataset summaries")

            # Aggregate identifications for YOLO model
            if args.save_csv:
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

            # Generate summary plots (into the run output directory). Each plot function
            # derives its own filename as "{model_label}_<suffix>_vs_dataset.png".
            summary_plot_fns = [
                plot_accuracy_summary,
                plot_counts_summary,
                plot_species_counts_with_unknowns_summary,
                plot_element_counts_summary,
                plot_molecule_counts_summary,
                plot_element_counts_excluding_unknowns_summary,
                plot_molecule_counts_excluding_unknowns_summary,
                plot_element_accuracy_pct_summary,
                plot_molecule_accuracy_pct_summary,
                plot_element_accuracy_pct_including_unknowns_summary,
                plot_molecule_accuracy_pct_including_unknowns_summary,
            ]
            for plot_fn in summary_plot_fns:
                plot_fn(all_stats, output_dir=out_base, model_label=args.model)
            plot_yolo_metrics_summary(all_stats, output_path=os.path.join(out_base, "yolo_metrics_vs_dataset.png"))

            if args.save_csv:
                print(f"\nBatch Processing Complete. Summary saved to {summary_file}")
            else:
                print("\nBatch Processing Complete. (--no-save_csv: summary CSVs not written)")
        else:
            print("\nNo datasets were successfully processed.")


if __name__ == "__main__":
    main()
