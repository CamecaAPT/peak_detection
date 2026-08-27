"""Plotting for the peak-detection/identification pipeline.

Split out of detect_peaks_refactor.py: everything here is write-only (renders
matplotlib figures to disk) and has no business logic of its own. Imported
explicitly (`from peak_detection.plotting import ...`) rather than re-exported
from peak_detection/__init__.py, so importing the base package doesn't force
matplotlib on callers that don't plot anything.
"""

from __future__ import annotations

import os
import re

import numpy as np
import matplotlib.pyplot as plt


def plot_yolo_comparison(stats, xlim=None, save_path=None, facecolor=None, model_label=None):
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
    model_label : str, optional
        Free-form name of the identification model that produced these predictions
        (e.g. ``'rf'``, ``'kde'``) — prefixed onto the chart title when given.
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
    model_prefix = f'[{str(model_label).upper()}] ' if model_label else ''
    plt.title(f'{model_prefix}YOLO Comparison: {dataset}{zoom_suffix}')
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
    model_label=None,
):
    """
    Shared per-dataset line-plot renderer used by the plot_*_summary/plot_yolo_metrics_summary
    functions below. ``stats`` must already be sorted by dataset name.

    ``primary_series``/``secondary_series`` are lists of dicts with keys ``values``, ``color``,
    ``label``, and optionally ``linestyle`` (default '-') and ``marker`` (default 'o'). Passing
    ``secondary_series`` adds a twin (right-hand) y-axis. ``annotate`` is an optional dict with
    ``correct``, ``total``, and ``color`` keys to draw correct/total percentage labels (via
    ``_annotate_count_percentages``) near each point on the primary axis.

    ``model_label`` is an optional free-form string naming which identification model produced
    this data (e.g. ``'rf'``, ``'kde'``) — when given, it's prefixed onto the chart title
    (``"[RF] ..."``) so a chart is never ambiguous about which model it's showing.
    """
    if not stats:
        return
    display_names = [d[:20] + '...' if len(d) > 20 else d for d in (s.dataset for s in stats)]
    if model_label:
        title = f'[{str(model_label).upper()}] {title}'

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


def plot_accuracy_summary(all_stats, output_dir=".", model_label=None):
    """Generates a summary plot for identification accuracy across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    overall_acc = [s.species_accuracy for s in stats]
    elemental_acc = [s.elemental_accuracy for s in stats]
    unk_frac_truth = [(getattr(s, 'unknown_count_with_truth', 0) or 0) / (s.predicted_peaks_count or 1) for s in stats]
    unk_frac_no_truth = [(getattr(s, 'unknown_count_no_truth', 0) or 0) / (s.predicted_peaks_count or 1) for s in stats]

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_accuracy_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title='Identification Accuracy and Unknown Peak Fraction (Truth-Matched vs Unmatched)',
        model_label=model_label,
        primary_series=[
            dict(values=overall_acc, color='black', label=f'Accuracy Overall (Avg: {np.mean(overall_acc):.1f}%)'),
            dict(values=elemental_acc, color='blue', label=f'Accuracy Elemental (Avg: {np.mean(elemental_acc):.1f}%)'),
        ],
        primary_ylabel='Accuracy (%)', primary_ylabel_color='black', primary_ylim=(-5, 105),
        secondary_series=[
            dict(values=unk_frac_truth, color='grey',
                 label=f'Unknown fraction (truth-matched / all predicted, Avg: {np.mean(unk_frac_truth):.3f})'),
            dict(values=unk_frac_no_truth, color='lightgrey', linestyle='--',
                 label=f'Unknown fraction (extra/no-truth / all predicted, Avg: {np.mean(unk_frac_no_truth):.3f})'),
        ],
        secondary_ylabel='Fraction of Unknowns', secondary_ylabel_color='grey', secondary_ylim=(-0.05, 1.05),
        log_label='accuracy summary plot',
    )


def plot_counts_summary(all_stats, output_dir=".", model_label=None):
    """Plot classification totals/correct counts across datasets (elemental-only and overall species)."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_ele = [int(getattr(s, 'elemental_total', 0) or 0) for s in stats]
    corr_ele = [int(getattr(s, 'elemental_correct', 0) or 0) for s in stats]
    true_all = [int(getattr(s, 'species_total', 0) or 0) for s in stats]
    corr_all = [int(getattr(s, 'species_correct', 0) or 0) for s in stats]

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_counts_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title='Classification Counts (True vs Correct)',
        model_label=model_label,
        primary_series=[
            dict(values=true_ele, color='blue', label='True elemental (count)'),
            dict(values=corr_ele, color='navy', label='Correct elemental (count)'),
            dict(values=true_all, color='black', label='True overall (count)'),
            dict(values=corr_all, color='green', label='Correct overall (count)'),
        ],
        primary_ylabel='Count', primary_ylim=(-1, None),
        log_label='counts summary plot',
    )


def plot_species_counts_with_unknowns_summary(all_stats, output_dir=".", model_label=None):
    """
    Plot overall (elements+molecules) evaluation counts across datasets:
      - total evaluated species (truth-matched peaks)
      - correct species
      - unknown species (truth-matched unknown predictions)
      - unknown extra predictions (unknown predictions with no truth match)
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    total_all = [int(getattr(s, 'species_total', 0) or 0) for s in stats]
    correct_all = [int(getattr(s, 'species_correct', 0) or 0) for s in stats]
    unknown_with_truth = [int(getattr(s, 'unknown_count_with_truth', 0) or 0) for s in stats]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_species_counts_with_unknowns_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title='Overall Species Counts (Total vs Correct vs Unknown, Split by Truth Match)',
        model_label=model_label,
        primary_series=[
            dict(values=total_all, color='black', label='Total species (truth-matched count)'),
            dict(values=correct_all, color='green', label='Correct species (count)'),
            dict(values=unknown_with_truth, color='grey', linestyle='--', label='Unknown species (truth-matched count)'),
            dict(values=unknown_no_truth, color='lightgrey', linestyle=':', label='Unknown extra predictions (no truth match count)'),
        ],
        primary_ylabel='Count', primary_ylim=(-1, None),
        log_label='overall species+unknown counts plot',
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


def plot_element_counts_summary(all_stats, output_dir=".", model_label=None):
    """Plot elemental-only true/correct counts including unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_ele = [int(getattr(s, 'elemental_total', 0) or 0) for s in stats]
    corr_ele = [int(getattr(s, 'elemental_correct', 0) or 0) for s in stats]
    true_ele_exc = [int(getattr(s, 'elemental_total_exc', 0) or 0) for s in stats]
    unknown_ele_truth = [max(0, ti - te) for ti, te in zip(true_ele, true_ele_exc)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_ele)
    total_correct = sum(corr_ele)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_ele + corr_ele + [1])

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_element_counts_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title=f'Elemental Classification Counts Including Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        model_label=model_label,
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
        log_label='elemental counts summary plot',
    )


def plot_molecule_counts_summary(all_stats, output_dir=".", model_label=None):
    """Plot molecule-only true/correct counts including unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_mol = [int(getattr(s, 'molecular_total', 0) or 0) for s in stats]
    corr_mol = [int(getattr(s, 'molecular_correct', 0) or 0) for s in stats]
    true_mol_exc = [int(getattr(s, 'molecular_total_exc', 0) or 0) for s in stats]
    unknown_mol_truth = [max(0, ti - te) for ti, te in zip(true_mol, true_mol_exc)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_mol)
    total_correct = sum(corr_mol)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_mol + corr_mol + [1])

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_molecule_counts_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title=f'Molecular Classification Counts Including Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        model_label=model_label,
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
        log_label='molecular counts summary plot',
    )


def plot_element_counts_excluding_unknowns_summary(
    all_stats,
    output_dir=".",
    model_label=None,
):
    """Plot elemental-only true/correct counts excluding unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_ele = [int(getattr(s, 'elemental_total_exc', 0) or 0) for s in stats]
    corr_ele = [int(getattr(s, 'elemental_correct_exc', 0) or 0) for s in stats]
    true_ele_inc = [int(getattr(s, 'elemental_total', 0) or 0) for s in stats]
    unknown_ele_truth = [max(0, ti - te) for ti, te in zip(true_ele_inc, true_ele)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_ele)
    total_correct = sum(corr_ele)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_ele + corr_ele + [1])

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_element_counts_excluding_unknowns_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title=f'Elemental Classification Counts Excluding Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        model_label=model_label,
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
        log_label='elemental counts excluding unknowns summary plot',
    )


def plot_molecule_counts_excluding_unknowns_summary(
    all_stats,
    output_dir=".",
    model_label=None,
):
    """Plot molecule-only true/correct counts excluding unknowns across datasets."""
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    true_mol = [int(getattr(s, 'molecular_total_exc', 0) or 0) for s in stats]
    corr_mol = [int(getattr(s, 'molecular_correct_exc', 0) or 0) for s in stats]
    true_mol_inc = [int(getattr(s, 'molecular_total', 0) or 0) for s in stats]
    unknown_mol_truth = [max(0, ti - te) for ti, te in zip(true_mol_inc, true_mol)]
    unknown_no_truth = [int(getattr(s, 'unknown_count_no_truth', 0) or 0) for s in stats]
    total_true = sum(true_mol)
    total_correct = sum(corr_mol)
    total_acc = (total_correct / total_true * 100.0) if total_true else 0.0
    main_ymax = max(true_mol + corr_mol + [1])

    output_path = os.path.join(output_dir, f"{model_label or 'model'}_molecule_counts_excluding_unknowns_vs_dataset.png")
    _plot_summary_series(
        stats, output_path,
        title=f'Molecular Classification Counts Excluding Unknowns (Total Correct: {total_correct}/{total_true}, {total_acc:.1f}%)',
        model_label=model_label,
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
        log_label='molecular counts excluding unknowns summary plot',
    )


def _mean_where_gated(values, gates):
    """Mean of `values` at positions where the matching `gates` entry is > 0; 0.0 if none."""
    filtered = [v for v, g in zip(values, gates) if g > 0]
    return float(np.mean(filtered)) if filtered else 0.0


def _accuracy_pct_series(stats, total_key, correct_key, total_inc_key=None):
    """Shared data prep for the plot_*_accuracy_pct_* functions below.

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


def _plot_accuracy_pct_summary(stats, output_path, title, species_label, color, pct_label,
                                total_key, correct_key, total_inc_key=None, log_label='', model_label=None):
    """Shared renderer for the 4 plot_*_accuracy_pct_* functions below."""
    pct, avg_pct, unk_frac_truth, avg_unk_truth, unk_frac_extra, avg_unk_extra = _accuracy_pct_series(
        stats, total_key, correct_key, total_inc_key
    )
    _plot_summary_series(
        stats, output_path,
        title=title,
        model_label=model_label,
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


def plot_element_accuracy_pct_summary(all_stats, output_dir=".", model_label=None):
    """
    Plot elemental-only correctness percentage across datasets (correct/true * 100),
    excluding unknown predictions from the denominator. Also overlays unknown fraction.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    output_path = os.path.join(output_dir, f"{model_label or 'model'}_element_accuracy_pct_vs_dataset.png")
    _plot_accuracy_pct_summary(
        stats, output_path,
        title='Elemental Classification Accuracy (Excluding Unknowns)',
        species_label='Element', color='navy', pct_label='Elemental correct (%)',
        total_key='elemental_total_exc', correct_key='elemental_correct_exc',
        total_inc_key='elemental_total',
        log_label='elemental accuracy percent plot',
        model_label=model_label,
    )


def plot_molecule_accuracy_pct_summary(all_stats, output_dir=".", model_label=None):
    """
    Plot molecule-only correctness percentage across datasets (correct/true * 100),
    excluding unknown predictions from the denominator. Also overlays unknown fraction.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    output_path = os.path.join(output_dir, f"{model_label or 'model'}_molecule_accuracy_pct_vs_dataset.png")
    _plot_accuracy_pct_summary(
        stats, output_path,
        title='Molecular Classification Accuracy (Excluding Unknowns)',
        species_label='Molecule', color='purple', pct_label='Molecular correct (%)',
        total_key='molecular_total_exc', correct_key='molecular_correct_exc',
        total_inc_key='molecular_total',
        log_label='molecular accuracy percent plot',
        model_label=model_label,
    )


def plot_element_accuracy_pct_including_unknowns_summary(
    all_stats,
    output_dir=".",
    model_label=None,
):
    """
    Plot elemental-only correctness percentage across datasets (correct/true * 100),
    counting Unknown predictions as incorrect.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    output_path = os.path.join(output_dir, f"{model_label or 'model'}_element_accuracy_pct_including_unknowns_vs_dataset.png")
    _plot_accuracy_pct_summary(
        stats, output_path,
        title='Elemental Classification Accuracy (Including Unknowns)',
        species_label='Element', color='navy', pct_label='Elemental correct incl. unknowns (%)',
        total_key='elemental_total', correct_key='elemental_correct',
        log_label='elemental accuracy percent including unknowns plot',
        model_label=model_label,
    )


def plot_molecule_accuracy_pct_including_unknowns_summary(
    all_stats,
    output_dir=".",
    model_label=None,
):
    """
    Plot molecule-only correctness percentage across datasets (correct/true * 100),
    counting Unknown predictions as incorrect.
    """
    if not all_stats:
        return
    stats = sorted(all_stats, key=lambda x: x.dataset)
    output_path = os.path.join(output_dir, f"{model_label or 'model'}_molecule_accuracy_pct_including_unknowns_vs_dataset.png")
    _plot_accuracy_pct_summary(
        stats, output_path,
        title='Molecular Classification Accuracy (Including Unknowns)',
        species_label='Molecule', color='purple', pct_label='Molecular correct incl. unknowns (%)',
        total_key='molecular_total', correct_key='molecular_correct',
        log_label='molecular accuracy percent including unknowns plot',
        model_label=model_label,
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
