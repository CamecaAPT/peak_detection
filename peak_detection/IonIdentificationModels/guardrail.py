"""Model-agnostic guardrails: unknown-flagging, context rescoring, mixed-unknown flagging,
accuracy-breakdown computation, and their CSV writers.

Peak *ranging* and per-model species classification are not part of this module — a
ClassifierPipeline calls these functions after it has assigned each PeakRange a winning
label/detailed_id/id_score, to apply guardrails that only ever touch PeakRange/DetailedId
fields and config thresholds. RF-specific guardrails (molecule rescue) live in
IonIdentificationModels/RF/molecule_rescue.py instead — see that module's docstring.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np
from pymatgen.core import Composition

from ..models import DetailedId, PeakRange
from ..utils import calculate_iou, is_molecule, simplify_label


def empty_accuracy_breakdown() -> dict:
    """Zeroed accuracy-breakdown shape used both on early-exit and as a failure fallback."""
    return {
        'species_including_unknowns': 0.0,
        'species_excluding_unknowns': 0.0,
        'elemental_including_unknowns': 0.0,
        'elemental_excluding_unknowns': 0.0,
        'molecular_including_unknowns': 0.0,
        'molecular_excluding_unknowns': 0.0,
        'counts': {
            'species_correct_including_unknowns': 0,
            'species_total_including_unknowns': 0,
            'species_correct_excluding_unknowns': 0,
            'species_total_excluding_unknowns': 0,
            'elemental_correct_including_unknowns': 0,
            'elemental_total_including_unknowns': 0,
            'elemental_correct_excluding_unknowns': 0,
            'elemental_total_excluding_unknowns': 0,
            'molecular_correct_including_unknowns': 0,
            'molecular_total_including_unknowns': 0,
            'molecular_correct_excluding_unknowns': 0,
            'molecular_total_excluding_unknowns': 0,
            'unknown_with_truth': 0,
        },
    }


def _is_elemental_label(label: str) -> bool:
    try:
        comp = Composition(str(label))
        if len(comp.elements) != 1:
            return False
        return list(comp.values())[0] == 1
    except Exception:
        return bool(re.fullmatch(r'[A-Z][a-z]?$', str(label)))


def _min_abs_distance_to_samples(sorted_samples: np.ndarray | None, value: float) -> float:
    if sorted_samples is None or len(sorted_samples) == 0:
        return float('inf')
    import bisect
    idx = bisect.bisect_left(sorted_samples, value)
    best = float('inf')
    if idx < len(sorted_samples):
        best = min(best, abs(float(sorted_samples[idx]) - value))
    if idx > 0:
        best = min(best, abs(float(sorted_samples[idx - 1]) - value))
    return best


def _nearest_sample_value(sorted_samples: np.ndarray | None, value: float) -> float | None:
    if sorted_samples is None or len(sorted_samples) == 0:
        return None
    import bisect
    idx = bisect.bisect_left(sorted_samples, value)
    best_val = None
    best_dist = float('inf')
    if idx < len(sorted_samples):
        v = float(sorted_samples[idx])
        d = abs(v - value)
        if d < best_dist:
            best_dist, best_val = d, v
    if idx > 0:
        v = float(sorted_samples[idx - 1])
        d = abs(v - value)
        if d < best_dist:
            best_dist, best_val = d, v
    return best_val


def _best_match_to_species_samples(
    species_key: str,
    mc_val: float,
    mc_samples_by_species: dict,
    mc_threshold: float,
    *,
    allow_scaling_for_elements: bool,
) -> tuple[float, float, float, float | None]:
    """Returns (best_dist, best_scale, scaled_mc, nearest_training_mc).

    - For molecules: always considers scaling to allow charge-aware matching:
      {1, 2, 3, 4, 0.5, 1/3, 0.25}.
    - For elements: only considers scaling when allow_scaling_for_elements=True (same set).
    """
    samples = mc_samples_by_species.get(species_key)
    if samples is None or len(samples) == 0:
        return float('inf'), 1.0, mc_val, None

    min_s = float(samples[0])
    max_s = float(samples[-1])

    consider_scaling = is_molecule(species_key) or allow_scaling_for_elements
    multipliers = (1.0, 2.0, 3.0, 4.0, 0.5, 1.0 / 3.0, 0.25) if consider_scaling else (1.0,)

    best_dist = float('inf')
    best_mult = 1.0
    best_scaled = mc_val
    best_nearest = None

    for mult in multipliers:
        scaled = mc_val * mult
        if mult != 1 and not ((min_s - mc_threshold) <= scaled <= (max_s + mc_threshold)):
            continue
        dist = _min_abs_distance_to_samples(samples, scaled)
        if dist < best_dist:
            best_dist = dist
            best_mult = mult
            best_scaled = scaled
            best_nearest = _nearest_sample_value(samples, scaled)

    return best_dist, best_mult, best_scaled, best_nearest


def _min_abs_distance_to_species_samples(
    species_key: str, mc_val: float, mc_samples_by_species: dict, mc_threshold: float,
) -> float:
    samples = mc_samples_by_species.get(species_key)
    if samples is None or len(samples) == 0:
        return float('inf')

    if not is_molecule(species_key):
        return _min_abs_distance_to_samples(samples, mc_val)

    min_s = float(samples[0])
    max_s = float(samples[-1])
    candidates = [mc_val]
    for mult in (2, 3, 4):
        scaled = mc_val * mult
        if (min_s - mc_threshold) <= scaled <= (max_s + mc_threshold):
            candidates.append(scaled)
    for div in (2, 3, 4):
        scaled = mc_val / float(div)
        if (min_s - mc_threshold) <= scaled <= (max_s + mc_threshold):
            candidates.append(scaled)
    return min(_min_abs_distance_to_samples(samples, c) for c in candidates)


def _format_confidence_unknown_label(det: DetailedId | None, fallback_label: str) -> str:
    parts = []
    if det is not None and det.el1:
        parts.append(f"{det.el1} {float(det.conf1) * 100:.0f}%")
    elif fallback_label:
        parts.append(str(fallback_label))
    if det is not None and det.el2 and float(det.conf2) > 0:
        parts.append(f"{det.el2} {float(det.conf2) * 100:.0f}%")
    return f"Unknown ({', '.join(parts)})" if parts else "Unknown"


def _format_mixed_element_molecule_unknown_label(det: DetailedId) -> str:
    parts = [
        f"{det.el1} {float(det.conf1) * 100:.0f}%",
        f"{det.el2} {float(det.conf2) * 100:.0f}%",
    ]
    return f"Unknown ({' / '.join(parts)})"


def flag_unknown_peaks(
    peaks: list[PeakRange],
    mc_samples_by_species: dict,
    peak_mcs,
    *,
    flag_unknowns: bool,
    mc_threshold: float,
    unknown_confidence_threshold: float,
) -> None:
    """Mutates each PeakRange in place: given a peak whose label/id_score/detailed_id already
    hold its model's winning candidate, relabel it 'Unknown (...)' when the winner is
    physically implausible (mc-distance check) or below a confidence floor. A no-op per peak
    when neither check trips (the caller's already-assigned winner fields are left alone)."""
    if not flag_unknowns:
        return
    for i, p in enumerate(peaks):
        mc_val = float(peak_mcs[i]) if len(peak_mcs) > i else float(getattr(p, 'pos', 0.0) or 0.0)
        winner_full = str(p.label)
        winner_main = re.split(r'\(|\s', winner_full)[0].strip()
        winner_conf = float(p.id_score)
        best_det = p.detailed_id

        winner_key = simplify_label(winner_main)
        winner_dist = _min_abs_distance_to_species_samples(winner_key, mc_val, mc_samples_by_species, mc_threshold)
        is_unphysical = winner_dist > mc_threshold

        confidence_unknown = (
            unknown_confidence_threshold is not None
            and float(unknown_confidence_threshold) > 0
            and best_det is not None
            and getattr(best_det, 'el1', '')
            and str(best_det.el1) != 'Unknown'
            and float(getattr(best_det, 'conf1', 0.0) or 0.0) < float(unknown_confidence_threshold)
        )

        if is_unphysical or winner_main == 'Unknown':
            p.label = f'Unknown ({winner_main})'
            p.id_score, p.is_unknown = 1.0, True
            p.method = 'RF-Unknown'
            p.detailed_id = DetailedId(el1=p.label, conf1=winner_conf, el2='Unknown', conf2=0.0)
        elif confidence_unknown:
            p.label = _format_confidence_unknown_label(best_det, winner_main)
            p.id_score, p.is_unknown = winner_conf, True
            p.method = 'RF-Unknown-LowConf'
            p.detailed_id = DetailedId(
                el1=p.label,
                conf1=float(getattr(best_det, 'conf1', winner_conf) or 0.0),
                el2=str(getattr(best_det, 'el2', '') or ''),
                conf2=float(getattr(best_det, 'conf2', 0.0) or 0.0),
            )


def context_rescore_peaks(
    peaks: list[PeakRange],
    peak_mcs,
    *,
    context_window_da: float,
    context_strength: float,
    context_min_confidence: float,
    context_min_candidate_confidence: float,
    context_override_margin: float,
    context_distance_sigma: float,
    context_rescue_unknown_same_label: bool,
    context_rescue_unknown_min_score: float,
    artifacts_dir: str,
    prefix: str,
) -> list[dict]:
    """Neighbor-weighted candidate rescoring over each peak's existing top-2 candidates.
    Mutates ``peaks`` in place. Writes `<prefix>_context_rescore_overrides.csv` whenever at
    least one override was applied (unconditional on any save-artifacts flag — that's the
    caller's call to gate, matching today's behavior). Returns the override rows."""
    context_override_rows: list[dict] = []

    def _format_context_candidates(cands):
        return "; ".join(f"{c['label']}:{float(c['conf']):.3f}" for c in cands)

    def _parse_unknown_confidence_candidates(label):
        raw = str(label or '').strip()
        if not raw.startswith('Unknown') or '(' not in raw or ')' not in raw:
            return []
        inner = raw.split('(', 1)[1].rsplit(')', 1)[0].strip()
        parsed = []
        for part in inner.split(','):
            text = part.strip()
            m = re.match(r'(.+?)\s+([0-9]+(?:\.[0-9]+)?)%$', text)
            if not m:
                continue
            parsed.append((m.group(1).strip(), float(m.group(2)) / 100.0))
        return parsed

    def _rf_candidates_for_peak(p):
        merged: dict[str, dict] = {}

        def add_candidate(label, conf_value, *, display_label=None):
            raw = str(label or '').strip()
            if not raw:
                return
            if raw.startswith('Unknown'):
                for parsed_label, parsed_conf in _parse_unknown_confidence_candidates(raw):
                    add_candidate(parsed_label, parsed_conf)
                return
            key = simplify_label(re.split(r'\(|,', raw)[0].strip())
            if not key or key == 'Unknown':
                return
            conf_f = max(0.0, float(conf_value or 0.0))
            if key not in merged or conf_f > float(merged[key]['conf']):
                merged[key] = {'label': key, 'display_label': display_label or raw, 'conf': conf_f}

        det = getattr(p, 'detailed_id', None)
        if det is not None:
            add_candidate(str(getattr(det, 'el1', '') or ''), float(getattr(det, 'conf1', 0.0) or 0.0))
            add_candidate(str(getattr(det, 'el2', '') or ''), float(getattr(det, 'conf2', 0.0) or 0.0))
        if not merged:
            add_candidate(str(getattr(p, 'label', '') or ''), float(getattr(p, 'id_score', 0.0) or 0.0))
        return sorted(merged.values(), key=lambda c: float(c['conf']), reverse=True)

    try:
        window = max(0.0, float(context_window_da))
        strength = max(0.0, float(context_strength))
        min_top_conf = float(context_min_confidence)
        min_cand_conf = max(0.0, float(context_min_candidate_confidence))
        margin = float(context_override_margin)
        sigma = max(1e-9, float(context_distance_sigma))
        rescue_min_score = float(context_rescue_unknown_min_score)

        peak_positions = [
            float(peak_mcs[i]) if len(peak_mcs) > i else float(getattr(p, 'pos', 0.0) or 0.0)
            for i, p in enumerate(peaks)
        ]
        all_candidates = [_rf_candidates_for_peak(p) for p in peaks]
        considered = 0

        for i, p in enumerate(peaks):
            target_candidates = [c for c in all_candidates[i] if float(c['conf']) >= min_cand_conf]
            if len(target_candidates) < 2:
                continue

            original = target_candidates[0]
            original_label = str(original['label'])
            original_conf = float(original['conf'])
            if not getattr(p, 'is_unknown', False) and original_conf >= min_top_conf:
                continue

            considered += 1
            target_labels = {str(c['label']) for c in target_candidates}
            support = {label: 0.0 for label in target_labels}
            neighbor_count = 0
            target_mc = peak_positions[i]

            for j, neighbor in enumerate(peaks):
                if i == j:
                    continue
                delta = abs(float(peak_positions[j]) - target_mc)
                if delta > window:
                    continue
                neighbor_candidates = all_candidates[j]
                if not neighbor_candidates:
                    continue
                distance_weight = float(np.exp(-0.5 * (delta / sigma) ** 2))
                contributed = False
                for cand in neighbor_candidates:
                    label = str(cand['label'])
                    if label not in support:
                        continue
                    support[label] += distance_weight * float(cand['conf'])
                    contributed = True
                if contributed:
                    neighbor_count += 1

            rescored = {
                str(c['label']): float(c['conf']) + strength * support.get(str(c['label']), 0.0)
                for c in target_candidates
            }
            best_label = max(rescored, key=rescored.get)
            best_score = float(rescored[best_label])
            original_score = float(rescored.get(original_label, original_conf))
            same_label_unknown_rescue = (
                bool(context_rescue_unknown_same_label)
                and bool(getattr(p, 'is_unknown', False))
                and best_label == original_label
                and support.get(original_label, 0.0) > 0.0
                and best_score >= rescue_min_score
                and best_score >= (original_conf + margin)
            )
            candidate_switch = (
                best_label != original_label
                and best_score >= (original_score + margin)
            )
            if not candidate_switch and not same_label_unknown_rescue:
                continue
            override_reason = 'same_label_unknown_rescue' if same_label_unknown_rescue else 'candidate_switch'

            old_label = str(getattr(p, 'label', '') or '')
            old_method = str(getattr(p, 'method', '') or '')
            old_is_unknown = bool(getattr(p, 'is_unknown', False))
            old_det = getattr(p, 'detailed_id', None)
            old_top2 = target_candidates[1] if len(target_candidates) > 1 else None

            p.label = best_label
            p.id_score = min(1.0, best_score)
            p.is_unknown = False
            p.method = f"{old_method}+context" if old_method else 'RF-context'
            p.detailed_id = DetailedId(
                el1=best_label,
                conf1=min(1.0, best_score),
                el2=original_label if original_label != best_label else (str(old_top2['label']) if old_top2 else ''),
                conf2=original_conf if original_label != best_label else (float(old_top2['conf']) if old_top2 else 0.0),
            )
            all_candidates[i] = _rf_candidates_for_peak(p)

            context_override_rows.append({
                'peak_start': float(getattr(p, 'start', np.nan)),
                'peak_end': float(getattr(p, 'end', np.nan)),
                'peak_mc': target_mc,
                'old_label': old_label,
                'old_method': old_method,
                'old_is_unknown': old_is_unknown,
                'old_top1': original_label,
                'old_top1_conf': original_conf,
                'old_top2': str(getattr(old_det, 'el2', '') or (str(old_top2['label']) if old_top2 else '')) if old_det is not None else (str(old_top2['label']) if old_top2 else ''),
                'old_top2_conf': float(getattr(old_det, 'conf2', 0.0) or 0.0) if old_det is not None else (float(old_top2['conf']) if old_top2 else 0.0),
                'new_label': best_label,
                'new_score': best_score,
                'original_candidate': original_label,
                'original_rescored_score': original_score,
                'support_new': support.get(best_label, 0.0),
                'support_original': support.get(original_label, 0.0),
                'neighbor_count': neighbor_count,
                'override_reason': override_reason,
                'candidates_before': _format_context_candidates(target_candidates),
                'scores_after': "; ".join(f"{k}:{v:.3f}" for k, v in sorted(rescored.items())),
                'context_window_da': window,
                'context_strength': strength,
                'context_distance_sigma': sigma,
                'context_override_margin': margin,
                'context_rescue_unknown_min_score': rescue_min_score,
            })

        if context_override_rows:
            print(f"  Context RF rescoring overrides applied: {len(context_override_rows)}/{considered} candidates")
            _ctx_dir = artifacts_dir or prefix
            os.makedirs(_ctx_dir, exist_ok=True)
            context_overrides_path = os.path.join(_ctx_dir, f"{prefix}_context_rescore_overrides.csv")
            cols = [
                'peak_start', 'peak_end', 'peak_mc',
                'old_label', 'old_method', 'old_is_unknown',
                'old_top1', 'old_top1_conf', 'old_top2', 'old_top2_conf',
                'new_label', 'new_score',
                'original_candidate', 'original_rescored_score',
                'support_new', 'support_original', 'neighbor_count',
                'override_reason',
                'candidates_before', 'scores_after',
                'context_window_da', 'context_strength',
                'context_distance_sigma', 'context_override_margin',
                'context_rescue_unknown_min_score',
            ]
            with open(context_overrides_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(context_override_rows)
        else:
            print(f"  Context RF rescoring considered {considered} candidates; no overrides")
    except Exception as e:
        print(f"  [Warn] Context RF rescoring failed: {e}")

    return context_override_rows


def flag_high_confidence_mixed_unknowns(
    peaks: list[PeakRange],
    *,
    flag_unknowns: bool,
    unknown_mixed_element_molecule_confidence_threshold: float,
) -> int:
    """Flags a peak Unknown when its top-2 candidates are a high-confidence element +
    molecule mix (ambiguous between an elemental and a molecular ID). Mutates ``peaks`` in
    place; returns the number flagged."""
    threshold = float(unknown_mixed_element_molecule_confidence_threshold or 0.0)
    if not flag_unknowns or threshold <= 0:
        return 0
    flagged = 0
    for p in peaks:
        if getattr(p, 'is_unknown', False):
            continue
        det = getattr(p, 'detailed_id', None)
        if det is None or not det.el1 or not det.el2:
            continue
        conf1 = float(getattr(det, 'conf1', 0.0) or 0.0)
        conf2 = float(getattr(det, 'conf2', 0.0) or 0.0)
        if conf1 < threshold or conf2 < threshold:
            continue
        label1 = simplify_label(str(det.el1))
        label2 = simplify_label(str(det.el2))
        if not label1 or not label2 or label1 == 'Unknown' or label2 == 'Unknown':
            continue
        mixed_element_molecule = (
            (_is_elemental_label(label1) and is_molecule(label2))
            or (_is_elemental_label(label2) and is_molecule(label1))
        )
        if not mixed_element_molecule:
            continue
        p.label = _format_mixed_element_molecule_unknown_label(det)
        p.id_score = max(conf1, conf2)
        p.is_unknown = True
        p.method = f"{p.method}+mixed-unknown" if p.method else "RF-mixed-unknown"
        flagged += 1
    return flagged


def compute_accuracy_breakdown(peaks: list[PeakRange], truth_data: list[PeakRange], *, rf_accuracy_top_n: int = 1) -> dict:
    """Compute counts for truth-matched peaks: overall/elemental-only/molecular-only,
    including and excluding unknowns."""
    def _get_pred_labels(pr: PeakRange) -> list[str]:
        if getattr(pr, 'is_unknown', False):
            return ['Unknown']
        top_n = max(1, int(rf_accuracy_top_n or 1))
        labels = []
        if pr.detailed_id is not None and pr.detailed_id.el1:
            labels.append(str(pr.detailed_id.el1))
            if top_n >= 2 and pr.detailed_id.el2 and float(getattr(pr.detailed_id, 'conf2', 0.0) or 0.0) > 0:
                labels.append(str(pr.detailed_id.el2))
            return labels[:top_n]
        raw = str(pr.label) if getattr(pr, 'label', None) is not None else ''
        return [re.split(r'\(|,', raw)[0].strip()] if raw else ['Unknown']

    total_inc = correct_inc = 0
    total_exc = correct_exc = 0
    total_ele_inc = correct_ele_inc = 0
    total_ele_exc = correct_ele_exc = 0
    total_mol_inc = correct_mol_inc = 0
    total_mol_exc = correct_mol_exc = 0
    unk_with_truth = 0

    for pr in peaks:
        best_iou, best_truth = 0.0, None
        for t in truth_data:
            iou_val = calculate_iou(pr, t)
            if iou_val > best_iou:
                best_iou, best_truth = iou_val, t
        if best_iou <= 0.1 or best_truth is None:
            continue
        true_label = str(best_truth.label)
        if not true_label or true_label == 'Unknown':
            continue
        pred_labels = _get_pred_labels(pr)
        is_pred_unknown = (not pred_labels) or (pred_labels[0] == 'Unknown')
        if is_pred_unknown:
            unk_with_truth += 1

        total_inc += 1
        is_ele_true = _is_elemental_label(true_label)
        if is_ele_true:
            total_ele_inc += 1
        else:
            total_mol_inc += 1

        is_correct = (not is_pred_unknown) and any(
            simplify_label(true_label) == simplify_label(pred_label)
            for pred_label in pred_labels[:max(1, int(rf_accuracy_top_n or 1))]
            if pred_label and pred_label != 'Unknown'
        )
        if is_correct:
            correct_inc += 1
            if is_ele_true:
                correct_ele_inc += 1
            else:
                correct_mol_inc += 1

        if not is_pred_unknown:
            total_exc += 1
            if is_ele_true:
                total_ele_exc += 1
            else:
                total_mol_exc += 1
            if is_correct:
                correct_exc += 1
                if is_ele_true:
                    correct_ele_exc += 1
                else:
                    correct_mol_exc += 1

    def pct(c, t):
        return (c / t * 100.0) if t > 0 else 0.0

    return {
        'species_including_unknowns': pct(correct_inc, total_inc),
        'species_excluding_unknowns': pct(correct_exc, total_exc),
        'elemental_including_unknowns': pct(correct_ele_inc, total_ele_inc),
        'elemental_excluding_unknowns': pct(correct_ele_exc, total_ele_exc),
        'molecular_including_unknowns': pct(correct_mol_inc, total_mol_inc),
        'molecular_excluding_unknowns': pct(correct_mol_exc, total_mol_exc),
        'counts': {
            'species_correct_including_unknowns': correct_inc,
            'species_total_including_unknowns': total_inc,
            'species_correct_excluding_unknowns': correct_exc,
            'species_total_excluding_unknowns': total_exc,
            'elemental_correct_including_unknowns': correct_ele_inc,
            'elemental_total_including_unknowns': total_ele_inc,
            'elemental_correct_excluding_unknowns': correct_ele_exc,
            'elemental_total_excluding_unknowns': total_ele_exc,
            'molecular_correct_including_unknowns': correct_mol_inc,
            'molecular_total_including_unknowns': total_mol_inc,
            'molecular_correct_excluding_unknowns': correct_mol_exc,
            'molecular_total_excluding_unknowns': total_mol_exc,
            'unknown_with_truth': unk_with_truth,
        },
    }


def write_detailed_results_csv(
    peaks: list[PeakRange], truth_data: list[PeakRange], *, save_artifacts: bool, artifacts_dir: str, prefix: str,
) -> list[dict]:
    """Builds the per-peak predicted-vs-truth rows and, when save_artifacts, writes them to
    `<prefix>_detailed_results.csv`. Always returns the rows regardless of save_artifacts —
    write_unknown_peak_error_report needs them either way."""
    detailed_rows = []
    for p in peaks:
        best_iou, best_truth = 0, None
        for t in truth_data:
            iou_val = calculate_iou(p, t)
            if iou_val > best_iou:
                best_iou, best_truth = iou_val, t
        det = p.detailed_id if p.detailed_id is not None else DetailedId(el1='Unknown')
        detailed_rows.append({
            'predicted peak start': p.start, 'predicted peak end': p.end,
            'true peak start': best_truth.start if best_iou > 0.1 else '',
            'true peak end': best_truth.end if best_iou > 0.1 else '',
            'true element label': best_truth.label if best_iou > 0.1 else 'Unknown',
            'pred display label': p.label,
            'pred element label 1': det.el1, 'pred confidence 1': round(det.conf1, 3),
            'pred element label 2': det.el2, 'pred confidence 2': round(det.conf2, 3),
            'discarded': p.is_unknown,
        })

    if save_artifacts:
        out_dir = artifacts_dir or prefix
        os.makedirs(out_dir, exist_ok=True)
        detailed_results_path = os.path.join(out_dir, f"{prefix}_detailed_results.csv")
        with open(detailed_results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'predicted peak start', 'predicted peak end', 'true peak start',
                'true peak end', 'true element label', 'pred display label',
                'pred element label 1', 'pred confidence 1', 'pred element label 2',
                'pred confidence 2', 'discarded',
            ])
            writer.writeheader()
            writer.writerows(detailed_rows)

    return detailed_rows


def write_unknown_peak_error_report(
    detailed_rows: list[dict],
    mc_samples_by_species: dict,
    x_exp,
    spectrum_log,
    *,
    save_artifacts: bool,
    flag_unknowns: bool,
    mc_threshold: float,
    artifacts_dir: str,
    prefix: str,
) -> None:
    """Writes `<prefix>_unknown_peak_error_report.csv` diagnosing why each discarded
    ('Unknown') peak was flagged, when there's at least one such peak. No-op otherwise.

    Unlike the original inline code, this never needs to rebuild ``mc_samples_by_species``
    itself: the caller (RFClassifierPipeline.run) always builds it once, up front, before
    any guardrail call — so it's never empty here."""
    if not (save_artifacts and flag_unknowns and any(bool(r.get('discarded')) for r in detailed_rows)):
        return

    def _parse_unknown_reason(label1: str) -> str:
        s = str(label1)
        if not s.startswith('Unknown'):
            return ''
        if '(' in s and ')' in s:
            return s.split('(', 1)[1].rsplit(')', 1)[0].strip()
        return ''

    def _safe_float(v):
        try:
            if v is None:
                return None
            if isinstance(v, str) and v.strip() == '':
                return None
            return float(v)
        except Exception:
            return None

    y_exp = spectrum_log.numpy() if hasattr(spectrum_log, 'numpy') else spectrum_log

    def _peak_max_mc(start, end):
        if start is None or end is None:
            return None
        try:
            mask = (x_exp >= float(start)) & (x_exp <= float(end))
            if np.any(mask):
                peak_idx = int(np.argmax(np.asarray(y_exp)[mask]))
                return float(np.asarray(x_exp)[mask][peak_idx])
            return float(start + end) / 2.0
        except Exception:
            return None

    report_rows = []
    for row in detailed_rows:
        discarded = str(row.get('discarded', '')).lower() in ('true', '1', 'yes')
        if not discarded:
            continue

        true_label_raw = str(row.get('true element label', 'Unknown'))
        true_label_simple = simplify_label(true_label_raw) if true_label_raw else 'Unknown'

        reason_raw = _parse_unknown_reason(str(row.get('pred element label 1', '')))
        reason_simple = simplify_label(reason_raw) if reason_raw else ''

        ts = _safe_float(row.get('true peak start'))
        te = _safe_float(row.get('true peak end'))
        ps = _safe_float(row.get('predicted peak start'))
        pe = _safe_float(row.get('predicted peak end'))

        true_peak_mc_max = _peak_max_mc(ts, te) if (ts is not None and te is not None) else None
        pred_peak_mc_max = _peak_max_mc(ps, pe) if (ps is not None and pe is not None) else None
        mc_used = true_peak_mc_max if true_peak_mc_max is not None else pred_peak_mc_max

        dist_true_code = float('inf')
        mult_true_code = 1
        scaled_true_code = mc_used if mc_used is not None else float('nan')
        nearest_true_code = None

        dist_true_scaled_any = float('inf')
        mult_true_scaled_any = 1
        scaled_true_scaled_any = mc_used if mc_used is not None else float('nan')
        nearest_true_scaled_any = None

        if mc_used is not None and true_label_simple and true_label_simple != 'Unknown':
            dist_true_code, mult_true_code, scaled_true_code, nearest_true_code = _best_match_to_species_samples(
                true_label_simple, float(mc_used), mc_samples_by_species, mc_threshold, allow_scaling_for_elements=False,
            )
            dist_true_scaled_any, mult_true_scaled_any, scaled_true_scaled_any, nearest_true_scaled_any = _best_match_to_species_samples(
                true_label_simple, float(mc_used), mc_samples_by_species, mc_threshold, allow_scaling_for_elements=True,
            )

        dist_reason_code = float('inf')
        mult_reason_code = 1
        scaled_reason_code = mc_used if mc_used is not None else float('nan')
        nearest_reason_code = None

        dist_reason_scaled_any = float('inf')
        mult_reason_scaled_any = 1
        scaled_reason_scaled_any = mc_used if mc_used is not None else float('nan')
        nearest_reason_scaled_any = None

        if mc_used is not None and reason_simple and reason_simple != 'Unknown':
            dist_reason_code, mult_reason_code, scaled_reason_code, nearest_reason_code = _best_match_to_species_samples(
                reason_simple, float(mc_used), mc_samples_by_species, mc_threshold, allow_scaling_for_elements=False,
            )
            dist_reason_scaled_any, mult_reason_scaled_any, scaled_reason_scaled_any, nearest_reason_scaled_any = _best_match_to_species_samples(
                reason_simple, float(mc_used), mc_samples_by_species, mc_threshold, allow_scaling_for_elements=True,
            )

        report_rows.append({
            'predicted peak start': row.get('predicted peak start', ''),
            'predicted peak end': row.get('predicted peak end', ''),
            'true peak start': row.get('true peak start', ''),
            'true peak end': row.get('true peak end', ''),
            'true_peak_mc_max': true_peak_mc_max if true_peak_mc_max is not None else '',
            'pred_peak_mc_max': pred_peak_mc_max if pred_peak_mc_max is not None else '',
            'mc_used_for_distance': mc_used if mc_used is not None else '',
            'true element label': true_label_raw,
            'true label simplified': true_label_simple,
            'pred element label 1': row.get('pred element label 1', ''),
            'unknown reason raw': reason_raw,
            'unknown reason simplified': reason_simple,
            'true is molecule': bool(is_molecule(true_label_simple)) if true_label_simple else False,
            'training missing (true)': bool(np.isinf(dist_true_scaled_any)),
            'dist_true_code': dist_true_code,
            'mult_true_code': mult_true_code,
            'scaled_mc_true_code': scaled_true_code,
            'nearest_training_mc_true_code': nearest_true_code if nearest_true_code is not None else '',
            'within_threshold_true_code': (dist_true_code <= mc_threshold) if not np.isinf(dist_true_code) else False,
            'dist_true_scaled_any': dist_true_scaled_any,
            'mult_true_scaled_any': mult_true_scaled_any,
            'scaled_mc_true_scaled_any': scaled_true_scaled_any,
            'nearest_training_mc_true_scaled_any': nearest_true_scaled_any if nearest_true_scaled_any is not None else '',
            'within_threshold_true_scaled_any': (dist_true_scaled_any <= mc_threshold) if not np.isinf(dist_true_scaled_any) else False,
            'dist_reason_code': dist_reason_code,
            'mult_reason_code': mult_reason_code,
            'scaled_mc_reason_code': scaled_reason_code,
            'nearest_training_mc_reason_code': nearest_reason_code if nearest_reason_code is not None else '',
            'dist_reason_scaled_any': dist_reason_scaled_any,
            'mult_reason_scaled_any': mult_reason_scaled_any,
            'scaled_mc_reason_scaled_any': scaled_reason_scaled_any,
            'nearest_training_mc_reason_scaled_any': nearest_reason_scaled_any if nearest_reason_scaled_any is not None else '',
        })

    out_dir = artifacts_dir or prefix
    os.makedirs(out_dir, exist_ok=True)
    unknown_report_path = os.path.join(out_dir, f"{prefix}_unknown_peak_error_report.csv")
    try:
        with open(unknown_report_path, 'w', newline='') as f:
            fieldnames = [
                'predicted peak start', 'predicted peak end',
                'true peak start', 'true peak end',
                'true_peak_mc_max', 'pred_peak_mc_max', 'mc_used_for_distance',
                'true element label', 'true label simplified',
                'pred element label 1',
                'unknown reason raw', 'unknown reason simplified',
                'true is molecule',
                'training missing (true)',
                'dist_true_code', 'mult_true_code', 'scaled_mc_true_code', 'nearest_training_mc_true_code', 'within_threshold_true_code',
                'dist_true_scaled_any', 'mult_true_scaled_any', 'scaled_mc_true_scaled_any', 'nearest_training_mc_true_scaled_any', 'within_threshold_true_scaled_any',
                'dist_reason_code', 'mult_reason_code', 'scaled_mc_reason_code', 'nearest_training_mc_reason_code',
                'dist_reason_scaled_any', 'mult_reason_scaled_any', 'scaled_mc_reason_scaled_any', 'nearest_training_mc_reason_scaled_any',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"  Unknown peak error report saved: {unknown_report_path} ({len(report_rows)} rows)")
    except OSError as e:
        print(f"  [Warn] Failed to write unknown peak error report: {unknown_report_path} ({e})")
