"""RF-specific molecule-rescue guardrails: the two steps that retrain and rerun a *second*
RF model restricted to molecular species, as the rescue mechanism for both unknown peaks
and elemental winners. These are RF-specific by construction (they depend on RF's
train/infer functions) and therefore live here rather than in the shared
IonIdentificationModels/guardrail.py.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np

from ...models import DetailedId, PeakRange
from ...utils import is_molecule, simplify_label
from ..guardrail import _best_match_to_species_samples, _is_elemental_label, _min_abs_distance_to_species_samples


def train_molecule_only_rf(
    *,
    training_data_path: str,
    truth_molecules: list[str],
    training_num_files: int,
    neighbor_threshold: float,
    use_signature: bool,
    augment_molecule_training_charge_ratios: bool,
    load_ion_training_data_fn,
    create_rf_model_fn,
):
    """Trains a second RF restricted to molecular species only (shared by both rescue
    guardrails below). Returns (scaler, model, target_decoder), or None if there wasn't
    enough training data for any of ``truth_molecules``."""
    X_train_mol2, ions_train_mol2 = load_ion_training_data_fn(
        path=training_data_path,
        element_list=truth_molecules,
        elements_to_get_molecules=[],
        num_files=int(training_num_files),
        neighbor_threshold=neighbor_threshold,
        use_signature=use_signature,
        augment_molecule_charge_ratios=bool(augment_molecule_training_charge_ratios),
    )
    if len(X_train_mol2) == 0:
        return None
    return create_rf_model_fn(X_train_mol2, ions_train_mol2)


def rescue_unknowns_with_molecule_rf(
    peaks: list[PeakRange],
    x_exp,
    spectrum_log,
    mol_rf,
    mc_samples_by_species: dict,
    *,
    mc_threshold: float,
    molecule_rf_threshold: float,
    eff_neighbor_threshold: float,
    use_signature: bool,
    run_rf_model_fn,
) -> None:
    """Second-pass molecule-only RF on peaks currently flagged Unknown. Mutates ``peaks`` in
    place; recovers a peak (clears is_unknown) when the molecule RF's top candidate beats
    ``molecule_rf_threshold`` and is physically plausible (mc-distance <= mc_threshold)."""
    if mol_rf is None:
        return
    unknown_indices = [i for i, p in enumerate(peaks) if getattr(p, 'is_unknown', False)]
    if not unknown_indices:
        return
    scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2 = mol_rf
    unknown_peaks = [peaks[i] for i in unknown_indices]
    mol_elements2, mol_confs2, mol_details2, mol_mcs2 = run_rf_model_fn(
        unknown_peaks, x_exp, spectrum_log, scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2,
        neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature,
    )

    recovered = 0
    for local_i, global_i in enumerate(unknown_indices):
        det2 = mol_details2[local_i]
        pred2 = str(det2.el1) if det2 else ''
        conf2 = float(mol_confs2[local_i]) if mol_confs2 else 0.0
        if not pred2 or pred2 == 'Unknown':
            continue
        if conf2 < float(molecule_rf_threshold):
            continue
        dist2 = _min_abs_distance_to_species_samples(simplify_label(pred2), float(mol_mcs2[local_i]), mc_samples_by_species, mc_threshold)
        if dist2 > mc_threshold:
            continue

        p = peaks[global_i]
        p.label = mol_elements2[local_i]
        p.id_score = conf2
        p.is_unknown = False
        p.method = 'RF-mol2'
        p.detailed_id = det2
        recovered += 1

    if recovered:
        print(f"  Molecule RF recovered {recovered}/{len(unknown_indices)} unknown peaks.")


def rescue_elements_with_molecule_rf(
    peaks: list[PeakRange],
    x_exp,
    spectrum_log,
    mol_rf,
    mc_samples_by_species: dict,
    *,
    mc_threshold: float,
    molecule_rf_rescue_threshold: float,
    molecule_rf_rescue_margin: float,
    molecule_rf_rescue_score_margin: float,
    molecule_rf_rescue_dist_margin: float,
    eff_neighbor_threshold: float,
    use_signature: bool,
    run_rf_model_fn,
) -> tuple[dict, list[dict]]:
    """Molecule-rescue pass on peaks currently labeled a single element (not unknown):
    reruns the molecule-only RF on each and either overrides to the molecule (clearly
    better physical + confidence fit) or records it as a 'mixed candidate' (comparable fit,
    kept as the detailed_id's secondary slot). Mutates ``peaks`` in place. Returns
    (rescue_stats, rescue_override_rows) — pass rescue_override_rows to
    write_molecule_rescue_candidates_csv after label-remapping."""
    rescue_stats = {'considered': 0, 'overrides': 0, 'mixed_candidates': 0}
    rescue_override_rows: list[dict] = []
    if mol_rf is None:
        return rescue_stats, rescue_override_rows
    scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2 = mol_rf

    candidate_indices = []
    candidate_peaks = []
    for i, p in enumerate(peaks):
        if getattr(p, 'is_unknown', False):
            continue
        det = getattr(p, 'detailed_id', None)
        pred1 = str(det.el1) if det is not None and det.el1 else (re.split(r'\(|,', str(p.label))[0].strip() if p.label else '')
        if not pred1 or pred1 == 'Unknown':
            continue
        if not _is_elemental_label(pred1):
            continue
        candidate_indices.append(i)
        candidate_peaks.append(p)

    if not candidate_peaks:
        return rescue_stats, rescue_override_rows

    mol_elements_r, mol_confs_r, mol_details_r, mol_mcs_r = run_rf_model_fn(
        candidate_peaks, x_exp, spectrum_log, scaler_rf_mol2, model_rf_mol2, target_decoder_rf_mol2,
        neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature,
    )

    for local_i, global_i in enumerate(candidate_indices):
        rescue_stats['considered'] += 1
        p = peaks[global_i]

        det_ele = getattr(p, 'detailed_id', None)
        ele_pred = str(det_ele.el1) if det_ele is not None and det_ele.el1 else ''
        ele_conf = float(det_ele.conf1) if det_ele is not None else float(getattr(p, 'id_score', 0.0) or 0.0)
        ele_key = simplify_label(ele_pred) if ele_pred else ''

        det_m = mol_details_r[local_i]
        mol_pred = str(det_m.el1) if det_m is not None and det_m.el1 else ''
        mol_conf = float(mol_confs_r[local_i]) if mol_confs_r else 0.0
        mol_key = simplify_label(mol_pred) if mol_pred else ''
        if not mol_key or mol_key == 'Unknown' or not is_molecule(mol_key):
            continue
        if mol_conf < float(molecule_rf_rescue_threshold):
            continue

        mc_val = float(mol_mcs_r[local_i]) if len(mol_mcs_r) > local_i else float(getattr(p, 'pos', 0.0) or 0.0)
        dist_m = _min_abs_distance_to_species_samples(mol_key, mc_val, mc_samples_by_species, mc_threshold)
        if dist_m > mc_threshold:
            continue

        dist_e = _min_abs_distance_to_species_samples(ele_key, mc_val, mc_samples_by_species, mc_threshold) if ele_key else float('inf')
        dist_margin = float(molecule_rf_rescue_dist_margin)
        better_physical_fit = bool(np.isinf(dist_e) or (dist_m + dist_margin < dist_e))
        comparable_physical_fit = bool(np.isinf(dist_e) or (dist_m <= dist_e + dist_margin))
        q_m = max(0.0, 1.0 - (dist_m / mc_threshold)) if mc_threshold > 0 else 0.0
        q_e = max(0.0, 1.0 - (dist_e / mc_threshold)) if (mc_threshold > 0 and not np.isinf(dist_e)) else 0.0
        score_m = mol_conf * q_m
        score_e = ele_conf * q_e

        conf_margin = float(molecule_rf_rescue_margin)
        score_margin = float(molecule_rf_rescue_score_margin)
        rescue_action = ""
        rescue_reason = ""
        if better_physical_fit and mol_conf >= (ele_conf + conf_margin):
            rescue_action, rescue_reason = "override", "conf_margin"
        elif better_physical_fit and score_m >= (score_e + score_margin):
            rescue_action, rescue_reason = "override", "score_margin"
        elif comparable_physical_fit and mol_conf >= max(0.0, ele_conf - conf_margin):
            rescue_action, rescue_reason = "mixed_candidate", "conf_close"
        elif comparable_physical_fit and score_m >= max(0.0, score_e - score_margin):
            rescue_action, rescue_reason = "mixed_candidate", "score_close"

        if not rescue_action:
            continue

        should_override = rescue_action == "override"

        mol_best_dist, mol_best_scale, mol_scaled_mc, mol_nearest_train = _best_match_to_species_samples(
            mol_key, mc_val, mc_samples_by_species, mc_threshold, allow_scaling_for_elements=False,
        )
        ele_best_dist, ele_best_scale, ele_scaled_mc, ele_nearest_train = (
            _best_match_to_species_samples(ele_key, mc_val, mc_samples_by_species, mc_threshold, allow_scaling_for_elements=False)
            if ele_key else (float('inf'), 1.0, mc_val, None)
        )

        rescue_override_rows.append({
            'peak_start': float(getattr(p, 'start', np.nan)),
            'peak_end': float(getattr(p, 'end', np.nan)),
            'peak_mc': mc_val,
            'element_pred_simple': ele_key if ele_key else ele_pred,
            'element_conf': ele_conf,
            'element_dist': float(dist_e) if not np.isinf(dist_e) else '',
            'element_best_scale': ele_best_scale,
            'element_scaled_mc': ele_scaled_mc,
            'element_nearest_training_mc': ele_nearest_train if ele_nearest_train is not None else '',
            'molecule_pred_simple': mol_key,
            'molecule_conf': mol_conf,
            'molecule_dist': float(dist_m),
            'molecule_best_scale': mol_best_scale,
            'molecule_scaled_mc': mol_scaled_mc,
            'molecule_nearest_training_mc': mol_nearest_train if mol_nearest_train is not None else '',
            'q_element': q_e,
            'q_molecule': q_m,
            'score_element': score_e,
            'score_molecule': score_m,
            'rescue_action': rescue_action,
            'rescue_reason': rescue_reason,
        })

        if should_override:
            p.label = mol_elements_r[local_i]
            p.id_score = mol_conf
            p.is_unknown = False
            p.method = 'RF-mol-rescue'
            p.detailed_id = DetailedId(el1=mol_pred, conf1=mol_conf, el2=ele_pred, conf2=ele_conf)
            rescue_stats['overrides'] += 1
        else:
            p.is_unknown = False
            p.method = f"{p.method}+mol-candidate" if p.method else "RF-mol-candidate"
            p.detailed_id = DetailedId(el1=ele_pred, conf1=ele_conf, el2=mol_pred, conf2=mol_conf)
            rescue_stats['mixed_candidates'] += 1

    if rescue_stats['overrides'] or rescue_stats['mixed_candidates']:
        print(
            "  Molecule rescue accepted: "
            f"{rescue_stats['overrides']} overrides, "
            f"{rescue_stats['mixed_candidates']} mixed candidates / "
            f"{rescue_stats['considered']} candidates"
        )
    else:
        print(f"  Molecule rescue considered {rescue_stats['considered']} candidates; no accepted rescues")

    return rescue_stats, rescue_override_rows


def write_molecule_rescue_candidates_csv(
    rescue_override_rows: list[dict], label_map: dict, artifacts_dir: str, prefix: str,
) -> None:
    """Writes `<prefix>_molecule_rescue_candidates.csv`, remapping each row's simplified
    element/molecule labels back to their RRNG display labels via ``label_map`` first.
    No-op when there are no rows."""
    if not rescue_override_rows:
        return
    mapped_rows = []
    for r in rescue_override_rows:
        r2 = dict(r)
        r2['element_pred'] = label_map.get(str(r2.get('element_pred_simple', '')), str(r2.get('element_pred_simple', '')))
        r2['molecule_pred'] = label_map.get(str(r2.get('molecule_pred_simple', '')), str(r2.get('molecule_pred_simple', '')))
        mapped_rows.append(r2)

    out_dir = artifacts_dir or prefix
    os.makedirs(out_dir, exist_ok=True)
    rescue_overrides_path = os.path.join(out_dir, f"{prefix}_molecule_rescue_candidates.csv")
    cols = [
        'peak_start', 'peak_end', 'peak_mc',
        'element_pred', 'element_pred_simple', 'element_conf', 'element_dist',
        'element_best_scale', 'element_scaled_mc', 'element_nearest_training_mc',
        'molecule_pred', 'molecule_pred_simple', 'molecule_conf', 'molecule_dist',
        'molecule_best_scale', 'molecule_scaled_mc', 'molecule_nearest_training_mc',
        'q_element', 'q_molecule', 'score_element', 'score_molecule',
        'rescue_action', 'rescue_reason',
    ]
    try:
        with open(rescue_overrides_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(mapped_rows)
    except Exception as e:
        print(f"  [Warn] Failed writing rescue overrides CSV ({e})")
