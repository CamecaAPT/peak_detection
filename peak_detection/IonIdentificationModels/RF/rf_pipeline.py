"""RF species-classification pipeline: registers "rf" with the classifier registry.

RFClassifierPipeline.run() is the real orchestrator: ranging (yolo_detection.run_yolo_ranging)
-> RF train/infer (rf_model.py) -> shared guardrails (IonIdentificationModels/guardrail.py) ->
RF-specific molecule rescue (RF/molecule_rescue.py) -> accuracy breakdown -> CSV artifacts.
Deleting this folder removes the RF model entirely (config + pipeline + underlying
rf_model.py all live together here).

flat_rf_kwargs() maps the merged model YAML (nested: ranging/training/guardrails.*) onto
the flat keyword-argument names this pipeline (and detect_peaks_headless/process_dataset,
which build ctx.cfg from their own already-flat parameters) uses. ctx.cfg is that flat
dict, not the nested one — flat_rf_kwargs() itself is unchanged, but it's now called once
at each entry point's CLI boundary rather than inside this pipeline.
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
import yaml

from ...classifiers import register
from ...classifiers.base import ClassifierContext, ClassifierPipeline
from ...models import PeakRange
from ...training import build_empirical_mc_samples, load_ion_training_data
from ...utils import is_molecule, simplify_label, yaml_safe
from ...yolo_detection import run_yolo_ranging
from .. import guardrail
from . import molecule_rescue
from .rf_model import create_RF_model, run_RF_model


def flat_rf_kwargs(cfg: dict) -> dict:
    """Flatten a merged universal+rf.yaml config dict into the flat kwarg names shared by
    process_dataset and detect_peaks_headless (both use the same parameter names, except
    they call it unknown_molecule_rf_threshold where this pipeline calls it
    molecule_rf_threshold)."""
    ranging = cfg.get("ranging", {})
    training = cfg.get("training", {})
    guardrails = cfg.get("guardrails", {})
    unknown_flagging = guardrails.get("unknown_flagging", {})
    mixed_unknown = guardrails.get("mixed_unknown", {})
    context_rescore = guardrails.get("context_rescore", {})
    molecule_rescue_cfg = guardrails.get("molecule_rescue", {})
    unknown_molecule_rf = guardrails.get("unknown_molecule_rf", {})

    return dict(
        # Ranging (fixed across all models)
        yolo_weights=ranging.get("yolo_weights", "best_v0_2026-06-23.pt"),
        iou=ranging.get("iou", 0.01),
        conf=ranging.get("conf", 0.05),
        max_det=ranging.get("max_det", 2000),
        mc_min=ranging.get("mc_min", 0.0),
        mc_max=ranging.get("mc_max", 307.2),
        # Training
        training_path=training.get("training_path"),
        training_num_files=training.get("training_num_files", 10000),
        augment_molecule_training_charge_ratios=training.get("augment_molecule_training_charge_ratios", False),
        include_molecules=training.get("include_molecules", False),
        use_neighborhood=training.get("use_neighborhood", False),
        neighbor_threshold=training.get("neighbor_threshold", 2.0),
        use_signature=training.get("use_signature", False),
        # Unknown flagging
        flag_unknowns=unknown_flagging.get("flag_unknowns", True),
        mc_threshold=unknown_flagging.get("mc_threshold", 0.2),
        unknown_confidence_threshold=unknown_flagging.get("unknown_confidence_threshold", 0.6),
        rf_accuracy_top_n=unknown_flagging.get("rf_accuracy_top_n", 1),
        unknown_mixed_element_molecule_confidence_threshold=mixed_unknown.get(
            "unknown_mixed_element_molecule_confidence_threshold", 0.95),
        # Context rescoring
        context_rescore=context_rescore.get("enabled", False),
        context_window_da=context_rescore.get("context_window_da", 2.0),
        context_strength=context_rescore.get("context_strength", 0.35),
        context_min_confidence=context_rescore.get("context_min_confidence", 0.75),
        context_min_candidate_confidence=context_rescore.get("context_min_candidate_confidence", 0.05),
        context_override_margin=context_rescore.get("context_override_margin", 0.05),
        context_distance_sigma=context_rescore.get("context_distance_sigma", 0.75),
        context_rescue_unknown_same_label=context_rescore.get("context_rescue_unknown_same_label", True),
        context_rescue_unknown_min_score=context_rescore.get("context_rescue_unknown_min_score", 0.7),
        # Molecule rescue / second-pass molecule RF
        molecule_rf_rescue_elements=molecule_rescue_cfg.get("enabled", False),
        molecule_rf_rescue_threshold=molecule_rescue_cfg.get("molecule_rf_rescue_threshold", 0.8),
        molecule_rf_rescue_margin=molecule_rescue_cfg.get("molecule_rf_rescue_margin", 0.15),
        molecule_rf_rescue_score_margin=molecule_rescue_cfg.get("molecule_rf_rescue_score_margin", 0.05),
        molecule_rf_rescue_dist_margin=molecule_rescue_cfg.get("molecule_rf_rescue_dist_margin", 0.05),
        unknown_molecule_rf=unknown_molecule_rf.get("enabled", False),
        unknown_molecule_rf_threshold=unknown_molecule_rf.get("unknown_molecule_rf_threshold", 0.8),
    )


@register("rf")
class RFClassifierPipeline(ClassifierPipeline):
    def run(self, ctx: ClassifierContext) -> dict:
        step_timings: dict[str, float] = {}

        def _timed_load_ion_training_data(*args, **kw):
            t0 = time.perf_counter()
            try:
                return load_ion_training_data(*args, **kw)
            finally:
                step_timings['rf_train'] = step_timings.get('rf_train', 0.0) + (time.perf_counter() - t0)

        def _timed_create_RF_model(*args, **kw):
            t0 = time.perf_counter()
            try:
                return create_RF_model(*args, **kw)
            finally:
                step_timings['rf_train'] = step_timings.get('rf_train', 0.0) + (time.perf_counter() - t0)

        def _timed_run_RF_model(*args, **kw):
            t0 = time.perf_counter()
            try:
                return run_RF_model(*args, **kw)
            finally:
                step_timings['rf_infer'] = step_timings.get('rf_infer', 0.0) + (time.perf_counter() - t0)

        kwargs = dict(ctx.cfg)
        kwargs["molecule_rf_threshold"] = kwargs.pop("unknown_molecule_rf_threshold")

        # --- Step 1: resolve target species/elements ---
        truth_data = ctx.truth_data
        if ctx.species_list is not None:
            label_map = {simplify_label(str(s)): str(s) for s in ctx.species_list if s and str(s) != 'Unknown'}
        else:
            label_map = {simplify_label(str(t.label)): t.label for t in truth_data if t.label and t.label != 'Unknown'}
        truth_species_all = sorted(label_map.keys())
        include_molecules = kwargs['include_molecules']
        truth_species_primary = truth_species_all if include_molecules else [s for s in truth_species_all if not is_molecule(s)]
        truth_molecules = [s for s in truth_species_all if is_molecule(s)]

        if ctx.elements_list is not None:
            elements_for_molecules = sorted({str(e) for e in ctx.elements_list if e})
        elif ctx.species_list is not None:
            elements_for_molecules = sorted({sym for s in truth_species_all for sym in re.findall(r'[A-Z][a-z]?', s)})
        else:
            elements_for_molecules = ctx.elements_for_molecules

        prefix = ctx.prefix or os.path.basename(ctx.apt_file).split('.')[0].lower()
        artifacts_dir = ctx.artifacts_dir

        # --- Step 2: ranging ---
        t0 = time.perf_counter()
        peaks = run_yolo_ranging(
            ctx.spectrum_log, yolo_weights=kwargs['yolo_weights'], iou=kwargs['iou'],
            conf=kwargs['conf'], max_det=kwargs['max_det'],
        )
        step_timings['ranging'] = time.perf_counter() - t0
        if peaks is None:
            ctx.peaks = []
            return {}
        ctx.peaks = peaks

        default_training_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'training_data', 'NewData', 'Data0001',
        )
        training_path = kwargs['training_path'] or default_training_path
        use_neighborhood = kwargs['use_neighborhood']
        neighbor_threshold = kwargs['neighbor_threshold']
        eff_neighbor_threshold = neighbor_threshold if use_neighborhood else 0.0
        use_signature = kwargs['use_signature']
        training_num_files = kwargs['training_num_files']
        augment_charge_ratios = kwargs['augment_molecule_training_charge_ratios']

        before_rescue_breakdown = None
        after_rescue_breakdown = None
        rescue_stats = {'considered': 0, 'overrides': 0, 'mixed_candidates': 0}
        mc_samples_by_species: dict = {}

        try:
            peak_mcs = np.array([])
            raw_elements_initial, rf_confs_initial, detailed_rf_initial = [], [], []

            X_train, ions_train = _timed_load_ion_training_data(
                path=training_path, element_list=truth_species_primary,
                elements_to_get_molecules=elements_for_molecules if include_molecules else [],
                num_files=int(training_num_files), neighbor_threshold=eff_neighbor_threshold,
                use_signature=use_signature, augment_molecule_charge_ratios=bool(augment_charge_ratios),
            )
            if len(X_train) > 0:
                scaler_rf, model_rf, target_decoder_rf = _timed_create_RF_model(X_train, ions_train)
                raw_elements_initial, rf_confs_initial, detailed_rf_initial, peak_mcs = _timed_run_RF_model(
                    peaks, ctx.x_exp, ctx.spectrum_log, scaler_rf, model_rf, target_decoder_rf,
                    neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature,
                )

            for i, p in enumerate(peaks):
                p.label = raw_elements_initial[i]
                p.id_score = float(rf_confs_initial[i])
                p.is_unknown = False
                p.method = 'RF'
                p.detailed_id = detailed_rf_initial[i]

            # --- Step 4: mc-sample lookup ---
            mc_samples_by_species = build_empirical_mc_samples(path=training_path, num_files=int(training_num_files))

            mol_rf = None
            if truth_molecules and (kwargs['unknown_molecule_rf'] or kwargs['molecule_rf_rescue_elements']):
                mol_rf = molecule_rescue.train_molecule_only_rf(
                    training_data_path=training_path, truth_molecules=truth_molecules,
                    training_num_files=training_num_files, neighbor_threshold=eff_neighbor_threshold,
                    use_signature=use_signature, augment_molecule_training_charge_ratios=augment_charge_ratios,
                    load_ion_training_data_fn=_timed_load_ion_training_data,
                    create_rf_model_fn=_timed_create_RF_model,
                )

            # --- Step 5: guardrails + molecule rescue, same order as today ---
            guardrail.flag_unknown_peaks(
                peaks, mc_samples_by_species, peak_mcs,
                flag_unknowns=kwargs['flag_unknowns'], mc_threshold=kwargs['mc_threshold'],
                unknown_confidence_threshold=kwargs['unknown_confidence_threshold'],
            )
            if kwargs['unknown_molecule_rf'] and kwargs['flag_unknowns'] and truth_molecules:
                molecule_rescue.rescue_unknowns_with_molecule_rf(
                    peaks, ctx.x_exp, ctx.spectrum_log, mol_rf, mc_samples_by_species,
                    mc_threshold=kwargs['mc_threshold'], molecule_rf_threshold=kwargs['molecule_rf_threshold'],
                    eff_neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature,
                    run_rf_model_fn=_timed_run_RF_model,
                )
            before_rescue_breakdown = guardrail.compute_accuracy_breakdown(
                peaks, truth_data, rf_accuracy_top_n=kwargs['rf_accuracy_top_n'])

            if kwargs['context_rescore']:
                guardrail.context_rescore_peaks(
                    peaks, peak_mcs,
                    context_window_da=kwargs['context_window_da'], context_strength=kwargs['context_strength'],
                    context_min_confidence=kwargs['context_min_confidence'],
                    context_min_candidate_confidence=kwargs['context_min_candidate_confidence'],
                    context_override_margin=kwargs['context_override_margin'],
                    context_distance_sigma=kwargs['context_distance_sigma'],
                    context_rescue_unknown_same_label=kwargs['context_rescue_unknown_same_label'],
                    context_rescue_unknown_min_score=kwargs['context_rescue_unknown_min_score'],
                    artifacts_dir=artifacts_dir, prefix=prefix,
                )

            rescue_override_rows = []
            if kwargs['molecule_rf_rescue_elements'] and truth_molecules and mol_rf is not None:
                rescue_stats, rescue_override_rows = molecule_rescue.rescue_elements_with_molecule_rf(
                    peaks, ctx.x_exp, ctx.spectrum_log, mol_rf, mc_samples_by_species,
                    mc_threshold=kwargs['mc_threshold'],
                    molecule_rf_rescue_threshold=kwargs['molecule_rf_rescue_threshold'],
                    molecule_rf_rescue_margin=kwargs['molecule_rf_rescue_margin'],
                    molecule_rf_rescue_score_margin=kwargs['molecule_rf_rescue_score_margin'],
                    molecule_rf_rescue_dist_margin=kwargs['molecule_rf_rescue_dist_margin'],
                    eff_neighbor_threshold=eff_neighbor_threshold, use_signature=use_signature,
                    run_rf_model_fn=_timed_run_RF_model,
                )

            mixed_unknown_count = guardrail.flag_high_confidence_mixed_unknowns(
                peaks, flag_unknowns=kwargs['flag_unknowns'],
                unknown_mixed_element_molecule_confidence_threshold=kwargs['unknown_mixed_element_molecule_confidence_threshold'],
            )
            if mixed_unknown_count:
                print(
                    "  High-confidence mixed element/molecule peaks flagged as Unknown: "
                    f"{mixed_unknown_count} (threshold={float(kwargs['unknown_mixed_element_molecule_confidence_threshold']):.2f})"
                )
            after_rescue_breakdown = guardrail.compute_accuracy_breakdown(
                peaks, truth_data, rf_accuracy_top_n=kwargs['rf_accuracy_top_n'])

            # --- Step 7: map labels back through label_map ---
            for p in peaks:
                orig = label_map.get(p.label)
                if orig:
                    p.label = orig
                if p.detailed_id:
                    p.detailed_id.el1 = label_map.get(p.detailed_id.el1, p.detailed_id.el1)
                    p.detailed_id.el2 = label_map.get(p.detailed_id.el2, p.detailed_id.el2)

            molecule_rescue.write_molecule_rescue_candidates_csv(rescue_override_rows, label_map, artifacts_dir, prefix)

        except Exception as e:
            print(f"RF identification failed: {e}")
            import traceback
            traceback.print_exc()
            before_rescue_breakdown = None
            after_rescue_breakdown = None
            rescue_stats = {'considered': 0, 'overrides': 0, 'mixed_candidates': 0}

        # --- Step 6: accuracy breakdown + CSV writers ---
        detailed_rows = guardrail.write_detailed_results_csv(
            peaks, truth_data, save_artifacts=ctx.save_artifacts, artifacts_dir=artifacts_dir, prefix=prefix)
        guardrail.write_unknown_peak_error_report(
            detailed_rows, mc_samples_by_species, ctx.x_exp, ctx.spectrum_log,
            save_artifacts=ctx.save_artifacts, flag_unknowns=kwargs['flag_unknowns'],
            mc_threshold=kwargs['mc_threshold'], artifacts_dir=artifacts_dir, prefix=prefix,
        )

        if after_rescue_breakdown is None:
            after_rescue_breakdown = guardrail.empty_accuracy_breakdown()

        accuracy_breakdown = dict(after_rescue_breakdown)
        if kwargs['molecule_rf_rescue_elements'] and before_rescue_breakdown is not None:
            accuracy_breakdown['before_rescue'] = before_rescue_breakdown
            accuracy_breakdown['after_rescue'] = after_rescue_breakdown
            accuracy_breakdown['rescue'] = rescue_stats

        # --- Step 8: args snapshot ---
        if ctx.save_artifacts:
            self._save_args_snapshot(ctx, kwargs, prefix, artifacts_dir)

        # --- Step 9: step timing ---
        print(
            "  Step timing: "
            f"ranging {step_timings.get('ranging', 0.0):.2f}s | "
            f"RF training (incl. data load) {step_timings.get('rf_train', 0.0):.2f}s | "
            f"RF inference {step_timings.get('rf_infer', 0.0):.2f}s"
        )

        return accuracy_breakdown

    @staticmethod
    def _save_args_snapshot(ctx: ClassifierContext, kwargs: dict, prefix: str, artifacts_dir: str | None) -> None:
        snapshot = {}
        extra_scalars = {
            'apt_file': ctx.apt_file, 'rrng_file': ctx.rrng_file, 'prefix': prefix,
            'artifacts_dir': artifacts_dir, 'save_artifacts': ctx.save_artifacts,
        }
        for k, v in {**kwargs, **extra_scalars}.items():
            try:
                snapshot[k] = yaml_safe(v)
            except TypeError:
                continue
        out_dir = artifacts_dir or prefix
        try:
            os.makedirs(out_dir, exist_ok=True)
            args_path = os.path.join(out_dir, f"{prefix}_yolo_args.yaml")
            with open(args_path, 'w') as f:
                yaml.safe_dump(snapshot, f, sort_keys=True, default_flow_style=False)
            print(f"  Saved YOLO call arguments: {args_path}")
        except OSError as e:
            print(f"  [Warning] Could not save YOLO call arguments: {e}")
