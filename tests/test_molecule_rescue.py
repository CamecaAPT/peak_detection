import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from peak_detection.models import DetailedId, PeakRange
from peak_detection.IonIdentificationModels.RF import molecule_rescue


def _peak(label, conf, el1=None, conf1=None, el2='', conf2=0.0, pos=10.0, is_unknown=False):
    el1 = el1 if el1 is not None else label
    conf1 = conf1 if conf1 is not None else conf
    return PeakRange(
        start=pos - 0.05, end=pos + 0.05, pos=pos,
        label=label, id_score=conf, method='RF', is_unknown=is_unknown,
        detailed_id=DetailedId(el1=el1, conf1=conf1, el2=el2, conf2=conf2),
    )


def test_train_molecule_only_rf_returns_none_without_data():
    result = molecule_rescue.train_molecule_only_rf(
        training_data_path='unused', truth_molecules=['FeO'], training_num_files=1,
        neighbor_threshold=0.0, use_signature=False, augment_molecule_training_charge_ratios=False,
        load_ion_training_data_fn=lambda **kw: ([], []),
        create_rf_model_fn=lambda *a: (None, None, None),
    )
    assert result is None


def test_train_molecule_only_rf_returns_model_tuple():
    result = molecule_rescue.train_molecule_only_rf(
        training_data_path='unused', truth_molecules=['FeO'], training_num_files=1,
        neighbor_threshold=0.0, use_signature=False, augment_molecule_training_charge_ratios=False,
        load_ion_training_data_fn=lambda **kw: ([1, 2], ['FeO', 'FeO']),
        create_rf_model_fn=lambda X, y: ('scaler', 'model', 'decoder'),
    )
    assert result == ('scaler', 'model', 'decoder')


def test_rescue_unknowns_recovers_high_confidence_physical_match():
    peaks = [_peak('Unknown (Fe)', 1.0, el1='Unknown (Fe)', conf1=1.0, pos=50.0, is_unknown=True)]

    def fake_run_rf_model(peaks_arg, x_exp, spectrum_log, scaler, model, decoder, **kw):
        det = DetailedId(el1='FeO', conf1=0.9, el2='', conf2=0.0)
        return ['FeO'], [0.9], [det], np.array([50.0])

    molecule_rescue.rescue_unknowns_with_molecule_rf(
        peaks, x_exp=None, spectrum_log=None,
        mol_rf=('scaler', 'model', 'decoder'),
        mc_samples_by_species={'FeO': np.array([50.0, 50.01])},
        mc_threshold=0.2, molecule_rf_threshold=0.8,
        eff_neighbor_threshold=0.0, use_signature=False,
        run_rf_model_fn=fake_run_rf_model,
    )
    assert peaks[0].is_unknown is False
    assert peaks[0].label == 'FeO'
    assert peaks[0].method == 'RF-mol2'


def test_rescue_unknowns_noop_without_mol_rf():
    peaks = [_peak('Unknown (Fe)', 1.0, is_unknown=True)]
    molecule_rescue.rescue_unknowns_with_molecule_rf(
        peaks, x_exp=None, spectrum_log=None, mol_rf=None, mc_samples_by_species={},
        mc_threshold=0.2, molecule_rf_threshold=0.8, eff_neighbor_threshold=0.0,
        use_signature=False, run_rf_model_fn=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert peaks[0].is_unknown is True


def test_rescue_elements_overrides_on_clear_molecule_win():
    peaks = [_peak('Fe', 0.6, el1='Fe', conf1=0.6, pos=50.0)]

    def fake_run_rf_model(peaks_arg, x_exp, spectrum_log, scaler, model, decoder, **kw):
        det = DetailedId(el1='FeO', conf1=0.95, el2='', conf2=0.0)
        return ['FeO'], [0.95], [det], np.array([50.0])

    stats, rows = molecule_rescue.rescue_elements_with_molecule_rf(
        peaks, x_exp=None, spectrum_log=None,
        mol_rf=('scaler', 'model', 'decoder'),
        mc_samples_by_species={'FeO': np.array([50.0, 50.01]), 'Fe': np.array([49.0])},
        mc_threshold=0.2, molecule_rf_rescue_threshold=0.8,
        molecule_rf_rescue_margin=0.15, molecule_rf_rescue_score_margin=0.05,
        molecule_rf_rescue_dist_margin=0.05, eff_neighbor_threshold=0.0, use_signature=False,
        run_rf_model_fn=fake_run_rf_model,
    )
    assert stats['overrides'] == 1
    assert peaks[0].label == 'FeO'
    assert peaks[0].method == 'RF-mol-rescue'
    assert len(rows) == 1


def test_write_molecule_rescue_candidates_csv_writes_mapped_labels(tmp_path):
    rows = [{
        'peak_start': 49.95, 'peak_end': 50.05, 'peak_mc': 50.0,
        'element_pred_simple': 'Fe', 'element_conf': 0.6, 'element_dist': 0.5,
        'element_best_scale': 1.0, 'element_scaled_mc': 50.0, 'element_nearest_training_mc': '',
        'molecule_pred_simple': 'FeO', 'molecule_conf': 0.95, 'molecule_dist': 0.01,
        'molecule_best_scale': 1.0, 'molecule_scaled_mc': 50.0, 'molecule_nearest_training_mc': 50.0,
        'q_element': 0.0, 'q_molecule': 0.95, 'score_element': 0.0, 'score_molecule': 0.9,
        'rescue_action': 'override', 'rescue_reason': 'conf_margin',
    }]
    molecule_rescue.write_molecule_rescue_candidates_csv(
        rows, label_map={'Fe': 'Fe', 'FeO': 'Fe:1 O:1'}, artifacts_dir=str(tmp_path), prefix='sample',
    )
    out_file = tmp_path / "sample_molecule_rescue_candidates.csv"
    assert out_file.exists()
    assert 'Fe:1 O:1' in out_file.read_text()
