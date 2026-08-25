import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from peak_detection.models import DetailedId, PeakRange
from peak_detection import guardrail


def _peak(label, conf, el1=None, conf1=None, el2='', conf2=0.0, pos=10.0):
    el1 = el1 if el1 is not None else label
    conf1 = conf1 if conf1 is not None else conf
    return PeakRange(
        start=pos - 0.05, end=pos + 0.05, pos=pos,
        label=label, id_score=conf, method='RF', is_unknown=False,
        detailed_id=DetailedId(el1=el1, conf1=conf1, el2=el2, conf2=conf2),
    )


def test_flag_unknown_peaks_flags_unphysical_mc_distance():
    peaks = [_peak('Fe', 0.9, pos=10.0)]
    mc_samples_by_species = {'Fe': np.array([50.0, 51.0])}
    guardrail.flag_unknown_peaks(
        peaks, mc_samples_by_species, np.array([10.0]),
        flag_unknowns=True, mc_threshold=0.2, unknown_confidence_threshold=0.0,
    )
    assert peaks[0].is_unknown is True
    assert peaks[0].label.startswith('Unknown')


def test_flag_unknown_peaks_leaves_physical_match_alone():
    peaks = [_peak('Fe', 0.9, pos=50.02)]
    mc_samples_by_species = {'Fe': np.array([50.0, 51.0])}
    guardrail.flag_unknown_peaks(
        peaks, mc_samples_by_species, np.array([50.02]),
        flag_unknowns=True, mc_threshold=0.2, unknown_confidence_threshold=0.0,
    )
    assert peaks[0].is_unknown is False
    assert peaks[0].label == 'Fe'


def test_flag_unknown_peaks_noop_when_disabled():
    peaks = [_peak('Fe', 0.9, pos=10.0)]
    mc_samples_by_species = {'Fe': np.array([50.0, 51.0])}
    guardrail.flag_unknown_peaks(
        peaks, mc_samples_by_species, np.array([10.0]),
        flag_unknowns=False, mc_threshold=0.2, unknown_confidence_threshold=0.0,
    )
    assert peaks[0].is_unknown is False
    assert peaks[0].label == 'Fe'


def test_flag_high_confidence_mixed_unknowns():
    peaks = [_peak('Fe', 0.97, el1='Fe', conf1=0.97, el2='FeO', conf2=0.96)]
    flagged = guardrail.flag_high_confidence_mixed_unknowns(
        peaks, flag_unknowns=True, unknown_mixed_element_molecule_confidence_threshold=0.95,
    )
    assert flagged == 1
    assert peaks[0].is_unknown is True


def test_compute_accuracy_breakdown_counts_correct_and_total():
    truth = [PeakRange(start=9.95, end=10.05, pos=10.0, label='Fe')]
    peaks = [_peak('Fe', 0.9, pos=10.0)]
    breakdown = guardrail.compute_accuracy_breakdown(peaks, truth, rf_accuracy_top_n=1)
    assert breakdown['species_excluding_unknowns'] == 100.0
    assert breakdown['counts']['species_total_including_unknowns'] == 1
    assert breakdown['counts']['species_correct_including_unknowns'] == 1


def test_empty_accuracy_breakdown_is_zeroed():
    breakdown = guardrail.empty_accuracy_breakdown()
    assert breakdown['species_excluding_unknowns'] == 0.0
    assert breakdown['counts']['species_total_including_unknowns'] == 0


def test_write_detailed_results_csv_writes_when_save_artifacts(tmp_path):
    truth = [PeakRange(start=9.95, end=10.05, pos=10.0, label='Fe')]
    peaks = [_peak('Fe', 0.9, pos=10.0)]
    rows = guardrail.write_detailed_results_csv(
        peaks, truth, save_artifacts=True, artifacts_dir=str(tmp_path), prefix='sample',
    )
    assert len(rows) == 1
    assert (tmp_path / "sample_detailed_results.csv").exists()


def test_write_detailed_results_csv_skips_file_when_not_saving(tmp_path):
    truth = []
    peaks = [_peak('Fe', 0.9, pos=10.0)]
    rows = guardrail.write_detailed_results_csv(
        peaks, truth, save_artifacts=False, artifacts_dir=str(tmp_path), prefix='sample',
    )
    assert len(rows) == 1
    assert not (tmp_path / "sample_detailed_results.csv").exists()


def test_context_rescore_peaks_candidate_switch_override(tmp_path):
    # Target: low-confidence own top candidate 'Fe' (0.5), second candidate 'Cr' (0.4).
    # A single nearby neighbor strongly supports 'Cr', enough to flip the label.
    target = _peak('Fe', 0.5, el2='Cr', conf2=0.4, pos=10.0)
    neighbor = _peak('Cr', 0.9, pos=10.5)
    peaks = [target, neighbor]

    rows = guardrail.context_rescore_peaks(
        peaks, np.array([]),
        context_window_da=2.0, context_strength=0.35,
        context_min_confidence=0.75, context_min_candidate_confidence=0.05,
        context_override_margin=0.05, context_distance_sigma=0.75,
        context_rescue_unknown_same_label=True, context_rescue_unknown_min_score=0.7,
        artifacts_dir=str(tmp_path), prefix='sample',
    )

    assert peaks[0].label == 'Cr'
    assert peaks[0].is_unknown is False
    assert peaks[0].method == 'RF+context'
    assert len(rows) == 1
    assert rows[0]['override_reason'] == 'candidate_switch'


def test_context_rescore_peaks_same_label_unknown_rescue(tmp_path):
    # Target flagged Unknown; its own top candidate ('Fe') matches what a nearby neighbor
    # strongly supports, enough to rescue it back to known under the same label.
    target = _peak('Unknown', 0.3, el1='Fe', conf1=0.3, el2='Cr', conf2=0.2, pos=10.0)
    target.is_unknown = True
    neighbor = _peak('Fe', 0.9, pos=10.3)
    peaks = [target, neighbor]

    rows = guardrail.context_rescore_peaks(
        peaks, np.array([]),
        context_window_da=2.0, context_strength=0.35,
        context_min_confidence=0.75, context_min_candidate_confidence=0.05,
        context_override_margin=0.05, context_distance_sigma=0.75,
        context_rescue_unknown_same_label=True, context_rescue_unknown_min_score=0.5,
        artifacts_dir=str(tmp_path), prefix='sample',
    )

    assert peaks[0].label == 'Fe'
    assert peaks[0].is_unknown is False
    assert peaks[0].method == 'RF+context'
    assert len(rows) == 1
    assert rows[0]['override_reason'] == 'same_label_unknown_rescue'
