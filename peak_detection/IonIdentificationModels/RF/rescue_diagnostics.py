"""RF-only diagnostics: the accuracy snapshot taken right before molecule-rescue runs,
plus rescue's own considered/overrides/mixed-candidate counts. Lives here, not in the
shared peak_detection/models.py, because no other registered model has an equivalent
internal stage to report — deleting this folder removes it along with the rest of RF.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RFRescueDiagnostics:
    """Snapshot of RF's accuracy right after unknown-flagging/context-rescore but before
    molecule-rescue runs, plus rescue's own considered/overrides/mixed-candidate counts.
    Implements the ApproachDiagnostics protocol (peak_detection.models) via to_row()."""
    species_total_before: int = 0
    species_correct_before: int = 0
    elemental_total_before: int = 0
    elemental_correct_before: int = 0
    molecular_total_before: int = 0
    molecular_correct_before: int = 0
    species_total_before_exc: int = 0
    species_correct_before_exc: int = 0
    elemental_total_before_exc: int = 0
    elemental_correct_before_exc: int = 0
    molecular_total_before_exc: int = 0
    molecular_correct_before_exc: int = 0
    molecule_rescue_considered: int = 0
    molecule_rescue_overrides: int = 0
    molecule_rescue_mixed_candidates: int = 0

    @classmethod
    def from_breakdown(cls, before_breakdown: dict, rescue_stats: dict) -> "RFRescueDiagnostics":
        """Build from guardrail.compute_accuracy_breakdown()'s pre-rescue return dict (its
        'counts' sub-dict) and molecule_rescue.rescue_elements_with_molecule_rf()'s
        rescue_stats dict ({'considered', 'overrides', 'mixed_candidates})."""
        counts = before_breakdown.get('counts', {}) if before_breakdown else {}

        def g(key: str) -> int:
            return int(counts.get(key, 0) or 0)

        kwargs = {}
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

            kwargs[f'species_total_before{suffix}'] = species_total
            kwargs[f'species_correct_before{suffix}'] = species_correct
            kwargs[f'elemental_total_before{suffix}'] = elemental_total
            kwargs[f'elemental_correct_before{suffix}'] = elemental_correct
            kwargs[f'molecular_total_before{suffix}'] = molecular_total
            kwargs[f'molecular_correct_before{suffix}'] = molecular_correct

        kwargs['molecule_rescue_considered'] = int((rescue_stats or {}).get('considered', 0) or 0)
        kwargs['molecule_rescue_overrides'] = int((rescue_stats or {}).get('overrides', 0) or 0)
        kwargs['molecule_rescue_mixed_candidates'] = int((rescue_stats or {}).get('mixed_candidates', 0) or 0)
        return cls(**kwargs)

    def to_row(self) -> dict:
        """Flatten to a flat dict of CSV-writable columns (the ApproachDiagnostics contract)."""
        return {
            'species_total_before': self.species_total_before,
            'species_correct_before': self.species_correct_before,
            'elemental_total_before': self.elemental_total_before,
            'elemental_correct_before': self.elemental_correct_before,
            'molecular_total_before': self.molecular_total_before,
            'molecular_correct_before': self.molecular_correct_before,
            'species_total_before_exc': self.species_total_before_exc,
            'species_correct_before_exc': self.species_correct_before_exc,
            'elemental_total_before_exc': self.elemental_total_before_exc,
            'elemental_correct_before_exc': self.elemental_correct_before_exc,
            'molecular_total_before_exc': self.molecular_total_before_exc,
            'molecular_correct_before_exc': self.molecular_correct_before_exc,
            'molecule_rescue_considered': self.molecule_rescue_considered,
            'molecule_rescue_overrides': self.molecule_rescue_overrides,
            'molecule_rescue_mixed_candidates': self.molecule_rescue_mixed_candidates,
        }
