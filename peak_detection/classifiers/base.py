"""Shared interfaces for pluggable species-classification pipelines.

Peak *ranging* (finding peak start/end positions) is fixed across all models and is not part
of this abstraction. A ClassifierPipeline only assigns labels/confidences to already-detected
PeakRange objects and applies its own guardrails (unknown-flagging, context-rescoring, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..models import PeakRange


@dataclass
class ClassifierContext:
    """Everything a ClassifierPipeline needs to identify species for one dataset.

    ``peaks`` is populated by ``ClassifierPipeline.run`` (today: predict_peak_ranges_yolo's
    internal ranging call); each PeakRange gets label/detailed_id/id_score/is_unknown/method set.
    """
    apt_file: str
    rrng_file: str | None
    x_exp: np.ndarray
    spectrum_log: object
    truth_data: list[PeakRange]
    elements_for_molecules: list[str]
    prefix: str
    artifacts_dir: str | None
    save_artifacts: bool
    cfg: dict
    species_list: list[str] | None = None
    elements_list: list[str] | None = None
    peaks: list[PeakRange] = field(default_factory=list)


class ClassifierPipeline(ABC):
    """Base class for a swappable species-identification model (selected via --model)."""

    name: str

    @abstractmethod
    def run(self, ctx: ClassifierContext) -> dict:
        """Run ranging + identification, populate ``ctx.peaks``, and return an accuracy
        breakdown dict (may be empty if truth data / accuracy scoring isn't applicable)."""
        raise NotImplementedError
