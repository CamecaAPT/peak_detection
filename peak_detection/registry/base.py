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

    ``peaks`` is populated by ``ClassifierPipeline.run`` (today: run_yolo_ranging's
    internal ranging call); each PeakRange gets label/detailed_id/id_score/is_unknown/method set.

    ``cfg`` is a FLAT, model-specific kwargs dict — each entry point script builds it via
    that model's own flattener function (e.g. RF's ``flat_rf_kwargs()``), not a generic
    nested-to-flat conversion.

    ``diagnostics`` is an optional out-param, same pattern as ``peaks``: a pipeline that has
    its own internal staging worth recording (e.g. RF's before/after molecule-rescue
    snapshot) sets it during ``run()``; the entry-point script reads it back afterward to
    populate ``DatasetStats.extras``. None for pipelines with no equivalent internal stage.
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
    diagnostics: object | None = None


class ClassifierPipeline(ABC):
    """Base class for a swappable species-identification model (selected via --model)."""

    name: str

    @abstractmethod
    def run(self, ctx: ClassifierContext) -> dict:
        """Run ranging + identification, populate ``ctx.peaks``, and return an accuracy
        breakdown dict (may be empty if truth data / accuracy scoring isn't applicable)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def flat_kwargs(cfg: dict) -> dict:
        """Flatten this model's merged YAML config (nested: ranging/training/guardrails.*)
        into the flat kwarg names its entry point call and ``ClassifierContext.cfg`` use."""
        raise NotImplementedError
