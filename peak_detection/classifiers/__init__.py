"""Registry of pluggable species-classification pipelines, selected via --model.

Each model's registration lives inside its own IonIdentificationModels/<Model>/ folder
(alongside its underlying model code), so evaluating and later deleting a model is a single
self-contained folder + its configs/models/<name>.yaml — no stray files left behind here.

Adding a new model:
  1. Create peak_detection/IonIdentificationModels/<Model>/<name>_pipeline.py with a
     ClassifierPipeline subclass decorated with @register("<name>").
  2. Import that module below so the decorator runs.
  3. Add configs/models/<name>.yaml with that model's tunables.
No changes to orchestrator.py are needed.
"""
from __future__ import annotations

from .base import ClassifierContext, ClassifierPipeline

_REGISTRY: dict[str, type[ClassifierPipeline]] = {}


def register(name: str):
    """Class decorator registering a ClassifierPipeline under ``name``."""
    def _decorator(cls: type[ClassifierPipeline]) -> type[ClassifierPipeline]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _decorator


def get_pipeline(name: str) -> ClassifierPipeline:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_models() -> list[str]:
    return sorted(_REGISTRY)


# Import built-in pipelines so their @register decorators run.
from ..IonIdentificationModels.RF import rf_pipeline  # noqa: E402,F401

__all__ = ["ClassifierPipeline", "ClassifierContext", "register", "get_pipeline", "list_models"]
