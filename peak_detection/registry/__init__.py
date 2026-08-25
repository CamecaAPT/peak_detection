"""Registry of pluggable species-classification pipelines, selected via --model.

Each model's registration lives inside its own IonIdentificationModels/<Model>/ folder
(alongside its underlying model code), so evaluating and later deleting a model is a single
self-contained folder + its configs/models/<name>.yaml — no stray files left behind here.

Adding a new model:
  1. Create peak_detection/IonIdentificationModels/<Model>/<name>_pipeline.py with a
     ClassifierPipeline subclass decorated with @register("<name>") that implements both
     `run()` and the `flat_kwargs()` staticmethod (mapping that model's merged YAML onto the
     flat kwargs its pipeline expects).
  2. Import that module below so the decorator runs.
  3. Add configs/models/<name>.yaml with that model's tunables.
  4. That's it — the --model flag is plug-and-play: both entry point scripts fetch the
     model's flattener via get_flattener(name) instead of hardcoding any one model's.
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


def get_flattener(name: str):
    """Return the registered model's `flat_kwargs` staticmethod: maps its merged YAML
    config dict onto the flat kwargs its entry point call expects."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name].flat_kwargs


def list_models() -> list[str]:
    return sorted(_REGISTRY)


# Import built-in pipelines so their @register decorators run.
from ..IonIdentificationModels.RF import rf_pipeline  # noqa: E402,F401

__all__ = [
    "ClassifierPipeline", "ClassifierContext", "register", "get_pipeline",
    "get_flattener", "list_models",
]
