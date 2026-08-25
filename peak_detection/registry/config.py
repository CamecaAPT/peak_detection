"""Loads configs/models/<model>.yaml (+ optional override)."""
from __future__ import annotations

import copy
import os
import sys
from datetime import datetime

import yaml

from ..utils import yaml_safe


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(data).__name__}.")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a copy of ``base`` (override wins on conflicts)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_merged_config(model_name: str, *, configs_dir: str, override_path: str | None = None) -> dict:
    """configs/models/<model_name>.yaml <- override_path (dicts deep-merged)."""
    model_path = os.path.join(configs_dir, "models", f"{model_name}.yaml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No config found for model '{model_name}' at {model_path}")

    cfg = _load_yaml(model_path)
    if override_path:
        cfg = _deep_merge(cfg, _load_yaml(override_path))
    return cfg


def _sanitize(obj):
    """Recursively coerce a (possibly nested) config value to plain YAML-safe scalars."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return yaml_safe(obj)


def write_effective_config(cfg: dict, path: str | None = None,
                           extra: dict | None = None, directory: str | None = None) -> str:
    """Write the effective (merged) config to YAML for provenance, plus a command/timestamp
    header. ``extra`` holds script-specific output-control flags to persist alongside it.
    ``directory`` places the auto-named ``effective_config_<timestamp>.yaml`` there (created if
    needed); ``path`` (an explicit file path) takes precedence over ``directory``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path is None:
        name = f"effective_config_{timestamp}.yaml"
        if directory:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, name)
        else:
            path = name
    data = _sanitize(cfg)
    if extra:
        data["output_control"] = _sanitize(extra)
    data["command"] = " ".join(sys.argv)
    data["timestamp"] = timestamp
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)
    print(f"Saved effective config to {path}")
    return path
