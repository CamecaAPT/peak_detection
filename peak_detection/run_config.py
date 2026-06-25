"""
run_config.py — single source of truth for the shared peak-detection run parameters.

`detect_peaks_refactor.py` (eval/benchmark) and `detect_peaks_headless.py` (production
inference) both expose the same ~39 YOLO / RF / unknown-flagging / context-rescoring
tunables. This module defines them once (``SHARED_PARAMS``) and provides everything both
scripts need:

  * ``add_shared_args(parser)``   — register all shared args on an argparse parser
                                     (underscore canonical flag + hyphen alias).
  * ``parse_with_config(...)``    — two-phase parse so ``--config FILE`` (a YAML run config)
                                     supplies defaults that explicit CLI flags still override.
  * ``RunConfig``                 — typed container built from ``SHARED_PARAMS``.
  * ``config_from_namespace(ns)`` — extract a ``RunConfig`` from parsed args.
  * ``RunConfig.to_kwargs()``     — dict to splat into ``process_dataset`` /
                                     ``detect_peaks_headless`` (param names match).
  * ``RunConfig.to_yolo_kwargs()``— same, but applies the
                                     ``unknown_molecule_rf_threshold`` -> ``molecule_rf_threshold``
                                     rename required to call ``predict_peak_ranges_yolo`` directly.
  * ``write_run_config(cfg)``     — serialize the effective config to YAML (scalars only,
                                     ``sort_keys=True`` for stable diffs) plus a
                                     command/timestamp header. Supersedes the old
                                     ``write_run_cli_args`` .txt dump.

Script-specific arguments (I/O paths, plotting, ``--elements``/``--expected-rrng``,
``--separate-molecule-rf``, etc.) are deliberately NOT part of the shared set; each script
keeps registering those itself.
"""

from __future__ import annotations

import argparse
import numbers
import sys
from dataclasses import dataclass, field, fields, make_dataclass
from datetime import datetime

import yaml


@dataclass(frozen=True)
class Param:
    """One shared parameter: its canonical (underscore) name, argparse type, default and help.

    ``type`` is the callable argparse should use to coerce the value (``int``/``float``/``str``);
    for booleans set ``is_bool=True`` and leave ``type=None`` (they use BooleanOptionalAction).
    """
    name: str
    type: type | None
    default: object
    help: str = ""
    is_bool: bool = False


# --------------------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH
# Converged defaults (confirmed): newest weights + training data for BOTH scripts.
# --------------------------------------------------------------------------------------
SHARED_PARAMS: list[Param] = [
    # --- YOLO parameters ---
    Param("yolo_weights", str, "best_v0_2026-06-23.pt"),
    Param("n_iter", int, 0),
    Param("iou", float, 0.01),
    Param("conf", float, 0.05),
    Param("max_det", int, 2000),
    Param("iter_min_intensity_quantile", float, 0.10,
          "For YOLO iterative reruns, use this first-pass peak-intensity quantile to set the minimum intensity gate"),
    Param("iter_min_intensity_fraction", float, 0.50,
          "For YOLO iterative reruns, require new ranges to be at least this fraction of the first-pass intensity quantile"),
    Param("iter_intensity_stat_quantile", float, 0.90,
          "Within each candidate range, use this intensity quantile as the robust peak intensity statistic"),
    Param("mc_min", float, 0.0),
    Param("mc_max", float, 307.2),

    # --- RF parameters ---
    Param("training_path", str,
          "peak_detection/Ionclassifier/training_data/NewData_truthcoverage_lightmol1p_C3_BO_C2O_2p_2026-06-10/Data0001"),
    Param("training_num_files", int, 10000,
          "Number of synthetic training CSV files to scan (default loads all 10k when present)"),
    Param("augment_molecule_training_charge_ratios", None, False,
          "Augment molecule training m/c values with charge-state ratios (adds 1/2 and 1/3 m/c samples for molecular species)",
          is_bool=True),
    Param("molecule_rf_rescue_elements", None, False,
          "Run a molecule-only RF pass on peaks currently labeled as single elements and allow molecule overrides or mixed element+molecule top-2 candidates",
          is_bool=True),
    Param("molecule_rf_rescue_threshold", float, 0.8,
          "Min molecule RF confidence to accept a molecule rescue candidate"),
    Param("molecule_rf_rescue_margin", float, 0.15,
          "Confidence margin for molecule rescue: above element by this amount overrides; within this amount may be stored as mixed top-2"),
    Param("molecule_rf_rescue_score_margin", float, 0.05,
          "Quality-weighted score margin for molecule rescue overrides or mixed top-2 candidates"),
    Param("molecule_rf_rescue_dist_margin", float, 0.05,
          "m/c distance tolerance for molecule rescue; strict improvements can override, close overlaps can become mixed top-2"),
    Param("include_molecules", None, False, is_bool=True),
    Param("use_neighborhood", None, False, is_bool=True),
    Param("neighbor_threshold", float, 2.0),
    Param("use_signature", None, False, is_bool=True),
    Param("unknown_molecule_rf", None, False,
          "Train a molecule-only RF and apply it only to peaks flagged as unknown", is_bool=True),
    Param("unknown_molecule_rf_threshold", float, 0.8,
          "Min confidence for molecule-only RF to un-flag an unknown peak"),
    Param("followon_mc_vector_rf", None, False,
          "Run a follow-on RF using a padded vector of unique m/c values per predicted species-group", is_bool=True),
    Param("followon_mc_vector_round_decimals", int, 3,
          "Rounding decimals used when determining unique m/c values for the follow-on mc-vector RF"),

    # --- Unknown flagging ---
    Param("flag_unknowns", None, True, is_bool=True),
    Param("mc_threshold", float, 0.2),
    Param("unknown_confidence_threshold", float, 0.6,
          "Flag RF IDs as Unknown when the top candidate confidence is below this cutoff; set <=0 to disable"),
    Param("rf_accuracy_top_n", int, 1,
          "Consider the top N stored RF candidates when scoring element/molecule classification accuracy"),

    # --- Context rescoring ---
    Param("context_rescore", None, False,
          "Use nearby peak labels to rescore ambiguous RF candidates after initial classification", is_bool=True),
    Param("context_window_da", float, 2.0,
          "m/c window around a peak used to collect neighboring RF label support for context rescoring"),
    Param("context_strength", float, 0.35,
          "Weight applied to neighboring-label support during context rescoring"),
    Param("context_min_confidence", float, 0.75,
          "Only rescore peaks that are Unknown or whose top RF confidence is below this value"),
    Param("context_min_candidate_confidence", float, 0.05,
          "Minimum RF candidate confidence for a label to be eligible during context rescoring"),
    Param("context_override_margin", float, 0.05,
          "Require the context-adjusted winning score to beat the original top candidate by this margin"),
    Param("context_distance_sigma", float, 0.75,
          "Gaussian distance scale, in Da, for weighting nearby peaks during context rescoring"),
    Param("context_rescue_unknown_same_label", None, True,
          "When context rescoring is enabled, unflag Unknown peaks if nearby context strongly supports their existing top RF candidate",
          is_bool=True),
    Param("context_rescue_unknown_min_score", float, 0.7,
          "Minimum context-adjusted score needed to unflag an Unknown peak whose top RF candidate remains the winner"),
]

# Guard against accidental duplicate entries in the SPEC above.
_NAMES = [p.name for p in SHARED_PARAMS]
assert len(_NAMES) == len(set(_NAMES)), \
    f"Duplicate names in SHARED_PARAMS: {sorted({n for n in _NAMES if _NAMES.count(n) > 1})}"

_BY_NAME = {p.name: p for p in SHARED_PARAMS}

# Header keys written into a run-config YAML for provenance; not part of the param set, so
# they are ignored on load rather than treated as unknown.
HEADER_KEYS = ("command", "timestamp")

# Downstream name bridge: predict_peak_ranges_yolo names this param differently.
_YOLO_RENAMES = {"unknown_molecule_rf_threshold": "molecule_rf_threshold"}


# --------------------------------------------------------------------------------------
# RunConfig — typed container generated from SHARED_PARAMS (no name/default drift possible)
# --------------------------------------------------------------------------------------
def _to_kwargs(self) -> dict:
    """Param dict for process_dataset / detect_peaks_headless (names match those functions)."""
    return {p.name: getattr(self, p.name) for p in SHARED_PARAMS}


def _to_yolo_kwargs(self) -> dict:
    """Param dict for calling predict_peak_ranges_yolo directly (applies the name bridge)."""
    d = _to_kwargs(self)
    for src, dst in _YOLO_RENAMES.items():
        d[dst] = d.pop(src)
    return d


RunConfig = make_dataclass(
    "RunConfig",
    [(p.name, (p.type or bool), field(default=p.default)) for p in SHARED_PARAMS],
    namespace={"to_kwargs": _to_kwargs, "to_yolo_kwargs": _to_yolo_kwargs},
)
RunConfig.__doc__ = "Typed container for the shared run parameters (see SHARED_PARAMS)."


# --------------------------------------------------------------------------------------
# argparse integration
# --------------------------------------------------------------------------------------
def _flags_for(name: str) -> list[str]:
    """Underscore canonical flag plus a hyphenated alias (when the name contains '_')."""
    primary = f"--{name}"
    hyphen = f"--{name.replace('_', '-')}"
    return [primary] if hyphen == primary else [primary, hyphen]


def add_shared_args(parser: argparse.ArgumentParser, *, overrides: dict | None = None) -> None:
    """Register every shared parameter on ``parser``.

    Each param gets an underscore-style flag (canonical) and a hyphen-style alias, both
    mapping to the same underscore ``dest``. ``overrides`` may change a default per-script
    (not currently needed since SHARED_PARAMS already holds the converged defaults).
    """
    overrides = overrides or {}
    for p in SHARED_PARAMS:
        default = overrides.get(p.name, p.default)
        flags = _flags_for(p.name)
        if p.is_bool:
            parser.add_argument(*flags, dest=p.name,
                                action=argparse.BooleanOptionalAction,
                                default=default, help=p.help or None)
        else:
            parser.add_argument(*flags, dest=p.name, type=p.type,
                                default=default, help=p.help or None)


def config_from_namespace(args: argparse.Namespace) -> "RunConfig":
    """Build a RunConfig from a parsed argparse namespace."""
    return RunConfig(**{p.name: getattr(args, p.name) for p in SHARED_PARAMS})


def apply_config_defaults(parser: argparse.ArgumentParser, argv=None) -> list[str]:
    """If ``--config FILE`` appears in ``argv``, load that YAML and push its values onto
    ``parser`` as defaults (so explicit CLI flags still override them). Mutates ``parser``
    in place and returns the ``argv`` list.

    Only *defaults* are set, so YAML cannot satisfy ``required=`` arguments — required I/O
    stays CLI-only. Unknown keys (and the command/timestamp header) are ignored with a notice
    so a stale config doesn't crash the run.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # Sniff --config without disturbing anything else. allow_abbrev=False is essential:
    # otherwise argparse treats the prefix "--conf" as an abbreviation of "--config" and
    # swallows the unrelated --conf value.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    if pre_args.config:
        loaded = load_config_yaml(pre_args.config)
        known = {a.dest for a in parser._actions}
        unknown = set(loaded) - known - set(HEADER_KEYS)
        if unknown:
            print(f"  [config] ignoring unknown keys from {pre_args.config}: {sorted(unknown)}")
        parser.set_defaults(**{k: v for k, v in loaded.items()
                               if k in known and k not in HEADER_KEYS})
    return argv


def parse_with_config(build_parser, argv=None) -> argparse.Namespace:
    """Two-phase parse: load ``--config FILE`` (YAML) as defaults, then let CLI flags override.

    ``build_parser`` is a zero-arg callable returning the fully-configured parser (it must
    register ``--config`` and all shared + script-specific args).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    apply_config_defaults(parser, argv)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# YAML serialization (scalars only, deterministic order)
# --------------------------------------------------------------------------------------
def _yaml_safe(v):
    """Coerce a value to a plain YAML/JSON scalar, or raise if it isn't one."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        return float(v)
    raise TypeError(f"Non-serializable run-config value: {v!r}")


def config_to_dict(cfg: "RunConfig") -> dict:
    """Plain scalar dict of the shared params (sorted-friendly)."""
    return {p.name: _yaml_safe(getattr(cfg, p.name)) for p in SHARED_PARAMS}


def load_config_yaml(path: str) -> dict:
    """Load a run-config YAML into a plain dict (mapping required)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(data).__name__}.")
    return data


def write_run_config(cfg: "RunConfig", path: str | None = None,
                     extra: dict | None = None) -> str:
    """Write the effective config to YAML (plus command/timestamp header). Returns the path.

    ``extra`` is an optional ``{name: value}`` mapping of script-specific tunables to persist
    alongside the shared params (sanitized to scalars). They load back via
    ``apply_config_defaults`` exactly like any other known argument, so a script can round-trip
    its own settings; keys a different script doesn't recognise are simply ignored on load.
    Keep per-run I/O paths out of ``extra`` — the ``command`` header already records them.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path or f"run_config_{timestamp}.yaml"
    data = config_to_dict(cfg)
    if extra:
        for k, v in extra.items():
            data[k] = _yaml_safe(v)
    data["command"] = " ".join(sys.argv)
    data["timestamp"] = timestamp
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)
    print(f"Saved run config to {path}")
    return path
