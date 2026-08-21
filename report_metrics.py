"""Shared metric-formatting and truth-label helpers for the run-report scripts.

Kept dependency-light (pandas only, no torch/sklearn/pymatgen): importing anything from
the `peak_detection` package pulls in the full ML stack via peak_detection/__init__.py,
which write_dataset_peak_summaries.py and write_classification_audit_summary.py should
not require just to regenerate a text report from existing output CSVs.
"""

from __future__ import annotations

import re

import pandas as pd


def is_molecule_label(label) -> bool:
    """
    True if a plain composition-format label (no display decoration, e.g. no
    "(85%)" confidence suffix) represents more than one atom, e.g.
    "Fe" -> False, "Fe2" -> True, "FeO" -> True.

    Ground-truth labels parsed from RRNG files are always in this plain form,
    which is the only kind of label this helper is meant to classify.
    """
    label = str(label or "").strip()
    if not label or label == "Unknown":
        return False
    return bool(re.search(r"\d", label)) or len(re.findall(r"[A-Z][a-z]?", label)) > 1


def format_ratio(num: float, den: float) -> str:
    """'12/34 = 0.353 (35.3%)', or 'n/a' if den is zero/NaN/negative."""
    if den is None or pd.isna(den) or den <= 0:
        return "n/a"
    return f"{num}/{den} = {num / den:.3f} ({100.0 * num / den:.1f}%)"


def unknown_truth_element_molecule_split(detailed_df: pd.DataFrame) -> dict:
    """
    Given one dataset's *_detailed_results.csv (already loaded as a DataFrame),
    split rows by whether they matched a ground-truth label, whether that truth
    is an element or a molecule, and whether the prediction was flagged Unknown
    (discarded).
    """
    truth = detailed_df["true element label"].fillna("").astype(str).str.strip()
    matched = truth.ne("") & truth.ne("Unknown")
    discarded = detailed_df["discarded"].astype(str).str.lower().isin({"true", "1", "yes"})

    truth_is_molecule = truth.map(is_molecule_label)
    is_element_row = matched & ~truth_is_molecule
    is_molecule_row = matched & truth_is_molecule

    return {
        "found_truth_rows": int(matched.sum()),
        "found_true_elements": int(is_element_row.sum()),
        "found_true_molecules": int(is_molecule_row.sum()),
        "unknown_true_elements": int((is_element_row & discarded).sum()),
        "unknown_true_molecules": int((is_molecule_row & discarded).sum()),
        "unknown_truth_matched": int((matched & discarded).sum()),
        "unknown_unmatched": int((~matched & discarded).sum()),
    }
