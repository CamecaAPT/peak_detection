"""
Build a consolidated truth summary (truth_molecules_canonical_summary.csv + a single
merged *_true_species.txt) from a detect_peaks_refactor.py batch-run output folder, so
generate_synthetic_data.py's --results_dir can point at a small, repo-committed resource
folder instead of the full external batch-run directory.

Usage:
    python scripts/build_truth_molecule_summary.py --results_dir "C:\\path\\to\\results-headless-dev"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic_data import REPO_ROOT, canonical_formula, is_molecule  # noqa: E402

DEFAULT_RESOURCE_DIR = REPO_ROOT / "scripts" / "synthetic_resources"


def collect_truth_molecules(results_dir: Path) -> pd.DataFrame:
    label_by_canonical: dict[str, str] = {}
    occurrences: dict[str, int] = defaultdict(int)

    for csv_path in sorted(results_dir.glob("*/*_detailed_results.csv")):
        df = pd.read_csv(csv_path, usecols=lambda c: c == "true element label")
        if "true element label" not in df.columns:
            continue
        for label in df["true element label"].dropna().astype(str):
            label = label.strip()
            if not label or label == "Unknown":
                continue
            canonical = canonical_formula(label)
            if not is_molecule(canonical):
                continue
            label_by_canonical.setdefault(canonical, label)
            occurrences[canonical] += 1

    rows = [
        {
            "canonical_formula": canonical,
            "truth_label": label_by_canonical[canonical],
            "instance_count": occurrences[canonical],
        }
        for canonical in sorted(label_by_canonical)
    ]
    return pd.DataFrame(rows, columns=["canonical_formula", "truth_label", "instance_count"])


def collect_merged_true_species(results_dir: Path) -> list[str]:
    labels: set[str] = set()
    for txt_path in sorted(results_dir.glob("*/*_true_species.txt")):
        for line in txt_path.read_text().splitlines():
            label = line.strip()
            if label:
                labels.add(label)
    return sorted(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Batch-run output folder (contains per-dataset */*_detailed_results.csv "
                             "and */*_true_species.txt).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RESOURCE_DIR,
                         help="Where to write the consolidated truth summary "
                              "(default: scripts/synthetic_resources).")
    args = parser.parse_args()

    detailed_files = list(args.results_dir.glob("*/*_detailed_results.csv"))
    species_files = list(args.results_dir.glob("*/*_true_species.txt"))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_truth_molecules(args.results_dir)
    summary_path = args.output_dir / "truth_molecules_canonical_summary.csv"
    summary.to_csv(summary_path, index=False)

    merged_labels = collect_merged_true_species(args.results_dir)
    species_path = args.output_dir / "merged_true_species.txt"
    species_path.write_text("\n".join(merged_labels) + "\n")

    print(f"Found {len(summary)} distinct truth molecules and {len(merged_labels)} distinct "
          f"true-species labels across {len(detailed_files)} detailed_results file(s) / "
          f"{len(species_files)} true_species file(s).")
    print(f"Saved: {summary_path}")
    print(f"Saved: {species_path}")


if __name__ == "__main__":
    main()
