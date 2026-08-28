"""This script generates truth coverage training data for the Ionclassification models."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = REPO_ROOT / "scripts" / "synthetic_resources"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results_summary_2026-06-09_context_top1_cleantruth_rerun"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "peak_detection"
    / "IonIdentificationModels"
    / "training_data"
    / "NewData_truthcoverage_2026-06-09"
    / "Data0001"
)

BACKGROUND_COMPONENTS = [
    {"composition": {"H": 0.667, "O": 0.333}, "min_pct": 0.01, "max_pct": 0.05},
    {"composition": {"O": 0.667, "C": 0.333}, "min_pct": 0.01, "max_pct": 0.05},
    {"composition": {"N": 1.0}, "min_pct": 0.01, "max_pct": 0.03},
]

LANTHANIDES_FOR_EXTRA_CHARGES = {
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
}

LIGHT_MOLECULE_CHARGE1_ONLY_CANONICAL = {
    "AlH",
    "AlH2",
    "BO",
    "BeH",
    "BeH2",
    "BeO",
    "C2",
    "C2O",
    "C3",
    "CN",
    "CSi",
    "H2",
    "H2O",
    "H2O2",
    "H3",
    "H3O",
    "H5O2",
    "HO",
    "HO2",
    "N2",
    "O2",
}

DEFAULT_LIGHT_MOLECULE_EXTRA_CHARGES = {
    "BO": (2,),
    "C2O": (2,),
    "C3": (2,),
}


def parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sym, count_str in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        counts[sym] += int(count_str) if count_str else 1
    return dict(counts)


def canonical_formula(formula: str) -> str:
    counts = parse_formula(formula)
    return "".join(
        sym + (str(counts[sym]) if counts[sym] > 1 else "")
        for sym in sorted(counts)
    )


def is_molecule(formula: str) -> bool:
    counts = parse_formula(formula)
    return sum(counts.values()) > 1


def merge_compositions(dict_list: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for d in dict_list:
        for key, value in d.items():
            result[key] = result.get(key, 0.0) + float(value)
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    return result


def add_background(composition: dict[str, float]) -> dict[str, float]:
    pieces = [composition]
    for bg in BACKGROUND_COMPONENTS:
        pct = random.randint(int(bg["min_pct"] * 100), int(bg["max_pct"] * 100)) * 0.01
        pieces.append({el: frac * pct for el, frac in bg["composition"].items()})
    return merge_compositions(pieces)


def find_groups(series: pd.Series, overlap_limit: float) -> pd.Series:
    groups: list[list[int]] = []
    current_group = [0]
    for i in range(1, len(series)):
        if series.iloc[i] - series.iloc[current_group[-1]] <= overlap_limit:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)
    return pd.Series([group_id for group_id, group in enumerate(groups) for _ in group])


def process_group(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) == 1:
        row = group.iloc[0]
        return pd.DataFrame({**row.to_dict(), "ion2": "", "charge2": ""}, index=[0])

    sorted_group = group.sort_values("counts", ascending=False)
    main_row = sorted_group.iloc[0].copy()
    main_row["counts"] = sorted_group["counts"].sum()
    main_row["ion2"] = sorted_group.iloc[1]["ion"]
    main_row["charge2"] = sorted_group.iloc[1]["charge"]
    return pd.DataFrame(main_row).T


def register_preferred_label(label_by_canonical: dict[str, str], label: str) -> None:
    label = str(label).strip()
    if not label:
        return
    canonical = canonical_formula(label)
    if is_molecule(canonical) and canonical not in label_by_canonical:
        label_by_canonical[canonical] = label


def load_truth_molecules(results_dir: Path) -> dict[str, str]:
    files = [
        results_dir / "truth_molecules_canonical_summary.csv",
        results_dir / "truth_molecule_instances_canonical.csv",
        results_dir / "missing_truth_molecules_vs_full_training_canonical.csv",
        results_dir / "truth_molecules_charge_scaled_match_summary.csv",
    ]
    molecules: dict[str, str] = {}
    for path in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "truth_label" in df.columns:
            for label in df["truth_label"].dropna().astype(str):
                register_preferred_label(molecules, label)
        if "truth_label_variants" in df.columns:
            for variants in df["truth_label_variants"].dropna().astype(str):
                for label in variants.split(";"):
                    register_preferred_label(molecules, label)
        if "canonical_formula" not in df.columns:
            continue
        for label in df["canonical_formula"].dropna().astype(str):
            canonical = canonical_formula(label)
            if is_molecule(canonical) and canonical not in molecules:
                molecules[canonical] = canonical
    return molecules


def load_existing_training_molecules(training_dir: Path, limit_files: int = 5000) -> dict[str, str]:
    molecules: dict[str, str] = {}
    for csv_path in sorted(training_dir.glob("*.csv"))[:limit_files]:
        df = pd.read_csv(csv_path, keep_default_na=False, usecols=lambda c: c in {"ion", "ion2"})
        for col in ("ion", "ion2"):
            if col not in df.columns:
                continue
            for label in df[col].dropna().astype(str):
                label = label.strip()
                if not label:
                    continue
                canonical = canonical_formula(label)
                if is_molecule(canonical) and canonical not in molecules:
                    molecules[canonical] = label
    return molecules


def load_periodic_table_elements() -> set[str]:
    with open(REPO_ROOT / "peak_detection" / "data" / "periodic_table.json", encoding="utf-8") as f:
        return set(json.load(f))


def load_isotope_abundances() -> pd.DataFrame:
    with open(REPO_ROOT / "peak_detection" / "data" / "periodic_table.json", encoding="utf-8") as f:
        periodic = json.load(f)

    rows = []
    for element, data in periodic.items():
        for isotope in data.get("isotopes", {}).values():
            rows.append({
                "Element": element,
                "Mass": isotope["mass"],
                "Abund": float(isotope["abundance"]) * 100.0,
            })
    return pd.DataFrame(rows, columns=["Element", "Mass", "Abund"])


def load_truth_elements(results_dir: Path, valid_elements: set[str]) -> set[str]:
    elements: set[str] = set()
    for path in results_dir.glob("*/*_true_species.txt"):
        for line in path.read_text().splitlines():
            label = line.strip()
            if label in valid_elements and not is_molecule(label):
                elements.add(label)
    return elements


def patch_charge_table(charges: pd.DataFrame) -> pd.DataFrame:
    charges = charges.copy()
    if "Rarely" not in charges.columns:
        charges["Rarely"] = np.nan

    for sym in LANTHANIDES_FOR_EXTRA_CHARGES:
        mask = charges["Symbol"].astype(str).eq(sym)
        if not mask.any():
            continue
        if pd.isna(charges.loc[mask, "Most Common Charge State"]).all():
            charges.loc[mask, "Most Common Charge State"] = 1
        if pd.isna(charges.loc[mask, "Second Most Common Charge State"]).all():
            charges.loc[mask, "Second Most Common Charge State"] = 2
        if pd.isna(charges.loc[mask, "Sometimes"]).all():
            charges.loc[mask, "Sometimes"] = 3

    return charges


def choose_random_composition(compound_all: pd.DataFrame) -> dict[str, float]:
    compound_n = random.randint(1, 5)
    indexlist = random.sample(range(0, len(compound_all)), compound_n)
    pieces = []
    for idx in indexlist:
        comp = ast.literal_eval(compound_all.iloc[idx]["composition"])
        pieces.append({str(k): float(v) for k, v in comp.items()})
    return add_background(merge_compositions(pieces))


def choose_coverage_compositions(truth_molecules: set[str], truth_elements: set[str]) -> list[dict[str, float]]:
    compositions: list[dict[str, float]] = []

    for formula in sorted(truth_molecules):
        counts = parse_formula(formula)
        total_atoms = sum(counts.values())
        if total_atoms <= 0:
            continue
        comp = {el: count / total_atoms for el, count in counts.items()}
        compositions.append(add_background(comp))

    for sym in sorted(truth_elements | LANTHANIDES_FOR_EXTRA_CHARGES):
        compositions.append(add_background({sym: 1.0}))

    return compositions


def build_isotope_profile(
    composition: dict[str, float],
    abundance_all: pd.DataFrame,
) -> pd.DataFrame:
    isotope_rows = []
    for element, element_fraction in composition.items():
        el_rows = abundance_all[abundance_all["Element"].astype(str).eq(element)].copy()
        if el_rows.empty:
            continue
        el_rows["ratio"] = el_rows["Abund"].astype(float) * 0.01 * float(element_fraction)
        isotope_rows.append(el_rows)
    if not isotope_rows:
        return pd.DataFrame(columns=["Element", "Mass", "ratio"])
    return pd.concat(isotope_rows, ignore_index=True)


def generate_single_element_peaks(
    isotopes_comp: pd.DataFrame,
    charges: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = isotopes_comp[["Element", "Mass", "ratio"]].copy()
    rows = rows[(rows["ratio"].astype(float) > 0) & (rows["Mass"].astype(float) < 307.2)]

    profile_parts = []
    label_parts = []
    charge_parts = []

    for _, row in rows.iterrows():
        label = str(row["Element"])
        mass = float(row["Mass"])
        ratio = float(row["ratio"])
        charge_row = charges[charges["Symbol"].astype(str).eq(label)]
        if charge_row.empty:
            charge_values = [1.0]
        else:
            charge_values = []
            for col in [
                "Most Common Charge State",
                "Second Most Common Charge State",
                "Sometimes",
                "Rarely",
            ]:
                val = charge_row.iloc[0].get(col, np.nan)
                if pd.notna(val):
                    charge_values.append(float(val))
            charge_values = charge_values or [1.0]

        for idx, charge in enumerate(charge_values):
            if charge <= 0:
                continue
            intensity = ratio
            if idx == 1:
                intensity *= random.randint(50, 67) * 0.01
            elif idx >= 2:
                intensity *= random.randint(30, 50) * 0.01
            profile_parts.append((mass / charge, intensity))
            label_parts.append(label)
            charge_parts.append(charge)

    if not profile_parts:
        return np.zeros((0, 2)), np.array([]), np.array([])
    return (
        np.asarray(profile_parts, dtype=float),
        np.asarray(label_parts, dtype=object),
        np.asarray(charge_parts, dtype=float),
    )


def generate_molecule_peaks(
    isotopes_comp: pd.DataFrame,
    molecule_formulas: set[str],
    molecule_charges: tuple[int, ...],
    charge1_only_canonical: set[str] | None = None,
    extra_charges_by_canonical: dict[str, tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile_parts = []
    label_parts = []
    charge_parts = []
    charge1_only_canonical = charge1_only_canonical or set()
    extra_charges_by_canonical = extra_charges_by_canonical or {}

    for formula in sorted(molecule_formulas):
        counts = parse_formula(formula)
        if not counts or sum(counts.values()) <= 1:
            continue
        canonical = canonical_formula(formula)
        if canonical in charge1_only_canonical:
            formula_charges = tuple(
                sorted({1, *[int(charge) for charge in extra_charges_by_canonical.get(canonical, ())]})
            )
        else:
            formula_charges = molecule_charges

        isotope_lists = []
        missing_element = False
        for element, count in counts.items():
            el_profile = isotopes_comp[isotopes_comp["Element"].astype(str).eq(element)][["Mass", "ratio"]]
            el_profile = el_profile[el_profile["ratio"].astype(float) > 0]
            if el_profile.empty:
                missing_element = True
                break
            values = np.asarray(el_profile, dtype=float)
            for _ in range(count):
                isotope_lists.append(values)
        if missing_element or not isotope_lists:
            continue

        base_scaling = random.randint(30, 50) * 0.01
        for isotope_combo in itertools.product(*isotope_lists):
            combo = np.vstack(isotope_combo)
            neutral_mass = float(combo[:, 0].sum())
            abundance_ratio = float(combo[:, 1].prod()) * base_scaling
            if neutral_mass <= 0 or abundance_ratio <= 0:
                continue

            for charge in formula_charges:
                charge_f = float(charge)
                if charge_f <= 0:
                    continue
                intensity = abundance_ratio
                if charge == 2:
                    intensity *= random.randint(50, 67) * 0.01
                elif charge >= 3:
                    intensity *= random.randint(30, 50) * 0.01
                profile_parts.append((neutral_mass / charge_f, intensity))
                label_parts.append(formula)
                charge_parts.append(charge_f)

    if not profile_parts:
        return np.zeros((0, 2)), np.array([]), np.array([])
    return (
        np.asarray(profile_parts, dtype=float),
        np.asarray(label_parts, dtype=object),
        np.asarray(charge_parts, dtype=float),
    )


def generate_profile(
    composition: dict[str, float],
    abundance_all: pd.DataFrame,
    charges: pd.DataFrame,
    molecule_formulas: set[str],
    molecule_charges: tuple[int, ...],
    charge1_only_canonical: set[str] | None = None,
    extra_charges_by_canonical: dict[str, tuple[int, ...]] | None = None,
    *,
    peak_shift: int,
    noise: float,
    noise_ground_level: float,
    overlap_limit: float,
) -> pd.DataFrame:
    isotopes_comp = build_isotope_profile(composition, abundance_all)
    single_profile, single_labels, single_charges = generate_single_element_peaks(isotopes_comp, charges)
    molecule_profile, molecule_labels, molecule_charge_labels = generate_molecule_peaks(
        isotopes_comp,
        molecule_formulas,
        molecule_charges,
        charge1_only_canonical,
        extra_charges_by_canonical,
    )

    if single_profile.size and molecule_profile.size:
        profile_all = np.vstack([single_profile, molecule_profile])
        ionlabels_all = np.hstack([single_labels, molecule_labels])
        chargelabels_all = np.hstack([single_charges, molecule_charge_labels])
    elif single_profile.size:
        profile_all = single_profile
        ionlabels_all = single_labels
        chargelabels_all = single_charges
    elif molecule_profile.size:
        profile_all = molecule_profile
        ionlabels_all = molecule_labels
        chargelabels_all = molecule_charge_labels
    else:
        return pd.DataFrame(columns=["mc", "counts", "ion", "charge", "ion2", "charge2"])

    count_scale = 1e7 * random.randint(10, 100)
    profile_all = profile_all.copy()
    profile_all[:, 1] *= count_scale

    keep = profile_all[:, 1] > noise_ground_level
    profile_all = profile_all[keep]
    ionlabels_all = ionlabels_all[keep]
    chargelabels_all = chargelabels_all[keep]

    if len(profile_all) == 0:
        return pd.DataFrame(columns=["mc", "counts", "ion", "charge", "ion2", "charge2"])

    for idx in range(profile_all.shape[0]):
        shift = (random.randint(0, peak_shift * 2) - peak_shift) * 0.01
        profile_all[idx, 0] += shift
        profile_all[idx, 1] += np.random.normal(0, noise)

    table = pd.DataFrame(
        {
            "mc": profile_all[:, 0],
            "counts": profile_all[:, 1],
            "ion": ionlabels_all,
            "charge": chargelabels_all,
        }
    )
    table = table[table["mc"].between(0, 307.2)]
    table = table[table["counts"] > 0]
    table = table.sort_values("mc", ignore_index=True)
    if table.empty:
        return pd.DataFrame(columns=["mc", "counts", "ion", "charge", "ion2", "charge2"])

    table["group"] = find_groups(table["mc"], overlap_limit)
    try:
        result = table.groupby("group", group_keys=False).apply(process_group, include_groups=False).reset_index(drop=True)
    except TypeError:
        result = table.groupby("group", group_keys=False).apply(process_group).reset_index(drop=True)
    result = result.reindex(columns=["mc", "counts", "ion", "charge", "ion2", "charge2"])
    result["counts"] = result["counts"].astype(int)
    return result.sort_values("mc", ignore_index=True)


def summarize_output(output_dir: Path, truth_molecules: set[str], num_files: int) -> pd.DataFrame:
    sample_counts: dict[str, int] = defaultdict(int)
    charge_values: dict[str, set[float]] = defaultdict(set)
    file_counts: dict[str, set[str]] = defaultdict(set)

    for csv_path in sorted(output_dir.glob("*.csv"))[:num_files]:
        df = pd.read_csv(csv_path, keep_default_na=False)
        for _, row in df.iterrows():
            for ion_col, charge_col in (("ion", "charge"), ("ion2", "charge2")):
                label = str(row.get(ion_col, "")).strip()
                if not label:
                    continue
                label = canonical_formula(label)
                if not is_molecule(label):
                    continue
                sample_counts[label] += 1
                file_counts[label].add(csv_path.name)
                try:
                    charge = float(row.get(charge_col, ""))
                    if charge > 0:
                        charge_values[label].add(charge)
                except Exception:
                    pass

    rows = []
    for formula in sorted(truth_molecules):
        rows.append(
            {
                "canonical_formula": formula,
                "sample_count": sample_counts.get(formula, 0),
                "file_count": len(file_counts.get(formula, set())),
                "charges": ";".join(f"{c:g}" for c in sorted(charge_values.get(formula, set()))),
            }
        )
    return pd.DataFrame(rows, columns=["canonical_formula", "sample_count", "file_count", "charges"])


def summarize_element_output(output_dir: Path, truth_elements: set[str], num_files: int) -> pd.DataFrame:
    sample_counts: dict[str, int] = defaultdict(int)
    charge_values: dict[str, set[float]] = defaultdict(set)
    file_counts: dict[str, set[str]] = defaultdict(set)

    for csv_path in sorted(output_dir.glob("*.csv"))[:num_files]:
        df = pd.read_csv(csv_path, keep_default_na=False)
        for _, row in df.iterrows():
            for ion_col, charge_col in (("ion", "charge"), ("ion2", "charge2")):
                label = str(row.get(ion_col, "")).strip()
                if label not in truth_elements:
                    continue
                sample_counts[label] += 1
                file_counts[label].add(csv_path.name)
                try:
                    charge = float(row.get(charge_col, ""))
                    if charge > 0:
                        charge_values[label].add(charge)
                except Exception:
                    pass

    rows = []
    for element in sorted(truth_elements):
        rows.append(
            {
                "element": element,
                "sample_count": sample_counts.get(element, 0),
                "file_count": len(file_counts.get(element, set())),
                "charges": ";".join(f"{c:g}" for c in sorted(charge_values.get(element, set()))),
            }
        )
    return pd.DataFrame(rows, columns=["element", "sample_count", "file_count", "charges"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate truth-coverage ion classifier training CSVs.")
    parser.add_argument("--num_files", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--existing_training_dir",
        type=Path,
        default=REPO_ROOT / "peak_detection/IonIdentificationModels/training_data/NewData_peakshift0_noise0_newchg/Data0001",
    )
    parser.add_argument("--peak_shift", type=int, default=5)
    parser.add_argument("--noise", type=float, default=10.0)
    parser.add_argument("--noise_ground_level", type=float, default=20.0)
    parser.add_argument("--overlap_limit", type=float, default=0.03)
    parser.add_argument(
        "--light_molecule_charge1_only",
        action="store_true",
        help="Restrict the predefined low-mass molecule list to charge state 1+ only.",
    )
    parser.add_argument(
        "--light_molecule_charge2_exceptions",
        nargs="*",
        default=sorted(DEFAULT_LIGHT_MOLECULE_EXTRA_CHARGES),
        help=(
            "Canonical low-mass molecule formulas that should also include 2+ "
            "when --light_molecule_charge1_only is enabled."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    compound_all = pd.read_csv(RESOURCE_DIR / "materials_project_compounds_filtered.csv")
    abundance_all = load_isotope_abundances()
    valid_elements = load_periodic_table_elements()
    charges = patch_charge_table(pd.read_csv(RESOURCE_DIR / "MostCommonChargeState.csv"))

    truth_molecule_labels = load_truth_molecules(args.results_dir)
    truth_elements = load_truth_elements(args.results_dir, valid_elements)
    existing_molecule_labels = load_existing_training_molecules(args.existing_training_dir, limit_files=5000)
    label_by_canonical = dict(existing_molecule_labels)
    label_by_canonical.update(truth_molecule_labels)

    truth_molecules = set(truth_molecule_labels)
    molecule_formulas = {
        label_by_canonical[canonical]
        for canonical in label_by_canonical
        if is_molecule(canonical)
    }
    charge2_exceptions = {
        canonical_formula(label): (2,)
        for label in args.light_molecule_charge2_exceptions
        if canonical_formula(label) in LIGHT_MOLECULE_CHARGE1_ONLY_CANONICAL
    }

    coverage_compositions = choose_coverage_compositions(truth_molecules, truth_elements)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in args.output_dir.glob("*.csv"):
        old_file.unlink()

    for idx in range(args.num_files):
        if idx < len(coverage_compositions):
            composition = coverage_compositions[idx]
        else:
            composition = choose_random_composition(compound_all)

        df = generate_profile(
            composition,
            abundance_all,
            charges,
            molecule_formulas,
            molecule_charges=(1, 2, 3, 4),
            charge1_only_canonical=LIGHT_MOLECULE_CHARGE1_ONLY_CANONICAL
            if args.light_molecule_charge1_only
            else set(),
            extra_charges_by_canonical=charge2_exceptions,
            peak_shift=args.peak_shift,
            noise=args.noise,
            noise_ground_level=args.noise_ground_level,
            overlap_limit=args.overlap_limit,
        )
        df.to_csv(args.output_dir / f"{idx:06d}.csv", index=False)
        if (idx + 1) % 500 == 0:
            print(f"Generated {idx + 1}/{args.num_files} files")

    charges.to_csv(args.output_dir.parent / "MostCommonChargeState_truthcoverage_2026-06-09.csv", index=False)

    coverage_summary = summarize_output(args.output_dir, truth_molecules, args.num_files)
    coverage_summary.to_csv(args.output_dir.parent / "truth_molecule_coverage_summary.csv", index=False)
    element_coverage_summary = summarize_element_output(args.output_dir, truth_elements, args.num_files)
    element_coverage_summary.to_csv(args.output_dir.parent / "truth_element_coverage_summary.csv", index=False)

    missing = coverage_summary[coverage_summary["sample_count"].eq(0)].copy()
    missing_elements = element_coverage_summary[element_coverage_summary["sample_count"].eq(0)].copy()
    manifest = [
        "Truth-coverage classifier training data",
        f"num_files = {args.num_files}",
        f"seed = {args.seed}",
        f"output_dir = {args.output_dir}",
        f"results_dir = {args.results_dir}",
        f"existing_training_dir = {args.existing_training_dir}",
        f"truth_element_count = {len(truth_elements)}",
        f"truth_molecule_count = {len(truth_molecules)}",
        f"molecule_formula_count_used_for_generation = {len(molecule_formulas)}",
        "molecule_label_policy = preserve truth-label spelling/order when available; canonical formulas used only for matching",
        "element_coverage_policy = include coverage compositions for every element present in *_true_species.txt",
        "element_charge_override = lanthanides La-Lu get 2+ and 3+ if missing",
        "molecule_charge_states = 1+, 2+, 3+, 4+ for all truth/current training molecule formulas",
        f"light_molecule_charge1_only = {args.light_molecule_charge1_only}",
        "light_molecule_charge2_exceptions = "
        + (", ".join(sorted(charge2_exceptions)) if charge2_exceptions else "none"),
        f"truth_elements_missing_after_generation = {len(missing_elements)}",
        f"truth_molecules_missing_after_generation = {len(missing)}",
    ]
    if args.light_molecule_charge1_only:
        manifest.append(
            "light_molecule_charge1_only_canonical = "
            + ", ".join(sorted(LIGHT_MOLECULE_CHARGE1_ONLY_CANONICAL))
        )
    if len(missing_elements):
        manifest.append("missing_truth_elements = " + ", ".join(missing_elements["element"].astype(str)))
    if len(missing):
        manifest.append("missing_truth_molecules = " + ", ".join(missing["canonical_formula"].astype(str)))
    (args.output_dir.parent / "generation_manifest.txt").write_text("\n".join(manifest) + "\n")

    print("\n".join(manifest))


if __name__ == "__main__":
    main()
