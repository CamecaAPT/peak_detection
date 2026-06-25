"""
detect_peaks_headless.py — Non-interactive peak detection + identification.

Designed to be driven by another program: activate the venv, then call this
script with an APT/CSV input, the expected species (a list, or an RRNG to parse
them from), an optional artifacts directory, and a path for the output range
file. Nothing is plotted. The output range file is always written; every other
on-disk output is optional.

Mirrors the model behavior of `detect_peaks_refactor.process_dataset` but without
the evaluation/plotting machinery. All YOLO + RF tunables exposed by
`process_dataset` are available here.

Usage:
    python detect_peaks_headless.py \
        --input "R13.apt" \
        --elements "Zr,O,Ti,ZrO,ZrH" \
        --output-rrng "out/R13_predicted.rrng" \
        --artifacts-dir "out/artifacts" --save-artifacts \
        --yolo-weights best_v0_2026-05-12.pt --include-molecules

    # Expected species parsed from an existing range file instead of a list:
    python detect_peaks_headless.py \
        --input "R13.csv" --expected-rrng "R13.RRNG" \
        --output-rrng "out/R13_predicted.rrng"

    # Callable from Python:
    from detect_peaks_headless import detect_peaks_headless
    ranges = detect_peaks_headless('R13.apt', output_rrng='out.rrng',
                                   elements=['Zr', 'O', 'ZrO'])
"""

import os
import sys
import time
import argparse

# Ensure project root is on path for the peak_detection package.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from peak_detection.data_io import (
    load_apt_from_file,
    parse_rrng,
    extract_elements_from_rrng,
    save_rrng,
)
from peak_detection.yolo_detection import predict_peak_ranges_yolo
from peak_detection.training import set_progress_min_fraction
from peak_detection.run_config import (
    add_shared_args,
    apply_config_defaults,
    config_from_namespace,
    write_run_config,
)

# Script-specific tunables (beyond the shared RunConfig) that are persisted to / loadable
# from the run-config YAML. Per-run I/O paths and expected-species inputs are deliberately
# omitted (those change every run and the `command` header records them).
SCRIPT_CONFIG_KEYS = ['save_artifacts', 'save_peak_ranges_txt',
                      'separate_molecule_rf', 'progress_min_fraction']


def _resolve_expected_species(elements=None, expected_rrng=None):
    """
    Resolve the expected species/elements that seed the RF classifier.

    Exactly one source is used:
      - `elements`: an explicit list (or comma-separated string) of species. May
        mix elements and molecules (e.g. ['Zr', 'O', 'ZrO']). Base elements are
        derived from these inside predict_peak_ranges_yolo.
      - `expected_rrng`: a range file whose labels define the species list and
        whose decomposed symbols define the base elements (same as the RRNG path
        in the original script).

    Returns (species_list, elements_list). Either may be None, meaning "let
    predict_peak_ranges_yolo derive it".
    """
    if elements is not None:
        if isinstance(elements, str):
            species = [s.strip() for s in elements.split(',') if s.strip()]
        else:
            species = [str(s).strip() for s in elements if str(s).strip()]
        if not species:
            raise ValueError("--elements was provided but contained no species.")
        return species, None

    if expected_rrng is not None:
        if not os.path.exists(expected_rrng):
            raise FileNotFoundError(f"Expected-species range file not found: {expected_rrng}")
        truth = parse_rrng(expected_rrng)
        species = sorted({str(t.label) for t in truth if t.label and t.label != 'Unknown'})
        if not species:
            raise ValueError(f"No usable species labels parsed from {expected_rrng}.")
        elements_list = extract_elements_from_rrng(expected_rrng)
        return species, elements_list

    raise ValueError("Provide expected species via either `elements` or `expected_rrng`.")


def detect_peaks_headless(
    input_file: str,
    output_rrng: str,
    *,
    # Expected species (exactly one of these)
    elements=None,
    expected_rrng: str = None,
    # Output / artifact control
    artifacts_dir: str = None,
    save_artifacts: bool = False,
    save_peak_ranges_txt: bool = False,
    # YOLO parameters
    yolo_weights: str = 'best_v0_2026-06-23.pt',
    n_iter: int = 0,
    iou: float = 0.01,
    conf: float = 0.05,
    max_det: int = 2000,
    iter_min_intensity_quantile: float = 0.10,
    iter_min_intensity_fraction: float = 0.50,
    iter_intensity_stat_quantile: float = 0.90,
    mc_min: float = 0.0,
    mc_max: float = 307.2,
    # RF parameters
    training_path: str = None,
    training_num_files: int = 10000,
    augment_molecule_training_charge_ratios: bool = False,
    molecule_rf_rescue_elements: bool = False,
    molecule_rf_rescue_threshold: float = 0.8,
    molecule_rf_rescue_margin: float = 0.15,
    molecule_rf_rescue_score_margin: float = 0.05,
    molecule_rf_rescue_dist_margin: float = 0.05,
    include_molecules: bool = False,
    use_neighborhood: bool = False,
    neighbor_threshold: float = 2.0,
    use_signature: bool = False,
    separate_molecule_rf: bool = False,
    unknown_molecule_rf: bool = False,
    unknown_molecule_rf_threshold: float = 0.8,
    followon_mc_vector_rf: bool = False,
    followon_mc_vector_round_decimals: int = 3,
    # Unknown flagging
    flag_unknowns: bool = True,
    mc_threshold: float = 0.2,
    unknown_confidence_threshold: float = 0.6,
    rf_accuracy_top_n: int = 1,
    # Progress reporting
    progress_min_fraction: float = None,
    # Context rescoring
    context_rescore: bool = False,
    context_window_da: float = 2.0,
    context_strength: float = 0.35,
    context_min_confidence: float = 0.75,
    context_min_candidate_confidence: float = 0.05,
    context_override_margin: float = 0.05,
    context_distance_sigma: float = 0.75,
    context_rescue_unknown_same_label: bool = True,
    context_rescue_unknown_min_score: float = 0.7,
):
    """
    Detect and identify peaks for a single dataset and write a range (.rrng)
    file. Returns the list of detected PeakRange objects.

    No plotting is performed. The only guaranteed on-disk output is `output_rrng`.
    Diagnostic CSVs are written only when `save_artifacts=True` (into
    `artifacts_dir` if given, otherwise a dataset-named folder).
    """
    set_progress_min_fraction(progress_min_fraction)
    _t_file = time.perf_counter()

    species_list, elements_list = _resolve_expected_species(elements, expected_rrng)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Detecting peaks for {input_file}")
    print(f"  Expected species ({len(species_list)}): {', '.join(species_list)}")

    x, spectrum, spectrum_log = load_apt_from_file(input_file)
    if x is None:
        raise RuntimeError(f"Failed to load input file: {input_file}")

    # prefix only matters for diagnostic-CSV filenames when save_artifacts is on.
    prefix = os.path.splitext(os.path.basename(output_rrng))[0]

    detected, _, _rf_acc, _rf_acc_ele, _unknown_count = predict_peak_ranges_yolo(
        input_file, spectrum_log, x, None,  # rrng_file=None -> no truth/eval
        n_iter=n_iter, prefix=prefix,
        flag_unknowns=flag_unknowns,
        mc_threshold=mc_threshold,
        training_path=training_path,
        training_num_files=training_num_files,
        augment_molecule_training_charge_ratios=augment_molecule_training_charge_ratios,
        molecule_rf_rescue_elements=molecule_rf_rescue_elements,
        molecule_rf_rescue_threshold=molecule_rf_rescue_threshold,
        molecule_rf_rescue_margin=molecule_rf_rescue_margin,
        molecule_rf_rescue_score_margin=molecule_rf_rescue_score_margin,
        molecule_rf_rescue_dist_margin=molecule_rf_rescue_dist_margin,
        include_molecules=include_molecules,
        yolo_weights=yolo_weights, iou=iou, conf=conf, max_det=max_det,
        iter_min_intensity_quantile=iter_min_intensity_quantile,
        iter_min_intensity_fraction=iter_min_intensity_fraction,
        iter_intensity_stat_quantile=iter_intensity_stat_quantile,
        mc_min=mc_min, mc_max=mc_max,
        use_neighborhood=use_neighborhood, neighbor_threshold=neighbor_threshold,
        use_signature=use_signature,
        separate_molecule_rf=separate_molecule_rf,
        unknown_molecule_rf=unknown_molecule_rf,
        molecule_rf_threshold=unknown_molecule_rf_threshold,
        unknown_confidence_threshold=unknown_confidence_threshold,
        rf_accuracy_top_n=rf_accuracy_top_n,
        context_rescore=context_rescore,
        context_window_da=context_window_da,
        context_strength=context_strength,
        context_min_confidence=context_min_confidence,
        context_min_candidate_confidence=context_min_candidate_confidence,
        context_override_margin=context_override_margin,
        context_distance_sigma=context_distance_sigma,
        context_rescue_unknown_same_label=context_rescue_unknown_same_label,
        context_rescue_unknown_min_score=context_rescue_unknown_min_score,
        followon_mc_vector_rf=followon_mc_vector_rf,
        followon_mc_vector_round_decimals=followon_mc_vector_round_decimals,
        species_list=species_list,
        elements_list=elements_list,
        save_artifacts=save_artifacts,
        artifacts_dir=artifacts_dir,
    )

    # --- REQUIRED OUTPUT: range file ---
    out_parent = os.path.dirname(os.path.abspath(output_rrng))
    os.makedirs(out_parent, exist_ok=True)
    save_rrng(output_rrng, detected)
    print(f"Output range file written: {output_rrng} ({len(detected)} ranges)")

    # --- OPTIONAL: plain-text peak ranges ---
    if save_peak_ranges_txt:
        ranges_dir = artifacts_dir or out_parent
        os.makedirs(ranges_dir, exist_ok=True)
        ranges_txt = os.path.join(ranges_dir, f"{prefix}_peak_ranges.txt")
        with open(ranges_txt, 'w') as f:
            f.write("peak_start, peak_end, round, peak_pos\n")
            for p in detected:
                f.write(f"{p.start:.4f}, {p.end:.4f}, 1, {p.pos:.4f}\n")
        print(f"Peak ranges text written: {ranges_txt}")

    print(f"Total processing time for {os.path.basename(input_file)}: {time.perf_counter() - _t_file:.2f}s")

    return detected


def main():
    parser = argparse.ArgumentParser(
        description="Non-interactive APT peak detection + identification (headless).",
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML run-config file. Its values become defaults that "
                             "explicit CLI flags still override. (Required I/O flags must still "
                             "be supplied on the command line.)")

    # I/O
    parser.add_argument("--input", required=True,
                        help="Path to the input .apt or .csv file.")
    parser.add_argument("--output-rrng", required=True,
                        help="Path for the output range (.rrng) file (always written).")

    # Expected species: exactly one of these
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--elements", type=str, default=None,
                               help="Comma-separated expected species (elements and/or "
                                    "molecules, e.g. 'Zr,O,Ti,ZrO,ZrH').")
    species_group.add_argument("--expected-rrng", type=str, default=None,
                               help="Range file to parse expected species/elements from.")

    # Artifact control (everything except the output range file is optional)
    parser.add_argument("--artifacts-dir", type=str, default=None,
                        help="Directory for optional diagnostic artifacts.")
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=False,
                        help="Write per-dataset diagnostic CSVs (detailed results, unknown report).")
    parser.add_argument("--save-peak-ranges-txt", action=argparse.BooleanOptionalAction, default=False,
                        help="Also write a plain-text peak_ranges.txt.")

    # Shared YOLO / RF / unknown-flagging / context-rescoring parameters
    # (single source of truth: peak_detection/run_config.py). These accept both
    # hyphen (e.g. --yolo-weights) and underscore (--yolo_weights) spellings.
    add_shared_args(parser)

    # Script-specific: molecule-only RF mode (not part of the shared config).
    parser.add_argument("--separate-molecule-rf", "--separate_molecule_rf",
                        dest="separate_molecule_rf",
                        action=argparse.BooleanOptionalAction, default=False)

    # Progress reporting
    parser.add_argument("--progress-min-fraction", type=float, default=None,
                        help="Throttle training-data progress bars to ~one update per this "
                             "fraction of progress (e.g. 0.2 = every 20%%). Default: continuous.")

    # Apply --config YAML as defaults (explicit CLI flags still override), then parse.
    apply_config_defaults(parser)
    args = parser.parse_args()

    cfg = config_from_namespace(args)
    # Script-specific tunables to persist in the run config (I/O paths + expected-species
    # inputs are excluded). These load back via --config too.
    write_run_config(cfg, extra={k: getattr(args, k) for k in SCRIPT_CONFIG_KEYS})

    try:
        detect_peaks_headless(
            args.input,
            args.output_rrng,
            elements=args.elements,
            expected_rrng=args.expected_rrng,
            artifacts_dir=args.artifacts_dir,
            save_artifacts=args.save_artifacts,
            save_peak_ranges_txt=args.save_peak_ranges_txt,
            separate_molecule_rf=args.separate_molecule_rf,
            progress_min_fraction=args.progress_min_fraction,
            **cfg.to_kwargs(),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
