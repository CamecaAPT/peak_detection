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
    yolo_weights: str = 'best_v0_2025-11-12.pt',
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

    return detected


def main():
    parser = argparse.ArgumentParser(
        description="Non-interactive APT peak detection + identification (headless).",
    )

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

    # YOLO parameters
    parser.add_argument("--yolo-weights", type=str, default='best_v0_2025-11-12.pt')
    parser.add_argument("--n-iter", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.01)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--iter-min-intensity-quantile", type=float, default=0.10,
                        help="For YOLO iterative reruns, use this first-pass peak-intensity quantile to set the minimum intensity gate")
    parser.add_argument("--iter-min-intensity-fraction", type=float, default=0.50,
                        help="For YOLO iterative reruns, require new ranges to be at least this fraction of the first-pass intensity quantile")
    parser.add_argument("--iter-intensity-stat-quantile", type=float, default=0.90,
                        help="Within each candidate range, use this intensity quantile as the robust peak intensity statistic")
    parser.add_argument("--mc-min", type=float, default=0.0)
    parser.add_argument("--mc-max", type=float, default=307.2)

    # RF parameters
    parser.add_argument("--training-path", type=str,
                        default='peak_detection/Ionclassifier/training_data/NewData_peakshift0_noise0/Data0001')
    parser.add_argument("--training-num-files", type=int, default=10000)
    parser.add_argument("--augment-molecule-training-charge-ratios",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--molecule-rf-rescue-elements", action=argparse.BooleanOptionalAction, default=False,
                        help="Run a molecule-only RF pass on peaks currently labeled as single elements and allow molecule overrides or mixed element+molecule top-2 candidates")
    parser.add_argument("--molecule-rf-rescue-threshold", type=float, default=0.8,
                        help="Min molecule RF confidence to accept a molecule rescue candidate")
    parser.add_argument("--molecule-rf-rescue-margin", type=float, default=0.15,
                        help="Confidence margin for molecule rescue: above element by this amount overrides; within this amount may be stored as mixed top-2")
    parser.add_argument("--molecule-rf-rescue-score-margin", type=float, default=0.05,
                        help="Quality-weighted score margin for molecule rescue overrides or mixed top-2 candidates")
    parser.add_argument("--molecule-rf-rescue-dist-margin", type=float, default=0.05,
                        help="m/c distance tolerance for molecule rescue; strict improvements can override, close overlaps can become mixed top-2")
    parser.add_argument("--include-molecules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-neighborhood", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--neighbor-threshold", type=float, default=2.0)
    parser.add_argument("--use-signature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--separate-molecule-rf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unknown-molecule-rf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unknown-molecule-rf-threshold", type=float, default=0.8)
    parser.add_argument("--followon-mc-vector-rf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--followon-mc-vector-round-decimals", type=int, default=3)

    # Unknown flagging
    parser.add_argument("--flag-unknowns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mc-threshold", type=float, default=0.2)
    parser.add_argument("--unknown-confidence-threshold", type=float, default=0.6,
                        help="Flag RF IDs as Unknown when the top candidate confidence is below this cutoff; set <=0 to disable")
    parser.add_argument("--rf-accuracy-top-n", type=int, default=1,
                        help="Consider the top N stored RF candidates when scoring element/molecule classification accuracy")

    # Context rescoring
    parser.add_argument("--context-rescore", action=argparse.BooleanOptionalAction, default=False,
                        help="Use nearby peak labels to rescore ambiguous RF candidates after initial classification")
    parser.add_argument("--context-window-da", type=float, default=2.0,
                        help="m/c window around a peak used to collect neighboring RF label support for context rescoring")
    parser.add_argument("--context-strength", type=float, default=0.35,
                        help="Weight applied to neighboring-label support during context rescoring")
    parser.add_argument("--context-min-confidence", type=float, default=0.75,
                        help="Only rescore peaks that are Unknown or whose top RF confidence is below this value")
    parser.add_argument("--context-min-candidate-confidence", type=float, default=0.05,
                        help="Minimum RF candidate confidence for a label to be eligible during context rescoring")
    parser.add_argument("--context-override-margin", type=float, default=0.05,
                        help="Require the context-adjusted winning score to beat the original top candidate by this margin")
    parser.add_argument("--context-distance-sigma", type=float, default=0.75,
                        help="Gaussian distance scale, in Da, for weighting nearby peaks during context rescoring")
    parser.add_argument("--context-rescue-unknown-same-label", action=argparse.BooleanOptionalAction, default=True,
                        help="When context rescoring is enabled, unflag Unknown peaks if nearby context strongly supports their existing top RF candidate")
    parser.add_argument("--context-rescue-unknown-min-score", type=float, default=0.7,
                        help="Minimum context-adjusted score needed to unflag an Unknown peak whose top RF candidate remains the winner")

    args = parser.parse_args()

    try:
        detect_peaks_headless(
            args.input,
            args.output_rrng,
            elements=args.elements,
            expected_rrng=args.expected_rrng,
            artifacts_dir=args.artifacts_dir,
            save_artifacts=args.save_artifacts,
            save_peak_ranges_txt=args.save_peak_ranges_txt,
            yolo_weights=args.yolo_weights,
            n_iter=args.n_iter,
            iou=args.iou,
            conf=args.conf,
            max_det=args.max_det,
            iter_min_intensity_quantile=args.iter_min_intensity_quantile,
            iter_min_intensity_fraction=args.iter_min_intensity_fraction,
            iter_intensity_stat_quantile=args.iter_intensity_stat_quantile,
            mc_min=args.mc_min,
            mc_max=args.mc_max,
            training_path=args.training_path,
            training_num_files=args.training_num_files,
            augment_molecule_training_charge_ratios=args.augment_molecule_training_charge_ratios,
            molecule_rf_rescue_elements=args.molecule_rf_rescue_elements,
            molecule_rf_rescue_threshold=args.molecule_rf_rescue_threshold,
            molecule_rf_rescue_margin=args.molecule_rf_rescue_margin,
            molecule_rf_rescue_score_margin=args.molecule_rf_rescue_score_margin,
            molecule_rf_rescue_dist_margin=args.molecule_rf_rescue_dist_margin,
            include_molecules=args.include_molecules,
            use_neighborhood=args.use_neighborhood,
            neighbor_threshold=args.neighbor_threshold,
            use_signature=args.use_signature,
            separate_molecule_rf=args.separate_molecule_rf,
            unknown_molecule_rf=args.unknown_molecule_rf,
            unknown_molecule_rf_threshold=args.unknown_molecule_rf_threshold,
            followon_mc_vector_rf=args.followon_mc_vector_rf,
            followon_mc_vector_round_decimals=args.followon_mc_vector_round_decimals,
            flag_unknowns=args.flag_unknowns,
            mc_threshold=args.mc_threshold,
            unknown_confidence_threshold=args.unknown_confidence_threshold,
            rf_accuracy_top_n=args.rf_accuracy_top_n,
            context_rescore=args.context_rescore,
            context_window_da=args.context_window_da,
            context_strength=args.context_strength,
            context_min_confidence=args.context_min_confidence,
            context_min_candidate_confidence=args.context_min_candidate_confidence,
            context_override_margin=args.context_override_margin,
            context_distance_sigma=args.context_distance_sigma,
            context_rescue_unknown_same_label=args.context_rescue_unknown_same_label,
            context_rescue_unknown_min_score=args.context_rescue_unknown_min_score,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
