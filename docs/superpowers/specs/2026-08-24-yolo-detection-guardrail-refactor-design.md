# yolo_detection.py breakup + RF guardrail extraction + single-yaml-per-model config

Date: 2026-08-24

## Context

`peak_detection/yolo_detection.py` currently holds a single ~1200-line function,
`predict_peak_ranges_yolo`, that does everything: the YOLO1D ranging call, RF
training/inference, and every RF-specific guardrail (mc-distance unknown flagging,
mixed element/molecule flagging, second-pass molecule-only RF, context rescoring,
molecule rescue), plus accuracy-breakdown computation and CSV artifact writing.

A prior session already scaffolded a plug-in architecture around this: `orchestrator.py`
(a model-agnostic CLI), `peak_detection/classifiers/` (a `ClassifierPipeline` ABC +
registry + merged-config loader), and `peak_detection/IonIdentificationModels/RF/rf_pipeline.py`
(registers `"rf"`, but today just delegates straight into the old monolith). Config is
split across `configs/universal.yaml` (ranging + shared guardrail defaults) and
`configs/models/rf.yaml` (RF-only tunables). `detect_peaks_refactor.py` and
`detect_peaks_headless.py` still call `predict_peak_ranges_yolo` directly (RF hardcoded),
bypassing the registry entirely.

The goal of this refactor: finish that migration so a future model can be added by
writing one `IonIdentificationModels/<Model>/` folder + one `configs/models/<name>.yaml`,
with no changes to `yolo_detection.py`, `classifiers/`, or the entry-point scripts.

## Goals

- `yolo_detection.py` contains only the model-agnostic YOLO1D ranging call.
- Guardrail logic that only touches `PeakRange`/`DetailedId` fields and config thresholds
  (unknown flagging, mixed-unknown flagging, context rescoring, accuracy-breakdown
  computation, and their CSV writers) moves to a new, genuinely shared
  `peak_detection/IonIdentificationModels/guardrail.py` — any future model's pipeline can
  call these without depending on RF.
- The molecule-rescue guardrails (steps 4 and 6 below) are RF-specific by construction —
  they train and rerun a *second* RF model as the rescue mechanism — and move to a new
  `RF/molecule_rescue.py` instead of the shared file.
- `RF/rf_pipeline.py` becomes a plain function (`run_rf(ctx) -> dict`, registered via
  `@register("rf")`) that orchestrates the RF model: ranging → train/infer → shared
  guardrails → RF-specific molecule rescue → accuracy breakdown. No class needed (see
  "Registry: plain functions, not a `ClassifierPipeline` ABC" below).
- One self-contained YAML per model (`configs/models/<name>.yaml`); `configs/universal.yaml`
  is deleted.
- `detect_peaks_refactor.py` and `detect_peaks_headless.py` route through the classifier
  registry (`get_pipeline(model_name)(ctx)`) instead of calling `yolo_detection.py`
  directly, gaining a `--model` flag (default `"rf"`).
- `peak_detection/classifiers/` stays a separate, generic package (framework, not tied to
  any model).

## Non-goals

- No second real model is implemented in this pass — only the seams for one are laid.
- No change to the accuracy-breakdown dict *shape* that `detect_peaks_refactor.py`'s
  plotting/summary code expects; a future model's breakdown just needs to satisfy that
  shape (or the existing graceful fallback when `'counts'` is absent kicks in).
- No behavior change to detection results themselves — this is a structural refactor,
  verified by before/after diffing on the same input.
- `DatasetStats` (`peak_detection/models.py`) and the `plot_rf_*` plotting functions are
  untouched. They're already decoupled from this refactor (`process_dataset()` builds
  `DatasetStats` itself from the returned breakdown `dict`) and are inherently RF-shaped
  today (species/elemental/molecular counts, molecule-rescue fields). Splitting it into a
  universal base + a per-model subclass (e.g. `RFDatasetStats`) is deferred until there's
  an actual second model to design that split against.

## Registry: plain functions, not a `ClassifierPipeline` ABC

The original scaffolding used a `ClassifierPipeline` ABC (one abstract method, `run`) plus
a registry that instantiates the class before calling `.run(ctx)`. RF's implementation
has no `__init__` and no instance state, so the class is pure ceremony. Simplify to:

- `classifiers/base.py`: keep only the `ClassifierContext` dataclass; drop
  `ClassifierPipeline`.
- `classifiers/__init__.py`: `_REGISTRY: dict[str, Callable[[ClassifierContext], dict]]`;
  `register(name)` decorates a plain function; `get_pipeline(name)` returns that function
  directly (no instantiation step).
- `RF/rf_pipeline.py`: `class RFClassifierPipeline(ClassifierPipeline): def run(self, ctx):
  ...` becomes `@register("rf")\ndef run_rf(ctx: ClassifierContext) -> dict: ...`.
- Every call site changes from `get_pipeline(name).run(ctx)` to `get_pipeline(name)(ctx)`.

If a future model needs to cache expensive state (a loaded checkpoint) across multiple
`run()` calls in a batch, that's a reason to reintroduce a class *for that model* — it
isn't needed today and isn't a breaking change to add later.

## Module layout

```
peak_detection/
  yolo_detection.py
    run_yolo_ranging(spectrum_log, *, yolo_weights, iou, conf, max_det) -> list[PeakRange] | None
      - Loads modelweights + prediction_args.yaml, runs DetectionPredictor, converts
        raw predictor output to PeakRange(start, end, pos) (the *0.01 multiplier).
      - Returns None (with a printed error) if the weights/cfg files are missing —
        preserves today's "missing model files" graceful path. Returns [] if the model
        ran but found nothing (a real, distinct outcome from None).
      - No RF, no guardrails, no CSV writing, no _STEP_TIMINGS global.

  classifiers/                     (unchanged home, simplified internals)
    base.py
      - Drops ClassifierPipeline (see "Registry" section above). ClassifierContext gains
        two optional fields: species_list: list[str] | None, elements_list: list[str] |
        None (headless's explicit-species mode, used when there's no RRNG truth file to
        derive the RF class list from).
    config.py
      - load_merged_config(model_name, *, configs_dir, override_path=None): loads
        configs/models/<model_name>.yaml, deep-merges override_path if given. No
        universal.yaml step.
    __init__.py                    - registry becomes function-based (see above).

  IonIdentificationModels/
    guardrail.py (NEW, shared across models)
      Standalone functions operating only on list[PeakRange]/DetailedId + config
      thresholds — no RF-specific calls — mirroring today's model-agnostic steps inside
      predict_peak_ranges_yolo:
        - flag_unknown_peaks(...)                     [step 3: strict mc-distance +
          low-confidence flagging]
        - context_rescore_peaks(...)                  [step 5: neighbor-weighted
          candidate rescoring over each peak's existing top-2 candidates; writes its own
          context-override CSV whenever there are override rows — unconditional on
          save_artifacts, matching today's behavior]
        - flag_high_confidence_mixed_unknowns(...)
        - compute_accuracy_breakdown(...), empty_accuracy_breakdown()
        - write_detailed_results_csv(...), write_unknown_peak_error_report(...)
        - private mc-distance helpers used by several of the above:
          _min_abs_distance_to_samples, _nearest_sample_value,
          _best_match_to_species_samples, _min_abs_distance_to_species_samples
    RF/
      rf_model.py                  - unchanged.
      molecule_rescue.py (NEW, RF-specific)
        The two guardrail steps that retrain/rerun a second RF model:
          - rescue_unknowns_with_molecule_rf(...)       [step 4: second-pass molecule RF
            on peaks flagged unknown]
          - rescue_elements_with_molecule_rf(...)        [step 6: molecule-rescue pass on
            elemental winners; writes its own rescue-candidate CSV whenever there are
            rescue rows — unconditional on save_artifacts, matching today]
        - train_molecule_only_rf(...) — the lazy-training helper both steps share.
      rf_pipeline.py
        flat_rf_kwargs(cfg)                              - unchanged.
        run_rf(ctx) -> dict, registered via @register("rf"), becomes the orchestrator:
          1. Resolve target species/elements (label_map, truth_species_primary,
             truth_molecules, elements_for_molecules) from ctx.truth_data or
             ctx.species_list/ctx.elements_list — moved here verbatim from the old
             function (RF-specific: this is how RF derives its class list).
          2. peaks = run_yolo_ranging(...); if None, return {} (ctx.peaks = []).
          3. Train + run RF (rf_model.create_RF_model / run_RF_model) on the primary
             class list.
          4. Build the mc-sample lookup (training.build_empirical_mc_samples).
          5. Call guardrail.flag_unknown_peaks, then molecule_rescue's
             rescue_unknowns_with_molecule_rf, then guardrail.context_rescore_peaks, then
             molecule_rescue's rescue_elements_with_molecule_rf, then
             guardrail.flag_high_confidence_mixed_unknowns — same order as today — lazily
             training the molecule-only RF once (up front) if either molecule_rescue step
             needs it (today's per-step laziness collapses to "trained once if
             truth_molecules and (unknown_molecule_rf or molecule_rf_rescue_elements)" —
             same actual trigger condition, no observable behavior change).
          6. guardrail.compute_accuracy_breakdown() before/after rescue; write
             detailed-results CSV and unknown-peak-error-report CSV via guardrail.py.
          7. Map labels back through label_map (RRNG display labels).
          8. Save the per-call args snapshot YAML when ctx.save_artifacts is True (folds
             the previously-separate `save_args` flag into `save_artifacts`, since nothing
             toggles them independently today).
          9. Time each stage locally (perf_counter deltas) and print the same
             "Step timing: ranging Xs | RF training Ys | RF inference Zs" summary line,
             replacing the removed _STEP_TIMINGS global.
```

## Config

- Delete `configs/universal.yaml`.
- `configs/models/rf.yaml` becomes fully self-contained: keeps its existing `model: rf`,
  `training:`, `guardrails.molecule_rescue`, `guardrails.unknown_molecule_rf` blocks, and
  gains the `ranging:` block and `guardrails.unknown_flagging` / `guardrails.mixed_unknown`
  / `guardrails.context_rescore` blocks moved verbatim (same values) from the old
  `universal.yaml`.
- `RUN_CONFIG.md` updated to describe the single-file-per-model config (drop references to
  `configs/universal.yaml` and the merge-order section; document that each model owns one
  yaml).

## Entry points

- `detect_peaks_headless.py`: replace the direct `predict_peak_ranges_yolo(...)` call with
  building a `ClassifierContext` (`rrng_file=None`, `truth_data=[]`,
  `elements_for_molecules=[]`, `species_list=species_list`, `elements_list=elements_list`,
  plus the existing prefix/artifacts_dir/save_artifacts/cfg) and calling
  `get_pipeline(args.model)(ctx)`; use `ctx.peaks` for `.rrng`/`.txt` output. Add
  `--model` (default `"rf"`); `cfg = load_merged_config(args.model, ...)`.
- `detect_peaks_refactor.py`: `process_dataset()` gains a `model_name: str = "rf"` parameter;
  replace its direct `predict_peak_ranges_yolo(...)` call the same way, building a
  `ClassifierContext` from its already-parsed truth data and calling
  `get_pipeline(model_name)(ctx)`. Add `--model` (default `"rf"`) to `main()`, threaded
  through `run_batch`. Plotting/batch-summary code is unchanged (consumes `ctx.peaks` +
  the breakdown dict, already model-agnostic).
- `orchestrator.py`: unchanged (already routes through the registry).

## Package exports

- `peak_detection/__init__.py`: replace `from .yolo_detection import predict_peak_ranges_yolo`
  with `from .yolo_detection import run_yolo_ranging` in both the import and `__all__`.

## Cleanup

- Delete the stale `peak_detection/classifiers/__pycache__/` directory (holds a `.pyc` for
  an `rf_pipeline.py` that no longer lives in that folder).

## Verification

No automated test suite covers this pipeline. Verify by running both
`detect_peaks_refactor.py` and `detect_peaks_headless.py --save-artifacts` against
`data/APT_test/R13_40310Zr Unsaved - Top Level ROI.csv` +
`data/RRNG_test/R13_40310Zr Top Level ROI.RRNG` before and after the refactor, and diffing:
- the `*_detailed_results.csv` (per-peak predicted labels, scores, method)
- the printed accuracy percentages (species/elemental/molecular, including/excluding
  unknowns)
- the peak count / precision / recall / F1 line

An exact match on all of the above confirms the restructuring didn't change behavior.
(`--save-artifacts` is passed explicitly on both runs so the args-snapshot-folding change
noted above — the one deliberate, approved behavior difference — doesn't show up as a
spurious diff: without it, a pre-refactor headless run still writes `*_yolo_args.yaml`
while a post-refactor one wouldn't, since that snapshot now follows `save_artifacts`.)
