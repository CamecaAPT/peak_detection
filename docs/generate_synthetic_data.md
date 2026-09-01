# scripts/generate_synthetic_data.py

Generates truth-coverage synthetic training CSVs for the ion-classification models: one CSV per synthetic spectrum, under `peak_detection/IonIdentificationModels/training_data/<output_name>/Data0001/`. Coverage compositions are seeded from the truth molecules/elements found in a prior evaluation run (via `--results_dir`, defaulting to the committed `scripts/synthetic_resources/` summary built by `build_truth_molecule_summary.py`), then padded out with random compositions drawn from a materials-project compound list.

## Usage

```powershell
.venv\Scripts\python.exe scripts\generate_synthetic_data.py --num_files 5000 --output_name NewData
```

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--num_files NUM_FILES` | No | `5000` | Number of synthetic spectrum CSVs to generate. |
| `--seed SEED` | No | `20260609` | Random seed for `random`/`numpy`. |
| `--output_name OUTPUT_NAME` | No | `NewData` | Name of the output folder created under `training_data/` (ignored if `--output_dir` is set). |
| `--output_dir OUTPUT_DIR` | No | `training_data/<output_name>/Data0001` | Directory for the generated `*.csv` files. |
| `--results_dir RESULTS_DIR` | No | `scripts/synthetic_resources` | Directory with the truth-summary CSVs/species files used to build coverage compositions (built via `build_truth_molecule_summary.py`). |
| `--existing_training_dir EXISTING_TRAINING_DIR` | No | none | Prior training dataset to scan for extra molecule labels to carry forward. |
| `--peak_shift PEAK_SHIFT` | No | `5` | Max random peak position jitter (in 0.01 Da units). |
| `--noise NOISE` | No | `10.0` | Std-dev of Gaussian count noise added per peak. |
| `--noise_ground_level NOISE_GROUND_LEVEL` | No | `20.0` | Minimum intensity a peak must exceed to be kept. |
| `--overlap_limit OVERLAP_LIMIT` | No | `0.03` | Da distance under which nearby peaks are grouped into one row (`ion`/`ion2`). |
| `--light_molecule_charge1_only` | No | off (flag) | Restrict the predefined low-mass molecule list to charge state 1+ only. |
| `--light_molecule_charge2_exceptions [...]` | No | `BO C2O C3` | Canonical low-mass molecule formulas that also get 2+ when `--light_molecule_charge1_only` is set. |

## Notes

- Overwrites any existing `*.csv` files already in `--output_dir` before generating.
- Also writes, alongside `Data0001/`: `MostCommonChargeState.csv`, `truth_molecule_coverage_summary.csv`, `truth_element_coverage_summary.csv`, and `generation_manifest.txt` (run parameters + any truth elements/molecules that ended up with zero generated samples).
- Reads `peak_detection/data/periodic_table.json` for isotope abundances — regenerate that file first via `generate_periodic_table.py` if it's missing or stale.

## Reproducibility

`--seed` fixes every `random`/`numpy` draw used *inside* a single run — `main()` calls `random.seed(args.seed)`/`np.random.seed(args.seed)` up front, and the output directory is fully cleared (`for old_file in args.output_dir.glob("*.csv"): old_file.unlink()`) and regenerated from scratch every time, so within one invocation the same `--seed` always produces byte-identical CSVs. The real reproducibility gap is that the molecule/element *universe* fed into those seeded draws is not self-contained — it's assembled from two external, unpinned inputs, making each generated dataset a link in a chain rather than a standalone, reproducible artifact:

- **`--results_dir` (default `scripts/synthetic_resources`)** is itself an output of `build_truth_molecule_summary.py` run against some earlier `detect_peaks_refactor.py` batch evaluation — which in turn ran with whatever RF/RangingNN model existed at that time. If `scripts/synthetic_resources/` is later regenerated from a newer eval run, re-running this script with the same `--seed` produces a **different** dataset, because `truth_molecules_canonical_summary.csv`/`merged_true_species.txt` (and therefore `choose_coverage_compositions()`'s coverage list) changed underneath it.
- **`--existing_training_dir`** lets a new generation carry forward molecule labels seen in a *previous* generated training set (`load_existing_training_molecules`). This means dataset *N+1* can silently inherit labels that only exist because dataset *N* happened to include them — a nested dependency chain back through however many prior generations were chained this way, not just on the immediate `--results_dir` snapshot.
- Neither dependency is content-hashed or copied into the output: `generation_manifest.txt` records the **paths** passed (`results_dir = ...`, `existing_training_dir = ...`), not the content of those directories at generation time. If either path's contents change or the referenced folder is deleted/regenerated, the manifest gives no way to tell that the same command would now produce different data.
- Net effect: two people (or the same person, later) running the identical CLI command with the identical `--seed` can get different training CSVs if their `scripts/synthetic_resources/` or `--existing_training_dir` differ. True reproducibility would require snapshotting (or content-hashing) both inputs alongside the seed, which this script does not currently do.
