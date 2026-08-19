import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import simplify_label, is_molecule, min_max_scale
from .IonIdentificationModels.RF.rf_model import get_signature_features


# --- Optional progress throttling ------------------------------------------------
# When set to a fraction in (0, 1], the training-data tqdm bars below refresh roughly
# once per that fraction of progress (e.g. 0.2 -> ~ every 20%) instead of continuously.
# None (default) keeps tqdm's current update rate. Set via set_progress_min_fraction()
# so public entry points can plumb it without threading it through every loader.
_progress_min_fraction = None


def set_progress_min_fraction(fraction):
    """Throttle the training-data tqdm bars to ~one update per `fraction` of progress.

    Pass a value in (0, 1] (e.g. 0.2 for every 20%). None preserves the current
    continuous update rate.
    """
    global _progress_min_fraction
    _progress_min_fraction = fraction


def _tqdm(iterable, **kwargs):
    """tqdm wrapper that honors the module-level progress throttle when set."""
    fraction = _progress_min_fraction
    if fraction and fraction > 0:
        total = kwargs.get('total')
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        if total and total > 0:
            # Explicit miniters disables tqdm's dynamic adjustment; mininterval=0 lets
            # miniters alone gate refreshes (~one per `fraction` of the iterations).
            kwargs.setdefault('mininterval', 0)
            kwargs.setdefault('miniters', max(1, int(total * fraction)))
    return tqdm(iterable, **kwargs)


def load_ion_training_data(path='peak_detection/IonIdentificationModels/training_data/NewData/Data0001',
                           element_list=list(),
                           elements_to_get_molecules=list(),
                           threshold_c=1e-8,
                           num_files=10000,
                           neighbor_threshold=0.0,
                           use_signature=False,
                           augment_molecule_charge_ratios: bool = False,
                           molecule_charge_ratios: tuple[float, ...] = (0.5, 1.0 / 3.0)):
    """
    Load the evaluation files, get input and gt, normalized counts,
    including neighborhood features.
    """
    features_all = list()
    ions_all = list()

    # Normalize element_list to ensure consistent matching with simplified training labels
    if element_list != 'all':
        element_list = [simplify_label(str(e)) for e in element_list]

    if not os.path.exists(path):
        print(f"Warning: Training data path {path} not found.")
        return np.array([]), np.array([])

    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])[:num_files]
    max_neigh = 0

    raw_data_per_file = []
    for file in _tqdm(files, desc='Loading and parsing classifier training data'):
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        if 'ion' not in df.columns or 'mc' not in df.columns or 'counts' not in df.columns:
            continue
        
        mc = df.get(['mc']).to_numpy().squeeze()
        counts = df.get(['counts']).to_numpy().squeeze()

        if counts.max() == counts.min():
            counts = np.zeros_like(counts)
        else:
            counts = min_max_scale(counts)

        indexes = counts > threshold_c
        if not np.any(indexes):
            continue

        mc_f = mc[indexes]
        ions_raw_f = df.get(['ion']).to_numpy().squeeze()[indexes]
        ions2_raw_f = df.get(['ion2']).to_numpy().squeeze()[indexes]

        # Filtering logic - resolves ion2/ion preference and applies the element list /
        # molecule discovery rules in a single pass over the threshold-surviving rows
        mc_k_list = []
        ions_k_list = []
        for m_val, i1, i2 in zip(mc_f, ions_raw_f, ions2_raw_f):
            ion_str = str(i2) if (i2 and i2 != "") else str(i1)
            label_simple = simplify_label(ion_str)

            if element_list == 'all':
                keep = True
            elif label_simple in element_list:
                # Priority 1: RRNG / element list whitelist
                keep = True
            elif elements_to_get_molecules:
                # Priority 2: Automatic discovery from base elements (if enabled)
                atoms = re.findall(r'[A-Z][a-z]?', ion_str)
                keep = bool(atoms) and all(a in elements_to_get_molecules for a in atoms) and is_molecule(label_simple)
            else:
                keep = False

            if keep:
                mc_k_list.append(m_val)
                ions_k_list.append(label_simple)

        mc_k = np.asarray(mc_k_list, dtype=float)
        ions_k = ions_k_list

        if augment_molecule_charge_ratios:
            mc_aug: list[float] = mc_k.astype(float).tolist()
            ions_aug: list[str] = list(ions_k)
            for mc_val, lab in zip(mc_k.astype(float).tolist(), ions_k):
                if not is_molecule(lab):
                    continue
                for ratio in molecule_charge_ratios:
                    try:
                        ratio_f = float(ratio)
                    except Exception:
                        continue
                    if ratio_f <= 0:
                        continue
                    mc_aug.append(float(mc_val) * ratio_f)
                    ions_aug.append(str(lab))
            mc_k = np.asarray(mc_aug, dtype=float)
            ions_k = ions_aug

        if len(mc_k) == 0:
            continue

        # Build features for the kept peaks
        file_features = []
        for target_mc in mc_k:
            neighbors = mc_f[(np.abs(mc_f - target_mc) < neighbor_threshold) & (mc_f != target_mc)]
            neigh_part = [target_mc] + sorted(neighbors.tolist())
            max_neigh = max(max_neigh, len(neigh_part))

            sigs_part = []
            if use_signature:
                sigs_part = get_signature_features(target_mc, mc_f, counts[indexes])

            file_features.append((neigh_part, sigs_part))

        raw_data_per_file.append((file_features, ions_k))

    if not raw_data_per_file:
        return np.array([]), np.array([])

    for features_pairs, labels in raw_data_per_file:
        padded_features = []
        for neigh, sigs in features_pairs:
            f_padded = neigh + [0.0] * (max_neigh - len(neigh)) + sigs
            padded_features.append(f_padded)
        features_all.append(np.array(padded_features))
        ions_all.extend(labels)

    if not features_all:
        return np.array([]), np.array([])

    all_features = np.vstack(features_all)
    all_ions = np.array(ions_all)

    # Balancing / Oversampling for rare species
    unique, counts = np.unique(all_ions, return_counts=True)
    target_count = 100
    if len(unique) > 1:
        new_features = [all_features]
        new_ions = [all_ions]
        for species, count in zip(unique, counts):
            if count < target_count:
                species_mask = (all_ions == species)
                s_feat = all_features[species_mask]
                s_ions = all_ions[species_mask]
                
                num_to_add = target_count - count
                if len(s_feat) > 0:
                    repeats = (num_to_add // len(s_feat)) + 1
                    added_feat = np.tile(s_feat, (repeats, 1))[:num_to_add]
                    added_ions = np.tile(s_ions, repeats)[:num_to_add]
                    
                    new_features.append(added_feat)
                    new_ions.append(added_ions)
        
        all_features = np.vstack(new_features)
        all_ions = np.concatenate(new_ions)

    print(f"Loaded training data with feature vector length: {all_features.shape[1]} (Neighborhood: {max_neigh}, Signature: {10 if use_signature else 0})")
    return all_features, all_ions


def build_empirical_mc_distributions(path, num_files=10000):
    """Builds an empirical mapping of [label]: {mean_mc, std_mc} from synthetic data."""
    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])[:num_files]
    mc_data = {}

    for file in files:
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        if 'ion' not in df.columns:
            continue

        ions = df['ion'].to_numpy()
        ions2 = df['ion2'].to_numpy() if 'ion2' in df.columns else np.array([''] * len(df))
        mc = df['mc'].to_numpy()

        for m, i1, i2 in zip(mc, ions, ions2):
            label = str(i2).strip() if str(i2).strip() else str(i1).strip()
            if not label:
                continue

            if label not in mc_data:
                mc_data[label] = []
            mc_data[label].append(m)

    stats = {}
    for label, vals in mc_data.items():
        if len(vals) > 5:
            stats[label] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'count': len(vals)
            }
    return stats


def build_empirical_mc_samples(path, num_files=10000):
    """
    Builds an empirical mapping of simplified species label -> sorted numpy array of mc samples.

    Used for mc-distance physicality checks:
    a peak at mc is "physical" for a species if it is within a threshold of ANY sample.
    """
    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])[:num_files]
    mc_data: dict[str, list[float]] = {}

    for file in files:
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        if 'ion' not in df.columns or 'mc' not in df.columns:
            continue

        ions = df['ion'].to_numpy()
        ions2 = df['ion2'].to_numpy() if 'ion2' in df.columns else np.array([''] * len(df))
        mc = df['mc'].to_numpy()

        for m, i1, i2 in zip(mc, ions, ions2):
            raw_label = str(i2).strip() if str(i2).strip() else str(i1).strip()
            if not raw_label:
                continue
            label = simplify_label(raw_label)
            if not label or label == 'Unknown':
                continue
            mc_data.setdefault(label, []).append(float(m))

    return {k: np.sort(np.asarray(v, dtype=float)) for k, v in mc_data.items() if len(v) > 0}


def load_ion_training_data_mc_vector(
    path: str = 'peak_detection/IonIdentificationModels/training_data/NewData/Data0001',
    element_list=list(),
    elements_to_get_molecules=list(),
    threshold_c: float = 1e-8,
    num_files: int = 10000,
    mc_round_decimals: int = 3,
    augment_molecule_charge_ratios: bool = False,
    molecule_charge_ratios: tuple[float, ...] = (0.5, 1.0 / 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training samples where each sample corresponds to a (file, species) pair, and the
    features are the sorted unique m/c values observed for that species within the file.

    This is intended for a follow-on classifier that can leverage multi-peak information
    (e.g., multiple charge states / isotopes) for the same species.

    Feature vectors are zero-padded to the maximum number of unique m/c values found in
    any (file, species) sample.
    """
    features_all: list[list[float]] = []
    ions_all: list[str] = []

    if element_list != 'all':
        element_list = [simplify_label(str(e)) for e in element_list]

    if not os.path.exists(path):
        print(f"Warning: Training data path {path} not found.")
        return np.array([]), np.array([])

    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])[:num_files]

    # First pass: collect all per-(file, species) mc lists and determine max vector length
    grouped_samples: list[tuple[list[float], str]] = []
    max_len = 0

    for file in _tqdm(files, desc='Loading and grouping training data (mc-vector)'):
        df = pd.read_csv(os.path.join(path, file), keep_default_na=False)
        if 'ion' not in df.columns or 'mc' not in df.columns or 'counts' not in df.columns:
            continue

        mc = df.get(['mc']).to_numpy().squeeze()
        counts = df.get(['counts']).to_numpy().squeeze()

        indexes = counts > threshold_c
        if not np.any(indexes):
            continue

        mc_f = mc[indexes]
        ions_raw_f = df.get(['ion']).to_numpy().squeeze()[indexes]
        ions2_raw_f = (df.get(['ion2']).to_numpy().squeeze() if 'ion2' in df.columns else np.array([''] * len(df)))[indexes]

        # Apply filtering (mirrors load_ion_training_data() logic) in a single pass
        mc_k_list = []
        ions_k_list = []
        for m_val, i1, i2 in zip(mc_f, ions_raw_f, ions2_raw_f):
            ion_str = str(i2) if (i2 and i2 != "") else str(i1)
            label_simple = simplify_label(ion_str)

            if element_list == 'all':
                keep = True
            elif label_simple in element_list:
                keep = True
            elif elements_to_get_molecules:
                atoms = re.findall(r'[A-Z][a-z]?', ion_str)
                keep = bool(atoms) and all(a in elements_to_get_molecules for a in atoms) and is_molecule(label_simple)
            else:
                keep = False

            if keep:
                mc_k_list.append(m_val)
                ions_k_list.append(label_simple)

        if not mc_k_list:
            continue

        mc_k = np.asarray(mc_k_list, dtype=float)
        ions_k = ions_k_list

        per_species: dict[str, list[float]] = {}
        for m, lab in zip(mc_k, ions_k):
            if not lab or lab == 'Unknown':
                continue
            per_species.setdefault(lab, []).append(float(m))

        if augment_molecule_charge_ratios and molecule_charge_ratios:
            for lab, mcs in list(per_species.items()):
                if not is_molecule(lab):
                    continue
                for mc_val in list(mcs):
                    for ratio in molecule_charge_ratios:
                        try:
                            ratio_f = float(ratio)
                        except Exception:
                            continue
                        if ratio_f <= 0:
                            continue
                        mcs.append(float(mc_val) * ratio_f)
                per_species[lab] = mcs

        for lab, mcs in per_species.items():
            if not mcs:
                continue
            uniq = np.unique(np.round(np.asarray(mcs, dtype=float), mc_round_decimals))
            uniq_sorted = sorted(float(x) for x in uniq.tolist())
            if not uniq_sorted:
                continue
            grouped_samples.append((uniq_sorted, lab))
            if len(uniq_sorted) > max_len:
                max_len = len(uniq_sorted)

    if not grouped_samples or max_len == 0:
        return np.array([]), np.array([])

    # Second pass: pad vectors
    for uniq_sorted, lab in grouped_samples:
        vec = uniq_sorted + [0.0] * (max_len - len(uniq_sorted))
        features_all.append(vec)
        ions_all.append(lab)

    all_features = np.asarray(features_all, dtype=float)
    all_ions = np.asarray(ions_all, dtype=str)
    print(f"Loaded mc-vector training data with feature vector length: {all_features.shape[1]}")
    return all_features, all_ions
