"""
TRIAL-MODE ROBUSTNESS -- the trial modes refit under other settings, for the mode-entropy vs LD1 supplement
==========================================================================================================
The paper's trial modes (trial_modes_8_k_10_bin_syllables_02-10-2026) are Functions/trial_modes.cluster_trials
with UMAP's defaults (n_neighbors 15, min_dist 0.1, random_state 42), a KDE on a 150 x 150 grid and a watershed
(threshold_pct 60, min_distance 10). This refits the partition on the SAME trials and the same encoding
('factored', ref_paw 1) under:

    umap     UMAP n_neighbors x min_dist grid at seed 42, and other seeds at the defaults -> one 2-D embedding each
             (the KDE + watershed, and its parameter grid, are cheap and run in the supplement notebook)
    kmeans   no UMAP at all: k-means on the encoded trials (360 binary features), several k

Inputs: the paper fit's trials.pqt (its epoch columns are the syllable codes in uniform paw numbering, as used
for that fit). Outputs in data/mode_robustness/:
    valid.npy                      rows of trials.pqt with complete trials (the rows every output refers to)
    umap_nn<n>_md<d>_s<seed>.npy   embedding (n_valid x 2, float32)
    kmeans_k<k>.npy                labels (n_valid,)

    python mode_entropy_robustness.py kmeans          # all k
    python mode_entropy_robustness.py umap            # the whole grid, skipping what exists
    python mode_entropy_robustness.py umap 15 0.1 42  # one embedding (n_neighbors, min_dist, seed)
"""
import sys
import time
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
sys.path.insert(0, str(PI.parent))
from Functions import trial_modes as tm

FIT_DIR = HERE / 'trial_modes_8_k_10_bin_syllables_02-10-2026'
OUT = PI / 'data' / 'mode_robustness'
ENCODING, REF_PAW = 'factored', 1                                  # as the paper fit (trial_modes_proficient PARAMS)
UMAP_GRID = [(nn, md, 42) for nn in (5, 15, 50) for md in (0.0, 0.1, 0.5)] + [(15, 0.1, s) for s in (0, 1)]
KMEANS_K = [4, 6, 8, 10, 12, 15, 17, 20, 25, 30, 40]


def encoded():
    trials = pd.read_parquet(FIT_DIR / 'trials.pqt')
    _, seqs = tm.trial_matrix_from_trials(trials)
    X = tm.encode(seqs, ENCODING, 8, REF_PAW)
    valid = ~np.isnan(X).any(axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'valid.npy', valid)
    return X[valid]


def run_umap(X, nn, md, seed):
    path = OUT / f'umap_nn{nn}_md{md}_s{seed}.npy'
    if path.exists():
        print(f'{path.name} exists', flush=True)
        return
    t0 = time.time()
    emb, _, _ = tm.embed(X, None, seed, n_neighbors=nn, min_dist=md)
    np.save(path, emb.astype(np.float32))
    print(f'{path.name}: {time.time() - t0:.0f} s', flush=True)


def run_kmeans(X, k):
    from sklearn.cluster import KMeans
    path = OUT / f'kmeans_k{k}.npy'
    if path.exists():
        print(f'{path.name} exists', flush=True)
        return
    t0 = time.time()
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
    np.save(path, lab.astype(np.int16))
    print(f'{path.name}: {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    X = encoded()
    print(f'{len(X)} complete trials x {X.shape[1]} features', flush=True)
    if sys.argv[1] == 'kmeans':
        for k in ([int(a) for a in sys.argv[2:]] or KMEANS_K):
            run_kmeans(X, k)
    elif sys.argv[1] == 'umap':
        grid = [(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))] if len(sys.argv) > 2 else UMAP_GRID
        for nn, md, seed in grid:
            run_umap(X, nn, md, seed)
