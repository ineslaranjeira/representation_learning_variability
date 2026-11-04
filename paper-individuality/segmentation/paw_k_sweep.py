"""
PAW-STATE SWEEP -- the paper's paw clustering at other numbers of states
=======================================================================
The production paw states are 3.3_wavelet_clusters_uniform's proficient fit (VAR = 'paw',
LOG = False, NORM = 'session', K = 8, n_init = 10). This re-runs exactly that fit at every K in KS
(same training frames, same PCA -- only the k-means differs), labels all sessions, and pushes each
K through syllable_pipeline (states files, then 10-bin syllables), so 4_mice/syllable_sweeps.py
can score mouse identification per K.

    K = 8 is not refitted: wavelet_clusters reproduces its states exactly (all 332 sessions), and
    the production syllables (8_k_10_bin_syllables_02-10-2026) are used for it.

Outputs, per K != 8 (vigor-numbered states, 0 = stillest; no relabelling):
    data/paw_states_uniform_raw_sessionz/most_likely_states_<K>_<mouse><eid>.npy   (no centroid distances)
    data/states_files_both_paws_k<K>_own_lick-standard/                            (syllable_pipeline)
    data/both_paws_k<K>_own_lick-standard_proficient_<K>_k_10_bin_syllables_<date>

    python paw_k_sweep.py            # every K, skipping what exists
    python paw_k_sweep.py 4 6        # some K
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
import glob
import warnings

import numpy as np

import wavelet_clusters as wc
import syllable_pipeline as sp

KS = [4, 6, 8, 10, 12, 16]
# 3.3's production configuration
VAR, LOG, NORM, BALANCE, CUTOFF, SEED, N_INIT = 'paw', False, 'session', 'set', 0.95, 2024, 10
SETS = {'proficient': ('proficient', None)}
STATES_DIR = wc.DATA + wc.states_dir_name(VAR, LOG, NORM)
TARGET_LENGTH = 10
N_JOBS = 8


def fit_and_label(ks):
    """One training set and PCA for all K; k-means and labelling per K."""
    tags, subs, eps = wc.load_subsamples(VAR, SETS)
    T = wc.training_frames(VAR, SETS, tags, subs, eps, LOG, NORM, BALANCE, SEED)
    pca, n_pc, _ = wc.fit_pca(T.X_train, CUTOFF, VAR)
    P = pca.transform(T.X_train)[:, :n_pc]
    for k in ks:
        S = wc.fit_states(P, T.X_train, k, SEED, N_INIT)
        print(f'K = {k}: inertia {S.km.inertia_:.5e}, training occupancy '
              f'{np.round(np.bincount(S.train_states, minlength=k) / len(S.train_states), 3)}', flush=True)
        for t in tags['proficient']:
            st, _, bins, _, _ = wc.label_session('proficient', VAR, t, T.std, pca, n_pc, S.CENTROIDS,
                                                 saved=subs[('proficient', t)])
            wc.save_states(STATES_DIR, k, t, st, bins)
        print(f'K = {k}: labelled {len(tags["proficient"])} sessions', flush=True)


def syllables(k):
    cfg = sp.resolve('proficient', 'both_paws', 'own', 'standard', k=k)
    if glob.glob(sp.syllables_glob('proficient', 'both_paws', 'own', 'standard', k=k, target_length=TARGET_LENGTH)):
        print(f'K = {k}: syllables exist, skipped', flush=True)
        return
    sessions = sp.sessions_with_inputs(cfg)
    print(f'K = {k}: states files {sp.run_states(cfg, sessions, n_jobs=N_JOBS)}', flush=True)
    seq = sp.run_syllables(cfg, sessions, TARGET_LENGTH, n_jobs=N_JOBS)
    path = sp.output_name(cfg, 'syllables', TARGET_LENGTH)
    seq.to_parquet(path, compression='gzip')
    print(f'K = {k}: {len(seq)} rows -> {path}', flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    ks = [int(a) for a in sys.argv[1:]] or KS
    ks = [k for k in ks if k != 8]
    have = lambda k: len(glob.glob(STATES_DIR + f'most_likely_states_{k}_*.npy')) >= 332     # noqa: E731
    todo = [k for k in ks if not have(k)]
    if todo:
        fit_and_label(todo)
    for k in ks:
        syllables(k)
