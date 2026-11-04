"""
Wavelet clusters -- the code shared by 3.2_wavelet_subsample_uniform.ipynb,
3.3_wavelet_clusters_uniform.ipynb and paw_k_sweep.py.

Part 1 is the notebooks' former "SHARED DEFINITIONS" cell, verbatim: data layout, feature
sources, loading one session.

Part 2 is 3.3's computation, as functions of its knobs instead of notebook globals:
    load_subsamples  -> the 3.2 subsamples of every set
    training_frames  -> the standardised training frames + the Standardiser that labelling reuses
    fit_states       -> PCA to CUTOFF variance, k-means (best of N_INIT), states numbered by vigor
    label_session    -> every valid frame of one session -> nearest centroid
    save_states      -> most_likely_states_<k>_<mouse><eid>.npy (and centroid distances)
The notebook reads exactly as before (it calls these and keeps its plots); see 3.3's header for
what each choice means and why.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ================================================================================================
# PART 1 -- SHARED DEFINITIONS (formerly identical cells in 3.2_..._uniform and 3.3_..._uniform)
# ================================================================================================
DATA = '/home/ines/repositories/representation_learning_variability/paper-individuality/data/'

# each dataset keeps its wavelet folders under its own root
DATASETS = {'proficient': '', 'neuromodulators': 'neuromodulators/', 'training': 'training/'}

BANDS = ['0.5', '1.0', '2.0', '4.0', '8.0']     # the bands the clustering uses
# source -> (wavelet folders to try in order, file prefix, columns, log floor)
#   log floor: log(x + floor), only used with LOG = True. Paw amplitudes have a median of ~1-6
#   and a 1st percentile of ~0.01-0.07, so 0.1 compresses a still paw's tracking jitter. Wheel
#   amplitudes are ~20x smaller (still ~1e-4, moving ~0.05-0.3), so its floor is 0.005.
#   16 / 32 Hz are left out everywhere: for the wheel they are encoder noise.
SOURCES = {
    'paw':      (['paw_wavelets/'], 'paw_vel_wavelets_',
                 [f'{p}_{a}{b}' for p in ('l_paw', 'r_paw') for a in 'xy' for b in BANDS], 0.1),
    # the proficient dataset keeps the left paw inside the both-paw files
    'left_paw': (['left_paw_wavelets/', 'paw_wavelets/'], 'paw_vel_wavelets_',
                 [f'l_paw_{a}{b}' for a in 'xy' for b in BANDS], 0.1),
    'wheel':    (['wheel_wavelets/'], 'wheel_vel_wavelets_',
                 [f'avg_wheel_vel{b}' for b in BANDS], 0.005),
}
# what can be clustered: one source, or several merged on Bin
VARS = {'paw': ['paw'], 'left_paw': ['left_paw'], 'wheel': ['wheel'],
        'left_paw_wheel': ['left_paw', 'wheel']}


def features(var):
    return [c for s in VARS[var] for c in SOURCES[s][2]]


def feature_sources(var):
    """The source of every feature column, in features() order."""
    return [s for s in VARS[var] for _ in SOURCES[s][2]]


def feature_eps(var):
    return np.array([SOURCES[s][3] for s in feature_sources(var)])


def source_dir(dataset, src):
    root = DATA + DATASETS[dataset]
    for d in SOURCES[src][0]:
        if os.path.isdir(root + d):
            return root + d
    raise FileNotFoundError(f'no {SOURCES[src][0]} under {root}')


def session_tags(dataset, var):
    """'<eid>_<mouse>' of every session that has a file for EVERY source of `var`."""
    have = []
    for s in VARS[var]:
        d, prefix = source_dir(dataset, s), SOURCES[s][1]
        have.append({f[len(prefix):] for f in os.listdir(d) if f.startswith(prefix)})
    return sorted(set.intersection(*have))


def load_frame(dataset, var, tag, extra=()):
    """One session, its sources merged on Bin -> Bin, the features, and whichever `extra`
    columns (raw traces for plotting) the files have."""
    out = None
    for s in VARS[var]:
        path = source_dir(dataset, s) + SOURCES[s][1] + tag
        names = set(pq.read_schema(path).names)
        taken = set() if out is None else set(out.columns)
        cols = ['Bin'] + SOURCES[s][2] + [c for c in extra if c in names and c not in taken]
        df = pd.read_parquet(path, columns=cols)
        out = df if out is None else out.merge(df, on='Bin', how='inner')
    return out


def load_session(dataset, var, tag):
    """-> (raw amplitudes of the frames with every feature finite, their Bin, the mask)."""
    df = load_frame(dataset, var, tag)
    X = df[features(var)].to_numpy(float)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], df['Bin'].to_numpy()[valid], valid


# TRAINING STAGES: each is a subfolder of design_matrices/1_camera_setup/ listing its sessions
STAGE_DIR = DATA + 'design_matrices/1_camera_setup/'


def stage_eids(stage):
    """The session eids of one training stage, e.g. 'session_1', 'last_training', 'biased'."""
    return {f.split('_')[2] for f in os.listdir(STAGE_DIR + stage) if 'design_matrix' in f}


def dataset_root(dataset, out_root=None):
    """Where a dataset's subsamples and states live: its own folder under DATA, or the same
    layout under `out_root` (test runs)."""
    return (out_root or DATA) + DATASETS[dataset]


# ================================================================================================
# PART 2 -- 3.3's computation
# ================================================================================================
def column_norm(var, norm):
    """NORM per feature column: one value for all, or a dict per source."""
    col = np.array([norm if isinstance(norm, str) else norm[x] for x in feature_sources(var)])
    assert set(col) <= {'session', 'global'}, col
    return col


def states_dir_name(var, log, norm, fit_name=None):
    """The states folder 3.3 writes, e.g. 'paw_states_uniform_raw_sessionz/'."""
    tag = norm if isinstance(norm, str) else '_'.join(f'{x}-{norm[x]}' for x in VARS[var])
    return f'{var}_states_uniform_{"log" if log else "raw"}_{tag}z' + (f'_fit-{fit_name}' if fit_name else '') + '/'


def set_tags(var, dataset, stage=None, out_root=None, exclude=(), max_sessions=None):
    """The sessions of one set that 3.2 subsampled (stage filter, exclusions, max_sessions)."""
    d = dataset_root(dataset, out_root) + f'{var}_subsampled_wavelets_uniform/'
    tags = sorted(t[:-4] for t in os.listdir(d) if t.endswith('.npz'))
    if stage:
        keep = stage_eids(stage)
        tags = [t for t in tags if t[:36] in keep]
    tags = [t for t in tags if t[:36] not in set(exclude)]
    return tags[:max_sessions]


def load_subsamples(var, sets, out_root=None, exclude=(), max_sessions=None):
    """-> (SET_TAGS {set: [tag]}, subs {(set, tag): arrays of 3.2's .npz}, EPS per feature)."""
    tags = {n: set_tags(var, ds, st, out_root, exclude, max_sessions) for n, (ds, st) in sets.items()}
    subs = {}
    for n, (ds, _) in sets.items():
        for t in tags[n]:
            # read every file ONCE into plain arrays (an NpzFile re-reads the whole array on every ['key'])
            with np.load(dataset_root(ds, out_root) + f'{var}_subsampled_wavelets_uniform/{t}.npz') as z:
                subs[(n, t)] = {k: z[k] for k in z.files}
    assert all(list(z['features']) == features(var) for z in subs.values())
    # the log floor per feature (older files store one number for all)
    eps = np.broadcast_to(np.asarray(next(iter(subs.values()))['eps'], float), (len(features(var)),)).copy()
    assert all(np.allclose(np.broadcast_to(z['eps'], eps.shape), eps) for z in subs.values())
    return tags, subs, eps


class Standardiser:
    """amp -> standardise, the same function for training and labelled frames.
    Per feature, the session's own stats where NORM is 'session', the pooled training ones where
    it is 'global'."""

    def __init__(self, log, eps, col_norm, pooled):
        self.log, self.eps, self.pooled = log, eps, pooled
        self.glob = np.asarray(col_norm) == 'global'

    def amp(self, raw):
        """The amplitude scale the clustering works in."""
        return np.log(raw + self.eps) if self.log else raw

    def session_stats(self, raw):
        """Full-session mean and SD in the working scale (ddof=0, as scipy.stats.zscore)."""
        x = self.amp(raw)
        return x.mean(0), x.std(0)

    def mix(self, sess):
        return np.where(self.glob, self.pooled[0], sess[0]), np.where(self.glob, self.pooled[1], sess[1])

    def saved_stats(self, z):
        """The stats 3.2 saved for a session (computed over ALL its frames), mixed per NORM."""
        return self.mix((z['mean_log'], z['sd_log']) if self.log else (z['mean_raw'], z['sd_raw']))

    def full_stats(self, raw):
        """The standardisation for a session's frames: its own full-session stats, mixed per NORM."""
        return self.mix(self.session_stats(raw))

    def standardise(self, raw, stats):
        mean, sd = stats
        return (self.amp(raw) - mean) / sd


def training_frames(var, fit_sets, tags, subs, eps, log, norm, balance, seed):
    """-> SimpleNamespace(X_train, TRAIN_SET, TRAIN_ROWS, std). The training frames are every fit
    set's subsamples; with balance = 'set', the same number of frames from each set (a random
    draw, without replacement, from the larger ones)."""
    rng = np.random.default_rng(seed)
    rows = {n: np.concatenate([np.c_[np.full(len(subs[(n, t)]['sub']), i), np.arange(len(subs[(n, t)]['sub']))]
                               for i, t in enumerate(tags[n])]) for n in fit_sets}
    if balance == 'set':
        m = min(len(r) for r in rows.values())
        rows = {n: r[np.sort(rng.choice(len(r), m, replace=False))] for n, r in rows.items()}
    elif balance != 'session':
        raise ValueError(f"BALANCE must be 'session' or 'set', got {balance!r}")
    train_rows = [(n, tags[n][i], j) for n, r in rows.items() for i, j in r]   # (set, tag, row)

    # 'global': ONE mean/SD from the training frames (every session, and with balance='set' every
    # set, weighing as in the fit)
    blocks = pd.DataFrame(train_rows, columns=['set', 'tag', 'row'])
    groups = list(blocks.groupby(['set', 'tag'], sort=False))
    plain = Standardiser(log, eps, column_norm(var, norm), None)
    pool = np.vstack([plain.amp(subs[k]['sub'][g['row'].to_numpy()]) for k, g in groups])
    std = Standardiser(log, eps, column_norm(var, norm), (pool.mean(0), pool.std(0)))
    # standardise per session (vectorised: one block per session, rows in train_rows order)
    X = np.vstack([std.standardise(subs[k]['sub'][g['row'].to_numpy()], std.saved_stats(subs[k])) for k, g in groups])
    train_set = np.concatenate([np.full(len(g), k[0]) for k, g in groups])
    return SimpleNamespace(X_train=X, TRAIN_SET=train_set, TRAIN_ROWS=train_rows, std=std)


def fit_pca(X_train, cutoff, var):
    """-> (pca, N_PC, cumulative variance): PCA keeping `cutoff` of the variance (all PCs for the wheel)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=X_train.shape[1]).fit(X_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pc = X_train.shape[1] if var == 'wheel' else int(np.searchsorted(cumvar, cutoff) + 1)
    return pca, n_pc, cumvar


def fit_states(P_train, X_train, k, seed, n_init):
    """k-means on the retained PCs, best of n_init; states renumbered by vigor (mean standardised
    power), 0 = stillest. -> SimpleNamespace(km, ORDER, CENTROIDS, train_states)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(P_train)
    power = np.array([X_train[km.labels_ == c].mean() for c in range(k)])
    order = np.argsort(power)                 # new state s = old cluster ORDER[s]
    return SimpleNamespace(km=km, ORDER=order, CENTROIDS=km.cluster_centers_[order],
                           train_states=np.argsort(order)[km.labels_])


def label_session(dataset, var, tag, std, pca, n_pc, centroids, saved=None):
    """Every valid frame of one session -> (states, centroid distances, Bin, valid mask, raw).
    The same standardise() the training frames went through, with stats recomputed from the full
    session and, if `saved` (its 3.2 .npz arrays) is given, checked against the ones 3.2 saved."""
    from scipy.spatial.distance import cdist
    raw, bins, valid = load_session(dataset, var, tag)
    stats = std.full_stats(raw)
    if saved is not None:
        ref = std.saved_stats(saved)
        assert np.allclose(stats[0], ref[0]) and np.allclose(stats[1], ref[1])
    P = pca.transform(std.standardise(raw, stats))[:, :n_pc]
    D = cdist(P, centroids)
    return D.argmin(1), D, bins, valid, raw


def save_states(out_dir, k, tag, states, bins, D=None):
    """Same file names and layout as the original, so 4.0 / syllable_pipeline can point here."""
    session, mouse = tag[:36], tag[37:]
    np.save(out_dir + f'most_likely_states_{k}_{mouse}{session}', np.array([states, bins]))
    if D is not None:
        np.save(out_dir + f'centroid_distances_{k}_{mouse}{session}', D)
