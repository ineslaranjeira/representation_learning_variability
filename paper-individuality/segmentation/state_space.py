"""
FROZEN STATE SPACES: fit a 3.3 clustering once, save it, label any dataset with it.
===================================================================================
3.3_wavelet_clusters_uniform.ipynb refits its PCA + k-means every run and keeps nothing but the labels.
To put a NEW dataset into an existing state space (e.g. the neuromodulator sessions into the stage-balanced
left paw + wheel states) the fitted model itself is needed. This module

    fit(config)                -> the model, by exactly 3.3's steps (same set order, balanced draw, per-session
                                  standardisation, PCA to CUTOFF, k-means SEED / N_INIT, vigor numbering)
    save(model) / load(name)   -> data/state_space_models/<name>.npz + <name>.json (the config, fit sets,
                                  frames per set, inertia, the state table)
    label_dataset(model, ds)   -> most_likely_states_<K>_<mouse><eid>.npy ([states, Bin]) and
                                  centroid_distances_<K>_*.npy, in <dataset root>/<the 3.3 states folder>/,
                                  i.e. where syllable_pipeline / 5_syllable_generation_fast already look;
                                  plus _model.json there, naming the model that wrote them
    verify(model, ds, ...)     -> labels sessions into a scratch folder and compares with files on disk

The shared definitions (sources, features, loading) are copied from 3.3's first cell, unchanged.
"""
import json
import os
import pathlib
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = '/home/ines/repositories/representation_learning_variability/paper-individuality/data/'
MODELS = DATA + 'state_space_models/'

# ------------------------------------------------------------------------------------------------
# 3.3's SHARED DEFINITIONS (identical in 3.2_..._uniform and 3.3_..._uniform)
# ------------------------------------------------------------------------------------------------
DATASETS = {'proficient': '', 'neuromodulators': 'neuromodulators/', 'training': 'training/'}
BANDS = ['0.5', '1.0', '2.0', '4.0', '8.0']
SOURCES = {
    'paw':      (['paw_wavelets/'], 'paw_vel_wavelets_',
                 [f'{p}_{a}{b}' for p in ('l_paw', 'r_paw') for a in 'xy' for b in BANDS], 0.1),
    'left_paw': (['left_paw_wavelets/', 'paw_wavelets/'], 'paw_vel_wavelets_',
                 [f'l_paw_{a}{b}' for a in 'xy' for b in BANDS], 0.1),
    'wheel':    (['wheel_wavelets/'], 'wheel_vel_wavelets_',
                 [f'avg_wheel_vel{b}' for b in BANDS], 0.005),
}
VARS = {'paw': ['paw'], 'left_paw': ['left_paw'], 'wheel': ['wheel'],
        'left_paw_wheel': ['left_paw', 'wheel']}
STAGE_DIR = DATA + 'design_matrices/1_camera_setup/'


def features(var):
    return [c for s in VARS[var] for c in SOURCES[s][2]]


def feature_sources(var):
    return [s for s in VARS[var] for _ in SOURCES[s][2]]


def source_dir(dataset, src):
    root = DATA + DATASETS[dataset]
    for d in SOURCES[src][0]:
        if os.path.isdir(root + d):
            return root + d
    raise FileNotFoundError(f'no {SOURCES[src][0]} under {root}')


def load_frame(dataset, var, tag):
    out = None
    for s in VARS[var]:
        path = source_dir(dataset, s) + SOURCES[s][1] + tag
        df = pd.read_parquet(path, columns=['Bin'] + SOURCES[s][2])
        out = df if out is None else out.merge(df, on='Bin', how='inner')
    return out


def load_session(dataset, var, tag):
    """-> (raw amplitudes of the frames with every feature finite, their Bin, the mask)."""
    df = load_frame(dataset, var, tag)
    X = df[features(var)].to_numpy(float)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], df['Bin'].to_numpy()[valid], valid


def stage_eids(stage):
    return {f.split('_')[2] for f in os.listdir(STAGE_DIR + stage) if 'design_matrix' in f}


def subsample_dir(dataset, var):
    return DATA + DATASETS[dataset] + f'{var}_subsampled_wavelets_uniform/'


def states_dir(dataset, var, log, norm, fit_name):
    """3.3's states folder name, under the dataset's root."""
    norm_tag = norm if isinstance(norm, str) else '_'.join(f'{x}-{norm[x]}' for x in VARS[var])
    return (DATA + DATASETS[dataset] + f'{var}_states_uniform_{"log" if log else "raw"}_{norm_tag}z'
            + (f'_fit-{fit_name}' if fit_name else '') + '/')


# ------------------------------------------------------------------------------------------------
# CONFIGS -- the knobs of 3.3's second cell
# ------------------------------------------------------------------------------------------------
STAGES = dict(name='left_paw_wheel_fit-stages', var='left_paw_wheel', fit_name='stages',
              fit_sets={'Early': ('training', 'session_1'), 'Late': ('training', 'last_training'),
                        'Pre-rec': ('training', 'biased'), 'Proficient': ('proficient', None)},
              balance='set', log=False, norm='session', k=8, cutoff=0.95, seed=2024, n_init=10)


# ------------------------------------------------------------------------------------------------
# FIT (3.3 cells 3, 5, 9)
# ------------------------------------------------------------------------------------------------
def set_tags(dataset, stage, var):
    d = subsample_dir(dataset, var)
    tags = sorted(t[:-4] for t in os.listdir(d) if t.endswith('.npz'))
    if stage:
        keep = stage_eids(stage)
        tags = [t for t in tags if t[:36] in keep]
    return tags


def fit(cfg, verbose=True):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    var, feats = cfg['var'], features(cfg['var'])
    src = np.array(feature_sources(var))
    norm = cfg['norm']
    col_norm = np.array([norm if isinstance(norm, str) else norm[x] for x in src])
    fit_sets = cfg['fit_sets']
    tags = {n: set_tags(ds, st, var) for n, (ds, st) in fit_sets.items()}
    subs = {}
    for n, (ds, _) in fit_sets.items():
        for t in tags[n]:
            with np.load(subsample_dir(ds, var) + f'{t}.npz') as z:
                subs[(n, t)] = {k: z[k] for k in z.files}
    assert all(list(z['features']) == feats for z in subs.values())
    eps = np.broadcast_to(np.asarray(next(iter(subs.values()))['eps'], float), (len(feats),)).copy()
    amp = (lambda raw: np.log(raw + eps)) if cfg['log'] else (lambda raw: raw)

    rng = np.random.default_rng(cfg['seed'])
    rows = {n: np.concatenate([np.c_[np.full(len(subs[(n, t)]['sub']), i), np.arange(len(subs[(n, t)]['sub']))]
                               for i, t in enumerate(tags[n])]) for n in fit_sets}
    if cfg['balance'] == 'set':
        m = min(len(r) for r in rows.values())
        rows = {n: r[np.sort(rng.choice(len(r), m, replace=False))] for n, r in rows.items()}
    train_rows = [(n, tags[n][i], j) for n, r in rows.items() for i, j in r]
    blocks = pd.DataFrame(train_rows, columns=['set', 'tag', 'row'])
    groups = list(blocks.groupby(['set', 'tag'], sort=False))
    pool = np.vstack([amp(subs[k]['sub'][g['row'].to_numpy()]) for k, g in groups])
    pooled = (pool.mean(0), pool.std(0))
    G = col_norm == 'global'

    def stats_for(k):
        z = subs[k]
        s = (z['mean_log'], z['sd_log']) if cfg['log'] else (z['mean_raw'], z['sd_raw'])
        return np.where(G, pooled[0], s[0]), np.where(G, pooled[1], s[1])
    X = np.vstack([(amp(subs[k]['sub'][g['row'].to_numpy()]) - stats_for(k)[0]) / stats_for(k)[1]
                   for k, g in groups])
    train_set = np.concatenate([np.full(len(g), k[0]) for k, g in groups])

    pca = PCA(n_components=len(feats)).fit(X)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), cfg['cutoff']) + 1)
    if var == 'wheel':
        n_pc = len(feats)
    P = pca.transform(X)[:, :n_pc]
    km = KMeans(n_clusters=cfg['k'], random_state=cfg['seed'], n_init=cfg['n_init']).fit(P)
    power = np.array([X[km.labels_ == c].mean() for c in range(cfg['k'])])
    order = np.argsort(power)                        # state s = cluster order[s]: 0 = stillest
    states = np.argsort(order)[km.labels_]
    table = {'vigor': [X[states == s].mean() for s in range(cfg['k'])]}
    for x in VARS[var] if len(VARS[var]) > 1 else []:
        table[f'{x} power'] = [X[states == s][:, src == x].mean() for s in range(cfg['k'])]
    table['share'] = list(np.bincount(states, minlength=cfg['k']) / len(states))
    model = dict(cfg=cfg, features=feats, eps=eps, col_norm=col_norm, pooled_mean=pooled[0], pooled_sd=pooled[1],
                 pca_mean=pca.mean_, pca_components=pca.components_[:n_pc], n_pc=n_pc,
                 centroids=km.cluster_centers_[order], inertia=float(km.inertia_),
                 frames_per_set={n: int((train_set == n).sum()) for n in fit_sets},
                 sessions_per_set={n: len(tags[n]) for n in fit_sets}, state_table=table,
                 fitted=date.today().isoformat())
    if verbose:
        print(f"{cfg['name']}: {len(X)} training frames {model['frames_per_set']}, {n_pc} PCs, "
              f"inertia {km.inertia_:.5e}")
    return model


# ------------------------------------------------------------------------------------------------
# SAVE / LOAD
# ------------------------------------------------------------------------------------------------
_ARRAYS = ['eps', 'col_norm', 'pooled_mean', 'pooled_sd', 'pca_mean', 'pca_components', 'centroids']


def save(model, overwrite=False):
    os.makedirs(MODELS, exist_ok=True)
    base = MODELS + model['cfg']['name']
    if os.path.exists(base + '.npz') and not overwrite:
        raise FileExistsError(base + '.npz exists; overwrite=True to replace a frozen model')
    np.savez(base + '.npz', **{k: np.asarray(model[k]) for k in _ARRAYS})
    meta = {k: v for k, v in model.items() if k not in _ARRAYS}
    with open(base + '.json', 'w') as f:
        json.dump(meta, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return base


def load(name):
    base = MODELS + name
    with open(base + '.json') as f:
        model = json.load(f)
    model['cfg']['fit_sets'] = {k: tuple(v) for k, v in model['cfg']['fit_sets'].items()}
    with np.load(base + '.npz') as z:
        model.update({k: z[k] for k in z.files})
    return model


# ------------------------------------------------------------------------------------------------
# LABEL (3.3 cell 11)
# ------------------------------------------------------------------------------------------------
def standardise(model, raw):
    """Each session by its own full-session stats (NORM 'session'), or the fit's pooled ones ('global')."""
    x = np.log(raw + model['eps']) if model['cfg']['log'] else raw
    G = np.asarray(model['col_norm']) == 'global'
    mean = np.where(G, model['pooled_mean'], x.mean(0))
    sd = np.where(G, model['pooled_sd'], x.std(0))
    return (x - mean) / sd


def label_session(model, dataset, tag):
    """-> (states, centroid distances, Bin) for every valid frame of one session."""
    from scipy.spatial.distance import cdist
    raw, bins, _ = load_session(dataset, model['cfg']['var'], tag)
    P = (standardise(model, raw) - model['pca_mean']) @ model['pca_components'].T
    D = cdist(P, model['centroids'])
    return D.argmin(1), D, bins


def dataset_tags(dataset, var, stage=None):
    """Sessions with a 3.2 subsample (the set 3.3 labels), optionally one training stage."""
    return set_tags(dataset, stage, var)


def label_dataset(model, dataset, stage=None, out_dir=None, overwrite=False, save_distances=True, tags=None,
                  verbose=True):
    """Label every session of a dataset; writes into 3.3's states folder for this model unless out_dir."""
    cfg, k = model['cfg'], model['cfg']['k']
    out = out_dir or states_dir(dataset, cfg['var'], cfg['log'], cfg['norm'], cfg['fit_name'])
    os.makedirs(out, exist_ok=True)
    stamp = os.path.join(out, '_model.json')
    if os.path.exists(stamp):
        with open(stamp) as f:
            prev = json.load(f)
        assert prev['model'] == cfg['name'] and prev['fitted'] == model['fitted'], \
            f'{out} was labelled by another model: {prev}'
    tags = tags if tags is not None else dataset_tags(dataset, cfg['var'], stage)
    occ, done = {}, 0
    quality = []
    for t in tags:
        session, mouse = t[:36], t[37:]
        f = os.path.join(out, f'most_likely_states_{k}_{mouse}{session}.npy')
        if os.path.exists(f) and not overwrite:
            continue
        states, D, bins = label_session(model, dataset, t)
        np.save(f, np.array([states, bins]))
        if save_distances:
            np.save(os.path.join(out, f'centroid_distances_{k}_{mouse}{session}.npy'), D)
        occ[t] = np.bincount(states, minlength=k) / len(states)
        quality.append(fit_quality(states, D, k, t))
        done += 1
    if quality:
        qf = os.path.join(out, '_fit_quality.csv')
        q = pd.DataFrame(quality).set_index('session')
        if os.path.exists(qf):
            old = pd.read_csv(qf, index_col='session')
            q = pd.concat([old[~old.index.isin(q.index)], q])
        q.to_csv(qf)
    palette = os.path.join(states_dir(*_fit_dataset(cfg)), 'state_palette.csv')
    if os.path.exists(palette) and not os.path.exists(os.path.join(out, 'state_palette.csv')):
        import shutil
        shutil.copy(palette, out)
    with open(stamp, 'w') as fh:
        json.dump({'model': cfg['name'], 'fitted': model['fitted'], 'file': MODELS + cfg['name'] + '.npz',
                   'dataset': dataset, 'stage': stage}, fh, indent=1)
    if verbose:
        print(f'{dataset}{"/" + stage if stage else ""}: labelled {done} of {len(tags)} sessions -> {out}')
    return pd.DataFrame(occ, index=[f'state{s}' for s in range(k)]).T


def fit_quality(states, D, k, tag):
    """Per session: how far its frames sit from their nearest centroid (quantiles), and its occupancy."""
    d = D[np.arange(len(states)), states]
    q = np.percentile(d, [10, 50, 90, 99])
    return dict(session=tag, frames=len(states), d10=q[0], d50=q[1], d90=q[2], d99=q[3],
                **{f'state{s}': c for s, c in enumerate(np.bincount(states, minlength=k) / len(states))})


def _fit_dataset(cfg):
    """(dataset, var, log, norm, fit_name) of the first fit set: where 3.3 wrote the model's palette."""
    ds = next(iter(cfg['fit_sets'].values()))[0]
    ds = 'proficient' if 'proficient' in [v[0] for v in cfg['fit_sets'].values()] else ds
    return ds, cfg['var'], cfg['log'], cfg['norm'], cfg['fit_name']


def reference_quality(model, dataset, stage=None):
    """fit_quality from the centroid_distances files 3.3 already wrote for a dataset of this model."""
    cfg, k = model['cfg'], model['cfg']['k']
    d = states_dir(dataset, cfg['var'], cfg['log'], cfg['norm'], cfg['fit_name'])
    rows = []
    for t in dataset_tags(dataset, cfg['var'], stage):
        f = os.path.join(d, f'centroid_distances_{k}_{t[37:]}{t[:36]}.npy')
        if os.path.exists(f):
            D = np.load(f)
            rows.append(fit_quality(D.argmin(1), D, k, t))
    return pd.DataFrame(rows).set_index('session')


def verify(model, dataset, stage=None, n=10, seed=0, scratch='/tmp/state_space_verify/'):
    """Label n random sessions into `scratch` and compare with the files of this model's states folder:
    per session, the fraction of frames whose state differs and whether the Bin grids match."""
    cfg, k = model['cfg'], model['cfg']['k']
    ref = states_dir(dataset, cfg['var'], cfg['log'], cfg['norm'], cfg['fit_name'])
    tags = [t for t in dataset_tags(dataset, cfg['var'], stage)
            if os.path.exists(os.path.join(ref, f'most_likely_states_{k}_{t[37:]}{t[:36]}.npy'))]
    pick = sorted(np.random.default_rng(seed).choice(tags, min(n, len(tags)), replace=False))
    rows = []
    for t in pick:
        states, _, bins = label_session(model, dataset, t)
        old = np.load(os.path.join(ref, f'most_likely_states_{k}_{t[37:]}{t[:36]}.npy'))
        same_grid = len(old[1]) == len(bins) and np.allclose(old[1], bins)
        rows.append(dict(session=t, frames=len(states), same_bins=same_grid,
                         frac_changed=float(np.mean(old[0] != states)) if same_grid else np.nan))
    return pd.DataFrame(rows)
