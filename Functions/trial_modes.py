"""
TRIAL MODES: cluster whole trials by their syllable sequences
=============================================================
Function version of paper-individuality/3_trial_modes/clustering_trial_syllables_watershed.ipynb, so the
same clustering runs on any syllable file (proficient, neuromodulators, ...).

    syllables (one row per trial x epoch, 10-bin binned_sequence)
      -> trial_matrix   one row per trial, the 4 epochs x 10 bins side by side (40 codes)
      -> encode         one-hot per bin ('factored': paw / whisking / licking, or 'onehot': all 32 codes)
      -> embed          PCA -> 2D UMAP
      -> density_map    gaussian KDE of the embedding on a grid
      -> watershed_map  watershed on the KDE: one basin = one trial mode
      -> assign         each trial gets the basin it falls in (0 = background -> NaN)

USE IT (from any folder in the repo):

    import sys, pathlib
    _p = pathlib.Path.cwd().resolve()
    while not (_p / 'Functions' / 'trial_modes.py').exists() and _p != _p.parent:
        _p = _p.parent
    sys.path.insert(0, str(_p))
    from Functions import trial_modes as tm

    trials, res = tm.cluster_trials(syllables)      # syllables = pd.read_parquet(<syllable file>)
    tm.plot_watershed(res); tm.plot_clusters(trials, res)

Syllable codes are mixed radix with the paw state fastest: code = paw + n_paw * whisking + 2 * n_paw * licking
(syllable_pipeline.make_states_file), so 0-31 for 8 paw states.

Differences from the notebook (all on purpose):
- trials come out in chronological order (numeric trial_id); the notebook's pivot sorted trial_id as a string,
  which cat-HMM.ipynb had to undo
- the KDE grid bounds come from the data (percentiles) unless given; the notebook hard-coded them for one embedding
- the KDE can be fitted on a subsample (kde_max_points) -- it is a smooth density, and the full fit on ~400k
  trials is slow. None = all trials, as the notebook did
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy import ndimage as ndi
from scipy import stats
from sklearn.decomposition import PCA
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries

EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']


def trial_matrix(syllables, epochs=EPOCHS):
    """One row per trial with complete epochs; returns (trials, seqs).

    trials: sample, trial_type, mouse_name, session, trial_id + one column per epoch (its 10-bin array)
    seqs:   n_trials x (n_epochs * n_bins) array of syllable codes, rows aligned with trials
    """
    df = syllables.copy()
    df['session'] = df['sample'].str.split(' ').str[0]
    df['trial_id'] = df['sample'].str.split(' ').str[1].astype(float)
    trials = (df.pivot(index=['sample', 'trial_type', 'mouse_name', 'session', 'trial_id'],
                       columns='broader_label', values='binned_sequence')
              .reset_index().dropna(subset=epochs)
              .sort_values(['session', 'trial_id']).reset_index(drop=True))
    trials.columns.name = None
    seqs = np.vstack([np.hstack(row) for row in trials[epochs].to_numpy()]).astype(float)
    return trials, seqs


def encode(seqs, mode='factored', n_paw=8, ref_paw=1):
    """Binarize the syllable codes per bin; rows with any NaN bin come back all-NaN.

    'factored' (the notebook's "alternative 9 dimensional encoding"): per bin, paw state one-hot (minus the
               reference state ref_paw, to avoid collinearity) + whisking (0/1) + licking (0/1)
    'onehot':  per bin, one-hot over all 2 * 2 * n_paw syllable codes
    """
    n_trials, n_steps = seqs.shape
    nan_rows = np.isnan(seqs).any(axis=1)
    codes = np.nan_to_num(seqs).astype(int)
    if mode == 'onehot':
        n_codes = 4 * n_paw
        X = np.zeros((n_trials, n_steps, n_codes), dtype=np.float32)
        np.put_along_axis(X, codes[..., None], 1, axis=2)
    elif mode == 'factored':
        X = np.zeros((n_trials, n_steps, n_paw + 2), dtype=np.float32)
        np.put_along_axis(X, (codes % n_paw)[..., None], 1, axis=2)
        X[..., n_paw] = (codes // n_paw) % 2
        X[..., n_paw + 1] = codes // (2 * n_paw)
        X = np.delete(X, ref_paw, axis=2)
    else:
        raise ValueError(f'Unknown encoding {mode!r}')
    X = X.reshape(n_trials, -1)
    X[nan_rows] = np.nan
    return X


def embed(X, n_pca=None, random_state=42, **umap_kwargs):
    """PCA (n_pca components; None = all, as the notebook, which is a rotation only) -> 2D UMAP."""
    import umap
    pca = PCA(n_pca)
    X_pca = pca.fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=random_state, **umap_kwargs)
    embedding = reducer.fit_transform(X_pca)
    return embedding, pca, reducer


def density_map(embedding, bounds=None, res=150, kde_max_points=50_000, pad_pct=0.5, seed=0):
    """Gaussian KDE of the embedding on a res x res grid. bounds = (xmin, xmax, ymin, ymax)."""
    if bounds is None:
        lo = np.percentile(embedding, pad_pct, axis=0)
        hi = np.percentile(embedding, 100 - pad_pct, axis=0)
        span = hi - lo
        bounds = (lo[0] - .05 * span[0], hi[0] + .05 * span[0], lo[1] - .05 * span[1], hi[1] + .05 * span[1])
    xmin, xmax, ymin, ymax = bounds
    fit_points = embedding
    if kde_max_points is not None and len(embedding) > kde_max_points:
        idx = np.random.default_rng(seed).choice(len(embedding), kde_max_points, replace=False)
        fit_points = embedding[idx]
    kernel = stats.gaussian_kde(fit_points.T)
    gx, gy = np.mgrid[xmin:xmax:complex(res), ymin:ymax:complex(res)]
    Z = kernel(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    return Z, bounds


def watershed_map(Z, threshold_pct=60, min_distance=10, footprint_radius=2):
    """Watershed basins of the KDE above its threshold_pct percentile; 0 = background."""
    mask = ndi.binary_fill_holes(Z > np.percentile(Z, threshold_pct))
    coords = peak_local_max(Z, footprint=disk(footprint_radius), min_distance=min_distance, labels=mask)
    markers = np.zeros(Z.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return watershed(-Z, markers, mask=mask)


def assign(embedding, labels, bounds):
    """Basin of each point (grid lookup); points on the background or outside the grid get NaN."""
    xmin, xmax, ymin, ymax = bounds
    nx, ny = labels.shape
    px = ((embedding[:, 0] - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
    py = ((embedding[:, 1] - ymin) / (ymax - ymin) * (ny - 1)).astype(int)
    outside = (px < 0) | (px >= nx) | (py < 0) | (py >= ny)
    out = labels[np.clip(px, 0, nx - 1), np.clip(py, 0, ny - 1)].astype(float)
    out[outside | (out == 0)] = np.nan
    return out


def cluster_trials(syllables, epochs=EPOCHS, encoding='factored', n_paw=8, ref_paw=1, n_pca=None,
                   random_state=42, umap_kwargs=None, bounds=None, res=150, kde_max_points=50_000,
                   threshold_pct=60, min_distance=10, footprint_radius=2):
    """The whole pipeline. Returns (trials, res):
    trials: one row per complete trial + trial_cluster, embedding_x, embedding_y
    res:    dict with the fitted pca / reducer (reducer.transform projects new trials), Z, labels, bounds
    """
    trials, seqs = trial_matrix(syllables, epochs)
    X = encode(seqs, encoding, n_paw, ref_paw)
    valid = ~np.isnan(X).any(axis=1)
    embedding, pca, reducer = embed(X[valid], n_pca, random_state, **(umap_kwargs or {}))
    Z, bounds = density_map(embedding, bounds, res, kde_max_points)
    labels = watershed_map(Z, threshold_pct, min_distance, footprint_radius)
    trials['embedding_x'] = np.nan
    trials['embedding_y'] = np.nan
    trials['trial_cluster'] = np.nan
    trials.loc[valid, 'embedding_x'] = embedding[:, 0]
    trials.loc[valid, 'embedding_y'] = embedding[:, 1]
    trials.loc[valid, 'trial_cluster'] = assign(embedding, labels, bounds)
    res = dict(pca=pca, reducer=reducer, embedding=embedding, Z=Z, labels=labels, bounds=bounds,
               params=dict(epochs=epochs, encoding=encoding, n_paw=n_paw, ref_paw=ref_paw, n_pca=n_pca,
                           random_state=random_state, umap_kwargs=umap_kwargs, res=res,
                           kde_max_points=kde_max_points, threshold_pct=threshold_pct,
                           min_distance=min_distance, footprint_radius=footprint_radius))
    return trials, res


# ---------------------------------------------------------------- plots

def plot_watershed(res):
    """KDE, mask, basins and boundaries side by side (the notebook's 4-panel check)."""
    Z, labels = res['Z'], res['labels']
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    ax[0].imshow(np.rot90(Z)); ax[0].set_title('KDE')
    ax[1].imshow(np.rot90(labels > 0)); ax[1].set_title('mask')
    ax[2].imshow(np.rot90(labels), cmap='tab20', interpolation='none'); ax[2].set_title('watershed')
    ax[3].imshow(np.rot90(Z)); ax[3].imshow(np.rot90(find_boundaries(labels, mode='outer')), cmap='Reds', alpha=.2)
    ax[3].set_title('boundaries')
    for a in ax:
        a.axis('off')
    return fig


def plot_clusters(trials, res, color_points=True, max_points=100_000, seed=0):
    """Embedding with the watershed boundaries and the cluster numbers at their centroids."""
    xmin, xmax, ymin, ymax = res['bounds']
    d = trials.dropna(subset=['embedding_x'])
    if len(d) > max_points:
        d = d.sample(max_points, random_state=seed)
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [cmap(int(k) % cmap.N) if k == k else (.6, .6, .6, 1) for k in d['trial_cluster']] \
        if color_points else 'k'
    ax.scatter(d['embedding_x'], d['embedding_y'], c=colors, s=1, alpha=.1 if color_points else .02,
               rasterized=True)
    ax.imshow(np.rot90(find_boundaries(res['labels'], mode='outer')), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax], aspect='auto', alpha=.6)
    for k, g in d.dropna(subset=['trial_cluster']).groupby('trial_cluster'):
        ax.text(g['embedding_x'].mean(), g['embedding_y'].mean(), str(int(k)), fontsize=11, fontweight='bold',
                ha='center', va='center', color='white',
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.axis('off')
    return fig


def mode_fingerprint(trials, cluster, epochs=EPOCHS, n_paw=8):
    """Mean paw-state occupancy and whisking/licking probability per bin for one trial mode:
    returns (paw: n_paw x n_steps, whisk: n_steps, lick: n_steps)."""
    _, seqs = trial_matrix_from_trials(trials[trials['trial_cluster'] == cluster], epochs)
    paw = np.stack([(seqs % n_paw == p).mean(axis=0) for p in range(n_paw)])
    return paw, ((seqs // n_paw) % 2).mean(axis=0), (seqs // (2 * n_paw)).mean(axis=0)


def trial_matrix_from_trials(trials, epochs=EPOCHS):
    """seqs from an already pivoted trials table (as returned by trial_matrix / cluster_trials)."""
    return trials, np.vstack([np.hstack(row) for row in trials[epochs].to_numpy()]).astype(float)
