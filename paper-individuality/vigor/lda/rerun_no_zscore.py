"""
THE COUNTERFACTUAL: REFIT THE PAW STATES WITHOUT THE PER-SESSION Z-SCORE
========================================================================
`zscore_cost.py` asks what the discarded scale contains. This asks the question the
decision actually turns on: if the clustering were refit WITHOUT the per-session
z-score, would the individuality axis be better?

It reruns 3.3_wavelet_clusters.ipynb end to end, twice, changing exactly one line:

  ZSCORED   supersession = vstack(zscore(subsample_s, axis=0) for s in sessions)
            assignment   = (zscore(session, axis=0) - g_mu) / g_sd     [CURRENT]
  RAW       supersession = vstack(subsample_s for s in sessions)
            assignment   = (session - g_mu) / g_sd                     [COUNTERFACTUAL]

Everything else is the notebook's: the 2000-sample per-session subsamples from
`data/paw_subsampled_wavelets/`, the 20 features {l,r}_paw_{x,y} x {0.5,1,2,4,8} Hz,
a global z-score, PCA to 95% variance, KMeans(k, random_state=2024), then
nearest-centroid assignment of every bin of every session.

The states are then binned exactly as 5_syllable_generation.ipynb does -- grouped by
(trial, epoch), `rescale_sequence(seq, 10, 'mode')` -- giving 4 epochs x 10 bins, and
one-hot to 280 paw-only features per session. Paw-only is the right comparison because
the whisk and lick channels are untouched by this change.

VALIDATION. The ZSCORED rerun is checked against the states already on disk
(`identifiable_states` first digit in data/states_files) with the adjusted Rand index,
so the counterfactual is known to differ from the current pipeline in one thing only.

WHAT TO LOOK AT. Not the mouse-ID score alone. A rig signature also identifies the
mouse, because a mouse is recorded on one rig -- so the table reports mouse ID, mouse ID
after lab-centring, lab eta^2 and lab ID together.
"""
import os
import re
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

import zscore_cost as Z

warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUB_DIR = ROOT / 'data' / 'paw_subsampled_wavelets'
WAV_DIR = ROOT / 'data' / 'paw_wavelets'
STATES_DIR = ROOT / 'data' / 'states_files'

FREQS = ['0.5', '1.0', '2.0', '4.0', '8.0']
ALL_F = ['0.5', '1.0', '2.0', '4.0', '8.0', '16.0', '32.0']
BLOCKS = ['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y']
VAR = [f'{b}{f}' for b in BLOCKS for f in FREQS]
# var_init = 4 positions, then 7 frequencies per block, in BLOCKS order
SUB_IDX = [4 + 7 * b + ALL_F.index(f) for b in range(4) for f in FREQS]

K = 8                 # match 8_k_10_bin_syllables_19-08-2026
PCA_CUTOFF = 0.95
TARGET_LEN, EPOCHS = 10, ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
CACHE = HERE / 'rerun_features.npz'


def _mode(a):
    """Mode of an integer array, NaNs already removed. Ties -> smallest, as scipy does."""
    if not len(a):
        return np.nan
    v, c = np.unique(a, return_counts=True)
    return float(v[np.argmax(c)])


def rescale_mode(seq, target=TARGET_LEN):
    """5_syllable_generation's rescale_sequence(., 10, 'mode'), NaNs dropped first."""
    seq = np.asarray(seq, float)
    seq = seq[np.isfinite(seq)]
    if not len(seq):
        return np.full(target, np.nan)
    if len(seq) == target:
        return seq
    if len(seq) > target:
        return np.array([_mode(b) for b in np.array_split(seq, target)])
    return seq[np.floor(np.linspace(0, len(seq) - 1, target)).astype(int)]


def session_list():
    pat = re.compile(r'^([0-9a-f\-]{36})_(.+)\.npy$')
    out = []
    for f in sorted(os.listdir(SUB_DIR)):
        m = pat.match(f)
        if m:
            out.append((m.group(1), m.group(2), f))
    return out


def fit_clustering(mode, sessions, seed=2024):
    """Supersession -> global z -> PCA(95%) -> KMeans(K). Returns everything needed
    to assign a new session the same way."""
    mats = []
    for eid, mouse, f in sessions:
        a = np.load(SUB_DIR / f)[:, SUB_IDX].astype(float)
        a = a[np.isfinite(a).all(1)]
        mats.append(stats.zscore(a, axis=0) if mode == 'zscored' else a)
    S = np.vstack(mats)
    g_mu, g_sd = np.nanmean(S, 0), np.nanstd(S, 0)
    Sz = (S - g_mu) / g_sd
    pca = PCA(n_components=20).fit(Sz)
    n_comp = int(min(20, np.argmax(np.cumsum(pca.explained_variance_ratio_) >= PCA_CUTOFF) + 1))
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(pca.transform(Sz)[:, :n_comp])
    print(f'  [{mode}] supersession {S.shape}, PCA -> {n_comp} comps, KMeans k={K}')
    return dict(mode=mode, g_mu=g_mu, g_sd=g_sd, pca=pca, n_comp=n_comp,
                centroids=km.cluster_centers_)


def assign(model, A):
    """A: (n_bins, 20) raw wavelet amplitudes for one session -> state per bin."""
    X = stats.zscore(A, axis=0) if model['mode'] == 'zscored' else A
    X = (X - model['g_mu']) / model['g_sd']
    P = model['pca'].transform(X)[:, :model['n_comp']]
    return np.argmin(cdist(P, model['centroids']), axis=1)


def build(models, sessions):
    """Per session: 4 epochs x 10 bins of paw state, one-hot -> 280 features.
    Also returns the ARI of each rerun against the states already on disk."""
    out = {m['mode']: {} for m in models}
    ari = {m['mode']: [] for m in models}
    for i, (eid, mouse, _) in enumerate(sessions):
        wf = WAV_DIR / f'paw_vel_wavelets_{eid}_{mouse}'
        sf = STATES_DIR / f'8_states_file_{eid}_{mouse}'
        if not wf.exists() or not sf.exists():
            continue
        W = pd.read_parquet(wf, columns=['Bin'] + VAR)
        S = pd.read_parquet(sf, columns=['Bin', 'trial_id', 'broader_label',
                                         'identifiable_states'])
        A = W[VAR].to_numpy(float)
        ok = np.isfinite(A).all(1)
        if ok.sum() < 1000:
            continue
        # align on Bin -- both come from the same design matrix, so the values are equal
        W = W.assign(_k=np.round(W['Bin'].to_numpy(), 6))
        S = S.assign(_k=np.round(S['Bin'].to_numpy(), 6))
        for m in models:
            st = np.full(len(W), np.nan)
            st[ok] = assign(m, A[ok])
            D = S.merge(pd.DataFrame({'_k': W['_k'], 'state': st}), on='_k', how='left')
            # paw digit of the existing states, for the ARI sanity check
            ex = pd.to_numeric(D['identifiable_states'], errors='coerce') // 100
            v = D['state'].notna() & ex.notna()
            if v.sum() > 1000:
                ari[m['mode']].append(adjusted_rand_score(ex[v].astype(int),
                                                          D['state'][v].astype(int)))
            D = D.dropna(subset=['trial_id'])
            D = D[D['broader_label'].isin(EPOCHS)]
            seqs = (D.groupby(['trial_id', 'broader_label'], sort=False)['state']
                    .apply(lambda s: rescale_mode(s.to_numpy())))
            piv = seqs.unstack('broader_label').reindex(columns=EPOCHS).dropna()
            if not len(piv):
                continue
            # one-hot each of the 40 timesteps into K columns, then session mean
            M = np.stack([np.hstack(r) for r in piv.to_numpy()])       # trials x 40
            oh = np.zeros((len(M), 40 * K))
            for t in range(40):
                v2 = M[:, t]
                f2 = np.isfinite(v2)
                oh[np.where(f2)[0], t * K + v2[f2].astype(int)] = 1
                oh[~f2, t * K:(t + 1) * K] = np.nan
            oh = np.delete(oh, [t * K + 1 for t in range(40)], axis=1)  # drop state 1
            out[m['mode']][eid] = np.nanmean(oh, axis=0)
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(sessions)}', flush=True)
    for m in models:
        print(f'  [{m["mode"]}] ARI vs the states on disk: '
              f'{np.mean(ari[m["mode"]]):.3f} (n={len(ari[m["mode"]])})')
    return out


def main():
    sessions = session_list()
    print(f'{len(sessions)} subsampled sessions')
    models = [fit_clustering('zscored', sessions), fit_clustering('raw', sessions)]

    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        feats = {k: d[k].item() for k in ('zscored', 'raw')}
    else:
        feats = build(models, sessions)
        np.savez(CACHE, zscored=feats['zscored'], raw=feats['raw'])
        print(f'cached {CACHE}')

    # restrict to the LDA cohort, so the comparison is on identical sessions
    import make_embedding as me
    ref, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
    from functions import lab_labels
    idx = [s for s in ref.index if s in feats['zscored'] and s in feats['raw']]
    print(f'\ncohort: {len(idx)} sessions, {mouse_of.loc[idx].nunique()} mice')
    y = pd.factorize(mouse_of.loc[idx])[0]
    lab_v = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx])))
    mice = mouse_of.loc[idx].to_numpy()

    W = pd.read_parquet(Z.CACHE).set_index('session').reindex(idx)
    amp = W[[c for c in W.columns if c.startswith('logmean_')]].to_numpy(float).mean(1)

    print('\n' + '=' * 96)
    print('PAW-ONLY STATES: WITH vs WITHOUT THE PER-SESSION Z-SCORE')
    print('=' * 96)
    print(f'{"variant":36s} {"dims":>5s} {"mouse":>7s} {"mouse|lab":>10s} '
          f'{"lab eta2":>9s} {"lab acc":>8s} {"ICC":>6s} {"r(LD1,amp)":>11s}')
    for name in ('zscored', 'raw'):
        X = np.vstack([feats[name][s] for s in idx])
        acc, _ = Z.loso_score(X, y)
        accc, _ = Z.loso_score(Z.lab_center(X, lab_v), y)
        lacc, _ = Z.loso_score(X, pd.factorize(lab_v)[0], n_repeats=1)
        ld1 = fit_ld1(X, y)
        r = stats.spearmanr(pd.Series(ld1).groupby(mice).mean(),
                            pd.Series(amp).groupby(mice).mean())[0]
        lbl = 'ZSCORED  (current pipeline)' if name == 'zscored' else 'RAW      (no per-session z)'
        print(f'{lbl:36s} {X.shape[1]:5d} {acc:7.3f} {accc:10.3f} '
              f'{Z.lab_eta2(X, lab_v):9.3f} {lacc:8.3f} {Z.icc1(X, mice):6.3f} {r:+11.3f}')


def fit_ld1(X, y):
    """LD1 of the shrinkage LDA fit on all sessions, centred as make_embedding does."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = np.nan_to_num(np.asarray(X, float))
    k = len(np.unique(y))
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.5,
                                     priors=np.ones(k) / k).fit(X, y)
    return ((X - lda.means_.mean(0)) @ lda.scalings_)[:, 0]


if __name__ == '__main__':
    main()
