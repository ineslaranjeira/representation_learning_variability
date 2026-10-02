"""
WHICH WAVELET NORMALISATION? THE FOUR UNIFORM-SUBSAMPLE PIPELINES AGAINST PRODUCTION
===================================================================================
`segmentation/3.2_wavelet_subsample_uniform.ipynb` + `3.3_wavelet_clusters_uniform.ipynb`
replace the density-weighted t-SNE subsample with a uniform one, standardise the training
frames and the labelled frames with the same statistics, and optionally log the amplitudes.
`run_variants.py` runs 3.3 once per (LOG, NORM) and writes one states folder each:

  data/paw_states_uniform_{raw|log}_{session|global}z/most_likely_states_8_<mouse><eid>.npy

This script puts them next to the production paw states (the first digit of
`identifiable_states` in data/states_files, K = 8) and asks two things:

  1. HOW DIFFERENT ARE THE STATES?  Frame-level adjusted Rand index against production,
     per session; and how evenly the frames spread over the 8 states.
  2. HOW MUCH RIG GOES IN?  On the downstream features the LDA actually sees -- paw-only,
     4 epochs x 10 bins, one-hot, 280 per session, built exactly as in
     vigor/lda/rerun_no_zscore.py -- with that script's metrics (mouse ID, mouse ID after
     lab-centring, lab eta^2, lab ID, ICC), plus lab ID from a held-out MOUSE, which is
     the one that cannot lean on mouse identity. Also lab eta^2 of the plain 8-state
     occupancy, against a null that shuffles lab across mice.

The two variants of rerun_no_zscore.py (old density subsample, no log; with and without
the per-session z) are REFIT here on the current data, not read from its cache.

Output: results_compare_pipelines.txt (run with `python compare_pipelines.py | tee ...`).
"""
import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                      # paper-individuality/
sys.path.insert(0, str(HERE.parent / 'lda'))
import rerun_no_zscore as R                                 # noqa: E402  rescale_mode, EPOCHS
import zscore_cost as Z                                     # noqa: E402  the LDA protocol
import make_embedding as me                                 # noqa: E402  the LDA cohort
from functions import lab_labels                            # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)

K = 8
DATA = ROOT / 'data'
STATES_FILES = DATA / 'states_files'
VARIANTS = {                     # name -> states folder; None = production
    'production (density subsample, session z)': None,
    'uniform, raw,  session z': 'paw_states_uniform_raw_sessionz',
    'uniform, log,  session z': 'paw_states_uniform_log_sessionz',
    'uniform, raw,  global z':  'paw_states_uniform_raw_globalz',
    'uniform, log,  global z':  'paw_states_uniform_log_globalz',
}
CACHE = HERE / 'compare_pipelines_features.npz'
OLD_CACHE = HERE / 'compare_pipelines_old_rerun.npz'
N_PERM = 2000


def onehot_280(D):
    """(trial, epoch)-rescaled paw states -> session mean of the one-hot, as rerun_no_zscore."""
    D = D.dropna(subset=['trial_id'])
    D = D[D['broader_label'].isin(R.EPOCHS)]
    seqs = (D.groupby(['trial_id', 'broader_label'], sort=False)['state']
            .apply(lambda s: R.rescale_mode(s.to_numpy())))
    piv = seqs.unstack('broader_label').reindex(columns=R.EPOCHS).dropna()
    if not len(piv):
        return None
    M = np.stack([np.hstack(r) for r in piv.to_numpy()])            # trials x 40
    oh = np.zeros((len(M), 40 * K))
    for t in range(40):
        v = M[:, t]
        f = np.isfinite(v)
        oh[np.where(f)[0], t * K + v[f].astype(int)] = 1
        oh[~f, t * K:(t + 1) * K] = np.nan
    oh = np.delete(oh, [t * K + 1 for t in range(40)], axis=1)       # one column is redundant
    return np.nanmean(oh, axis=0)


def build(eids_mice):
    feats = {v: {} for v in VARIANTS}
    occ = {v: {} for v in VARIANTS}
    ari = {v: {} for v in VARIANTS}
    for i, (eid, mouse) in enumerate(eids_mice):
        sf = STATES_FILES / f'8_states_file_{eid}_{mouse}'
        if not sf.exists():
            continue
        S = pd.read_parquet(sf, columns=['Bin', 'trial_id', 'broader_label', 'identifiable_states'])
        S['_k'] = np.round(S['Bin'].to_numpy(float), 6)
        prod = pd.to_numeric(S['identifiable_states'], errors='coerce') // 100
        for v, folder in VARIANTS.items():
            if folder is None:
                st = prod
            else:
                f = DATA / folder / f'most_likely_states_{K}_{mouse}{eid}.npy'
                if not f.exists():
                    continue
                a = np.load(f)
                new = pd.DataFrame({'_k': np.round(a[1], 6), 'state': a[0]})
                st = S[['_k']].merge(new, on='_k', how='left')['state']
                ok = st.notna().to_numpy() & prod.notna().to_numpy()
                if ok.sum() > 1000:
                    ari[v][eid] = adjusted_rand_score(prod[ok].astype(int), st[ok].astype(int))
                # occupancy over ALL labelled frames of the session, not just trial bins
                occ[v][eid] = np.bincount(a[0].astype(int), minlength=K) / a.shape[1]
            if folder is None:
                s = st.dropna().astype(int)
                occ[v][eid] = np.bincount(s, minlength=K) / len(s)
            x = onehot_280(S.assign(state=st.to_numpy()))
            if x is not None:
                feats[v][eid] = x
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(eids_mice)}', flush=True)
    return feats, occ, ari


def lomo_lab_acc(X, labs, mice):
    """Lab from a held-out MOUSE: none of that mouse's sessions are in training."""
    X = np.nan_to_num(np.asarray(X, float))
    y = pd.factorize(labs)[0]
    hit = []
    for m in np.unique(mice):
        te = mice == m
        pred = Z._fit(X[~te], y[~te]).predict(X[te])
        hit.extend(pred == y[te])
    return float(np.mean(hit))


def occ_lab_eta2(O, labs, mice, rng):
    """Multivariate eta^2 of lab on the 8-state occupancy, and a null that shuffles
    lab across MICE (so nesting is kept: a mouse's sessions stay together)."""
    def eta(lb):
        g = pd.DataFrame(O).groupby(lb)
        between = ((g.mean() - O.mean(0)) ** 2).mul(g.size(), axis=0).to_numpy().sum()
        return between / ((O - O.mean(0)) ** 2).sum()
    obs = eta(labs)
    m_lab = pd.Series(labs, index=mice).groupby(level=0).first()
    null = []
    for _ in range(N_PERM):
        perm = pd.Series(rng.permutation(m_lab.to_numpy()), index=m_lab.index)
        null.append(eta(perm.reindex(mice).to_numpy()))
    null = np.array(null)
    return obs, null.mean(), (1 + (null >= obs).sum()) / (1 + N_PERM)


def main():
    tags = sorted(f[len('paw_vel_wavelets_'):] for f in os.listdir(DATA / 'paw_wavelets')
                  if f.startswith('paw_vel_wavelets_'))
    eids_mice = [(t[:36], t[37:]) for t in tags]
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        feats, occ, ari = d['feats'].item(), d['occ'].item(), d['ari'].item()
    else:
        feats, occ, ari = build(eids_mice)
        np.savez(CACHE, feats=feats, occ=occ, ari=ari)
        print(f'cached {CACHE}')

    # the old counterfactual, REFIT here on the current data rather than read from
    # rerun_no_zscore's cache. Its SUB_DIR (data/paw_subsampled_wavelets) was emptied on
    # 2026-09-30; the original subsamples now live in paw_subsampled_wavelets18ago.
    if OLD_CACHE.exists():
        old = np.load(OLD_CACHE, allow_pickle=True)
        old = {k: old[k].item() for k in ('zscored', 'raw')}
    else:
        R.SUB_DIR = DATA / 'paw_subsampled_wavelets18ago'
        sessions = R.session_list()
        print(f'refitting rerun_no_zscore on {len(sessions)} sessions from {R.SUB_DIR}')
        old = R.build([R.fit_clustering('zscored', sessions),
                       R.fit_clustering('raw', sessions)], sessions)
        np.savez(OLD_CACHE, **old)
    feats['density subsample, session z (refit)'] = old['zscored']
    feats['density subsample, no z'] = old['raw']

    # ---------------------------------------------------------------- 1. how different
    mouse_all = pd.Series({e: m for e, m in eids_mice})
    print('\n' + '=' * 100)
    print('1. HOW DIFFERENT ARE THE STATES?   (all sessions with a states file)')
    print('=' * 100)
    print(f'{"variant":44s} {"n":>4s} {"ARI vs prod":>12s} {"[IQR]":>14s} '
          f'{"state share min-max":>20s} {"sessions w/ a state <1%":>24s}')
    for v in VARIANTS:
        O = np.vstack(list(occ[v].values()))
        pooled = O.mean(0)
        a = np.array(list(ari[v].values())) if ari[v] else np.array([1.0])
        rare = (O < 0.01).any(1).mean()
        print(f'{v:44s} {len(O):4d} {np.median(a):12.3f} '
              f'{f"[{np.percentile(a, 25):.2f}, {np.percentile(a, 75):.2f}]":>14s} '
              f'{f"{pooled.min():.3f} - {pooled.max():.3f}":>20s} {rare:24.2f}')

    # ---------------------------------------------------------------- 2. how much rig
    ref, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
    idx = [s for s in ref.index if all(s in feats[v] for v in feats)]
    mice = mouse_of.loc[idx].to_numpy()
    labs = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx])))
    y = pd.factorize(mice)[0]
    print(f'\nLDA cohort: {len(idx)} sessions, {len(set(mice))} mice, {len(set(labs))} labs')

    rng = np.random.default_rng(0)
    print('\n' + '=' * 100)
    print('2a. RIG IN THE 8-STATE OCCUPANCY   (frame level, LDA cohort; null shuffles lab across mice)')
    print('=' * 100)
    print(f'{"variant":44s} {"lab eta2":>9s} {"null":>7s} {"excess":>7s} {"p":>7s}')
    for v in VARIANTS:
        O = np.vstack([occ[v][s] for s in idx])
        obs, nul, p = occ_lab_eta2(O, labs, mice, rng)
        print(f'{v:44s} {obs:9.3f} {nul:7.3f} {obs - nul:7.3f} {p:7.4f}')

    print('\n' + '=' * 100)
    print('2b. THE LDA FEATURES (paw only, 280 per session)   chance: mouse 1/%d, lab ~0.10'
          % len(set(mice)))
    print('=' * 100)
    print(f'{"variant":44s} {"mouse":>6s} {"mouse|lab":>9s} {"lab eta2":>8s} '
          f'{"lab LOSO":>8s} {"lab LOMO":>8s} {"ICC":>6s}')
    for v in feats:
        X = np.vstack([feats[v][s] for s in idx])
        acc, _ = Z.loso_score(X, y)
        accc, _ = Z.loso_score(Z.lab_center(X, labs), y)
        lacc, _ = Z.loso_score(X, pd.factorize(labs)[0], n_repeats=1)
        print(f'{v:44s} {acc:6.3f} {accc:9.3f} {Z.lab_eta2(X, labs):8.3f} '
              f'{lacc:8.3f} {lomo_lab_acc(X, labs, mice):8.3f} {Z.icc1(X, mice):6.3f}', flush=True)


if __name__ == '__main__':
    main()
