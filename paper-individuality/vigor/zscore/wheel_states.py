"""
WHEEL STATES: HOW MANY, WHETHER TO Z-SCORE, AND DO THEY ADD INDIVIDUALITY TO THE PAW?
====================================================================================
The wheel is the one channel measured without a camera: a rotary encoder, physical units,
identical on every rig. So the reason to z-score paw amplitudes per session (camera
distance, pixel scale) does not apply to it, and its level may carry real vigor -- which
LD1 tracks (vigor/lda: wheel speed survives partialling paw speed, not vice versa).

Four questions:
  A. DOES THE WHEEL NEED Z-SCORING?  Lab share of each session's wheel amplitude level,
     against a null that shuffles lab across mice. (Compare the paw, which is in pixels.)
  B. DO PAW AND WHEEL DECORRELATE?  Frame level: r(log paw power, log wheel power) within
     session. State level: normalised mutual information between the production paw
     states and the wheel states. Session level: r between the two amplitude levels.
  C. HOW MANY WHEEL STATES?  K = 2..8, clustered with and without the per-session z.
  D. DO THEY ADD INDIVIDUALITY?  Mouse ID / mouse|lab / lab from a held-out mouse / ICC for
     wheel alone, paw + wheel, and the paper's 360 features + wheel, under the protocol
     of vigor/lda/zscore_cost.py.

Wheel pipeline, mirroring segmentation/3.*_uniform: uniform 2,000 frames per session;
log(x + EPS) of the 0.5-8 Hz bands (16/32 Hz are encoder noise, median amplitude < 0.001);
NORM = 'global' (one pooled mean/SD, only to put bands on a common scale -- session level
stays in) or 'session' (each session's own full-session mean/SD); KMeans(K, seed 2024) on
the 5 bands directly (no PCA at 5 dims); every frame labelled by nearest centroid; states
renumbered by power. Features: 4 epochs x 10 bins, mode state per bin, one-hot, one
redundant column per timestep dropped, session mean -- as for the paw.

Selecting K on the same cross-validated score that is then reported is mildly optimistic.
Read the curve, not its maximum: prefer the smallest K on the plateau.

Everything is built from current data. Output: results_wheel_states.txt
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')       # parallelise over rows, not inside LDA
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from joblib import Parallel, delayed

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'lda'))
import rerun_no_zscore as R                                 # noqa: E402  rescale_mode, EPOCHS
import zscore_cost as Z                                     # noqa: E402
import make_embedding as me                                 # noqa: E402
from functions import lab_labels                            # noqa: E402

warnings.filterwarnings('ignore')

DATA = ROOT / 'data'
BANDS = ['0.5', '1.0', '2.0', '4.0', '8.0']
WHEEL = [f'avg_wheel_vel{b}' for b in BANDS]
PAW = [f'{p}_{a}{b}' for p in ('l_paw', 'r_paw') for a in 'xy' for b in BANDS]
EPS = 0.005               # wheel still ~1e-4, moving ~0.05-0.3: the floor sits between
PAW_EPS = 0.1             # as in 3.2_wavelet_subsample_uniform
KS = range(2, 9)
NORMS = ('global', 'session')
N_SUB, SEED = 2000, 0
N_JOBS = 18
CACHE = HERE / 'wheel_states_features.npz'


def sessions():
    tags = sorted(f[len('wheel_vel_wavelets_'):] for f in os.listdir(DATA / 'wheel_wavelets')
                  if f.startswith('wheel_vel_wavelets_'))
    return [(t[:36], t[37:]) for t in tags]


def load_wheel(eid, mouse):
    W = pd.read_parquet(DATA / 'wheel_wavelets' / f'wheel_vel_wavelets_{eid}_{mouse}',
                        columns=['Bin'] + WHEEL)
    A = W[WHEEL].to_numpy(float)
    ok = np.isfinite(A).all(1)
    return np.log(A[ok] + EPS), np.round(W['Bin'].to_numpy(float)[ok], 6)


def onehot(D, k):
    """(trial, epoch)-rescaled states -> session mean of the one-hot (40 * (k - 1))."""
    D = D.dropna(subset=['trial_id'])
    D = D[D['broader_label'].isin(R.EPOCHS)]
    seqs = (D.groupby(['trial_id', 'broader_label'], sort=False)['state']
            .apply(lambda s: R.rescale_mode(s.to_numpy())))
    piv = seqs.unstack('broader_label').reindex(columns=R.EPOCHS).dropna()
    if not len(piv):
        return None
    M = np.stack([np.hstack(r) for r in piv.to_numpy()])
    oh = np.zeros((len(M), 40 * k))
    for t in range(40):
        v = M[:, t]
        f = np.isfinite(v)
        oh[np.where(f)[0], t * k + v[f].astype(int)] = 1
        oh[~f, t * k:(t + 1) * k] = np.nan
    oh = np.delete(oh, [t * k for t in range(40)], axis=1)          # drop state 0 per step
    return np.nanmean(oh, axis=0)


def session_features(eid, mouse, models, norm_stats):
    """One session: wheel states for every (norm, K), their one-hot features, the NMI with
    the production paw states, and the frame-level paw/wheel correlation. Runs in a worker."""
    sf = DATA / 'states_files' / f'8_states_file_{eid}_{mouse}'
    pf = DATA / 'paw_wavelets' / f'paw_vel_wavelets_{eid}_{mouse}'
    if not sf.exists():
        return None
    S = pd.read_parquet(sf, columns=['Bin', 'trial_id', 'broader_label', 'identifiable_states'])
    S['_k'] = np.round(S['Bin'].to_numpy(float), 6)
    L, kb = load_wheel(eid, mouse)
    r = dict(eid=eid, level=L.mean(0), frame_r=None, feats={}, nmi={})
    if pf.exists():                                                    # frame-level coupling
        P = pd.read_parquet(pf, columns=['Bin'] + PAW)
        pa = P[PAW].to_numpy(float)
        okp = np.isfinite(pa).all(1)
        pp = pd.Series(np.log(pa[okp] + PAW_EPS).mean(1),
                       index=np.round(P['Bin'].to_numpy(float)[okp], 6))
        j = pd.Series(L.mean(1), index=kb).to_frame('w').join(pp.rename('p'), how='inner')
        j = j[~j.index.duplicated()]
        r['frame_r'] = np.corrcoef(j['w'], j['p'])[0, 1]
    prod = pd.to_numeric(S['identifiable_states'], errors='coerce') // 100
    for norm in NORMS:
        mu, sd = norm_stats[norm]
        Xs = (L - mu) / sd
        for k in KS:
            st = cdist(Xs, models[norm, k]).argmin(1)
            col = S[['_k']].merge(pd.DataFrame({'_k': kb, 'state': st}).drop_duplicates('_k'),
                                  on='_k', how='left')['state']
            ok = col.notna().to_numpy() & prod.notna().to_numpy()
            if ok.sum() > 1000:
                r['nmi'][norm, k] = normalized_mutual_info_score(prod[ok].astype(int),
                                                                 col[ok].astype(int))
            x = onehot(S.assign(state=col.to_numpy()), k)
            if x is not None:
                r['feats'][norm, k] = x
    return r


def build():
    ses = sessions()
    rng = np.random.default_rng(SEED)
    # ---- training set: uniform subsample, plus each session's full-session stats
    subs, sstats = {}, {}
    for eid, mouse in ses:
        L, _ = load_wheel(eid, mouse)
        subs[eid] = L[np.sort(rng.choice(len(L), N_SUB, replace=False))]
        sstats[eid] = (L.mean(0), L.std(0))
    pool = np.vstack(list(subs.values()))
    pooled = (pool.mean(0), pool.std(0))
    stats_of = {'global': lambda e: pooled, 'session': lambda e: sstats[e]}
    models = {}
    for norm in NORMS:
        X = np.vstack([(subs[e] - stats_of[norm](e)[0]) / stats_of[norm](e)[1] for e in subs])
        for k in KS:
            km = KMeans(n_clusters=k, random_state=2024).fit(X)
            order = np.argsort([X[km.labels_ == c].mean() for c in range(k)])
            models[norm, k] = km.cluster_centers_[order]
    print(f'fit {len(models)} wheel models on {pool.shape}', flush=True)

    out = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(session_features)(eid, mouse, models, {'global': pooled, 'session': sstats[eid]})
        for eid, mouse in ses)
    feats = {(n, k): {} for n in NORMS for k in KS}
    nmi = {(n, k): [] for n in NORMS for k in KS}
    frame_r, level = [], {}
    for r in out:
        if r is None:
            continue
        eid = r['eid']
        level[eid] = r['level']
        if r['frame_r'] is not None:
            frame_r.append(r['frame_r'])
        for key, x in r['feats'].items():
            feats[key][eid] = x
        for key, v in r['nmi'].items():
            nmi[key].append(v)
    return dict(feats=feats, nmi=nmi, frame_r=np.array(frame_r), level=level)


def score(name, X, y, labs, mice):
    """The protocol's metrics for one block; runs in a worker."""
    from compare_pipelines import lomo_lab_acc
    acc, _ = Z.loso_score(X, y)
    accc, _ = Z.loso_score(Z.lab_center(X, labs), y)
    return (name, X.shape[1], acc, accc, Z.lab_eta2(X, labs),
            lomo_lab_acc(X, labs, mice), Z.icc1(X, mice))


def eta(v, g):
    s = pd.Series(v).groupby(g)
    return ((s.mean() - v.mean()) ** 2 * s.size()).sum() / ((v - v.mean()) ** 2).sum()


def main():
    if CACHE.exists():
        B = np.load(CACHE, allow_pickle=True)['B'].item()
    else:
        B = build()
        np.savez(CACHE, B=B)
        print(f'cached {CACHE}')

    syl, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
    paw280 = np.load(HERE / 'compare_pipelines_features.npz', allow_pickle=True)['feats'].item()
    paw280 = paw280['production (density subsample, session z)']
    idx = [s for s in syl.index if s in paw280 and s in B['level']
           and all(s in B['feats'][key] for key in B['feats'])]
    mice = mouse_of.loc[idx].to_numpy()
    labs = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx], verbose=False)))
    y = pd.factorize(mice)[0]
    m_lab = pd.Series(labs, index=mice).groupby(level=0).first()
    rng = np.random.default_rng(0)
    print(f'\ncohort: {len(idx)} sessions, {len(set(mice))} mice, {len(set(labs))} labs')

    # ------------------------------------------------------------ A. z-score needed?
    print('\n' + '=' * 96)
    print('A. LAB SHARE OF THE SESSION AMPLITUDE LEVEL   (null shuffles lab across mice)')
    print('=' * 96)
    Wl = np.vstack([B['level'][s] for s in idx])
    paw_level = []                                       # rebuilt here, not read from a cache
    for s in idx:
        m = mouse_of.loc[s]
        a = pd.read_parquet(DATA / 'paw_wavelets' / f'paw_vel_wavelets_{s}_{m}',
                            columns=PAW).dropna().to_numpy(float)
        paw_level.append(np.log(a + PAW_EPS).mean(0))
    Pl = np.array(paw_level)
    print(f'{"signal":34s} {"lab eta2":>9s} {"null":>7s} {"p":>7s} {"ICC mouse":>10s}')
    rows = ([('wheel, mean log amplitude', Wl.mean(1)), ('paw, mean log amplitude', Pl.mean(1))]
            + [(f'  wheel {b} Hz', Wl[:, j]) for j, b in enumerate(BANDS)])
    for name, M in rows:
        obs = eta(M, labs)
        null = np.array([eta(M, pd.Series(rng.permutation(m_lab.to_numpy()), index=m_lab.index)
                             .reindex(mice).to_numpy()) for _ in range(2000)])
        s = pd.Series(M).groupby(mice)
        kk = s.size().mean()
        msb, msw = s.mean().var(ddof=1) * kk, s.var(ddof=1).mean()
        print(f'{name:34s} {obs:9.3f} {null.mean():7.3f} {(1 + (null >= obs).sum()) / 2001:7.4f} '
              f'{(msb - msw) / (msb + (kk - 1) * msw):10.2f}')

    # ------------------------------------------------------------ B. decorrelation
    print('\n' + '=' * 96)
    print('B. DO PAW AND WHEEL DECORRELATE?')
    print('=' * 96)
    fr = B['frame_r']
    print(f'frame level, within session: r(log paw power, log wheel power) median {np.median(fr):.2f} '
          f'[IQR {np.percentile(fr, 25):.2f}, {np.percentile(fr, 75):.2f}], n={len(fr)}')
    mw = pd.Series(Wl.mean(1)).groupby(mice).mean()
    mp = pd.Series(Pl.mean(1)).groupby(mice).mean()
    print(f'session level (levels):  r = {stats.spearmanr(Wl.mean(1), Pl.mean(1))[0]:+.2f} across '
          f'sessions, {stats.spearmanr(mw, mp)[0]:+.2f} across mice')
    print('state level: NMI(production paw states, wheel states), median over sessions')
    for norm in NORMS:
        print(f'  {norm:8s} ' + '  '.join(f'K={k}: {np.median(B["nmi"][norm, k]):.3f}' for k in KS))

    # ------------------------------------------------------------ C/D. how many, and do they add
    blocks = [('paw 280 (production) alone', paw280, None, None),
              ('360 syllables alone', None, None, None)]
    for norm in NORMS:
        for k in KS:
            blocks += [(f'wheel {norm:7s} K={k} alone', None, norm, k),
                       (f'paw 280 + wheel {norm:7s} K={k}', paw280, norm, k),
                       (f'360 + wheel {norm:7s} K={k}', 'syl', norm, k)]

    def matrix(base, norm, k):
        parts = []
        if base is paw280:
            parts.append(np.vstack([paw280[s] for s in idx]))
        elif base == 'syl' or (base is None and norm is None):
            parts.append(syl.loc[idx].to_numpy(float))
        if norm is not None:
            parts.append(np.vstack([B['feats'][norm, k][s] for s in idx]))
        return np.hstack(parts)

    jobs = [delayed(score)(n, matrix(b, nm, k), y, labs, mice) for n, b, nm, k in blocks]
    res = Parallel(n_jobs=N_JOBS, verbose=0)(jobs)
    print('\n' + '=' * 96)
    print(f'C/D. MOUSE vs LAB   (LOSO; chance: mouse {1 / len(set(mice)):.3f}, lab ~0.100)')
    print('=' * 96)
    print(f'{"block":34s} {"dims":>5s} {"mouse":>6s} {"mouse|lab":>9s} {"lab eta2":>8s} '
          f'{"lab LOMO":>8s} {"ICC":>6s}')
    for name, d, acc, accc, le, lomo, icc in res:
        print(f'{name:34s} {d:5d} {acc:6.3f} {accc:9.3f} {le:8.3f} {lomo:8.3f} {icc:6.3f}')


if __name__ == '__main__':
    main()
