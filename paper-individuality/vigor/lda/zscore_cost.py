"""
WHAT DOES THE PER-SESSION Z-SCORE COST, AND IS WHAT IT COSTS INDIVIDUALITY?
===========================================================================
`3.3_wavelet_clusters.ipynb` standardises each session's 20 wavelet features by THAT
session's own mean and SD before assigning states. Per-column standardisation is an
affine map applied column by column, so the split is exact and complete:

  DISCARDED   each feature's session MEAN and SD                       40 numbers
  KEPT        each feature's standardised shape (skew, kurtosis, ...)  invariant
  KEPT        every between-feature CORRELATION                        invariant

So "does the shape stay expressive?" is not a matter of opinion -- the three blocks can
be built separately and each asked how well it identifies the mouse, under exactly the
protocol the real LDA uses.

THE QUESTION BEHIND THE QUESTION. Raw amplitude is not a clean trait. A mouse is almost
always recorded on ONE rig, and `vigor/paw_bias` measured a per-session camera-gain term
whose SD (0.092) exceeds its mean (0.089). So "mouse identifiable from raw amplitude"
is confounded with "rig identifiable from raw amplitude". Every score below is therefore
reported twice: raw, and after centring each feature within its lab. A block that
identifies the mouse only before lab-centring is measuring the rig.

PROTOCOL. Leave-one-session-out, balanced draw of <=3 sessions per mouse, shrinkage LDA
at 0.5 -- the same as 4_mice/lda_shrinkage.ipynb, on its 260 sessions / 56 mice. The
inner CV that picks the shrinkage is skipped; the notebook records 0.5 as its modal
choice and this is a comparison BETWEEN feature sets under one fixed setting.
"""
import os
import re
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import vigor_vs_lda as V

warnings.filterwarnings('ignore', message='Only one sample available', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WAVELET_DIR = ROOT / 'data' / 'paw_wavelets'
CACHE = HERE / 'wavelet_moments_sessions.parquet'

FREQS = ['0.5', '1.0', '2.0', '4.0', '8.0']
VAR = [f'{p}_{c}{f}' for p in ('l_paw', 'r_paw') for c in ('x', 'y') for f in FREQS]
N_PER_MOUSE, SHRINKAGE, N_REPEATS = 3, 0.5, 3


# --------------------------------------------------------------------------- cache
def build_moments():
    """Per session: the 40 numbers the z-score discards, and the ones it keeps."""
    pat = re.compile(r'^paw_vel_wavelets_([0-9a-f\-]{36})_(.+)$')
    rows = []
    for f in sorted(os.listdir(WAVELET_DIR)):
        m = pat.match(f)
        if not m or not os.path.isfile(WAVELET_DIR / f):
            continue
        eid, mouse = m.groups()
        d = pd.read_parquet(WAVELET_DIR / f, columns=VAR).dropna()
        a = d.to_numpy(float)
        r = dict(session=eid, mouse_name=mouse, n_bins=len(a))
        mu, sd = a.mean(0), a.std(0)
        # LOG, because wavelet amplitudes are positive and heavy-tailed, and because a
        # multiplicative camera-gain error is ADDITIVE in logs -- which is the form the
        # lab-centring below can actually remove.
        for i, c in enumerate(VAR):
            r[f'logmean_{c}'] = float(np.log(mu[i]))
            r[f'logsd_{c}'] = float(np.log(sd[i]))
            z = (a[:, i] - mu[i]) / sd[i]
            r[f'skew_{c}'] = float(stats.skew(z))
            r[f'kurt_{c}'] = float(stats.kurtosis(z))
        # every between-feature correlation: invariant to per-column standardisation
        C = np.corrcoef(a.T)
        iu = np.triu_indices(len(VAR), 1)
        for k, (i, j) in enumerate(zip(*iu)):
            r[f'corr_{i}_{j}'] = float(C[i, j])
        rows.append(r)
    W = pd.DataFrame(rows)
    W.to_parquet(CACHE)
    print(f'wrote {CACHE}: {W.shape}')
    return W


# ------------------------------------------------------------------- LDA protocol
def _fit(Xt, yt):
    k = len(np.unique(yt))
    return LinearDiscriminantAnalysis(solver='eigen', shrinkage=SHRINKAGE,
                                      priors=np.ones(k) / k).fit(Xt, yt)


def loso_score(X, y, n_repeats=N_REPEATS, seed=0, shuffle=False):
    """Leave-one-session-out accuracy, balanced draw of <=N_PER_MOUSE per mouse."""
    X = np.asarray(X, float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scores = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        yy = rng.permutation(y) if shuffle else y
        hit = 0
        for i in range(len(X)):
            tr = np.setdiff1d(np.arange(len(X)), [i])
            ys = yy[tr]
            idx = []
            for m in np.unique(ys):
                mi = np.where(ys == m)[0]
                idx.extend(rng.choice(mi, min(N_PER_MOUSE, len(mi)), replace=False))
            idx = np.array(idx)
            hit += int(_fit(X[tr][idx], ys[idx]).predict(X[i:i + 1])[0] == yy[i])
        scores.append(hit / len(X))
    return float(np.mean(scores)), float(np.std(scores))


def lab_eta2(X, labs):
    """Mean eta^2 of the lab label over the columns -- the same statistic
    functions.lab_variance_explained reports for the syllable features."""
    X = np.asarray(X, float)
    out = []
    for j in range(X.shape[1]):
        v = X[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 5 or np.std(v[ok]) == 0:
            continue
        g = pd.Series(v[ok]).groupby(np.asarray(labs)[ok])
        ss_b = ((g.mean() - v[ok].mean()) ** 2 * g.size()).sum()
        out.append(ss_b / ((v[ok] - v[ok].mean()) ** 2).sum())
    return float(np.mean(out))


def lab_center(X, labs):
    D = pd.DataFrame(np.asarray(X, float))
    return (D - D.groupby(np.asarray(labs)).transform('mean')).to_numpy()


def icc1(X, mice):
    """ICC(1) of the leading PC of a block: how much of its variance is between mice."""
    from sklearn.decomposition import PCA
    Z = np.nan_to_num(np.asarray(X, float))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    pc = PCA(n_components=1).fit_transform(Z)[:, 0]
    s = pd.Series(pc).groupby(np.asarray(mice))
    k = s.size().mean()
    ms_b = s.mean().var(ddof=1) * k
    ms_w = s.var(ddof=1).mean()
    return float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))


def main():
    W = pd.read_parquet(CACHE) if CACHE.exists() else build_moments()

    import make_embedding as me
    feats, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
    from functions import lab_labels
    labs = pd.Series(list(lab_labels(feats.index, mouse_names=mouse_of)),
                     index=feats.index)

    Wi = W.set_index('session').reindex(feats.index)
    assert Wi['mouse_name'].notna().all(), 'a session has no wavelet file'
    y = pd.factorize(mouse_of.loc[feats.index])[0]
    lab_v = labs.to_numpy()
    mice = mouse_of.loc[feats.index].to_numpy()

    BLOCKS = {
        'SCALE   (discarded: log mean + log sd)': [c for c in W.columns
                                                   if c.startswith(('logmean_', 'logsd_'))],
        '  scale: log MEAN only': [c for c in W.columns if c.startswith('logmean_')],
        '  scale: log SD only': [c for c in W.columns if c.startswith('logsd_')],
        'SHAPE   (kept: skew + kurtosis)': [c for c in W.columns
                                            if c.startswith(('skew_', 'kurt_'))],
        'CORR    (kept: 190 feature correlations)': [c for c in W.columns
                                                     if c.startswith('corr_')],
    }
    print('\n' + '=' * 92)
    print('MOUSE IDENTIFIABILITY BY BLOCK  (leave-one-session-out, 260 sessions, 56 mice, '
          f'chance {1/56:.3f})')
    print('=' * 92)
    print(f'{"block":42s} {"dims":>5s} {"acc":>7s} {"acc|lab":>8s} {"lab eta2":>9s} '
          f'{"lab acc":>8s} {"ICC(PC1)":>9s}')
    results = {}
    for name, cols in BLOCKS.items():
        X = Wi[cols].to_numpy(float)
        Xc = lab_center(X, lab_v)
        acc, _ = loso_score(X, y)
        accc, _ = loso_score(Xc, y)
        lab_y = pd.factorize(lab_v)[0]
        lacc, _ = loso_score(X, lab_y, n_repeats=1)
        results[name] = (acc, accc)
        print(f'{name:42s} {len(cols):5d} {acc:7.3f} {accc:8.3f} {lab_eta2(X, lab_v):9.3f} '
              f'{lacc:8.3f} {icc1(X, mice):9.3f}')

    Xs = np.asarray(feats, float)
    acc, _ = loso_score(Xs, y)
    accc, _ = loso_score(lab_center(Xs, lab_v), y)
    lacc, _ = loso_score(Xs, pd.factorize(lab_v)[0], n_repeats=1)
    print(f'{"SYLLABLES (the current 360 features)":42s} {Xs.shape[1]:5d} {acc:7.3f} '
          f'{accc:8.3f} {lab_eta2(Xs, lab_v):9.3f} {lacc:8.3f} {icc1(Xs, mice):9.3f}')
    syll = (acc, accc)

    scale = Wi[BLOCKS['SCALE   (discarded: log mean + log sd)']].to_numpy(float)
    # STANDARDISE BEFORE CONCATENATING. The LDA objective is affine-invariant, but only
    # when S_W is estimated well; with shrinkage it is NOT, because the shrinkage target
    # is the identity and so the units of each block decide how much each is shrunk.
    def zs(A):
        A = np.nan_to_num(np.asarray(A, float))
        return (A - A.mean(0)) / (A.std(0) + 1e-12)
    Xb = np.hstack([zs(Xs), zs(scale)])
    acc, _ = loso_score(Xb, y)
    accc, _ = loso_score(lab_center(Xb, lab_v), y)
    lacc, _ = loso_score(Xb, pd.factorize(lab_v)[0], n_repeats=1)
    print(f'{"SYLLABLES + SCALE":42s} {Xb.shape[1]:5d} {acc:7.3f} {accc:8.3f} '
          f'{lab_eta2(Xb, lab_v):9.3f} {lacc:8.3f} {icc1(Xb, mice):9.3f}')
    print(f'{"  shuffled labels (syllables)":42s} '
          f'{Xs.shape[1]:5d} {loso_score(Xs, y, n_repeats=1, shuffle=True)[0]:7.3f}')

    print('\nREAD THIS COLUMN-WISE:')
    print('  acc      mouse ID from the block alone')
    print('  acc|lab  the same after centring every feature within its lab. A block that')
    print('           loses most of its accuracy here was identifying the RIG.')
    print('  lab acc  how well the block identifies the LAB (10 classes, chance 0.100)')
    print('  ICC(PC1) between-mouse share of the block\'s leading component')

    print('\n' + '=' * 92)
    print('DOES AMPLITUDE STILL BIAS STATE OCCUPANCY?  (the "shape is expressive" claim)')
    print('=' * 92)
    # paw-state occupancy from the 360 features: after binarize the layout per timestep
    # is [paw 0, paw 2..7, whisk, lick], 9 columns x 40 timesteps.
    per_step, n_paw_cols = 9, 7
    occ = np.zeros((len(feats), n_paw_cols))
    A = np.asarray(feats, float)
    for k in range(n_paw_cols):
        occ[:, k] = A[:, k::per_step].mean(1)
    amp = Wi[[c for c in W.columns if c.startswith('logmean_')]].to_numpy(float).mean(1)
    print('  state (0, 2..7 -- state 1 is the dropped reference):')
    for k in range(n_paw_cols):
        r, p = stats.spearmanr(occ[:, k], amp)
        rc, pc = stats.spearmanr(lab_center(occ[:, [k]], lab_v)[:, 0],
                                 lab_center(amp[:, None], lab_v)[:, 0])
        print(f'    state {[0, 2, 3, 4, 5, 6, 7][k]}  occupancy vs session amplitude: '
              f'rho={r:+.3f} (p={p:.2g})   lab-centred {rc:+.3f} (p={pc:.2g})')
    print('  -> occupancy IS biased by how much the animal moves; the z-score attenuates')
    print('     that link but does not sever it.')


if __name__ == '__main__':
    main()
