"""
DOES SEGMENTING INTO SYLLABLES REMOVE LAB, AND WHAT DOES IT COST IN INDIVIDUALITY?
================================================================================
The case for segmenting has to be made against the UNSEGMENTED signal, built the same way.
So every raw block below is the syllables' own input, with only the clustering taken out:

  SYLLABLES   per session, 4 epochs x 10 bins, session mean of the one-hot state
  RAW-TIME    per session, 4 epochs x 10 bins, session mean of the CONTINUOUS signal
              the states were fit to -- paw wavelet log-amplitudes (20 features), and for
              the full comparison whisker motion energy and lick count
  RAW-SESSION whole-session statistics: log mean/SD, shape (skew, kurtosis), correlations

Time binning follows 5_syllable_generation: each (trial, epoch) split into 10 equal bins,
trials kept only when all 4 epochs are present, then averaged over trials with equal
weight per trial. The raw bins take the MEAN of the frames in a bin; the syllables take
the MODE state, then one-hot.

RAW-TIME comes in two normalisations, because the question has two parts:
  global     one scale for everyone; between-session level (rig included) stays in
  session z  each feature z-scored by that session's full-session mean/SD -- what the
             current states see before clustering, so this isolates the clustering itself

Everything is rebuilt from current data (states_files, paw_wavelets, SYLLABLE_FILE below and
its cohort, the 4Sep26 QC sheet). Nothing is read from vigor/lda caches.

PROTOCOL: zscore_cost.py's, on the LDA cohort (260 sessions, 56 mice, 10 labs):
leave-one-session-out mouse ID with <=3 sessions per mouse, shrinkage LDA at 0.5. Every
block's columns are standardised ACROSS sessions before the LDA (the shrinkage target is
the identity, so column units would otherwise decide how much each column is shrunk);
this does not remove any between-session difference.

  mouse       mouse ID                                         chance 1/56
  mouse|lab   mouse ID after centring every feature within its lab   = individuality that is not lab
  lab eta2    mean over columns of the lab share of variance
  lab LOSO    lab ID, leave-one-session-out (can lean on the mouse's other sessions)
  lab LOMO    lab ID from a held-out MOUSE                     chance ~0.10
  ICC         between-mouse share of the block's first PC
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'lda'))
import zscore_cost as Z                                     # noqa: E402  the LDA protocol
import make_embedding as me                                 # noqa: E402  the LDA cohort
from functions import lab_labels                            # noqa: E402
from compare_pipelines import lomo_lab_acc                  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DATA = ROOT / 'data'
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_BINS = 10
BANDS = ['0.5', '1.0', '2.0', '4.0', '8.0']
PAW = [f'{p}_{a}{b}' for p in ('l_paw', 'r_paw') for a in 'xy' for b in BANDS]
EPS = 0.1                         # the same log floor as 3.2_wavelet_subsample_uniform
CACHE = HERE / 'raw_vs_syllables_features.npz'
# THE SYLLABLES COMPARED. Every script in this comparison (raw_velocity_vs_syllables.py,
# ci_segmentation_vs_raw.py) takes it from here. It also sets the cohort, through
# make_embedding.build_design_matrix. The paw-only and whisk+lick blocks are cut out of
# the same 360 features, so all syllable rows come from one file.
SYLLABLE_FILE = str(ROOT / 'data' / '8_k_10_bin_syllables_02-10-2026')


def paw_only(S360):
    """The 360 features' paw columns: per timestep [paw 0, paw 2..7], i.e. 7 of 9 -> 280."""
    return np.hstack([S360[:, t * 9: t * 9 + 7] for t in range(40)])


def whisk_lick_only(S360):
    """The 360 features' whisk and lick columns: 2 of 9 per timestep -> 80."""
    return np.hstack([S360[:, t * 9 + 7: t * 9 + 9] for t in range(40)])


def time_resolved(frame, cols):
    """(frames with trial_id, broader_label, cols) -> 4 x 10 x len(cols), equal weight per trial."""
    f = frame.dropna(subset=['trial_id'])
    f = f[f['broader_label'].isin(EPOCHS)]
    full = f.groupby('trial_id')['broader_label'].nunique()
    f = f[f['trial_id'].isin(full.index[full == len(EPOCHS)])].copy()
    g = f.groupby(['trial_id', 'broader_label'], sort=False)
    f['bin'] = (g.cumcount() * N_BINS // g['broader_label'].transform('size')).to_numpy()
    per_trial = f.groupby(['trial_id', 'broader_label', 'bin'])[cols].mean()
    sess = per_trial.groupby(level=['broader_label', 'bin']).mean()
    idx = pd.MultiIndex.from_product([EPOCHS, range(N_BINS)], names=['broader_label', 'bin'])
    return sess.reindex(idx).to_numpy()                      # (40, len(cols))


def session_block(eid, mouse):
    """One session's raw blocks. Runs in a worker."""
    r = {}
    sf = DATA / 'states_files' / f'8_states_file_{eid}_{mouse}'
    wf = DATA / 'paw_wavelets' / f'paw_vel_wavelets_{eid}_{mouse}'
    if not (sf.exists() and wf.exists()):
        return None
    S = pd.read_parquet(sf, columns=['Bin', 'trial_id', 'broader_label',
                                     'whisker_me', 'Lick count'])
    W = pd.read_parquet(wf, columns=['Bin'] + PAW)
    A = W[PAW].to_numpy(float)
    ok = np.isfinite(A).all(1)
    raw = A[ok]

    # whole-session statistics of the raw amplitudes (as zscore_cost.build_moments)
    mu, sd = raw.mean(0), raw.std(0)
    r['scale'] = np.r_[np.log(mu), np.log(sd)]
    zr = (raw - mu) / sd
    r['shape'] = np.r_[stats.skew(zr, axis=0), stats.kurtosis(zr, axis=0)]
    r['corr'] = np.corrcoef(raw.T)[np.triu_indices(len(PAW), 1)]

    # continuous signals per frame, in the working scale
    L = np.full_like(A, np.nan)
    L[ok] = np.log(raw + EPS)
    Lz = np.full_like(A, np.nan)
    Lz[ok] = (L[ok] - L[ok].mean(0)) / L[ok].std(0)
    Wd = pd.DataFrame(np.hstack([L, Lz]), columns=PAW + [c + '_z' for c in PAW])
    Wd['_k'] = np.round(W['Bin'].to_numpy(float), 6)
    S['_k'] = np.round(S['Bin'].to_numpy(float), 6)
    wm = np.log(S['whisker_me'].to_numpy(float))
    S['whisk'] = wm
    S['whisk_z'] = (wm - np.nanmean(wm)) / np.nanstd(wm)
    lk = S['Lick count'].to_numpy(float)
    S['lick'] = lk
    S['lick_z'] = (lk - np.nanmean(lk)) / np.nanstd(lk)
    D = S.merge(Wd, on='_k', how='left')

    paw_g = time_resolved(D, PAW)
    paw_z = time_resolved(D, [c + '_z' for c in PAW])
    wl_g = time_resolved(D, ['whisk', 'lick'])
    wl_z = time_resolved(D, ['whisk_z', 'lick_z'])
    r['paw_global'] = paw_g.ravel()
    r['paw_sessz'] = paw_z.ravel()
    r['wl_global'] = wl_g.ravel()
    r['wl_sessz'] = wl_z.ravel()
    return r


def build(sessions):
    res = Parallel(n_jobs=18)(delayed(session_block)(e, m) for e, m in sessions)
    out = {k: {} for k in ('paw_global', 'paw_sessz', 'wl_global', 'wl_sessz',
                           'scale', 'shape', 'corr')}
    for (eid, _), r in zip(sessions, res):
        if r is not None:
            for k, v in r.items():
                out[k][eid] = v
    return out


def zs(A):
    A = np.asarray(A, float)
    A = np.where(np.isfinite(A), A, np.nanmean(A, 0))       # a missing bin -> column mean
    return (A - A.mean(0)) / (A.std(0) + 1e-12)


def main():
    tags = sorted(f[len('paw_vel_wavelets_'):] for f in os.listdir(DATA / 'paw_wavelets')
                  if f.startswith('paw_vel_wavelets_'))
    sessions = [(t[:36], t[37:]) for t in tags]
    if CACHE.exists():
        raw = np.load(CACHE, allow_pickle=True)['raw'].item()
    else:
        raw = build(sessions)
        np.savez(CACHE, raw=raw)
        print(f'cached {CACHE}')

    syl, mouse_of = me.build_design_matrix(SYLLABLE_FILE)
    idx = [s for s in syl.index if all(s in raw[k] for k in raw)]
    mice = mouse_of.loc[idx].to_numpy()
    labs = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx])))
    y = pd.factorize(mice)[0]
    print(f'\nsyllables: {SYLLABLE_FILE}')
    print(f'cohort: {len(idx)} sessions, {len(set(mice))} mice, {len(set(labs))} labs')

    S360 = syl.loc[idx].to_numpy(float)
    R = {k: np.vstack([raw[k][s] for s in idx]) for k in raw}
    BLOCKS = [
        ('SYLLABLES', None),
        ('  paw + whisk + lick (the 360 the LDA uses)', S360),
        ('  paw only', paw_only(S360)),
        ('  whisk + lick only', whisk_lick_only(S360)),
        ('RAW, TIME-RESOLVED (same 4 x 10 bins, no clustering)', None),
        ('  paw + whisk + lick, global', np.hstack([R['paw_global'], R['wl_global']])),
        ('  paw + whisk + lick, session z', np.hstack([R['paw_sessz'], R['wl_sessz']])),
        ('  paw only, global', R['paw_global']),
        ('  paw only, session z', R['paw_sessz']),
        ('  whisk + lick only, global', R['wl_global']),
        ('  whisk + lick only, session z', R['wl_sessz']),
        ('RAW, WHOLE-SESSION STATISTICS (paw amplitudes)', None),
        ('  scale: log mean + log SD', R['scale']),
        ('  shape: skew + kurtosis', R['shape']),
        ('  between-feature correlations', R['corr']),
    ]

    print('\n' + '=' * 104)
    print(f'MOUSE vs LAB, SEGMENTED vs RAW   (LOSO, {len(idx)} sessions; chance: mouse '
          f'{1 / len(set(mice)):.3f}, lab ~0.100)')
    print('=' * 104)
    print(f'{"block":52s} {"dims":>5s} {"mouse":>6s} {"mouse|lab":>9s} {"lab eta2":>8s} '
          f'{"lab LOSO":>8s} {"lab LOMO":>8s} {"ICC":>6s}')
    lab_y = pd.factorize(labs)[0]
    def score(name, X):
        X = zs(X)
        acc, _ = Z.loso_score(X, y)
        accc, _ = Z.loso_score(Z.lab_center(X, labs), y)
        lacc, _ = Z.loso_score(X, lab_y, n_repeats=1)
        return (f'{name:52s} {X.shape[1]:5d} {acc:6.3f} {accc:9.3f} {Z.lab_eta2(X, labs):8.3f} '
                f'{lacc:8.3f} {lomo_lab_acc(X, labs, mice):8.3f} {Z.icc1(X, mice):6.3f}')

    rows = Parallel(n_jobs=18)(delayed(score)(n, X) for n, X in BLOCKS if X is not None)
    rows = iter(rows)
    for name, X in BLOCKS:
        print(name if X is None else next(rows))

if __name__ == '__main__':
    main()
