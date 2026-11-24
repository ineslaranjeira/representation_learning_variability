"""
RAW-SIGNAL LDA EMBEDDING -- does an LDA on the unsegmented signal find the same LD1?
====================================================================================
The production embedding (mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026) is fit on the
360 syllable features. Here the same fit -- equal-mouse-weight shrinkage LDA at 0.5
(lda_shrinkage_core.fit_mouse_weighted), columns standardised across sessions, coordinates
centred on the mean of the mouse means -- is applied to raw blocks built the same way
(4 epochs x 10 bins, trial-averaged), read from this folder's caches:

  spd_wl_global   speed per paw (log |v|, L/R) + log whisker ME + lick count     160 dims
  spd_wl_sessz    the same, each signal z-scored within its session             160 dims
  wav_wl_global   paw wavelet log-amplitudes (20) + whisker ME + lick           880 dims

The first is the interpretable, dimension-matched comparison: every column is a named signal at
a named time in the trial.

Writes (this folder):
  raw_lda_scores.csv     session, mouse_name, prod_LD1..5, <block>_LD1..5  (LD1 sign aligned to
                         production at the mouse level)
  raw_lda_loadings.csv   structure coefficients: r(feature, LD1) across sessions, per block, with
                         channel / epoch / bin -- the readable profile of each LD1
Prints r(raw LD1, production LD1) at session and mouse level, best-matching raw LD for each
production LD, and the between-mouse share of each LD1.

Run on the analysis machine (needs raw_velocity_features.npz and raw_vs_syllables_features.npz,
built by raw_velocity_vs_syllables.py / raw_vs_syllables.py):
    python raw_lda_embedding.py
Then copy raw_lda_scores.csv to the Mac for the lab_controls correlates.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # paper-individuality/
sys.path.insert(0, str(ROOT / 'learning_individuality'))
from lda_shrinkage_core import fit_mouse_weighted        # noqa: E402

SHRINKAGE = 0.5
N_LD = 5
LDA_FILE = ROOT / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_BINS = 10
BANDS = ['0.5', '1.0', '2.0', '4.0', '8.0']
PAW = [f'{p}_{a}{b}' for p in ('l_paw', 'r_paw') for a in 'xy' for b in BANDS]


def load_cache(name):
    return np.load(HERE / name, allow_pickle=True)['raw'].item()


def feature_names(channels):
    """time_resolved(...).ravel() is (epoch, bin) major, channel minor."""
    return [(ep, b, ch) for ep in EPOCHS for b in range(N_BINS) for ch in channels]


def zs(A):
    A = np.asarray(A, float)
    A = np.where(np.isfinite(A), A, np.nanmean(A, 0))
    return (A - A.mean(0)) / (A.std(0) + 1e-12)


def embed(X, mice):
    X = zs(X)
    y = pd.factorize(mice, sort=True)[0]
    W, _, ev = fit_mouse_weighted(X, y, SHRINKAGE)
    Z = X @ W[:, :N_LD]
    M = pd.DataFrame(Z).groupby(y).mean()
    return Z - M.mean(0).to_numpy(), X, ev


def main():
    lda = pd.read_pickle(LDA_FILE)
    vel = load_cache('raw_velocity_features.npz')
    wav = load_cache('raw_vs_syllables_features.npz')
    sess = [s for s in lda['session']
            if s in vel['spd_wl_global'] and s in wav['paw_global'] and s in wav['wl_global']]
    lda = lda.set_index('session').loc[sess]
    mice = lda['mouse_name'].to_numpy()
    print(f'cohort: {len(sess)} of {len(lda)} production sessions, {len(set(mice))} mice')

    spd = ['speed L paw', 'speed R paw', 'whisker ME', 'lick']
    blocks = {
        'spd_wl_global': (np.vstack([vel['spd_wl_global'][s] for s in sess]), feature_names(spd)),
        'spd_wl_sessz': (np.vstack([vel['spd_wl_sessz'][s] for s in sess]), feature_names(spd)),
        'wav_wl_global': (np.hstack([np.vstack([wav['paw_global'][s] for s in sess]),
                                     np.vstack([wav['wl_global'][s] for s in sess])]),
                          feature_names(PAW) + feature_names(['whisker ME', 'lick'])),
    }

    prod = lda[list(range(N_LD))].to_numpy(float)
    out = pd.DataFrame({'session': sess, 'mouse_name': mice})
    for k in range(N_LD):
        out[f'prod_LD{k + 1}'] = prod[:, k]
    mm = lambda v: pd.Series(v).groupby(mice).mean()         # noqa: E731
    load_rows = []
    for name, (X, names) in blocks.items():
        assert X.shape[1] == len(names), (name, X.shape, len(names))
        Z, Xz, ev = embed(X, mice)
        sign = np.sign(np.corrcoef(mm(Z[:, 0]), mm(prod[:, 0]))[0, 1])
        Z[:, 0] *= sign
        for k in range(N_LD):
            out[f'{name}_LD{k + 1}'] = Z[:, k]
        r_s = np.corrcoef(Z[:, 0], prod[:, 0])[0, 1]
        r_m = np.corrcoef(mm(Z[:, 0]), mm(prod[:, 0]))[0, 1]
        share = mm(Z[:, 0]).var(ddof=0) / pd.Series(Z[:, 0]).var(ddof=0)
        # which raw LD best matches each production LD (mouse level, |r|)
        C = np.array([[abs(np.corrcoef(mm(Z[:, i]), mm(prod[:, j]))[0, 1]) for i in range(N_LD)]
                      for j in range(N_LD)])
        print(f'\n{name}  ({X.shape[1]} dims)')
        print(f'  r(raw LD1, production LD1): sessions {r_s:.3f}, mice {r_m:.3f}; '
              f'between-mouse share of raw LD1 {share:.2f}')
        print('  |r| mouse level, rows = production LD1..5, cols = raw LD1..5')
        print(pd.DataFrame(C, index=[f'prod LD{j + 1}' for j in range(N_LD)],
                           columns=[f'raw LD{i + 1}' for i in range(N_LD)]).round(2).to_string())
        sc = np.array([np.corrcoef(Xz[:, f], Z[:, 0])[0, 1] for f in range(Xz.shape[1])])
        for (ep, b, ch), r in zip(names, sc):
            load_rows.append(dict(block=name, epoch=ep, bin=b, channel=ch, r_LD1=r))
        top = pd.DataFrame(load_rows[-len(names):]).groupby('channel')['r_LD1'].agg(['mean', 'min', 'max'])
        print('  structure coefficients by channel (r of feature with raw LD1):')
        print(top.round(2).to_string())
    out.to_csv(HERE / 'raw_lda_scores.csv', index=False)
    pd.DataFrame(load_rows).to_csv(HERE / 'raw_lda_loadings.csv', index=False)
    print(f'\nwrote {HERE / "raw_lda_scores.csv"} and raw_lda_loadings.csv')


if __name__ == '__main__':
    main()
