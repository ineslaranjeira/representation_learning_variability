"""
DOES ADDING RAW (UN-Z-SCORED) WHEEL AMPLITUDE IMPROVE THE INDIVIDUALITY FEATURES?
=================================================================================
Adding the PAW scale block bought +0.004 -- it was redundant, because the syllables
are built from the paw and already encode much of its amplitude (ridge CV R2 = 0.23).

The wheel is the better candidate on two counts:
  * it is NOT in the syllables at all (they are paw + whisk + lick), so it is new
    information rather than a re-encoding of what is already there;
  * it comes from a rotary encoder, not a camera, so the per-session pixel-scale and
    smoother-attenuation objections do not apply. Measured: wheel amplitude vs the
    lick-tube ruler r = -0.042 (p = 0.49), against the paw's +0.077.

It is not independent of the paw, though: log wheel amplitude and log paw amplitude
correlate rho = +0.634 across sessions (+0.589 across mice), so some of it is the same
trait seen through a different sensor. That is exactly what the incremental test below
is for.
"""
import warnings, pathlib
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import zscore_cost as Z

HERE = pathlib.Path(__file__).resolve().parent
P = pd.read_parquet(HERE / 'wavelet_moments_sessions.parquet')
Wh = pd.read_parquet(HERE / 'wheel_moments_sessions.parquet')

import make_embedding as me
feats, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
from functions import lab_labels
idx = feats.index
lab_v = np.array(list(lab_labels(idx, mouse_names=mouse_of)))
y = pd.factorize(mouse_of.loc[idx])[0]
mice = mouse_of.loc[idx].to_numpy()

paw_cols = [c for c in P.columns if c.startswith(('logmean_', 'logsd_'))]
wh_cols = [c for c in Wh.columns if c.startswith(('wlogmean_', 'wlogsd_'))]
paw = P.set_index('session').reindex(idx)[paw_cols].to_numpy(float)
wheel = Wh.set_index('session').reindex(idx)[wh_cols].to_numpy(float)
syll = np.asarray(feats, float)
print(f'wheel scale coverage: {np.isfinite(wheel).all(1).sum()}/{len(idx)} sessions, '
      f'{len(wh_cols)} features')


def zs(A):
    A = np.nan_to_num(np.asarray(A, float))
    return (A - A.mean(0)) / (A.std(0) + 1e-12)


SETS = {
    'SYLLABLES (current)': zs(syll),
    'SYLLABLES + WHEEL scale': np.hstack([zs(syll), zs(wheel)]),
    'SYLLABLES + PAW scale': np.hstack([zs(syll), zs(paw)]),
    'SYLLABLES + BOTH scales': np.hstack([zs(syll), zs(wheel), zs(paw)]),
    'WHEEL scale only': zs(wheel),
    'PAW scale only': zs(paw),
}
print(f'\n{"feature set":28s} {"dims":>5s} {"mouse":>7s} {"mouse|lab":>10s} '
      f'{"lab eta2":>9s} {"lab ID":>7s} {"ICC":>7s}')
for name, X in SETS.items():
    a = Z.loso_score(X, y, n_repeats=1)[0]
    al = Z.loso_score(Z.lab_center(X, lab_v), y, n_repeats=1)[0]
    lb = Z.loso_score(X, pd.factorize(lab_v)[0], n_repeats=1)[0]
    print(f'{name:28s} {X.shape[1]:5d} {a:7.3f} {al:10.3f} '
          f'{Z.lab_eta2(X, lab_v):9.3f} {lb:7.3f} {Z.icc1(X, mice):7.3f}', flush=True)
print(f'{"chance":28s} {"":5s} {1/56:7.3f} {1/56:10.3f} {"":9s} {1/10:7.3f}')
