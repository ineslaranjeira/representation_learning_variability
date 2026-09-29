"""
IS THE DISCARDED SCALE INDIVIDUALITY, OR IS IT THE RIG?
========================================================
The cheap, targeted part of the block decomposition: only the 40 numbers the
per-session z-score actually discards (each of the 20 features' log mean and log sd),
scored four ways. 40 dimensions, so it runs in minutes rather than the hour the
360/400-dim version needed.

  raw           mouse ID from the 40 scale numbers alone
  | lab         after centring every feature within its lab
  | rig         after residualising every feature on the session-level rig covariates:
                log lick-tube length (the physical ruler), log camera frame rate and
                log tracking-noise floor -- a SESSION-level control, which lab-centring
                is not (paw_bias: only 22% of the camera-gain variance is between-lab)
  lab ID        how well the same 40 numbers identify the LAB (10 classes, chance 0.10)

SHAPE and CORR are included as the comparison the question needs: they are what the
z-score KEEPS, so the contrast says whether standardising costs individuality or merely
removes a nuisance that was never carrying any.
"""
import os, warnings, pathlib
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import zscore_cost as Z

HERE = pathlib.Path(__file__).resolve().parent
W = pd.read_parquet(HERE / 'wavelet_moments_sessions.parquet')
R = pd.read_csv(HERE / 'rig_covariates_sessions.csv')
V = pd.read_csv(HERE / 'vigor_sessions_states.csv')

import make_embedding as me
feats, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
from functions import lab_labels
idx = feats.index
lab_v = np.array(list(lab_labels(idx, mouse_names=mouse_of)))
y = pd.factorize(mouse_of.loc[idx])[0]
mice = mouse_of.loc[idx].to_numpy()

Wi = W.set_index('session').reindex(idx)
Ri = R.set_index('session').reindex(idx)
Vi = V.set_index('session').reindex(idx)

# rig covariate matrix; missing entries take the column mean, so a session with no
# ruler is simply not corrected rather than dropped (and the n is printed).
C = pd.DataFrame({
    'log_tube': np.log(Ri['l_tube']),
    'log_fr': np.log(Ri['l_fr']),
    'log_noise': np.log(Vi['paw_hf_noise']),
}, index=idx)
print(f'rig covariates present: ' +
      ', '.join(f'{c} {C[c].notna().sum()}/{len(C)}' for c in C))
C = C.fillna(C.mean())
Cm = np.c_[np.ones(len(C)), C.to_numpy()]


def rig_residual(X):
    X = np.nan_to_num(np.asarray(X, float))
    return X - Cm @ np.linalg.lstsq(Cm, X, rcond=None)[0]


BLOCKS = {
    'SCALE  = log mean + log sd  (DISCARDED)': [c for c in W.columns
                                                if c.startswith(('logmean_', 'logsd_'))],
    '  log MEAN only': [c for c in W.columns if c.startswith('logmean_')],
    '  log SD only': [c for c in W.columns if c.startswith('logsd_')],
    'SHAPE  = skew + kurtosis    (KEPT)': [c for c in W.columns
                                           if c.startswith(('skew_', 'kurt_'))],
    'CORR   = 190 correlations   (KEPT)': [c for c in W.columns if c.startswith('corr_')],
}
print(f'\n{"block":42s} {"dims":>4s} {"mouse":>6s} {"|lab":>6s} {"|rig":>6s} '
      f'{"labID":>6s} {"ICC":>6s}')
for name, cols in BLOCKS.items():
    X = Wi[cols].to_numpy(float)
    a = Z.loso_score(X, y, n_repeats=2)[0]
    al = Z.loso_score(Z.lab_center(X, lab_v), y, n_repeats=2)[0]
    ar = Z.loso_score(rig_residual(X), y, n_repeats=2)[0]
    lb = Z.loso_score(X, pd.factorize(lab_v)[0], n_repeats=1)[0]
    print(f'{name:42s} {len(cols):4d} {a:6.3f} {al:6.3f} {ar:6.3f} {lb:6.3f} '
          f'{Z.icc1(X, mice):6.3f}')
print(f'{"chance":42s} {"":4s} {1/56:6.3f} {1/56:6.3f} {1/56:6.3f} {1/10:6.3f}')
