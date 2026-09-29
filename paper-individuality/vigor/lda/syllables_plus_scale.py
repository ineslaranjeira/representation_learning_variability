"""
THE PRACTICAL ALTERNATIVE: KEEP THE Z-SCORE, ADD THE SCALE BACK AS FEATURES
===========================================================================
The counterfactual showed that removing the per-session z-score does NOT buy mouse
identification (0.756 -> 0.744): the k=8 nearest-centroid step is a bottleneck that
converts almost none of the discarded scale into occupancy differences.

But the scale itself identifies the mouse at 0.690 from 40 numbers, and survives the
rig controls. So the question becomes whether handing those 40 numbers to the LDA
DIRECTLY -- alongside the syllables, rather than through the clustering -- does what
refitting the clustering could not.

Both blocks are standardised before concatenation: the LDA objective is affine-
invariant only when S_W is well estimated, and with shrinkage the target is the
identity, so raw units would decide how much each block gets shrunk.

N_REPEATS=1 here: this is a comparison between two feature sets under one fixed
setting, and the repeats only average over the random balanced draw.
"""
import warnings, pathlib
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import zscore_cost as Z

HERE = pathlib.Path(__file__).resolve().parent
W = pd.read_parquet(HERE / 'wavelet_moments_sessions.parquet')

import make_embedding as me
feats, mouse_of = me.build_design_matrix(me.SYLLABLE_FILE)
from functions import lab_labels
idx = feats.index
lab_v = np.array(list(lab_labels(idx, mouse_names=mouse_of)))
y = pd.factorize(mouse_of.loc[idx])[0]
mice = mouse_of.loc[idx].to_numpy()

scale_cols = [c for c in W.columns if c.startswith(('logmean_', 'logsd_'))]
scale = W.set_index('session').reindex(idx)[scale_cols].to_numpy(float)
syll = np.asarray(feats, float)


def zs(A):
    A = np.nan_to_num(np.asarray(A, float))
    return (A - A.mean(0)) / (A.std(0) + 1e-12)


SETS = {
    'SYLLABLES (current)': zs(syll),
    'SYLLABLES + SCALE': np.hstack([zs(syll), zs(scale)]),
    'SCALE only': zs(scale),
}
print(f'\n{"feature set":26s} {"dims":>5s} {"mouse":>7s} {"mouse|lab":>10s} '
      f'{"lab eta2":>9s} {"lab ID":>7s}')
for name, X in SETS.items():
    a = Z.loso_score(X, y, n_repeats=1)[0]
    al = Z.loso_score(Z.lab_center(X, lab_v), y, n_repeats=1)[0]
    lb = Z.loso_score(X, pd.factorize(lab_v)[0], n_repeats=1)[0]
    print(f'{name:26s} {X.shape[1]:5d} {a:7.3f} {al:10.3f} '
          f'{Z.lab_eta2(X, lab_v):9.3f} {lb:7.3f}', flush=True)
print(f'{"chance":26s} {"":5s} {1/56:7.3f} {1/56:10.3f} {"":9s} {1/10:7.3f}')
