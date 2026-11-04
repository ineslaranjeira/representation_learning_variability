"""
SYLLABLE SPACE, FIRST k FEATURES -- the no-transformation reference for fig_lda's "first k dimensions" panel
==========================================================================================================
fig_lda's panel adds dimensions in a meaningful order (LD1, LD2, ... or PC1, PC2, ...). The syllable
features have no order, so here they are added in RANDOM orders (N_ORDERS of them, averaged): mouse ID
from the first k z-scored syllable features, nearest centroid -- the "syllable space" score of
lda_shrinkage_figures' panel b, restricted to k features. At k = 360 it is that score.

Leave-one-session-out as lda_shrinkage_figures' loso_fold: z-scored by the mean / SD of ALL training
sessions, centroids from <= 3 training sessions per mouse. Features: the paper's 360 (identical to
data/syllable_sweeps/bins/features_10.pqt, i.e. the LDA notebook's X).

Output: figure_generation/cache/syllable_space_curve.npz  (curve: accuracy per k, by order)
"""
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import syllable_sweeps as ss

OUT = ss.PI / 'figure_generation' / 'cache' / 'syllable_space_curve.npz'
N_ORDERS, N_REPEATS = 50, 3


def fold(X, y, orders, repeat, i):
    rng = ss.fold_rng(0, repeat, i)
    tr = np.setdiff1d(np.arange(len(X)), [i])
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    sel = tr[ss.balanced_draw(y[tr], rng)]
    Z = (X[sel] - mu) / sd
    classes = np.unique(y[sel])
    M = np.stack([Z[y[sel] == c].mean(0) for c in classes])
    d2 = (M - (X[i] - mu) / sd) ** 2                                    # classes x features
    return np.stack([classes[np.argmin(np.cumsum(d2[:, o], axis=1), axis=0)] == y[i] for o in orders])


if __name__ == '__main__':
    sf, mouse = ss.features('bins', 10)
    X, y = sf.to_numpy(float), pd.factorize(mouse.to_numpy())[0]
    orders = np.stack([np.random.default_rng([7, j]).permutation(X.shape[1]) for j in range(N_ORDERS)])
    hits = Parallel(n_jobs=-1)(delayed(fold)(X, y, orders, r, i) for r in range(N_REPEATS) for i in range(len(X)))
    curve = np.mean(hits, axis=0)                                       # orders x k
    np.savez(OUT, curve=curve, n_orders=N_ORDERS, n_repeats=N_REPEATS, n_sessions=len(X))
    print(f'k = 1, 5, 10, 45, 360: {np.round(curve.mean(0)[[0, 4, 9, 44, -1]], 3)}  -> {OUT.name}')
