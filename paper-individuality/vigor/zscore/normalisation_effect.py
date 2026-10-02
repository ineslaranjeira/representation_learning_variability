"""
HOW MUCH DOES THE TRAIN/LABEL NORMALISATION MISMATCH CHANGE THE STATES?  (K = 8, n_init = 10)
  A  production recipe: density subsample, z-scored by ITS OWN stats; labelling uses full stats
  B  same subsample, z-scored by FULL-SESSION stats (the fix, and nothing else)
  C  uniform subsample, full-session stats (the new pipeline)
All three models then label the SAME frames (every session's uniform 2,000, full-session
z-scored, i.e. exactly what labelling sees). Reported: ARI between models, and the fraction of
frames whose state changes after optimal (Hungarian) matching of state numbers.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import why_states_changed as W


def fit(Z):
    pca = PCA(20).fit(Z)
    n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    km = KMeans(8, random_state=2024, n_init=10).fit(pca.transform(Z)[:, :n])
    return lambda F: cdist(pca.transform(F)[:, :n], km.cluster_centers_).argmin(1)


def changed(a, b):
    C = np.zeros((8, 8))
    np.add.at(C, (a, b), 1)
    r, c = linear_sum_assignment(-C)
    return 1 - C[r, c].sum() / C.sum()


if __name__ == '__main__':
    models = {k[0]: fit(W.X[k]) for k in W.X}
    F = W.X['C uniform + full stats']                    # what labelling sees
    L = {k: m(F) for k, m in models.items()}
    for a, b, what in [('A', 'B', 'normalisation fix only'), ('B', 'C', 'sampling fix only'),
                       ('A', 'C', 'both (production -> new)')]:
        print(f'{a} vs {b}  ({what:26s}):  ARI {adjusted_rand_score(L[a], L[b]):.3f}   '
              f'frames changing state {changed(L[a], L[b]):.1%}')
