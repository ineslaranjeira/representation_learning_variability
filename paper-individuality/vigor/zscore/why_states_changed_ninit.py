"""Follow-up to why_states_changed.py: the same fits with n_init = 10 (the pre-1.4 sklearn
default, and what rerun_no_zscore.py uses), reporting inertia so the tilings can be ranked."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import why_states_changed as W


def solve(name, seed, n_init):
    Z = W.X[name]
    pca = PCA(20).fit(Z)
    n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    km = KMeans(8, random_state=seed, n_init=n_init).fit(pca.transform(Z)[:, :n])
    rows = sorted((Z[km.labels_ == c].mean(), Z[km.labels_ == c][:, :10].mean()
                   - Z[km.labels_ == c][:, 10:].mean(), (km.labels_ == c).mean()) for c in range(8))
    return name, seed, n_init, km.inertia_, rows


def main():
    jobs = [('C uniform + full stats', s, 10) for s in (2024, 0, 1, 2)] + \
           [('A density + subsample stats', s, 10) for s in (2024, 0)] + \
           [('C uniform + full stats', s, 1) for s in (2024, 0)]
    res = Parallel(n_jobs=4)(delayed(solve)(*j) for j in jobs)
    print('per state, sorted by vigor:  vigor / L-R gap / share.   |gap| > 0.3 marked *')
    for name, seed, ni, inertia, rows in res:
        cells = '  '.join(f'{v:+.2f}/{g:+.2f}{"*" if abs(g) > 0.3 else " "}' for v, g, s in rows)
        print(f'{name:28s} seed {seed:4d} n_init {ni:2d}  inertia {inertia:.5e}  '
              f'lateralised {sum(abs(g) > 0.3 for _, g, _ in rows)}   {cells}')


if __name__ == '__main__':
    main()
