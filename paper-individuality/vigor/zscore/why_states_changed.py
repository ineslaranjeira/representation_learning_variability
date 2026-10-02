"""
WHY DID THE MODERATE PAIR (3/4) LOSE ITS LATERALITY?
Two things changed between production and paw_states_uniform_raw_sessionz:
  (1) the training SAMPLE: density-weighted (p ~ t-SNE KDE) -> uniform
  (2) the training STATS: z-scored by the subsample's own stats -> full-session stats
This separates them, and asks whether the moderate region's tiling is stable at all:
  A  density subsample + subsample stats        (= production)
  B  density subsample + full-session stats     (fixes (2) only)
  C  uniform subsample + full-session stats     (= the new states), k-means seeds 2024, 0..4
Each: PCA 95% -> KMeans(8). Per state, sorted by vigor: vigor, left-right gap, share.
"""
import os, pathlib
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

DATA = pathlib.Path(__file__).resolve().parents[2] / 'data'
ALL = ['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y'] + [f'{p}_{a}{f}' for p in ['l_paw', 'r_paw']
       for a in 'xy' for f in ['0.5', '1.0', '2.0', '4.0', '8.0', '16.0', '32.0']]
USE = [f'{p}_{a}{f}' for p in ['l_paw', 'r_paw'] for a in 'xy' for f in ['0.5', '1.0', '2.0', '4.0', '8.0']]
UI = [ALL.index(c) for c in USE]

uni = {f[:-4]: np.load(DATA / 'paw_subsampled_wavelets_uniform' / f)
       for f in sorted(os.listdir(DATA / 'paw_subsampled_wavelets_uniform'))}
den = {f[:-4]: np.load(DATA / 'paw_subsampled_wavelets18ago' / f)[:, UI]
       for f in sorted(os.listdir(DATA / 'paw_subsampled_wavelets18ago'))}
tags = sorted(set(uni) & set(den))
full = {t: (uni[t]['mean_raw'], uni[t]['sd_raw']) for t in tags}     # full-session stats

X = {
    'A density + subsample stats': np.vstack([stats.zscore(den[t], axis=0) for t in tags]),
    'B density + full stats': np.vstack([(den[t] - full[t][0]) / full[t][1] for t in tags]),
    'C uniform + full stats': np.vstack([(uni[t]['sub'] - full[t][0]) / full[t][1] for t in tags]),
}


def solve(name, seed):
    Z = X[name]
    pca = PCA(20).fit(Z)
    n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    lab = KMeans(8, random_state=seed).fit_predict(pca.transform(Z)[:, :n])
    rows = []
    for c in range(8):
        m = Z[lab == c]
        rows.append((m.mean(), m[:, :10].mean() - m[:, 10:].mean(), len(m) / len(Z)))
    return name, seed, sorted(rows)


def main():
    jobs = [(k, 2024) for k in X] + [('C uniform + full stats', s) for s in range(5)] \
         + [('A density + subsample stats', s) for s in range(3)]
    res = Parallel(n_jobs=5)(delayed(solve)(k, s) for k, s in jobs)
    print('per state, sorted by vigor:  vigor / L-R gap / share.   |gap| > 0.3 marked *')
    for name, seed, rows in res:
        cells = '  '.join(f'{v:+.2f}/{g:+.2f}{"*" if abs(g) > 0.3 else " "}/{s:.0%}' for v, g, s in rows)
        nlat = sum(abs(g) > 0.3 for _, g, _ in rows)
        print(f'{name:30s} seed {seed:4d}  lateralised states: {nlat}   {cells}')


if __name__ == '__main__':
    main()
