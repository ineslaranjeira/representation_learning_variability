"""Reproduce the k=8 paw clustering of 3.3_wavelet_clusters.ipynb (read-only) and describe
each state's wavelet signature under the 19-Ago-2026 fix_mapping."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE = '/home/ines/repositories/representation_learning_variability/paper-individuality/data/'
var_init = ['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y',
            'l_paw_x0.5','l_paw_x1.0','l_paw_x2.0','l_paw_x4.0','l_paw_x8.0','l_paw_x16.0','l_paw_x32.0',
            'l_paw_y0.5','l_paw_y1.0','l_paw_y2.0','l_paw_y4.0','l_paw_y8.0','l_paw_y16.0','l_paw_y32.0',
            'r_paw_x0.5','r_paw_x1.0','r_paw_x2.0','r_paw_x4.0','r_paw_x8.0','r_paw_x16.0','r_paw_x32.0',
            'r_paw_y0.5','r_paw_y1.0','r_paw_y2.0','r_paw_y4.0','r_paw_y8.0','r_paw_y16.0','r_paw_y32.0']
FRQ = ['0.5','1.0','2.0','4.0','8.0']
var_interest = [f'{p}_paw_{ax}{f}' for p in 'lr' for ax in 'xy' for f in FRQ]
# match notebook ordering exactly: l_x(5), l_y(5), r_x(5), r_y(5)
var_interest = ([f'l_paw_x{f}' for f in FRQ] + [f'l_paw_y{f}' for f in FRQ] +
                [f'r_paw_x{f}' for f in FRQ] + [f'r_paw_y{f}' for f in FRQ])

ss = np.load(BASE + 'session_zscored_supersession_wavelets_paw08-19-2026')
df = pd.DataFrame(ss, columns=var_init)
X = np.array(stats.zscore(np.array(df[var_interest]), axis=0))
print('supersession', X.shape)

pca = PCA(n_components=min(20, len(var_interest)))
Xp = pca.fit_transform(X)
cum = np.cumsum(pca.explained_variance_ratio_)
k_opt = min(len(var_interest), np.where(cum >= 0.95)[0][0] + 1)
print('PCA components retained (95% var):', k_opt)
Xp = Xp[:, :k_opt]

km = KMeans(n_clusters=8, random_state=2024).fit(Xp)
raw = km.predict(Xp)

fix_mapping = {0:0, 1:2, 2:1, 3:6, 4:7, 5:5, 6:4, 7:3}   # 19 Ago 2026
new = np.vectorize(fix_mapping.get)(raw)

prof = pd.DataFrame(X, columns=var_interest)
prof['state'] = new
m = prof.groupby('state')[var_interest].mean()
occ = pd.Series(new).value_counts(normalize=True).sort_index()

pd.set_option('display.width', 250, 'display.max_columns', 50)
print('\n=== occupancy (fraction of supersession frames) ===')
print((occ * 100).round(2).to_string())

print('\n=== mean z-scored wavelet power per state (rows = remapped state) ===')
print(m.round(2).to_string())

print('\n=== summary per state ===')
lx = [f'l_paw_x{f}' for f in FRQ]; ly = [f'l_paw_y{f}' for f in FRQ]
rx = [f'r_paw_x{f}' for f in FRQ]; ry = [f'r_paw_y{f}' for f in FRQ]
FN = np.array([0.5, 1, 2, 4, 8])
floor = m[var_interest].to_numpy().min()
rows = []
for s_ in sorted(m.index):
    r = m.loc[s_]
    L, R = r[lx + ly].mean(), r[rx + ry].mean()
    def centroid(cx, cy):
        w = (r[cx].to_numpy() + r[cy].to_numpy()) / 2 - floor
        return float((w * FN).sum() / w.sum())
    rows.append(dict(state=s_, occ_pct=occ[s_] * 100, overall=r[var_interest].mean(),
                     left=L, right=R, LI=(L - R) / (abs(L) + abs(R)),
                     centroid_L_Hz=centroid(lx, ly), centroid_R_Hz=centroid(rx, ry),
                     x_minus_y=r[lx + rx].mean() - r[ly + ry].mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- dwell times + transitions, by propagating to real sessions (notebook procedure) ----
import os
gm, gs = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
from scipy.spatial.distance import cdist
WV = BASE + 'paw_wavelets/'
files = sorted(f for f in os.listdir(WV) if f.startswith('paw_vel_wavelets_'))[:25]
dwell = {k: [] for k in range(8)}
trans = np.zeros((8, 8))
for f in files:
    d = pd.read_parquet(WV + f, columns=var_interest)
    a = d.to_numpy(float)
    ok = ~np.isnan(a).any(axis=1)
    if ok.sum() < 10000:
        continue
    z = stats.zscore(a[ok], axis=0, nan_policy='omit')
    z = (z - gm) / gs
    st = np.vectorize(fix_mapping.get)(np.argmin(cdist(pca.transform(z)[:, :k_opt], km.cluster_centers_), axis=1))
    ch = np.r_[0, np.nonzero(np.diff(st))[0] + 1, len(st)]
    for a0, b0 in zip(ch[:-1], ch[1:]):
        dwell[int(st[a0])].append((b0 - a0) / 60 * 1000)   # ms
    for u, v in zip(st[:-1], st[1:]):
        trans[u, v] += 1
print('\n=== dwell time per state (ms), %d sessions ===' % len(files))
print(pd.DataFrame({'median_ms': {k: np.median(v) for k, v in dwell.items() if v},
                    'mean_ms': {k: np.mean(v) for k, v in dwell.items() if v},
                    'n_bouts': {k: len(v) for k, v in dwell.items() if v}}).round(1).to_string())
np.fill_diagonal(trans, 0)
tp = trans / trans.sum(axis=1, keepdims=True)
print('\n=== transition probabilities (row -> col, self-transitions removed) ===')
print(pd.DataFrame(tp).round(2).to_string())

m.to_csv("state_profiles_19Ago2026.csv")
