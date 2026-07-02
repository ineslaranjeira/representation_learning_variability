"""Block (0.2 vs 0.8) decodability in Isocortex with the IBL L1-logistic scheme + pseudosession null,
correlated with LDA 1. Features = Isocortex neurons (window-averaged, no PCA); L1 logistic regression,
class_weight balanced, balanced accuracy. Biased-only IBL pseudosession null -> corrected = real - pseudo."""
import numpy as np, pandas as pd, pickle, os, warnings
import matplotlib.pyplot as plt
from iblatlas.regions import BrainRegions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

prefix = '/home/ines/repositories/representation_learning_variability/paper-individuality/'
firing_rates_dir = prefix + 'data/firing_rates/'
DROP = ['root', 'void']
REGION = 'Isocortex'
WINDOW = (-0.4, 0.0)
C_REG = 1.0
CV_FOLDS = 5
N_PSEUDO = 50
MIN_NEURONS = 10
MIN_CLASS = CV_FOLDS
SEED = 0
BLK_MEAN, BLK_MIN, BLK_MAX = 60, 20, 100
rng = np.random.default_rng(SEED)

trials_df = pd.read_parquet(prefix + '4_mice/all_trials_04-05-2026')
idx = pd.read_parquet(prefix + 'neural/decoding/firing_session_pid_index.parquet')
lda = pd.read_pickle(prefix + 'clustering/mouse_LDA_5_bins_cut19-06-2026').rename(columns={0: 'lda_1'})
block_by_session = {s: g.sort_values('trial_id').set_index('trial_id')['block'] for s, g in trials_df.groupby('session')}
br = BrainRegions()

def gen_pseudo_biased(n, rng):
    out = []; side = float(rng.choice([0.2, 0.8]))
    while len(out) < n:
        x = rng.exponential(BLK_MEAN)
        while x < BLK_MIN or x > BLK_MAX: x = rng.exponential(BLK_MEAN)
        out += [side] * int(x); side = 0.8 if side == 0.2 else 0.2
    return np.array(out[:n])

def make_folds(X, y_split):
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED); folds = []
    for tr, te in skf.split(X, y_split):
        sc = StandardScaler().fit(X[tr]); folds.append((tr, te, sc.transform(X[tr]), sc.transform(X[te])))
    return folds

def acc_logreg(folds, y):
    a = []
    for tr, te, Xtr, Xte in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: a.append(np.nan); continue
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=C_REG, tol=1e-3,
                                 max_iter=20000, fit_intercept=True, class_weight='balanced')
        clf.fit(Xtr, y[tr]); a.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return np.nanmean(a)

def load_region(eid):
    pivs = []
    for fn in idx.loc[idx.session == eid, 'filename']:
        d = pickle.load(open(firing_rates_dir + fn, 'rb'))
        d = d[~d['area'].isin(DROP)].copy()
        if len(d) == 0: continue
        uac = d['area'].dropna().unique(); cmap = dict(zip(uac, br.acronym2acronym(uac, mapping='Cosmos')))
        d = d[d['area'].map(cmap) == REGION]
        if len(d) == 0: continue
        d['nuid'] = d['pid'].astype(str) + '__' + d['neuron_id'].astype(str)
        tcols = sorted([c for c in d.columns if c.startswith('t_')], key=lambda x: float(x.split('_')[1]))
        tsec = np.array([float(c.split('_')[1]) for c in tcols])
        cols = [c for c, m in zip(tcols, (tsec >= WINDOW[0]) & (tsec <= WINDOW[1])) if m]
        d['fr'] = d[cols].mean(axis=1)
        pivs.append(d.pivot_table(index='trial_id', columns='nuid', values='fr'))
    return pd.concat(pivs, axis=1) if pivs else None

sessions = sorted(set(idx.session) & set(lda.session))
rows = []
for si, eid in enumerate(sessions):
    X = load_region(eid)
    if X is None or X.shape[1] < MIN_NEURONS: continue
    blk = block_by_session.get(eid)
    if blk is None: continue
    full = blk.values; bpa = np.where(np.isin(full, [0.2, 0.8]))[0]
    if len(bpa) == 0: continue
    fb = bpa[0]; tids = blk.index.values[bpa]; keep = np.isin(tids, X.index.values)
    biased_pos, biased_tids = bpa[keep], tids[keep]
    Xr = X.loc[biased_tids].values; ok = ~np.isnan(Xr).any(axis=1)
    Xr, offs = Xr[ok], (biased_pos - fb)[ok]
    y = pd.Series(full[biased_pos][ok]).map({0.2: 0, 0.8: 1}).values.astype(int)
    if Xr.shape[0] < 60 or np.bincount(y).min() < MIN_CLASS: continue
    folds = make_folds(Xr, y)
    real = acc_logreg(folds, y)
    n_bias = len(bpa)
    pv = []
    for _ in range(N_PSEUDO):
        pf = gen_pseudo_biased(n_bias, rng)
        pv.append(acc_logreg(folds, pd.Series(pf[offs]).map({0.2: 0, 0.8: 1}).values.astype(int)))
    pm = np.nanmean(pv)
    rows.append(dict(session=eid, n_neurons=Xr.shape[1], n_trials=Xr.shape[0],
                     real=real, pseudo_mean=pm, corrected=real - pm))
    if (si + 1) % 40 == 0: print(f"  {si+1}/{len(sessions)} scanned, {len(rows)} usable")

res = pd.DataFrame(rows).merge(lda[['session', 'lda_1']], on='session').dropna(subset=['lda_1'])
out = prefix + f'neural/decoding/probe_{REGION}_block_logreg_vs_lda.parquet'
res.to_parquet(out)
print(f"\n{REGION} | block | window {WINDOW} | {len(res)} sessions | "
      f"real={res.real.mean():.3f} pseudo={res.pseudo_mean.mean():.3f} corrected={res.corrected.mean():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, m in zip(axes, ['real', 'corrected']):
    r, pvl = pearsonr(res.lda_1, res[m]); rho, pvv = spearmanr(res.lda_1, res[m])
    sc = ax.scatter(res.lda_1, res[m], c=res.lda_1, cmap='viridis', s=45, edgecolor='k', linewidth=0.3)
    z = np.polyfit(res.lda_1, res[m], 1); xl = np.linspace(res.lda_1.min(), res.lda_1.max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'k-', lw=2)
    if m == 'corrected': ax.axhline(0, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('LDA 1'); ax.set_ylabel(f'{m} block decodability ({REGION})')
    ax.set_title(f'{m}: r={r:.3f} p={pvl:.3f} | rho={rho:.3f} p={pvv:.3f} (n={len(res)})', fontsize=10)
fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label='LDA 1')
fig.suptitle(f'{REGION} block decodability (L1 logistic + pseudosession null) vs LDA 1', fontsize=12)
plt.savefig(prefix + f'neural/decoding/probe_{REGION}_block_logreg_vs_lda.png', dpi=110, bbox_inches='tight')
for m in ['real', 'corrected']:
    r, pvl = pearsonr(res.lda_1, res[m]); print(f"{m:10s} vs LDA1: r={r:+.3f} p={pvl:.3f}")
print("saved plot + parquet")
