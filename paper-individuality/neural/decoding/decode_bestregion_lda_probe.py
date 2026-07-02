"""Simplest first probe: pick the Cosmos region with the most sessions, decode a target with a
plain ridge linear regression (5-fold CV R2) per session, correlate CV R2 with LDA 1.
Deliberately crude (no nested CV / C-tuning / null)."""
import numpy as np, pandas as pd, pickle, os, warnings
import matplotlib.pyplot as plt
from iblatlas.regions import BrainRegions
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

prefix = '/home/ines/repositories/representation_learning_variability/paper-individuality/'
firing_rates_dir = prefix + 'data/firing_rates/'
DROP = ['root', 'void']
TARGET = 'feedback'          # 'feedback' | 'stim' | 'choice' | 'block'
WINDOW = (-0.2, 0.0)
MIN_NEURONS = 10
CV_FOLDS = 5
SEED = 0

trials_df = pd.read_parquet(prefix + '4_mice/all_trials_04-05-2026')
idx = pd.read_parquet(prefix + 'neural/decoding/firing_session_pid_index.parquet')
lda = pd.read_pickle(prefix + 'clustering/mouse_LDA_5_bins_cut19-06-2026').rename(columns={0: 'lda_1'})
br = BrainRegions()

# region with the most sessions (data-driven, from the block per-session parquet)
blk = pd.read_parquet(prefix + 'neural/decoding/decoding_block_pseudo_persession.parquet')
REGION = blk.groupby('region')['session'].nunique().idxmax()
print(f"region with most sessions: {REGION} ({blk[blk.region==REGION].session.nunique()} sessions)")

def target_labels(lab):
    cc = (lab['choice'].values == 'right').astype(float); corr = lab['correct'].values
    ct = lab['contrast'].values; blk = lab['block'].values
    if TARGET == 'feedback': return ~np.isnan(corr), corr
    if TARGET == 'stim':     return ct > 0, np.where(corr == 1, cc, 1 - cc)
    if TARGET == 'choice':   return ~np.isnan(cc), cc
    if TARGET == 'block':    return np.isin(blk, [0.2, 0.8]), (blk == 0.8).astype(float)

sessions = sorted(set(idx.session) & set(lda.session))
rows = []
for si, eid in enumerate(sessions):
    pivs = []
    for fn in idx.loc[idx.session == eid, 'filename']:
        d = pickle.load(open(firing_rates_dir + fn, 'rb'))
        d = d[~d['area'].isin(DROP)].copy()
        if len(d) == 0: continue
        uac = d['area'].dropna().unique()
        cmap = dict(zip(uac, br.acronym2acronym(uac, mapping='Cosmos')))
        d = d[d['area'].map(cmap) == REGION]
        if len(d) == 0: continue
        d['nuid'] = d['pid'].astype(str) + '__' + d['neuron_id'].astype(str)
        tcols = sorted([c for c in d.columns if c.startswith('t_')], key=lambda x: float(x.split('_')[1]))
        tsec = np.array([float(c.split('_')[1]) for c in tcols])
        cols = [c for c, m in zip(tcols, (tsec >= WINDOW[0]) & (tsec <= WINDOW[1])) if m]
        d['fr'] = d[cols].mean(axis=1)
        pivs.append(d.pivot_table(index='trial_id', columns='nuid', values='fr'))
    if not pivs: continue
    X = pd.concat(pivs, axis=1)
    lab = trials_df[trials_df.session == eid].set_index('trial_id').reindex(X.index)
    mask, y = target_labels(lab)
    good = mask & ~np.isnan(y) & ~np.isnan(X.values).any(axis=1)
    Xg, yg = X.values[good], y[good].astype(float)
    if Xg.shape[1] < MIN_NEURONS or Xg.shape[0] < 50 or len(np.unique(yg)) < 2: continue
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    r2 = cross_val_score(pipe, Xg, yg, cv=KFold(CV_FOLDS, shuffle=True, random_state=SEED), scoring='r2').mean()
    rows.append(dict(session=eid, n_neurons=Xg.shape[1], n_trials=Xg.shape[0], r2=r2))
    if (si + 1) % 40 == 0: print(f"  {si+1}/{len(sessions)} scanned, {len(rows)} usable")

res = pd.DataFrame(rows).merge(lda[['session', 'lda_1']], on='session').dropna(subset=['lda_1'])
out = prefix + f'neural/decoding/probe_{REGION}_{TARGET}_r2_vs_lda.parquet'
res.to_parquet(out)
print(f"\n{REGION} | {TARGET} | window {WINDOW} | {len(res)} sessions | mean CV R2={res.r2.mean():.3f}")
r, pv = pearsonr(res.lda_1, res.r2); rho, pvv = spearmanr(res.lda_1, res.r2)
print(f"CV R2 vs LDA 1:  r={r:+.3f} p={pv:.3f} | rho={rho:+.3f} p={pvv:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(res.lda_1, res.r2, c=res.lda_1, cmap='viridis', s=45, edgecolor='k', linewidth=0.3)
z = np.polyfit(res.lda_1, res.r2, 1); xl = np.linspace(res.lda_1.min(), res.lda_1.max(), 100)
ax.plot(xl, np.polyval(z, xl), 'k-', lw=2); ax.axhline(0, color='gray', ls=':', alpha=0.7)
ax.set_xlabel('LDA 1'); ax.set_ylabel(f'CV R2 ({TARGET} decoding, {REGION})')
ax.set_title(f'{REGION} {TARGET} decodability vs LDA 1\nr={r:.3f} p={pv:.3f} | rho={rho:.3f} p={pvv:.3f} (n={len(res)})', fontsize=11)
plt.colorbar(sc, label='LDA 1'); plt.tight_layout()
plt.savefig(prefix + f'neural/decoding/probe_{REGION}_{TARGET}_r2_vs_lda.png', dpi=110, bbox_inches='tight')
print("saved plot + parquet")
