"""
2. Habituation + Early training pooled into one NAIVE stage: are its raw-signal traits stable, and do they last?
==============================================================================================================
Same three raw per-session signals as 1_early_behaviour_stability.py (lick rate, whisker motion energy, left
paw speed; read from its cache, session_features.csv -- run that script first if it is missing), but the
habituation days and the first training session (Early) are pooled into one naive stage, so a mouse's naive
value rests on up to 4 sessions instead of 1-3.

POOLING ACROSS SESSION TYPES. Habituation and Early are both 1-camera, 30 Hz, same rig per lab, but Early has
trials and rewards, so absolute levels differ (licking above all). Each feature is therefore z-scored ACROSS
MICE within each session type (habituation day 1, 2, 3, Early) before pooling: a session contributes how the
mouse ranks against the others on that day, and a mouse's naive score is the mean of those. The same is done for
habituation alone and Early alone, so the three are comparable.

Analyses (every across-mouse correlation raw and lab-controlled, as in script 1):
  1. reliability of the naive score: repeatability R (one-way ICC over a mouse's naive sessions, lab-centred)
     and split-half (habituation-day mean vs Early, and odd vs even naive sessions), Spearman-Brown corrected;
  2. naive score vs Late, Pre-rec and Proficient (mouse means), next to habituation-only and Early-only;
  3. naive score vs learning speed (log training_days);
  4. mouse identification from the three signals across naive sessions: leave-one-session-out nearest mouse
     centroid (z-scored features), mice with >= 3 naive sessions having all three; chance = 1 / n mice, null
     from shuffling session -> mouse labels within lab.

@author: Ines
"""
#%%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
REPO = HERE.parent
HAB = REPO / 'data' / 'individuality' / 'habituation'
FIG = HERE / 'figures'
sys.path.insert(0, str(REPO / 'paper-individuality'))
import paper_style as ps

FEATURES = {'lick_rate': 'Lick rate', 'whisk_me': 'Whisker ME', 'paw_speed': 'Left paw speed'}
LATER = ['Late', 'Pre-rec', 'Proficient']
N_PERM = 5000
N_PERM_ID = 1000
rng = np.random.default_rng(0)
LAB_CONTROL = False   # figures: True = rho_lab / q_lab, False = plain Spearman rho / BH q of its p
RHO, Q = ('rho_lab', 'q_lab') if LAB_CONTROL else ('rho', 'q')

#%%
# ---- sessions: the cache of script 1, with its habituation exclusions -----------
sessions = pd.read_csv(HERE / 'session_features.csv')
hab_log = pd.read_csv(HAB / 'processing_log.csv').set_index('eid')
inventory = pd.read_csv(HERE / 'habituation_lp_sessions.csv').set_index('eid')
is_hab = sessions['timepoint'] == 'Habituation'
sessions = sessions[~is_hab | sessions['eid'].isin(hab_log.index[hab_log['status'] == 'ok'])].copy()
is_hab = sessions['timepoint'] == 'Habituation'
for flag, feat in [('lick_ok', 'lick_rate'), ('whisk_ok', 'whisk_me'), ('paw_ok', 'paw_speed')]:
    bad = is_hab & ~sessions['eid'].map(hab_log[flag]).fillna(False).astype(bool)
    sessions.loc[bad, feat] = np.nan
sessions['hab_day'] = sessions['eid'].map(inventory['hab_day'])
lab = inventory.groupby('mouse_name')['lab'].first()        # mice with habituation sessions

# session type for the naive stage: habituation day 1-3 (day 4: 8 mice, folded into day 3) or Early
naive = sessions[sessions['timepoint'].isin(['Habituation', 'Early']) & sessions['mouse_name'].isin(lab.index)].copy()
naive['stype'] = np.where(naive['timepoint'] == 'Early', 'Early',
                          'hab' + naive['hab_day'].clip(upper=3).astype('Int64').astype(str))
for f in FEATURES:      # z across mice within each session type (each mouse once per type: mean if repeated)
    naive[f + '_z'] = naive.groupby('stype')[f].transform(lambda v: (v - v.mean()) / v.std())
Z = [f + '_z' for f in FEATURES]
print(naive.groupby('stype').agg(sessions=('eid', 'size'), mice=('mouse_name', 'nunique')).to_string())
print('naive sessions per mouse:', naive.groupby('mouse_name').size().describe().round(1).to_dict())

def score(d):
    return d.groupby('mouse_name')[Z].mean().rename(columns=dict(zip(Z, FEATURES)))

STAGES = {'Naive (hab + Early)': score(naive),
          'Habituation only': score(naive[naive['stype'] != 'Early']),
          'Early only': score(naive[naive['stype'] == 'Early'])}
later = sessions[sessions['timepoint'].isin(LATER)].groupby(['timepoint', 'mouse_name'])[list(FEATURES)].mean()

#%%
# ---- helpers (as in script 1) ----------------------------------------------------
def lab_residual_ranks(values, labs):
    r = pd.Series(rankdata(values), index=values.index)
    return r - r.groupby(labs.values).transform('mean')

def correlate(x, y, labs, n_perm=N_PERM):
    raw = spearmanr(x, y)
    rx, ry = lab_residual_ranks(x, labs), lab_residual_ranks(y, labs)
    r_lab = np.corrcoef(rx, ry)[0, 1]
    groups = [np.where(labs.values == l)[0] for l in np.unique(labs.values)]
    ryv, null = ry.values, np.empty(n_perm)
    for i in range(n_perm):
        perm = np.arange(len(ryv))
        for g in groups:
            perm[g] = rng.permutation(g)
        null[i] = np.corrcoef(rx.values, ryv[perm])[0, 1]
    return {'n': len(x), 'rho': raw.statistic, 'p': raw.pvalue, 'rho_lab': r_lab,
            'p_lab': (np.sum(np.abs(null) >= abs(r_lab)) + 1) / (n_perm + 1)}

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1)
    return out

def icc(values, groups):
    """One-way ANOVA repeatability (unbiased, as non_emerging/session_metric_individuality)."""
    v = np.asarray(values, float); ok = np.isfinite(v); v = v[ok]
    g = pd.factorize(np.asarray(groups)[ok])[0]
    M_, N_ = g.max() + 1, len(v)
    n = np.bincount(g).astype(float); mu = np.bincount(g, weights=v) / n
    keep = n >= 2
    ms_w = ((v - mu[g]) ** 2).sum() / (N_ - M_)
    ms_b = (n * (mu - v.mean()) ** 2).sum() / (M_ - 1)
    n0 = (N_ - (n ** 2).sum() / N_) / (M_ - 1)
    s2 = (ms_b - ms_w) / n0
    return max(s2, 0) / (max(s2, 0) + ms_w), int(keep.sum())

#%%
# ---- 1. reliability of the naive score -------------------------------------------
rel = []
for f, z in zip(FEATURES, Z):
    d = naive.dropna(subset=[z])
    d = d[d.groupby('mouse_name')[z].transform('size') >= 2]
    centred = d[z] - d.groupby(d['mouse_name'].map(lab))[z].transform('mean')    # lab-centred
    R, n_m = icc(centred, d['mouse_name'])
    # split halves: habituation mean vs Early; odd vs even sessions (chronological within mouse)
    h = d[d['stype'] != 'Early'].groupby('mouse_name')[z].mean()
    e = d[d['stype'] == 'Early'].groupby('mouse_name')[z].mean()
    he = pd.concat([h, e], axis=1, keys=['a', 'b']).dropna()
    d = d.sort_values(['mouse_name', 'stype'])
    k = d.groupby('mouse_name').cumcount()
    oe = pd.concat([d[k % 2 == 0].groupby('mouse_name')[z].mean(), d[k % 2 == 1].groupby('mouse_name')[z].mean()],
                   axis=1, keys=['a', 'b']).dropna()
    for name, pair in [('habituation vs Early', he), ('odd vs even sessions', oe)]:
        c = correlate(pair['a'], pair['b'], lab.reindex(pair.index))
        rel.append(dict(feature=f, test=name, R_naive=R, mice_R=n_m, **c,
                        spearman_brown=2 * c['rho_lab'] / (1 + c['rho_lab']) if c['rho_lab'] > -1 else np.nan))
rel = pd.DataFrame(rel)
print('\n== 1. Reliability of the naive score (lab-controlled) ==')
print(rel.round(3).to_string(index=False))

#%%
# ---- 2. naive vs later stages ------------------------------------------------------
cross = []
for stage, S in STAGES.items():
    for f in FEATURES:
        for tp in LATER:
            both = pd.concat([S[f], later.loc[tp, f]], axis=1, keys=['x', 'y']).dropna()
            both = both[both.index.isin(lab.index)]
            if len(both) >= 5:
                cross.append(dict(stage=stage, feature=f, later=tp, **correlate(both['x'], both['y'], lab.reindex(both.index))))
cross = pd.DataFrame(cross)
cross['q_lab'] = cross.groupby('stage')['p_lab'].transform(bh)
cross['q'] = cross.groupby('stage')['p'].transform(bh)
print('\n== 2. Naive score vs later stages (rho_lab; q_lab = BH within each stage row family) ==')
print(cross.pivot_table(index=['feature', 'later'], columns='stage', values='rho_lab').round(2).to_string())
print(cross[cross['stage'] == 'Naive (hab + Early)'].round(3).to_string(index=False))

#%%
# ---- 3. naive vs learning speed -----------------------------------------------------
training = pd.read_parquet(REPO / 'learning_prediction' / 'training_time_05-05-2026')
training['log_training'] = np.log(training['training_days'])
training = training.loc[~training['log_training'].isin([0, 1])]
speed = np.log(training.set_index('mouse_name')['training_days'])
learn = []
for stage, S in STAGES.items():
    for f in FEATURES:
        both = pd.concat([S[f], speed], axis=1, keys=['x', 'y']).dropna()
        both = both[both.index.isin(lab.index)]
        learn.append(dict(stage=stage, feature=f, **correlate(both['x'], both['y'], lab.reindex(both.index))))
learn = pd.DataFrame(learn)
learn['q_lab'] = bh(learn['p_lab'])
print('\n== 3. Naive score vs learning speed (log training days; + = slower) ==')
print(learn.round(3).to_string(index=False))

#%%
# ---- 4. mouse identification across naive sessions ---------------------------------
def nearest_centroid_loso(X, y):
    hits = np.zeros(len(X), bool)
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        mice = np.unique(y[tr])
        C = np.stack([X[tr][y[tr] == m].mean(0) for m in mice])
        hits[i] = mice[np.argmin(((C - X[i]) ** 2).sum(1))] == y[i]
    return hits.mean()

idd = naive.dropna(subset=Z)
idd = idd[idd.groupby('mouse_name')['eid'].transform('size') >= 3]
X, y = idd[Z].to_numpy(), idd['mouse_name'].to_numpy()
labs_s = idd['mouse_name'].map(lab).to_numpy()
acc = nearest_centroid_loso(X, y)
null = []
for _ in range(N_PERM_ID):
    yp = y.copy()
    for l in np.unique(labs_s):          # shuffle sessions between mice of the same lab
        idx = np.where(labs_s == l)[0]
        yp[idx] = rng.permutation(yp[idx])
    null.append(nearest_centroid_loso(X, yp))
null = np.array(null)
ID = dict(accuracy=acc, n_sessions=len(X), n_mice=len(np.unique(y)), chance=1 / len(np.unique(y)),
          null_within_lab=null.mean(), p=(np.sum(null >= acc) + 1) / (len(null) + 1))
print('\n== 4. Mouse identification from the 3 raw signals, naive sessions ==')
print({k: round(v, 4) if isinstance(v, float) else v for k, v in ID.items()})

#%%
# ---- figure: rho (or rho_lab, see LAB_CONTROL) per feature x later stage, naive vs hab-only vs Early-only -------
ps.use('paper')
fig, axes = plt.subplots(1, len(FEATURES), figsize=(7.0, 2.2), sharey=True, constrained_layout=True)
style = {'Naive (hab + Early)': ('k', 'o', 0), 'Habituation only': ('#B0B0B0', 's', -0.18),
         'Early only': (ps.TIMEPOINT['Early'], '^', 0.18)}
for ax, f in zip(axes, FEATURES):
    for stage, (col, mk, dx) in style.items():
        d = cross[(cross['stage'] == stage) & (cross['feature'] == f)].set_index('later').reindex(LATER)
        x = np.arange(len(LATER)) + dx
        ax.plot(x, d[RHO], mk, color=col, ms=4, mfc=col if stage.startswith('Naive') else 'white', label=stage)
        for xi, (r_, q_) in zip(x, d[[RHO, Q]].to_numpy()):
            if q_ < 0.05:
                ax.text(xi, r_ + 0.05, '*', ha='center', fontsize=8)
    ax.axhline(0, color='0.7', lw=0.6)
    ax.set_xticks(range(len(LATER))); ax.set_xticklabels(LATER)
    ax.set_title(FEATURES[f])
axes[0].set_ylabel('ρ with later stage' + ('\n(lab-controlled)' if LAB_CONTROL else ''))
axes[0].set_ylim(top=cross[RHO].max() + 0.15)   # room for the stars
axes[-1].legend(frameon=False, fontsize=6, loc='upper right')
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'naive_pooled_vs_later.png', dpi=200)
plt.show()
