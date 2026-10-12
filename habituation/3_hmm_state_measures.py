"""
3. Whisking and licking STATES at habituation: are their amount and timing individual, and do they last?
=====================================================================================================
The 2-state whisking (Gaussian, z-scored whisker ME, 30 Hz) and licking (Poisson, 100 ms bins) HMMs of
segmentation/4.0_hmm_fit.ipynb (DATASET = 'habituation'), fitted on habituation sessions only. The 1-camera
training stages (Early = session_1, Late = last_training, Pre-rec = biased) were fitted with the SAME settings
(data/training/hmm/), so their measures are comparable to habituation in absolute terms; Proficient was fitted
from the 2-camera, 60 Hz data with the unbinned lick HMM (data/hmm/), so it is compared by RANK only.

Per session, per signal (state 1 = whisking / licking; no trial alignment -- habituation has no reliable trials):
  frac      fraction of time in the active state
  bout_rate active bouts per minute
  bout_dur  median active-bout duration (s)
Whisking: only fits passing the pipeline's screens (fit_ok). Licking: 'degenerate occupancy' / 'collapsed' fits
are near-silent sessions, not failures, so they are kept (collapsed = no licking); flickering / failed fits dropped. Bouts are counted within contiguous stretches of frames (a gap in the frame times ends a bout).

Analyses (Spearman across mice, raw and lab-controlled, as in 1_early_behaviour_stability.py):
  1. day-to-day stability within habituation (day 1 vs 2, 2 vs 3, 1 vs 3);
  2. mouse identification across habituation sessions from the six measures (leave-one-session-out nearest
     centroid, z-scored; null = shuffling sessions between mice of the same lab);
  3. habituation (mouse mean) vs Early, Late, Pre-rec (absolute level and rank) and Proficient (rank only).

@author: Ines
"""
#%%
import re
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, wilcoxon

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
REPO = HERE.parent
PI = REPO / 'paper-individuality'
DM = PI / 'data' / 'design_matrices'
FIG = HERE / 'figures'
CACHE = HERE / 'hmm_state_measures.csv'      # per-session measures; delete to recompute
sys.path.insert(0, str(PI))
import paper_style as ps

TAG = {'whisker_me': '5_gaussian_kmeans_em_zsc_True_k0_s2',
       'Lick count': '5_poisson_prior_em_zsc_False_k0_s2_bin100ms'}
PROF_TAG = {'whisker_me': '5_gaussian_kmeans_em_zsc_True_k0_s2', 'Lick count': '5_poisson_prior_em_zsc_False_k0_s2'}
SOURCES = {   # stage -> (hmm root, tag per variable, fps)
    'Habituation': (REPO / 'data' / 'individuality' / 'habituation' / 'hmm', TAG, 30.0),
    'Training': (PI / 'data' / 'training' / 'hmm', TAG, 30.0),
    'Proficient': (PI / 'data' / 'hmm', PROF_TAG, 60.0),
}
STAGE_DIRS = {'Early': DM / '1_camera_setup' / 'session_1', 'Late': DM / '1_camera_setup' / 'last_training',
              'Pre-rec': DM / '1_camera_setup' / 'biased'}
SHORT = {'whisker_me': 'whisk', 'Lick count': 'lick'}
MEASURES = ['frac', 'bout_rate', 'bout_dur']
FEATS = [f'{SHORT[v]}_{m}' for v in TAG for m in MEASURES]
LABEL = {'frac': 'fraction of time', 'bout_rate': 'bouts / min', 'bout_dur': 'bout duration (s)'}
ORDER = ['Habituation', 'Early', 'Late', 'Pre-rec', 'Proficient']
N_PERM, N_PERM_ID = 5000, 1000
rng = np.random.default_rng(0)
LAB_CONTROL = False   # figures: True = rho_lab / q_lab, False = plain Spearman rho / BH q of its p
RHO, Q = ('rho_lab', 'q_lab') if LAB_CONTROL else ('rho', 'q')

#%%
# ---- per-session state measures -----------------------------------------------------
NAME = re.compile(r'^(whisker_me|Lick count)_(.+)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$')

def runs(active, breaks):
    """Lengths (in frames) of runs of True, a run ending at every break (gap in frame times)."""
    seg = np.cumsum(np.r_[True, breaks])           # id of the contiguous stretch each frame belongs to
    d = np.diff(np.r_[0, active.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for s, e in zip(starts, ends):                  # split a run where it crosses a break
        ids = seg[s:e]
        out += list(np.diff(np.r_[0, np.flatnonzero(np.diff(ids)) + 1, e - s]))
    return np.array(out, float)

def measures(path, fps):
    obj = pickle.load(open(path, 'rb'))
    states = np.asarray(obj[0]).astype(int)
    if len(obj) == 4:                               # (states, Bin, ...): gaps in Bin end bouts
        bins = np.asarray(obj[1], float)
        breaks = np.diff(bins) > 1.5 / fps
    else:                                           # older 3-element fits: no frame times, treat as contiguous
        breaks = np.zeros(len(states) - 1, bool)
    minutes = len(states) / fps / 60
    L = runs(states == 1, breaks)
    return dict(frac=(states == 1).mean(), bout_rate=len(L) / minutes,
                bout_dur=np.median(L) / fps if len(L) else np.nan, minutes=minutes)

if CACHE.exists():
    S = pd.read_csv(CACHE)
else:
    rows = []
    for source, (root, tags, fps) in SOURCES.items():
        for var, tag in tags.items():
            ass = pd.read_csv(root / 'fits' / tag / f'assessments_{var}.csv').set_index('eid')
            flag = lambda c: ass[c].fillna(False).astype(bool) if c in ass else pd.Series(False, ass.index)
            failed = flag('flickering') | (ass.get('n_folds_failed', pd.Series(0, ass.index)).fillna(0) > 0) | \
                     (ass['error'].fillna('') != '')
            # A 'degenerate' or 'collapsed' LICK fit is a session with almost no licking (habituation: median 4.5
            # lick bins / min against 37 in the rest), not a failed fit -- dropping it would drop exactly the
            # low lickers. Degenerate ones are kept as fitted; collapsed ones (one state for the whole session,
            # oriented 'high') are set to no licking. Whisking fits pass every screen.
            keep = set(ass.index[~failed]) if var == 'Lick count' else set(ass.index[ass['fit_ok'].astype(bool)])
            collapsed = set(ass.index[flag('collapsed')])
            for f in sorted((root / 'most_likely_states' / tag).iterdir()):
                m = NAME.match(f.name)
                if not m or m.group(3) not in keep:
                    continue
                meas = measures(f, fps)
                if var == 'Lick count' and m.group(3) in collapsed:
                    meas.update(frac=0.0, bout_rate=0.0, bout_dur=np.nan)
                rows.append(dict(source=source, var=SHORT[var], mouse_name=m.group(2), eid=m.group(3), **meas))
    S = pd.DataFrame(rows)
    S.to_csv(CACHE, index=False)

# training fits share one folder: the stage of a session is the design-matrix folder holding it
stage_of = {}
for st, d in STAGE_DIRS.items():
    for f in d.glob('design_matrix_*'):
        stage_of[f.name[14:50]] = st
S['stage'] = np.where(S['source'] == 'Training', S['eid'].map(stage_of), S['source'])
S = S.dropna(subset=['stage'])
wide = S.pivot_table(index=['stage', 'mouse_name', 'eid'], columns='var', values=MEASURES)
wide.columns = [f'{v}_{m}' for m, v in wide.columns]
wide = wide.reset_index()

inventory = pd.read_csv(HERE / 'habituation_lp_sessions.csv').set_index('eid')
lab = inventory.groupby('mouse_name')['lab'].first()
hab = wide[wide['stage'] == 'Habituation'].copy()
hab['hab_day'] = hab['eid'].map(inventory['hab_day'])
print(S.groupby(['stage', 'var']).agg(sessions=('eid', 'size'), mice=('mouse_name', 'nunique')).to_string())
print('\nhabituation, per-session medians:')
print(hab[FEATS].median().round(3).to_string())
mouse = wide.groupby(['stage', 'mouse_name'])[FEATS].mean()

#%%
# ---- helpers (as in script 1) ----------------------------------------------------------
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

#%%
# ---- 1. day-to-day stability within habituation ----------------------------------------
stab = []
for f in FEATS:
    w = hab.pivot_table(index='mouse_name', columns='hab_day', values=f)
    for a, b in [(1, 2), (2, 3), (1, 3)]:
        if a in w and b in w:
            pair = w[[a, b]].dropna()
            pair = pair[pair.index.isin(lab.index)]
            if len(pair) >= 8:
                stab.append(dict(feature=f, pair=f'day {a} vs {b}', **correlate(pair[a], pair[b], lab.reindex(pair.index))))
stab = pd.DataFrame(stab)
stab['q_lab'] = bh(stab['p_lab'])
stab['q'] = bh(stab['p'])
print('\n== 1. Day-to-day stability within habituation ==')
print(stab.round(3).to_string(index=False))

#%%
# ---- 2. mouse identification across habituation sessions --------------------------------
def nearest_centroid_loso(X, y):
    hits = np.zeros(len(X), bool)
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        mice = np.unique(y[tr])
        C = np.stack([X[tr][y[tr] == m].mean(0) for m in mice])
        hits[i] = mice[np.argmin(((C - X[i]) ** 2).sum(1))] == y[i]
    return hits.mean()

def identification(cols):
    d = hab.dropna(subset=cols)
    d = d[d['mouse_name'].isin(lab.index) & (d.groupby('mouse_name')['eid'].transform('size') >= 2)]
    X = ((d[cols] - d[cols].mean()) / d[cols].std()).to_numpy()
    y, labs_s = d['mouse_name'].to_numpy(), d['mouse_name'].map(lab).to_numpy()
    acc, null = nearest_centroid_loso(X, y), []
    for _ in range(N_PERM_ID):
        yp = y.copy()
        for l in np.unique(labs_s):
            idx = np.where(labs_s == l)[0]
            yp[idx] = rng.permutation(yp[idx])
        null.append(nearest_centroid_loso(X, yp))
    null = np.array(null)
    return dict(features=', '.join(cols), accuracy=acc, n_sessions=len(X), n_mice=len(np.unique(y)),
                chance=1 / len(np.unique(y)), null_within_lab=null.mean(),
                p=(np.sum(null >= acc) + 1) / (len(null) + 1))

ID = pd.DataFrame([identification(FEATS),
                   identification([f for f in FEATS if f.startswith('whisk')]),
                   identification([f for f in FEATS if f.startswith('lick')])])
print('\n== 2. Mouse identification across habituation sessions (>= 2 sessions per mouse) ==')
print(ID.round(3).to_string(index=False))

#%%
# ---- 3. habituation vs later stages -------------------------------------------------------
cross, level = [], []
for f in FEATS:
    for st in ORDER[1:]:
        both = pd.concat([mouse.loc['Habituation', f], mouse.loc[st, f]], axis=1, keys=['x', 'y']).dropna()
        both = both[both.index.isin(lab.index)]
        if len(both) < 8:
            continue
        cross.append(dict(feature=f, stage=st, **correlate(both['x'], both['y'], lab.reindex(both.index))))
        if st != 'Proficient':      # absolute levels only where the fits are comparable
            level.append(dict(feature=f, stage=st, n=len(both), hab_median=both['x'].median(),
                              stage_median=both['y'].median(), p_wilcoxon=wilcoxon(both['x'], both['y']).pvalue))
cross = pd.DataFrame(cross)
cross['q_lab'] = bh(cross['p_lab'])
cross['q'] = bh(cross['p'])
print('\n== 3a. Habituation (mouse mean) vs later stages, by rank (Proficient: rank only) ==')
print(cross.pivot_table(index='feature', columns='stage', values='rho_lab')[ORDER[1:]].round(2).to_string())
print(cross.round(3).to_string(index=False))
level = pd.DataFrame(level)
print('\n== 3b. Absolute levels, same mice: habituation vs Early / Late / Pre-rec (paired Wilcoxon) ==')
print(level.round(3).to_string(index=False))

#%%
# ---- figure ---------------------------------------------------------------------------------
ps.use('paper')
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharey=True, constrained_layout=True)
for r, sig in enumerate(['whisk', 'lick']):
    for c, m in enumerate(MEASURES):
        ax = axes[r, c]
        f = f'{sig}_{m}'
        s = stab[(stab['feature'] == f) & (stab['pair'] == 'day 1 vs 2')]
        d = cross[cross['feature'] == f].set_index('stage').reindex(ORDER[1:])
        xs = np.arange(len(ORDER))
        ys = np.r_[s[RHO].iloc[0] if len(s) else np.nan, d[RHO].to_numpy()]
        qs = np.r_[s[Q].iloc[0] if len(s) else np.nan, d[Q].to_numpy()]
        ax.bar(xs, ys, color=['#B0B0B0'] + [ps.TIMEPOINT.get(t, '0.5') for t in ORDER[1:]], width=0.7)
        for x, y_, q_ in zip(xs, ys, qs):
            if np.isfinite(q_) and q_ < 0.05:
                ax.text(x, y_ + 0.03, '*', ha='center')
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(['Hab d1↔d2'] + ORDER[1:], rotation=40, ha='right')
        ax.set_title(f'{sig}: {LABEL[m]}')
    axes[r, 0].set_ylabel('ρ' + (' (lab-controlled)' if LAB_CONTROL else ''))
fig.suptitle('Whisking / licking HMM states: habituation stability and carry-over (* q < 0.05)', fontsize=9)
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'habituation_hmm_states.png', dpi=200)
plt.show()
