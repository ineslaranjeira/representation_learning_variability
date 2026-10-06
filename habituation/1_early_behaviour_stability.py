"""
1. Is habituation behaviour stable, and does it predict behaviour later in learning?
=========================================================================================
Three raw signals per session, straight from the design matrices (no segmentation):
  lick rate   : bins with a lick event per minute ('Lick count')
  whisking    : mean whisker-pad motion energy ('whisker_me')
  paw speed   : mean speed of the left paw (l_paw_x / l_paw_y, px/s)
averaged per session, then per mouse within each timepoint.

  Habituation  data/individuality/habituation/          (habituation_design_matrix.ipynb)
  Early        design_matrices/1_camera_setup/session_1/
  Late         design_matrices/1_camera_setup/last_training/
  Pre-rec      design_matrices/1_camera_setup/biased/
  Proficient   design_matrices/                         (two-camera; left camera used)

Absolute levels are NOT comparable across timepoints: grid rate (30 vs 60 Hz), one- vs
two-camera lick merging, rig and camera geometry all differ. Only the ORDER of mice within a
timepoint is used (Spearman). Lab is the other trap: habituation and training run on the same
rig per lab, and camera geometry carries lab (see the LD1 lab-tilt result), so a
habituation-vs-training correlation can be pure rig. Every correlation is therefore also
reported lab-controlled: ranks residualised on lab, p from permuting mice within lab.

Session exclusions: left paw NaN > 20% -> no paw speed for that session (all timepoints);
habituation sessions flagged lick_ok / whisk_ok False (face out of frame) -> no lick / whisk.

@author: Ines
"""
#%%
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
REPO = HERE.parent
DM = REPO / 'paper-individuality' / 'data' / 'design_matrices'
HAB = REPO / 'data' / 'individuality' / 'habituation'
FIG = HERE / 'figures'
CACHE = HERE / 'session_features.csv'      # per-session features; delete to recompute

sys.path.insert(0, str(REPO / 'paper-individuality'))
import paper_style as ps

TIMEPOINTS = {'Habituation': HAB,
              'Early': DM / '1_camera_setup' / 'session_1',
              'Late': DM / '1_camera_setup' / 'last_training',
              'Pre-rec': DM / '1_camera_setup' / 'biased',
              'Proficient': DM}
ORDER = list(TIMEPOINTS)
COLOR = {'Habituation': '#B0B0B0', **ps.TIMEPOINT}
FEATURES = {'lick_rate': 'Lick rate (bins/min)', 'whisk_me': 'Whisker motion energy',
            'paw_speed': 'Left paw speed (px/s)'}
MAX_PAW_NAN = 0.2
N_PERM = 5000
rng = np.random.default_rng(0)


#%%
# ---- per-session features --------------------------------------------------------
def session_features(path):
    dm = pd.read_parquet(path)
    fs = 1 / np.median(np.diff(dm['Bin']))
    minutes = len(dm) / fs / 60
    speed = np.hypot(dm['l_paw_x'].diff(), dm['l_paw_y'].diff()) / dm['Bin'].diff()
    paw_nan = dm['l_paw_x'].isna().mean()
    return {'lick_rate': dm['Lick count'].sum() / minutes,
            'whisk_me': dm['whisker_me'].mean(),
            'paw_speed': speed.mean() if paw_nan <= MAX_PAW_NAN else np.nan,
            'paw_nan': paw_nan, 'minutes': minutes, 'fs': round(fs)}


NAME = re.compile(r'design_matrix_([0-9a-f-]{36})_(.+)$')
cached = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=['timepoint', 'eid'])
done = set(zip(cached['timepoint'], cached['eid']))
rows = []
for tp, folder in TIMEPOINTS.items():
    for f in sorted(folder.glob('design_matrix_*')):
        m = NAME.match(f.name)
        if m and (tp, m.group(1)) not in done:
            rows.append({'timepoint': tp, 'eid': m.group(1), 'mouse_name': m.group(2),
                         **session_features(f)})
sessions = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
sessions.to_csv(CACHE, index=False)

# Habituation: drop signals the per-signal checks flagged, and attach day + lab
hab_log = pd.read_csv(HAB / 'processing_log.csv').set_index('eid')
inventory = pd.read_csv(HERE / 'habituation_lp_sessions.csv').set_index('eid')
is_hab = sessions['timepoint'] == 'Habituation'
sessions = sessions[~is_hab | sessions['eid'].isin(hab_log.index[hab_log['status'] == 'ok'])].copy()
is_hab = sessions['timepoint'] == 'Habituation'
for flag, feat in [('lick_ok', 'lick_rate'), ('whisk_ok', 'whisk_me'), ('paw_ok', 'paw_speed')]:
    bad = is_hab & ~sessions['eid'].map(hab_log[flag]).fillna(False).astype(bool)
    sessions.loc[bad, feat] = np.nan
sessions['hab_day'] = sessions['eid'].map(inventory['hab_day'])

lab = inventory.groupby('mouse_name')['lab'].first()     # only mice with habituation matter here
print(sessions.groupby('timepoint').agg(sessions=('eid', 'size'), mice=('mouse_name', 'nunique'),
                                        fs=('fs', 'median')).loc[ORDER].to_string())
print('\nhabituation sessions with each signal usable:',
      sessions.loc[is_hab, list(FEATURES)].notna().sum().to_dict())

# Per mouse, per timepoint
mouse = sessions.groupby(['timepoint', 'mouse_name'])[list(FEATURES)].mean()


#%%
# ---- correlation helpers ---------------------------------------------------------
def lab_residual_ranks(values, labs):
    """Ranks with each lab's mean rank removed."""
    r = pd.Series(rankdata(values), index=values.index)
    return r - r.groupby(labs.values).transform('mean')


def correlate(x, y, labs):
    """Spearman across mice, raw and lab-controlled (p by permuting mice within lab)."""
    raw = spearmanr(x, y)
    rx, ry = lab_residual_ranks(x, labs), lab_residual_ranks(y, labs)
    r_lab = np.corrcoef(rx, ry)[0, 1]
    groups = [np.where(labs.values == l)[0] for l in np.unique(labs.values)]
    ryv, null = ry.values, np.empty(N_PERM)
    for i in range(N_PERM):
        perm = np.arange(len(ryv))
        for g in groups:
            perm[g] = rng.permutation(g)
        null[i] = np.corrcoef(rx.values, ryv[perm])[0, 1]
    p_lab = (np.sum(np.abs(null) >= abs(r_lab)) + 1) / (N_PERM + 1)
    return {'n': len(x), 'n_labs': len(groups), 'rho': raw.statistic, 'p': raw.pvalue,
            'rho_lab': r_lab, 'p_lab': p_lab}


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


#%%
# ---- 1. stability within habituation: day to day ---------------------------------
hab = sessions[is_hab]
stab = []
for feat in FEATURES:
    wide = hab.pivot_table(index='mouse_name', columns='hab_day', values=feat)
    for a, b in [(1, 2), (2, 3), (1, 3)]:
        if a in wide and b in wide:
            pair = wide[[a, b]].dropna()
            if len(pair) >= 5:
                stab.append({'feature': feat, 'pair': f'day {a} vs {b}',
                             **correlate(pair[a], pair[b], lab.reindex(pair.index))})
stab = pd.DataFrame(stab)
print('\n== Day-to-day stability within habituation (Spearman across mice) ==')
print(stab.round(3).to_string(index=False))


#%%
# ---- 2. habituation vs later timepoints ------------------------------------------
cross = []
for feat in FEATURES:
    for tp in ORDER[1:]:
        both = pd.concat([mouse.loc['Habituation', feat], mouse.loc[tp, feat]], axis=1,
                         keys=['hab', 'later']).dropna()
        both = both[both.index.isin(lab.index)]
        cross.append({'feature': feat, 'timepoint': tp,
                      **correlate(both['hab'], both['later'], lab.reindex(both.index))})
cross = pd.DataFrame(cross)
cross['q'] = bh(cross['p'])
cross['q_lab'] = bh(cross['p_lab'])
print('\n== Habituation (mouse mean) vs later timepoints ==')
print(cross.round(3).to_string(index=False))

# Benchmark: the same correlation between consecutive later timepoints, on mice that
# also have habituation (so the lab control is identical).
bench = []
for feat in FEATURES:
    for a, b in [('Early', 'Late'), ('Late', 'Pre-rec'), ('Pre-rec', 'Proficient'),
                 ('Early', 'Proficient')]:
        both = pd.concat([mouse.loc[a, feat], mouse.loc[b, feat]], axis=1, keys=['a', 'b']).dropna()
        both = both[both.index.isin(lab.index)]
        if len(both) >= 5:
            bench.append({'feature': feat, 'pair': f'{a} vs {b}',
                          **correlate(both['a'], both['b'], lab.reindex(both.index))})
bench = pd.DataFrame(bench)
print('\n== Benchmark: later timepoint pairs, same mice pool ==')
print(bench.round(3).to_string(index=False))


#%%
# ---- 3. habituation vs learning speed --------------------------------------------
# Learning speed = training_days, the file the rest of the paper uses (loaded as in
# learning_individuality/syllable_correlations.ipynb), with its exclusion: log_training of
# exactly 0 or 1.
training = pd.read_parquet(REPO / 'learning_prediction' / 'training_time_05-05-2026')
training['log_training'] = np.log(training['training_days'])
training = training.loc[~training['log_training'].isin([0, 1])]
speed = training.set_index('mouse_name')['training_days']
print(f'\nlearning speed: {len(speed)} mice, median {int(speed.median())} training days '
      f'(range {speed.min()}-{speed.max()})')

learn = []
for tp in ['Habituation', 'Early']:          # Early = the benchmark the paper already uses
    for feat in FEATURES:
        both = pd.concat([mouse.loc[tp, feat], np.log(speed)], axis=1, keys=['x', 'y']).dropna()
        both = both[both.index.isin(lab.index)]
        learn.append({'timepoint': tp, 'feature': feat,
                      **correlate(both['x'], both['y'], lab.reindex(both.index))})
learn = pd.DataFrame(learn)
learn['q'] = bh(learn['p'])
learn['q_lab'] = bh(learn['p_lab'])
print('\n== Behaviour vs learning speed (log training_days; + = slower learner) ==')
print(learn.round(3).to_string(index=False))


#%%
# ---- figure: lab-controlled ranks, habituation vs each timepoint -----------------
ps.use('paper')
fig, axes = plt.subplots(len(FEATURES), len(ORDER), figsize=(2.1 * len(ORDER), 2.0 * len(FEATURES)),
                         constrained_layout=True)
for i, feat in enumerate(FEATURES):
    # column 0: habituation day 1 vs day 2
    wide = hab.pivot_table(index='mouse_name', columns='hab_day', values=feat)
    panels = [('Habituation', wide[[1, 2]].dropna().set_axis(['x', 'y'], axis=1)
               if {1, 2} <= set(wide.columns) else pd.DataFrame(columns=['x', 'y']), 'Hab day 1', 'Hab day 2')]
    for tp in ORDER[1:]:
        both = pd.concat([mouse.loc['Habituation', feat], mouse.loc[tp, feat]], axis=1,
                         keys=['x', 'y']).dropna()
        panels.append((tp, both[both.index.isin(lab.index)], 'Habituation', tp))
    for j, (tp, d, xl, yl) in enumerate(panels):
        ax = axes[i, j]
        if len(d) >= 5:
            labs = lab.reindex(d.index)
            rx, ry = lab_residual_ranks(d['x'], labs), lab_residual_ranks(d['y'], labs)
            ax.scatter(rx, ry, s=14, color=COLOR[tp], edgecolor='white', linewidth=0.5)
            res = correlate(d['x'], d['y'], labs)
            ax.set_title(f'ρ = {res["rho"]:.2f}   ρ$_{{lab}}$ = {res["rho_lab"]:.2f}\n'
                         f'n = {res["n"]}, p$_{{lab}}$ = {res["p_lab"]:.3f}', fontsize=7)
        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl if j else f'{FEATURES[feat]}\n{yl}', fontsize=7)
fig.suptitle('Habituation vs later behaviour: lab-residualised ranks across mice', fontsize=9)
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'habituation_stability_raw_signals.png', dpi=200)
plt.show()

# %%
