"""
Per-mouse session heatmap (same style as data_query_overview_principled) with
every session from session_qc_overview.csv marked on top, coloured by source_file.

Rows = mice with >=3 sessions in the bwm_qc_new pool (same set as the principled
plot). Fully offline: sessions missing from the cached session table are placed
using the CSV's own date + task_protocol, then each mouse is renumbered.

@author: Ines
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from matplotlib.lines import Line2D
from datetime import datetime

prefix = '/home/ines/repositories/'
repo = prefix + 'representation_learning_variability/'
dq = repo + 'paper-individuality/segmentation/data_query/'
sessions_cache = repo + 'Video and wheel/Video QC/all_session_details_bwm'
csv_file = dq + 'session_qc_overview.csv'
POOL_SOURCE = 'bwm_qc_new_08-03-2026'
MIN_SESSIONS = 3

# marker + colour per source_file (drawn on top of the dimmed protocol heatmap)
LAYERS = {
    'bwm_qc_new_08-03-2026':           {'m': 's', 'c': 'black',      'lbl': 'BWM pool (ephys)'},
    'first_training_eids.csv':         {'m': '^', 'c': 'tab:green',  'lbl': 'First training'},
    'last_training_eids.csv':          {'m': '*', 'c': 'magenta',    'lbl': 'Last training'},
    'biased_before_ephys_3_eids.csv':  {'m': 'o', 'c': 'teal',       'lbl': 'Biased -3 (pre-ephys)'},
    'biased_before_ephys_2_eids.csv':  {'m': 'o', 'c': 'deepskyblue','lbl': 'Biased -2 (pre-ephys)'},
    'biased_before_ephys_1_eids.csv':  {'m': 'o', 'c': 'blue',       'lbl': 'Biased -1 (closest to ephys)'},
}

#%%
csv = pd.read_csv(csv_file)
qual_counts = csv[csv['source_file'] == POOL_SOURCE]['mouse_name'].value_counts()
qualifying = sorted(qual_counts[qual_counts >= MIN_SESSIONS].index)
print(f'{len(qualifying)} qualifying mice (>= {MIN_SESSIONS} pool sessions).')

all_sessions = pd.read_csv(sessions_cache, compression='gzip').drop(columns=['Unnamed: 0'], errors='ignore')
all_sessions['session_start_time'] = pd.to_datetime(all_sessions['session_start_time'], errors='coerce', utc=True)

# --- place any CSV session of a qualifying mouse that's missing from the cache ---
cached = set(all_sessions['session'])
patch = []
for _, r in csv[csv['mouse_name'].isin(qualifying)].iterrows():
    if r['eid'] not in cached:
        patch.append({'subject': r['mouse_name'], 'session': r['eid'],
                      'session_start_time': pd.to_datetime(r['date'], utc=True),
                      'task_protocol': r['task_protocol_full'], 'training_status': np.nan})
if patch:
    all_sessions = pd.concat([all_sessions, pd.DataFrame(patch)], ignore_index=True)
print(f'{len(patch)} CSV sessions patched into the session table.')

#%%
# protocol colour code (identical to data_query_overview_principled)
def add_protocol_number(df):
    df = df.copy()
    df['protocol_number'] = np.nan
    df.loc[df['task_protocol'].str.contains('training', na=False), 'protocol_number'] = 0
    df.loc[df['task_protocol'].str.contains('biased', na=False), 'protocol_number'] = 1
    df.loc[df['task_protocol'].str.contains('ephys', na=False), 'protocol_number'] = 2
    df['training_status'] = df['training_status'].astype(str)
    for kw, v in [('trained 1b', 3), ('ready4ephysrig', 4), ('ready4delay', 5),
                  ('ready4recording', 6), ('unbiasable', 7), ('untrainable', 8)]:
        df.loc[df['training_status'].str.contains(kw), 'protocol_number'] = v
    return df

# renumber each qualifying mouse chronologically so patched rows get a slot
for s in qualifying:
    idx = all_sessions.index[all_sessions['subject'] == s]
    order = all_sessions.loc[idx].sort_values('session_start_time').index
    all_sessions.loc[order, 'session_number'] = np.arange(len(order))
all_sessions = add_protocol_number(all_sessions)

colors = ['#add8ff', '#ffd8a8', '#a8ffb0']
ps, pe = to_rgb('#d8b0ff'), to_rgb('#4b0082')
for i in range(4):
    t = i / 3
    colors.append(tuple(a + t * (b - a) for a, b in zip(ps, pe)))
colors += ['#ff9999', '#990000']
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], cmap.N)

row_of = {s: i for i, s in enumerate(qualifying)}
n_rows = len(qualifying)
max_sn = int(all_sessions.loc[all_sessions['subject'].isin(qualifying), 'session_number'].max()) + 1
protocol_matrix = np.full((n_rows, max_sn), np.nan)
for s in qualifying:
    sub = all_sessions[all_sessions['subject'] == s]
    protocol_matrix[row_of[s], sub['session_number'].astype(int)] = sub['protocol_number']

lookup = all_sessions.drop_duplicates('session').set_index('session')[['subject', 'session_number']]

#%%
fig, ax = plt.subplots(figsize=(22, 0.16 * n_rows))
ax.imshow(protocol_matrix, aspect='auto', cmap=cmap, norm=norm, alpha=0.30)
ax.set_xlabel('Session number'); ax.set_ylabel('Mouse')
ax.set_yticks(np.arange(n_rows)); ax.set_yticklabels(qualifying, fontsize=8)
ax.set_xticks(np.arange(-0.5, max_sn, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=0.2)
ax.tick_params(which='minor', length=0)

counts = {}
for src, spec in LAYERS.items():
    xs, ys = [], []
    for eid in csv[csv['source_file'] == src]['eid']:
        if eid in lookup.index:
            r = lookup.loc[eid]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            if r['subject'] in row_of and not np.isnan(r['session_number']):
                ys.append(row_of[r['subject']]); xs.append(int(r['session_number']))
    counts[src] = len(xs)
    if xs:
        ax.scatter(xs, ys, marker=spec['m'], s=90, facecolor='none', edgecolor=spec['c'],
                   linewidth=1.5, zorder=3)

handles = [Line2D([0], [0], marker=s['m'], color='none', markerfacecolor='none',
                  markeredgecolor=s['c'], markeredgewidth=1.6, markersize=11,
                  label=f"{s['lbl']} (n={counts.get(src, 0)})")
           for src, s in LAYERS.items()]
ax.legend(handles=handles, loc='upper right', framealpha=0.9, fontsize=9)

cbar = plt.colorbar(ax.images[0], ax=ax, ticks=range(9))
cbar.ax.set_yticklabels(['Training', 'Biased', 'Ephys', 'Trained', 'Ready4ephysrig',
                         'Ready4delay', 'Ready4recording', 'Unbiasable', 'Untrainable'])
ax.set_title(f'session_qc_overview.csv sessions on the per-mouse timeline '
             f'({n_rows} mice with >= {MIN_SESSIONS} pool sessions)')
plt.xlim([-.5, 150]); plt.tight_layout()
plt.savefig(dq + 'data_query_overview_csv.png', dpi=150, bbox_inches='tight')
plt.show()
print('marked per source_file:', counts)
# %%
