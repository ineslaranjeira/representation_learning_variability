"""
0. Habituation sessions with LightningPose: what exists for each one.
=========================================================================================
Supersedes the availability part of segmentation/data_query/4_habituation_sessions_video_qc.py,
which was written before LightningPose was run on habituation (its "no pose on any
habituation session" no longer holds) and whose output CSVs are not on disk.

One Alyx `datasets/list` query per dataset name, filtered to habituation protocols, so the
whole inventory is ~15 REST calls. Writes `habituation_lp_sessions.csv` next to this file:
one row per habituation session with a leftCamera LightningPose file.

What it showed on 2026-10-06 (138 sessions, 49 mice):
  * pose + timing are there: leftCamera LP 138, leftCamera.times 136, whisker-pad
    ROIMotionEnergy 136.
  * ALF trials only on 71; the rest need extraction from _iblrig_taskData.raw.jsonable
    (present on all 138).
  * NO WHEEL on any iblrig-v6 session: no ALF wheel anywhere, and the raw
    _iblrig_encoderPositions.raw.ssv exists only on the 19 iblrig-v8 / NPH2 sessions
    (MM*, ZFM-04019, ZFM-04026), none of which are LDA-cohort mice.

@author: Ines
"""
#%%
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
PAPER_DATA = HERE.parent / 'paper-individuality' / 'data'
LDA_SYLLABLE_FILE = PAPER_DATA / '8_k_10_bin_syllables_19-08-2026'   # mice of the LDA cohort
OUT = HERE / 'habituation_lp_sessions.csv'

# column name -> substring of the dataset name
DATASETS = {
    'lp_left': '_ibl_leftCamera.lightningPose',
    'lp_right': '_ibl_rightCamera.lightningPose',
    'lp_body': '_ibl_bodyCamera.lightningPose',
    'cam_times_left': '_ibl_leftCamera.times',
    'rig_timestamps_left': '_iblrig_leftCamera.timestamps',
    'me_left': 'leftCamera.ROIMotionEnergy',
    'trials_alf': '_ibl_trials.stimOn_times',
    'wheel_alf': '_ibl_wheel.position',
    'encoder_positions_raw': '_iblrig_encoderPositions.raw',
    'task_data_raw': '_iblrig_taskData.raw',
}

one = ONE(mode='remote')


def sessions_with(name):
    ds = one.alyx.rest('datasets', 'list', no_cache=True,
                       django=f'name__icontains,{name},session__task_protocol__icontains,habituation')
    return {d['session'].split('/')[-1] for d in ds}


#%%
have = {col: sessions_with(name) for col, name in DATASETS.items()}

sess = one.alyx.rest('sessions', 'list', task_protocol='habituation', no_cache=True,
                     django='data_dataset_session_related__name__icontains,lightningPose')
df = (pd.DataFrame([{'mouse_name': s['subject'], 'eid': s['id'], 'date': s['start_time'][:10],
                     'start_time': s['start_time'], 'lab': s['lab'],
                     'task_protocol': s['task_protocol']} for s in sess])
      .drop_duplicates('eid'))
df = df[df['eid'].isin(have['lp_left'])]
for col, eids in have.items():
    df[col] = df['eid'].isin(eids)

# Habituation day among the sessions WITH pose, not among all habituation sessions on Alyx
df = df.sort_values(['mouse_name', 'start_time']).reset_index(drop=True)
df['hab_day'] = df.groupby('mouse_name').cumcount() + 1
df['iblrig_v8'] = df['task_protocol'].str.contains(r'8\.\d', regex=True)

lda_mice = set(pd.read_parquet(LDA_SYLLABLE_FILE, columns=['mouse_name'])['mouse_name'])
df['in_lda_cohort'] = df['mouse_name'].isin(lda_mice)

df.drop(columns='start_time').to_csv(OUT, index=False)


#%%
print(f'{len(df)} habituation sessions with leftCamera LP, {df.mouse_name.nunique()} mice '
      f'({df.loc[df.in_lda_cohort, "mouse_name"].nunique()} in the LDA cohort of '
      f'{len(lda_mice)})')
print(df[list(DATASETS)].sum().to_string())
print('\nsessions per mouse (all / LDA cohort):')
print(pd.concat([df.groupby('mouse_name').size().value_counts().sort_index().rename('all'),
                 df[df.in_lda_cohort].groupby('mouse_name').size().value_counts()
                 .sort_index().rename('lda')], axis=1).fillna(0).astype(int).to_string())
print(f'\nwrote {OUT.name}')

# %%
