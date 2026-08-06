"""
Populate a per-session QC overview CSV, across one or more source files.

Columns: eid, source_file, mouse_name, date, public,
         public_task, public_video, public_lp, public_ephys  (public/private/NaN
           per datatype: available publicly / exists internally only / non-existent),
         lp_status, task_protocol (training/biased/ephys), task_protocol_full, lab,
         rig_name, task_qc, video_qc_left, video_qc_right, alyx_qc,
         + the detailed per-metric QC used in the 1_sessions_query_qc filter
           (10 _task_* metrics and the 14 left/right video + lightningPose metrics).

Each input file contributes its eids, tagged with its filename in `source_file`.
An eid already in the CSV is never re-described (first source wins), so files can
be added incrementally. All detailed QC is free: sessions/read returns the full
extended_qc in one call.

@author: Ines
"""
#%%
import os
import numpy as np
import pandas as pd
from one.api import ONE

prefix = '/home/ines/repositories/'
# prefix = '/Users/ineslaranjeira/Documents/Repositories/'
data_query_path = prefix + 'representation_learning_variability/paper-individuality/segmentation/data_query/'

# Source files to include, in priority order. Gzip pickles (no extension) or CSVs;
# each must expose an 'eid' column. Add more here to grow the CSV.
INPUT_FILES = [
    'bwm_qc_new_08-03-2026',      # gzip pickle
    'first_training_eids.csv',
    'last_training_eids.csv',
    'biased_before_ephys_1_eids.csv',   # last biased before ephys (closest)
    'biased_before_ephys_2_eids.csv',   # second-to-last
    'biased_before_ephys_3_eids.csv',   # third-to-last
]
out_csv = data_query_path + 'session_qc_overview.csv'   # output + incremental cache

one = ONE(base_url='https://alyx.internationalbrainlab.org', silent=True)

# Detailed QC metrics used by the 1_sessions_query_qc filter -------------------
TASK_METRICS = [
    '_task_stimOn_goCue_delays', '_task_response_feedback_delays',
    '_task_wheel_move_before_feedback', '_task_wheel_freeze_during_quiescence',
    '_task_error_trial_event_sequence', '_task_correct_trial_event_sequence',
    '_task_reward_volumes', '_task_reward_volume_set',
    '_task_stimulus_move_before_goCue', '_task_audio_pre_trial']
VIDEO_METRICS = [
    '_lightningPoseLeft_lick_detection', '_lightningPoseLeft_time_trace_length_match',
    '_videoLeft_pin_state', '_lightningPoseLeft_trace_all_nan', '_videoLeft_camera_times',
    '_videoLeft_dropped_frames', '_videoLeft_timestamps',
    '_lightningPoseRight_lick_detection', '_lightningPoseRight_time_trace_length_match',
    '_videoRight_pin_state', '_lightningPoseRight_trace_all_nan', '_videoRight_camera_times',
    '_videoRight_dropped_frames', '_videoRight_timestamps']


def protocol_word(proto):
    p = (proto or '').lower()
    if 'ephys' in p:
        return 'ephys'
    if 'biased' in p:
        return 'biased'
    if 'training' in p:
        return 'training'
    return 'other'


# Per-datatype presence, inferred from dataset relative paths / names.
DATATYPES = ['task', 'video', 'lp', 'ephys']
def classify_datatypes(names):
    p = [str(x).lower() for x in names]
    return {
        'task':  any('trials' in x for x in p),
        'video': any(('camera' in x or 'roimotionenergy' in x or 'dlc' in x) and 'lightningpose' not in x
                     for x in p),
        'lp':    any('lightningpose' in x for x in p),
        'ephys': any(k in x for x in p
                     for k in ['spikes', 'clusters', 'channels', 'pykilosort', '.ap.', '.lf.']),
    }


def load_eids(name):
    path = data_query_path + name
    df = pd.read_csv(path) if name.endswith('.csv') else pd.read_pickle(path, compression='gzip')
    return list(dict.fromkeys(df['eid']))

#%%
# Public-availability: one bulk lookup against the open (public) database
pub = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
public_eids = {str(x) for x in pub.search()}
print(f'{len(public_eids)} sessions in the public database (openalyx).')


def describe(eid, source_file):
    d = one.alyx.rest('sessions', 'read', id=eid)
    ext = d.get('extended_qc') or {}

    # internal datasets come free with the session read
    internal = classify_datatypes([r.get('name', '') for r in (d.get('data_dataset_session_related') or [])])
    # public datasets only exist for sessions in the public release
    is_public = eid in public_eids
    if is_public:
        try:
            pds = pub.alyx.rest('datasets', 'list', session=eid)
            public = classify_datatypes([r.get('name') or r.get('rel_path', '') for r in pds])
        except Exception:
            public = {t: False for t in DATATYPES}
    else:
        public = {t: False for t in DATATYPES}

    def pub_state(t):                              # public / private / NaN(non-existent)
        if public.get(t):
            return 'public'
        if internal.get(t):
            return 'private'
        return np.nan

    row = {
        'eid': eid,
        'source_file': source_file,
        'mouse_name': d.get('subject'),
        'date': str(d.get('start_time'))[:10],
        'public': 'public' if is_public else 'private',
        'public_task': pub_state('task'),
        'public_video': pub_state('video'),
        'public_lp': pub_state('lp'),
        'public_ephys': pub_state('ephys'),
        'lp_status': 'available' if internal['lp'] else 'missing',
        'task_protocol': protocol_word(d.get('task_protocol')),
        'task_protocol_full': d.get('task_protocol'),
        'lab': d.get('lab'),
        'rig_name': d.get('location') or (d.get('json') or {}).get('PYBPOD_BOARD'),
        'task_qc': ext.get('task'),
        'video_qc_left': ext.get('videoLeft'),
        'video_qc_right': ext.get('videoRight'),
        'alyx_qc': f'https://alyx.internationalbrainlab.org/ibl_reports/gallery/{eid}/gallery',
    }
    for m in TASK_METRICS + VIDEO_METRICS:
        row[m] = ext.get(m)
    return row

#%%
existing = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=['eid'])
seen = set(existing['eid']) if len(existing) else set()
print(f'{len(seen)} eids already in {os.path.basename(out_csv)}.')

new_rows = []
for src in INPUT_FILES:
    eids = load_eids(src)
    todo = [e for e in eids if e not in seen]
    print(f'{src}: {len(eids)} eids, {len(todo)} new to describe.')
    for i, eid in enumerate(todo):
        try:
            new_rows.append(describe(eid, src))
        except Exception as err:
            new_rows.append({'eid': eid, 'source_file': src, 'error': str(err)})
        seen.add(eid)
        if i % 25 == 0:
            pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True).to_csv(out_csv, index=False)
            print(f'  {src}: {i + 1}/{len(todo)}')

df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
df.to_csv(out_csv, index=False)
print(f'wrote {len(df)} rows ({len(new_rows)} newly described) -> {out_csv}')

#%%
# quick sanity summary
if 'lp_status' in df:
    print('\nby source_file:', dict(df['source_file'].value_counts(dropna=False)))
    print('public   :', dict(df['public'].value_counts(dropna=False)))
    for t in DATATYPES:
        print(f'public_{t:6}:', dict(df[f'public_{t}'].value_counts(dropna=False)))
    print('LP status:', dict(df['lp_status'].value_counts(dropna=False)))
    print('protocol :', dict(df['task_protocol'].value_counts(dropna=False)))
    print('task QC  :', dict(df['task_qc'].value_counts(dropna=False)))
    print('video L  :', dict(df['video_qc_left'].value_counts(dropna=False)))
    print('video R  :', dict(df['video_qc_right'].value_counts(dropna=False)))
    print('labs     :', df['lab'].nunique(), '| mice:', df['mouse_name'].nunique())

# %%
