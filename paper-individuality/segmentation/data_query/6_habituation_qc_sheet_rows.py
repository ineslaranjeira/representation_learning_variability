"""
6. Append the habituation sessions to the curated QC sheet.
=========================================================================================
`individuality-paper_data_<date>.csv` (paper-individuality/data/) is the sheet every LDA
analysis filters on, via `learning_individuality/session_filters.py`. Its automatic columns
are produced by `session_qc_overview.py`; the `Mayo's algo` and `Data inclusion` block is
filled in by hand afterwards. This script adds one row per habituation session, filling
exactly the columns `session_qc_overview.describe()` fills and leaving the hand-curated
ones empty.

Only the sessions with an mp4 go in (`has_raw_leftCam`, 144 of 152). The other 8 are left
out because the sheet could not show what they are: `classify_datatypes` keys `video` off
any dataset name containing 'camera', and those sessions still carry
`_iblrig_leftCamera.timestamps.ssv`, so they come out as `public_video = private` exactly
like a session with real video. `video_qc_left` is blank on them, but it is also blank on
54 sessions that do have an mp4 and were simply never QC'd -- so nothing in the sheet would
tell the two apart. A session with no video has no inclusion decision to record anyway;
`habituation_sessions.csv` keeps the row.

This is also why `session_qc_overview.py` does NOT list habituation in its INPUT_FILES:
it takes every eid in an input file, so it would add all 152. Habituation rows come from
here, and only from here.

It does NOT re-run `session_qc_overview.py`. That script makes one `sessions/read` call per
eid; `4_habituation_sessions_video_qc.py` already cached the identical payload for every
habituation session in `habituation_alyx_cache.pkl`, so every column but the four
`public_*` ones is recovered offline. The public lookup is the one live query, and it is a
single bulk `pub.search()` plus one `datasets/list` per publicly released session.

The output is a NEW dated sheet. The old one is never touched -- it is a hand-curated
artifact, and the rows added here carry an empty `Used in paper`, which no strictness level
excludes. Nothing changes for any existing analysis until `session_filters.CSV_NAME` is
pointed at the new file.

Two things the sheet cannot say about these sessions, both structural rather than missing:

  * 8 of the 10 `_task_*` columns do not exist for habituationChoiceWorld. The protocol
    runs its own QC set (`_task_habituation_time`, `_task_phase`, `_task_stimCenter_delays`,
    ...); only `_task_stimOn_goCue_delays` and `_task_reward_volumes` are shared with
    choiceWorld. The other eight are blank because the check was never defined for this
    protocol, not because it was skipped. The overall `task_qc` verdict is populated and is
    the column to curate on.
  * Every `_lightningPose*` and `_videoRight_*` column is blank. There is no pose estimation
    and no right camera on any habituation session in the cohort -- see the header of
    `4_habituation_sessions_video_qc.py`.

@author: Ines
"""
#%%
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = HERE.parent.parent / 'data'                 # paper-individuality/data/

SHEET_IN = DATA_DIR / 'individuality-paper_data_4Sep26.csv'
SHEET_OUT = DATA_DIR / f'individuality-paper_data_{date.today():%-d%b%y}.csv'
CACHE_FILE = HERE / 'habituation_alyx_cache.pkl'
SESSIONS_CSV = HERE / 'habituation_sessions.csv'       # written by script 4; the eid list

SOURCE_FILE = SESSIONS_CSV.name    # what goes in the sheet's `source_file` column
QUERY_PUBLIC = True                # False leaves the four public_* columns empty (offline)

# Same metric lists as session_qc_overview.py -- the sheet's column order depends on them.
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

# Filled by hand in the spreadsheet, never by a script. Written empty.
MANUAL_COLS = ['Results', 'Run', 'Outcome', 'Used in paper', 'Notes']

DATATYPES = ['task', 'video', 'lp', 'ephys']


#%%
# ---- verbatim from session_qc_overview.py -----------------------------------------
def protocol_word(proto):
    p = (proto or '').lower()
    if 'ephys' in p:
        return 'ephys'
    if 'biased' in p:
        return 'biased'
    if 'habituation' in p:          # added here: habituationChoiceWorld matched 'other'
        return 'habituation'
    if 'training' in p:
        return 'training'
    return 'other'


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


def describe(det, source_file, public_eids, pub=None):
    """session_qc_overview.describe(), fed a cached `sessions/read` payload."""
    eid = det['id']
    ext = det.get('extended_qc') or {}

    internal = classify_datatypes([r.get('name', '')
                                   for r in (det.get('data_dataset_session_related') or [])])
    is_public = eid in public_eids
    if is_public and pub is not None:
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
        'mouse_name': det.get('subject'),
        'date': str(det.get('start_time'))[:10],
        'task_protocol': protocol_word(det.get('task_protocol')),
        'rig_name': det.get('location') or (det.get('json') or {}).get('PYBPOD_BOARD'),
        'public_task': pub_state('task'),
        'public_video': pub_state('video'),
        'public_lp': pub_state('lp'),
        'public_ephys': pub_state('ephys'),
        'lp_status': 'available' if internal['lp'] else 'missing',
        'task_qc': ext.get('task'),
        'video_qc_left': ext.get('videoLeft'),
        'video_qc_right': ext.get('videoRight'),
        'alyx_qc': f'https://alyx.internationalbrainlab.org/ibl_reports/gallery/{eid}/gallery',
    }
    for col in MANUAL_COLS:
        row[col] = np.nan
    for m in TASK_METRICS + VIDEO_METRICS:
        row[m] = ext.get(m)
    return row


#%%
# ---- STEP 1: the rows to add ------------------------------------------------------
records = pickle.load(open(CACHE_FILE, 'rb'))
by_eid = {d['id']: d for dets in records.values() for d in dets}
_sess = pd.read_csv(SESSIONS_CSV)
wanted = _sess.loc[_sess['has_raw_leftCam'], 'eid'].tolist()   # the filter is a column
missing = [e for e in wanted if e not in by_eid]
if missing:
    raise KeyError(f'{len(missing)} eids in {SESSIONS_CSV.name} are not in {CACHE_FILE.name} -- '
                   'rerun 4_habituation_sessions_video_qc.py with REFRESH=True')
print(f'{len(wanted)} habituation eids with an mp4 to describe '
      f'({len(_sess) - len(wanted)} with none left out, of {len(by_eid)} cached)')

public_eids, pub = set(), None
if QUERY_PUBLIC:
    from one.api import ONE
    pub = ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)
    public_eids = {str(x) for x in pub.search()}
    print(f'{len(public_eids)} sessions in the public database (openalyx); '
          f'{len(set(wanted) & public_eids)} of ours')

new = pd.DataFrame([describe(by_eid[e], SOURCE_FILE, public_eids, pub) for e in wanted])


#%%
# ---- STEP 2: append, preserving the two-row header --------------------------------
# The sheet has merged spreadsheet cells in row 0, so it is read with header=1 and row 0
# is carried across verbatim -- reading it with a single header would rename every column.
group_header = (pd.read_csv(SHEET_IN, header=None, nrows=1, dtype=str)
                .iloc[0].fillna('').tolist())
old = pd.read_csv(SHEET_IN, header=1, dtype=str)
old.columns = [c.strip() for c in old.columns]

unexpected = [c for c in new.columns if c not in old.columns]
if unexpected:
    raise ValueError(f'columns not present in {SHEET_IN.name}: {unexpected}')
already = sorted(set(new['eid']) & set(old['eid'].dropna()))
if already:
    print(f'!! {len(already)} eids are already in the sheet and will be skipped')
    new = new[~new['eid'].isin(already)]

out = pd.concat([old, new.reindex(columns=old.columns)], ignore_index=True)

with open(SHEET_OUT, 'w', newline='') as fh:
    fh.write(','.join(group_header) + '\n')
    out.to_csv(fh, index=False)

print(f'wrote {SHEET_OUT.name}: {len(old)} existing + {len(new)} habituation = {len(out)} rows')
print(f'  filled       : {[c for c in new.columns if c not in MANUAL_COLS and new[c].notna().any()]}')
print(f'  left for you : {MANUAL_COLS}')
print(f'  blank (check does not exist for this protocol): '
      f'{[c for c in TASK_METRICS + VIDEO_METRICS if new[c].isna().all()]}')
print(f'\nTo put it in front of the analyses, point session_filters.CSV_NAME at it and add\n'
      f"  '{SOURCE_FILE}': 'Habituation'\nto SOURCE_TO_TIMEPOINT.")

# %%
