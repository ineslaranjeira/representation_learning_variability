"""
Wheel-only design matrices for every training session of the individuality cohort.

One parquet per session, sampled at FS (200 Hz), no video:
    design_matrix_{eid}_{mouse}   times (s, float64), position (rad), velocity (rad/s)
    trials_{eid}_{mouse}          the session's ALF trials table
plus manifest.csv (one row per session: status, n samples, duration).

Position is the raw encoder trace interpolated at 1 kHz and taken every 5th sample.
Velocity is computed at 1 kHz with brainbox's zero-phase 20 Hz low-pass (as SessionLoader),
then taken every 5th sample; the filter removes everything above 20 Hz, so there is no aliasing.
Position keeps content up to 100 Hz, so velocity can be recomputed with another filter later.
Nothing is segmented here.

Run (memory-capped):
    systemd-run --user --scope -p MemoryMax=12G python wheel_design_matrices.py [--test]
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent / 'paper-individuality'
TRIALS_DIR = PI / 'data' / 'training' / 'training_data'
OUT_DIR = PI / 'data' / 'training' / 'training_wheel_200Hz'

FS_INTERP = 1000        # Hz; interpolation + filtering rate (SessionLoader default)
FS = 200                # Hz; saved rate
CORNER = 20             # Hz; velocity low-pass, as SessionLoader
ORDER = 8
N_JOBS = 4


def training_sessions():
    """(mouse, eid, date, session_index) for every training session in the cohort."""
    rows = []
    for f in sorted(TRIALS_DIR.glob('training_data_trials_*')):
        mouse = f.name.replace('training_data_trials_', '')
        t = pd.read_parquet(f, columns=['session', 'session_date'])
        d = pd.to_datetime(t['session_date']).groupby(t['session']).min().sort_values()
        for i, (eid, date) in enumerate(d.items()):
            rows.append((mouse, eid, date, i + 1))
    return pd.DataFrame(rows, columns=['mouse', 'eid', 'session_date', 'session_index'])


def process(mouse, eid):
    out = OUT_DIR / f'design_matrix_{eid}_{mouse}'
    row = dict(mouse=mouse, eid=eid, status='ok', n_samples=np.nan, duration_s=np.nan, n_trials=np.nan)
    if out.exists():
        row['status'] = 'exists'
        return row
    from one.api import ONE
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered
    one = ONE(mode='remote')
    try:
        w = one.load_object(eid, 'wheel', collection='alf')
        if w['position'].shape[0] != w['timestamps'].shape[0]:
            raise ValueError('wheel position/timestamps length mismatch')
        pos, t = interpolate_position(w['timestamps'], w['position'], freq=FS_INTERP)
        vel, _ = velocity_filtered(pos, fs=FS_INTERP, corner_frequency=CORNER, order=ORDER)
        step = FS_INTERP // FS
        dm = pd.DataFrame({'times': t[::step].astype(np.float64),
                           'position': pos[::step].astype(np.float32),
                           'velocity': vel[::step].astype(np.float32)})
        trials = one.load_object(eid, 'trials', collection='alf')
        n = len(trials['intervals'])
        # some sessions carry an empty attribute (e.g. quiescencePeriod); drop it rather than fail
        trials = type(trials)({k: v for k, v in trials.items() if len(v) == n}).to_df()
        trials.to_parquet(OUT_DIR / f'trials_{eid}_{mouse}', compression='zstd')
        dm.to_parquet(out, compression='zstd')
        row.update(n_samples=len(dm), duration_s=dm['times'].iloc[-1] - dm['times'].iloc[0],
                   n_trials=len(trials))
    except Exception as e:
        row['status'] = f'error: {type(e).__name__}: {e}'[:300]
    return row


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = training_sessions()
    if '--test' in sys.argv:
        sessions = sessions.iloc[[0, 500, 1500]]
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process)(m, e) for m, e in zip(sessions['mouse'], sessions['eid']))
    man = sessions.merge(pd.DataFrame(rows), on=['mouse', 'eid'])
    f = OUT_DIR / 'manifest.csv'
    if f.exists():      # keep earlier rows for sessions this run skipped as 'exists'
        old = pd.read_csv(f)
        keep = old[~old['eid'].isin(man.loc[man['status'] != 'exists', 'eid'])]
        man = pd.concat([keep, man[man['status'] != 'exists']]).drop_duplicates('eid', keep='last')
    man.to_csv(f, index=False)
    print(man['status'].str.split(':').str[0].value_counts())
