"""
GEOMETRY AND PAW LATERALITY FROM lightningPose, FOR EVERY TIMEPOINT
====================================================================
Replaces the DLC-based extraction. The design matrices and states files are built from
lightningPose, so measuring geometry or paw speed from DLC mixes two trackers with different
error structure -- fine for a naming check, wrong for anything quantitative that is then
compared against LP-derived features.

LP carries every landmark needed: nose_tip, tube_top/bottom, paw_l, paw_r, the four pupil
points and the two tongue points. (The note in lda_rig_geometry_validation.py that LP lacks
tube and nose is out of date for these files.)

STREAMS TO KEEP DISK FLAT. Ephys LP files average ~147 MB and only 46 of 269 are cached, so
the rest are fetched one at a time, summarised, and the parquet deleted immediately -- peak
cost is one file. Files that were ALREADY on disk are never deleted.

Covers all four timepoints in one pass: training (Early, Late), biased (Pre-rec) and ephys
(Proficient), so every timepoint is measured with the same tracker, the same camera and the
same convention.
"""
import os
import sys
import glob
import pathlib
import argparse
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions
from session_filters import find_csv

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
FN = '_ibl_leftCamera.lightningPose.pqt'
LIK = 0.9
PUPIL = [f'pupil_{s}_r' for s in ('top', 'bottom', 'left', 'right')]
OUT = HERE / 'lp_session_metrics.csv'


def conf(d, kp):
    m = np.isfinite(d[f'{kp}_x']) & np.isfinite(d[f'{kp}_y'])
    c = f'{kp}_likelihood'
    if c in d:
        m &= d[c].to_numpy() > LIK
    return m.to_numpy()


def med_dist(d, a, b):
    m = conf(d, a) & conf(d, b)
    if m.sum() < 500:
        return np.nan
    return float(np.median(np.hypot(d[f'{a}_x'][m] - d[f'{b}_x'][m],
                                    d[f'{a}_y'][m] - d[f'{b}_y'][m])))


def metrics(path):
    d = pd.read_parquet(path)
    rec = {'n_frames': len(d)}
    # --- geometry
    rec['tube_len_px'] = med_dist(d, 'tube_top', 'tube_bottom')
    rec['nose_to_tube_px'] = med_dist(d, 'nose_tip', 'tube_top')
    pv, ph = med_dist(d, 'pupil_top_r', 'pupil_bottom_r'), med_dist(d, 'pupil_left_r', 'pupil_right_r')
    rec['pupil_diam_px'] = np.nanmean([pv, ph])
    m = conf(d, 'nose_tip')
    for k in PUPIL:
        m = m & conf(d, k)
    if m.sum() > 500:
        px = np.mean([d[f'{k}_x'][m].to_numpy() for k in PUPIL], axis=0)
        py = np.mean([d[f'{k}_y'][m].to_numpy() for k in PUPIL], axis=0)
        rec['nose_to_pupil_px'] = float(np.median(np.hypot(
            d['nose_tip_x'][m].to_numpy() - px, d['nose_tip_y'][m].to_numpy() - py)))
    else:
        rec['nose_to_pupil_px'] = np.nan
    for kp, short in [('nose_tip', 'nose'), ('tube_top', 'tube'),
                      ('paw_r', 'pawNear'), ('paw_l', 'pawFar')]:
        c = conf(d, kp)
        rec[f'{short}_x'] = float(np.median(d[f'{kp}_x'][c])) if c.sum() > 500 else np.nan
        rec[f'{short}_y'] = float(np.median(d[f'{kp}_y'][c])) if c.sum() > 500 else np.nan
        rec[f'conf_{short}'] = float(c.mean())
    # --- paw speed and laterality, both paws from THIS camera (one convention everywhere)
    sp = {}
    for paw in ['paw_r', 'paw_l']:
        c = conf(d, paw)
        v = np.hypot(np.diff(d[f'{paw}_x'].to_numpy()), np.diff(d[f'{paw}_y'].to_numpy()))
        v = v[c[1:]]
        sp[paw] = float(np.nanmean(v)) if len(v) else np.nan
        rec[f'speed_{paw}'] = sp[paw]
    a, b = sp['paw_r'], sp['paw_l']
    rec['li_paw'] = (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and (a + b) > 0 else np.nan
    rec['vigor_px'] = np.nanmean([a, b])
    return rec


def alf_dir(one, eid):
    p = one.eid2path(eid)
    if p is None:
        return None
    q = str(p).split('/')
    return os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true',
                    help='fetch missing LP files, summarise, then delete what was fetched')
    args = ap.parse_args()

    c = pd.read_csv(find_csv(), header=1)
    c = c[c['Used in paper'].astype(str).str.lower() != 'filtered out']
    tp_of = {'training': 'Training', 'biased': 'Pre-rec', 'ephys': 'Proficient'}
    todo = [(r.eid, r.mouse_name, tp_of.get(r.task_protocol)) for r in c.itertuples()
            if tp_of.get(r.task_protocol)]

    from one.api import ONE
    one = ONE()
    done = pd.read_csv(OUT).set_index('session') if OUT.exists() else None
    rows = [] if done is None else done.reset_index().to_dict('records')
    seen = set() if done is None else set(done.index)

    got = skipped = absent = 0
    for i, (eid, mouse, tp) in enumerate(todo, 1):
        if eid in seen:
            continue
        d = alf_dir(one, eid)
        if d is None:
            absent += 1
            continue
        f = os.path.join(d, FN)
        had = os.path.exists(f)
        if not had:
            if not args.download:
                absent += 1
                continue
            try:
                f = str(one.load_dataset(eid, FN, collection='alf', download_only=True))
            except Exception:
                absent += 1
                continue
            got += 1
        else:
            skipped += 1
        try:
            rec = metrics(f)
        except Exception as e:
            print(f'  !! {eid[:8]}: {type(e).__name__}', flush=True)
            rec = None
        if not had and os.path.exists(f):
            os.remove(f)                      # only ever delete what this run downloaded
        if rec:
            rec.update(session=eid, mouse=mouse, timepoint=tp)
            rows.append(rec)
        if i % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f'  {i}/{len(todo)}  measured {len(rows)}  downloaded {got}  '
                  f'already local {skipped}  unavailable {absent}', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(f'\n{len(R)} sessions measured from lightningPose -> {OUT}')
    print(R.groupby('timepoint').agg(sessions=('session', 'nunique'), mice=('mouse', 'nunique'),
                                     li=('li_paw', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
