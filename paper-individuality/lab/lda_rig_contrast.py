"""
EXPLOITING WITHIN-LAB RIG DIFFERENCES
=====================================
The problem with the previous validation: rig, lab and mouse are aliased, so
controlling for lab removes the rig variance and the test becomes circular.

The way out is a lab that used MORE THAN ONE RIG. Then rig varies WITHIN lab and
the two can be separated. angelakilab is one: its sessions fall into sharply
distinct camera framings (nose_y ~180 vs ~525 vs ~677 -- not a nudge, a different
aim or a different sensor). This script finds every such case across all 11 labs.

TWO FIXES TO THE PREVIOUS RUN
  1. zoom proxy. tube_len_px was wrong: the lick tube's APPARENT length depends on
     its angle to the camera and on how much of it is occluded, and the evidence
     said so -- under a real zoom change every distance scales together, but
     tube_len correlated only +0.12 / -0.07 with nose-to-tube and pupil diameter.
     nose_tip -> pupil centre is a MOUSE-anatomical distance: physically fixed for
     a given mouse, on the face, viewed near face-on. Within a mouse, any change
     in it is optical, not anatomical -- which is exactly the zoom measurement the
     within-mouse test needs.
  2. image extent. A different framing can mean a different camera AIM or a
     different SENSOR/crop. Those are not the same thing: a resolution change also
     breaks the RESOLUTION['left'] = 2 assumption in get_speed. The 99th percentile
     of the keypoint coordinates separates them.
"""
import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = '/Users/ineslaranjeira/Downloads/FlatIron'
LIK = 0.9
OUT = HERE / 'lda_rig_contrast.csv'
PUPIL = ['pupil_top_r', 'pupil_bottom_r', 'pupil_left_r', 'pupil_right_r']


def conf(d, kp):
    c = f'{kp}_likelihood'
    m = np.isfinite(d[f'{kp}_x']) & np.isfinite(d[f'{kp}_y'])
    if c in d:
        m &= d[c].to_numpy() > LIK
    return m


def extra_geometry(alf):
    f = os.path.join(alf, '_ibl_leftCamera.dlc.pqt')
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    rec = {}

    # pupil centre = mean of the four rim points, on frames where all four are confident
    m = np.ones(len(d), bool)
    for kp in PUPIL:
        if f'{kp}_x' not in d:
            return None
        m &= conf(d, kp)
    if m.sum() < 500:
        return None
    px = np.mean([d[f'{kp}_x'][m].to_numpy() for kp in PUPIL], axis=0)
    py = np.mean([d[f'{kp}_y'][m].to_numpy() for kp in PUPIL], axis=0)
    rec['pupil_x'], rec['pupil_y'] = float(np.median(px)), float(np.median(py))

    # nose -> pupil: fixed physical distance for a given mouse, so within-mouse
    # variation in it is optical (zoom / working distance), not anatomical
    mn = m & conf(d, 'nose_tip')
    if mn.sum() < 500:
        return None
    pxn = np.mean([d[f'{kp}_x'][mn].to_numpy() for kp in PUPIL], axis=0)
    pyn = np.mean([d[f'{kp}_y'][mn].to_numpy() for kp in PUPIL], axis=0)
    rec['nose_to_pupil_px'] = float(np.median(
        np.hypot(d['nose_tip_x'][mn].to_numpy() - pxn,
                 d['nose_tip_y'][mn].to_numpy() - pyn)))

    # sensor/crop extent vs camera aim
    xs = np.concatenate([d[c].to_numpy() for c in d.columns if c.endswith('_x')])
    ys = np.concatenate([d[c].to_numpy() for c in d.columns if c.endswith('_y')])
    rec['img_x99'] = float(np.nanpercentile(xs, 99.9))
    rec['img_y99'] = float(np.nanpercentile(ys, 99.9))
    return rec


def main():
    base = pd.read_csv(HERE / 'lda_rig_geometry_validation.csv', index_col=0)
    from one.api import ONE
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)

    print(f"rescanning {len(base)} sessions for nose->pupil and image extent")
    rows = []
    for i, eid in enumerate(base.index, 1):
        p = one.eid2path(eid)
        if p is None:
            continue
        q = p.parts
        alf = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf')
        try:
            g = extra_geometry(alf)
        except Exception:
            g = None
        if g is None:
            continue
        g['session'] = eid
        g['date'] = str(q[-2])
        rows.append(g)
        if i % 75 == 0:
            print(f'  ... {i}/{len(base)}, {len(rows)} measured')
    ex = pd.DataFrame(rows).set_index('session')
    df = base.join(ex, how='inner')
    df.to_csv(OUT)
    print(f"✓ {len(df)} sessions with the extra measures; saved {OUT}")
    return df


if __name__ == '__main__':
    main()
