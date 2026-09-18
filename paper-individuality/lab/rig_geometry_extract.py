"""
PER-SESSION RIG GEOMETRY, FROM THE LOCAL ONE CACHE
==================================================
Writes rig_geometry.csv: one row per session, the geometry terms that
lda_rig_geometry_validation.py defines, for every session of the LDA set whose
leftCamera DLC file is already on this machine.

DIFFERENCES FROM lda_rig_geometry_validation.py, and why:
  * reads the LOCAL ONE cache (one.eid2path) instead of a hardcoded Mac download
    folder, so it runs here;
  * takes the session list from functions.build_design_matrix, so the rows line up
    with the variance analysis rather than with a separately rebuilt LDA;
  * SKIPS LI_within. The Welch band-power laterality is the expensive part and it
    answers a different question (is LD1 contaminated by paw asymmetry) than the one
    this file feeds (how much of the lab variance component is measurement geometry).
    The helper functions `conf` and `med_dist` are imported from that script rather
    than re-implemented, so the geometry definitions cannot drift apart.
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lda_rig_geometry_validation import conf, med_dist, KP, COLS      # same definitions
import functions

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
OUT = HERE / 'rig_geometry.csv'   # nose_to_pupil_px added 18-09-2026


def session_geometry(alf):
    f = os.path.join(alf, '_ibl_leftCamera.dlc.pqt')
    if not os.path.exists(f):
        return None
    have = pd.read_parquet(f).columns if False else pd.read_parquet(f, columns=None).columns
    d = pd.read_parquet(f, columns=[c for c in COLS if c in have])

    fps = np.nan
    tpath = os.path.join(alf, '_ibl_leftCamera.times.npy')
    if os.path.exists(tpath):
        t = np.load(tpath)
        t = t[np.isfinite(t)]
        if len(t) > 100:
            fps = float(1.0 / np.median(np.diff(t)))

    rec = dict(fps=fps, n_frames=len(d))
    rec['tube_len_px'] = med_dist(d, 'tube_top', 'tube_bottom')
    rec['nose_to_tube_px'] = med_dist(d, 'nose_tip', 'tube_top')
    pv = med_dist(d, 'pupil_top_r', 'pupil_bottom_r')
    ph = med_dist(d, 'pupil_left_r', 'pupil_right_r')
    rec['pupil_diam_px'] = np.nanmean([pv, ph])

    # NOSE -> PUPIL: the better zoom ruler. The lick tube is NOT fixed hardware -- it can be
    # cut longer or shorter between rigs -- so its apparent length confounds zoom with how
    # someone trimmed the spout. Nose tip to pupil centre is a distance on the ANIMAL, at
    # roughly the depth the paws move in, so it scales with magnification and not with rig
    # assembly. Pupil centre is the mean of the four pupil landmarks, each gated on
    # likelihood, rather than a single keypoint.
    _pupil = [f'pupil_{s}_r' for s in ('top', 'bottom', 'left', 'right')]
    _m = conf(d, 'nose_tip')
    for _k in _pupil:
        _m = _m & conf(d, _k)
    if _m.sum() > 500:
        _px = np.mean([d[f'{k}_x'][_m].to_numpy() for k in _pupil], axis=0)
        _py = np.mean([d[f'{k}_y'][_m].to_numpy() for k in _pupil], axis=0)
        rec['nose_to_pupil_px'] = float(np.median(np.hypot(
            d['nose_tip_x'][_m].to_numpy() - _px, d['nose_tip_y'][_m].to_numpy() - _py)))
    else:
        rec['nose_to_pupil_px'] = np.nan
    for kp, short in [('nose_tip', 'nose'), ('tube_top', 'tube'),
                      ('paw_r', 'pawNear'), ('paw_l', 'pawFar')]:
        m = conf(d, kp)
        rec[f'{short}_x'] = float(np.median(d[f'{kp}_x'][m])) if m.sum() > 500 else np.nan
        rec[f'{short}_y'] = float(np.median(d[f'{kp}_y'][m])) if m.sum() > 500 else np.nan
    # tracking quality: the share of frames the tracker is confident about. A video-quality
    # term, not a geometry one, and the one most likely to differ between rigs.
    for kp, short in [('nose_tip', 'nose'), ('paw_r', 'pawNear'), ('paw_l', 'pawFar')]:
        rec[f'conf_{short}'] = float(conf(d, kp).mean())
    return rec


def main():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = ss.index.map(dd[['mouse_name', 'session']].drop_duplicates()
                         .set_index('session')['mouse_name'])
    from one.api import ONE
    one = ONE(mode='local')

    rows, missing = [], 0
    for i, (eid, mname) in enumerate(zip(ss.index, mouse), 1):
        p = one.eid2path(eid)
        if p is None:
            missing += 1
            continue
        q = str(p).split('/')
        alf = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf')
        try:
            g = session_geometry(alf)
        except Exception as e:
            print(f'  !! {eid[:8]}: {type(e).__name__}: {e}')
            continue
        if g is None:
            missing += 1
            continue
        g.update(session=eid, mouse_name=mname, lab=q[-5])
        rows.append(g)
        if i % 25 == 0:
            print(f'  ... {i}/{len(ss)} scanned, {len(rows)} measured', flush=True)

    df = pd.DataFrame(rows).set_index('session')
    df.to_csv(OUT)
    print(f'\n{len(df)} sessions ({missing} without a local DLC file), '
          f'{df.mouse_name.nunique()} mice, {df.lab.nunique()} labs')
    print('saved', OUT)


if __name__ == '__main__':
    main()
