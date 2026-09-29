"""
PER-SESSION RIG GEOMETRY: A PHYSICAL RULER, AND THE TEMPORAL GAIN PROXIES
=========================================================================
Lab-centring is a weak rig control here. `vigor/paw_bias` section 5 measured the
camera-gain term gamma and found its SESSION-level sd (0.092) exceeds its mean (0.089),
with only 22% of its variance between labs -- so centring within lab removes at most a
fifth of it. Mice are each recorded on one rig, so whatever is left inflates any
"mouse identity from raw amplitude" score.

THREE SESSION-LEVEL COVARIATES, all measured rather than assumed:

  tube_px      THE RULER. The lick tube is rig hardware of fixed physical size, so its
               apparent length in pixels IS that session's pixels-per-millimetre.
               Dividing paw amplitude by it converts px/s into physical units and
               removes camera-distance and rig-geometry differences directly.
               `vigor/paw_bias/measure_camera_scale.py` validated this reference:
               the left/right ratio it gives is 1.980 (p = 0.25 against 2.0), matching
               pupil diameter at 1.986.

  frame_rate   THE TEMPORAL GAIN. lightningPose's smoother has a time constant set per
               FRAME, so a 60 fps recording is smoothed over a 2.5x longer real-time
               window than a 150 fps one -- it removes ~28% of real 8 Hz movement at
               60 fps against ~4% at 150 fps. That is frequency-dependent gain a
               spatial ruler cannot touch.

  hf_noise     TRACKING QUALITY, from vigor_sessions_states.csv -- the 20-29 Hz band
               power of the paw speed trace, i.e. the noise floor.

WHICH TRACKER. The cohort mostly has DLC cached, not lightningPose (266 vs 49 of 332).
That is fine HERE and only here: the tube does not move, so measuring its length is not
a test of tracker quality. lightningPose is used where present and the two are compared
on the sessions that have both, so the mix is checkable rather than assumed.

COVERAGE HONESTY. `l_paw` comes from the LEFT camera and `r_paw` from the RIGHT, so a
left-camera ruler strictly rescales only the left paw. Few sessions have the right
camera cached, so the right-camera ruler is measured where possible and used to test
whether one camera's ruler stands in for the rig as a whole.
"""
import os
import re
import sys
import warnings
import pathlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATES_DIR = ROOT / 'data' / 'states_files'
CACHE = HERE / 'rig_covariates_sessions.csv'
LIK = 0.9


def _dist(d, a, b, lik=LIK):
    need = [f'{a}_x', f'{a}_y', f'{b}_x', f'{b}_y']
    if not all(c in d.columns for c in need):
        return np.nan, 0
    m = np.all([np.isfinite(d[c]) for c in need], axis=0)
    for p in (a, b):
        c = f'{p}_likelihood'
        if c in d.columns:
            m &= d[c].to_numpy() > lik
    if m.sum() < 500:
        return np.nan, int(m.sum())
    r = np.hypot(d[f'{a}_x'][m] - d[f'{b}_x'][m], d[f'{a}_y'][m] - d[f'{b}_y'][m])
    return float(np.median(r)), int(m.sum())


def one_session(args):
    eid, mouse, alf = args
    rec = dict(session=eid, mouse_name=mouse)
    for cam in ('left', 'right'):
        k = cam[0]
        for tracker, suffix in (('lp', 'lightningPose'), ('dlc', 'dlc')):
            f = f'{alf}/_ibl_{cam}Camera.{suffix}.pqt'
            if not os.path.exists(f):
                continue
            try:
                d = pd.read_parquet(f)
            except Exception:
                continue
            tube, n = _dist(d, 'tube_top', 'tube_bottom')
            rec[f'{k}_tube_{tracker}'] = tube
            rec[f'{k}_tube_n_{tracker}'] = n
            rec[f'{k}_nose_tube_{tracker}'], _ = _dist(d, 'nose_tip', 'tube_top')
            if 'tube_top_x' in d.columns:
                # a static object that jitters is a tracking failure, not a ruler
                rec[f'{k}_tube_jit_{tracker}'] = float(np.nanstd(d['tube_top_x']))
        t = f'{alf}/_ibl_{cam}Camera.times.npy'
        if os.path.exists(t):
            try:
                tt = np.load(t)
                if len(tt) > 100:
                    rec[f'{k}_fr'] = float(1 / np.median(np.diff(tt)))
            except Exception:
                pass
    return rec


def build(n_workers=8):
    from one.api import ONE
    one = ONE(mode='local')
    pat = re.compile(r'^8_states_file_([0-9a-f\-]{36})_(.+)$')
    jobs = []
    for f in sorted(os.listdir(STATES_DIR)):
        m = pat.match(f)
        if not m:
            continue
        eid, mouse = m.groups()
        p = one.eid2path(eid)
        if p is None:
            continue
        jobs.append((eid, mouse, str(p) + '/alf'))
    print(f'{len(jobs)} sessions resolved to paths', flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        res = list(ex.map(one_session, jobs, chunksize=2))
    R = pd.DataFrame(res)

    # ONE RULER COLUMN, preferring lightningPose and falling back to DLC, with the
    # agreement between them reported rather than assumed.
    for k in ('l', 'r'):
        lp, dlc = R.get(f'{k}_tube_lp'), R.get(f'{k}_tube_dlc')
        if lp is None:
            lp = pd.Series(np.nan, index=R.index)
        if dlc is None:
            dlc = pd.Series(np.nan, index=R.index)
        R[f'{k}_tube'] = lp.where(lp.notna(), dlc)
        R[f'{k}_tube_src'] = np.where(lp.notna(), 'lp', np.where(dlc.notna(), 'dlc', 'none'))
    R.to_csv(CACHE, index=False)
    print(f'wrote {CACHE}: {R.shape}')
    return R


def report(R):
    from scipy import stats
    print('\ncoverage:')
    for c in ['l_tube', 'r_tube', 'l_fr', 'r_fr']:
        if c in R:
            print(f'  {c:10s} {R[c].notna().sum():4d}/{len(R)}')
    if 'l_tube_src' in R:
        print('  left ruler source:', R['l_tube_src'].value_counts().to_dict())

    both = R.dropna(subset=['l_tube_lp', 'l_tube_dlc']) if \
        {'l_tube_lp', 'l_tube_dlc'} <= set(R.columns) else pd.DataFrame()
    if len(both) > 5:
        r = stats.spearmanr(both['l_tube_lp'], both['l_tube_dlc'])[0]
        print(f'\nDLC vs lightningPose on the same sessions (n={len(both)}): '
              f'rho={r:+.3f}, median ratio '
              f'{(both["l_tube_lp"] / both["l_tube_dlc"]).median():.3f}')
        print('  -> the tube is static, so the two trackers should agree; this checks it.')

    b2 = R.dropna(subset=['l_tube', 'r_tube'])
    if len(b2) > 5:
        r = stats.spearmanr(b2['l_tube'], b2['r_tube'])[0]
        print(f'\nDoes ONE camera\'s ruler stand in for the rig? (n={len(b2)} two-camera '
              f'sessions)\n  rho(l_tube, r_tube) = {r:+.3f}; '
              f'median l/r = {(b2["l_tube"] / b2["r_tube"]).median():.3f}')
        print('  A high rho means the left-camera ruler carries the rig-level geometry')
        print('  and can be used for both paws; a low one means it cannot.')

    print('\nhow much does the ruler vary? (it is the thing the correction removes)')
    v = np.log(R['l_tube'].dropna())
    print(f'  log l_tube: sd={v.std():.3f} across {len(v)} sessions '
          f'(median {np.exp(v.median()):.1f} px)')
    if 'l_fr' in R:
        print('  left camera frame rate:',
              R['l_fr'].round(0).value_counts().head(5).to_dict())


if __name__ == '__main__':
    R = pd.read_csv(CACHE) if (CACHE.exists() and '--refresh' not in sys.argv) else build()
    report(R)
