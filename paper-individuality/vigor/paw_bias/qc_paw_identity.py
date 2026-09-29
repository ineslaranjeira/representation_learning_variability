"""QC: establish which lightningPose paw label corresponds to which physical forepaw.

Both side cameras track `paw_l` and `paw_r`. The design-matrix pipeline takes `paw_r` from each
camera and calls them `l_paw` / `r_paw`. Two things have to be true for that to be a valid
left/right contrast:
  1. left-camera paw_r and right-camera paw_r must be DIFFERENT physical paws
     -> tested by which cross-camera pairing of speed traces correlates best.
  2. `paw_r` must be the NEAR paw in each camera (the well-resolved one)
     -> tested by which camera sees a given physical paw with the larger excursion.
Run over every session with both cameras cached, to check the convention is rig-independent.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats, interpolate
from concurrent.futures import ProcessPoolExecutor

RES = {'left': 2, 'right': 1}   # left cam is 1280x1024, right cam 640x512


def _traces(d, t, res):
    """60 Hz-gridded speed magnitude + scaled position sd, for both paw labels."""
    out = {}
    fr = 1 / np.median(np.diff(t))
    for p in ['paw_l', 'paw_r']:
        x = d[f'{p}_x'].to_numpy(float) / res
        y = d[f'{p}_y'].to_numpy(float) / res
        sp = np.r_[np.hypot(np.diff(x), np.diff(y)) * fr, np.nan]
        out[p] = (t, sp, np.nanstd(x), np.nanstd(y))
    return out


def run(alf):
    try:
        cams = {}
        for cam in ['left', 'right']:
            d = pd.read_parquet(f'{alf}/_ibl_{cam}Camera.lightningPose.pqt',
                                columns=['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y'])
            t = np.load(f'{alf}/_ibl_{cam}Camera.times.npy')
            n = min(len(d), len(t))
            cams[cam] = _traces(d.iloc[:n], t[:n], RES[cam])

        onset = max(cams[c][p][0].min() for c in cams for p in ['paw_r'])
        offset = min(cams[c][p][0].max() for c in cams for p in ['paw_r'])
        ref = np.arange(onset, offset, 1 / 60)
        if len(ref) < 6000:
            return None

        g = {}
        for cam in cams:
            for p in cams[cam]:
                t, sp, sx, sy = cams[cam][p]
                m = np.isfinite(sp) & np.isfinite(t)
                g[f'{cam[0].upper()}:{p}'] = interpolate.interp1d(
                    t[m], sp[m], bounds_error=False)(ref)

        def r(a, b):
            m = np.isfinite(g[a]) & np.isfinite(g[b])
            return float(stats.pearsonr(g[a][m], g[b][m])[0]) if m.sum() > 1000 else np.nan

        # pairing A ("crossed"): leftcam paw_r <-> rightcam paw_l  == same physical paw
        # pairing B ("same-name"): leftcam paw_r <-> rightcam paw_r == same physical paw
        crossed = 0.5 * (r('L:paw_r', 'R:paw_l') + r('L:paw_l', 'R:paw_r'))
        samename = 0.5 * (r('L:paw_r', 'R:paw_r') + r('L:paw_l', 'R:paw_l'))

        # near/far: for the crossed pairing, compare scaled excursion between the two views
        sd = {f'{c[0].upper()}:{p}': np.hypot(cams[c][p][2], cams[c][p][3]) for c in cams for p in cams[c]}
        # physical paw 1 = (L:paw_r, R:paw_l); physical paw 2 = (L:paw_l, R:paw_r)
        paw1_near = 'left' if sd['L:paw_r'] > sd['R:paw_l'] else 'right'
        paw2_near = 'left' if sd['L:paw_l'] > sd['R:paw_r'] else 'right'

        parts = alf.split('/Subjects/')
        return dict(lab=parts[0], sess='/'.join(parts[1].split('/')[:3]),
                    mouse=parts[1].split('/')[0],
                    r_cross=crossed, r_same=samename,
                    r_Lr_Rl=r('L:paw_r', 'R:paw_l'), r_Ll_Rr=r('L:paw_l', 'R:paw_r'),
                    r_Lr_Rr=r('L:paw_r', 'R:paw_r'), r_Ll_Rl=r('L:paw_l', 'R:paw_l'),
                    r_within_L=r('L:paw_l', 'L:paw_r'), r_within_R=r('R:paw_l', 'R:paw_r'),
                    sd_Lr=sd['L:paw_r'], sd_Rl=sd['R:paw_l'],
                    sd_Ll=sd['L:paw_l'], sd_Rr=sd['R:paw_r'],
                    paw1_near=paw1_near, paw2_near=paw2_near, n=len(ref))
    except Exception as e:
        return dict(err=f'{type(e).__name__}: {e}', sess=alf)


if __name__ == '__main__':
    dirs = open(sys.argv[1]).read().split()
    root = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, [root + d for d in dirs]))
    res = [r for r in res if r]
    pd.DataFrame(res).to_csv(sys.argv[2], index=False)
    print('done', len(res))
