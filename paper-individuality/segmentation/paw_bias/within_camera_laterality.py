"""Estimate paw laterality WITHOUT ever comparing across the two cameras.

The pipeline's l_paw/r_paw contrast takes one paw from each camera, so any residual difference in
camera gain, focus or noise lands directly on the laterality index. But each side camera tracks
BOTH paws, which allows two independent single-camera estimates:

    LI_leftcam  = LI( leftcam paw_r  [= mouse's LEFT paw, NEAR],
                      leftcam paw_l  [= mouse's RIGHT paw, FAR ] )
    LI_rightcam = LI( rightcam paw_l [= mouse's LEFT paw, FAR ],
                      rightcam paw_r [= mouse's RIGHT paw, NEAR] )

(paw identity per qc_paw_identity.py). Neither involves a cross-camera comparison, so neither can
be produced by the resolution correction. Each is still biased by near-vs-far foreshortening -- but
in OPPOSITE directions, because the left paw is the near one in the left camera and the far one in
the right camera. Their mean therefore cancels the near/far bias to first order, giving the
cleanest laterality estimate this dataset can support.

Two diagnostics follow:
  * sign agreement between the two single-camera estimates and the cross-camera one. If the
    population bias is biological, all three agree; if it is a camera artefact, the cross-camera
    estimate stands apart.
  * the near/far bias itself, recovered as (LI_leftcam - LI_rightcam) / 2.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_downsampling import morlet_amp, li

FS = 60.0
FREQS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 32.0])
BAND = slice(0, 5)
RES = {'left': 2, 'right': 1}
ROOT = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'


def band_power(x, y, ref, t):
    """Resample to the common 60 Hz grid, differentiate, Morlet-transform, return
    (0.5-8 Hz amplitude, 32 Hz amplitude) with x and y combined via hypot."""
    out = []
    for v in (x, y):
        m = np.isfinite(v) & np.isfinite(t)
        g = interpolate.interp1d(t[m], v[m], bounds_error=False)(ref)
        ok = np.isfinite(g)
        # one contiguous run only, so differencing never crosses a gap
        idx = np.flatnonzero(ok)
        g = g[idx[0]:idx[-1] + 1]
        g = pd.Series(g).interpolate(limit=6).to_numpy()
        if not np.all(np.isfinite(g)):
            g = g[np.isfinite(g)]
        out.append(morlet_amp(np.diff(g) * FS, FREQS).mean(axis=1))
    a = np.hypot(out[0], out[1])
    return a[BAND].sum(), a[5]


def run(alf):
    try:
        cams = {}
        for cam in ['left', 'right']:
            d = pd.read_parquet(f'{alf}/_ibl_{cam}Camera.lightningPose.pqt',
                                columns=['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y'])
            t = np.load(f'{alf}/_ibl_{cam}Camera.times.npy')
            n = min(len(d), len(t))
            cams[cam] = (d.iloc[:n].to_numpy(float) / RES[cam], t[:n])
        onset = max(t.min() for _, t in cams.values())
        offset = min(t.max() for _, t in cams.values())
        ref = np.arange(onset, offset, 1 / FS)
        if len(ref) < 20000:
            return None

        P, N = {}, {}
        for cam, (arr, t) in cams.items():
            for j, paw in [(0, 'paw_l'), (2, 'paw_r')]:
                P[(cam, paw)], N[(cam, paw)] = band_power(arr[:, j], arr[:, j + 1], ref, t)

        parts = alf.replace(ROOT, '').split('/Subjects/')
        rec = dict(lab=parts[0], mouse=parts[1].split('/')[0],
                   sess='/'.join(parts[1].split('/')[:3]), n=len(ref))
        # mouse's LEFT paw  = leftcam paw_r (near) / rightcam paw_l (far)
        # mouse's RIGHT paw = leftcam paw_l (far)  / rightcam paw_r (near)
        rec['li_leftcam'] = li(P[('left', 'paw_r')], P[('left', 'paw_l')])
        rec['li_rightcam'] = li(P[('right', 'paw_l')], P[('right', 'paw_r')])
        rec['li_within_mean'] = 0.5 * (rec['li_leftcam'] + rec['li_rightcam'])
        rec['nearfar_bias'] = 0.5 * (rec['li_leftcam'] - rec['li_rightcam'])
        # the cross-camera index the pipeline actually uses (both near views)
        rec['li_crosscam'] = li(P[('left', 'paw_r')], P[('right', 'paw_r')])
        # and the same two physical paws measured by the SWAPPED cameras (both far views)
        rec['li_crosscam_far'] = li(P[('right', 'paw_l')], P[('left', 'paw_l')])
        rec['li_noise_crosscam'] = li(N[('left', 'paw_r')], N[('right', 'paw_r')])
        return rec
    except Exception as e:
        return dict(sess=alf, err=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    dirs = [ROOT + d for d in open(sys.argv[2]).read().split()]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = [r for r in ex.map(run, dirs, chunksize=1) if r]
    df = pd.DataFrame(res)
    df.to_csv(sys.argv[1], index=False)
    if 'err' in df:
        print(f'{df.err.notna().sum()} failures'); print(df[df.err.notna()].err.head(3).to_string())
        df = df[df.err.isna()]
    print(f'\n== {len(df)} sessions, {df.mouse.nunique()} mice, {df.lab.nunique()} labs ==')
    print('\nLaterality estimates (LI > 0 = LEFT paw moves more), 0.5-8 Hz band power:')
    for c, lab in [('li_crosscam', 'cross-camera, both NEAR views  (what the pipeline uses)'),
                   ('li_crosscam_far', 'cross-camera, both FAR views   (cameras swapped)'),
                   ('li_leftcam', 'within LEFT camera only'),
                   ('li_rightcam', 'within RIGHT camera only'),
                   ('li_within_mean', 'mean of the two within-camera estimates  <-- cleanest'),
                   ('nearfar_bias', '[near-vs-far foreshortening bias, for reference]')]:
        v = df[c].dropna()
        t = stats.ttest_1samp(v, 0)
        print(f'  {lab:56s} mean={v.mean():+.4f} sd={v.std():.3f} '
              f'p={t.pvalue:.2e}  ({(v > 0).sum()}/{len(v)} positive)')
    print('\nDo the estimates agree per session? (correlation across sessions)')
    for a, b in [('li_crosscam', 'li_within_mean'), ('li_crosscam', 'li_leftcam'),
                 ('li_crosscam', 'li_rightcam'), ('li_leftcam', 'li_rightcam'),
                 ('li_crosscam', 'li_crosscam_far'), ('li_within_mean', 'li_noise_crosscam')]:
        d = df[[a, b]].dropna()
        r, p = stats.pearsonr(d[a], d[b])
        print(f'  {a:16s} vs {b:20s} r={r:+.3f} p={p:.2e} (n={len(d)})')
