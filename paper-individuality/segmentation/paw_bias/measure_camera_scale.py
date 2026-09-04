"""Measure the left/right camera spatial-scale ratio DIRECTLY, from landmarks seen by both cameras.

The pipeline assumes the ratio is exactly 2 (left camera 1280x1024, right 640x512) and divides
left-camera pixels by 2. The paw-movement analysis in within_camera_laterality.py implies the
factor that actually makes the two cameras agree is ~1.67, not 2. That inference rests on modelling
assumptions, so here the ratio is measured directly instead.

The lick tube is rig hardware: one rigid object of fixed physical size, seen by both cameras at the
same moment. Its apparent length in pixels therefore gives each camera's pixels-per-millimetre, and
the ratio of the two is the resolution ratio, with no reference to the mouse or its movement.
Two further references cross-check it:
  * nose_tip -> tube_top   : a mouse-to-rig distance, identical physical value from either side
  * pupil diameter         : bilaterally symmetric, viewed nearly face-on by its own side camera

If the measured ratio is ~2, the pipeline's assumption is right and the residual bias must come from
somewhere else. If it is ~1.67, the assumption is simply wrong and correcting it fixes the bias.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

ROOT = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
LIK = 0.9


def dist(d, a, b, lik=LIK):
    """Median apparent distance between two landmarks, over confidently-tracked frames."""
    need = [f'{a}_x', f'{a}_y', f'{b}_x', f'{b}_y']
    m = np.all([np.isfinite(d[c]) for c in need], axis=0)
    for p in (a, b):
        c = f'{p}_likelihood'
        if c in d:
            m &= d[c].to_numpy() > lik
    if m.sum() < 500:
        return np.nan, 0
    r = np.hypot(d[f'{a}_x'][m] - d[f'{b}_x'][m], d[f'{a}_y'][m] - d[f'{b}_y'][m])
    return float(np.median(r)), int(m.sum())


def run(alf):
    try:
        rec = {}
        for cam in ['left', 'right']:
            d = pd.read_parquet(f'{alf}/_ibl_{cam}Camera.lightningPose.pqt')
            k = cam[0]
            rec[f'{k}_tube'], rec[f'{k}_tube_n'] = dist(d, 'tube_top', 'tube_bottom')
            rec[f'{k}_nose_tube'], _ = dist(d, 'nose_tip', 'tube_top')
            v, _ = dist(d, 'pupil_top_r', 'pupil_bottom_r')
            h, _ = dist(d, 'pupil_left_r', 'pupil_right_r')
            rec[f'{k}_pupil'] = np.nanmean([v, h])
            # tube position spread: a static object should barely move
            rec[f'{k}_tube_jit'] = float(np.nanstd(d['tube_top_x']))
        for tag in ['tube', 'nose_tube', 'pupil']:
            rec[f'ratio_{tag}'] = rec[f'l_{tag}'] / rec[f'r_{tag}'] if rec[f'r_{tag}'] else np.nan
        parts = alf.replace(ROOT, '').split('/Subjects/')
        rec.update(lab=parts[0], mouse=parts[1].split('/')[0],
                   sess='/'.join(parts[1].split('/')[:3]))
        return rec
    except Exception as e:
        return dict(sess=alf, err=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    dirs = [ROOT + d for d in open(sys.argv[2]).read().split()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [r for r in ex.map(run, dirs, chunksize=1) if r]
    df = pd.DataFrame(res)
    df.to_csv(sys.argv[1], index=False)
    if 'err' in df:
        print(f'{df.err.notna().sum()} failures'); df = df[df.err.isna()]

    print(f'== {len(df)} sessions, {df.mouse.nunique()} mice, {df.lab.nunique()} labs ==\n')
    print('apparent size in each camera (raw pixels, no scaling applied):')
    for tag, lab in [('tube', 'lick tube length      (rig hardware, rigid)'),
                     ('nose_tube', 'nose tip -> tube top  (mouse-to-rig)'),
                     ('pupil', 'pupil diameter        (bilaterally symmetric)')]:
        print(f'  {lab:44s} left {df[f"l_{tag}"].median():7.2f} px   '
              f'right {df[f"r_{tag}"].median():7.2f} px')

    print('\nMEASURED left/right resolution ratio (pipeline assumes 2.000):')
    for tag, lab in [('tube', 'lick tube length'), ('nose_tube', 'nose tip -> tube top'),
                     ('pupil', 'pupil diameter')]:
        v = df[f'ratio_{tag}'].dropna()
        v = v[(v > 0.5) & (v < 5)]
        if len(v) < 5:
            print(f'  {lab:24s} too few usable sessions ({len(v)})'); continue
        t2 = stats.ttest_1samp(v, 2.0)
        t167 = stats.ttest_1samp(v, 1.675)
        print(f'  {lab:24s} median={v.median():.3f}  mean={v.mean():.3f} '
              f'sd={v.std():.3f}  n={len(v)}')
        print(f'  {"":24s}   vs 2.000: t={t2.statistic:+7.2f} p={t2.pvalue:.1e}   '
              f'vs 1.675: t={t167.statistic:+7.2f} p={t167.pvalue:.2f}')

    print('\nis the ratio rig-dependent?')
    for tag in ['tube']:
        d = df[df[f'ratio_{tag}'].between(0.5, 5)]
        g = d.groupby('lab')[f'ratio_{tag}'].agg(n='count', median='median').round(3)
        print(g[g.n >= 3].to_string())
        gr = [x[f'ratio_{tag}'].values for _, x in d.groupby('lab') if len(x) >= 3]
        if len(gr) >= 3:
            f = stats.f_oneway(*gr)
            print(f'  ANOVA across labs: F={f.statistic:.2f} p={f.pvalue:.2e}')
