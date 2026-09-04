"""Is the cross-camera bias a SPATIAL scale error or a TEMPORAL one?

measure_camera_scale.py shows the left/right spatial ratio really is 2.0, so the /2 correction is
right and gamma is not a scale error. The two candidate mechanisms left are both temporal, and both
attenuate the LEFT camera:

  * motion blur -- the left camera runs at 60 fps (exposure up to ~16.7 ms), the right at 150 fps
    (<=6.7 ms). A longer exposure smears a moving paw, so the tracked point is drawn toward the
    middle of its excursion.
  * resampling -- the left camera is 60 Hz native and is linearly interpolated onto a different
    60 Hz grid, which is a triangular smoothing kernel one full sample wide. The right camera has
    2.5 source samples per output bin, so its interpolation error is far smaller.

These make a prediction a spatial gain error cannot: attenuation must GROW WITH FREQUENCY. A pure
scale error is flat across frequency. So compute gamma per frequency band rather than band-summed.
"""
import sys
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '.')
from test_downsampling import morlet_amp

FS = 60.0
FREQS = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
RES = {'left': 2, 'right': 1}
ROOT = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'


def amp(x, y, ref, t):
    out = []
    for v in (x, y):
        m = np.isfinite(v) & np.isfinite(t)
        g = interpolate.interp1d(t[m], v[m], bounds_error=False)(ref)
        idx = np.flatnonzero(np.isfinite(g))
        g = pd.Series(g[idx[0]:idx[-1] + 1]).interpolate(limit=6).to_numpy()
        g = g[np.isfinite(g)]
        out.append(morlet_amp(np.diff(g) * FS, FREQS).mean(axis=1))
    return np.hypot(out[0], out[1])


def run(alf):
    try:
        cams = {}
        for cam in ['left', 'right']:
            d = pd.read_parquet(f'{alf}/_ibl_{cam}Camera.lightningPose.pqt',
                                columns=['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y'])
            t = np.load(f'{alf}/_ibl_{cam}Camera.times.npy')
            n = min(len(d), len(t))
            cams[cam] = (d.iloc[:n].to_numpy(float) / RES[cam], t[:n],
                         1 / np.median(np.diff(t[:n])))
        onset = max(t.min() for _, t, _ in cams.values())
        offset = min(t.max() for _, t, _ in cams.values())
        ref = np.arange(onset, offset, 1 / FS)
        if len(ref) < 20000:
            return None
        P = {}
        for cam, (arr, t, _) in cams.items():
            for j, paw in [(0, 'paw_l'), (2, 'paw_r')]:
                P[(cam, paw)] = amp(arr[:, j], arr[:, j + 1], ref, t)
        # LI per frequency; positive = first argument larger
        def LI(a, b):
            return (a - b) / (a + b)
        li_cross = LI(P[('left', 'paw_r')], P[('right', 'paw_r')])       # tau + gamma_signed
        li_cross_far = LI(P[('right', 'paw_l')], P[('left', 'paw_l')])   # tau - gamma_signed
        rec = {'sess': alf.replace(ROOT, ''), 'lab': alf.replace(ROOT, '').split('/')[0],
               'fr_left': cams['left'][2], 'fr_right': cams['right'][2]}
        for i, f in enumerate(FREQS):
            # gamma as reported elsewhere: positive = LEFT camera reads LOW
            rec[f'gamma_{f}'] = (li_cross_far[i] - li_cross[i]) / 2
            rec[f'tau_{f}'] = (li_cross_far[i] + li_cross[i]) / 2
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
        print(f'{df.err.notna().sum()} failures'); df = df[df.err.isna()]
    print(f'== {len(df)} sessions ==\n')
    print('native frame rates: left %s Hz, right %s Hz'
          % (sorted(df.fr_left.round().unique())[:5], sorted(df.fr_right.round().unique())[:5]))
    print('\ngamma per frequency  (positive = LEFT camera reads LOW).')
    print('A spatial scale error would be FLAT; temporal attenuation GROWS with frequency.\n')
    print(f'  {"freq":>6s} {"gamma":>9s} {"sem":>7s} {"left/right amp ratio":>21s} {"tau (true lat.)":>16s}')
    for f in FREQS:
        g = df[f'gamma_{f}'].dropna()
        t = df[f'tau_{f}'].dropna()
        print(f'  {f:6.1f} {g.mean():+9.4f} {g.sem():7.4f} {np.exp(-2*g.mean()):21.3f} '
              f'{t.mean():+10.4f} (p={stats.ttest_1samp(t,0).pvalue:.2f})')
    lo, hi = df['gamma_0.5'].dropna(), df['gamma_8.0'].dropna()
    d = df[['gamma_0.5', 'gamma_8.0']].dropna()
    tt = stats.ttest_rel(d['gamma_8.0'], d['gamma_0.5'])
    print(f'\n  gamma at 8 Hz vs 0.5 Hz (paired): {hi.mean():+.4f} vs {lo.mean():+.4f}, '
          f't={tt.statistic:+.2f} p={tt.pvalue:.2e}')
    print(f'  => {"GROWS with frequency: temporal" if tt.pvalue < 0.01 and tt.statistic > 0 else "flat: spatial"}')
