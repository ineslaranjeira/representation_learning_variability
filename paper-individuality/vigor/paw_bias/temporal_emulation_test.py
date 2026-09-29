"""How much of the frame-rate attenuation is the design matrix's doing, and how much is upstream?

Take a genuinely 150 Hz camera and process its paw trace two ways:
  (A) the pipeline's actual route  : low-pass 30 Hz -> linear resample to 60 Hz
  (B) as if it had been a 60 fps camera: box-average over 1/60 s (a model of the longer exposure),
      decimate to 60 Hz, then the same route as (A)
The ratio B/A is the part of the 60-vs-150 Hz penalty that exposure and sampling can explain, i.e.
the most a "match the cameras in time" fix could recover. Whatever gap remains against the observed
0.68 ratio at 8 Hz must originate before the design matrix -- in lightningPose itself, whose
ensemble-Kalman post-processing would smooth over a longer real-time window at a lower frame rate.
"""
import sys
import numpy as np
import pandas as pd
from scipy import interpolate, signal
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '.')
from test_downsampling import morlet_amp

FREQS = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
ROOT = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
OUT_FS = 60.0


def to60(x, t, ref, src_fs, boxcar=False):
    m = np.isfinite(x) & np.isfinite(t)
    x, t = x[m], t[m]
    if boxcar:
        # emulate a 1/60 s exposure: box-average, then keep only ~60 Hz worth of samples
        k = max(int(round(src_fs / OUT_FS)), 1)
        x = np.convolve(x, np.ones(k) / k, mode='same')
        x, t = x[::k], t[::k]
        src_fs = src_fs / k
    if src_fs > 60:
        b, a = signal.butter(4, 30 / (src_fs / 2), btype='low')
        x = signal.filtfilt(b, a, x)
    g = interpolate.interp1d(t, x, bounds_error=False)(ref)
    idx = np.flatnonzero(np.isfinite(g))
    g = g[idx[0]:idx[-1] + 1]
    return g[np.isfinite(g)]


def run(alf):
    try:
        d = pd.read_parquet(f'{alf}/_ibl_rightCamera.lightningPose.pqt',
                            columns=['paw_r_x', 'paw_r_y'])
        t = np.load(f'{alf}/_ibl_rightCamera.times.npy')
        n = min(len(d), len(t))
        d, t = d.iloc[:n], t[:n]
        fs = 1 / np.median(np.diff(t))
        if fs < 120:
            return None
        ref = np.arange(t[np.isfinite(t)].min(), t[np.isfinite(t)].max(), 1 / OUT_FS)
        if len(ref) < 20000:
            return None
        out = {}
        for tag, box in [('A', False), ('B', True)]:
            amps = []
            for c in ['paw_r_x', 'paw_r_y']:
                g = to60(d[c].to_numpy(float), t, ref, fs, boxcar=box)
                amps.append(morlet_amp(np.diff(g) * OUT_FS, FREQS).mean(axis=1))
            out[tag] = np.hypot(*amps)
        rec = {'sess': alf.replace(ROOT, ''), 'fs': fs}
        for i, f in enumerate(FREQS):
            rec[f'ratio_{f}'] = out['B'][i] / out['A'][i]
        return rec
    except Exception as e:
        return dict(sess=alf, err=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    dirs = [ROOT + x for x in open(sys.argv[2]).read().split()][:24]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = [r for r in ex.map(run, dirs, chunksize=1) if r]
    df = pd.DataFrame(res)
    df.to_csv(sys.argv[1], index=False)
    if 'err' in df:
        print(f'{df.err.notna().sum()} failures'); df = df[df.err.isna()]
    print(f'== {len(df)} sessions with a genuine 150 Hz right camera '
          f'(median {df.fs.median():.0f} Hz) ==\n')
    obs = {0.5: 0.929, 1.0: 0.926, 2.0: 0.906, 4.0: 0.841, 8.0: 0.681}
    print('  amplitude ratio after emulating a 60 fps camera, vs the observed 60-vs-150 penalty:\n')
    print(f'  {"freq":>6s} {"emulated B/A":>13s} {"OBSERVED":>10s} {"gap explained":>15s}')
    for f in FREQS:
        v = df[f'ratio_{f}'].mean()
        expl = (1 - v) / (1 - obs[f]) * 100 if obs[f] < 1 else np.nan
        print(f'  {f:6.1f} {v:13.3f} {obs[f]:10.3f} {expl:14.0f}%')
    print('\n  -> whatever fraction is NOT explained here happens before the design matrix,')
    print('     i.e. inside lightningPose, and no change to the resampling code can recover it.')
