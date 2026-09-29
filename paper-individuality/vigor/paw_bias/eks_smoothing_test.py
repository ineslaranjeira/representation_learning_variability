"""Is the frame-rate-dependent attenuation caused by lightningPose's own temporal smoothing?

The pose files carry two versions of every keypoint: `<kp>_ens_median`, the median across the
ensemble of networks (no temporal smoothing), and `<kp>_x`, the final output of lightningPose's
ensemble Kalman smoother. The ratio of their velocity spectra IS the smoother's transfer function.

Hypothesis: the smoother's time constant is set per FRAME rather than per second, so at 60 fps it
smooths over a 2.5x longer real-time window than at 150 fps. That predicts the transfer function
should roll off at a lower frequency for a 60 Hz camera than for a 150 Hz one -- which would produce
exactly the frequency-growing, frame-rate-following bias measured in gamma_spectrum.py.

Measured at each camera's NATIVE rate, so nothing here depends on the design matrix's resampling.
"""
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '.')
from test_downsampling import morlet_amp

FREQS = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
ROOT = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'


def run(alf):
    try:
        rec = {'sess': alf.replace(ROOT, ''), 'lab': alf.replace(ROOT, '').split('/')[0]}
        for cam in ['left', 'right']:
            cols = [f'paw_r_{a}{s}' for a in 'xy' for s in ['', '_ens_median']]
            d = pd.read_parquet(f'{alf}/_ibl_{cam}Camera.lightningPose.pqt', columns=cols)
            t = np.load(f'{alf}/_ibl_{cam}Camera.times.npy')
            n = min(len(d), len(t))
            d, t = d.iloc[:n], t[:n]
            fs = 1 / np.median(np.diff(t))
            m = d.notna().all(axis=1).to_numpy()
            if m.sum() < 30000:
                return None
            k = cam[0]
            rec[f'fs_{k}'] = fs
            amp = {}
            for tag, suf in [('smooth', ''), ('raw', '_ens_median')]:
                a = [morlet_amp(np.diff(d[f'paw_r_{ax}{suf}'].to_numpy(float)[m]) * fs,
                                FREQS, dt=1 / fs).mean(axis=1) for ax in 'xy']
                amp[tag] = np.hypot(*a)
            for i, f in enumerate(FREQS):
                # smoother transfer function: <1 means the smoother removes movement
                rec[f'eks_{k}_{f}'] = amp['smooth'][i] / amp['raw'][i]
        return rec
    except Exception as e:
        return dict(sess=alf, err=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    dirs = [ROOT + x for x in open(sys.argv[2]).read().split()]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = [r for r in ex.map(run, dirs, chunksize=1) if r]
    df = pd.DataFrame(res)
    df.to_csv(sys.argv[1], index=False)
    if 'err' in df:
        print(f'{df.err.notna().sum()} failures'); df = df[df.err.isna()]
    std = df[(df.fs_l < 90) & (df.fs_r > 120)]           # left 60, right 150 (the usual wiring)
    rev = df[(df.fs_l > 120) & (df.fs_r < 90)]           # left 150, right 60 (reversed)
    print(f'== {len(df)} sessions:  {len(std)} standard (left 60 / right 150), '
          f'{len(rev)} reversed ==\n')
    print("lightningPose smoother transfer function, measured at each camera's native rate")
    print("(value < 1 = the smoother removes real movement at that frequency)\n")
    obs = {0.5: 0.929, 1.0: 0.926, 2.0: 0.906, 4.0: 0.841, 8.0: 0.681}
    print(f'  {"freq":>6s} {"60 Hz cam":>11s} {"150 Hz cam":>11s} {"ratio 60/150":>13s} '
          f'{"OBSERVED L/R":>13s} {"explained":>10s}')
    for f in FREQS:
        slow = std[f'eks_l_{f}'].mean()
        fast = std[f'eks_r_{f}'].mean()
        pred = slow / fast
        expl = (1 - pred) / (1 - obs[f]) * 100 if obs[f] < 1 else np.nan
        print(f'  {f:6.1f} {slow:11.3f} {fast:11.3f} {pred:13.3f} {obs[f]:13.3f} {expl:9.0f}%')
    if len(rev) >= 3:
        print(f'\n  in the {len(rev)} REVERSED rigs the roles swap, as they must if this is the cause:')
        for f in [0.5, 8.0]:
            print(f'    {f:4.1f} Hz: left(150 Hz)={rev[f"eks_l_{f}"].mean():.3f}  '
                  f'right(60 Hz)={rev[f"eks_r_{f}"].mean():.3f}  '
                  f'ratio left/right={rev[f"eks_l_{f}"].mean()/rev[f"eks_r_{f}"].mean():.3f}')
