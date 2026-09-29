"""
WHEEL SCALE: THE ONE AMPLITUDE THAT HAS NO CAMERA IN IT
========================================================
Paw amplitude is measured in camera pixels, so it carries a per-session spatial gain
(measured here: sd(log lick-tube px) = 0.159) and a frame-rate-dependent temporal gain
from the pose smoother. Wheel velocity comes from a ROTARY ENCODER. There is no lens,
no working distance and no smoother -- the units are physical and identical on every
rig. So if raw amplitude is worth adding to the individuality features anywhere, the
wheel is where the rig objection is weakest.

It is also the only channel that is not already in the syllables: the 360 features are
paw + whisk + lick, and nothing about the wheel enters them.

Builds the same SCALE block for the wheel that `zscore_cost.py` built for the paw --
log mean and log sd of each wavelet band, pre-standardisation -- from
`data/wheel_wavelets/`, using the 0.5-8 Hz bands to match the paw convention.
"""
import os, re, sys, warnings, pathlib
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WHEEL_DIR = ROOT / 'data' / 'wheel_wavelets'
CACHE = HERE / 'wheel_moments_sessions.parquet'
FREQS = ['0.5', '1.0', '2.0', '4.0', '8.0']
WVAR = [f'avg_wheel_vel{f}' for f in FREQS]


def build():
    pat = re.compile(r'^wheel_vel_wavelets_([0-9a-f\-]{36})_(.+)$')
    rows = []
    for i, f in enumerate(sorted(os.listdir(WHEEL_DIR))):
        m = pat.match(f)
        if not m or not os.path.isfile(WHEEL_DIR / f):
            continue
        eid, mouse = m.groups()
        d = pd.read_parquet(WHEEL_DIR / f, columns=WVAR).dropna()
        a = d.to_numpy(float)
        a = a[np.isfinite(a).all(1)]
        if len(a) < 1000:
            continue
        r = dict(session=eid, mouse_name=mouse)
        mu, sd = a.mean(0), a.std(0)
        for j, c in enumerate(WVAR):
            r[f'wlogmean_{c}'] = float(np.log(mu[j]))
            r[f'wlogsd_{c}'] = float(np.log(sd[j]))
            z = (a[:, j] - mu[j]) / sd[j]
            r[f'wskew_{c}'] = float(stats.skew(z))
            r[f'wkurt_{c}'] = float(stats.kurtosis(z))
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}', flush=True)
    W = pd.DataFrame(rows)
    W.to_parquet(CACHE)
    print(f'wrote {CACHE}: {W.shape}')
    return W


if __name__ == '__main__':
    build() if ('--build' in sys.argv or not CACHE.exists()) else None
