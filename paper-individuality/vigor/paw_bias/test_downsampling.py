"""Does degrading the left camera to right-camera resolution fix the left/right bias artefact?

`get_speed` halves left-camera pixels (1280x1024 -> right camera's 640x512 scale), which equates
spatial scale but also halves the left camera's tracking noise. Raw LI[0.5-8 Hz power] therefore
correlates r=+0.80 with LI[32 Hz power], a band where a forepaw cannot physically oscillate.

Real downsampling means re-encoding the video at 640x512 and re-running lightningPose, which cannot
be done post-hoc. Two surrogates bracket what it would do, plus one alternative that destroys no
signal:

  QUANT  snap left-paw coordinates onto the right camera's pixel grid. This is the FLOOR of the
         downsampling effect: it removes sub-pixel precision only (uniform noise, sd = 1/sqrt(12)
         right-cam px) and does not reproduce the larger pose-estimation error a lower-resolution
         frame actually causes.
  MATCH  add Gaussian positional noise to the left paw, CALIBRATED per session so its 32 Hz
         wavelet power equals the right paw's. This emulates the full effect of degrading the
         camera, and the calibration is verified rather than assumed.
  WHITEN divide each paw's 0.5-8 Hz band amplitudes by its own 32 Hz amplitude. Cancels the
         per-camera gain and the noise floor without adding any noise.

Success criteria, fixed before looking at the results:
  C1  LI[32 Hz] -> 0                      (noise floors actually matched)
  C2  r(LI[band power], LI[32 Hz]) -> 0   (amplitude bias no longer tracks noise)
  C3  the per-frequency LI trend flattens (no monotonic right-bias growing with frequency)
  C4  between-mouse ICC survives          (individuality was not purely the artefact)
  C5  LI x wheel-direction bias survives  (the task link was not purely the artefact)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import signal, stats

from numpy.fft import fft, ifft, fftshift

FS = 60.0
OMEGA0 = 5
FREQS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 32.0])
BAND = slice(0, 5)          # 0.5-8 Hz
NOISE = 5                   # the 32 Hz channel
BASE = ('/home/ines/repositories/representation_learning_variability/'
        'paper-individuality/data/design_matrices/')
RNG_SEED = 7


def li(a, b):
    return (a - b) / (a + b) if (a + b) > 0 else np.nan


def white_sd(x, m):
    """Positional noise sd, estimated from the >20 Hz residual. Real forepaw movement has
    negligible power there, and for noise flat over 0-30 Hz that band holds 1/3 of the variance."""
    b, a = signal.butter(4, 20 / (FS / 2), btype='high')
    return float(np.std(signal.filtfilt(b, a, x[m])) * np.sqrt(3))


def morlet_amp(x, f, omega0=OMEGA0, dt=1 / FS):
    """Vectorised equivalent of segmentation_functions.fast_wavelet_morlet_convolution_parallel's
    `amp` output (verified to 5e-15 relative). The original calls joblib Parallel(n_jobs=-1) per
    invocation, which oversubscribes badly when the caller is itself a process pool; this version
    reuses one FFT across frequencies and runs ~7x faster single-threaded."""
    x = np.asarray(x, float)
    N = len(x)
    odd = N % 2 == 1
    if odd:
        x = np.append(x, 0)
        N += 1
    M = N
    x = np.concatenate((np.zeros(N // 2), x, np.zeros(N // 2)))
    N = len(x)
    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * np.asarray(f, float))
    omega = 2 * np.pi * np.arange(-N // 2, N // 2) / (N * dt)
    xh = fftshift(fft(x))
    idx = np.arange(M // 2, M // 2 + M - 1) if odd else np.arange(M // 2, M // 2 + M)
    W = np.pi**-0.25 * np.exp(-0.5 * (-omega[None, :] * scales[:, None] - omega0)**2)
    q = ifft(W * xh[None, :], axis=1)[:, idx] * np.sqrt(scales)[:, None]
    return (np.abs(q) * np.pi**-0.25 *
            np.exp(0.25 * (omega0 - np.sqrt(omega0**2 + 2))**2) / np.sqrt(2 * scales)[:, None])


def wavelet_amp(pos, m, freqs=FREQS):
    """Velocity wavelet amplitude per frequency, following the notebook: differentiate the
    resolution-corrected position, then Morlet-transform."""
    return morlet_amp(np.diff(pos[m]) * FS, freqs)


def lp_speed_median(x, y, m):
    """Median 2D speed after a common 8 Hz low-pass -- the frame-difference-style measure."""
    b, a = signal.butter(4, 8 / (FS / 2), btype='low')
    xs, ys = signal.filtfilt(b, a, x[m]), signal.filtfilt(b, a, y[m])
    return float(np.median(np.hypot(np.diff(xs), np.diff(ys)) * FS))


def session_metrics(lx, ly, rx, ry, m, freqs=FREQS):
    """Mean band amplitude (x and y combined via hypot) per paw, and the derived indices."""
    out = {}
    out['spmed_L'] = lp_speed_median(lx, ly, m)
    out['spmed_R'] = lp_speed_median(rx, ry, m)
    out['li_spmed'] = li(out['spmed_L'], out['spmed_R'])
    aL = np.hypot(wavelet_amp(lx, m, freqs), wavelet_amp(ly, m, freqs)).mean(axis=1)
    aR = np.hypot(wavelet_amp(rx, m, freqs), wavelet_amp(ry, m, freqs)).mean(axis=1)
    out['bandpow_L'], out['bandpow_R'] = aL[BAND].sum(), aR[BAND].sum()
    out['noisepow_L'], out['noisepow_R'] = aL[NOISE], aR[NOISE]
    out['li_bandpow'] = li(out['bandpow_L'], out['bandpow_R'])
    out['li_noisepow'] = li(out['noisepow_L'], out['noisepow_R'])
    out['li_snr'] = li(out['bandpow_L'] / out['noisepow_L'], out['bandpow_R'] / out['noisepow_R'])
    for i, f in enumerate(FREQS[BAND]):
        out[f'li_pow_{f}'] = li(aL[i], aR[i])
    return out


def run(item):
    eid, mouse = item
    try:
        rng = np.random.default_rng(abs(hash(eid)) % 2**31 + RNG_SEED)
        dm = pd.read_parquet(BASE + f'design_matrix_{eid}_{mouse}',
                             columns=['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y',
                                      'avg_wheel_vel'])
        raw = {k: dm[k].to_numpy(float) for k in
               ['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y']}
        m = np.all([np.isfinite(v) for v in raw.values()], axis=0)
        if m.sum() < 20000:
            return None

        # right camera is already in its own pixel units; left is halved to match
        rx, ry = raw['r_paw_x'], raw['r_paw_y']
        lx, ly = raw['l_paw_x'] / 2, raw['l_paw_y'] / 2

        rec = dict(eid=eid, mouse=mouse, n=int(m.sum()))
        rec['sd_L'] = 0.5 * (white_sd(lx, m) + white_sd(ly, m))
        rec['sd_R'] = 0.5 * (white_sd(rx, m) + white_sd(ry, m))
        wh = dm['avg_wheel_vel'].to_numpy(float)
        rec['wheel_bias'] = float(np.nanmean(wh > 0) - np.nanmean(wh < 0))
        tf = BASE + f'session_trials_{eid}_{mouse}'
        if os.path.exists(tf):
            tr = pd.read_parquet(tf, columns=['choice'])
            ch = tr['choice'].to_numpy(float)
            ch = ch[np.isfinite(ch)]
            if len(ch) > 50:
                rec['frac_right_choice'] = float(np.mean(ch == -1))

        # ---------- variant RAW ----------
        base = session_metrics(lx, ly, rx, ry, m)
        for k, v in base.items():
            rec[f'raw_{k}'] = v

        # ---------- variant QUANT: snap onto the right camera's pixel grid ----------
        qx, qy = np.round(lx), np.round(ly)
        for k, v in session_metrics(qx, qy, rx, ry, m).items():
            rec[f'quant_{k}'] = v

        # ---------- variant MATCH: calibrate added noise so 32 Hz power matches ----------
        # P32(sigma)^2 ~ P32(0)^2 + (c*sigma)^2 for independent added noise; measure c with a probe
        p0 = base['noisepow_L']
        target = base['noisepow_R']
        sp = max(rec['sd_L'], 1e-3)
        px = lx + rng.normal(0, sp, lx.shape)
        py = ly + rng.normal(0, sp, ly.shape)
        pp = np.hypot(wavelet_amp(px, m, np.array([32.0])),
                      wavelet_amp(py, m, np.array([32.0]))).mean()
        c2 = max((pp**2 - p0**2) / sp**2, 1e-9)
        need = (target**2 - p0**2) / c2
        sigma = float(np.sqrt(need)) if need > 0 else 0.0
        rec['sigma_added'] = sigma
        mx = lx + rng.normal(0, sigma, lx.shape) if sigma > 0 else lx
        my = ly + rng.normal(0, sigma, ly.shape) if sigma > 0 else ly
        for k, v in session_metrics(mx, my, rx, ry, m).items():
            rec[f'match_{k}'] = v

        # ---------- variant WHITEN: derived from RAW, nothing added ----------
        rec['whiten_li_bandpow'] = li(base['bandpow_L'] / base['noisepow_L'],
                                      base['bandpow_R'] / base['noisepow_R'])
        rec['whiten_li_noisepow'] = 0.0
        return rec
    except Exception as e:
        return dict(eid=eid, mouse=mouse, err=f'{type(e).__name__}: {e}')


def pick_sessions(min_sess=3, max_per_mouse=4, n_mice=None):
    wv = ('/home/ines/repositories/representation_learning_variability/'
          'paper-individuality/data/paw_wavelets/')
    from extract_paw_bias import SESSIONS_TO_EXCLUDE
    items = sorted({(f[17:53], f[54:]) for f in os.listdir(wv)
                    if f.startswith('paw_vel_wavelets_') and f[17:53] not in SESSIONS_TO_EXCLUDE
                    and os.path.exists(BASE + f'design_matrix_{f[17:53]}_{f[54:]}')})
    by = {}
    for eid, mo in items:
        by.setdefault(mo, []).append(eid)
    keep = {mo: e[:max_per_mouse] for mo, e in by.items() if len(e) >= min_sess}
    if n_mice:
        keep = dict(sorted(keep.items())[:n_mice])
    return [(e, mo) for mo, es in keep.items() for e in es]


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    items = pick_sessions()
    if len(sys.argv) > 2:
        items = items[:int(sys.argv[2])]
    print(f'{len(items)} sessions, {len(set(m for _, m in items))} mice', flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [r for r in ex.map(run, items, chunksize=1) if r]
    df = pd.DataFrame(res)
    df.to_csv(sys.argv[1], index=False)
    print('wrote', sys.argv[1], df.shape)
    if 'err' in df:
        print(df[df.err.notna()][['eid', 'err']].head().to_string())
