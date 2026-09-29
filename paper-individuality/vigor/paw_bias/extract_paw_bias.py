"""Per-session left/right forepaw movement-bias metrics.

Paw identity (established empirically in qc_paw_identity.py over 68 two-camera sessions, 10 labs):
    l_paw_* = leftCamera  'paw_r' = the mouse's LEFT  forepaw, near view
    r_paw_* = rightCamera 'paw_r' = the mouse's RIGHT forepaw, near view
The left camera is 1280x1024 and the right 640x512, so left-camera pixels are halved to put both
paws in a common spatial unit (this is what segmentation_functions.get_speed already does).

THE CONFOUND THAT DRIVES EVERY DESIGN CHOICE HERE
Halving the left camera also halves its tracking noise, while the right camera's noise is kept at
full size. Raw frame-to-frame speed therefore makes the right paw look ~2x faster in essentially
every session -- that is pixel noise, not behaviour. Countermeasures:
  * positions are low-pass filtered at a common cutoff before differentiating,
  * the primary amplitude measure is 0.5-8 Hz wavelet band power, not frame-difference speed,
  * a per-paw jitter proxy is stored so it can be regressed out downstream,
  * scale-free metrics (bout structure at matched duty cycle, spectral shape, lead-lag) are
    reported alongside, since a multiplicative per-camera error cannot touch them.
Consequence: the POPULATION MEAN laterality index is not interpretable as biology. Only
individual differences around it are, and only after the jitter check.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal
from concurrent.futures import ProcessPoolExecutor

FS = 60.0                      # design-matrix bin rate
LP_CUTOFF = 8.0                # common low-pass before differentiating (Hz)
BAND = ['0.5', '1.0', '2.0', '4.0', '8.0']   # wavelet bands used as the amplitude measure
NOISE_BAND = ['32.0']          # negligible real paw movement here -> tracking-noise proxy
DUTY = 0.20                    # matched duty cycle for scale-free bout metrics
MAX_LAG = 30                   # +-0.5 s for the lead-lag cross-correlation

BASE = '/home/ines/repositories/representation_learning_variability/paper-individuality/data/'
DM_PATH = BASE + 'design_matrices/'
WV_PATH = BASE + 'paw_wavelets/'

# Sessions dropped upstream in 3.3_wavelet_clusters.ipynb (bad fits / bad video), kept in sync.
SESSIONS_TO_EXCLUDE = [
    'a8a8af78-16de-4841-ab07-fde4b5281a03', '8c33abef-3d3e-4d42-9f27-445e9def08f9',
    'ebe2efe3-e8a1-451a-8947-76ef42427cc9', '91bac580-76ed-41ab-ac07-89051f8d7f6e',
    '8a1cf4ef-06e3-4c72-9bc7-e1baa189841b', '64977c74-9c04-437a-9ea1-50386c4996db',
    '90e524a2-aa63-47ce-b5b8-1b1941a1223a', '30af8629-7b96-45b7-8778-374720ddbc5e',
    'f3eeb2d4-87ce-49ae-8a74-21665f6f1536', 'fcd49e34-f07b-441c-b2ac-cb8c462ec5ac']


def li(left, right):
    """Laterality index in [-1, 1]; positive = left paw larger. NaN if the sum is degenerate."""
    tot = left + right
    return (left - right) / tot if np.isfinite(tot) and tot > 0 else np.nan


def _speed(x, y, res, lp=True):
    """Resolution-corrected 2D speed. Low-pass is applied on the valid samples only, so that
    NaN gaps (camera dropouts) are never interpolated across."""
    x = x / res
    y = y / res
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 600:
        return np.full(len(x) - 1, np.nan), np.nan
    xs, ys = x.copy(), y.copy()
    if lp:
        b, a = signal.butter(4, LP_CUTOFF / (FS / 2), btype='low')
        xs[m] = signal.filtfilt(b, a, x[m])
        ys[m] = signal.filtfilt(b, a, y[m])
    jitter = float(np.nanstd(np.hypot(x[m] - xs[m], y[m] - ys[m])))   # high-freq residual
    sp = np.hypot(np.diff(xs), np.diff(ys)) * FS
    sp[~(m[:-1] & m[1:])] = np.nan
    return sp, jitter


def _bout_stats(sp, duty=DUTY):
    """Bout count per minute and mean bout duration with the threshold set so that this paw
    spends exactly `duty` of its valid time above it. Matching the duty cycle removes any
    multiplicative scale/noise difference, leaving only how movement is parcelled in time."""
    v = sp[np.isfinite(sp)]
    if len(v) < 600:
        return np.nan, np.nan
    thr = np.quantile(v, 1 - duty)
    above = np.isfinite(sp) & (sp > thr)
    d = np.diff(above.astype(np.int8))
    n_bouts = int((d == 1).sum())
    minutes = len(v) / FS / 60
    if n_bouts == 0 or minutes == 0:
        return np.nan, np.nan
    return n_bouts / minutes, above.sum() / FS / n_bouts


def _lead_lag(a, b, max_lag=MAX_LAG):
    """Peak of the cross-correlation of the two paw speed traces.
    Positive lag = the left paw leads. Also returns the peak height (bilateral coupling)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 6000:
        return np.nan, np.nan
    x = stats.zscore(a[m])
    y = stats.zscore(b[m])
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.array([np.dot(x[max_lag:n - max_lag], y[max_lag + L:n - max_lag + L]) /
                  (n - 2 * max_lag) for L in lags])
    k = int(np.argmax(r))
    return float(lags[k] / FS), float(r[k])


def run(item):
    eid, mouse = item
    try:
        out = dict(eid=eid, mouse=mouse)
        dm = pd.read_parquet(DM_PATH + f'design_matrix_{eid}_{mouse}',
                             columns=['Bin', 'l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y',
                                      'avg_wheel_vel'])
        out['dur_min'] = float((dm['Bin'].iloc[-1] - dm['Bin'].iloc[0]) / 60)
        out['nan_L'] = float(dm['l_paw_x'].isna().mean())
        out['nan_R'] = float(dm['r_paw_x'].isna().mean())

        sp_L, jit_L = _speed(dm['l_paw_x'].to_numpy(float), dm['l_paw_y'].to_numpy(float), 2)
        sp_R, jit_R = _speed(dm['r_paw_x'].to_numpy(float), dm['r_paw_y'].to_numpy(float), 1)
        out['jitter_L'], out['jitter_R'] = jit_L, jit_R
        out['li_jitter'] = li(jit_L, jit_R)

        # ---- amplitude, on bins where BOTH paws are tracked (paired within session) ----
        both = np.isfinite(sp_L) & np.isfinite(sp_R)
        out['n_both'] = int(both.sum())
        if both.sum() > 6000:
            L, R = sp_L[both], sp_R[both]
            for tag, fn in [('med', np.median), ('mean', np.mean),
                            ('p90', lambda v: np.quantile(v, 0.90))]:
                out[f'sp{tag}_L'], out[f'sp{tag}_R'] = float(fn(L)), float(fn(R))
                out[f'li_sp{tag}'] = li(float(fn(L)), float(fn(R)))
            # bilateral coupling / temporal precedence
            out['lag_s'], out['xcorr_peak'] = _lead_lag(sp_L, sp_R)
            out['r_LR_speed'] = float(stats.pearsonr(L, R)[0])
            # scale-free bout structure at matched duty cycle
            for tag, sp in [('L', sp_L), ('R', sp_R)]:
                out[f'boutrate_{tag}'], out[f'boutdur_{tag}'] = _bout_stats(sp)
            out['li_boutrate'] = li(out['boutrate_L'], out['boutrate_R'])
            out['li_boutdur'] = li(out['boutdur_L'], out['boutdur_R'])
            # wheel engagement, for the task-relation analysis
            wh = dm['avg_wheel_vel'].to_numpy(float)[:-1][both]
            out['wheel_bias'] = float(np.nanmean(wh > 0) - np.nanmean(wh < 0))
            mv = np.isfinite(wh) & (np.abs(wh) > np.nanquantile(np.abs(wh), 0.75))
            if mv.sum() > 2000:   # amplitude bias restricted to active wheel turning
                out['li_spmed_wheel'] = li(float(np.median(L[mv])), float(np.median(R[mv])))

        # ---- wavelet band power: the primary amplitude measure ----
        wv_file = WV_PATH + f'paw_vel_wavelets_{eid}_{mouse}'
        if os.path.exists(wv_file):
            cols = [f'{p}_paw_{ax}{f}' for p in 'lr' for ax in 'xy'
                    for f in BAND + NOISE_BAND]
            wv = pd.read_parquet(wv_file, columns=cols)
            ok = wv.notna().all(axis=1).to_numpy()
            out['n_wv'] = int(ok.sum())
            if ok.sum() > 6000:
                w = wv[ok]
                pw = {}
                for p, tag in [('l', 'L'), ('r', 'R')]:
                    for f in BAND + NOISE_BAND:
                        # combine x and y into one magnitude per band
                        pw[(tag, f)] = float(np.mean(np.hypot(w[f'{p}_paw_x{f}'],
                                                              w[f'{p}_paw_y{f}'])))
                for tag in 'LR':
                    out[f'bandpow_{tag}'] = sum(pw[(tag, f)] for f in BAND)
                    out[f'noisepow_{tag}'] = pw[(tag, NOISE_BAND[0])]
                out['li_bandpow'] = li(out['bandpow_L'], out['bandpow_R'])
                out['li_noisepow'] = li(out['noisepow_L'], out['noisepow_R'])
                # per-frequency LI, and scale-free spectral shape (power normalised within paw)
                for f in BAND:
                    out[f'li_pow_{f}'] = li(pw[('L', f)], pw[('R', f)])
                    for tag in 'LR':
                        out[f'shape_{tag}_{f}'] = pw[(tag, f)] / out[f'bandpow_{tag}']
                    out[f'li_shape_{f}'] = li(out[f'shape_L_{f}'], out[f'shape_R_{f}'])
                # split-half reliability of the amplitude LI (measurement-noise ceiling)
                h = ok.nonzero()[0]
                for half, idx in [('h1', h[:len(h) // 2]), ('h2', h[len(h) // 2:])]:
                    ww = wv.iloc[idx]   # positional: the parquet index is not 0-based
                    bp = {tag: sum(float(np.mean(np.hypot(ww[f'{p}_paw_x{f}'], ww[f'{p}_paw_y{f}'])))
                                   for f in BAND) for p, tag in [('l', 'L'), ('r', 'R')]}
                    out[f'li_bandpow_{half}'] = li(bp['L'], bp['R'])

        # ---- trials: choice / performance, for the task-relation analysis ----
        tf = DM_PATH + f'session_trials_{eid}_{mouse}'
        if os.path.exists(tf):
            tr = pd.read_parquet(tf)
            if 'choice' in tr:
                ch = tr['choice'].to_numpy(float)
                out['n_trials'] = int(np.isfinite(ch).sum())
                out['frac_right_choice'] = float(np.mean(ch[np.isfinite(ch)] == -1))
                if 'contrastLeft' in tr and 'contrastRight' in tr:
                    cl = tr['contrastLeft'].fillna(0).to_numpy(float)
                    cr = tr['contrastRight'].fillna(0).to_numpy(float)
                    zero = (cl == 0) & (cr == 0) & np.isfinite(ch)
                    if zero.sum() > 20:   # unbiased readout of side preference
                        out['frac_right_zero'] = float(np.mean(ch[zero] == -1))
                if 'feedbackType' in tr:
                    out['perf'] = float(np.mean(tr['feedbackType'] == 1))
        return out
    except Exception as e:
        return dict(eid=eid, mouse=mouse, err=f'{type(e).__name__}: {e}')


def main(out_csv):
    files = [f for f in os.listdir(WV_PATH) if f.startswith('paw_vel_wavelets_')]
    items = sorted({(f[17:53], f[54:]) for f in files
                    if f[17:53] not in SESSIONS_TO_EXCLUDE
                    and os.path.exists(DM_PATH + f'design_matrix_{f[17:53]}_{f[54:]}')})
    print(f'{len(items)} sessions to process')
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, items, chunksize=2))
    df = pd.DataFrame(res)

    # attach lab from the local ONE cache (no network needed)
    try:
        s = pd.read_parquet('/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/sessions.pqt')
        df = df.merge(s[['lab']], left_on='eid', right_index=True, how='left')
    except Exception as e:
        print('lab merge skipped:', e)
    df.to_csv(out_csv, index=False)
    if 'err' in df:
        bad = df[df.err.notna()]
        print(f'{len(bad)} failures')
        for _, r in bad.head(10).iterrows():
            print('  ', r.eid, r.err)
    print('wrote', out_csv, df.shape)


if __name__ == '__main__':
    main(sys.argv[1])
