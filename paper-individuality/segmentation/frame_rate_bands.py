"""
Which wavelet bands does the camera frame rate change?
======================================================
Proficient sessions are filmed at 60 Hz; Pre-rec (on the same recording rig), the earlier training stages and the
neuromodulator sessions at 30 Hz. The left paw + wheel states cluster 0.5-8 Hz Morlet amplitudes (3.1.1 / 3.1.2), each band z-scored
within its session (3.3, NORM = 'session'). So a pure per-band GAIN difference between frame rates cancels (the
2x of get_speed's hard-coded 60 fps, the finite-difference roll-off, the wheel's longer bins). What can
survive is anything not multiplicative: an added noise floor (tracking jitter aliased from 15-30 Hz into
the top band, Lightning Pose's frame-rate-dependent smoothing) or a change in how bands co-vary.

The measures are therefore gain-invariant, per session and band:
  floor      q10 / q50 of the amplitude: a noise floor lifts the quiet frames towards the median
  peak       q95 / q50: dynamic range above the median
  r_2hz      correlation of the band with the same signal's 2 Hz band (band coupling)
  r_wheel    correlation of the left-paw x band with the wheel's same band (paw-wheel coupling)

Two tests:
  A  DECIMATION. A 60 Hz proficient session, every other frame kept (wheel: mean of each frame pair, as a
     30 Hz bin), through the pipeline's own get_speed + fast_wavelet_morlet_convolution_parallel with
     its dt rounding. Isolates sampling (aliasing, binning, dt) -- NOT the pose estimation, which was
     run on the 60 Hz video.
  B  SAME MICE, SAME RIG, 30 vs 60 Hz. The existing wavelet files: Pre-rec (biased sessions on the recording
     rig, camera at 30 Hz) vs Proficient (same rig, 60 Hz) for the mice that have both. Everything that comes
     with acquiring at 30 Hz (exposure, motion blur, pose estimated on 30 fps video), plus a few days of
     behaviour and the protocol. Baseline: Pre-rec vs Pre-rec of the same mouse (same settings, different days).
"""
import os
import pathlib

import numpy as np
import pandas as pd

from segmentation_functions import get_speed, fast_wavelet_morlet_convolution_parallel

DATA = pathlib.Path(__file__).resolve().parents[1] / 'data'
BANDS = [0.5, 1.0, 2.0, 4.0, 8.0]
SIGNALS = ['l_paw_x', 'l_paw_y', 'avg_wheel_vel']
F = np.array([.25, .5, 1, 2, 4, 8, 16, 32])       # as 3.1.1 / 3.1.2 (the clustering keeps 0.5-8)
OMEGA0 = 5
OUT = DATA / 'frame_rate_bands'


def cols(sig):
    return [f'{sig}{b}' for b in BANDS]


# ------------------------------------------------------------------------------------------------
# The pipeline, as 3.1.1 (paw, velocity, split per axis) and 3.1.2 (wheel)
# ------------------------------------------------------------------------------------------------
def wavelets(dm):
    """Design-matrix frame (Bin, l_paw_x, l_paw_y, avg_wheel_vel) -> the wavelet columns 3.1.x write.
    NaNs dropped before the transform and the amplitudes written back at their frames, as there."""
    out = pd.DataFrame({'Bin': dm['Bin'].to_numpy()})
    rate = np.round(1 / np.nanmean(dm['Bin'].diff()))
    dt = np.round(np.mean(np.diff(dm['Bin'])), 3)
    speeds = get_speed(dm, dm['Bin'], 'left', rate, split=True, feature='l_paw')
    arrays = {'l_paw_x': np.asarray(speeds[1]), 'l_paw_y': np.asarray(speeds[2]),
              'avg_wheel_vel': dm['avg_wheel_vel'].to_numpy(float)}
    for sig, a in arrays.items():
        ok = ~np.isnan(a)
        amp, _, _ = fast_wavelet_morlet_convolution_parallel(a[ok], F, OMEGA0, dt)
        for i, f in enumerate(F):
            if f in BANDS:
                col = np.full(len(a), np.nan)
                col[ok] = amp[i]
                out[f'{sig}{f}'] = col
    return out


def decimate(dm):
    """Every other frame of a 60 Hz design matrix; the wheel as the mean of each frame pair (a 30 Hz bin)."""
    n = len(dm) // 2 * 2
    d = dm.iloc[:n:2, :].reset_index(drop=True).copy()
    w = dm['avg_wheel_vel'].to_numpy(float)[:n]
    d['avg_wheel_vel'] = np.nanmean(w.reshape(-1, 2), axis=1)
    return d


# ------------------------------------------------------------------------------------------------
# Gain-invariant band measures
# ------------------------------------------------------------------------------------------------
def band_measures(w):
    """One row per (signal, band): floor, peak, r_2hz, r_wheel (see the module docstring)."""
    rows = []
    for sig in SIGNALS:
        for b in BANDS:
            c = f'{sig}{b}'
            if c not in w:
                continue
            x = w[c].to_numpy(float)
            ok = np.isfinite(x)
            q10, q50, q95 = np.percentile(x[ok], [10, 50, 95])
            ref = w[f'{sig}2.0'].to_numpy(float)
            m2 = ok & np.isfinite(ref)
            row = dict(signal=sig, band=b, floor=q10 / q50, peak=q95 / q50,
                       r_2hz=np.corrcoef(x[m2], ref[m2])[0, 1] if b != 2.0 else np.nan)
            wh = w.get(f'avg_wheel_vel{b}')
            if sig == 'l_paw_x' and wh is not None:
                whv = wh.to_numpy(float)
                mw = ok & np.isfinite(whv)
                row['r_wheel'] = np.corrcoef(x[mw], whv[mw])[0, 1]
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# Test A: decimation
# ------------------------------------------------------------------------------------------------
def decimation_session(path):
    """Measures at 60 Hz and decimated to 30 Hz, plus the frame-matched correlation of each band's
    session-z-scored amplitude between the two (60 Hz at the kept frames)."""
    dm = pd.read_parquet(path, columns=['Bin', 'l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y', 'avg_wheel_vel'])
    w60, w30 = wavelets(dm), wavelets(decimate(dm))
    m60, m30 = band_measures(w60), band_measures(w30)
    k60 = w60.iloc[:len(w30) * 2:2].reset_index(drop=True)
    r = []
    for sig in SIGNALS:
        for b in BANDS:
            a, c = k60[f'{sig}{b}'].to_numpy(float), w30[f'{sig}{b}'].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(c)
            r.append(dict(signal=sig, band=b, r_60_30=np.corrcoef(a[ok], c[ok])[0, 1]))
    out = m60.merge(m30, on=['signal', 'band'], suffixes=('_60', '_30')).merge(pd.DataFrame(r), on=['signal', 'band'])
    return out


def decimation_test(n_sessions=30, seed=0, cache=True):
    f = OUT / f'decimation_{n_sessions}.csv'
    if cache and f.exists():
        return pd.read_csv(f)
    dms = sorted(p for p in (DATA / 'design_matrices').glob('design_matrix_*') if p.is_file())
    pick = np.random.default_rng(seed).choice(len(dms), min(n_sessions, len(dms)), replace=False)
    t = pd.concat([decimation_session(dms[i]).assign(session=dms[i].name.split('_')[2]) for i in sorted(pick)])
    OUT.mkdir(exist_ok=True)
    t.to_csv(f, index=False)
    return t


# ------------------------------------------------------------------------------------------------
# Test B: same mice, same rig, 30 vs 60 Hz camera (existing wavelet files)
# ------------------------------------------------------------------------------------------------
def _wavelet_file(root, kind, eid, mouse):
    pre = 'paw_vel_wavelets_' if kind == 'paw' else 'wheel_vel_wavelets_'
    sub = 'left_paw_wavelets' if (kind == 'paw' and root == 'training') else f'{kind}_wavelets'
    return DATA / (f'{root}/' if root else '') / sub / f'{pre}{eid}_{mouse}'


def stored_wavelets(root, eid, mouse):
    """The wavelet columns the clustering read for one session (paw and wheel files merged on Bin)."""
    p = pd.read_parquet(_wavelet_file(root, 'paw', eid, mouse), columns=['Bin'] + cols('l_paw_x') + cols('l_paw_y'))
    w = pd.read_parquet(_wavelet_file(root, 'wheel', eid, mouse), columns=['Bin'] + cols('avg_wheel_vel'))
    return p.merge(w, on='Bin', how='inner')


def rate_pairs():
    """Sessions of the mice with both Pre-rec (30 Hz) and Proficient (60 Hz) wavelets, same recording rig."""
    def ok(root, eid, mouse):
        return all(_wavelet_file(root, k, eid, mouse).exists() for k in ('paw', 'wheel'))
    pre = [(x.split('_')[2], x.split('_', 3)[3]) for x in os.listdir(DATA / 'design_matrices/1_camera_setup/biased')
           if x.startswith('design_matrix_')]
    pre = [(e, m) for e, m in pre if ok('training', e, m)]
    prof = [(x.split('_')[3], x.split('_', 4)[4]) for x in os.listdir(DATA / 'paw_wavelets') if x.startswith('paw_vel')]
    prof = [(e, m) for e, m in prof if ok('', e, m)]
    mice = {m for _, m in pre} & {m for _, m in prof}
    return pd.DataFrame([dict(session=e, mouse=m, camera='Pre-rec (30 Hz)', root='training') for e, m in pre if m in mice]
                        + [dict(session=e, mouse=m, camera='Proficient (60 Hz)', root='') for e, m in prof if m in mice])


def rate_test(cache=True):
    f = OUT / 'rate_pairs.csv'
    if cache and f.exists():
        return pd.read_csv(f)
    S = rate_pairs()
    t = pd.concat([band_measures(stored_wavelets(r.root, r.session, r.mouse))
                   .assign(session=r.session, mouse=r.mouse, camera=r.camera) for r in S.itertuples()])
    OUT.mkdir(exist_ok=True)
    t.to_csv(f, index=False)
    return t


def rate_contrast(t, measure):
    """Per mouse and (signal, band): Proficient minus Pre-rec (mean over each setting's sessions), and the
    same-setting baseline |Pre-rec_i - Pre-rec_j| (first two Pre-rec sessions). -> per (signal, band): median
    rig difference, how many mice in its direction, median same-setting |difference|."""
    g = t.groupby(['mouse', 'signal', 'band', 'camera'])[measure].mean().unstack('camera')
    diff = (g['Proficient (60 Hz)'] - g['Pre-rec (30 Hz)']).rename('d')
    pre = t[t['camera'] == 'Pre-rec (30 Hz)'].sort_values('session')
    same = (pre.groupby(['mouse', 'signal', 'band'])[measure]
               .apply(lambda s: abs(s.iloc[0] - s.iloc[1]) if len(s) >= 2 else np.nan).rename('same_rig'))
    from scipy.stats import wilcoxon
    rows = []
    for (sig, b), d in diff.dropna().groupby(level=['signal', 'band']):
        s = same.loc[(slice(None), sig, b)].dropna()
        rows.append(dict(signal=sig, band=b, measure=measure, n_mice=len(d), diff_60_30=d.median(),
                         frac_same_sign=np.mean(np.sign(d) == np.sign(d.median())),
                         p=wilcoxon(d).pvalue if len(d) > 5 else np.nan,
                         abs_diff=d.abs().median(), same_setting_abs=s.median()))
    return pd.DataFrame(rows)
