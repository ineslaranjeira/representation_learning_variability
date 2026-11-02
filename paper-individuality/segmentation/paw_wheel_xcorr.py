"""
Paw x wheel cross-correlation QC for the paper-individuality (BWM) sessions.
============================================================================
The same timing/tracking QC the neuromodulator collaborators run on their Lightning Pose sessions
(davcrom/ibl_neuromodulators: iblnm.analysis.per_third_crosscorr, iblnm.data.run_pose_qc; columns
peak_val_* / peak_lag_* / drift in their LightningPoseSessions sheet; their pose_qc filter keeps
sessions with min peak value >= 0.5). Ported here so the numbers are directly comparable:

  paw speed    sum over paw_l and paw_r of the frame-to-frame keypoint displacement (pixels / frame),
               frames with likelihood < 0.9 set to NaN; frames where both paws are NaN dropped
  wheel speed  |velocity| of the wheel position interpolated onto a 100 Hz grid and differentiated
               with brainbox's velocity_filtered (their `differentiate`)
  per third    both traces linearly interpolated onto a common 100 Hz grid over their overlap, the
               session split into three equal-time thirds; in each, the normalized cross-correlation
               (z-scored, divided by n) over lags of +-5 s
  outputs      peak_val (max of the cross-correlation) and peak_lag (its lag, s; > 0 = paw leads
               wheel) per third, drift = late lag - early lag

Done for the left and, since the paper's design matrices use both cameras, the right camera.

DISK. Lightning Pose files are 80-300 MB each and the disk is nearly full, so every LP file this
script downloads is deleted once the session is done. Files already in the ONE cache are left alone.

Usage (iblenv):  python paw_wheel_xcorr.py        # resumes: sessions already in the output are skipped
Output: data_query/paw_wheel_xcorr.csv, one row per session x camera.
"""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault('TQDM_DISABLE', '1')
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_query' / 'paw_wheel_xcorr.csv'
FS = 100                  # Hz, common grid (theirs: CROSSCORR_FS = WHEEL_FS)
LAG_WINDOW = 5.0          # s
LIKELIHOOD_THRESHOLD = 0.9
PAWS = ['paw_l', 'paw_r']
CAMERAS = ['left', 'right']


def keypoint_speed(x, y, likelihood, threshold=LIKELIHOOD_THRESHOLD):
    speed = np.concatenate(([np.nan], np.hypot(np.diff(x), np.diff(y))))
    speed[likelihood < threshold] = np.nan
    return speed


def paw_speed(pose):
    speeds = np.column_stack([keypoint_speed(pose[f'{k}_x'].to_numpy(), pose[f'{k}_y'].to_numpy(),
                                             pose[f'{k}_likelihood'].to_numpy()) for k in PAWS])
    trace = np.nansum(speeds, axis=1)
    trace[np.isnan(speeds).all(axis=1)] = np.nan
    return trace


def wheel_speed(position, times, fs=FS):
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered
    pos, t = interpolate_position(times, position, freq=fs)
    vel, _ = velocity_filtered(pos, fs=fs)
    return np.abs(vel), t


def normalized_crosscorr(a, b, fs, lag_window=LAG_WINDOW):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    cc = correlate((b - b.mean()) / b.std(), (a - a.mean()) / a.std(), mode='full') / n
    lags = correlation_lags(len(b), n, mode='full') / fs
    keep = np.abs(lags) <= lag_window
    cc, lags = cc[keep], lags[keep]
    return cc, lags, lags[np.argmax(cc)]


def per_third_crosscorr(paw, paw_t, wheel, wheel_t, fs=FS, lag_window=LAG_WINDOW):
    grid = np.arange(max(paw_t[0], wheel_t[0]), min(paw_t[-1], wheel_t[-1]), 1 / fs)
    p = np.interp(grid, paw_t, paw)
    w = np.interp(grid, wheel_t, wheel)
    edges = np.linspace(grid[0], grid[-1], 4)
    peak_vals, peak_lags = np.full(3, np.nan), np.full(3, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (grid >= lo) & (grid < hi)
        cc, _, peak_lags[i] = normalized_crosscorr(p[m], w[m], fs, lag_window)
        peak_vals[i] = cc.max()
    return peak_vals, peak_lags, peak_lags[-1] - peak_lags[0]


def session_qc(one, eid):
    from brainbox.io.one import SessionLoader
    rows = []
    wheel = one.load_object(eid, 'wheel', collection='alf')
    w, wt = wheel_speed(wheel['position'], wheel['timestamps'])
    for cam in CAMERAS:
        row = dict(eid=eid, camera=cam)
        lp_path = one.eid2path(eid) / 'alf' / f'_ibl_{cam}Camera.lightningPose.pqt'
        cached = lp_path.exists()
        try:
            sl = SessionLoader(eid=eid, one=one)
            sl.load_pose(views=[cam], tracker='lightningPose')
            pose = sl.pose[f'{cam}Camera']
            sp = paw_speed(pose)
            ok = np.isfinite(sp)
            vals, lags, drift = per_third_crosscorr(sp[ok], pose['times'].to_numpy()[ok], w, wt)
            row.update(frame_rate=1 / np.median(np.diff(pose['times'])), frac_tracked=ok.mean(),
                       **{f'peak_val_{s}': v for s, v in zip(['early', 'mid', 'late'], vals)},
                       **{f'peak_lag_{s}': v for s, v in zip(['early', 'mid', 'late'], lags)}, drift=drift)
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'[:200]
        finally:
            if not cached and lp_path.exists():
                lp_path.unlink()
        rows.append(row)
    return rows


def main():
    sys.path.insert(0, str(HERE.parent / 'GLM-HMM' / 'transition_prediction'))
    import tp_data as td
    from one.api import ONE
    one = ONE(base_url='https://alyx.internationalbrainlab.org', silent=True)
    eids = list(td.load_lda()['session'].unique())
    done = set(pd.read_csv(OUT)['eid']) if OUT.exists() else set()
    todo = [e for e in eids if e not in done]
    print(f'{len(eids)} sessions, {len(todo)} to do', flush=True)
    for i, eid in enumerate(todo):
        try:
            rows = session_qc(one, eid)
        except Exception:
            rows = [dict(eid=eid, camera=c, error=traceback.format_exc(limit=1)[-200:]) for c in CAMERAS]
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not OUT.exists(), index=False)
        print(f'{i + 1}/{len(todo)} {eid}', flush=True)


if __name__ == '__main__':
    main()
