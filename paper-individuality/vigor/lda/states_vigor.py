"""
PER-SESSION MOVEMENT VIGOR FROM THE CURRENT states_files
=========================================================
Rebuilds the vigor metrics of `vigor/wheel/wheel_vigor.py` directly from
`data/states_files/`, and adds the same metrics for each FOREPAW, which had never
been computed.

WHY REBUILD RATHER THAN REUSE THE CACHE
    The cached `vigor_sessions.parquet` was built on a machine whose proficient
    design matrices came from a Google Drive mount, and neither the cache nor that
    mount is present here. The raw signals are however identical: for a session
    present in both, `design_matrices/design_matrix_<eid>_<mouse>` and
    `states_files/8_states_file_<eid>_<mouse>` agree to max|diff| = 0 on
    avg_wheel_vel, l_paw_x, r_paw_x and whisker_me, on the same Bin grid. So this
    is a re-derivation on the current file set (332 sessions, 101 mice), not a
    different measurement -- and it additionally carries `broader_label`, so vigor
    can be split by task epoch, which is the space the LDA features live in.

WHEEL -- reproduces wheel_vigor.session_vigor exactly
    Same resample-to-30 Hz, same MOVE_THRESH, same winsorising, same metric set, so
    the numbers are comparable with the early-vs-proficient analysis. All proficient
    sessions here are natively 60 Hz (checked, printed), so the resampling is a
    no-op for between-mouse ranking; it is kept for comparability.

PAW -- new
    `l_paw_*` is in native LEFT-camera pixels (1280x1024) and `r_paw_*` in native
    RIGHT-camera pixels (640x512); the design-matrix pipeline's `get_speed` divides
    the left camera by 2 and that factor is correct (segmentation/paw_bias measured
    the rig's spatial symmetry from the lick tube: 1.980, p = 0.25 against 2.0). The
    same /2 is applied here, so both paws are in right-camera pixel units.

    Speed is computed at the native 60 Hz, NOT on the 30 Hz grid: differencing an
    averaged position is a different estimator and there is no mixed-rate problem to
    solve inside the proficient set.

    Tracking NaNs are DROPPED, never interpolated, and a difference is only taken
    between samples that are one frame apart -- otherwise a gap would be read as a
    huge excursion.

    THE CAMERA CAVEAT. Paw amplitude is NOT gain-free: the left camera usually runs
    at 60 fps and the right at 150 fps, and lightningPose's smoother removes ~28% of
    real 8 Hz movement at 60 fps against ~4% at 150 fps. That biases the two paws
    differently and it is per-rig. Hence:
      - both paws are reported separately as well as averaged;
      - `hf_noise_*` (20-29 Hz velocity band power) is carried as a covariate so the
        tracking-noise/gain axis can be partialled out downstream;
      - `bouts_p75` metrics use a threshold set from the session's OWN speed
        distribution, so they are invariant to any per-session multiplicative gain.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
STATES_DIR = os.path.join(ROOT, 'data', 'states_files')
PREFIX = '8_'                      # the 8-state files: the 332-session paper cohort

RESAMPLE_HZ = 30.0                 # wheel only, as in wheel_vigor.py
MOVE_THRESH = 0.1                  # wheel |velocity| above this counts as moving
PAW_MOVE_THRESH = 20.0             # px/s; a fixed global threshold (see README)
MIN_BOUT_S = 0.1
WINSOR_P = 99.5
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']

PAW_COLS = ['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y']
READ_COLS = ['Bin', 'avg_wheel_vel', 'trial_id', 'broader_label'] + PAW_COLS


# --------------------------------------------------------------------------- utils
def _bouts(moving, width, min_bout_s=MIN_BOUT_S):
    """(n_bouts, mean duration in s) for contiguous runs of `moving`."""
    if not moving.any():
        return 0, np.nan
    d = np.diff(np.concatenate([[0], moving.view(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    dur = (ends - starts) * width
    keep = dur >= min_bout_s
    return int(keep.sum()), (float(dur[keep].mean()) if keep.any() else np.nan)


def _resample(bins, vel, hz=RESAMPLE_HZ):
    """Mean velocity within fixed-width windows on a common grid (wheel_vigor.py)."""
    if len(bins) < 2:
        return np.array([]), np.nan
    width = 1.0 / hz
    idx = np.floor((bins - bins[0]) / width).astype(np.int64)
    n = idx[-1] + 1
    s = np.bincount(idx, weights=vel, minlength=n)
    c = np.bincount(idx, minlength=n)
    ok = c > 0
    return s[ok] / c[ok], width


def _speed_metrics(speed, width, thresh, prefix, dur_min, p75_bouts=False):
    """The shared metric block: rates and distributional summaries, never totals."""
    if not len(speed):
        return {}
    cap = np.percentile(speed, WINSOR_P)
    sw = np.minimum(speed, cap)
    moving = speed > thresh
    n_bout, bout_dur = _bouts(moving, width)
    out = {
        f'{prefix}_mean_speed': float(sw.mean()),
        f'{prefix}_mean_speed_moving': float(sw[moving].mean()) if moving.any() else np.nan,
        f'{prefix}_median_speed': float(np.median(speed)),
        f'{prefix}_p95_speed': float(np.percentile(speed, 95)),
        f'{prefix}_sd_speed': float(sw.std()),
        f'{prefix}_frac_moving': float(moving.mean()),
        f'{prefix}_bout_rate_per_min': float(n_bout / dur_min) if dur_min > 0 else np.nan,
        f'{prefix}_mean_bout_s': bout_dur,
        f'{prefix}_distance_per_min': float(sw.sum() * width / dur_min) if dur_min > 0 else np.nan,
    }
    if p75_bouts:
        # GAIN-INVARIANT. The threshold is the session's own 75th percentile, so the
        # duty cycle is 0.25 by construction and any per-session multiplicative gain
        # (camera distance, smoother attenuation) cancels exactly. What is left is
        # the TEMPORAL structure: is the movement in few long bouts or many short ones.
        mv = speed > np.percentile(speed, 75)
        nb, bd = _bouts(mv, width)
        out[f'{prefix}_bout_rate_p75'] = float(nb / dur_min) if dur_min > 0 else np.nan
        out[f'{prefix}_mean_bout_p75_s'] = bd
    return out


def _band_power(v, fs, lo, hi):
    """Mean PSD of a velocity trace in [lo, hi) Hz. Welch, 4 s segments."""
    if len(v) < int(4 * fs):
        return np.nan
    f, p = signal.welch(v - np.nanmean(v), fs=fs, nperseg=int(4 * fs))
    m = (f >= lo) & (f < hi)
    return float(p[m].mean()) if m.any() else np.nan


def _paw_speed(x, y, bins, fs):
    """Frame-to-frame speed of one paw, NaNs dropped and gaps respected.

    Returns (speed, n_used, nan_frac). Only differences between samples exactly one
    frame apart are kept: a difference across a dropout is a tracking gap, not a
    movement, and interpolating it would invent one.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    nan_frac = float(1 - ok.mean())
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return np.array([]), 0, nan_frac
    adj = np.diff(idx) == 1
    dx = np.diff(x[idx])[adj]
    dy = np.diff(y[idx])[adj]
    return np.hypot(dx, dy) * fs, int(adj.sum()), nan_frac


# --------------------------------------------------------------------- one session
def session_vigor(path, mouse_name, session):
    d = pd.read_parquet(path, columns=READ_COLS)
    b = d['Bin'].to_numpy(float)
    fs = 1.0 / np.median(np.diff(b))
    row = dict(mouse_name=mouse_name, session=session, native_hz=float(fs))

    # ---- wheel: exactly wheel_vigor.session_vigor -----------------------------
    v = d['avg_wheel_vel'].to_numpy(float)
    ok = np.isfinite(b) & np.isfinite(v)
    vr, width = _resample(b[ok], v[ok])
    if len(vr):
        speed = np.abs(vr)
        cap = np.percentile(speed, WINSOR_P)
        dur_min = len(vr) * width / 60.0
        row.update(_speed_metrics(speed, width, MOVE_THRESH, 'wheel', dur_min,
                                  p75_bouts=True))
        mv = speed > MOVE_THRESH
        row['wheel_mean_vel_signed'] = float(
            np.minimum(np.abs(vr), cap).mean() * np.sign(vr).mean())
        row['wheel_turn_bias'] = float(np.sign(vr[mv]).mean()) if mv.any() else np.nan
        row['duration_min'] = float(dur_min)
        row['n_samples'] = int(len(vr))
    row['wheel_hf_noise'] = _band_power(v[ok], fs, 20, 29)

    # ---- paws -----------------------------------------------------------------
    # /2 on the left camera: native 1280x1024 vs 640x512. The factor is correct
    # (paw_bias measured 1.980 +/- from rig landmarks), so both paws end up in the
    # same spatial units.
    xy = {'l_paw': (d['l_paw_x'].to_numpy(float) / 2.0, d['l_paw_y'].to_numpy(float) / 2.0),
          'r_paw': (d['r_paw_x'].to_numpy(float), d['r_paw_y'].to_numpy(float))}
    per_paw = {}
    for paw, (x, y) in xy.items():
        sp, n_used, nan_frac = _paw_speed(x, y, b, fs)
        row[f'{paw}_nan_frac'] = nan_frac
        if not len(sp):
            continue
        dur_min = n_used / fs / 60.0
        row.update(_speed_metrics(sp, 1.0 / fs, PAW_MOVE_THRESH, paw, dur_min,
                                  p75_bouts=True))
        per_paw[paw] = sp
        # noise / amplitude band powers on the SPEED trace, for the gain controls
        row[f'{paw}_hf_noise'] = _band_power(sp, fs, 20, 29)
        row[f'{paw}_band_0.5_8'] = _band_power(sp, fs, 0.5, 8)

    # BOTH PAWS TOGETHER = "paw vigor" proper. Averaging the two per-paw scalars,
    # not concatenating the traces, so a paw with more dropouts does not get less
    # weight in a way that depends on its dropout rate.
    for k in [c.split('l_paw_')[-1] for c in row if c.startswith('l_paw_')
              and not c.endswith('nan_frac')]:
        a, c = row.get(f'l_paw_{k}', np.nan), row.get(f'r_paw_{k}', np.nan)
        row[f'paw_{k}'] = float(np.nanmean([a, c])) if np.isfinite([a, c]).any() else np.nan
    # laterality of amplitude, for continuity with vigor/laterality
    a, c = row.get('l_paw_band_0.5_8', np.nan), row.get('r_paw_band_0.5_8', np.nan)
    row['paw_LI_band'] = float((a - c) / (a + c)) if np.isfinite(a) and np.isfinite(c) else np.nan

    # ---- per-epoch wheel and paw speed ----------------------------------------
    # The LDA features are per-epoch syllable occupancies, so epoch-resolved vigor is
    # the matched readout. Native rate, means only -- an epoch is short and the
    # threshold-crossing metrics would be dominated by its length.
    lab = d['broader_label'].to_numpy(object)
    wheel_speed_native = np.abs(v)
    lp = np.hypot(np.diff(xy['l_paw'][0]), np.diff(xy['l_paw'][1])) * fs
    rp = np.hypot(np.diff(xy['r_paw'][0]), np.diff(xy['r_paw'][1])) * fs
    both = np.nanmean(np.vstack([lp, rp]), axis=0)
    for ep in EPOCHS:
        m = (lab == ep)
        row[f'wheel_speed_{ep}'] = float(np.nanmean(wheel_speed_native[m])) if m.any() else np.nan
        m2 = m[:-1]
        row[f'paw_speed_{ep}'] = float(np.nanmean(both[m2])) if m2.any() else np.nan
    return row


# --------------------------------------------------------------------------- build
def build(out_csv=None, prefix=PREFIX, limit=None, verbose=True):
    pat = re.compile(r'^' + re.escape(prefix) + r'states_file_([0-9a-f\-]{36})_(.+)$')
    files = sorted(f for f in os.listdir(STATES_DIR) if pat.match(f))
    if limit:
        files = files[:limit]
    rows = []
    for i, f in enumerate(files):
        eid, mouse = pat.match(f).groups()
        try:
            rows.append(session_vigor(os.path.join(STATES_DIR, f), mouse, eid))
        except Exception as e:                                   # noqa: BLE001
            print(f'  FAILED {f}: {type(e).__name__}: {e}', file=sys.stderr)
        if verbose and (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(files)}', flush=True)
    V = pd.DataFrame(rows)
    if out_csv:
        V.to_csv(out_csv, index=False)
        print(f'wrote {out_csv}: {V.shape[0]} sessions x {V.shape[1]} cols')
    return V


if __name__ == '__main__':
    out = os.path.join(HERE, 'vigor_sessions_states.csv')
    V = build(out_csv=out)
    print(V['native_hz'].round(1).value_counts().to_string())
