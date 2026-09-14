"""
SESSION-LEVEL MOVEMENT VIGOR FROM THE RAW DESIGN MATRIX
=======================================================
A deliberately simple readout: how much does this mouse move the wheel, over the
whole session. No go-cue alignment, no epoch labelling, no time warping -- which is
not just a shortcut, it is what makes this analysis better powered than the 40-d
profile. A 40-number profile estimated from ~200 early trials is noisy in every
dimension; a scalar pools all of it into one well-estimated number, and attenuation
by measurement error scales with how badly each feature is measured.

THE CONFOUND THAT HAD TO BE FIXED FIRST
    Training design matrices are sampled at ~30 Hz, proficient ones at ~60 Hz.
    Averaging over a bin twice as long smooths the signal, which pushes DOWN any
    threshold-crossing fraction and any extreme-value statistic. Comparing a 30 Hz
    session against a 60 Hz one would therefore show a difference that is pure
    sampling rate.

    Correlation across mice is immune to a constant offset between the two
    timepoints -- but NOT to sampling rate varying BETWEEN MICE within a timepoint,
    which would inject real noise into the ranking. Rather than rely on it being
    constant, every session is resampled onto a common grid (RESAMPLE_HZ) before any
    metric is computed. `sampling_hz` is kept as a column so the assumption stays
    checkable instead of implicit.

    Wheel traces also carry occasional huge spikes (|v| up to ~98 in one training
    session vs a 95th percentile near 2), so speeds are winsorised before the
    mean-like metrics are taken. The percentile statistics are unaffected either way.
"""
import os
import numpy as np
import pandas as pd

import wheel_preprocessing as wp

RESAMPLE_HZ = 30.0        # common grid; the coarser of the two native rates
MOVE_THRESH = 0.1         # |velocity| above this counts as "moving"
MIN_BOUT_S = 0.1          # a bout must last at least this long
WINSOR_P = 99.5           # clip speeds above this percentile before averaging


def _resample(bins, vel, hz=RESAMPLE_HZ):
    """Mean velocity within fixed-width windows on a common grid.

    Averaging (not decimation) is what makes a 60 Hz trace comparable with a 30 Hz
    one: it applies the same smoothing the coarser recording already underwent.
    """
    if len(bins) < 2:
        return np.array([]), np.nan
    width = 1.0 / hz
    t0 = bins[0]
    idx = np.floor((bins - t0) / width).astype(np.int64)
    n = idx[-1] + 1
    s = np.bincount(idx, weights=vel, minlength=n)
    c = np.bincount(idx, minlength=n)
    ok = c > 0
    return s[ok] / c[ok], width


def _bouts(moving, width, min_bout_s=MIN_BOUT_S):
    """(n_bouts, mean duration in s) for contiguous runs of `moving`."""
    if not moving.any():
        return 0, np.nan
    d = np.diff(np.concatenate([[0], moving.view(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    dur = (ends - starts) * width
    keep = dur >= min_bout_s
    return int(keep.sum()), float(dur[keep].mean()) if keep.any() else np.nan


def session_vigor(mouse_name, session, source, hz=RESAMPLE_HZ,
                  thresh=MOVE_THRESH, winsor_p=WINSOR_P):
    """One row of movement-vigor metrics for one session.

    Every metric is a RATE or a DISTRIBUTIONAL summary, never a total, because
    sessions differ in length (first sessions run shorter than proficient ones) and
    a total would mostly measure how long the mouse sat there.
    """
    f = os.path.join(wp.SOURCE_DIR[source], f'design_matrix_{session}_{mouse_name}')
    d = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel'])
    b = d['Bin'].to_numpy(float)
    v = d['avg_wheel_vel'].to_numpy(float)
    ok = np.isfinite(b) & np.isfinite(v)
    b, v = b[ok], v[ok]
    native_hz = 1.0 / np.median(np.diff(b)) if len(b) > 1 else np.nan

    vr, width = _resample(b, v, hz)
    if not len(vr):
        return None
    speed = np.abs(vr)
    cap = np.percentile(speed, winsor_p)
    sp_w = np.minimum(speed, cap)
    moving = speed > thresh
    n_bout, bout_dur = _bouts(moving, width)
    dur_min = len(vr) * width / 60.0

    return dict(
        mouse_name=mouse_name, session=session, source=source,
        # --- vigor ---
        mean_speed=float(sp_w.mean()),
        # vigor PROPER: speed given that the animal is moving. mean_speed averages
        # over the still samples too, which is why it correlates 0.84 with
        # frac_moving and muddles "how often" with "how hard"; conditioning on
        # movement makes the two constructs orthogonal (rho 0.03) and raises
        # reliability from 0.54 to 0.62.
        mean_speed_moving=float(sp_w[moving].mean()) if moving.any() else np.nan,
        median_speed=float(np.median(speed)),
        p95_speed=float(np.percentile(speed, 95)),
        sd_speed=float(sp_w.std()),
        frac_moving=float(moving.mean()),
        bout_rate_per_min=float(n_bout / dur_min) if dur_min > 0 else np.nan,
        mean_bout_s=bout_dur,
        distance_per_min=float(sp_w.sum() * width / dur_min) if dur_min > 0 else np.nan,
        # --- direction (signed; the toggle's scalar analogue) ---
        mean_vel_signed=float(np.minimum(np.abs(vr), cap).mean() * np.sign(vr).mean()),
        turn_bias=float(np.sign(vr[speed > thresh]).mean()) if moving.any() else np.nan,
        # --- covariates, so confounds stay checkable ---
        duration_min=float(dur_min), n_samples=int(len(vr)),
        sampling_hz=float(native_hz),
    )


def build_vigor(table, **kw):
    """Metrics for every (mouse, session, source) row of `table`."""
    out, bad = [], []
    for _, r in table.iterrows():
        try:
            row = session_vigor(r['mouse_name'], r['session'], r['source'], **kw)
            if row is not None:
                out.append(row)
        except Exception as e:
            bad.append((r['mouse_name'], r['session'], r['source'], repr(e)[:80]))
    if bad:
        print(f'!! {len(bad)} sessions failed, e.g. {bad[0]}')
    return pd.DataFrame(out)


METRICS = ['mean_speed', 'mean_speed_moving', 'median_speed', 'p95_speed',
           'sd_speed', 'frac_moving',
           'bout_rate_per_min', 'mean_bout_s', 'distance_per_min',
           'mean_vel_signed', 'turn_bias']
