"""
Analysis 3: does the NM signal lead or follow spontaneous movement (onsets, and stops)?
Functions and the cache runner; the analysis and figures are in 4_nm_movement_lead_lag.ipynb.

    python neuromodulators/behavioural_states/movement_onsets.py [n_jobs]

Only the second before stimulus onset is used, the one stretch of the ±1 s snippets without task events (the
previous feedback is >= 1.5 s earlier; the stimulus response starts after 0). Within it, per trial and signal:
  onset   paw speed, whisker ME, wheel speed: log(1 + |x|), z-scored over the session's pre-stimulus samples;
          an onset is where z rises above OFF after >= QUIET_N samples below it and reaches ON within RISE_N.
          licks: a lick after >= QUIET_N samples without one.
Stops (offsets) of paw, whisking and wheel are taken the same way: before the stimulus the mouse often settles,
so stops are more common than starts. Events are kept if they fall in ONSET_RANGE, so the whole NM window WIN around them lies before the stimulus.

Onset-locked NM, corrected for where in the trial the onset happened: for an onset at time τ (relative to the
stimulus), the control is the NM at the same τ + lags averaged over the session's trials without an onset of that
signal in ONSET_RANGE. This removes the pre-stimulus ramp and anything else locked to the stimulus. Per fiber and
signal: mean corrected NM curve over onsets, the same for every vigor signal (what else moves at the onset), and n.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from Functions import nm_syllables as ns  # noqa: E402

OUT = ns.DATA / 'movement_onsets'
SIGNALS = {'paw': 'paw speed', 'whisking': 'whisker ME', 'wheel': 'wheel speed', 'lick': 'lick count'}
ON, OFF = 1.0, 0.0           # z thresholds: the movement reaches ON; quiet = below OFF
QUIET_N = 6                  # samples (200 ms) of quiet before an onset
RISE_N = 5                   # samples after the onset within which the signal must reach ON
ONSET_RANGE = (-0.7, -0.27)  # s relative to stimulus onset
WIN = (-9, 8)                # samples around the onset: -300 ms .. +233 ms (ends before the stimulus)
LAGS = np.arange(WIN[0], WIN[1] + 1) / ns.FS
MIN_ONSETS = 10


def events(x, t, kind, edge='on'):
    """(trial, sample) of the first event in each row of x (n_trials x T) within ONSET_RANGE.
    edge='on': the sample where the signal rises above OFF after >= QUIET_N samples below it and reaches ON within
    RISE_N samples (licks: a lick after >= QUIET_N samples without one).
    edge='off': the sample where it drops below OFF and stays there for QUIET_N samples, having been >= ON within
    the RISE_N samples before (the mouse stops moving)."""
    out = []
    lo, hi = np.searchsorted(t, ONSET_RANGE[0]), np.searchsorted(t, ONSET_RANGE[1])
    for i, row in enumerate(x):
        quiet = row < (0.5 if kind == 'lick' else OFF)
        for j in range(max(lo, QUIET_N, RISE_N), min(hi, len(row) - QUIET_N)):
            if edge == 'on':
                hit = (not quiet[j] and quiet[j - QUIET_N:j].all()
                       and (kind == 'lick' or (row[j:j + RISE_N] >= ON).any()))
            else:
                hit = quiet[j:j + QUIET_N].all() and not quiet[j - 1] and (row[j - RISE_N:j] >= ON).any()
            if hit:
                out.append((i, j))
                break
    return out


EVENTS = [(k, 'on') for k in SIGNALS] + [(k, 'off') for k in SIGNALS if k != 'lick']


def fiber_curves(fiber):
    """Per signal: corrected onset-locked NM and vigor curves (mean over onsets), and the number of onsets."""
    t, S = fiber['time'], fiber['stim']
    pre = t < 0
    P = S['photometry']
    V = {k: np.log1p(np.abs(S['movement'][c])) for k, c in SIGNALS.items() if k != 'lick'}
    Z = {}
    for k, x in V.items():
        mu, sd = np.nanmean(x[:, pre]), np.nanstd(x[:, pre])
        Z[k] = (x - mu) / (sd if sd > 0 else 1)
    Z['lick'] = S['movement']['lick count']
    trace = {'NM': P, **{f'v {k}': Z[k] for k in SIGNALS}}
    out = {}
    for k, edge in EVENTS:
        ev = [(i, j) for i, j in events(np.nan_to_num(Z[k], nan=OFF), t, k, edge)
              if j + WIN[0] >= 0 and j + WIN[1] < len(t) and np.isfinite(Z[k][i, j + WIN[0]:j + WIN[1] + 1]).all()]
        if len(ev) < MIN_ONSETS:
            continue
        has = np.zeros(len(P), bool)
        has[[i for i, _ in ev]] = True
        res = {'n': len(ev), 'n_control': int((~has).sum())}
        for name, A in trace.items():
            ctrl = np.nanmean(A[~has], 0)
            seg = np.array([A[i, j + WIN[0]:j + WIN[1] + 1] - ctrl[j + WIN[0]:j + WIN[1] + 1] for i, j in ev])
            res[name] = np.nanmean(seg, 0)
        out[f'{k} {edge}'] = res
    return out


def one_session(eid):
    warnings.filterwarnings('ignore')
    rows = []
    try:
        for f in ns.load_session(eid, mask=False):
            m = f['meta']
            for k, r in fiber_curves(f).items():
                rows.append({'eid': eid, 'subject': m['subject'].iat[0], 'region': f['region'],
                             'target_NM': m['target_NM'].iat[0], 'signal': k, **r})
    except Exception as ex:
        print(f'{eid}: {type(ex).__name__}: {ex}', flush=True)
    return rows


def run(n_jobs=12):
    from joblib import Parallel, delayed
    OUT.mkdir(exist_ok=True)
    eids = ns.load_recordings()['eid'].unique()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_session)(e) for e in eids)
    tab = pd.DataFrame([r for s in res for r in s])
    tab.to_pickle(OUT / f'onset_curves_{ns.RUN}.pkl')
    return tab


def load():
    return pd.read_pickle(OUT / f'onset_curves_{ns.RUN}.pkl')


def mouse_curves(tab, column='NM'):
    """Mean curve per (target_NM, signal, subject), onset-weighted over that mouse's fibers and sessions."""
    rows = []
    for k, g in tab.groupby(['target_NM', 'signal', 'subject']):
        w = g['n'].to_numpy(float)
        rows.append({'target_NM': k[0], 'signal': k[1], 'subject': k[2], 'n': int(w.sum()),
                     column: np.average(np.stack(g[column]), axis=0, weights=w)})
    return pd.DataFrame(rows)


def window_mean(curve, lo, hi):
    m = (LAGS >= lo) & (LAGS <= hi)
    return curve[..., m].mean(-1)


def half_rise(curve, base=(-0.3, -0.2), peak=(0.1, 0.233)):
    """Lag (s) at which the curve first crosses halfway from its baseline to its post-onset level (sign-aware);
    NaN if it never does or the change is ~0."""
    b, p = window_mean(curve, *base), window_mean(curve, *peak)
    if not np.isfinite(b - p) or abs(p - b) < 1e-9:
        return np.nan
    x = (curve - b) / (p - b)
    start = np.searchsorted(LAGS, base[1])
    idx = np.nonzero(x[start:] >= 0.5)[0]
    return LAGS[start + idx[0]] if len(idx) else np.nan


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
