"""
Sub-movement (primitive) decomposition of wheel movements -- used by 1_primitives.ipynb.

A movement is the stretch between two stillness periods, as detected by IBL
(alf/_ibl_wheelMoves.intervals, brainbox.behavior.wheel.movements). Its velocity is modelled as a
sum of N stereotyped primitives, each with a signed displacement D (rad):

  minimum-jerk (Flash & Hogan 1985): bell-shaped, symmetric, lasts exactly T
      v(t) = D/T * (30 tau^2 - 60 tau^3 + 30 tau^4),  tau = (t - t0)/T in [0, 1]
  lognormal (Plamondon's kinematic theory): skewed, fast rise and slow decay
      v(t) = D / (sigma sqrt(2 pi) (t - t0)) * exp(-(ln(t - t0) - mu)^2 / (2 sigma^2)),  t > t0

Each primitive's |D| is bounded by the movement's path length (sum |v| dt).

Fitting: greedy-then-joint. Start from zero primitives; repeatedly put a new primitive on the
largest lobe of the residual, then refit all parameters jointly (scipy least_squares, bounded),
until the fit explains R2_STOP of the velocity energy or N_MAX is reached. This is a simple
stand-in for Rohrer & Hogan's (2003) globally optimal decomposition: cheaper, but it can split one
primitive in two or miss overlapping ones.

The 200 Hz velocity is low-passed at 20 Hz, so primitives shorter than ~80 ms can't be resolved:
T_MIN sets that floor.
"""
import numpy as np
from scipy.optimize import least_squares

PAD_S = 0.05            # s of stillness kept either side of the movement
R2_STOP = 0.95          # stop adding primitives once this much velocity energy is explained
N_MAX = 15
T_MIN, T_MAX = 0.08, 1.5        # minimum-jerk duration bounds (s)
SIG_MIN, SIG_MAX = 0.1, 1.0     # lognormal sigma bounds


# ---------------------------------------------------------------------------------- shapes
def minjerk(t, t0, T, D):
    tau = np.clip((t - t0) / T, 0, 1)
    return D / T * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4)


def lognormal(t, t0, mu, sigma, D):
    dt = t - t0
    out = np.zeros_like(t)
    ok = dt > 1e-6
    out[ok] = D / (sigma * np.sqrt(2 * np.pi) * dt[ok]) * np.exp(
        -(np.log(dt[ok]) - mu) ** 2 / (2 * sigma ** 2))
    return out


KINDS = {
    'minjerk':   dict(f=minjerk, n=3),
    'lognormal': dict(f=lognormal, n=4),
}


def model(kind, t, p):
    f, n = KINDS[kind]['f'], KINDS[kind]['n']
    return sum(f(t, *p[i:i + n]) for i in range(0, len(p), n))


def components(kind, t, p):
    f, n = KINDS[kind]['f'], KINDS[kind]['n']
    return [f(t, *p[i:i + n]) for i in range(0, len(p), n)]


# ---------------------------------------------------------------------------------- fitting
def _lobe(t, r):
    """The largest lobe of the residual: (start, peak, end time, area)."""
    i = np.argmax(np.abs(r))
    s = np.sign(r[i])
    a = i
    while a > 0 and np.sign(r[a - 1]) == s:
        a -= 1
    b = i
    while b < len(r) - 1 and np.sign(r[b + 1]) == s:
        b += 1
    dt = t[1] - t[0]
    return t[a], t[i], t[b], r[a:b + 1].sum() * dt


def _init(kind, t, r):
    a, pk, b, area = _lobe(t, r)
    width = np.clip(b - a, T_MIN, T_MAX)
    if kind == 'minjerk':
        return [pk - width / 2, width, area]
    sigma = 0.3
    t0 = a - 0.02
    mu = np.log(max(pk - t0, 0.01)) + sigma ** 2      # puts the mode at the residual peak
    return [t0, mu, sigma, area]


def _bounds(kind, t, n_prim, d_max):
    lo_t, hi_t = t[0] - 0.5, t[-1]
    if kind == 'minjerk':
        lo, hi = [lo_t, T_MIN, -d_max], [hi_t, T_MAX, d_max]
    else:
        lo, hi = [lo_t, np.log(0.01), SIG_MIN, -d_max], [hi_t, np.log(2.0), SIG_MAX, d_max]
    return lo * n_prim, hi * n_prim


def fit_movement(kind, t, v, r2_stop=R2_STOP, n_max=N_MAX):
    """Greedy-then-joint fit. Returns params (flat), the R2 after each added primitive."""
    n = KINDS[kind]['n']
    energy = np.sum(v ** 2)
    # no primitive may move further than the whole movement's path length; without this, pairs of
    # huge opposite primitives cancel each other (the spurious decompositions Rohrer & Hogan warn of)
    d_max = np.sum(np.abs(v)) * (t[1] - t[0])
    p, r2s = [], []
    for k in range(1, n_max + 1):
        resid = v - (model(kind, t, np.array(p)) if p else 0)
        p0 = np.array(p + _init(kind, t, resid))
        lo, hi = _bounds(kind, t, k, d_max)
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        sol = least_squares(lambda q: model(kind, t, q) - v, p0, bounds=(lo, hi), max_nfev=2000)
        p = list(sol.x)
        r2s.append(1 - np.sum(sol.fun ** 2) / energy)
        if r2s[-1] >= r2_stop:
            break
    return np.array(p), np.array(r2s)


def movement_window(times, velocity, onset, offset, pad=PAD_S):
    i0, i1 = np.searchsorted(times, [onset - pad, offset + pad])
    return times[i0:i1] - onset, velocity[i0:i1]


def primitive_table(kind, p):
    """One row per primitive: start, peak time, duration (5-95 % of displacement), D."""
    n = KINDS[kind]['n']
    rows = []
    for i in range(0, len(p), n):
        if kind == 'minjerk':
            t0, T, D = p[i:i + n]
            start, peak, dur = t0, t0 + T / 2, T
        else:
            t0, mu, sigma, D = p[i:i + n]
            q05, q95 = np.exp(mu - 1.645 * sigma), np.exp(mu + 1.645 * sigma)
            start, peak, dur = t0 + q05, t0 + np.exp(mu - sigma ** 2), q95 - q05
        rows.append(dict(start=start, peak=peak, duration=dur, D=D))
    return rows


# ---------------------------------------------------------------------------------- sessions
EPOCHS = ['pre-stim', 'response', 'post-feedback']


def stim_times(trials):
    """stimOn_times, filled with goCue_times where missing (some sessions have no stimOn_times at
    all; in training the go cue is played at stimulus onset)."""
    return trials.stimOn_times.fillna(trials.goCue_times)


def epoch_of(on, trials):
    """(epoch, trial index) of a time. pre-stim: trial start -> stimulus (quiescence, where moving
    restarts the timer); response: stimulus -> feedback; post-feedback: feedback -> next trial."""
    i = np.searchsorted(trials.intervals_0.values, on, side='right') - 1
    if i < 0:
        return 'post-feedback', -1
    r = trials.iloc[i]
    if on < stim_times(trials).iloc[i]:
        return 'pre-stim', i
    if on <= r.feedback_times:
        return 'response', i
    return 'post-feedback', i


def session_primitives(times, velocity, trials, intervals, kind='minjerk'):
    """Decompose every movement of a session. Returns (movements, primitives) tables; primitive
    times are absolute (s)."""
    import pandas as pd
    mv = pd.DataFrame(intervals, columns=['onset', 'offset'])
    ep = [epoch_of(o, trials) for o in mv.onset]
    mv['epoch'] = [e for e, _ in ep]
    mv['trial'] = np.array([i for _, i in ep], int)
    rows, n, r2 = [], [], []
    for m, (on, off) in enumerate(zip(mv.onset, mv.offset)):
        tt, vv = movement_window(times, velocity, on, off)
        p, r2s = fit_movement(kind, tt, vv)
        n.append(len(r2s)); r2.append(r2s[-1])
        for j, row in enumerate(sorted(primitive_table(kind, p), key=lambda x: x['start'])):
            row.update(movement=m, order=j, start=row['start'] + on, peak=row['peak'] + on)
            rows.append(row)
    mv['n_primitives'], mv['r2'] = n, r2
    pr_ = pd.DataFrame(rows)
    pr_ = pr_.merge(mv[['epoch', 'trial']], left_on='movement', right_index=True)
    return mv, pr_
