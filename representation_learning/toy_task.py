"""
Toy version of the IBL training task, and three actor-critic agents that differ only in
their state representation.
=========================================================================================
This is a sketch for thinking about the modelling project, NOT the model we would fit:
one contrast (100%), fixed 3-bin movement bouts.

Two response modes
  wheel=False   the first bout during the stimulus is the response, registered at onset.
  wheel=True    each bout moves the stimulus by a random displacement (amplitude, plus an
                optional reversal that swings back). The response is the first crossing
                of +-1 (threshold units): towards the side of the stimulus is correct, the
                other way is an error. Action labels are defined so that "left" is correct
                for a left stimulus, as in wheel=False. The stimulus position is visible to
                M0 and M1 (5 bins between -1 and +1).

Task, in 100 ms bins
  ITI           10 bins after feedback. Moving here has no consequence.
  quiescence    the mouse must stay still for 4-7 bins (drawn per trial); a movement
                restarts the count.
  stimulus      left or right. Correct response -> reward 1, error -> 0. No response
                within the response window -> timeout (0).
  feedback      the next ITI starts at the response.

Actions, at every bin where the agent is still
  wait | start a LEFT bout | start a RIGHT bout
  A bout lasts 3 bins and the agent is committed for its duration (no decisions).

What the agent can observe, x = (stim, tf, tm)
  stim   0 none / 1 left / 2 right
  tf     bins since the last feedback (capped)
  tm     bins since the last bout ended (capped)

Models (the learning rule is identical; only the features change)
  M0  one feature per stim value: all no-stimulus time is ONE state.
  M1  known structure: the stimulus feature, or, with no stimulus, two clocks coded as
      tapped delay lines (one feature per tf value + one per tm value).
  M2  state discovery: a tree that starts as a single state for everything and splits a
      leaf when the TD-error advantage of one action over another differs reliably across
      a cut of stim, tf or tm.

Learning rule: actor-critic on decision points (semi-Markov, a bout spans k = 3 bins),
with eligibility traces (trace_lambda = 0 is plain TD(0))
  e_V, e_H  *= gamma**k_prev * trace_lambda       (older tags fade)
  e_V(s)    += 1;  e_H(s)[a'] += 1[a'=a] - pi(a') (tag the current state and action)
  delta      = R + gamma**k * V(s') - V(s)
  V         += alpha * delta * e_V                (critic)
  H         += beta * delta * e_H                 (actor, softmax policy gradient)
  H(s)      -= lam * H(s)                         (preferences relax to the motor baseline)
  pi         = softmax(H / tau - [0, bias, bias]) (bias sets the baseline movement rate)

@author: Ines (sketch drafted with Claude)
"""
import numpy as np

ITI, QMIN, QMAX, BOUT, STIM_TO = 10, 4, 7, 3, 40
POS_EDGES = np.array([-0.6, -0.2, 0.2, 0.6])   # stimulus-position bins seen by M0/M1
MOTOR = {   # mean bout amplitude (threshold units), reversal probability
    'expert': dict(amp=1.2, rev=0.0),
    'naive': dict(amp=0.35, rev=0.35),
}
TF_CAP, TM_CAP = 20, 8
GAMMA = 0.98
DT = 0.1                      # seconds per bin
FEATURE_NAMES = ['stimulus', 'time since feedback', 'time since movement']
PHASES = {'iti': 0, 'q': 1, 'stim': 2}


class StateTree:
    """M2's state partition. Leaves are states; a leaf splits on one feature threshold."""

    def __init__(self, tcrit, nmin=15):
        self.nodes = [dict(leaf=True, stats={})]
        self.tcrit, self.nmin, self.log = tcrit, nmin, []

    def leaf(self, x):
        i = 0
        while not self.nodes[i]['leaf']:
            n = self.nodes[i]
            i = n['lo'] if x[n['f']] <= n['thr'] else n['hi']
        return i

    def record(self, i, a, x, delta):
        st = self.nodes[i]['stats']
        for f in range(3):
            s = st.setdefault((a, f, x[f]), [0, 0., 0.])
            s[0] += 1; s[1] += delta; s[2] += delta * delta

    def try_split(self, trial, V, H):
        """For each leaf, find the feature cut across which the advantage (mean TD error of
        one action minus another) differs most, in t units. Split if it exceeds tcrit.
        Statistics are reset after every check (a sliding window of experience)."""
        for i in range(len(self.nodes)):
            n = self.nodes[i]
            if not n['leaf']:
                continue
            best = None
            for f in range(3):
                vals = sorted({k[2] for k in n['stats'] if k[1] == f})
                for thr in vals[:-1]:
                    cell = {}
                    for a in range(3):
                        lo, hi = np.zeros(3), np.zeros(3)
                        for v in vals:
                            s = n['stats'].get((a, f, v))
                            if s:
                                (lo if v <= thr else hi)[:] += s
                        cell[a] = (lo, hi)
                    for a, b in [(1, 0), (2, 0), (1, 2)]:
                        parts = []
                        for side in (0, 1):
                            A, B = cell[a][side], cell[b][side]
                            if A[0] < self.nmin or B[0] < self.nmin:
                                break
                            ma, mb = A[1] / A[0], B[1] / B[0]
                            va = max(A[2] / A[0] - ma ** 2, 1e-4)
                            vb = max(B[2] / B[0] - mb ** 2, 1e-4)
                            parts.append((ma - mb, va / A[0] + vb / B[0]))
                        if len(parts) < 2:
                            continue
                        t = abs(parts[0][0] - parts[1][0]) / np.sqrt(parts[0][1] + parts[1][1])
                        if best is None or t > best[0]:
                            best = (t, f, thr)
            n['stats'] = {}
            if best and best[0] > self.tcrit:
                _, f, thr = best
                lo, hi = len(self.nodes), len(self.nodes) + 1
                self.nodes += [dict(leaf=True, stats={}), dict(leaf=True, stats={})]
                n.update(leaf=False, f=f, thr=thr, lo=lo, hi=hi)
                V[lo] = V[hi] = V.get(i, 0.)
                H[lo] = H.get(i, np.zeros(3)).copy()
                H[hi] = H.get(i, np.zeros(3)).copy()
                self.log.append(dict(trial=trial, feature=FEATURE_NAMES[f], threshold=thr,
                                     n_states=sum(nd['leaf'] for nd in self.nodes)))


def run(model, n_trials=2000, alpha=0.05, beta=0.05, tau=0.1, bias=2.2, tcrit=5, seed=0,
        lam=0.002, split_every=3000, block=50, record=None,
        wheel=False, motor='expert', motor_bouts=3000, trace_lambda=0., resp_window=None,
        naive_amp=None, naive_rev=None):
    """
    Simulate one agent. Returns (blocks, splits, trace).

    wheel        : response by threshold crossing (see module docstring).
    motor        : 'expert', 'naive' or 'improving' (naive -> expert as bouts accumulate,
                   1 - exp(-n_bouts / motor_bouts)).
    trace_lambda : eligibility-trace decay; 0 = TD(0).
    resp_window  : bins before a timeout (default 100 with the wheel, 40 without).
    naive_amp, naive_rev : override the naive motor's mean amplitude and reversal probability.
    blocks       : one row per `block` trials: accuracy (timeouts count as not correct),
                   movement onsets per still bin in the ITI and in quiescence, trials per minute,
                   bouts per response, reaction time, fraction of errors caused by a reversal,
                   fraction of timeouts.
    splits       : M2 only, list of split events.
    trace        : per-bin record for trials in `record` = (first, last) trial, else None.
                   Columns: trial, phase (0 ITI, 1 quiescence, 2 stimulus), action (0 wait,
                   1 left onset, 2 right onset, -1 inside a bout), side, reward, stimulus position.
    """
    rng = np.random.default_rng(int(seed))
    V, H, eV, eH = {}, {}, {}, {}
    tree = StateTree(tcrit) if model == 'M2' else None
    resp_window = resp_window or (100 if wheel else STIM_TO)

    def feats(x, xb):
        stim, tf, tm = x
        if model == 'M0':
            return [('s', stim, xb)]
        if model == 'M1':
            return [('s', stim, xb)] if stim else [('tf', tf), ('tm', tm)]
        return [tree.leaf(x)]

    def val(F):
        return sum(V.get(f, 0.) for f in F)

    def pref(F):
        return sum((H.get(f, np.zeros(3)) for f in F), np.zeros(3))

    naive = dict(amp=naive_amp if naive_amp is not None else MOTOR['naive']['amp'],
                 rev=naive_rev if naive_rev is not None else MOTOR['naive']['rev'])

    def bout(n_bouts):
        if motor == 'improving':
            w = 1 - np.exp(-n_bouts / motor_bouts)
            mu = (1 - w) * naive['amp'] + w * MOTOR['expert']['amp']
            pr = (1 - w) * naive['rev'] + w * MOTOR['expert']['rev']
        else:
            mu, pr = (naive['amp'], naive['rev']) if motor == 'naive' else (MOTOR[motor]['amp'], MOTOR[motor]['rev'])
        amp = mu * rng.lognormal(-0.125, 0.5)          # mean-preserving lognormal noise
        back = amp * rng.uniform(0.5, 1.5) if rng.random() < pr else 0.
        return amp, back

    tf, tm, still = 0, TM_CAP, 0
    phase, qreq, side, stim_t, xpos = 'iti', 0, 0, 0, 0.
    trial, t, n_bouts, k_prev = 0, 0, 0, 1
    blocks, trace = [], []
    new_cur = lambda: dict(c=0, n=0, mi=0, si=0, mq=0, sq=0, bins=0, bouts=0, rt=0, err=0, rev_err=0, to=0)
    cur, resp_bouts = new_cur(), 0
    prev = None
    while trial < n_trials and t < 400 * n_trials:
        stim = side if phase == 'stim' else 0
        x = (stim, min(tf, TF_CAP), min(tm, TM_CAP))
        xb = int(np.searchsorted(POS_EDGES, xpos)) if (stim and wheel) else 0
        F = feats(x, xb)
        if prev is not None:
            pF, pa, pR, pk, px, pp = prev
            delta = pR + GAMMA ** pk * val(F) - val(pF)
            decay = GAMMA ** k_prev * trace_lambda
            for d in (eV, eH):
                for f in list(d):
                    d[f] = d[f] * decay
                    if np.max(np.abs(d[f])) < 1e-2:
                        del d[f]
            onehot = np.zeros(3); onehot[pa] = 1
            for f in pF:
                eV[f] = eV.get(f, 0.) + 1. / len(pF)
                eH[f] = eH.get(f, np.zeros(3)) + (onehot - pp) / len(pF)
            for f, e in eV.items():
                V[f] = V.get(f, 0.) + alpha * delta * e
            for f, e in eH.items():
                H[f] = H.get(f, np.zeros(3)) + beta * delta * e
            for f in pF:
                H[f] -= lam * H[f]
            k_prev = pk
            if tree is not None:
                tree.record(pF[0], pa, px, delta)
        logits = pref(F) / tau - np.array([0, bias, bias])
        p = np.exp(logits - logits.max()); p /= p.sum()
        a = rng.choice(3, p=p)
        if phase == 'iti':
            cur['si'] += 1; cur['mi'] += a > 0
        elif phase == 'q':
            cur['sq'] += 1; cur['mq'] += a > 0
        rec = record is not None and record[0] <= trial <= record[1]
        R, k, ph0, tr0, side0, x0 = 0., 1, PHASES[phase], trial, side, xpos
        outcome = None
        if a == 0:
            t += 1; tf += 1; tm += 1; cur['bins'] += 1
            if phase == 'iti' and tf >= ITI:
                phase, still, qreq = 'q', 0, rng.integers(QMIN, QMAX + 1)
            elif phase == 'q':
                still += 1
                if still >= qreq:
                    phase, side, stim_t, xpos, resp_bouts = 'stim', rng.integers(1, 3), 0, 0., 0
            elif phase == 'stim':
                stim_t += 1
                if stim_t >= resp_window:
                    outcome = 'timeout'
        else:
            k = BOUT
            n_bouts += 1
            if phase == 'stim':
                resp_bouts += 1
                stim_t += BOUT
                if not wheel:
                    outcome = 'correct' if a == side else 'error'
                else:
                    sgn = -1 if a == 1 else 1           # left bouts move the stimulus to -x
                    target = -1 if side == 1 else 1     # correct crossing for this stimulus
                    amp, back = bout(n_bouts)
                    path = [xpos + sgn * amp] + ([xpos + sgn * (amp - back)] if back else [])
                    for j, xp in enumerate(path):
                        if xp * target >= 1:
                            outcome = 'correct'
                        elif xp * target <= -1:
                            outcome = 'error'
                            if j == 1 and sgn == target:
                                cur['rev_err'] += 1     # right direction, undone by the swing back
                        if outcome:
                            break
                    xpos = path[-1]
                    if not outcome and stim_t >= resp_window:
                        outcome = 'timeout'
            elif phase == 'q':
                still = 0
            t += BOUT; tf += BOUT; cur['bins'] += BOUT; tm = 0
            if phase == 'iti' and tf >= ITI:
                phase, still, qreq = 'q', 0, rng.integers(QMIN, QMAX + 1)
        if outcome:
            R = 1. if outcome == 'correct' else 0.
            trial += 1; cur['n'] += 1; cur['c'] += outcome == 'correct'
            cur['err'] += outcome == 'error'; cur['to'] += outcome == 'timeout'
            cur['bouts'] += resp_bouts; cur['rt'] += stim_t
            phase, tf, xpos = 'iti', 0 if a == 0 else BOUT, 0.
            if a > 0 and tf >= ITI:
                phase = 'q'
        if rec:
            trace.append((tr0, ph0, a, side0 if ph0 == 2 else 0, R, x0))
            for _ in range(k - 1):
                trace.append((tr0, ph0 if not outcome else 0, -1, 0, 0., xpos))
        prev = (F, a, R, k, x, p)
        if tree is not None and t // split_every != (t - k) // split_every:
            tree.try_split(trial, V, H)
        if cur['n'] >= block:
            blocks.append(dict(model=model, seed=seed, trial=trial, accuracy=cur['c'] / cur['n'],
                               move_iti=cur['mi'] / max(cur['si'], 1),
                               move_quiescence=cur['mq'] / max(cur['sq'], 1),
                               trials_per_min=cur['n'] / (cur['bins'] * DT / 60),
                               bouts_per_response=cur['bouts'] / cur['n'],
                               rt_s=cur['rt'] * DT / cur['n'],
                               reversal_error_frac=cur['rev_err'] / max(cur['err'], 1),
                               timeout_frac=cur['to'] / cur['n']))
            cur = new_cur()
    return blocks, (tree.log if tree else None), (np.array(trace) if record else None)


def trials_to_criterion(acc, trials, thr=0.8, n_blocks=2):
    """First trial count at which accuracy stays >= thr for n_blocks consecutive blocks."""
    run_len = 0
    for i, a in enumerate(acc):
        run_len = run_len + 1 if a >= thr else 0
        if run_len >= n_blocks:
            return trials[i - n_blocks + 1]
    return np.nan
