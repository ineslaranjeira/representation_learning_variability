"""
Does a mouse learn WHERE in the trial it must keep still? Per-session wheel measures.
=====================================================================================
Used by 2_epoch_structure_wheel.ipynb.

For every training session with a wheel design matrix (`avg_wheel_vel`, ~30 Hz), movement
ONSETS are detected (speed > MOVE_THRESH for at least MIN_BOUT_S, the thresholds of
paper-individuality/vigor/wheel/wheel_vigor.py; dips shorter than MIN_GAP_S are bridged) and
counted in three windows per trial:

  still    trial start (intervals_0) -> go cue. The quiescence timer restarts on any
           movement, so every onset here is a rule violation.
  free     feedback -> next trial start (trial end for the last trial). Moving here has
           no consequence.
  response go cue -> first movement (reaction time) -> feedback.

Onsets rather than time-moving, so the tail of a response movement that runs into the free
window is not counted as free movement. Rates are per second, because the windows differ
in length -- and the still window is set by the mouse itself (it ends only once the mouse
has been still long enough), so a mouse that violates more gets longer windows.

    selectivity = log((n_free + 0.5) / T_free) - log((n_still + 0.5) / T_still)

0 = the mouse starts movements equally often whether or not stillness is required;
> 0 = it keeps still selectively where the task demands it. A mouse that just calms down
everywhere lowers both rates and leaves the index unchanged.
"""
import os
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PI = REPO / 'paper-individuality'
TRIALS_DIR = PI / 'data' / 'training' / 'training_data'
DM_DIR = PI / 'data' / 'training' / 'training_design_matrices'
TRAINING_TIME = REPO / 'learning_prediction' / 'training_time_05-05-2026'
CACHE = HERE / 'epoch_structure_sessions.parquet'

MOVE_THRESH = 0.1       # rad/s
MIN_BOUT_S = 0.1        # s
MIN_GAP_S = 0.1         # s; a dip below threshold shorter than this does not end a bout
FLAT_MAX = 0.01        # sessions with more interpolated (flat) movement than this are flagged
EASY = 0.5              # contrast >= EASY counts as an easy trial


def mouse_trials(mouse):
    """All training trials of one mouse, sessions in date order, trials in time order."""
    t = pd.read_parquet(TRIALS_DIR / f'training_data_trials_{mouse}')
    t['session_date'] = pd.to_datetime(t['session_date'])
    order = t.groupby('session')['session_date'].min().sort_values()
    t['session_index'] = t['session'].map({s: i + 1 for i, s in enumerate(order.index)})
    return t.sort_values(['session_index', 'intervals_0']).reset_index(drop=True)


def wheel_onsets(b, v, thresh=MOVE_THRESH, min_bout_s=MIN_BOUT_S, min_gap_s=MIN_GAP_S):
    """Onset times of movement bouts, plus the 0/1 moving trace (for time-moving).
    Gaps below threshold shorter than min_gap_s are filled first, so one wobbly movement
    is one bout, not several."""
    moving = np.abs(v) > thresh
    dt = np.median(np.diff(b))
    gap_n = int(np.ceil(min_gap_s / dt))
    d = np.diff(np.r_[0, moving.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for e, s_next in zip(ends[:-1], starts[1:]):           # fill short still gaps between bouts
        if s_next - e < gap_n:
            moving[e:s_next] = True
    min_n = max(1, int(np.ceil(min_bout_s / dt)))
    d = np.diff(np.r_[0, moving.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_n
    return b[starts[keep]], moving, dt


def flat_share(v, thresh=MOVE_THRESH, tol=1e-3, min_s=0.3, dt=1 / 30):
    """Share of above-threshold samples that sit in near-constant runs (|dv| < tol for >= min_s):
    velocity interpolated across a gap in the wheel data, not real movement. QC only."""
    v = np.asarray(v, float)
    small = (np.abs(np.diff(v)) < tol) & (np.abs(v[1:]) > thresh)
    d = np.diff(np.r_[0, small.astype(int), 0])
    L = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    n_mov = np.sum(np.abs(v) > thresh)
    return float(L[L >= int(min_s / dt)].sum() / n_mov) if n_mov else 0.0


def count_in(times, lo, hi):
    """Number of `times` in each window (lo, hi]."""
    return np.searchsorted(times, hi, 'right') - np.searchsorted(times, lo, 'right')


def time_moving_in(b, moving, dt, lo, hi):
    """Seconds spent moving in each window (lo, hi], from the sampled moving trace."""
    cum = np.r_[0, np.cumsum(moving)]
    return (cum[np.searchsorted(b, hi, 'right')] - cum[np.searchsorted(b, lo, 'right')]) * dt


def trial_windows(tt):
    """Per-trial window bounds for one session's trial table (time-ordered)."""
    i0 = tt['intervals_0'].to_numpy(float)
    go = tt['goCue_times'].to_numpy(float)
    fb = tt['feedback_times'].to_numpy(float)
    nxt = np.r_[i0[1:], tt['intervals_1'].to_numpy(float)[-1]]
    return dict(still=(i0, go), free=(fb, nxt))


def session_measures(mouse, session, tt):
    """One row of measures for one session; None if the session has no wheel data."""
    f = DM_DIR / f'design_matrix_{session}_{mouse}'
    if not f.exists():
        return None
    dm = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel']).dropna().sort_values('Bin')
    b, v = dm['Bin'].to_numpy(float), dm['avg_wheel_vel'].to_numpy(float)
    onsets, moving, dt = wheel_onsets(b, v)
    w = trial_windows(tt)
    flat = flat_share(v)

    out = dict(mouse_name=mouse, session=session,
               session_index=int(tt['session_index'].iloc[0]),
               session_date=tt['session_date'].iloc[0], n_trials=len(tt),
               duration_min=(tt['intervals_1'].max() - tt['intervals_0'].min()) / 60,
               sampling_hz=1 / dt, flat_share=flat)
    for name, (lo, hi) in w.items():
        ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        n = count_in(onsets, lo[ok], hi[ok])
        T = (hi - lo)[ok]
        out[f'{name}_onsets'] = int(n.sum())
        out[f'{name}_time_s'] = float(T.sum())
        out[f'{name}_rate'] = n.sum() / T.sum() if T.sum() > 0 else np.nan
        out[f'{name}_window_median_s'] = float(np.median(T)) if len(T) else np.nan
        out[f'{name}_frac_moving'] = float(time_moving_in(b, moving, dt, lo[ok], hi[ok]).sum() / T.sum()) \
            if T.sum() > 0 else np.nan
        if name == 'still':
            out['p_trial_violation'] = float(np.mean(n > 0)) if len(n) else np.nan
    out['selectivity'] = (np.log((out['free_onsets'] + 0.5) / out['free_time_s'])
                          - np.log((out['still_onsets'] + 0.5) / out['still_time_s'])
                          if out['free_time_s'] > 0 and out['still_time_s'] > 0 else np.nan)
    elong = tt['goCue_times'] - tt['quiescence'] - tt['intervals_0']
    out['elongation_median_s'] = float(np.nanmedian(elong))

    # task performance
    contrast = tt['contrastLeft'].fillna(0) + tt['contrastRight'].fillna(0)
    easy = contrast >= EASY
    out['accuracy_easy'] = float((tt.loc[easy, 'feedbackType'] == 1).mean()) if easy.any() else np.nan
    rt = tt['firstMovement_times'] - tt['goCue_times']
    out['rt_median_s'] = float(np.nanmedian(rt))
    out['trials_per_min'] = len(tt) / out['duration_min'] if out['duration_min'] > 0 else np.nan
    out['n_contrasts'] = int(np.unique(contrast.round(4)).size)
    return out


def build(refresh=False, verbose=True):
    """Per-session table for every training session with wheel data (cached)."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    rows = []
    for f in sorted(os.listdir(TRIALS_DIR)):
        mouse = f.replace('training_data_trials_', '')
        t = mouse_trials(mouse)
        n_sess = t['session'].nunique()
        for session, tt in t.groupby('session', sort=False):
            r = session_measures(mouse, session, tt.reset_index(drop=True))
            if r is not None:
                r['n_sessions_total'] = n_sess
                rows.append(r)
    S = pd.DataFrame(rows).sort_values(['mouse_name', 'session_index']).reset_index(drop=True)
    have = S.groupby('mouse_name')['session'].size()
    S['complete'] = S['mouse_name'].map(have) == S['n_sessions_total']
    S['qc_flat'] = S['flat_share'] > FLAT_MAX
    S['training_fraction'] = np.where(S['n_sessions_total'] > 1,
                                      (S['session_index'] - 1) / (S['n_sessions_total'] - 1).clip(lower=1), 0.0)
    S.to_parquet(CACHE)
    if verbose:
        print(f'{len(S)} sessions, {S.mouse_name.nunique()} mice, '
              f'{S.loc[S.complete, "mouse_name"].nunique()} with every training session')
    return S


def sessions_to_third_contrast(verbose=True):
    """Ordinal of the first training session that uses >= 3 absolute contrasts (0.25 added to
    1.0 / 0.5 once the stage criterion is met). Same definition as
    vigor/wheel/wheel_preprocessing.sessions_to_n_levels(3, persist=1).
    NaN for LEFT-CENSORED mice (session 1 already has >= 3 contrasts: recording started after
    the start of training, so the value would only be a lower bound) and for mice that never
    reach it."""
    out, censored, never = {}, [], []
    for f in sorted(os.listdir(TRIALS_DIR)):
        mouse = f.replace('training_data_trials_', '')
        t = mouse_trials(mouse)
        c = (t['contrastLeft'].fillna(0) + t['contrastRight'].fillna(0)).abs()
        nl = c[c > 0].round(4).groupby(t.loc[c > 0, 'session_index']).nunique().sort_index()
        hit = nl.index[nl >= 3]
        if not len(hit):
            never.append(mouse); out[mouse] = np.nan
        elif hit[0] == nl.index.min():
            censored.append(mouse); out[mouse] = np.nan
        else:
            out[mouse] = float(hit[0])
    if verbose:
        print(f'third contrast: {len(censored)} left-censored {censored}, {len(never)} never reach it {never}')
    return pd.Series(out, name='sessions_to_3c')


def learning_speed(definition='proficiency'):
    """Per-mouse log learning time; HIGHER = SLOWER.
    'proficiency'     log(training sessions before proficiency), as lab_controls.py
                      (drops log == 0 or 1, i.e. PL034).
    'third_contrast'  log(sessions until the third contrast is added), censored mice dropped."""
    if definition == 'proficiency':
        tt = pd.read_parquet(TRAINING_TIME).set_index('mouse_name')
        ls = np.log(tt['training_days'])
        return ls[~ls.isin([0, 1])].rename('log_train')
    if definition == 'third_contrast':
        return np.log(sessions_to_third_contrast()).dropna().rename('log_train')
    raise ValueError(f"definition must be 'proficiency' or 'third_contrast', got {definition!r}")


# ------------------------------------------------------------------------------------------
# Movement-onset rate as a function of time since the previous feedback
# ------------------------------------------------------------------------------------------
# The mouse is not told when a trial starts: the stillness window begins silently ~0.65 s after
# the previous trial ends. The only cue it has for "the stimulus is coming" is elapsed time since
# feedback, so this asks how onset rate depends on that time, pooled over the trials of a session.
HAZ_EDGES = np.arange(0, 6.01, 0.1)


def hazard_since_feedback(mouse, session, tt, edges=HAZ_EDGES):
    """For trial t, onsets between feedback(t-1) and goCue(t), binned by time since feedback(t-1),
    separately after a correct and after an error trial (the ITI is ~1 s longer after errors).
    Returns, per previous outcome, onset counts and exposure (trial-seconds at risk) per bin and
    the median trial-start and go-cue delays (time since feedback(t-1)) for marking the plot."""
    f = DM_DIR / f'design_matrix_{session}_{mouse}'
    if not f.exists():
        return None
    dm = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel']).dropna().sort_values('Bin')
    onsets, _, _ = wheel_onsets(dm['Bin'].to_numpy(float), dm['avg_wheel_vel'].to_numpy(float))
    fb_prev = tt['feedback_times'].to_numpy(float)[:-1]
    prev_ok = tt['feedbackType'].to_numpy(float)[:-1] == 1
    go = tt['goCue_times'].to_numpy(float)[1:]
    start = tt['intervals_0'].to_numpy(float)[1:]
    ok = np.isfinite(fb_prev) & np.isfinite(go) & (go > fb_prev)
    out = {}
    for name, sel in [('correct', ok & prev_ok), ('error', ok & ~prev_ok)]:
        a_, z_, s_ = fb_prev[sel], go[sel], start[sel]
        counts = np.zeros(len(edges) - 1)
        for a, z in zip(a_, z_):
            i, j = np.searchsorted(onsets, a, 'right'), np.searchsorted(onsets, z, 'right')
            counts += np.histogram(onsets[i:j] - a, edges)[0]
        span = z_ - a_                                    # each trial is at risk until its go cue
        exposure = np.clip(span[:, None] - edges[None, :-1], 0, np.diff(edges)[None, :]).sum(0)
        out[name] = dict(counts=counts, exposure=exposure,
                         start_delay=np.median(s_ - a_) if len(a_) else np.nan,
                         go_delay=np.median(z_ - a_) if len(a_) else np.nan)
    return out


def build_hazard(S, refresh=False, cache=HERE / 'epoch_structure_hazard.parquet'):
    """One row per (session, previous outcome): onset counts and exposure per time-since-feedback bin."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = []
    for mouse, g in S.groupby('mouse_name'):
        t = mouse_trials(mouse)
        for session in g['session']:
            tt = t[t['session'] == session].reset_index(drop=True)
            h = hazard_since_feedback(mouse, session, tt)
            if h is None:
                continue
            for prev, d in h.items():
                rows.append(dict(mouse_name=mouse, session=session, prev=prev,
                                 start_delay=d['start_delay'], go_delay=d['go_delay'],
                                 **{f'n{k}': c for k, c in enumerate(d['counts'])},
                                 **{f'e{k}': e for k, e in enumerate(d['exposure'])}))
    H = pd.DataFrame(rows)
    H.to_parquet(cache)
    return H


# ------------------------------------------------------------------------------------------
# Time-since-feedback summaries, one row per (session, previous outcome)
# ------------------------------------------------------------------------------------------
# Windows are defined per trial from its own previous feedback and its own (silent) trial
# start, and all end before the earliest possible go cue (trial start + minimum quiescence
# + rig delay >= ~0.47 s), so no window is shaped by the go-cue selection: a go cue only
# arrives after stillness, so near it the trials still at risk are enriched for movers.
#
#   reward   feedback -> +0.5 s                 response tail / reward consumption
#   iti      feedback + 0.5 s -> start - PRE_S  free, well before the silent trial start
#   pre      start - PRE_S -> start             free, the last moment before the rule applies
#   post     start -> start + POST_S            stillness required, go cue not yet possible
#   free_matched  start - 1.2 s -> start - 0.4 s  free; SAME length and position relative to the
#                 trial start after correct and after error, unlike `iti`, whose length follows the
#                 outcome-dependent ITI (~1 s longer after errors)
#
#   boundary = log rate(post) - log rate(pre)   (+0.5 onset smoothing, as `selectivity`)
# 0 = the mouse does not distinguish the moment the silent trial start passes; < 0 = it keeps
# stiller once stillness becomes required, which it can only do from elapsed time since feedback.
PRE_S, POST_S, REWARD_S = 0.4, 0.4, 0.5
MATCHED = (-1.2, -0.4)      # 'free_matched': same-length free window, relative to trial start, both outcomes
FB_WINDOWS = ('reward', 'iti', 'free_matched', 'pre', 'post')


def feedback_summaries(mouse, session, tt):
    f = DM_DIR / f'design_matrix_{session}_{mouse}'
    if not f.exists():
        return None
    dm = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel']).dropna().sort_values('Bin')
    onsets, _, _ = wheel_onsets(dm['Bin'].to_numpy(float), dm['avg_wheel_vel'].to_numpy(float))
    fb = tt['feedback_times'].to_numpy(float)[:-1]
    prev_ok = tt['feedbackType'].to_numpy(float)[:-1] == 1
    start = tt['intervals_0'].to_numpy(float)[1:]
    go = tt['goCue_times'].to_numpy(float)[1:]
    ok = np.isfinite(fb) & np.isfinite(start) & np.isfinite(go) & (start + MATCHED[0] > fb + REWARD_S) \
        & (go >= start + POST_S)
    win = dict(reward=(fb, fb + REWARD_S), iti=(fb + REWARD_S, start - PRE_S),
               free_matched=(start + MATCHED[0], start + MATCHED[1]),
               pre=(start - PRE_S, start), post=(start, start + POST_S))
    rows = []
    for prev, sel in [('correct', ok & prev_ok), ('error', ok & ~prev_ok)]:
        r = dict(mouse_name=mouse, session=session, prev=prev, n_trials=int(sel.sum()),
                 start_delay=float(np.median((start - fb)[sel])) if sel.any() else np.nan)
        for name in FB_WINDOWS:
            lo, hi = win[name][0][sel], win[name][1][sel]
            n, T = count_in(onsets, lo, hi).sum(), (hi - lo).sum()
            r[f'{name}_n'], r[f'{name}_T'] = int(n), float(T)
            r[f'{name}_rate'] = n / T if T > 0 else np.nan
        r['boundary'] = (np.log((r['post_n'] + 0.5) / r['post_T']) - np.log((r['pre_n'] + 0.5) / r['pre_T'])
                         if r['post_T'] > 0 and r['pre_T'] > 0 else np.nan)
        rows.append(r)
    return rows


def build_feedback(S, refresh=False, cache=HERE / 'epoch_structure_feedback.parquet'):
    """Time-since-feedback summaries for every session in table S (cached)."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = []
    for mouse, g in S.groupby('mouse_name'):
        t = mouse_trials(mouse)
        for session in g['session']:
            r = feedback_summaries(mouse, session, t[t['session'] == session].reset_index(drop=True))
            if r:
                rows += r
    F = pd.DataFrame(rows)
    F.to_parquet(cache)
    return F


# ------------------------------------------------------------------------------------------
# Movement-onset rate aligned to the (silent) trial start
# ------------------------------------------------------------------------------------------
# Same counts/exposure construction as hazard_since_feedback, but time is measured from each
# trial's own start. Trial t is at risk from feedback(t-1) to start(t) + POST_S, i.e. it stops
# before the earliest possible go cue, so no bin is shaped by the go-cue selection.
START_EDGES = np.round(np.arange(-3.4, POST_S + 1e-9, 0.1), 1)


def hazard_from_start(mouse, session, tt, edges=START_EDGES):
    f = DM_DIR / f'design_matrix_{session}_{mouse}'
    if not f.exists():
        return None
    dm = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel']).dropna().sort_values('Bin')
    onsets, _, _ = wheel_onsets(dm['Bin'].to_numpy(float), dm['avg_wheel_vel'].to_numpy(float))
    fb = tt['feedback_times'].to_numpy(float)[:-1]
    prev_ok = tt['feedbackType'].to_numpy(float)[:-1] == 1
    start = tt['intervals_0'].to_numpy(float)[1:]
    go = tt['goCue_times'].to_numpy(float)[1:]
    ok = np.isfinite(fb) & np.isfinite(start) & np.isfinite(go) & (start > fb) & (go >= start + POST_S)
    out = {}
    for name, sel in [('correct', ok & prev_ok), ('error', ok & ~prev_ok)]:
        counts = np.zeros(len(edges) - 1)
        for a, st in zip(fb[sel], start[sel]):
            i, j = np.searchsorted(onsets, a, 'right'), np.searchsorted(onsets, st + POST_S, 'right')
            counts += np.histogram(onsets[i:j] - st, edges)[0]
        lo = (fb[sel] - start[sel])[:, None]                       # at risk from feedback ...
        exposure = np.clip(np.minimum(edges[None, 1:], POST_S) - np.maximum(edges[None, :-1], lo), 0, None).sum(0)
        out[name] = dict(counts=counts, exposure=exposure)
    return out


def build_start_hazard(S, refresh=False, cache=HERE / 'epoch_structure_start_hazard.parquet'):
    """One row per (session, previous outcome): onset counts and exposure per bin of time from trial start."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = []
    for mouse, g in S.groupby('mouse_name'):
        t = mouse_trials(mouse)
        for session in g['session']:
            h = hazard_from_start(mouse, session, t[t['session'] == session].reset_index(drop=True))
            if h is None:
                continue
            for prev, d in h.items():
                rows.append(dict(mouse_name=mouse, session=session, prev=prev,
                                 **{f'n{k}': c for k, c in enumerate(d['counts'])},
                                 **{f'e{k}': e for k, e in enumerate(d['exposure'])}))
    Hs = pd.DataFrame(rows)
    Hs.to_parquet(cache)
    return Hs
