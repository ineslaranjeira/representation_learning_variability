"""
What predicts WHEN the stimulus starts to drive a mouse's choices?
==================================================================
Used by 4_onset_predictors.ipynb.

OUTCOME -- onset of stimulus use, per mouse. The per-session stimulus weight `w_stim`
(choice_strategy.py) is noisy, so it is smoothed with an isotonic (monotone non-decreasing)
fit across sessions, and onset is where that fit first reaches ONSET_W, linearly interpolated
between sessions. Onset is also expressed in TRIALS: trials completed before the onset point.

PREDICTORS -- measured in a FIXED window, training sessions 1-2, for every mouse. Using each
mouse's own pre-onset period would give slow learners longer windows and more data.

  experience loop (trial tables)
    p_stay_err, p_stay_cor     raw stay probabilities after an error / a correct trial
    w_lose, w_win              history weights from the choice model fitted on sessions 1-2 pooled
    abs_bias                   side bias, balanced over stimulus side
    run_length                 mean length of runs of same-side choices
    p_against_bias_err         how often the protocol put the stimulus against the mouse's recent
                               bias after an error (what the mouse was exposed to)
    reward_rate                rewarded trials per minute
    trials_per_session, trials_per_min
  hidden stimulus effects (trial tables + wheel)
    rt_contrast_effect         log RT(contrast 0.5) - log RT(contrast 1.0): does stimulus strength
                               change reaction time, whatever the choice?
    rt_error_effect            log RT(error) - log RT(correct)
    first_dir_toward_stim      P(first wheel movement goes toward the stimulus side) - 0.5
    first_dir_minus_choice     the same, minus P(choice toward the stimulus) - 0.5: is the FIRST
                               movement more stimulus-guided than the final choice?
    com_rate, com_benefit      changes of mind (first movement opposite the final choice) and
                               P(correct | change of mind) - P(correct | no change of mind)
    wheel_post_error_pause     log(free-window movement rate after error / after correct),
                               from notebook 2
  early performance (reference only: this is the outcome measured early, not a predictor)
    accuracy_easy, w_stim

WHEEL DIRECTION. The sign convention of `avg_wheel_vel` is not assumed: per session, the sign of
the wheel displacement from first movement to response is matched to the chosen side (stimulus
coordinates, as choice_strategy), and the session's mapping is the majority one. The share of
trials that agree is kept as a QC column (`wheel_side_agreement`).
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import epoch_structure as es
import choice_strategy as cs

ONSET_W = 1.0
EARLY = (1, 2)
FIRST_DIR_S = 0.1          # first-movement direction = net wheel displacement in this window


# ------------------------------------------------------------------------------------------
# Outcome
# ------------------------------------------------------------------------------------------
def onset_table(S=None, w=ONSET_W):
    """Per mouse: onset of stimulus use in sessions and in trials (NaN if never reached)."""
    S = cs.build() if S is None else S
    S = S[S['protocol'].str.contains('training')]
    rows = []
    for m, g in S.groupby('mouse_name'):
        g = g.sort_values('session_index').dropna(subset=['w_stim'])
        x, y = g['session_index'].to_numpy(float), g['w_stim'].to_numpy(float)
        fit = IsotonicRegression(increasing=True).fit_transform(x, y)
        cum_trials = np.r_[0, np.cumsum(g['n_trials'].to_numpy())]      # trials before session k
        on_s = on_t = np.nan
        hit = np.flatnonzero(fit >= w)
        if len(hit):
            k = hit[0]
            if k == 0:
                frac = 0.0
            else:
                frac = (w - fit[k - 1]) / (fit[k] - fit[k - 1]) if fit[k] > fit[k - 1] else 1.0
            on_s = x[k - 1] + frac if k > 0 else x[0]
            # trials: all trials of sessions before k-1, plus the interpolated share of session k-1 -> k
            on_t = cum_trials[k] if k == 0 else cum_trials[k - 1] + frac * g['n_trials'].iloc[k - 1]
        rows.append(dict(mouse_name=m, onset_sessions=on_s, onset_trials=on_t,
                         n_sessions_total=int(g['session_index'].max())))
    return pd.DataFrame(rows).set_index('mouse_name')


# ------------------------------------------------------------------------------------------
# Predictors, sessions 1-2
# ------------------------------------------------------------------------------------------
def _wheel_first_direction(mouse, session, tt):
    """Per trial (rows of tt): side of the first movement and of the whole movement, in stimulus
    coordinates via the session's wheel->side mapping; NaN where undefined."""
    f = es.DM_DIR / f'design_matrix_{session}_{mouse}'
    n = len(tt)
    nan = np.full(n, np.nan)
    if not f.exists():
        return nan, nan, np.nan
    dm = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel']).dropna().sort_values('Bin')
    b, v = dm['Bin'].to_numpy(float), dm['avg_wheel_vel'].to_numpy(float)
    dt = np.median(np.diff(b))
    cum = np.r_[0, np.cumsum(v) * dt]                       # wheel position (arbitrary units)

    def disp(lo, hi):
        i, j = np.searchsorted(b, lo, 'right'), np.searchsorted(b, hi, 'right')
        return cum[j] - cum[i]

    fm = tt['firstMovement_times'].to_numpy(float)
    resp = tt['response_times'].to_numpy(float)
    ok = np.isfinite(fm) & np.isfinite(resp) & (resp > fm)
    d_first, d_all = nan.copy(), nan.copy()
    d_first[ok] = disp(fm[ok], fm[ok] + FIRST_DIR_S)
    d_all[ok] = disp(fm[ok], resp[ok])
    side = tt['side'].to_numpy(float)
    use = ok & (side != 0) & (d_all != 0)
    if use.sum() < 20:
        return nan, nan, np.nan
    agree = np.mean(np.sign(d_all[use]) == side[use])
    sgn = 1.0 if agree >= 0.5 else -1.0                     # wheel sign -> side mapping for this session
    first_side = np.where(ok & (d_first != 0), sgn * np.sign(d_first), np.nan)
    all_side = np.where(use, sgn * np.sign(d_all), np.nan)
    return first_side, all_side, max(agree, 1 - agree)


def early_predictors(mouse, sessions=EARLY, wheel=True, prepared=None):
    """Predictors pooled over `sessions`. `prepared` = cs.prepare(es.mouse_trials(mouse)), to
    avoid re-reading the trial table when looping over sessions."""
    t = cs.prepare(es.mouse_trials(mouse)) if prepared is None else prepared
    t = t[t['session_index'].isin(sessions)]
    if t.empty:
        return None
    t = t.copy()
    t['recent_bias'] = t.groupby('session_index')['side'].transform(
        lambda x: x.where(x != 0).shift().rolling(10, min_periods=5).mean())
    out = dict(mouse_name=mouse, n_early_sessions=int(t['session_index'].nunique()))

    # pacing and reward
    dur = t.groupby('session_index').apply(lambda g: (g['intervals_1'].max() - g['intervals_0'].min()) / 60)
    out['trials_per_session'] = float(t.groupby('session_index').size().mean())
    out['trials_per_min'] = float(len(t) / dur.sum())
    out['reward_rate'] = float(t['correct'].sum() / dur.sum())

    v = t[(t['side'] != 0) & t['prev_side'].isin([-1, 1])]
    stay = v['side'] == v['prev_side']
    out['p_stay_err'] = float(stay[v['prev_correct'] == 0].mean())
    out['p_stay_cor'] = float(stay[v['prev_correct'] == 1].mean())
    pr = v[v['stim'] != 0].groupby('stim')['side'].apply(lambda s: (s > 0).mean())
    out['abs_bias'] = float(abs(pr.mean() - 0.5)) if len(pr) == 2 else np.nan
    runs = []
    for _, g in t[t['side'] != 0].groupby('session_index'):
        s = g['side'].to_numpy()
        change = np.flatnonzero(np.diff(s) != 0)
        runs += list(np.diff(np.r_[-1, change, len(s) - 1]))
    out['run_length'] = float(np.mean(runs)) if runs else np.nan
    e = v[(v['prev_correct'] == 0) & (v['stim'] != 0) & v['recent_bias'].notna() & (v['recent_bias'] != 0)]
    out['p_against_bias_err'] = float((e['stim'] == -np.sign(e['recent_bias'])).mean()) if len(e) else np.nan
    out.update({k: v_ for k, v_ in cs.fit_choice_model(v).items() if k in ('w_win', 'w_lose', 'w_stim')})
    easy = v['contrast'] >= cs.EASY
    out['accuracy_easy'] = float(v.loc[easy, 'correct'].mean())

    # reaction-time effects of the stimulus
    r = v[(v['rt'] > 0.05) & np.isfinite(v['rt'])]
    med = lambda s: np.log(s.median()) if len(s) >= 10 else np.nan
    out['rt_contrast_effect'] = med(r.loc[np.isclose(r['contrast'], 0.5), 'rt']) - med(r.loc[np.isclose(r['contrast'], 1.0), 'rt'])
    out['rt_error_effect'] = med(r.loc[r['correct'] == 0, 'rt']) - med(r.loc[r['correct'] == 1, 'rt'])

    # wheel: first-movement direction and changes of mind
    if wheel:
        fs, al, agr = [], [], []
        for sess, g in t.groupby('session', sort=False):
            a, b_, q = _wheel_first_direction(mouse, sess, g.reset_index(drop=True))
            fs.append(a); al.append(b_); agr.append(q)
        t['first_side'] = np.concatenate(fs); t['all_side'] = np.concatenate(al)
        out['wheel_side_agreement'] = float(np.nanmean(agr)) if np.isfinite(agr).any() else np.nan
        w = t[(t['stim'] != 0) & (t['side'] != 0) & t['first_side'].notna()]
        if len(w) >= 50:
            out['first_dir_toward_stim'] = float((w['first_side'] == w['stim']).mean() - 0.5)
            out['first_dir_minus_choice'] = float((w['first_side'] == w['stim']).mean() - (w['side'] == w['stim']).mean())
            com = w['first_side'] != w['side']
            out['com_rate'] = float(com.mean())
            out['com_benefit'] = float(w.loc[com, 'correct'].mean() - w.loc[~com, 'correct'].mean()) \
                if com.sum() >= 10 else np.nan
    return out


def build(refresh=False, cache=es.HERE / 'onset_predictors_mice.parquet'):
    """Per-mouse table: early predictors (sessions 1-2) + onset of stimulus use (cached)."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = [r for f in sorted(es.os.listdir(es.TRIALS_DIR))
            if (r := early_predictors(f.replace('training_data_trials_', ''))) is not None]
    P = pd.DataFrame(rows).set_index('mouse_name')
    # wheel post-error pause from notebook 2, sessions 1-2
    SW = es.build()
    F = es.build_feedback(SW).merge(SW[['session', 'session_index']], on='session')
    F = F[F['session_index'].isin(EARLY)].groupby(['mouse_name', 'prev'])[['free_matched_n', 'free_matched_T']].sum()
    rate = (F['free_matched_n'] / F['free_matched_T']).unstack('prev')
    P['wheel_post_error_pause'] = np.log(rate['error'] / rate['correct']).replace([np.inf, -np.inf], np.nan)
    P = P.join(onset_table())
    P.to_parquet(cache)
    return P


# ------------------------------------------------------------------------------------------
# The same measures, session by session (for temporal precedence within each mouse)
# ------------------------------------------------------------------------------------------
def build_per_session(refresh=False, cache=es.HERE / 'onset_predictors_sessions.parquet'):
    """One row per (mouse, training session): every early_predictors measure computed on that
    single session, plus the wheel post-error pause (sessions with wheel data only)."""
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = []
    for f in sorted(es.os.listdir(es.TRIALS_DIR)):
        mouse = f.replace('training_data_trials_', '')
        t = cs.prepare(es.mouse_trials(mouse))
        for si in sorted(t['session_index'].unique()):
            r = early_predictors(mouse, sessions=(si,), prepared=t)
            if r is not None:
                r['session_index'] = int(si)
                r['session'] = t.loc[t['session_index'] == si, 'session'].iloc[0]
                rows.append(r)
    Q = pd.DataFrame(rows)
    SW = es.build()
    F = es.build_feedback(SW).pivot_table(index='session', columns='prev', values='free_matched_rate')
    Q['wheel_post_error_pause'] = Q['session'].map(np.log(F['error'] / F['correct']).replace([np.inf, -np.inf], np.nan))
    Q.to_parquet(cache)
    return Q
