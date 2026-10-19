"""
What drives the mouse's choices during training: the stimulus, a side bias, or the last trial?
==============================================================================================
Used by 3_choice_strategy.ipynb. One row per training session, from the trial tables alone
(every pre-proficiency trainingChoiceWorld session of all 102 mice; no wheel or video needed).

SIDE CONVENTION. Choices are expressed in STIMULUS coordinates, never through the `choice`
labels (see the mirrored-label caveat in the project notes):
    stim  = sign(contrastRight - contrastLeft)       +1 / -1, 0 for a zero-contrast trial
    side  = -choice                                   so that side == stim on a correct trial
which holds for 100% of correct trials in every session (checked in the notebook).
No-go trials (choice == 0) are dropped from every choice measure and counted separately.

CHOICE MODEL, per session (logistic regression, mild L2 so separable sessions stay finite):
    logit P(side = +1) = b0 + w_stim * signed_contrast
                            + w_win  * prev_side * [prev correct]
                            + w_lose * prev_side * [prev error]
  b0      side bias (|b0| = how strongly the mouse favours one side)
  w_stim  stimulus sensitivity
  w_win   > 0: repeats a rewarded choice (win-stay) beyond what the stimulus explains
  w_lose  < 0: switches after an error (lose-shift) beyond what the stimulus explains

WHY A MODEL AND NOT "STAY PROBABILITIES" ALONE. trainingChoiceWorld often repeats the stimulus
after an error, so after an error the correct answer is usually "switch". A raw lose-shift rate
cannot tell stimulus use from a switch rule. The repeats are not flagged and not deterministic
(the same stimulus returns after 60-70% of high-contrast errors, depending on rig software
version), so they cannot be removed trial by trial; including the stimulus in the model is what
separates the two. The raw rates are kept as well, for reference.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import epoch_structure as es

CACHE = es.HERE / 'choice_strategy_sessions.parquet'
EASY = 0.5
L2_C = 10.0          # inverse L2 strength; large = barely regularised


def prepare(t):
    """Add stimulus-coordinate columns to one mouse's trial table (sorted as es.mouse_trials)."""
    t = t.copy()
    t['signed_contrast'] = t['contrastRight'].fillna(0) - t['contrastLeft'].fillna(0)
    t['contrast'] = t['signed_contrast'].abs()
    t['stim'] = np.sign(t['signed_contrast'])
    t['side'] = -t['choice']                      # +1 = the response that is correct for a right stimulus
    t['correct'] = (t['feedbackType'] == 1).astype(float)
    t['rt'] = t['firstMovement_times'] - t['goCue_times']
    g = t.groupby('session_index', sort=False)
    for c in ['side', 'correct', 'stim', 'signed_contrast', 'rt']:
        t[f'prev_{c}'] = g[c].shift()
    return t


def fit_choice_model(d):
    """d: valid trials (side != 0, previous trial a valid choice). Returns the four weights."""
    X = np.column_stack([d['signed_contrast'],
                         d['prev_side'] * (d['prev_correct'] == 1),
                         d['prev_side'] * (d['prev_correct'] == 0)])
    y = (d['side'] > 0).astype(int)
    if len(d) < 50 or y.nunique() < 2:
        return dict(b0=np.nan, w_stim=np.nan, w_win=np.nan, w_lose=np.nan)
    m = LogisticRegression(C=L2_C, max_iter=1000).fit(X, y)
    return dict(b0=float(m.intercept_[0]), w_stim=float(m.coef_[0, 0]),
                w_win=float(m.coef_[0, 1]), w_lose=float(m.coef_[0, 2]))


def session_measures(g):
    """One row of choice-strategy measures for one session's (prepared) trials."""
    out = dict(n_trials=len(g), nogo_frac=float((g['side'] == 0).mean()),
               n_contrasts=int(g.loc[g['contrast'] > 0, 'contrast'].round(4).nunique()),
               protocol=str(g['task_protocol'].iloc[0]))
    v = g[(g['side'] != 0) & g['prev_side'].isin([-1, 1])]           # valid choice, valid previous choice
    easy = v['contrast'] >= EASY
    out['accuracy_easy'] = float(v.loc[easy, 'correct'].mean()) if easy.any() else np.nan
    # bias balanced over stimulus side, so stimulus repeats after errors do not leak into it
    pr = v[v['stim'] != 0].groupby('stim')['side'].apply(lambda s: (s > 0).mean())
    out['bias'] = float(pr.mean() - 0.5) if len(pr) == 2 else np.nan
    out['abs_bias'] = abs(out['bias']) if np.isfinite(out['bias']) else np.nan
    stay = v['side'] == v['prev_side']
    out['p_stay_after_correct'] = float(stay[v['prev_correct'] == 1].mean())
    out['p_stay_after_error'] = float(stay[v['prev_correct'] == 0].mean())
    # same-stimulus probability after an easy error = how often the protocol repeated it here
    pe = (v['prev_correct'] == 0) & (v['prev_signed_contrast'].abs() >= EASY)
    out['p_same_stim_after_error'] = float((v['signed_contrast'] == v['prev_signed_contrast'])[pe].mean()) \
        if pe.any() else np.nan
    # post-error adjustments
    ac, ae = v.loc[v['prev_correct'] == 1, 'correct'].mean(), v.loc[v['prev_correct'] == 0, 'correct'].mean()
    out['acc_after_correct'], out['acc_after_error'] = float(ac), float(ae)
    rc = v.loc[v['prev_correct'] == 1, 'rt'].median()
    re_ = v.loc[v['prev_correct'] == 0, 'rt'].median()
    out['rt_after_correct'], out['rt_after_error'] = float(rc), float(re_)
    out['post_error_rt_logratio'] = float(np.log(re_ / rc)) if rc > 0 and re_ > 0 else np.nan
    out.update(fit_choice_model(v))
    return out


def build(refresh=False, verbose=True):
    """Per-session table for every training session in the trial tables (cached)."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    rows = []
    for f in sorted(os.listdir(es.TRIALS_DIR)):
        mouse = f.replace('training_data_trials_', '')
        t = prepare(es.mouse_trials(mouse))
        n_sess = t['session_index'].max()
        for (si, sess), g in t.groupby(['session_index', 'session'], sort=True):
            r = session_measures(g)
            r.update(mouse_name=mouse, session=sess, session_index=int(si), n_sessions_total=int(n_sess),
                     training_fraction=(si - 1) / max(n_sess - 1, 1))
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_parquet(CACHE)
    if verbose:
        print(f'{len(S)} sessions, {S.mouse_name.nunique()} mice')
    return S
