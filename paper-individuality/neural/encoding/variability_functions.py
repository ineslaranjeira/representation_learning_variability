"""
Residual variability *quench* analysis (separate from encoding_functions.py).

Motivation: what tracks LDA1 is the stimulus-driven QUENCH in variability
(the drop from pre-stim/spontaneous to evoked), not the session-average level.
Here we compute that quench on the ENCODING RESIDUALS at three removal levels
(raw / task-removed / task+motor-removed), so we can ask whether the quench-vs-LDA1
relationship is behavioural (dies after removing task+movement) or internal
(survives). Uses cross-validated residuals from encoding_functions.

Measure per region, aligned to stimulus onset:
  * rSC_*  : mean off-diagonal across-TRIAL correlation of the windowed residual
             (shared / coordinated variability = noise-correlation analogue).
Quench = value(pre-stim window) - value(evoked window).

We deliberately do NOT recompute Fano factor here: the FF quench is the established
ff_psth_ldabin.ipynb analysis (condition-adjusted mean-matched window_ff over the
same PRE/POST windows, ff_quench = ff_pre - ff_post, tested with the perm+BF
age_effect_test == lda1_perm_bf). Correlation residualises cleanly; Fano does not.
Windows match ff_psth_ldabin: PRE=(-0.2, 0.0), POST=(0.1, 0.3).
"""

import numpy as np
import pandas as pd
import encoding_functions as ef

FS = ef.FS
MIN_WINDOW_COUNT = 0.5   # min mean spike count in a window for a stable FF (as in ff_psth_ldabin)


def _windowed(Rfull, onset_b, lo, hi):
    """Per-trial mean of the residual over [onset+lo, onset+hi). -> (n_trials, n_neurons).

    NaN bins (tracking gaps) are ignored (nanmean); trials whose window falls off the
    session edge become an all-NaN row and drop out of downstream nan-aware stats.
    """
    n_total, m = Rfull.shape
    out = np.full((len(onset_b), m), np.nan)
    for i, ob in enumerate(onset_b):
        if not np.isfinite(ob):
            continue
        a, b = int(ob) + lo, int(ob) + hi
        if a < 0 or b > n_total:
            continue
        out[i] = np.nanmean(Rfull[a:b, :], axis=0)
    return out


def _windowed_sum(Rfull, onset_b, lo, hi):
    """Per-trial SUM of the signal over [onset+lo, onset+hi) -> (n_trials, n_neurons).
    NaN bins ignored; a fully-missing window becomes NaN so it drops from stats."""
    n_total, m = Rfull.shape
    out = np.full((len(onset_b), m), np.nan)
    for i, ob in enumerate(onset_b):
        if not np.isfinite(ob):
            continue
        a, b = int(ob) + lo, int(ob) + hi
        if a < 0 or b > n_total:
            continue
        w = Rfull[a:b, :]
        with np.errstate(invalid="ignore"):
            out[i] = np.where(np.all(np.isnan(w), axis=0), np.nan, np.nansum(w, axis=0))
    return out


def _region_ff(Rlvl, Yc, onset_b, lo, hi):
    """Residual Fano factor for one window, region-averaged: var(residual count) /
    mean(observed count), i.e. ff_psth_ldabin.ipynb's var/mean formula with the
    numerator variance taken on the encoding residual. Rlvl/Yc: (n_bins, n_neurons)."""
    rcount = _windowed_sum(Rlvl, onset_b, lo, hi)      # trials x neurons (residual)
    ocount = _windowed_sum(Yc, onset_b, lo, hi)        # trials x neurons (observed)
    mean = np.nanmean(ocount, axis=0)
    var = np.nanvar(rcount, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ff = var / (mean + 1e-6)
    ff[(mean <= MIN_WINDOW_COUNT) | ~np.isfinite(ff)] = np.nan
    return float(np.nanmean(ff))


def _mean_offdiag_corr(M):
    """Mean off-diagonal Pearson correlation across rows (trials) of M (trials x neurons)."""
    if M.shape[1] < 2:
        return np.nan
    C = pd.DataFrame(M).corr().values          # pairwise-complete, NaN-aware
    iu = np.triu_indices(C.shape[0], k=1)
    return np.nanmean(C[iu])


def residual_matrices(df, alpha=None, n_splits=5, motor_lags=(), motor_continuous=False):
    """Cross-validated residuals of neural activity after removing nothing (raw-
    centred), task only, and task+motor. Held-out (GroupKFold by trial) so they
    aren't biased by in-sample overfitting.

    Returns (R, acronyms, alpha): R is a dict level -> (n_bins, n_neurons) residual.
    """
    X, groups, tids, keep = ef.build_design_matrix(
        df, motor_continuous=motor_continuous, motor_lags=motor_lags)
    spike_cols = [c for c in df.columns if c.endswith("_spike_count")]
    Y = df.loc[keep, spike_cols].values.astype(float)
    task = groups["task"]
    allc = task + groups["motor_states"] + groups.get("motor_continuous", [])
    if alpha is None:
        alphas = np.logspace(-2, 5, 8)
        preds = ef._ridge_cv_predict(X[allc].values, Y, tids, alphas, n_splits)
        alpha = max(alphas, key=lambda a: np.nanmean(ef._r2(Y, preds[a])))
    yhat_task = ef.cv_predictions(X[task], Y, tids, alpha, n_splits)
    yhat_all = ef.cv_predictions(X[allc], Y, tids, alpha, n_splits)
    R = {"raw": Y - Y.mean(0), "task": Y - yhat_task, "taskmotor": Y - yhat_all}
    acr = [c.split("_neuron_")[0] for c in spike_cols]
    return R, acr, alpha


def residual_noise_corr(df, region_of, alpha=None, min_neurons=5, **kw):
    """Session-AVERAGE within-region noise correlation of the encoding residuals at
    three removal levels (raw / task / task+motor). Mean off-diagonal correlation of
    the residual time series among the >= min_neurons units of each region.

    Returns (DataFrame [region, n_neurons, rSC_raw, rSC_task, rSC_taskmotor], alpha).
    """
    R, acr, alpha = residual_matrices(df, alpha=alpha, **kw)
    regions = np.array([region_of(a) for a in acr], dtype=object)
    rows = []
    for rg in pd.unique(regions):
        if rg is None:
            continue
        idx = np.where(regions == rg)[0]
        if len(idx) < min_neurons:
            continue
        row = {"region": rg, "n_neurons": int(len(idx))}
        iu = np.triu_indices(len(idx), k=1)
        for lvl, Rm in R.items():
            C = np.corrcoef(Rm[:, idx].T)
            row[f"rSC_{lvl}"] = float(np.nanmean(C[iu]))
        rows.append(row)
    return pd.DataFrame(rows), alpha


def residual_quench(df, region_of, pre=(-0.2, 0.0), post=(0.1, 0.3),
                    alpha=None, min_neurons=5, min_trials=50,
                    motor_lags=(), motor_continuous=False, n_splits=5):
    """Stimulus-onset variability quench of the encoding residuals, per region.

    Returns (DataFrame rows per region, alpha). Per level in {raw, task, taskmotor}:
    rSC_pre_*, rSC_post_* (noise-correlation analogue) and ff_pre_*, ff_post_*
    (residual Fano). Quench = pre - post, computed by the caller. Plus region,
    n_neurons, n_trials.
    """
    X, groups, tids, keep = ef.build_design_matrix(
        df, motor_continuous=motor_continuous, motor_lags=motor_lags)
    spike = [c for c in df.columns if c.endswith("_spike_count")]
    Y = df.loc[keep, spike].values.astype(float)
    task = groups["task"]
    allc = task + groups["motor_states"] + groups.get("motor_continuous", [])

    if alpha is None:
        alphas = np.logspace(-2, 5, 8)
        preds = ef._ridge_cv_predict(X[allc].values, Y, tids, alphas, n_splits)
        alpha = max(alphas, key=lambda a: np.nanmean(ef._r2(Y, preds[a])))

    resid = {
        "raw": Y - Y.mean(0),
        "task": Y - ef.cv_predictions(X[task], Y, tids, alpha, n_splits),
        "taskmotor": Y - ef.cv_predictions(X[allc], Y, tids, alpha, n_splits),
    }

    # scatter residuals back onto the full session timeline (NaN where dropped)
    n_total = len(df)
    keep_idx = np.where(keep)[0]
    Rf = {}
    for lvl, R in resid.items():
        full = np.full((n_total, R.shape[1]), np.nan)
        full[keep_idx] = R
        Rf[lvl] = full
    # observed spike counts on the full timeline (denominator of the residual Fano)
    Ycount = np.full((n_total, Y.shape[1]), np.nan)
    Ycount[keep_idx] = Y

    bt = df["Bin"].values
    stimon_b = ef._nearest_bin(bt, ef.per_trial_table(df)["goCueTrigger_times"].values)
    n_trials = int(np.isfinite(stimon_b).sum())
    if n_trials < min_trials:
        return pd.DataFrame(), alpha

    lo_pre, hi_pre = int(round(pre[0] * FS)), int(round(pre[1] * FS))
    lo_post, hi_post = int(round(post[0] * FS)), int(round(post[1] * FS))
    acr = np.array([c.split("_neuron_")[0] for c in spike], dtype=object)
    regions = np.array([region_of(a) for a in acr], dtype=object)

    rows = []
    for rg in pd.unique(regions):
        if rg is None:
            continue
        idx = np.where(regions == rg)[0]
        if len(idx) < min_neurons:
            continue
        row = {"region": rg, "n_neurons": int(len(idx)), "n_trials": n_trials}
        # At each removal level (raw / task / task+motor) compute, in the pre- and
        # post-stimulus windows:
        #   rSC : mean off-diagonal across-trial correlation (noise-corr analogue)
        #   ff  : residual Fano = var(residual count)/mean(observed count)
        # Quench = pre - post is computed by the caller.
        for lvl in resid:
            pre_m = _windowed(Rf[lvl][:, idx], stimon_b, lo_pre, hi_pre)
            post_m = _windowed(Rf[lvl][:, idx], stimon_b, lo_post, hi_post)
            row[f"rSC_pre_{lvl}"] = _mean_offdiag_corr(pre_m)
            row[f"rSC_post_{lvl}"] = _mean_offdiag_corr(post_m)
            row[f"ff_pre_{lvl}"] = _region_ff(Rf[lvl][:, idx], Ycount[:, idx],
                                              stimon_b, lo_pre, hi_pre)
            row[f"ff_post_{lvl}"] = _region_ff(Rf[lvl][:, idx], Ycount[:, idx],
                                               stimon_b, lo_post, hi_post)
        rows.append(row)
    return pd.DataFrame(rows), alpha
