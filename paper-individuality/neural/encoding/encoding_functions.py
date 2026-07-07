"""
Encoding-model helpers (Musall/Churchland-style, adapted to the IBL binned neuron files).

Design philosophy
-----------------
* One continuous 60 Hz design matrix per session, built from the neuron file only
  (no ONE call needed: feedback = goCue + response, firstMovement = goCue + reaction
  all land in the same 16.67 ms bin as the canonical ONE times).
* Three kinds of regressors, exactly as in Musall et al. 2019 (Nat. Neurosci.):
    1. peri-event kernels  -> a pulse at the event bin, expanded over a time window.
    2. whole-trial kernels  -> aligned to stimulus onset, amplitude = a decision
                               variable (choice / prior / signed contrast).
    3. instantaneous regressors (no lags) -> the discrete motor STATES
                               (paw / whisk / lick), one-hot per bin.
* Temporal kernels are represented with a small raised-cosine basis (Pillow 2005)
  instead of one column per lag: smooth, interpretable kernels and a small design
  matrix that fits the whole session in seconds.
* Task-vs-motor comparison is done with cross-validated unique variance (dR2),
  which is robust to shared variance between correlated task and motor regressors.

The two variable GROUPS ('task' vs 'motor') are what the dR2 partition compares.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

FS = 60.0            # Hz, binning of the neuron files
DT = 1.0 / FS        # 16.67 ms


# ---------------------------------------------------------------------------
# Temporal basis
# ---------------------------------------------------------------------------
def raised_cosine_basis(win_len, n_basis):
    """Raised-cosine ('bump') basis tiling a window of `win_len` bins.

    Returns an array (win_len, n_basis). Each column is a smooth bump; adjacent
    bumps overlap so any smooth kernel over the window is well approximated by a
    weighted sum of them. This is the standard neural-GLM temporal basis.
    """
    win_len = int(win_len)
    t = np.arange(win_len)
    if n_basis == 1:
        return np.ones((win_len, 1))
    centers = np.linspace(0, win_len - 1, n_basis)
    width = centers[1] - centers[0]          # bump half-period == center spacing
    B = np.zeros((win_len, n_basis))
    for k, c in enumerate(centers):
        x = (t - c) * np.pi / (2.0 * width)
        B[:, k] = np.where(np.abs(x) < np.pi / 2, np.cos(x) ** 2, 0.0)
    return B


def _event_regressors(n_bins, onset_bins, lag_lo_bins, win_len, n_basis,
                      amplitude=None):
    """Expand an event train into raised-cosine temporal regressors.

    Places a (possibly amplitude-weighted) pulse at each onset, then convolves
    with each basis bump. `lag_lo_bins` shifts the window start relative to the
    onset (negative -> the kernel can precede the event).

    Returns array (n_bins, n_basis).
    """
    onset_bins = np.asarray(onset_bins)
    if amplitude is None:
        amplitude = np.ones(len(onset_bins))
    amplitude = np.asarray(amplitude, dtype=float)

    # impulse train with pulses placed at onset + lag_lo (window then runs forward)
    imp = np.zeros(n_bins)
    pos = onset_bins + lag_lo_bins
    ok = np.isfinite(pos) & (pos >= 0) & (pos < n_bins) & np.isfinite(amplitude)
    np.add.at(imp, pos[ok].astype(int), amplitude[ok])

    B = raised_cosine_basis(win_len, n_basis)
    out = np.zeros((n_bins, n_basis))
    for k in range(n_basis):
        out[:, k] = np.convolve(imp, B[:, k])[:n_bins]
    return out


def _filter_signal(s, lo_b, win, k):
    """Filter an analog/indicator signal with a raised-cosine lag basis.

    Returns (n, k): copies of `s` convolved with each basis bump, covering lags
    [lo_b, lo_b+win-1]. NaNs in `s` propagate to every output bin whose lag window
    overlaps them (so those bins are dropped downstream rather than interpolated).
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    B = raised_cosine_basis(win, k)
    out = np.full((n, k), np.nan)
    for kk in range(k):
        full = np.convolve(s, B[:, kk])          # full[i] = sum_j s[i-j] B[j]
        src = np.arange(n) - lo_b                 # out[t] = full[t - lo_b]
        ok = (src >= 0) & (src < len(full))
        out[ok, kk] = full[src[ok]]
    return out


def _nearest_bin(bin_times, event_times):
    """Nearest bin index for each event time (float array; NaN if out of range)."""
    t0 = bin_times[0]
    idx = np.round((np.asarray(event_times, dtype=float) - t0) / DT)
    n = len(bin_times)
    idx = np.where((idx >= 0) & (idx < n), idx, np.nan)
    return idx


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------
# Default kernel windows, in SECONDS: (lag_lo, lag_hi, n_basis).
# lag_lo/lag_hi are relative to the aligning event.
DEFAULT_KERNELS = {
    # --- TASK: peri-event kernels ---
    "stimOn":     dict(lo=0.0,  hi=0.6, k=6, group="task"),
    "fb_correct": dict(lo=0.0,  hi=1.0, k=8, group="task"),
    "fb_error":   dict(lo=0.0,  hi=1.0, k=8, group="task"),
    # --- TASK: decision kernels ---
    # choice is aligned to FIRST MOVEMENT (the action that reports it), not stimOn:
    # RTs are variable, so a stimOn-locked choice kernel would smear the
    # movement-locked choice signal across the RT. prior & signed contrast stay
    # aligned to stimOn.
    "choice":     dict(lo=-0.25, hi=0.75, k=8, group="task"),   # aligned to firstMove
    "prior":      dict(lo=0.0,   hi=1.0,  k=6, group="task"),    # aligned to stimOn
    "scontrast":  dict(lo=0.0,   hi=0.6,  k=6, group="task"),    # aligned to stimOn
}

# Motor STATE columns (discrete HMM states), one-hot encoded.
MOTOR_STATE_COLS = ["paw", "whisk", "lick"]

# Symmetric temporal filter applied to BOTH motor groups (states + continuous) when
# motor_lags=True, so the states-vs-continuous comparison is on equal footing. A
# neuron's response to movement is not instantaneous; this gives each motor
# regressor a short raised-cosine temporal kernel.
MOTOR_LAG = dict(lo=-0.15, hi=0.15, k=4)


def per_trial_table(df):
    """Collapse the binned file to one row per trial (task variables are broadcast)."""
    cols = ["trial_id", "goCueTrigger_times", "reaction", "response",
            "correct", "choice", "contrast", "block"]
    pt = (df.dropna(subset=["goCueTrigger_times"])
            .drop_duplicates("goCueTrigger_times")[cols]
            .sort_values("goCueTrigger_times")
            .reset_index(drop=True))
    return pt


def build_design_matrix(df, kernels=DEFAULT_KERNELS, mask_peritrial=True,
                        mask_pre=0.5, mask_post=2.0, motor_lags=True,
                        motor_continuous=True):
    """Build the Musall-style design matrix for a single session.

    Parameters
    ----------
    df : the binned neuron-file DataFrame (one session).
    kernels : dict of kernel specs (see DEFAULT_KERNELS).
    mask_peritrial : if True, only keep bins within [stimOn-mask_pre, feedback+mask_post].

    Returns
    -------
    X          : DataFrame (n_kept_bins, n_features), z-scored continuous columns.
    groups     : dict {group_name: [column names]} for the dR2 partition.
    trial_ids  : array (n_kept_bins,) trial id per bin (for grouped CV).
    keep_mask  : boolean array over the original bins that were kept.
    """
    df = df.reset_index(drop=True)
    bt = df["Bin"].values
    n = len(bt)

    pt = per_trial_table(df)
    gocue = pt["goCueTrigger_times"].values
    # timings derived from the file (validated to be <1 bin from ONE):
    stimon_t = gocue                                   # stimOn ~= goCue at 60 Hz
    feedback_t = gocue + pt["response"].values         # response_times ~= feedback
    firstmove_t = gocue + pt["reaction"].values        # firstMovement_times

    stimon_b = _nearest_bin(bt, stimon_t)
    feedback_b = _nearest_bin(bt, feedback_t)
    firstmove_b = _nearest_bin(bt, firstmove_t)

    correct = pt["correct"].values                     # 1 correct / 0 error
    choice_right = (pt["choice"].values == "right").astype(float)
    choice_sign = np.where(choice_right == 1, 1.0, -1.0)   # +1 right, -1 left
    prior = pt["block"].values - 0.5                    # -0.3 / 0 / +0.3
    contrast = pt["contrast"].values
    # signed contrast: stimulus side inferred from choice & correctness.
    # stim on left  <=>  chose left AND correct, OR chose right AND error.
    stim_left = (choice_right == 0) == (correct == 1)
    scontrast = np.where(stim_left, -1.0, 1.0) * contrast   # +right / -left

    cols = {}
    groups = {"task": [], "motor_states": [], "motor_continuous": []}

    def add_event(name, onset_b, amp=None):
        spec = kernels[name]
        lo_b = int(round(spec["lo"] * FS))
        win = int(round((spec["hi"] - spec["lo"]) * FS)) + 1
        R = _event_regressors(n, onset_b, lo_b, win, spec["k"], amplitude=amp)
        for k in range(spec["k"]):
            cname = f"{name}_b{k}"
            cols[cname] = R[:, k]
            groups[spec["group"]].append(cname)

    # TASK peri-event kernels
    add_event("stimOn", stimon_b)
    add_event("fb_correct", feedback_b[correct == 1])
    add_event("fb_error",   feedback_b[correct == 0])
    # TASK decision kernels: choice aligned to firstMove, prior/contrast to stimOn
    add_event("choice",    firstmove_b, amp=choice_sign)
    add_event("prior",     stimon_b,    amp=prior)
    add_event("scontrast", stimon_b,    amp=scontrast)

    X = pd.DataFrame(cols, index=df.index)

    # --- collect BASE motor signals (before optional temporal lagging) ---
    motor_base = {"motor_states": {}, "motor_continuous": {}}

    # discrete STATES: one-hot. Unknown (NaN) state -> NaN so the bin is DROPPED
    # (not silently folded into the reference level).
    for c in MOTOR_STATE_COLS:
        nan_rows = df[c].isna().values
        dummies = pd.get_dummies(df[c].astype("Int64"), prefix=c, dtype=float)
        if dummies.shape[1] > 1:                 # drop one level to avoid redundancy
            dummies = dummies.iloc[:, 1:]
        for cc in dummies.columns:
            v = dummies[cc].values.copy()
            v[nan_rows] = np.nan
            motor_base["motor_states"][cc] = v

    # CONTINUOUS analog signals. DLC paw tracking has gaps (NaN): NOT interpolated,
    # affected bins are dropped by the finite-mask below. log1p the heavy-tailed
    # signals (paw speed, lick; raw skew 3-6) so z-scoring isn't dominated by
    # outliers; whisker ME is ~symmetric and left as-is.
    def _paw_speed(xcol, ycol):
        x = df[xcol].values.astype(float)
        y = df[ycol].values.astype(float)
        return np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))

    if motor_continuous:
        motor_base["motor_continuous"] = {
            "l_paw_speed": np.log1p(_paw_speed("l_paw_x", "l_paw_y")),
            "r_paw_speed": np.log1p(_paw_speed("r_paw_x", "r_paw_y")),
            "whisker_me":  df["whisker_me"].values.astype(float),
            "lick_count":  np.log1p(df["Lick count"].values.astype(float)),
        }
    # else: leave motor_continuous empty (states-only encoding; avoids collinearity
    # and the tracking-gap bin loss from paw signals).

    # --- expand motor signals into the design matrix ---
    # `motor_lags` selects which motor group(s) get a temporal lag basis:
    #   True  -> both groups        False/None -> neither
    #   a collection of group names -> only those (e.g. ('motor_states',) is the
    #   Musall-faithful choice: lick/whisk are "motor events" and ARE lagged, but
    #   continuous analog signals are NOT).
    def _lag_this(grp):
        if motor_lags is True:
            return True
        if not motor_lags:
            return False
        return grp in motor_lags

    lo_b = int(round(MOTOR_LAG["lo"] * FS))
    win = int(round((MOTOR_LAG["hi"] - MOTOR_LAG["lo"]) * FS)) + 1
    kk = MOTOR_LAG["k"]
    for grp, sigs in motor_base.items():
        for name, s in sigs.items():
            if _lag_this(grp):
                F = _filter_signal(s, lo_b, win, kk)
                for b in range(kk):
                    cname = f"{name}_L{b}"
                    X[cname] = F[:, b]
                    groups[grp].append(cname)
            else:
                X[name] = s
                groups[grp].append(name)

    # --- peri-trial mask ---
    if mask_peritrial:
        keep = np.zeros(n, dtype=bool)
        pre = int(round(mask_pre * FS))
        post = int(round(mask_post * FS))
        for s, f in zip(stimon_b, feedback_b):
            if not (np.isfinite(s) and np.isfinite(f)):
                continue
            lo = max(0, int(s) - pre)
            hi = min(n, int(f) + post + 1)
            keep[lo:hi] = True
    else:
        keep = np.ones(n, dtype=bool)

    # drop bins with any NaN regressor (tracking gaps) rather than interpolating
    keep = keep & np.isfinite(X.values).all(axis=1)

    trial_ids = df["trial_id"].values
    return X.loc[keep].reset_index(drop=True), groups, trial_ids[keep], keep


# ---------------------------------------------------------------------------
# Ridge fitting + variance partition
# ---------------------------------------------------------------------------
def _ridge_cv_predict(X, Y, trial_ids, alphas, n_splits=5):
    """Grouped-CV ridge for all neurons jointly, evaluated over an alpha grid.

    Uses an SVD per fold so every alpha is essentially free. Returns
    Ypred[a] (n, m) held-out predictions for each alpha, plus the fold labels.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    Ypred = {a: np.zeros_like(Y) for a in alphas}
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, groups=trial_ids):
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        ymu = Y[tr].mean(0)
        Ytr = Y[tr] - ymu
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ Ytr
        for a in alphas:
            d = s / (s ** 2 + a)
            W = Vt.T @ (d[:, None] * UtY)       # (p, m)
            Ypred[a][te] = Xte @ W + ymu
    return Ypred


def _r2(Y, Yhat):
    """Per-column R^2 (cross-validated when Yhat are held-out predictions)."""
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    ss_tot[ss_tot == 0] = np.nan
    return 1.0 - ss_res / ss_tot


def fit_and_partition(X, Y, groups, trial_ids, alphas=None, n_splits=5):
    """Fit the full model and compute cvR2 + unique dR2 for each variable group.

    Works with any number of groups. A single ridge alpha is chosen on the FULL
    model (maximizing mean cvR2 across neurons) and reused for every reduced
    model, so the dR2 partition is apples-to-apples.

    Returns a DataFrame with one row per neuron:
        cv_r2            full-model cross-validated R^2
        dR2_<group>      variance UNIQUE to that group (full - model-without-group)
        r2_<group>_only  that-group-only cvR2
    """
    if alphas is None:
        alphas = np.logspace(-2, 5, 8)
    Yv = np.asarray(Y, dtype=float)
    group_names = [g for g in groups if len(groups[g]) > 0]
    all_cols = [c for g in group_names for c in groups[g]]

    def cvr2(cols, alpha):
        preds = _ridge_cv_predict(X[cols].values, Yv, trial_ids, [alpha], n_splits)
        return _r2(Yv, preds[alpha])

    # choose alpha on the full model
    preds = _ridge_cv_predict(X[all_cols].values, Yv, trial_ids, alphas, n_splits)
    best_alpha, best_mean, full_r2 = None, -np.inf, None
    for a in alphas:
        r2 = _r2(Yv, preds[a])
        m = np.nanmean(r2)
        if m > best_mean:
            best_alpha, best_mean, full_r2 = a, m, r2

    out = {"cv_r2": full_r2}
    for g in group_names:
        g_cols = set(groups[g])
        others = [c for c in all_cols if c not in g_cols]
        r2_without = cvr2(others, best_alpha) if others else np.zeros_like(full_r2)
        out[f"dR2_{g}"] = full_r2 - r2_without         # unique to group g
        out[f"r2_{g}_only"] = cvr2(groups[g], best_alpha)

    res = pd.DataFrame(out)
    res.attrs["best_alpha"] = best_alpha
    return res


# ---------------------------------------------------------------------------
# Coefficients, predictions and kernel reconstruction (for plotting)
# ---------------------------------------------------------------------------
def fit_coefficients(X, Y, alpha):
    """Ridge fit on ALL data; returns standardized coefficients (comparable across
    regressors), the column names, and per-neuron means.

    Coefficients are in z-scored-predictor units (effect on firing rate per 1 SD
    of the regressor), so magnitudes are directly comparable between regressors.
    """
    Xv = np.asarray(X, dtype=float)
    Yv = np.asarray(Y, dtype=float)
    mu, sd = Xv.mean(0), Xv.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xv - mu) / sd
    ymu = Yv.mean(0)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    d = s / (s ** 2 + alpha)
    W = Vt.T @ (d[:, None] * (U.T @ (Yv - ymu)))     # (p, m) standardized coefs
    return W, list(X.columns), ymu


def cv_predictions(X, Y, trial_ids, alpha, n_splits=5):
    """Held-out (cross-validated) full-model predictions, shape (n_bins, n_neurons)."""
    preds = _ridge_cv_predict(np.asarray(X, dtype=float), np.asarray(Y, dtype=float),
                              trial_ids, [alpha], n_splits)
    return preds[alpha]


def event_kernel(name, w_col, colnames, kernels=DEFAULT_KERNELS):
    """Reconstruct one event's temporal kernel from its basis weights.

    Returns (t_seconds, kernel) for a single neuron's coefficient vector `w_col`.
    """
    spec = kernels[name]
    lo_b = int(round(spec["lo"] * FS))
    win = int(round((spec["hi"] - spec["lo"]) * FS)) + 1
    B = raised_cosine_basis(win, spec["k"])
    idx = [colnames.index(f"{name}_b{k}") for k in range(spec["k"])]
    t = (np.arange(win) + lo_b) / FS
    return t, B @ np.asarray(w_col)[idx]


# ---------------------------------------------------------------------------
# Shuffle null for dR2 significance
# ---------------------------------------------------------------------------
def _prep_folds(Xsub, trial_ids, alpha, n_splits):
    """Precompute per-fold ridge projections so a shuffled Y can be scored cheaply.

    For each fold, prediction is linear in the training responses:
        Ypred_te = Xte_std @ (P @ (Ytr - ymu)) + ymu,  P = V diag(s/(s^2+a)) U^T.
    We cache P (p x n_tr) and the standardized Xte so only cheap matmuls remain.
    """
    Xsub = np.asarray(Xsub, dtype=float)
    folds = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(Xsub, groups=trial_ids):
        Xtr, Xte = Xsub[tr], Xsub[te]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        d = s / (s ** 2 + alpha)
        P = (Vt.T * d) @ U.T                     # (p, n_tr)
        folds.append({"tr": tr, "te": te, "P": P, "Xte": Xte})
    return folds


def _cv_r2_from_folds(folds, Y):
    """Cross-validated per-neuron R^2 from precomputed folds (Y may be shuffled)."""
    n, m = Y.shape
    Ypred = np.zeros((n, m))
    for f in folds:
        Ytr = Y[f["tr"]]
        ymu = Ytr.mean(0)
        W = f["P"] @ (Ytr - ymu)                 # (p, m)
        Ypred[f["te"]] = f["Xte"] @ W + ymu
    return _r2(Y, Ypred)


def shuffle_null(X, Y, groups, trial_ids, alpha, n_shuffles=100, n_splits=5):
    """Circular-shift null for each group's unique dR2.

    Each neuron's spike train is circularly shifted (preserving autocorrelation
    but breaking alignment to the regressors), and dR2 is recomputed. Returns:
        obs   : dict group -> observed dR2 per neuron (n_neurons,)
        pvals : dict group -> p-value per neuron  (fraction of null >= observed)
        null  : dict group -> null dR2 samples (n_shuffles, n_neurons)
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    group_names = [g for g in groups if len(groups[g]) > 0]
    all_cols = [c for g in group_names for c in groups[g]]

    prep = {"full": _prep_folds(X[all_cols].values, trial_ids, alpha, n_splits)}
    for g in group_names:
        gc = set(groups[g])
        others = [c for c in all_cols if c not in gc]
        prep[f"wo_{g}"] = _prep_folds(X[others].values, trial_ids, alpha, n_splits)

    def dr2(Yv):
        fr = _cv_r2_from_folds(prep["full"], Yv)
        return {g: fr - _cv_r2_from_folds(prep[f"wo_{g}"], Yv) for g in group_names}

    obs = dr2(Y)
    null = {g: np.zeros((n_shuffles, m)) for g in group_names}
    guard = max(1, n // 10)                      # keep shifts well away from 0 / n
    span = max(1, n - 2 * guard)
    for i in range(n_shuffles):
        # deterministic per-(shuffle, neuron) shifts (no RNG needed for resumability)
        shifts = guard + (((np.arange(m) + 1) * (i + 1) * 2654435761) % span)
        Ys = np.empty_like(Y)
        for j in range(m):
            Ys[:, j] = np.roll(Y[:, j], shifts[j])
        d = dr2(Ys)
        for g in group_names:
            null[g][i] = d[g]

    pvals = {g: (1 + (null[g] >= obs[g][None, :]).sum(0)) / (n_shuffles + 1)
             for g in group_names}
    return obs, pvals, null


# ---------------------------------------------------------------------------
# One-call per-session pipeline (for scaling across sessions)
# ---------------------------------------------------------------------------
def fit_session(df, motor_lags=True, motor_continuous=True, n_shuffles=50, n_splits=5):
    """Build design, fit + partition, and (optionally) run the shuffle null for a
    single session. Returns a per-neuron results DataFrame (self-contained columns,
    safe to concatenate / write to parquet).

    For the scaled encoding analysis use motor_continuous=False (task + motor states
    only): avoids task/motor collinearity and the paw tracking-gap bin loss.
    """
    X, groups, trial_ids, keep = build_design_matrix(
        df, motor_lags=motor_lags, motor_continuous=motor_continuous)
    spike_cols = [c for c in df.columns if c.endswith("_spike_count")]
    Y = df.loc[keep, spike_cols].values.astype(float)

    res = fit_and_partition(X, Y, groups, trial_ids, n_splits=n_splits)
    alpha = res.attrs["best_alpha"]
    res.insert(0, "neuron", spike_cols)
    res.insert(1, "area", [c.split("_neuron_")[0] for c in spike_cols])

    if n_shuffles:
        _, pvals, _ = shuffle_null(X, Y, groups, trial_ids, alpha,
                                   n_shuffles=n_shuffles, n_splits=n_splits)
        for g in pvals:
            res[f"p_{g}"] = pvals[g]

    res["best_alpha"] = alpha
    res["n_bins"] = int(keep.sum())
    res["frac_bins_kept"] = float(keep.mean())
    return res


def fit_session_motor_split(df, motor_lags=(), n_splits=5):
    """Per-session encoding fit with the motor-STATES group split into paw / whisk /
    lick, to compare how much each state uniquely explains. Groups: task, paw, whisk,
    lick (motor_continuous excluded). Returns a per-neuron DataFrame with cv_r2,
    dR2_<g> and r2_<g>_only for each group.
    """
    X, groups, trial_ids, keep = build_design_matrix(
        df, motor_continuous=False, motor_lags=motor_lags)
    g = {"task": groups["task"], "paw": [], "whisk": [], "lick": []}
    for c in groups["motor_states"]:
        g[c.split("_")[0]].append(c)          # 'paw_1'->paw, 'whisk_1'->whisk, ...
    g = {k: v for k, v in g.items() if v}     # drop any empty group

    spike_cols = [c for c in df.columns if c.endswith("_spike_count")]
    Y = df.loc[keep, spike_cols].values.astype(float)
    res = fit_and_partition(X, Y, g, trial_ids, n_splits=n_splits)
    res.insert(0, "neuron", spike_cols)
    res.insert(1, "area", [c.split("_neuron_")[0] for c in spike_cols])
    res["n_cols_paw"] = len(g.get("paw", []))
    res["n_cols_whisk"] = len(g.get("whisk", []))
    res["n_cols_lick"] = len(g.get("lick", []))
    res["best_alpha"] = res.attrs["best_alpha"]
    res["n_bins"] = int(keep.sum())
    return res


# ---------------------------------------------------------------------------
# Effect of the behavioral LDA1 axis on encoding (total R2, task vs motor share)
# ---------------------------------------------------------------------------
def _lda_model_df(pop, lda, target, min_area_n=10):
    """Merge session-level LDA1 onto per-neuron results and add the modeling target.

    target='task_fraction' -> dR2_task / (dR2_task + dR2_motor_states), restricted
    to neurons where that denominator is positive (i.e. they encode something).
    """
    lda1 = lda[["session", "lda_1"]].drop_duplicates("session")
    d = pop.merge(lda1, on="session", how="inner").copy()
    if target == "task_fraction":
        denom = d["dR2_task"] + d["dR2_motor_states"]
        d = d[denom > 0].copy()
        d["task_fraction"] = d["dR2_task"] / (d["dR2_task"] + d["dR2_motor_states"])
    d = d.dropna(subset=[target, "lda_1"])
    if min_area_n:
        vc = d["area"].value_counts()
        d = d[d["area"].isin(vc[vc >= min_area_n].index)]
    d["zlda1"] = (d["lda_1"] - d["lda_1"].mean()) / d["lda_1"].std()
    return d


def lda1_effect(pop, lda, target="cv_r2", control_region=True,
                cluster="mouse_name", min_area_n=10):
    """Mixed-effects test of the LDA1 axis on an encoding target.

    Model: target ~ zLDA1 [+ C(area)], random intercept per `cluster` (mouse) so
    the LDA1 effect is estimated across mice rather than pseudo-replicated neurons.
    zLDA1 is z-scored, so its coefficient is the change in target per 1 SD of LDA1.

    Returns the fitted statsmodels result (or None if too few mice/sessions).
    """
    import statsmodels.formula.api as smf

    d = _lda_model_df(pop, lda, target, min_area_n)
    n_sess, n_mice = d["session"].nunique(), d[cluster].nunique()
    if n_sess < 3 or n_mice < 3:
        print(f"[{target}] too few sessions/mice ({n_sess} sessions, {n_mice} mice) "
              f"— scale up the sweep first.")
        return None

    rhs = "zlda1 + C(area)" if control_region else "zlda1"
    try:
        res = smf.mixedlm(f"{target} ~ {rhs}", d, groups=d[cluster]).fit(method="lbfgs")
    except Exception as e:
        print(f"[{target:14s}] mixedlm failed to converge ({type(e).__name__}) — skipped")
        return None
    beta, se, p = res.params["zlda1"], res.bse["zlda1"], res.pvalues["zlda1"]
    print(f"[{target:14s}] LDA1 beta (per SD) = {beta:+.4f}  SE={se:.4f}  p={p:.3g}  "
          f"| region {'controlled' if control_region else 'ignored'}, "
          f"{n_sess} sessions, {n_mice} mice, {len(d)} neurons")
    return res


def _bf_label(bf):
    if bf > 10:
        return "strong H1"
    if bf > 3:
        return "moderate H1"
    if bf < 1 / 10:
        return "strong H0"
    if bf < 1 / 3:
        return "moderate H0"
    return "inconclusive"


def lda1_perm_bf(pop, lda, target="cv_r2", region_col="area", extra_covars=("n_bins",),
                 perm_group="session", n_perm=2000, seed=0, min_area_n=10):
    """LDA1 effect via session-stratified permutation test + partial-correlation
    JZS Bayes factor -- the same framework as fano_factor/ff_psth_ldabin.ipynb and
    firing_rate/fr_psth_ldabin.ipynb (Wetzels & Wagenmakers 2012 BF).

    Frisch-Waugh-Lovell: residualize target and LDA1 on [intercept + region dummies
    + extra covariates]; the slope of the residuals is LDA1's partial coefficient.
    Significance from shuffling LDA1 across `perm_group` (default session -> respects
    neuron-within-session pseudoreplication, like the paper's neural-metric test);
    evidence from the JZS BF on the partial correlation. `extra_covars` defaults to
    n_bins (amount of data per session, analogous to the paper's n_trials).
    """
    from scipy.stats import pearsonr
    import pingouin as pg

    d = _lda_model_df(pop, lda, target, min_area_n).reset_index(drop=True)
    d = d.dropna(subset=[region_col, perm_group, *extra_covars])
    n_groups = d[perm_group].nunique()
    if n_groups < 3:
        print(f"[{target}] too few {perm_group} groups ({n_groups}) — scale up first.")
        return None

    y = d[target].values.astype(float)
    x = d["lda_1"].values.astype(float)
    dummies = pd.get_dummies(d[region_col], drop_first=True)
    Z = pd.concat([pd.Series(1.0, index=d.index, name="Intercept"), dummies,
                   d[list(extra_covars)].astype(float)], axis=1).values.astype(float)
    Zpinv = np.linalg.pinv(Z)

    y_resid = y - Z @ (Zpinv @ y)
    x_resid = x - Z @ (Zpinv @ x)
    slope_obs = np.sum(x_resid * y_resid) / np.sum(x_resid ** 2)
    r_partial, _ = pearsonr(x_resid, y_resid)
    n_eff = max(len(d) - (Z.shape[1] - 1), 3)
    bf10 = pg.bayesfactor_pearson(r_partial, n_eff)

    grp_to_x = d.groupby(perm_group)["lda_1"].first()
    grp_index = d[perm_group].values
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        sh = pd.Series(rng.permutation(grp_to_x.values), index=grp_to_x.index)
        xp = sh.reindex(grp_index).values.astype(float)
        xpr = xp - Z @ (Zpinv @ xp)
        perm[i] = np.sum(xpr * y_resid) / np.sum(xpr ** 2)
    p_perm = float(np.mean(np.abs(perm) >= np.abs(slope_obs)))

    print(f"[{target:14s}] slope={slope_obs:+.4g}  r_partial={r_partial:+.3f}  "
          f"BF10={bf10:.3g} ({_bf_label(bf10)})  p_perm={p_perm:.4f}  "
          f"| {len(d)} neurons, {n_groups} {perm_group}s")
    return dict(target=target, n=len(d), n_groups=n_groups, slope=slope_obs,
                r_partial=r_partial, bf10=bf10, bf_evidence=_bf_label(bf10), p_perm=p_perm)


def plot_lda1_effect(pop, lda, target="cv_r2", ax=None):
    """Descriptive session-level scatter of `target` vs LDA1 (LDA1 is per session),
    with an OLS line and Pearson r. Statistical test lives in lda1_effect()."""
    import matplotlib.pyplot as plt
    from scipy import stats as ss

    d = _lda_model_df(pop, lda, target, min_area_n=0)
    s = (d.groupby("session")
           .agg(y=(target, "mean"), lda_1=("lda_1", "first"), n=(target, "size"))
           .reset_index())
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(s["lda_1"], s["y"], s=6 + s["n"], alpha=.6, color="#3a7ca5",
               edgecolor="none")
    if len(s) >= 3:
        b, a = np.polyfit(s["lda_1"], s["y"], 1)
        xs = np.array([s["lda_1"].min(), s["lda_1"].max()])
        ax.plot(xs, a + b * xs, "k--", lw=1.5)
        r, p = ss.pearsonr(s["lda_1"], s["y"])
        ax.set_title(f"{target}  (session-level r={r:+.2f}, p={p:.2g}, N={len(s)})",
                     fontsize=10)
    ax.set(xlabel="LDA1", ylabel=f"mean {target}")
    return ax


# Residual-variability functions (residual_matrices, residual_noise_corr,
# residual_quench) live in variability_functions.py.
