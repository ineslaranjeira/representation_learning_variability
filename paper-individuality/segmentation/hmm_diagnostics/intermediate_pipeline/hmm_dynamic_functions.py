"""
Functions for the dynamic HMM search (4.1 + 4.2 joined into one pass).

Design decisions, and why:

  kappa is FIXED AT 0.
      kappa enters the M-step as pseudo-counts on the diagonal of the transition-count
      matrix, so its effect scales with the number of transitions the data contains:
      moving dwell by one frame costs kappa ~ N_exits (thousands here). Across the grids
      previously searched (kappa <= 1000) it changed nothing measurable -- not held-out
      likelihood, not state labels, not durations. It is therefore not searched.
      Fixing kappa=0 also means initialize(method='prior') no longer depends on kappa,
      so the cross-validation baseline is one fixed reference across the whole lag grid.

  The lag grid GROWS instead of being fixed.
      A fixed grid either stops too early (the old [1,10,20,30] pinned ~55% of sessions
      at its ceiling once the fold-pairing was corrected) or wastes fits at lags nothing
      selects. Here the grid doubles from 1 and only extends while the adopted lag is
      still at the ceiling, so each session gets exactly as much grid as it needs.

  Selection uses PAIRED per-fold differences on RAW held-out log-likelihood.
      The 5 CV folds are the same time blocks for every lag, so lag comparisons are
      paired; using each cell's own across-fold SD (as find_2_best_param does) inflates
      the error bar ~14x and collapses the choice onto the grid minimum. Raw held-out LL
      is used rather than baseline-subtracted bits because bits is only comparable when
      the baseline is shared -- true along the lag axis at fixed kappa, but a trap in
      general (it is what made kappa=0 look best when kappa had no effect at all).

  There is NO minimum-gain floor.
      Longer lags keep improving held-out LL because the signal really is autocorrelated
      out to ~1 s, so significance alone would not stop the search -- which is why the
      grid is capped at the decorrelation time instead. A floor was considered and
      dropped: any threshold on the gain is arbitrary at this data size, and the one
      principled version of it (the BIC parameter penalty, ~1e-5 per frame against
      LL differences of ~1e-3) is far too small to bind, besides double-counting
      complexity that cross-validation already charges for. The full per-fold profile is
      saved, so a floor can be applied after the fact if it ever becomes justified.

Everything that fits or cross-validates is imported from segmentation_functions, so the
fitting logic is unchanged from 4.1.
"""

import os
import gc
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import zscore, t as tdist
from joblib import Parallel, delayed

import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM, PoissonHMM

# --- reuse the fitting logic from 4.1 unchanged ---
from segmentation_functions import (
    compute_inputs,
    cross_validate_armodel,
    cross_validate_poismodel,
)


# ============================================================================
# DATA PREPARATION
# ============================================================================


def load_fit_variable(data_path, session, mouse_name, var_interest, zsc):
    """Load one session and return only the column(s) being fitted, NaNs dropped.

    Two differences from the original 4.1 cell, both deliberate:

    1. Only `var_interest` is read from the parquet and only `var_interest` is
       z-scored. The old code read every column and z-scored the whole matrix.
    2. NaN rows are dropped AFTER selecting `var_interest`. The old code dropped a
       timestep if ANY column was NaN, so whisker and lick fits were gated on right-paw
       tracking -- in the single-camera sessions that discarded up to 98% of usable
       frames.
    """
    filename = os.path.join(data_path, "design_matrix_" + str(session) + '_' + mouse_name)
    original_design_matrix = pd.read_parquet(filename, columns=list(var_interest))

    # mask on the fitted variable(s) only
    design_matrix = original_design_matrix[list(var_interest)].dropna()
    array_matrix = np.array(design_matrix, dtype=float)

    if zsc:
        array_matrix = zscore(array_matrix, axis=0, nan_policy='omit')

    return array_matrix


def prepare_batches(design_matrix, num_train_batches):
    """Split into equal CV folds, exactly as 4.1 did.

    Returns (shortened_array, train_emissions, fold_len). Same variable names as 4.1
    so the arrays are recognisable downstream.
    """
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(
        design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    train_emissions = jnp.stack(jnp.split(shortened_array, num_train_batches))
    fold_len = len(shortened_array) / num_train_batches
    return shortened_array, train_emissions, fold_len


def decorrelation_time(x, max_lag=4000, threshold=1 / np.e):
    """First lag at which the autocorrelation drops below `threshold`, in frames.

    ACF(L) = corr(x(t), x(t+L)). For an exponentially decaying autocorrelation,
    ACF(L) = exp(-L/T), so ACF(L) = 1/e exactly at L = T -- the 1/e crossing recovers
    the decay time constant rather than being an arbitrary cut. Computed by FFT, so it
    costs nothing next to a single EM fit.

    Measured on the 60 Hz whisker cohort: median 34 frames (567 ms), IQR 24-44,
    90th percentile 55, max 293.

    NOT the integral autocorrelation time (1 + 2*sum(ACF)): that is dominated by slow
    session-level drift -- median 99 frames but IQR 54-722 and a max of 3450, which
    would imply AR orders of several thousand. Slow drift is what the STATE should
    explain, not the AR filter.

    Returns np.nan if the ACF never crosses within max_lag, so the caller can fall
    back to a hard cap and flag the session.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 1000:
        return np.nan
    x = (x - x.mean()) / (x.std() + 1e-12)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag].real
    ac /= ac[0]
    below = np.flatnonzero(ac < threshold)
    return float(below[0]) if len(below) else np.nan


def lag_grid_for_session(x, hard_cap=256, fallback_cap=128, start=1):
    """Build this session's lag grid: doubling from `start`, capped near the ACF 1/e crossing.

    The cap is snapped to the NEAREST POINT OF THE DOUBLING GRID, so every grid value is
    a power of two and the last one is a genuine grid point rather than a ragged tail.
    "Nearest" is taken in log2 space because the grid is geometric: the midpoint between
    32 and 64 is sqrt(32*64) = 45.3, not 48. So tau = 34 -> cap 32, tau = 49 -> cap 64.

    The cap is a MODELLING DECISION, not a selection: on data of this size no
    likelihood-based criterion identifies the AR order -- held-out CV, AIC and BIC all
    increase monotonically to whatever the largest order tested is (with N ~ 2e5 frames
    and ~1e2 parameters, a k*ln(N) penalty is ~1e-5 per frame while the likelihood
    differences are ~1e-3). So the ceiling has to come from outside the likelihood. The
    argument used here is that an AR filter longer than the signal's own decorrelation
    time absorbs temporal structure that the latent state should be explaining.

    Doubling is used because the order of magnitude is not known in advance, so equal
    ratios cover more range per fit than equal differences.

    Returns (grid, cap, tau, used_fallback).
    """
    tau = decorrelation_time(x)
    used_fallback = not np.isfinite(tau)
    raw_cap = fallback_cap if used_fallback else max(float(tau), float(start))

    # snap to the nearest power of two, in log2 space, then clip
    cap = int(2 ** int(np.round(np.log2(raw_cap))))
    cap = int(np.clip(cap, start, hard_cap))

    grid = []
    v = int(start)
    while v <= cap:
        grid.append(v)
        v *= 2
    return sorted(set(grid)), cap, (np.nan if used_fallback else float(tau)), used_fallback


# ============================================================================
# THE LAG GRID
# ============================================================================


def increment_pays(prof, i_from, i_to, alpha=0.05):
    """Paired test: is fold-wise LL at i_to reliably better than at i_from?

    prof is (n_lags, n_folds) of PER-FRAME RAW held-out log-likelihood. The folds are the
    same time blocks for every lag, so we test the per-fold DIFFERENCE -- that removes
    the between-fold variance (early vs late session, engagement drift) which otherwise
    dominates the error bar. Using each cell's own across-fold SD instead (as
    find_2_best_param does) inflates the error bar ~14x and collapses the choice onto the
    grid minimum: on this cohort lag 10 beats lag 1 in 275/292 sessions paired, but only
    2/292 unpaired.

    RAW held-out LL, not baseline-subtracted bits. At kappa=0 the CV baseline still
    drifts slightly with lag (0.00076 nats/frame, ~1.7% of the fitted spread) and it
    drifts DOWNWARD, so subtracting it inflates the apparent gain at long lags. That is
    enough to change the selection in 54/323 sessions, systematically toward longer lags.
    """
    diff = np.asarray(prof[i_to]) - np.asarray(prof[i_from])
    ok = np.isfinite(diff)
    n = int(ok.sum())
    if n < 2:
        return False, np.nan, np.nan
    mu = diff[ok].mean()
    se = diff[ok].std(ddof=1) / np.sqrt(n)
    if se == 0:
        return bool(mu > 0), mu, 0.0
    crit = tdist.ppf(1 - alpha / 2, n - 1) * se
    return bool((mu - crit) > 0), float(mu), float(se)


def select_lag(prof, lags, alpha=0.05):
    """Walk up the grid; adopt a longer lag only if the paired increment is significant.

    Consecutive-increment testing, NOT every-cell-against-the-best. The best cell is the
    maximum of several noisy estimates, so its mean is biased upward by selection and
    testing candidates against it drags the choice toward the argmax's lag.

    Skipping is allowed: if the next step does not pay, the following comparison is still
    made against the currently ADOPTED lag, so a larger jump can pay even when each
    single step does not.

    NOTE this is deliberately not the argmax of held-out LL. The two differ in 53% of
    sessions -- argmax sits at the grid ceiling in 95% of them, this rule in 44% --
    because most later steps are not significant. No minimum-gain floor is applied: the
    full per-fold profile is saved, so any floor, alpha or alternative rule can be
    re-derived offline without refitting.

    Returns dict: best_lag, at_cap, lag_profile, steps.
    """
    order = np.argsort(lags)
    lags_sorted = [lags[i] for i in order]
    prof = np.asarray(prof)[order]

    adopted = 0
    steps = []
    for j in range(1, len(lags_sorted)):
        pays, gain, se = increment_pays(prof, adopted, j, alpha=alpha)
        steps.append(dict(lag_from=lags_sorted[adopted], lag_to=lags_sorted[j],
                          gain=gain, se=se, pays=pays))
        if pays:
            adopted = j

    return dict(best_lag=int(lags_sorted[adopted]),
                at_cap=bool(lags_sorted[adopted] == max(lags_sorted)),
                lag_profile={int(l): float(np.nanmean(prof[i]))
                             for i, l in enumerate(lags_sorted)},
                steps=steps)


# ============================================================================
# FITTING
# ============================================================================


def fit_ar_lag(shortened_array, train_emissions, lag, num_states, emission_dim,
               num_train_batches, method, fit_method, kappa=0.0, num_iters=100):
    """Cross-validate one AR-HMM lag. Wraps 4.1's cross_validate_armodel unchanged.

    Returns (raw_ll_per_frame_per_fold, baseline_ll_per_frame_per_fold, fit_params,
             my_inputs). compute_inputs materialises a (T, emission_dim*lag) array,
             the single biggest allocation in the loop, so the caller drops it after
             each lag and rebuilds it once for the selected lag (see run_session).
    """
    my_inputs = compute_inputs(shortened_array, lag, emission_dim)
    train_inputs = jnp.stack(jnp.split(my_inputs, num_train_batches))
    fold_len = len(shortened_array) / num_train_batches

    test_arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=lag,
                                        transition_matrix_stickiness=kappa)
    all_val_lls, fit_params, init_params, baseline_lls = cross_validate_armodel(
        test_arhmm, jr.PRNGKey(0), train_emissions, train_inputs, method,
        num_train_batches, fit_method, num_iters=num_iters)

    return (np.asarray(all_val_lls) / fold_len,
            np.asarray(baseline_lls) / fold_len,
            fit_params, my_inputs)


def fit_poisson(train_emissions, num_states, emission_dim, num_train_batches,
                method, fit_method, kappa=0.0, num_iters=100):
    """Cross-validate the Poisson-HMM (lick). No lag to search, so a single fit.

    NOTE: the lick signal in these design matrices is BINARY (max 1 per bin at both
    30 and 60 Hz), so a Bernoulli emission is the correctly specified model and gains
    ~0.023 bits/frame of raw held-out likelihood. It is left as Poisson here to stay
    comparable with the existing fits -- the state sequences agree to 99%.
    Note also that PoissonHMM.initialize() does not accept `emissions`, so 'kmeans'
    initialisation is not available for this model; `method` must be 'prior'.
    """
    fold_len = train_emissions.shape[1]
    test_poishmm = PoissonHMM(num_states, emission_dim,
                              transition_matrix_stickiness=kappa)
    all_val_lls, fit_params, init_params, baseline_lls = cross_validate_poismodel(
        test_poishmm, jr.PRNGKey(0), train_emissions, num_train_batches, fit_method,
        num_iters=num_iters)
    return (np.asarray(all_val_lls) / fold_len,
            np.asarray(baseline_lls) / fold_len,
            fit_params)


# ============================================================================
# DECODING AND ASSESSMENT
# ============================================================================


def pick_fold(raw_ll):
    """Best fold by held-out likelihood, or None if too many folds failed.

    4.2 chose the fold maximising bits and treated a cell as unusable when at least
    half the folds were NaN; the same rule is kept here on raw LL.
    """
    raw_ll = np.asarray(raw_ll)
    if np.sum(~np.isfinite(raw_ll)) >= len(raw_ll) / 2:
        return None
    return int(np.nanargmax(raw_ll))


def decode_ar_states(shortened_array, my_inputs, lag, num_states, emission_dim,
                     fit_params, use_fold, method, kappa=0.0):
    """Rebuild the best-fold AR-HMM and Viterbi-decode. Same steps as 4.2 cell 6."""
    new_arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=lag,
                                        transition_matrix_stickiness=kappa)
    best_fold_params, props = new_arhmm.initialize(
        key=jr.PRNGKey(0), method=method,
        initial_probs=fit_params[0].probs[use_fold],
        transition_matrix=fit_params[1].transition_matrix[use_fold],
        emission_weights=fit_params[2].weights[use_fold],
        emission_biases=fit_params[2].biases[use_fold],
        emission_covariances=fit_params[2].covs[use_fold],
        emissions=shortened_array)
    return np.asarray(new_arhmm.most_likely_states(
        best_fold_params, shortened_array, my_inputs)), best_fold_params


def decode_poisson_states(shortened_array, num_states, emission_dim, fit_params,
                          use_fold, method, kappa=0.0):
    """Rebuild the best-fold Poisson-HMM and Viterbi-decode. Same steps as 4.2 cell 6."""
    test_phmm = PoissonHMM(num_states, emission_dim,
                           transition_matrix_stickiness=kappa)
    best_fold_params, props = test_phmm.initialize(
        key=jr.PRNGKey(0), method=method,
        initial_probs=fit_params[0].probs[use_fold],
        transition_matrix=fit_params[1].transition_matrix[use_fold],
        emission_rates=fit_params[2].rates[use_fold])
    return np.asarray(test_phmm.most_likely_states(
        best_fold_params, shortened_array)), best_fold_params


def orient_states(most_likely_states, shortened_array):
    """Relabel so state 1 is the higher-amplitude / higher-rate state.

    State identity is arbitrary in an HMM, so without this the labels are not
    comparable across sessions or across fits of the same session.
    """
    s = np.asarray(most_likely_states)
    if s.max() == s.min():
        return s.astype(np.int8)
    hi = shortened_array[s == 1, 0].mean()
    lo = shortened_array[s == 0, 0].mean()
    s = 1 - s if hi < lo else s
    # int8, not the int32 jax hands back: with num_states=2 the sequence is the single
    # biggest thing in the pickle, and 4x smaller for free (680 KB -> 170 KB per session).
    return s.astype(np.int8)


def dwell_times(most_likely_states):
    """Run lengths of the state sequence, in frames."""
    s = np.asarray(most_likely_states)
    changes = np.where(np.diff(s) != 0)[0]
    return np.diff(np.concatenate(([0], changes + 1, [len(s)])))


def assess_fit(most_likely_states, raw_ll, baseline_ll, fps,
               min_dwell_frames=10, target_dwell_ms=None):
    """Quality screens for one fitted session.

    These are independent failure modes -- held-out likelihood does NOT predict a
    degenerate segmentation, so both have to be checked:

      collapsed    the state never really switches (one state for the session)
      flickering   median dwell below `min_dwell_frames`; a state flipping every frame
                   or two carries no behavioural information
      degenerate_occupancy   one state occupies ~everything

    `bits_LL` is reported for continuity with 4.2, but note it is only comparable
    between fits that share a baseline (same model class, same kappa).
    """
    s = np.asarray(most_likely_states)
    d = dwell_times(s)
    med = float(np.median(d))
    occ = float(s.mean())

    out = dict(
        n_segments=int(len(d)),
        median_dwell_frames=med,
        median_dwell_ms=med / fps * 1000.0,
        mean_dwell_frames=float(d.mean()),
        occupancy_state1=round(occ, 4),
        raw_ll_per_frame=float(np.nanmean(raw_ll)),
        bits_LL=float(np.nanmean(np.asarray(raw_ll) - np.asarray(baseline_ll)) / np.log(2)),
        n_folds_failed=int(np.sum(~np.isfinite(np.asarray(raw_ll)))),
        collapsed=bool(len(d) <= 1),
        degenerate_occupancy=bool(occ <= 0.002 or occ >= 0.998),
        flickering=bool(med <= min_dwell_frames),
    )
    if target_dwell_ms is not None:
        out['dwell_vs_target'] = out['median_dwell_ms'] / target_dwell_ms
    out['fit_ok'] = not (out['collapsed'] or out['degenerate_occupancy']
                         or out['flickering'] or out['n_folds_failed'] > 0)
    return out


# ============================================================================
# ONE SESSION, END TO END  (this is 4.1 + 4.2 joined)
# ============================================================================


def run_session(id, var_interest, zsc, num_states, num_train_batches, method,
                fit_method, save_path, data_path, fps, num_iters=100, alpha=0.05,
                hard_cap=256, fallback_cap=128, kappa=0.0, sticky=False,
                extra_lags=(), states_save_path=None):
    """Grid-search, select, decode and assess one session in a single pass.

    Joining 4.1 and 4.2 avoids re-reading the pickle, recomputing the LL table and
    re-initialising the model, which 4.2 did for every session.

    Returns one assessment dict (also the row written to the cohort CSV). Never raises:
    failures are returned with an `error` field instead of being swallowed the way the
    bare `except:` in 4.2 did.
    """
    mouse_name, session = id
    fit_id = str(mouse_name + session)
    result_filename = os.path.join(
        save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_{fit_id}")

    row = dict(mouse=mouse_name, eid=session, var=var_interest[0], error='')
    try:
        design_matrix = load_fit_variable(data_path, session, mouse_name, var_interest, zsc)
        shortened_array, train_emissions, fold_len = prepare_batches(
            design_matrix, num_train_batches)
        emission_dim = np.shape(design_matrix)[1]
        row.update(n_frames=int(len(shortened_array)))

        if var_interest == ['Lick count']:
            # Poisson-HMM: no lag to search, so a single fit.
            raw_ll, base_ll, fit_params = fit_poisson(
                train_emissions, num_states, emission_dim, num_train_batches,
                method, fit_method, kappa=kappa, num_iters=num_iters)
            all_lls = {int(0): raw_ll}
            all_baseline_lls = {int(0): base_ll}
            all_fit_params = {int(0): fit_params}
            best_lag, tau, cap, at_cap, steps, lag_profile = 0, np.nan, 0, False, [], {}
            use_fold = pick_fold(raw_ll)
            if use_fold is None:
                row['error'] = 'all folds NaN'
                return row
            most_likely_states, best_fold_params = decode_poisson_states(
                shortened_array, num_states, emission_dim, fit_params, use_fold,
                method, kappa=kappa)
            my_inputs = None
        else:
            # AR-HMM: doubling grid capped near the signal's decorrelation time.
            grid, cap, tau, used_fallback = lag_grid_for_session(
                shortened_array[:, 0], hard_cap=hard_cap, fallback_cap=fallback_cap)
            grid = sorted(set(list(grid) + [int(l) for l in extra_lags]))
            row.update(tau=tau, cap=int(cap), used_tau_fallback=bool(used_fallback),
                       grid=str(grid))

            all_lls, all_baseline_lls, all_fit_params = {}, {}, {}
            for lag in grid:
                raw_ll, base_ll, fit_params, my_inputs = fit_ar_lag(
                    shortened_array, train_emissions, lag, num_states, emission_dim,
                    num_train_batches, method, fit_method, kappa=kappa,
                    num_iters=num_iters)
                all_lls[int(lag)] = raw_ll
                all_baseline_lls[int(lag)] = base_ll
                all_fit_params[int(lag)] = fit_params
                del my_inputs      # see below: kept for every lag, this was ~0.1-0.4 GB
                                   # of live JAX arrays per worker, growing with the cap

            prof = np.vstack([all_lls[int(l)] for l in grid])
            sel = select_lag(prof, list(grid), alpha=alpha)
            best_lag, at_cap = sel['best_lag'], sel['at_cap']
            steps, lag_profile = sel['steps'], sel['lag_profile']

            use_fold = pick_fold(all_lls[int(best_lag)])
            if use_fold is None:
                row['error'] = f'all folds NaN at selected lag {best_lag}'
                return row
            # Rebuilt for the selected lag only. compute_inputs is a column_stack of
            # slices (milliseconds), so recomputing it once is far cheaper than holding
            # one (T x lag) array per grid cell alive until selection is finished.
            my_inputs = compute_inputs(shortened_array, best_lag, emission_dim)
            most_likely_states, best_fold_params = decode_ar_states(
                shortened_array, my_inputs, best_lag, num_states, emission_dim,
                all_fit_params[int(best_lag)], use_fold, method, kappa=kappa)

        most_likely_states = orient_states(most_likely_states, shortened_array)
        assessment = assess_fit(most_likely_states, all_lls[int(best_lag)],
                                all_baseline_lls[int(best_lag)], fps)
        row.update(assessment)
        row.update(best_lag=int(best_lag), at_cap=bool(at_cap), use_fold=int(use_fold))

        # `design_matrix` is NOT stored: it was 1.24 MB of the old 1.25 MB pickle and is
        # already on disk as parquet. A fingerprint keeps the provenance check (is this
        # fit still consistent with the current design matrix?) at ~zero cost.
        fingerprint = dict(n_rows=int(len(shortened_array)),
                           sha1=hashlib.sha1(np.ascontiguousarray(
                               shortened_array).tobytes()).hexdigest())

        best_parameters = num_states, best_lag, kappa
        to_save = dict(
            all_lls=all_lls,                       # per-fold RAW held-out LL per frame
            all_baseline_lls=all_baseline_lls,
            all_fit_params=all_fit_params,          # every grid cell -- 0.01 MB, and what
                                                    # makes any later re-analysis possible
            most_likely_states=most_likely_states,
            use_fold=use_fold,
            best_parameters=best_parameters,
            best_lag=best_lag,
            lag_profile=lag_profile,
            selection_steps=steps,
            tau=tau, cap=cap, at_cap=at_cap,
            assessment=assessment,
            fingerprint=fingerprint,
            config=dict(var_interest=var_interest, zsc=zsc, num_states=num_states,
                        num_train_batches=num_train_batches, method=method,
                        fit_method=fit_method, kappa=kappa, num_iters=num_iters,
                        alpha=alpha, fps=fps),
        )
        atomic_dump(to_save, result_filename)

        # ---- ALSO write 4.2's exact output format, so downstream code is unchanged ----
        # 5_syllable_generation.ipynb (and the other consumers) do:
        #     most_likely_states, _, _ = pickle.load(open(states_filename, "rb"))
        # i.e. a 3-tuple at  <states_save_path>/<var_interest[0]>_<fit_id>
        # `best_parameters` keeps 4.2's shape: (num_states, lag, kappa), with lag = np.nan
        # for the Poisson model exactly as 4.2 did.
        if states_save_path is not None:
            if not os.path.exists(states_save_path):
                os.makedirs(states_save_path, exist_ok=True)
            states_filename = os.path.join(states_save_path,
                                          var_interest[0] + '_' + fit_id)
            legacy_parameters = (num_states,
                                 np.nan if var_interest == ['Lick count'] else best_lag,
                                 kappa)
            atomic_dump((most_likely_states, use_fold, legacy_parameters), states_filename)

        del to_save, all_lls, all_baseline_lls, all_fit_params
        gc.collect()

    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def atomic_dump(obj, path):
    """Pickle to `path` so that the file is either absent or complete — never partial.

    Why this matters: `open(path, "wb")` creates and truncates the file IMMEDIATELY, and
    it only gains content once `pickle.dump` finishes. A worker killed in between (kernel
    interrupt, OOM kill) therefore leaves a **0-byte file** behind. That is worse than no
    file at all, because `run_all` skips a session whose output already exists, so the
    dead file silently and permanently excludes that session from the cohort — and
    `assessments_from_pickles` cannot read it either.

    Writing to a temporary name and renaming fixes it: `os.replace` is atomic within one
    filesystem, so a kill either leaves the untouched previous state plus a stray `.tmp`
    (harmless, and overwritten next time) or the finished file.
    """
    tmp = path + '.tmp'
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())     # so the rename cannot land before the bytes do
    os.replace(tmp, path)


def is_complete(path):
    """Does `path` hold a usable result? Used instead of a bare `os.path.exists` so an
    interrupted write (0 bytes) is retried rather than mistaken for finished work.

    Only the size is checked, not the whole unpickling: with `atomic_dump` a present file
    is complete by construction, so this just has to catch the pre-atomic leftovers.
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def assessments_from_pickles(save_path, var_interest, sticky=False):
    """Rebuild the whole cohort assessment table by reading the saved pickles.

    Needed because the run is resumable: `run_all` only ever has rows for the sessions
    THAT call fitted, so if a run is interrupted (kernel restart, OOM) the sessions
    already on disk are skipped next time and would never get a CSV row. Every field in
    the row is recoverable from the pickle, so the table is rebuilt from disk rather
    than appended to.

    Sessions that errored have no pickle and so are absent here -- which is right: they
    are also the ones `run_all` will retry.
    """
    pre = f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_"
    rows, unreadable = [], []
    for name in sorted(os.listdir(save_path)):
        if not name.startswith(pre) or name.endswith('.tmp'):
            continue
        rest = name[len(pre):]
        mouse_name, session = rest[:-36], rest[-36:]   # eid is the trailing 36 chars
        try:
            with open(os.path.join(save_path, name), 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            # one dead file must not take the whole table down with it
            unreadable.append((name, f'{type(e).__name__}: {e}'))
            continue

        row = dict(mouse=mouse_name, eid=session, var=var_interest[0], error='')
        row.update(n_frames=int(d['fingerprint']['n_rows']))
        if d['cap']:                       # AR-HMM; the Poisson fit has no lag grid
            # tau is stored as NaN exactly when lag_grid_for_session fell back, so the
            # fallback flag is recoverable rather than needing to be stored separately.
            row.update(tau=d['tau'], cap=int(d['cap']),
                       used_tau_fallback=bool(not np.isfinite(d['tau'])),
                       grid=str(sorted(int(l) for l in d['all_lls'])))
        row.update(d['assessment'])
        row.update(best_lag=int(d['best_lag']), at_cap=bool(d['at_cap']),
                   use_fold=int(d['use_fold']))
        rows.append(row)

    if unreadable:
        print(f"WARNING: {len(unreadable)} unreadable result file(s) — delete them and "
              f"rerun the launcher; they are being skipped, NOT refitted:")
        for name, err in unreadable:
            print(f"    {name}  ({err})")
    return pd.DataFrame(rows)


def run_all(idxs, var_interest, zsc, num_states, num_train_batches, method, fit_method,
            save_path, data_path, fps, n_jobs=1, csv_path=None,
            states_save_path=None, sticky=False, **kw):
    """Same skip-if-exists / parallel-or-serial structure as run_grid_search_* in 4.1."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sessions_to_process, revived = [], 0
    for m, mat in enumerate(idxs):
        mouse_name = mat[37:]
        session = mat[:36]
        fit_id = str(mouse_name + session)
        result_filename = os.path.join(
            save_path, f"{'best_sticky' if sticky else 'best'}_results_{var_interest[0]}_{fit_id}")
        # CHANGED: `is_complete` rather than `os.path.exists`. A 0-byte file left by an
        # interrupted write used to count as finished work, so the session was skipped
        # forever and vanished from the cohort without any warning.
        if not is_complete(result_filename):
            if os.path.exists(result_filename):
                revived += 1
            sessions_to_process.append((mouse_name, session))
    print(f"Found {len(sessions_to_process)} sessions to process.")
    if revived:
        print(f"  ({revived} of them had an empty/incomplete result file and will be refitted)")

    args = dict(var_interest=var_interest, zsc=zsc, num_states=num_states,
                num_train_batches=num_train_batches, method=method,
                fit_method=fit_method, save_path=save_path, data_path=data_path,
                fps=fps, states_save_path=states_save_path, sticky=sticky, **kw)
    if n_jobs and n_jobs > 1:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(run_session)(id, **args) for id in sessions_to_process)
    else:
        rows = [run_session(id, **args) for id in sessions_to_process]

    fitted_now = pd.DataFrame([r for r in rows if r is not None])

    # CHANGED: the CSV is rebuilt from every pickle in save_path, not appended from just
    # this call. Appending lost a row for every session that a previous, interrupted run
    # had already finished (they get skipped above, so they never reappear). Rebuilding
    # is also idempotent -- rerunning the launcher cannot duplicate rows.
    if csv_path is not None:
        table = assessments_from_pickles(save_path, var_interest, sticky=sticky)
        # sessions that raised have no pickle; keep their error rows from this call
        errs = fitted_now[fitted_now.error != ''] if len(fitted_now) else fitted_now
        if len(errs):
            table = pd.concat([table, errs], ignore_index=True)
        table.to_csv(csv_path, index=False)
        print(f"{len(fitted_now)} fitted this call; wrote {len(table)} rows -> {csv_path}")
    return fitted_now
