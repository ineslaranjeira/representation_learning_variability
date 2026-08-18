"""
HMM fitting for behavioural segmentation — Gaussian for whisking, Poisson for licking.

Self-contained on purpose: this module imports only numpy/pandas/jax/dynamax, so the
5.x notebooks do not depend on any of the exploratory 4.0.x files and those can be
deleted freely.

WHY THERE IS NO HYPERPARAMETER SEARCH
-------------------------------------
The AR-HMM needed a lag, and choosing it was the whole problem: held-out likelihood rises
monotonically with lag at this data size (no criterion -- CV, AIC or BIC -- identifies the
order), while the SEGMENTATION converges by lag ~16 and syllable duration drifts away from
the model-free changepoint duration (~450 ms). Dropping the autoregressive emission
removes that axis entirely: a Gaussian HMM is the lag-0 limit, and the hypothesis it
encodes -- whisking is high or low -- is the thing actually wanted downstream.

What is left was measured, one knob at a time, on sessions spanning the good and the
pathological cases (hmm_diagnostics/gaussian_hyperparams.csv). Frame agreement against
the default fit:

    transition_matrix_stickiness (kappa) = 1e3    1.0000   (bit-identical)
    kappa = 1e5                                   0.9973   (100x past anything defensible)
    method = 'prior' instead of 'kmeans'          1.0000
    emission_prior_scale         x 1e2            1.0000
    emission_prior_scale         x 1e4            1.0000
    emission_prior_concentration x 1e4            1.0000
    transition_matrix_concentration = 10          1.0000

Every prior and the initialisation are irrelevant -- not "small", identical. So they are
left at their defaults and not searched.

`num_states` is the one knob that does change the answer (2 -> 3 states moves median dwell
from 567 ms to 200 ms), and it is deliberately NOT selected by cross-validation: held-out
LL keeps improving as states are added (-0.154 -> +0.043 -> +0.113 nats/frame for 2/3/4)
while dwell collapses toward frame-by-frame flicker. That is the same non-convergence that
made lag selection undefendable. `num_states = 2` is therefore a stated modelling
commitment -- high vs low -- not a fitted quantity.

WHAT IS SAVED
-------------
Per session, two files:
  <save_path>/best_results_<var>_<mouse><eid>   the full record (per-fold LLs, fitted
      params, decoded states, assessment, config, data fingerprint)
  <states_save_path>/<var>_<mouse><eid>         4.2's exact 3-tuple
      (most_likely_states, use_fold, (num_states, nan, kappa)), so 5_syllable_generation
      and the other consumers need no change.
Writes are atomic, so an interrupted run leaves no half-written file to be mistaken for
finished work.
"""

import os
import gc
import pickle
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import zscore
from joblib import Parallel, delayed

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model import GaussianHMM, PoissonHMM, BernoulliHMM


# ============================================================================
# MODEL REGISTRY
# ============================================================================
# Each emission family differs in three ways that matter here: the names of its emission
# parameters (needed to rebuild the best fold for decoding), whether `initialize` accepts
# `emissions` (so whether kmeans is available at all), and whether the data must be
# binarised first.

MODELS = {
    'gaussian': dict(
        cls=GaussianHMM,
        # (attribute on the fitted params, keyword for initialize)
        emission_params=[('means', 'emission_means'), ('covs', 'emission_covariances')],
        supports_kmeans=True,      # GaussianHMM.initialize takes `emissions`
        binarise=False,
    ),
    'poisson': dict(
        cls=PoissonHMM,
        emission_params=[('rates', 'emission_rates')],
        supports_kmeans=False,     # PoissonHMM.initialize has no `emissions` argument
        binarise=False,
    ),
    'bernoulli': dict(
        cls=BernoulliHMM,
        emission_params=[('probs', 'emission_probs')],
        supports_kmeans=False,
        binarise=True,             # counts -> 0/1; at 30 Hz a frame can hold 2 licks
    ),
}


def make_model(model, num_states, emission_dim, kappa=0.0):
    """Construct one HMM. Priors are left at their defaults: measured irrelevant."""
    if model not in MODELS:
        raise ValueError(f'unknown model {model!r}; use one of {sorted(MODELS)}')
    return MODELS[model]['cls'](num_states, emission_dim,
                               transition_matrix_stickiness=kappa)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_fit_variable(data_path, session, mouse_name, var_interest, zsc, binarise=False):
    """Read ONLY the fitted column(s) and drop NaN rows on those columns alone.

    Reading just `var_interest` matters: a NaN in an unrelated column (an untracked right
    paw, say) must not drop the bin, since the fit does not depend on it. Dropping after
    selecting is what makes the row set correct.
    """
    filename = os.path.join(data_path, "design_matrix_" + str(session) + '_' + mouse_name)
    original_design_matrix = pd.read_parquet(filename, columns=list(var_interest))
    design_matrix = original_design_matrix[list(var_interest)].dropna()
    array_matrix = np.array(design_matrix, dtype=float)
    if binarise:
        # before z-scoring, and z-scoring a 0/1 variable would defeat the point
        array_matrix = (array_matrix > 0).astype(float)
    elif zsc:
        array_matrix = zscore(array_matrix, axis=0, nan_policy='omit')
    return array_matrix


def prepare_batches(design_matrix, num_train_batches):
    """Split into equal contiguous CV folds. Returns (shortened_array, train_emissions,
    fold_len); the array is truncated so the folds divide it exactly."""
    num_timesteps = np.shape(design_matrix)[0]
    shortened_array = np.array(
        design_matrix[:(num_timesteps // num_train_batches) * num_train_batches])
    train_emissions = jnp.stack(jnp.split(shortened_array, num_train_batches))
    fold_len = len(shortened_array) / num_train_batches
    return shortened_array, train_emissions, fold_len


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validate_hmm(model, key, train_emissions, num_train_batches, fit_method,
                       method='prior', num_iters=100, return_train_lps=False):
    """Leave-one-fold-out CV over contiguous blocks, for any emission family here.

    Same structure as the original `cross_validate_poismodel`: hold out fold i, fit on the
    rest, score the held-out fold. The baseline is the same model left UNFITTED, which is
    what `bits_LL` is measured against.

    `train_lps` is returned behind a flag so EM convergence can be checked rather than
    assumed (it used to be computed and thrown away).
    """
    if method == 'kmeans':
        # kmeans needs the data; only some families accept it
        flat = jnp.concatenate([train_emissions[i] for i in range(num_train_batches)])
        init_params, props = model.initialize(key=key, method='kmeans', emissions=flat)
    else:
        init_params, props = model.initialize(key=key, method=method)

    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])

    def _fit_fold_baseline(y_train, y_val):
        return model.marginal_log_prob(init_params, y_val)

    if fit_method == 'em':
        def _fit_fold(y_train, y_val):
            fit_params, train_lps = model.fit_em(init_params, props, y_train,
                                                 num_iters=num_iters, verbose=False)
            return model.marginal_log_prob(fit_params, y_val), fit_params, train_lps
    elif fit_method == 'sgd':
        def _fit_fold(y_train, y_val):
            fit_params, train_lps = model.fit_sgd(init_params, props, y_train,
                                                  num_epochs=num_iters)
            return model.marginal_log_prob(fit_params, y_val), fit_params, train_lps
    else:
        raise ValueError(f'unknown fit_method {fit_method!r}')

    val_lls, fit_params, train_lps = vmap(_fit_fold)(folds, train_emissions)
    baseline_val_lls = vmap(_fit_fold_baseline)(folds, train_emissions)

    if return_train_lps:
        return val_lls, fit_params, init_params, baseline_val_lls, train_lps
    return val_lls, fit_params, init_params, baseline_val_lls


def fit_hmm(train_emissions, model, num_states, emission_dim, num_train_batches,
            method, fit_method, kappa=0.0, num_iters=100):
    """One cross-validated fit. Returns per-frame per-fold (raw LL, baseline LL, params).

    Dividing by fold length up front means every likelihood in this pipeline is per frame,
    so sessions of different lengths are directly comparable.
    """
    spec = MODELS[model]
    if method == 'kmeans' and not spec['supports_kmeans']:
        raise ValueError(f"method='kmeans' is unavailable for the {model} model "
                         f"({spec['cls'].__name__}.initialize takes no `emissions`); "
                         f"use method='prior'")
    hmm = make_model(model, num_states, emission_dim, kappa)
    all_val_lls, fit_params, init_params, baseline_lls = cross_validate_hmm(
        hmm, jr.PRNGKey(0), train_emissions, num_train_batches, fit_method,
        method=method, num_iters=num_iters)
    fold_len = train_emissions.shape[1]
    return (np.asarray(all_val_lls) / fold_len,
            np.asarray(baseline_lls) / fold_len,
            fit_params)


def pick_fold(raw_ll):
    """Index of the best-scoring fold, or None if every fold failed."""
    raw_ll = np.asarray(raw_ll, dtype=float)
    return None if not np.isfinite(raw_ll).any() else int(np.nanargmax(raw_ll))


def decode_states(shortened_array, model, num_states, emission_dim, fit_params, use_fold,
                  kappa=0.0):
    """Rebuild the best fold's model and Viterbi-decode the whole session.

    Initialisation is by 'prior' with every parameter supplied explicitly, so the prior is
    overwritten and only the fitted values are used.
    """
    spec = MODELS[model]
    hmm = make_model(model, num_states, emission_dim, kappa)
    kwargs = {kw: getattr(fit_params[2], attr)[use_fold]
              for attr, kw in spec['emission_params']}
    best_fold_params, _ = hmm.initialize(
        key=jr.PRNGKey(0), method='prior',
        initial_probs=fit_params[0].probs[use_fold],
        transition_matrix=np.asarray(fit_params[1].transition_matrix)[use_fold],
        **kwargs)
    return np.asarray(hmm.most_likely_states(best_fold_params, shortened_array)), \
        best_fold_params


# ============================================================================
# ASSESSMENT
# ============================================================================

def orient_states(most_likely_states, shortened_array):
    """Relabel so state 1 is always the HIGH state (larger mean signal).

    Without this the label is arbitrary per fit, and nothing downstream -- nor any
    comparison between fits -- can be pooled. int8 keeps the pickle small.
    """
    s = np.asarray(most_likely_states)
    x = np.asarray(shortened_array)[:, 0]
    if s.max() == 1 and np.nanmean(x[s == 1]) < np.nanmean(x[s == 0]):
        s = 1 - s
    return s.astype(np.int8)


def dwell_times(most_likely_states):
    """Length in frames of every run of a constant state."""
    s = np.asarray(most_likely_states)
    change = np.flatnonzero(np.diff(s) != 0)
    edges = np.concatenate(([-1], change, [len(s) - 1]))
    return np.diff(edges)


def assess_fit(most_likely_states, raw_ll, baseline_ll, fps, min_dwell_ms=167.,
               occupancy_limits=(0.02, 0.98)):
    """Quality screens. Three INDEPENDENT failure modes -- held-out likelihood does not
    predict a degenerate segmentation, so all of them have to be checked separately.

    `min_dwell_ms` is in milliseconds, not frames: 10 frames means 167 ms at 60 Hz but
    333 ms at 30 Hz, so a frame-based screen silently means different things in the two
    cohorts.
    """
    s = np.asarray(most_likely_states)
    dw = dwell_times(s)
    raw_ll = np.asarray(raw_ll, dtype=float)
    baseline_ll = np.asarray(baseline_ll, dtype=float)
    med_frames = float(np.median(dw))
    occ = float(np.mean(s == 1))
    lo, hi = occupancy_limits
    out = dict(
        n_segments=int(len(dw)),
        median_dwell_frames=med_frames,
        median_dwell_ms=med_frames * 1000. / fps,
        mean_dwell_frames=float(np.mean(dw)),
        occupancy_state1=round(occ, 4),
        raw_ll_per_frame=float(np.nanmean(raw_ll)),
        bits_LL=float(np.nanmean(raw_ll - baseline_ll) * np.log(2)),
        n_folds_failed=int(np.sum(~np.isfinite(raw_ll))),
    )
    out['collapsed'] = bool(out['n_segments'] <= 1)
    out['degenerate_occupancy'] = bool(occ < lo or occ > hi)
    out['flickering'] = bool(out['median_dwell_ms'] <= min_dwell_ms)
    out['fit_ok'] = bool(not (out['collapsed'] or out['degenerate_occupancy']
                              or out['flickering'] or out['n_folds_failed'] > 0))
    return out


# ============================================================================
# SAVING
# ============================================================================

def atomic_dump(obj, path):
    """Pickle so the file is either absent or complete, never partial.

    `open(path, "wb")` truncates immediately and the content only arrives when
    `pickle.dump` finishes, so a worker killed in between leaves a 0-byte file -- which is
    worse than nothing, because the runner would treat it as finished work and skip that
    session forever. Writing to a temporary name and renaming avoids the window entirely.
    """
    tmp = path + '.tmp'
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def is_complete(path):
    """Whether `path` holds usable output. Used instead of a bare `os.path.exists` so a
    0-byte file from an interrupted write is retried rather than counted as done."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def result_path(save_path, var, mouse_name, session):
    return os.path.join(save_path, f"best_results_{var}_{mouse_name}{session}")


# ============================================================================
# ONE SESSION / ALL SESSIONS
# ============================================================================

def run_session(id, var_interest, model, zsc, num_states, num_train_batches, method,
                fit_method, save_path, data_path, fps, num_iters=100, kappa=0.0,
                min_dwell_ms=167., states_save_path=None):
    """Fit, decode and assess one session.

    Never raises: a failure is returned in the row's `error` field instead of being
    swallowed, so a bad session is visible in the assessment table rather than silently
    missing from the output directory.
    """
    mouse_name, session = id
    fit_id = str(mouse_name + session)
    filename = result_path(save_path, var_interest[0], mouse_name, session)

    row = dict(mouse=mouse_name, eid=session, var=var_interest[0], model=model, error='')
    try:
        design_matrix = load_fit_variable(data_path, session, mouse_name, var_interest,
                                          zsc, binarise=MODELS[model]['binarise'])
        shortened_array, train_emissions, fold_len = prepare_batches(
            design_matrix, num_train_batches)
        emission_dim = np.shape(design_matrix)[1]
        row.update(n_frames=int(len(shortened_array)))

        raw_ll, base_ll, fit_params = fit_hmm(
            train_emissions, model, num_states, emission_dim, num_train_batches,
            method, fit_method, kappa=kappa, num_iters=num_iters)

        use_fold = pick_fold(raw_ll)
        if use_fold is None:
            row['error'] = 'all folds NaN'
            return row

        most_likely_states, best_fold_params = decode_states(
            shortened_array, model, num_states, emission_dim, fit_params, use_fold,
            kappa=kappa)
        most_likely_states = orient_states(most_likely_states, shortened_array)

        assessment = assess_fit(most_likely_states, raw_ll, base_ll, fps,
                                min_dwell_ms=min_dwell_ms)
        row.update(assessment)
        row.update(use_fold=int(use_fold))
        # the fitted emission parameters, per state, low state first -- for a Gaussian fit
        # these two numbers ARE the model's claim about high vs low
        attr = MODELS[model]['emission_params'][0][0]
        levels = np.sort(np.asarray(getattr(fit_params[2], attr))[use_fold].ravel())
        row.update(level_low=float(levels[0]), level_high=float(levels[-1]))

        # the design matrix is NOT stored (it is already on disk as parquet); a
        # fingerprint keeps the provenance check at no cost
        fingerprint = dict(n_rows=int(len(shortened_array)),
                           sha1=hashlib.sha1(np.ascontiguousarray(
                               shortened_array).tobytes()).hexdigest())
        to_save = dict(
            all_lls=raw_ll, all_baseline_lls=base_ll, fit_params=fit_params,
            most_likely_states=most_likely_states, use_fold=use_fold,
            emission_levels=levels, assessment=assessment, fingerprint=fingerprint,
            config=dict(var_interest=var_interest, model=model, zsc=zsc,
                        num_states=num_states, num_train_batches=num_train_batches,
                        method=method, fit_method=fit_method, kappa=kappa,
                        num_iters=num_iters, fps=fps, min_dwell_ms=min_dwell_ms),
        )
        atomic_dump(to_save, filename)

        # 4.2's exact output format, so downstream notebooks are unchanged:
        #     most_likely_states, _, _ = pickle.load(open(states_filename, "rb"))
        if states_save_path is not None:
            os.makedirs(states_save_path, exist_ok=True)
            atomic_dump((most_likely_states, use_fold, (num_states, np.nan, kappa)),
                        os.path.join(states_save_path, var_interest[0] + '_' + fit_id))

        del to_save
        gc.collect()
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def assessments_from_pickles(save_path, var):
    """Rebuild the whole assessment table from the saved pickles.

    Needed because the run is resumable: a given call only knows about the sessions IT
    fitted, so after an interruption the sessions already on disk are skipped and would
    never get a row. Rebuilding from disk is also idempotent.
    """
    pre = f'best_results_{var}_'
    rows, unreadable = [], []
    for name in sorted(os.listdir(save_path)):
        if not name.startswith(pre) or name.endswith('.tmp'):
            continue
        rest = name[len(pre):]
        mouse_name, session = rest[:-36], rest[-36:]      # eid is the trailing 36 chars
        try:
            with open(os.path.join(save_path, name), 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            unreadable.append((name, f'{type(e).__name__}: {e}'))
            continue
        row = dict(mouse=mouse_name, eid=session, var=var,
                   model=d['config']['model'], error='')
        row.update(n_frames=int(d['fingerprint']['n_rows']))
        row.update(d['assessment'])
        row.update(use_fold=int(d['use_fold']),
                   level_low=float(np.min(d['emission_levels'])),
                   level_high=float(np.max(d['emission_levels'])))
        rows.append(row)
    if unreadable:
        print(f'WARNING: {len(unreadable)} unreadable result file(s); delete them and '
              f'rerun the launcher -- they are skipped, NOT refitted:')
        for n, e in unreadable:
            print(f'    {n}  ({e})')
    return pd.DataFrame(rows)


def run_all(idxs, var_interest, model, zsc, num_states, num_train_batches, method,
            fit_method, save_path, data_path, fps, n_jobs=1, csv_path=None,
            states_save_path=None, **kw):
    """Fit every session that does not already have complete output."""
    os.makedirs(save_path, exist_ok=True)

    todo, revived = [], 0
    for mat in idxs:
        mouse_name, session = mat[37:], mat[:36]
        fn = result_path(save_path, var_interest[0], mouse_name, session)
        if not is_complete(fn):
            if os.path.exists(fn):
                revived += 1
            todo.append((mouse_name, session))
    print(f'Found {len(todo)} sessions to process.')
    if revived:
        print(f'  ({revived} had an empty/incomplete file and will be refitted)')

    args = dict(var_interest=var_interest, model=model, zsc=zsc, num_states=num_states,
                num_train_batches=num_train_batches, method=method,
                fit_method=fit_method, save_path=save_path, data_path=data_path,
                fps=fps, states_save_path=states_save_path, **kw)
    if n_jobs and n_jobs > 1:
        rows = Parallel(n_jobs=n_jobs)(delayed(run_session)(i, **args) for i in todo)
    else:
        rows = [run_session(i, **args) for i in todo]

    fitted_now = pd.DataFrame([r for r in rows if r is not None])
    if csv_path is not None:
        table = assessments_from_pickles(save_path, var_interest[0])
        errs = fitted_now[fitted_now.error != ''] if len(fitted_now) else fitted_now
        if len(errs):
            table = pd.concat([table, errs], ignore_index=True)
        table.to_csv(csv_path, index=False)
        print(f'{len(fitted_now)} fitted this call; wrote {len(table)} rows -> {csv_path}')
    return fitted_now
