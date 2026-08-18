"""
Plain 2-state GAUSSIAN HMM on whisker motion energy — no AR filter at all.

The idea being tested: whisking is high or low, and that is the whole latent structure.
The within-state dynamics may well differ between the two states, but if the STATE is
just a level then an autoregressive emission is not needed to find it.

Why this is worth trying rather than another AR variant:

  * there is NO lag to select. The entire lag question -- the grid, its cap, paired vs
    unpaired tests, bits vs raw LL -- simply does not arise. A Gaussian HMM is the lag-0
    limit of the AR-HMM.
  * the AR-HMM's held-out likelihood rises with lag while the SEGMENTATION converges by
    lag ~16 and syllable duration drifts away from the model-free changepoint anchor.
    That is the signature of a filter earning likelihood without earning structure.
  * `kmeans` initialisation is available here (`GaussianHMM.initialize` takes
    `emissions=`, unlike `PoissonHMM`), and a 2-state high/low split is exactly what
    kmeans finds -- so initialisation is well posed rather than a prior draw.

Everything is saved in the same format as `hmm_dynamic_functions.run_session`, with the
lag fields left neutral (best_lag=0, empty lag_profile), so `4.0.1_hmm_dynamic_inspect`
and `hmm_dynamic_plots` work on these fits unchanged.
"""

import os
import gc
import pickle
import hashlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from dynamax.hidden_markov_model import GaussianHMM

# the shared machinery: identical data preparation, orientation and assessment as the
# AR pipeline, so the two are directly comparable frame for frame
from hmm_dynamic_functions import (load_fit_variable, prepare_batches, orient_states,
                                   assess_fit, pick_fold, dwell_times, atomic_dump,
                                   is_complete)


# ============================================================================
# CROSS-VALIDATION  (mirrors cross_validate_poismodel, same structure and names)
# ============================================================================

def cross_validate_gaussmodel(model, key, train_emissions, num_train_batches, fit_method,
                              method='kmeans', num_iters=100, return_train_lps=False):
    """Same 5-fold structure as `cross_validate_poismodel`, for a Gaussian HMM.

    Differences from the Poisson version, both deliberate:
      * `method` is a real argument. GaussianHMM.initialize accepts `emissions`, so
        'kmeans' is available; the Poisson model had no such option, which is why that
        function hard-codes the default.
      * `train_lps` is returned behind a flag, as in the other two CV functions, so EM
        convergence can be checked instead of assumed.
    """
    if method == 'kmeans':
        # kmeans needs the data; use the whole training set, as the docstring of the
        # Poisson version claims (that one actually falls back to the prior)
        flat = jnp.concatenate([train_emissions[i] for i in range(num_train_batches)])
        init_params, props = model.initialize(key=key, method='kmeans', emissions=flat)
    else:
        init_params, props = model.initialize(key=key, method=method)

    # Split the training data into folds.
    folds = jnp.stack([
        jnp.concatenate([train_emissions[:i], train_emissions[i+1:]])
        for i in range(num_train_batches)])

    # Baseline: same number of states, unfitted initialisation
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


# ============================================================================
# FIT AND DECODE
# ============================================================================

def fit_gaussian(train_emissions, num_states, emission_dim, num_train_batches, method,
                 fit_method, kappa=0.0, num_iters=100):
    """One cross-validated Gaussian-HMM fit. Returns per-frame per-fold LLs."""
    test_gausshmm = GaussianHMM(num_states, emission_dim,
                                transition_matrix_stickiness=kappa)
    all_val_lls, fit_params, init_params, baseline_lls = cross_validate_gaussmodel(
        test_gausshmm, jr.PRNGKey(0), train_emissions, num_train_batches, fit_method,
        method=method, num_iters=num_iters)
    fold_len = train_emissions.shape[1]
    return (np.asarray(all_val_lls) / fold_len,
            np.asarray(baseline_lls) / fold_len,
            fit_params)


def decode_gaussian_states(shortened_array, num_states, emission_dim, fit_params,
                           use_fold, method, kappa=0.0):
    """Rebuild the best-fold model and Viterbi-decode. Same shape as the AR/Poisson
    decoders in hmm_dynamic_functions, so the outputs are interchangeable."""
    new_gausshmm = GaussianHMM(num_states, emission_dim,
                               transition_matrix_stickiness=kappa)
    best_fold_params, props = new_gausshmm.initialize(
        key=jr.PRNGKey(0), method='prior',
        initial_probs=fit_params[0].probs[use_fold],
        transition_matrix=fit_params[1].transition_matrix[use_fold],
        emission_means=fit_params[2].means[use_fold],
        emission_covariances=fit_params[2].covs[use_fold])
    return np.asarray(new_gausshmm.most_likely_states(
        best_fold_params, shortened_array)), best_fold_params


# ============================================================================
# ONE SESSION / ALL SESSIONS
# ============================================================================

def run_session_gaussian(id, var_interest, zsc, num_states, num_train_batches, method,
                         fit_method, save_path, data_path, fps, num_iters=100,
                         kappa=0.0, states_save_path=None):
    """Fit, decode and assess one session. Never raises; errors land in the row.

    Saves the SAME dict layout as hmm_dynamic_functions.run_session with the lag fields
    neutral, so the existing viewer and assessment table work without modification.
    """
    mouse_name, session = id
    fit_id = str(mouse_name + session)
    result_filename = os.path.join(save_path,
                                   f"best_results_{var_interest[0]}_{fit_id}")
    row = dict(mouse=mouse_name, eid=session, var=var_interest[0], error='')
    try:
        design_matrix = load_fit_variable(data_path, session, mouse_name, var_interest, zsc)
        shortened_array, train_emissions, fold_len = prepare_batches(
            design_matrix, num_train_batches)
        emission_dim = np.shape(design_matrix)[1]
        row.update(n_frames=int(len(shortened_array)))

        raw_ll, base_ll, fit_params = fit_gaussian(
            train_emissions, num_states, emission_dim, num_train_batches, method,
            fit_method, kappa=kappa, num_iters=num_iters)

        use_fold = pick_fold(raw_ll)
        if use_fold is None:
            row['error'] = 'all folds NaN'
            return row
        most_likely_states, best_fold_params = decode_gaussian_states(
            shortened_array, num_states, emission_dim, fit_params, use_fold, method,
            kappa=kappa)
        most_likely_states = orient_states(most_likely_states, shortened_array)

        assessment = assess_fit(most_likely_states, raw_ll, base_ll, fps)
        row.update(assessment)
        row.update(best_lag=0, at_cap=False, use_fold=int(use_fold))
        # the two fitted state means, in z units -- the thing the model is actually about
        means = np.asarray(fit_params[2].means)[use_fold].ravel()
        row.update(mean_low=float(np.min(means)), mean_high=float(np.max(means)))

        fingerprint = dict(n_rows=int(len(shortened_array)),
                           sha1=hashlib.sha1(np.ascontiguousarray(
                               shortened_array).tobytes()).hexdigest())
        to_save = dict(
            all_lls={0: raw_ll}, all_baseline_lls={0: base_ll},
            all_fit_params={0: fit_params},
            most_likely_states=most_likely_states, use_fold=use_fold,
            best_parameters=(num_states, 0, kappa),
            best_lag=0, lag_profile={}, selection_steps=[],
            tau=np.nan, cap=0, at_cap=False,
            assessment=assessment, fingerprint=fingerprint,
            emission_means=means,
            config=dict(var_interest=var_interest, zsc=zsc, num_states=num_states,
                        num_train_batches=num_train_batches, method=method,
                        fit_method=fit_method, kappa=kappa, num_iters=num_iters,
                        alpha=np.nan, fps=fps, model='gaussian'),
        )
        atomic_dump(to_save, result_filename)

        if states_save_path is not None:
            os.makedirs(states_save_path, exist_ok=True)
            atomic_dump((most_likely_states, use_fold, (num_states, np.nan, kappa)),
                        os.path.join(states_save_path, var_interest[0] + '_' + fit_id))

        del to_save
        gc.collect()
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def run_all_gaussian(idxs, var_interest, zsc, num_states, num_train_batches, method,
                     fit_method, save_path, data_path, fps, n_jobs=1, csv_path=None,
                     states_save_path=None, **kw):
    """Same skip-if-complete / parallel structure as hmm_dynamic_functions.run_all."""
    os.makedirs(save_path, exist_ok=True)
    todo = []
    for mat in idxs:
        mouse_name, session = mat[37:], mat[:36]
        fn = os.path.join(save_path,
                          f"best_results_{var_interest[0]}_{mouse_name}{session}")
        if not is_complete(fn):
            todo.append((mouse_name, session))
    print(f'Found {len(todo)} sessions to process.')

    args = dict(var_interest=var_interest, zsc=zsc, num_states=num_states,
                num_train_batches=num_train_batches, method=method,
                fit_method=fit_method, save_path=save_path, data_path=data_path,
                fps=fps, states_save_path=states_save_path, **kw)
    if n_jobs and n_jobs > 1:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(run_session_gaussian)(id, **args) for id in todo)
    else:
        rows = [run_session_gaussian(id, **args) for id in todo]

    out = pd.DataFrame([r for r in rows if r is not None])
    if csv_path is not None and len(out):
        out.to_csv(csv_path, index=False)
        print(f'wrote {len(out)} rows -> {csv_path}')
    return out
