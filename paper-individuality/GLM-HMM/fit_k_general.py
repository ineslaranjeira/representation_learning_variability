"""
Shared fitting utilities for K >= 2, extending the K<=3 hardcoded-init recipe
(from engaged.py / compare_k2_k3_pilot.py) to arbitrary K.

For K<=3, behaves exactly like before: truncate the hardcoded paper-fit init.
For K>3, there's no principled reference init, so extra states are built as
small random perturbations of the 3 existing templates (engaged + 2 biased-
direction states), and several random-seeded restarts are fit; whichever
restart has the best TRAINING log-likelihood is kept (held-out data is never
used to pick a restart, only to evaluate the winner).
"""
import numpy as np
import ssm

from compare_k2_k3_pilot import (
    TRANSITION_ALPHA, PRIOR_SIGMA, INPUT_DIM, NUM_CATEGORIES, OBS_DIM,
    N_EM_ITERS, FULL_PARAMS,
)

BASE_PI0 = FULL_PARAMS[0][0]
BASE_TRANS = FULL_PARAMS[1][0]
BASE_OBS = FULL_PARAMS[2]


def make_init_params(num_states, seed):
    """Construct (pi0, trans, obs) init arrays for any num_states >= 2."""
    rng = np.random.RandomState(seed)

    if num_states <= 3:
        return (BASE_PI0[:num_states].copy(),
                BASE_TRANS[:num_states, :num_states].copy(),
                BASE_OBS[:num_states].copy())

    pi0 = np.concatenate([BASE_PI0, np.full(num_states - 3, BASE_PI0.min())])

    obs_list = [BASE_OBS[k].copy() for k in range(3)]
    for _ in range(num_states - 3):
        template_idx = rng.randint(3)
        perturb = BASE_OBS[template_idx] + rng.normal(scale=0.5, size=BASE_OBS[template_idx].shape)
        obs_list.append(perturb)
    obs = np.stack(obs_list)

    diag_val = np.diag(BASE_TRANS).mean()
    off_diag_val = BASE_TRANS[~np.eye(3, dtype=bool)].mean()
    trans = np.full((num_states, num_states), off_diag_val)
    trans[:3, :3] = BASE_TRANS
    for i in range(3, num_states):
        trans[i, :] = off_diag_val
        trans[i, i] = diag_val

    return pi0, trans, obs


def fit_one_restart(inputs_list, datas_list, masks_list, num_states, seed):
    pi0, trans, obs = make_init_params(num_states, seed)
    glmhmm = ssm.HMM(
        num_states, OBS_DIM, INPUT_DIM,
        observations="input_driven_obs",
        observation_kwargs=dict(C=NUM_CATEGORIES, prior_sigma=PRIOR_SIGMA),
        transitions="sticky",
        transition_kwargs=dict(alpha=TRANSITION_ALPHA, kappa=0),
    )
    glmhmm.params = [[pi0], [trans], obs]
    train_lps = glmhmm.fit(datas_list, inputs=inputs_list, masks=masks_list, method="em",
                            num_iters=N_EM_ITERS, initialize=False, tolerance=1e-4, verbose=0)
    train_ll = glmhmm.log_likelihood(datas_list, inputs=inputs_list, masks=masks_list)
    return glmhmm, train_ll


def fit_k_multistart(inputs_list, datas_list, masks_list, num_states, n_restarts=3, base_seed=0):
    """Fit num_states with n_restarts random-seeded inits (n_restarts=1 for K<=3 is
    plenty since that init is a good, literature-derived starting point; more
    restarts matter more as K grows and the extra states start from perturbed
    guesses). Returns the restart with the best TRAINING log-likelihood."""
    if num_states <= 3:
        n_restarts = 1
    best_glmhmm, best_train_ll = None, -np.inf
    for r in range(n_restarts):
        glmhmm, train_ll = fit_one_restart(inputs_list, datas_list, masks_list, num_states, seed=base_seed + r)
        if train_ll > best_train_ll:
            best_glmhmm, best_train_ll = glmhmm, train_ll
    return best_glmhmm, best_train_ll
