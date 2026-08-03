"""
Full leave-one-session-out (LOSO) K=2 vs K=3 GLM-HMM sweep.

For every mouse with >=3 fitted-and-LDA-matched sessions (54 mice, 242
sessions total), and for each of that mouse's sessions in turn: fit K=2 and
K=3 by pooling the mouse's OTHER sessions (joint fit, shared weights and
transition matrix across those sessions - same design as engaged.py's
model_single_mouse), then score held-out log-likelihood on the excluded
session alone. This keeps each session's own LDA1 label paired with a
genuine held-out evaluation, while still giving the model enough pooled
data to fit stably (unlike a within-session train/test split).

The stimulus regressor is z-scored using only the training sessions' pooled
mean/std (not the held-out session), and the null/bias baseline for the
bits-per-trial metric uses the training sessions' pooled P(choice=right)
(not the held-out session's own), so nothing about the held-out session
leaks into anything used to evaluate it.

Output: one row per (mouse, held_out_session, K) with held-out LL and
bits/trial, saved incrementally (per mouse) to loso_k2_k3_results.csv so
partial progress survives an interruption.

Run with the `glmhmm` conda env (estimated ~16 min total, see
time_estimate_loso.py):
    /opt/anaconda3/envs/glmhmm/bin/python3 loso_k2_k3_sweep.py
"""
import sys
import time
import numpy as np
import numpy.random as npr
import pandas as pd
import ssm
from pathlib import Path

from compare_k2_k3_pilot import (
    GLM_HMM_DIR, OUT_DIR, N_EM_ITERS, TRANSITION_ALPHA, PRIOR_SIGMA,
    INPUT_DIM, NUM_CATEGORIES, OBS_DIM, FULL_PARAMS,
)

npr.seed(0)

RESULTS_PATH = OUT_DIR / 'loso_k2_k3_results.csv'


def load_data():
    lda = pd.read_csv(OUT_DIR / 'lda1_export.csv').rename(columns={'0': 'lda_1'})
    states_df = pd.read_parquet(GLM_HMM_DIR / 'merged_behavioral_and_states.pqt')
    fitted_sessions = set(states_df['eid'].unique())
    matched = lda[lda['session'].isin(fitted_sessions)].copy()
    return matched, states_df


def build_session_covariates(session_df):
    """Un-normalized per-session covariates: raw_stim, prev_choice_bin, wsls, y, mask.
    Stimulus is NOT z-scored here - that happens later using training-only stats."""
    contrast_left = session_df['contrastLeft'].fillna(0).values
    contrast_right = session_df['contrastRight'].fillna(0).values
    raw_stim = contrast_right - contrast_left

    right_correct = session_df['contrastLeft'].isna() & (session_df['rewarded'] == 1)
    right_incorrect = session_df['contrastRight'].isna() & (session_df['rewarded'] == -1)
    choice_right = (right_correct | right_incorrect).astype(int).values

    prev_choice = np.hstack([choice_right[0], choice_right[:-1]])
    prev_choice_bin = 2 * prev_choice - 1

    reward = session_df['rewarded'].values
    prev_reward = np.hstack([reward[0], reward[:-1]])
    wsls = (prev_reward * prev_choice_bin).astype(float)
    wsls[wsls == 0] = -1

    T = len(session_df)
    y = choice_right[:, None].astype(int)
    mask = np.ones((T, 1), dtype=int)
    return raw_stim, prev_choice_bin, wsls, y, mask


def assemble_inpt(raw_stim, prev_choice_bin, wsls, stim_mean, stim_std):
    stim_z = (raw_stim - stim_mean) / stim_std
    T = len(raw_stim)
    return np.column_stack([stim_z, prev_choice_bin, wsls, np.ones(T)])


def fit_k_pooled(inputs_list, datas_list, masks_list, num_states):
    glmhmm = ssm.HMM(
        num_states, OBS_DIM, INPUT_DIM,
        observations="input_driven_obs",
        observation_kwargs=dict(C=NUM_CATEGORIES, prior_sigma=PRIOR_SIGMA),
        transitions="sticky",
        transition_kwargs=dict(alpha=TRANSITION_ALPHA, kappa=0),
    )
    glmhmm.params = [
        [FULL_PARAMS[0][0][:num_states]],
        [FULL_PARAMS[1][0][:num_states, :num_states]],
        FULL_PARAMS[2][:num_states],
    ]
    glmhmm.fit(datas_list, inputs=inputs_list, masks=masks_list, method="em",
               num_iters=N_EM_ITERS, initialize=False, tolerance=1e-4, verbose=0)
    return glmhmm


def null_bernoulli_ll_given_p(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def bits_per_trial(ll_model, ll_null, n_trials):
    return (ll_model - ll_null) / n_trials / np.log(2)


def main(max_mice=None):
    matched, states_df = load_data()
    session_counts = matched.groupby('mouse_name').size()
    qualifying_mice = session_counts[session_counts >= 3].index.tolist()
    if max_mice is not None:
        qualifying_mice = qualifying_mice[:max_mice]

    total_start = time.time()
    all_rows = []
    first_write = True

    for mi, mouse in enumerate(qualifying_mice):
        mouse_sessions = matched[matched['mouse_name'] == mouse]['session'].tolist()
        lda_by_session = dict(zip(matched['session'], matched['lda_1']))
        n_sess = len(mouse_sessions)
        mouse_t0 = time.time()

        covariates = {
            eid: build_session_covariates(states_df[states_df['eid'] == eid].reset_index(drop=True))
            for eid in mouse_sessions
        }

        mouse_rows = []
        for held_out in mouse_sessions:
            train_sessions = [e for e in mouse_sessions if e != held_out]

            train_stim_pool = np.concatenate([covariates[e][0] for e in train_sessions])
            stim_mean, stim_std = train_stim_pool.mean(), train_stim_pool.std()
            train_y_pool = np.concatenate([covariates[e][3].ravel() for e in train_sessions])
            null_p_train = train_y_pool.mean()

            inputs_list, datas_list, masks_list = [], [], []
            for e in train_sessions:
                raw_stim, prev_choice_bin, wsls, y, mask = covariates[e]
                inpt = assemble_inpt(raw_stim, prev_choice_bin, wsls, stim_mean, stim_std)
                inputs_list.append(inpt)
                datas_list.append(y)
                masks_list.append(mask)

            raw_stim_ho, prev_choice_bin_ho, wsls_ho, y_ho, mask_ho = covariates[held_out]
            inpt_ho = assemble_inpt(raw_stim_ho, prev_choice_bin_ho, wsls_ho, stim_mean, stim_std)
            n_heldout_trials = len(y_ho)
            ll_null_ho = null_bernoulli_ll_given_p(y_ho, null_p_train)

            for K in (2, 3):
                glmhmm = fit_k_pooled(inputs_list, datas_list, masks_list, K)
                ll_ho = glmhmm.log_likelihood([y_ho], inputs=[inpt_ho], masks=[mask_ho])
                bpt = bits_per_trial(ll_ho, ll_null_ho, n_heldout_trials)
                mouse_rows.append({
                    'mouse_name': mouse, 'held_out_session': held_out,
                    'lda_1': lda_by_session[held_out], 'n_sessions_mouse': n_sess,
                    'n_heldout_trials': n_heldout_trials, 'K': K,
                    'held_out_ll': ll_ho, 'held_out_bpt': bpt,
                })

        all_rows.extend(mouse_rows)
        mouse_elapsed = time.time() - mouse_t0
        total_elapsed = time.time() - total_start
        print(f"[{mi + 1}/{len(qualifying_mice)}] {mouse} ({n_sess} sessions): "
              f"{mouse_elapsed:.1f}s (total so far: {total_elapsed / 60:.1f} min)", flush=True)

        pd.DataFrame(mouse_rows).to_csv(
            RESULTS_PATH, mode='w' if first_write else 'a',
            header=first_write, index=False)
        first_write = False

    total_elapsed = time.time() - total_start
    print(f"\nDone: {len(qualifying_mice)} mice, {len(all_rows)} rows, "
          f"{total_elapsed / 60:.1f} min total")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == '__main__':
    max_mice = None
    if '--smoke-test' in sys.argv:
        max_mice = 2
    main(max_mice=max_mice)
