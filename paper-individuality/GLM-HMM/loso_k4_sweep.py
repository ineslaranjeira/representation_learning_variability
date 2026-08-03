"""
Leave-one-session-out (LOSO) sweep for K=4 only, using 3 random restarts per
fit (see fit_k_general.py - no principled reference init exists beyond K=3,
so extra states start from perturbed copies of the K=3 template and the
restart with the best TRAINING log-likelihood is kept).

Identical design to loso_k2_k3_sweep.py (same mice, same held-out folds, same
train-derived stimulus z-scoring and null baseline) so results merge directly
with loso_k2_k3_results.csv for K4-vs-K2 and K4-vs-K3 comparisons.

Run with the `glmhmm` conda env (projected ~30 min, see time_estimate_higher_k.py):
    /opt/anaconda3/envs/glmhmm/bin/python3 loso_k4_sweep.py
"""
import time
import numpy as np
import numpy.random as npr
import pandas as pd

from loso_k2_k3_sweep import (
    load_data, build_session_covariates, assemble_inpt,
    null_bernoulli_ll_given_p, bits_per_trial, OUT_DIR,
)
from fit_k_general import fit_k_multistart

npr.seed(0)

RESULTS_PATH = OUT_DIR / 'loso_k4_results.csv'
N_RESTARTS = 3
K = 4


def main():
    matched, states_df = load_data()
    session_counts = matched.groupby('mouse_name').size()
    qualifying_mice = session_counts[session_counts >= 3].index.tolist()
    lda_by_session = dict(zip(matched['session'], matched['lda_1']))

    total_start = time.time()
    first_write = True

    for mi, mouse in enumerate(qualifying_mice):
        mouse_sessions = matched[matched['mouse_name'] == mouse]['session'].tolist()
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

            glmhmm, _ = fit_k_multistart(inputs_list, datas_list, masks_list, K, n_restarts=N_RESTARTS)
            ll_ho = glmhmm.log_likelihood([y_ho], inputs=[inpt_ho], masks=[mask_ho])
            bpt = bits_per_trial(ll_ho, ll_null_ho, n_heldout_trials)
            mouse_rows.append({
                'mouse_name': mouse, 'held_out_session': held_out,
                'lda_1': lda_by_session[held_out], 'n_sessions_mouse': n_sess,
                'n_heldout_trials': n_heldout_trials, 'K': K,
                'held_out_ll': ll_ho, 'held_out_bpt': bpt,
            })

        mouse_elapsed = time.time() - mouse_t0
        total_elapsed = time.time() - total_start
        print(f"[{mi + 1}/{len(qualifying_mice)}] {mouse} ({n_sess} sessions): "
              f"{mouse_elapsed:.1f}s (total so far: {total_elapsed / 60:.1f} min)", flush=True)

        pd.DataFrame(mouse_rows).to_csv(
            RESULTS_PATH, mode='w' if first_write else 'a',
            header=first_write, index=False)
        first_write = False

    total_elapsed = time.time() - total_start
    print(f"\nDone: {len(qualifying_mice)} mice, {total_elapsed / 60:.1f} min total")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
