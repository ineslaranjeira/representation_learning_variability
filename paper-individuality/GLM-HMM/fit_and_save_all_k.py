"""
Consolidated fitting pass for K=2,3,4 GLM-HMM (full-data, all of a mouse's
sessions pooled - same design as k3_state_interpretation.py /
k4_state_interpretation.py / n_transitions_vs_lda1_multi_k.py /
posterior_uncertainty_multi_k.py / mean_engaged_posterior_multi_k.py).

This replaces those five separate single-purpose refits: it saves the RAW
per-trial posterior probability arrays (long format) plus the fitted GLM
weights, so ANY future derived per-session statistic (occupancy under any
state grouping, n_transitions, entropy, mean posterior of any state, etc.)
can be computed directly from these files without ever fitting again.

Outputs (in k2_k3_pilot/):
  - all_k_posteriors.parquet: one row per (mouse, eid, K, trial_idx, state),
    column 'posterior' = p(state | data) for that trial.
  - all_k_weights.csv: one row per (mouse, K, state), GLM weight vector.

Run with the `glmhmm` conda env:
    /opt/anaconda3/envs/glmhmm/bin/python3 fit_and_save_all_k.py
"""
import time
import numpy as np
import numpy.random as npr
import pandas as pd

from loso_k2_k3_sweep import load_data, build_session_covariates, assemble_inpt, fit_k_pooled, OUT_DIR
from fit_k_general import fit_k_multistart

npr.seed(0)

INPUT_NAMES = ['stim', 'prev_choice', 'wsls', 'bias']
POSTERIORS_PATH = OUT_DIR / 'all_k_posteriors.parquet'
WEIGHTS_PATH = OUT_DIR / 'all_k_weights.csv'


def main():
    matched, states_df = load_data()
    session_counts = matched.groupby('mouse_name').size()
    qualifying_mice = session_counts[session_counts >= 3].index.tolist()
    lda_by_session = dict(zip(matched['session'], matched['lda_1']))

    posterior_rows = []
    weight_rows = []
    t0 = time.time()

    for mi, mouse in enumerate(qualifying_mice):
        mouse_sessions = matched[matched['mouse_name'] == mouse]['session'].tolist()

        covariates = {
            eid: build_session_covariates(states_df[states_df['eid'] == eid].reset_index(drop=True))
            for eid in mouse_sessions
        }
        stim_pool = np.concatenate([covariates[e][0] for e in mouse_sessions])
        stim_mean, stim_std = stim_pool.mean(), stim_pool.std()

        inputs_list, datas_list, masks_list = [], [], []
        for e in mouse_sessions:
            raw_stim, prev_choice_bin, wsls, y, mask = covariates[e]
            inputs_list.append(assemble_inpt(raw_stim, prev_choice_bin, wsls, stim_mean, stim_std))
            datas_list.append(y)
            masks_list.append(mask)

        for K in (2, 3, 4):
            if K <= 3:
                glmhmm = fit_k_pooled(inputs_list, datas_list, masks_list, K)
            else:
                glmhmm, _ = fit_k_multistart(inputs_list, datas_list, masks_list, K, n_restarts=3)

            weights = glmhmm.observations.params  # (K, 1, 4)
            for state in range(K):
                weight_rows.append({
                    'mouse_name': mouse, 'K': K, 'state': state,
                    **{name: weights[state, 0, j] for j, name in enumerate(INPUT_NAMES)},
                })

            posterior_probs = [glmhmm.expected_states(data=d, input=inp, mask=m)[0]
                               for d, inp, m in zip(datas_list, inputs_list, masks_list)]

            for eid, post in zip(mouse_sessions, posterior_probs):
                n_trials = post.shape[0]
                for state in range(K):
                    posterior_rows.append(pd.DataFrame({
                        'mouse_name': mouse, 'eid': eid, 'lda_1': lda_by_session[eid],
                        'K': K, 'trial_idx': np.arange(n_trials), 'state': state,
                        'posterior': post[:, state],
                    }))

        elapsed = time.time() - t0
        print(f"[{mi + 1}/{len(qualifying_mice)}] {mouse}: total so far {elapsed / 60:.1f} min", flush=True)

    weights_df = pd.DataFrame(weight_rows)
    posteriors_df = pd.concat(posterior_rows, ignore_index=True)
    weights_df.to_csv(WEIGHTS_PATH, index=False)
    posteriors_df.to_parquet(POSTERIORS_PATH, index=False)
    print(f"\nSaved weights ({len(weights_df)} rows) to {WEIGHTS_PATH}")
    print(f"Saved posteriors ({len(posteriors_df)} rows, {posteriors_df.memory_usage(deep=True).sum()/1e6:.1f} MB) to {POSTERIORS_PATH}")


if __name__ == '__main__':
    main()
