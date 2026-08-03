"""
Pilot: compare K=2 vs K=3 GLM-HMM fits on a handful of individual sessions,
picked as the most extreme low- and high-LDA1 sessions among the sessions
that already have a GLM-HMM K=2 fit in merged_behavioral_and_states.pqt.

The model class, hyperparameters, and hardcoded initialization below are
copied verbatim from engaged.py's `model_single_mouse` (the code that
generated merged_behavioral_and_states.pqt), truncated to K states exactly
as that function does. This means the K=2 fit produced here is a faithful,
directly-comparable re-fit of the existing data (not a different recipe),
and K=3 is the natural one-state extension of that same recipe.

Each picked session is fit independently (not pooled across an animal's
other sessions, unlike engaged.py's per-animal fitting), since the point
is a session-level "would this particular session support a 3rd state"
check across a small, deliberately mixed sample of low- and high-LDA1
sessions.

IMPORTANT CAVEAT: the comparison metric below (bits/trial over a bias-only
null) is computed IN-SAMPLE (no held-out split). More states can never
mechanically decrease training log-likelihood, so a positive K=3-over-K=2
gap here is suggestive, not conclusive - it's meant to decide whether a
proper cross-validated (held-out) comparison is worth building next, not
to replace one.

Assumes no violation trials remain in the input dataframe (consistent with
it already being the output of a successful upstream GLM-HMM fit that
required a clean binary choice on every trial).

Run with the `glmhmm` conda env (already has Zoe Ashwood's ssm fork built
and was smoke-tested against this exact API before use):
    /opt/anaconda3/envs/glmhmm/bin/python3 compare_k2_k3_pilot.py
"""
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import ssm

npr.seed(0)

GLM_HMM_DIR = Path(__file__).resolve().parent
PREFIX = GLM_HMM_DIR.parent
CLUSTERING_DIR = PREFIX / 'clustering'
OUT_DIR = GLM_HMM_DIR / 'k2_k3_pilot'
OUT_DIR.mkdir(exist_ok=True)

N_EM_ITERS = 200
TRANSITION_ALPHA = 2
PRIOR_SIGMA = 2
INPUT_DIM = 4  # stim, prev_choice, wsls, bias
NUM_CATEGORIES = 2
OBS_DIM = 1
N_PER_GROUP = 2

# Exact hardcoded initialization from engaged.py's model_single_mouse - the same
# init used to fit the existing K=2 posteriors in merged_behavioral_and_states.pqt.
FULL_PARAMS = [
    [np.array([-0.52493862, -1.64298306, -1.53708898])],
    [np.array([[-0.02608457, -4.25327563, -4.46282725],
                [-3.04799552, -0.05351097, -5.370778],
                [-3.09435783, -5.83956581, -0.04941527]])],
    np.array([[[-7.20614670e+00, -3.12209531e-01, -1.61175163e-03, 1.43813500e-01]],
               [[-1.22616681e+00, -3.61431579e-01, -1.80390841e-01, 1.70656106e+00]],
               [[-1.20628874e+00, -3.20944513e-01, -1.98757460e-01, -1.62621352e+00]]]),
]

COLORS = ['#ff7f00', '#4daf4a', '#377eb8']


def select_pilot_sessions(n_per_group=N_PER_GROUP):
    """2 lowest- and 2 highest-LDA1 sessions among sessions with an existing GLM-HMM fit.

    Uses the same LDA1 file and same GLM-HMM output file as load_states.ipynb,
    so the picked sessions and their LDA1 values are directly consistent with
    everything analyzed there.

    The original file is a pickle saved with a newer pandas/pickle protocol
    that this env's Python 3.7 can't unpickle, so a one-time CSV export
    (same rows/columns, made with a newer env, see k2_k3_pilot/lda1_export.csv)
    is read instead; column '0' becomes the string '0' after the CSV round-trip."""
    lda_csv = OUT_DIR / 'lda1_export.csv'
    lda = pd.read_csv(lda_csv).rename(columns={'0': 'lda_1'})
    states_df = pd.read_parquet(GLM_HMM_DIR / 'merged_behavioral_and_states.pqt')
    fitted_sessions = set(states_df['eid'].unique())
    matched = lda[lda['session'].isin(fitted_sessions)].copy()

    low = matched.sort_values('lda_1', ascending=True).head(n_per_group)
    high = matched.sort_values('lda_1', ascending=False).head(n_per_group)
    picked = pd.concat([low.assign(group='low_lda1'), high.assign(group='high_lda1')])
    return picked[['session', 'mouse_name', 'lda_1', 'group']].reset_index(drop=True), states_df


def build_session_inputs(session_df):
    """Build (inpt, y, mask) for one session's trials, following engaged.py's
    process_bwm_mouse/model_single_mouse covariate construction, except the
    stimulus regressor is z-scored within this single session rather than
    pooled across an animal's sessions (not applicable here, since each
    pilot session is fit independently)."""
    contrast_left = session_df['contrastLeft'].fillna(0).values
    contrast_right = session_df['contrastRight'].fillna(0).values
    stim = contrast_right - contrast_left
    stim = (stim - stim.mean()) / stim.std()

    right_correct = session_df['contrastLeft'].isna() & (session_df['rewarded'] == 1)
    right_incorrect = session_df['contrastRight'].isna() & (session_df['rewarded'] == -1)
    choice_right = (right_correct | right_incorrect).astype(int).values  # y in {0,1}, 1 = chose right

    prev_choice = np.hstack([choice_right[0], choice_right[:-1]])
    prev_choice_bin = 2 * prev_choice - 1  # {-1, 1}

    reward = session_df['rewarded'].values  # already {-1, 1}
    prev_reward = np.hstack([reward[0], reward[:-1]])
    wsls = (prev_reward * prev_choice_bin).astype(float)
    wsls[wsls == 0] = -1

    T = len(session_df)
    inpt = np.column_stack([stim, prev_choice_bin, wsls, np.ones(T)])
    y = choice_right[:, None].astype(int)
    mask = np.ones((T, 1), dtype=int)
    return inpt, y, mask


def fit_k(inpt, y, mask, num_states):
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
    glmhmm.fit([y], inputs=[inpt], masks=[mask], method="em",
               num_iters=N_EM_ITERS, initialize=False, tolerance=1e-4, verbose=0)
    posterior_probs = glmhmm.expected_states(data=y, input=inpt, mask=mask)[0]
    ll_model = glmhmm.log_likelihood([y], inputs=[inpt], masks=[mask])
    return glmhmm, posterior_probs, ll_model


def null_bernoulli_ll(y):
    """Log-likelihood of a constant-P(right) bias-only model, as the comparison baseline."""
    p = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def bits_per_trial(ll_model, ll_null, n_trials):
    return (ll_model - ll_null) / n_trials / np.log(2)


def validate_against_existing_k2(session_df, our_k2_post):
    """Sanity check: compare our from-scratch K=2 refit against the K=2 posteriors
    already stored in merged_behavioral_and_states.pqt for this same session. Not
    expected to match exactly (that fit pooled the animal's other sessions too;
    ours doesn't), but should be in the same ballpark if the covariates were
    built correctly."""
    existing_p_state1 = session_df['p_state1'].values
    # our state ordering is arbitrary (K=2 states aren't labeled "engaged"/"disengaged"),
    # so compare against whichever of our two states correlates better
    corr_as_is = np.corrcoef(our_k2_post[:, 0], existing_p_state1)[0, 1]
    corr_flipped = np.corrcoef(our_k2_post[:, 1], existing_p_state1)[0, 1]
    return max(corr_as_is, corr_flipped)


def plot_session_comparison(eid, group, lda1, results, out_path):
    """results: dict K -> (glmhmm, posterior_probs, ll_model, bpt)"""
    ks = sorted(results.keys())
    fig, axes = plt.subplots(2, len(ks), figsize=(6 * len(ks), 7))
    if len(ks) == 1:
        axes = axes[:, None]

    for col, K in enumerate(ks):
        glmhmm, post, ll_model, bpt = results[K]
        weights = glmhmm.observations.params
        cols = COLORS[:K]

        ax_w = axes[0, col]
        for k in range(K):
            ax_w.plot(weights[k][0], marker='o', color=cols[k])
        ax_w.axhline(0, color='k', ls='--', alpha=0.5)
        ax_w.set_xticks(range(4))
        ax_w.set_xticklabels(['stim', 'prev_choice', 'wsls', 'bias'], rotation=45, fontsize=8)
        ax_w.set_title(f'K={K} weights  (train bpt={bpt:.4f})', fontsize=10)

        ax_p = axes[1, col]
        for k in range(K):
            ax_p.plot(post[:, k], color=cols[k], lw=0.8, label=f'state{k + 1}')
        ax_p.set_ylim(-0.02, 1.02)
        ax_p.set_xlabel('trial')
        ax_p.set_ylabel('posterior prob.')
        ax_p.legend(fontsize=7)

    fig.suptitle(f'{eid[:8]}... ({group}, LDA1={lda1:.2f})', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    picked, states_df = select_pilot_sessions()
    print("Pilot sessions:")
    print(picked.to_string(index=False))
    print()

    summary_rows = []
    for _, row in picked.iterrows():
        eid = row['session']
        session_df = states_df[states_df['eid'] == eid].reset_index(drop=True)
        inpt, y, mask = build_session_inputs(session_df)
        n_trials = len(y)
        ll_null = null_bernoulli_ll(y)

        results = {}
        for K in (2, 3):
            print(f"Fitting K={K} for {eid[:8]}... ({row['group']}, n_trials={n_trials})")
            glmhmm, post, ll_model = fit_k(inpt, y, mask, K)
            bpt = bits_per_trial(ll_model, ll_null, n_trials)
            results[K] = (glmhmm, post, ll_model, bpt)

            row_out = {
                'eid': eid, 'mouse_name': row['mouse_name'], 'group': row['group'],
                'lda_1': row['lda_1'], 'n_trials': n_trials, 'K': K,
                'train_ll': ll_model, 'train_bpt': bpt,
            }
            if K == 2:
                row_out['corr_with_existing_k2_fit'] = validate_against_existing_k2(session_df, post)
            summary_rows.append(row_out)

        out_path = OUT_DIR / f"{row['group']}_{eid[:8]}_k2_vs_k3.png"
        plot_session_comparison(eid, row['group'], row['lda_1'], results, out_path)
        print(f"  saved {out_path}")

    summary = pd.DataFrame(summary_rows)
    id_cols = ['eid', 'mouse_name', 'group', 'lda_1', 'n_trials']
    # built via merge rather than .pivot(list) for compatibility with this env's old pandas,
    # which only supports a single column in pivot()'s index=
    k2 = summary[summary['K'] == 2][id_cols + ['train_bpt']].rename(columns={'train_bpt': 'bpt_K2'})
    k3 = summary[summary['K'] == 3][id_cols + ['train_bpt']].rename(columns={'train_bpt': 'bpt_K3'})
    summary_wide = k2.merge(k3, on=id_cols)
    summary_wide['bpt_gain_K3_over_K2'] = summary_wide['bpt_K3'] - summary_wide['bpt_K2']

    validation = summary.dropna(subset=['corr_with_existing_k2_fit'])[
        ['eid', 'group', 'corr_with_existing_k2_fit']]
    print("\n=== Sanity check: our from-scratch K=2 refit vs the K=2 fit already in the file ===")
    print("(correlation of posterior state-1 probability across trials; not expected to be")
    print(" ~1.0 since the existing fit pooled each animal's other sessions too, but should")
    print(" be clearly positive/high if the covariates were built correctly)\n")
    print(validation.to_string(index=False))

    print("\n=== Summary: in-sample bits/trial over a bias-only null ===")
    print("(K=3 mechanically can never do worse in-sample, so a positive gain here is")
    print(" suggestive, not conclusive - see the module docstring.)\n")
    print(summary_wide.sort_values('group').to_string(index=False))

    summary.to_csv(OUT_DIR / 'k2_k3_pilot_summary_long.csv', index=False)
    summary_wide.to_csv(OUT_DIR / 'k2_k3_pilot_summary_wide.csv', index=False)
    print(f"\nSaved summary tables and per-session figures to {OUT_DIR}")


if __name__ == '__main__':
    main()
