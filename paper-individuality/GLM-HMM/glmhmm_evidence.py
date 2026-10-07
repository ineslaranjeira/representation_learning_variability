"""
Where does the K=2 GLM-HMM's engaged/disengaged label come from?
=================================================================
The label on trial t is the argmax of the smoothed posterior p(z_t | all choices of the
session). That posterior splits EXACTLY into two additive parts, on the log-odds scale:

    logit p(engaged_t | y_1..T) = N_t + LLR_t

    LLR_t  this trial's own evidence: log p(y_t | x_t, engaged) - log p(y_t | x_t, disengaged).
           x_t holds the current stimulus AND the history regressors (previous choice, WSLS) and
           the bias, so "current trial" still includes one trial of history.
    N_t    the evidence from every OTHER trial, carried to t by the transition matrix:
           logit p(engaged_t | y_s, s != t), from forward-backward with trial t's emission left out.

LLR_t is then split over its inputs (stimulus / history / bias) with Shapley values: the
log-likelihood ratio is recomputed with each subset of input groups set to 0 (stimulus 0 = an
average, z-scored stimulus; history 0 = neutral) and the marginal contributions averaged over
orderings. That attribution is a convention (the model is nonlinear), but with 3 groups it is
exact Shapley, not an approximation.

Also reported:
  * weight-level: (w_engaged - w_disengaged) x SD(input), how much each regressor separates the
    two states' predictions on a typical trial;
  * reach: how far along the session one trial's choice moves the posterior. Trial s's emission
    is removed and the posterior recomputed; |change in logit p(engaged_{s+k})| vs k. Plus the
    prior's own memory, lambda = 1 - p(e->d) - p(d->e), half-life log(0.5)/log(lambda).

THE FIT. Mice are refit with session_glmhmm_em.py (numpy EM, the same model as engaged.py's
ssm fit: sessions of a mouse pooled, stimulus z-scored over the mouse, paper init), because the
ssm parameters were never saved. Agreement with the stored p_state1 is printed per mouse.

Usage (iblenv):
    python glmhmm_evidence.py                 # mice in the LDA x syllable set (56)
    python glmhmm_evidence.py --all           # every mouse in merged_behavioral_and_states.pqt
    python glmhmm_evidence.py --max-mice 3    # smoke test
Outputs in GLM-HMM/glmhmm_evidence/.
"""
import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'transition_prediction'))
from session_glmhmm_em import fit_glmhmm, log_likelihoods, forward_backward   # noqa: E402
from lda_predicts_session_glmhmm import build_unit_covariates                  # noqa: E402

OUT = HERE / 'glmhmm_evidence'
STATES_PATH = HERE / 'merged_behavioral_and_states.pqt'
GROUPS = {'stimulus': [0], 'history': [1, 2], 'bias': [3]}   # columns of X: stim, prev_choice, wsls, bias
REACH = 30            # trials either side for the influence kernel
N_REACH_PROBES = 25   # trials per session whose emission is removed for the kernel


def forward_backward_logs(ll, pi0, log_Ps):
    """log_alpha, log_beta (T, K) for one sequence."""
    T, K = ll.shape
    la = np.zeros((T, K))
    la[0] = np.log(pi0 + 1e-300) + ll[0]
    for t in range(1, T):
        la[t] = ll[t] + logsumexp(la[t - 1][:, None] + log_Ps, axis=0)
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(log_Ps + (ll[t + 1] + lb[t + 1])[None, :], axis=1)
    return la, lb


def llr(X, y, W, eng, dis, keep_cols):
    """Log-likelihood ratio engaged vs disengaged with only keep_cols of X kept (others 0)."""
    Xm = np.zeros_like(X)
    Xm[:, keep_cols] = X[:, keep_cols]
    ll = log_likelihoods(Xm, y, W)
    return ll[:, eng] - ll[:, dis]


def shapley(X, y, W, eng, dis):
    names = list(GROUPS)
    n = len(names)
    value = {}
    for r in range(n + 1):
        for S in itertools.combinations(names, r):
            cols = [c for g in S for c in GROUPS[g]]
            value[frozenset(S)] = llr(X, y, W, eng, dis, cols)
    phi = {}
    for g in names:
        others = [h for h in names if h != g]
        tot = 0.0
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot = tot + w * (value[frozenset(S) | {g}] - value[frozenset(S)])
        phi[g] = tot
    return phi


def reach_kernel(ll, pi0, log_Ps, eng, dis, rng):
    """|change in posterior logit at s+k| when trial s's emission is removed, for k in
    -REACH..REACH, averaged over N_REACH_PROBES random s."""
    la, lb = forward_backward_logs(ll, pi0, log_Ps)
    base = (la + lb)[:, eng] - (la + lb)[:, dis]
    T = len(ll)
    out = np.full((N_REACH_PROBES, 2 * REACH + 1), np.nan)
    probes = rng.choice(np.arange(REACH, T - REACH), size=min(N_REACH_PROBES, max(T - 2 * REACH, 0)),
                        replace=False) if T > 2 * REACH + 1 else []
    for i, s in enumerate(probes):
        ll2 = ll.copy()
        ll2[s] = 0.0
        la2, lb2 = forward_backward_logs(ll2, pi0, log_Ps)
        new = (la2 + lb2)[:, eng] - (la2 + lb2)[:, dis]
        out[i] = np.abs(new - base)[s - REACH:s + REACH + 1]
    return out


def analyse_mouse(states_df, animal, rng):
    adf = states_df[states_df['animal'] == animal]
    eids = list(pd.unique(adf['eid']))
    inputs, datas = build_unit_covariates(states_df, eids, pool_zscore=True)
    fit = fit_glmhmm(inputs, datas, num_states=2)
    W, pi0, log_Ps = fit['W'], fit['pi0'], fit['log_Ps']
    eng = int(np.argmin(W[:, 0]))      # most negative stim weight = follows the stimulus
    dis = 1 - eng

    rows, kernels = [], []
    for eid, X, y, post in zip(eids, inputs, datas, fit['posteriors']):
        ll = log_likelihoods(X, y, W)
        la, lb = forward_backward_logs(ll, pi0, log_Ps)
        L = (la + lb)[:, eng] - (la + lb)[:, dis]              # posterior logit (normaliser cancels)
        LLR = ll[:, eng] - ll[:, dis]
        N = L - LLR                                            # leave-trial-out logit
        phi = shapley(X, y, W, eng, dis)
        sdf = adf[adf['eid'] == eid].reset_index(drop=True)
        rows.append(pd.DataFrame({
            'animal': animal, 'eid': eid, 'trial': np.arange(len(y)),
            'p_engaged_refit': post[:, eng], 'p_state1_stored': sdf['p_state1'].to_numpy(),
            'logit_post': L, 'llr_own': LLR, 'logit_others': N,
            **{f'phi_{g}': v for g, v in phi.items()},
            'abs_contrast': sdf['signed_contrast'].abs().to_numpy(),
            'correct': (sdf['rewarded'] == 1).to_numpy(),
        }))
        kernels.append(reach_kernel(ll, pi0, log_Ps, eng, dis, rng))

    sd = np.concatenate(inputs).std(axis=0)
    trans = np.exp(log_Ps)
    lam = 1 - trans[eng, dis] - trans[dis, eng]
    params = dict(animal=animal, n_sessions=len(eids), n_trials=sum(len(y) for y in datas),
                  converged=fit['converged'], n_iter=fit['n_iter'],
                  p_stay_engaged=trans[eng, eng], p_stay_disengaged=trans[dis, dis], memory_lambda=lam,
                  memory_half_life=np.log(0.5) / np.log(lam) if 0 < lam < 1 else np.nan,
                  **{f'w_eng_{n}': W[eng, j] for j, n in enumerate(['stim', 'prev_choice', 'wsls', 'bias'])},
                  **{f'w_dis_{n}': W[dis, j] for j, n in enumerate(['stim', 'prev_choice', 'wsls', 'bias'])},
                  **{f'sep_{n}': abs(W[eng, j] - W[dis, j]) * (sd[j] if n != 'bias' else 1)
                     for j, n in enumerate(['stim', 'prev_choice', 'wsls', 'bias'])})
    return pd.concat(rows, ignore_index=True), params, np.vstack(kernels)


def summarise(trials, params, kernel):
    t = trials
    stored_r = t.groupby('animal').apply(
        lambda d: abs(np.corrcoef(d['p_engaged_refit'], d['p_state1_stored'])[0, 1]), include_groups=False)
    agree = ((t['p_engaged_refit'] >= .5) == (t['p_state1_stored'] >= .5)).groupby(t['animal']).mean()
    print(f"\nrefit vs stored ssm posterior: median r = {stored_r.median():.3f} "
          f"(min {stored_r.min():.3f}); label agreement median {agree.median():.1%} (min {agree.min():.1%})")

    print("\n1. CURRENT TRIAL vs OTHER TRIALS (posterior logit = others + own, exactly)")
    own, oth = t['llr_own'].abs(), t['logit_others'].abs()
    print(f"   median |own evidence| = {own.median():.2f} nats, median |other trials| = {oth.median():.2f} nats")
    flip = np.sign(t['logit_others']) != np.sign(t['logit_post'])
    print(f"   labels decided by the trial's own choice (removing it flips the label): {flip.mean():.2%} of trials")
    dis = t['logit_post'] < 0
    print(f"   ... of disengaged-labelled trials: {flip[dis].mean():.2%}; of engaged-labelled: {flip[~dis].mean():.2%}")
    per_mouse = t.assign(flip=flip).groupby('animal')['flip'].mean()
    print(f"   per mouse: median {per_mouse.median():.2%}, range {per_mouse.min():.2%}-{per_mouse.max():.2%}")

    print("\n2. WITHIN THE TRIAL'S OWN EVIDENCE: Shapley split (mean |phi|, nats; and share)")
    phis = t[[c for c in t.columns if c.startswith('phi_')]].abs().mean()
    print((pd.DataFrame({'mean_abs_phi': phis, 'share': phis / phis.sum()})).round(3).to_string())
    print("   own evidence by |contrast| x outcome (mean LLR, + = engaged):")
    print(t.groupby(['abs_contrast', 'correct'])['llr_own'].mean().unstack().round(2).to_string())

    print("\n3. WEIGHT-LEVEL SEPARATION |w_eng - w_dis| x SD(input), median across mice")
    p = pd.DataFrame(params)
    print(p[[c for c in p.columns if c.startswith('sep_')]].median().round(3).to_string())

    print("\n4. REACH: median |change in posterior logit| at distance k when one trial is removed")
    kmed = np.nanmean(kernel, axis=0)
    ks = np.arange(-REACH, REACH + 1)
    for k in [0, 1, 2, 5, 10, 20, 30]:
        print(f"   k=+-{k:2d}: {0.5 * (kmed[ks == k][0] + kmed[ks == -k][0]):.3f}")
    print(f"   prior memory: median p_stay engaged {p['p_stay_engaged'].median():.3f}, disengaged "
          f"{p['p_stay_disengaged'].median():.3f}, half-life {p['memory_half_life'].median():.1f} trials")
    return kmed, ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--max-mice', type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    states_df = pd.read_parquet(STATES_PATH)
    animals = list(pd.unique(states_df['animal']))
    if not a.all:
        import tp_data as td
        lda = td.load_lda()
        syl_eids = set(td.load_syllable_features()['eid'])
        keep = set(lda.loc[lda['session'].isin(syl_eids), 'mouse_name'])
        animals = [m for m in animals if m in keep]
    if a.max_mice:
        animals = animals[:a.max_mice]
    rng = np.random.default_rng(0)
    trials, params, kernels = [], [], []
    t0 = time.time()
    for i, animal in enumerate(animals):
        tr, pa, ke = analyse_mouse(states_df, animal, rng)
        trials.append(tr); params.append(pa); kernels.append(ke)
        print(f"[{i + 1}/{len(animals)}] {animal}: {pa['n_sessions']} sessions, converged={pa['converged']} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    trials = pd.concat(trials, ignore_index=True)
    kernel = np.vstack(kernels)
    tag = 'all' if a.all else 'lda_syllable_mice'
    trials.to_parquet(OUT / f'trial_evidence_{tag}.parquet')
    pd.DataFrame(params).to_csv(OUT / f'mouse_params_{tag}.csv', index=False)
    kmed, ks = summarise(trials, params, kernel)
    pd.DataFrame({'k': ks, 'mean_abs_dlogit': kmed}).to_csv(OUT / f'reach_kernel_{tag}.csv', index=False)


if __name__ == '__main__':
    main()
