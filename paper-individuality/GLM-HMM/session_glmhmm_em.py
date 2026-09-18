"""
Self-contained input-driven GLM-HMM (Bernoulli choice) fit by EM in numpy/scipy.

WHY THIS EXISTS: every other GLM-HMM script in this directory imports `ssm`
(Zoe Ashwood's fork) and is run with a mac-only interpreter
(/opt/anaconda3/envs/glmhmm/bin/python3). That package is not installed on
this machine, so this module reimplements exactly the model those scripts fit,
using only numpy/scipy, so the per-session sweep can run in `iblenv` here.

The implementation is deliberately a match, not an approximation, of
ssm.HMM(K, 1, 4, observations="input_driven_obs",
        observation_kwargs=dict(C=2, prior_sigma=2),
        transitions="sticky", transition_kwargs=dict(alpha=2, kappa=0)):

  - Observations. ssm's InputDrivenObservations with C=2 stores weights of
    shape (K, C-1, M) = (K, 1, 4) and pads a zero row for the LAST category,
    so the logit of category 0 is x @ W_k and of category 1 is 0. Hence
        p(y_t = 1 | z_t = k) = sigmoid(-x_t @ W_k).
    That sign convention is why the paper's engaged state has stimulus weight
    -7.2 (with stim = contrastRight - contrastLeft and y = chose right, a
    NEGATIVE weight means "follows the stimulus"). Keeping it means the
    weights this module returns are directly comparable to FULL_PARAMS in
    compare_k2_k3_pilot.py and to the existing per-mouse fits.
  - Weight prior. Gaussian N(0, prior_sigma^2) on every weight including the
    bias column, added to the M-step objective (MAP, as in ssm).
  - Transitions. Sticky, i.e. a Dirichlet(alpha + kappa*I) prior per row with
    the closed-form M-step P_kj propto (expected joint counts + alpha - 1 +
    kappa*delta_kj). With alpha=2, kappa=0 this adds one pseudo-count per cell.
  - Initial distribution. Normalized posterior over the first trial of each
    sequence, matching ssm's InitialStateDistribution.m_step.
  - EM stops when the change in log posterior falls below `tolerance`, or
    after `num_iters` iterations - same stopping rule as ssm's fit().

`fit_glmhmm` accepts a LIST of sessions, so it can fit one session alone
(what lda_predicts_session_glmhmm.py needs) or pool a mouse's sessions with
shared parameters (what engaged.py's model_single_mouse did) - the latter is
what --validate uses to check this code against the stored ssm posteriors.
"""
import numpy as np
from scipy.special import logsumexp, expit
from scipy.optimize import minimize

# Hyperparameters, copied from engaged.py / compare_k2_k3_pilot.py.
N_EM_ITERS = 200
TOLERANCE = 1e-4
TRANSITION_ALPHA = 2
TRANSITION_KAPPA = 0
PRIOR_SIGMA = 2
INPUT_DIM = 4  # stim, prev_choice, wsls, bias
INPUT_NAMES = ['stim', 'prev_choice', 'wsls', 'bias']

# Global fit from the paper's scripts, used as the initialization (truncated to
# K states) exactly as engaged.py's model_single_mouse does.
FULL_PARAMS = [
    [np.array([-0.52493862, -1.64298306, -1.53708898])],
    [np.array([[-0.02608457, -4.25327563, -4.46282725],
               [-3.04799552, -0.05351097, -5.370778],
               [-3.09435783, -5.83956581, -0.04941527]])],
    np.array([[[-7.20614670e+00, -3.12209531e-01, -1.61175163e-03, 1.43813500e-01]],
              [[-1.22616681e+00, -3.61431579e-01, -1.80390841e-01, 1.70656106e+00]],
              [[-1.20628874e+00, -3.20944513e-01, -1.98757460e-01, -1.62621352e+00]]]),
]


def init_params(num_states, seed=None):
    """(pi0, log_Ps, W) initialization. seed=None -> the paper init, truncated
    to num_states (the deterministic default, identical to every other script
    here). An int seed additionally jitters it, for multi-restart fitting."""
    if num_states > 3:
        raise ValueError("paper init only defines 3 states; this module targets K<=3")
    log_pi0 = FULL_PARAMS[0][0][:num_states].copy()
    pi0 = np.exp(log_pi0 - logsumexp(log_pi0))
    trans = FULL_PARAMS[1][0][:num_states, :num_states].copy()
    log_Ps = trans - logsumexp(trans, axis=1, keepdims=True)
    W = FULL_PARAMS[2][:num_states, 0, :].copy()  # (K, M)
    if seed is not None:
        rng = np.random.RandomState(seed)
        W = W + rng.normal(scale=0.5, size=W.shape)
        log_Ps = log_Ps + rng.normal(scale=0.2, size=log_Ps.shape)
        log_Ps = log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
    return pi0, log_Ps, W


def log_likelihoods(X, y, W):
    """(T, K) log p(y_t | x_t, z_t = k), with p(y=1) = sigmoid(-x @ W_k)."""
    z = -X @ W.T                       # (T, K)
    logp1 = -np.logaddexp(0.0, -z)     # log sigmoid(z)
    logp0 = -np.logaddexp(0.0, z)      # log(1 - sigmoid(z))
    yc = y[:, None]
    return yc * logp1 + (1.0 - yc) * logp0


def forward_backward(ll, pi0, log_Ps):
    """Returns (gamma (T,K), xi_sum (K,K), loglik) for one sequence."""
    T, K = ll.shape
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi0 + 1e-300) + ll[0]
    for t in range(1, T):
        log_alpha[t] = ll[t] + logsumexp(log_alpha[t - 1][:, None] + log_Ps, axis=0)
    loglik = logsumexp(log_alpha[-1])

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_Ps + (ll[t + 1] + log_beta[t + 1])[None, :], axis=1)

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # xi_sum[i, j] = sum_t p(z_t = i, z_{t+1} = j | data)
    log_xi = (log_alpha[:-1, :, None] + log_Ps[None, :, :]
              + (ll[1:] + log_beta[1:])[:, None, :] - loglik)
    xi_sum = np.exp(logsumexp(log_xi, axis=0))
    return gamma, xi_sum, loglik


def _obs_objective(w, X, y, weights, prior_sigma):
    """Negative expected log-likelihood + Gaussian prior, for one state."""
    z = -X @ w
    p1 = expit(z)
    logp1 = -np.logaddexp(0.0, -z)
    logp0 = -np.logaddexp(0.0, z)
    ll = np.sum(weights * (y * logp1 + (1.0 - y) * logp0))
    prior = -0.5 * np.sum(w ** 2) / prior_sigma ** 2
    # d(-ll)/dw: dz/dw = -x, so the usual logistic gradient picks up a sign.
    grad = X.T @ (weights * (y - p1)) + w / prior_sigma ** 2
    return -(ll + prior), grad


def m_step_observations(X_all, y_all, gamma_all, W, prior_sigma):
    """Weighted MAP logistic regression per state, warm-started at W."""
    K = W.shape[0]
    W_new = W.copy()
    for k in range(K):
        res = minimize(_obs_objective, W[k], jac=True, method='L-BFGS-B',
                       args=(X_all, y_all, gamma_all[:, k], prior_sigma),
                       options=dict(maxiter=200))
        W_new[k] = res.x
    return W_new


def log_prior_obs(W, prior_sigma):
    return -0.5 * np.sum(W ** 2) / prior_sigma ** 2


def log_prior_trans(log_Ps, alpha, kappa):
    K = log_Ps.shape[0]
    coef = (alpha - 1.0) * np.ones((K, K)) + kappa * np.eye(K)
    return float(np.sum(coef * log_Ps))


def fit_glmhmm(inputs, datas, num_states=2, seed=None, num_iters=N_EM_ITERS,
               tolerance=TOLERANCE, prior_sigma=PRIOR_SIGMA,
               alpha=TRANSITION_ALPHA, kappa=TRANSITION_KAPPA):
    """EM fit with parameters shared across all sequences in `inputs`/`datas`.

    inputs: list of (T_i, M) design matrices; datas: list of (T_i,) 0/1 choices.
    Returns a dict with pi0, log_Ps, W, per-sequence posteriors, loglik, and
    convergence info.
    """
    X_all = np.concatenate(inputs)
    y_all = np.concatenate(datas).astype(float)
    datas = [np.asarray(d, dtype=float).ravel() for d in datas]

    pi0, log_Ps, W = init_params(num_states, seed=seed)
    lp_prev, converged, n_iter = -np.inf, False, 0

    for n_iter in range(1, num_iters + 1):
        gammas, xi_total, loglik = [], np.zeros((num_states, num_states)), 0.0
        for X, y in zip(inputs, datas):
            gamma, xi_sum, ll = forward_backward(log_likelihoods(X, y, W), pi0, log_Ps)
            gammas.append(gamma)
            xi_total += xi_sum
            loglik += ll

        lp = loglik + log_prior_obs(W, prior_sigma) + log_prior_trans(log_Ps, alpha, kappa)
        if abs(lp - lp_prev) < tolerance:
            converged = True
            break
        lp_prev = lp

        # M-step
        pi0_new = np.sum([g[0] for g in gammas], axis=0) + 1e-8
        pi0 = pi0_new / pi0_new.sum()

        counts = xi_total + (alpha - 1.0) + kappa * np.eye(num_states) + 1e-8
        log_Ps = np.log(counts / counts.sum(axis=1, keepdims=True))

        W = m_step_observations(X_all, y_all, np.concatenate(gammas), W, prior_sigma)

    # Final E-step so the returned posteriors match the returned parameters.
    gammas, loglik = [], 0.0
    for X, y in zip(inputs, datas):
        gamma, _, ll = forward_backward(log_likelihoods(X, y, W), pi0, log_Ps)
        gammas.append(gamma)
        loglik += ll

    return dict(pi0=pi0, log_Ps=log_Ps, trans=np.exp(log_Ps), W=W,
                posteriors=gammas, loglik=float(loglik),
                log_posterior=float(loglik + log_prior_obs(W, prior_sigma)
                                    + log_prior_trans(log_Ps, alpha, kappa)),
                converged=converged, n_iter=n_iter)


def fit_glmhmm_multistart(inputs, datas, num_states=2, n_restarts=1, base_seed=0, **kw):
    """Restart 0 is always the deterministic paper init (seed=None); further
    restarts jitter it. The restart with the best TRAINING log posterior wins."""
    best = None
    for r in range(n_restarts):
        res = fit_glmhmm(inputs, datas, num_states=num_states,
                         seed=None if r == 0 else base_seed + r, **kw)
        if best is None or res['log_posterior'] > best['log_posterior']:
            best = res
    return best
