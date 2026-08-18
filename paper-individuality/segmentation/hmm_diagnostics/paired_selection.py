"""
Paired-fold model selection for the HMM grid search.

Drop-in alternative to find_best_param / find_2_best_param in
segmentation_functions.py. Same inputs, same return signature, so 4.2 can call
either one.

Why this exists
---------------
The CV folds are the same 5 contiguous time blocks for every (lag, kappa) cell,
so the held-out log-likelihoods are PAIRED across cells. find_2_best_param
compares cells using each cell's own across-fold SD, which is an unpaired
comparison: it charges the between-fold variance (early vs late session,
engagement drift) to the error bar on the difference. On this dataset the
across-fold SD of bits_LL is ~0.125 while the SD of the per-fold difference
(lag 30 - lag 1) is ~0.009, so the unpaired test is ~14x too conservative and
the parsimony rule collapses to the smallest grid value for ~98% of sessions.

Selection rule here
-------------------
Sequential increment test along the lag axis ("only buy complexity that pays"):

1. Average bits_LL over kappa to get a (lag, fold) profile. kappa is a prior
   concentration that does not change the parameter count and is inert at this
   data scale, so averaging over it just reduces noise on the lag comparison.
2. Start at the smallest lag. Walk up the grid; adopt a longer lag only if its
   PAIRED per-fold increment over the currently adopted lag is reliably > 0.
3. Report kappa as the best-scoring value at the adopted lag.

Why sequential rather than "compare every cell to the best cell": the best cell
is the maximum of n_kappa x n_lag noisy estimates, so its mean is upward-biased
by selection (winner's curse). Testing candidates against it makes them look
worse than they are and drags the choice toward the argmax's lag. Testing
consecutive increments has no such bias, and it is also the question you
actually care about -- does the next step up the grid pay for itself.

Two rules are available:
  'ttest' - adopt the longer lag if the paired increment is significantly > 0 at
            alpha (one-sided in effect, t with n_folds-1 df).
  '1se'   - adopt it only if mean(increment) > n_se * SE(increment). The
            Breiman/glmnet 1-SE rule on the paired scale; stricter.

BOUNDARY CHECK: the returned diagnostics include `at_ceiling`, true when the
adopted lag is the largest in the grid. If that fires often, the grid is too
short and the search has not converged -- extend it and refit.
"""

import numpy as np
from scipy.stats import t as _tdist


def _paired_survivors(bits_flat, rule, alpha, n_se):
    """bits_flat: (n_cells, n_folds). Returns (best_idx, survivor_mask, means)."""
    means = np.nanmean(bits_flat, axis=1)
    best = int(np.nanargmax(means))

    survives = np.zeros(len(means), dtype=bool)
    for i in range(len(means)):
        if i == best:
            survives[i] = True
            continue
        diff = bits_flat[best] - bits_flat[i]          # >0 means best is better
        ok = ~np.isnan(diff)
        n = int(ok.sum())
        if n < 2:
            continue                                    # cannot judge, drop it
        m = diff[ok].mean()
        se = diff[ok].std(ddof=1) / np.sqrt(n)
        if se == 0:
            survives[i] = m <= 0
        elif rule == 'ttest':
            crit = _tdist.ppf(1 - alpha / 2, n - 1)
            survives[i] = (m - crit * se) <= 0          # not reliably worse
        elif rule == '1se':
            survives[i] = m <= n_se * se
        else:
            raise ValueError(f"unknown rule {rule!r}")
    return best, survives, means


def _increment_pays(prof, i_from, i_to, rule, alpha, n_se):
    """Paired test: is prof[i_to] reliably better than prof[i_from] across folds?"""
    diff = prof[i_to] - prof[i_from]
    ok = ~np.isnan(diff)
    n = int(ok.sum())
    if n < 2:
        return False, np.nan
    m = diff[ok].mean()
    se = diff[ok].std(ddof=1) / np.sqrt(n)
    if se == 0:
        return m > 0, m
    if rule == 'ttest':
        return bool((m - _tdist.ppf(1 - alpha / 2, n - 1) * se) > 0), m
    if rule == '1se':
        return bool(m > n_se * se), m
    raise ValueError(f"unknown rule {rule!r}")


def find_2_best_param_paired(bits_LL, kappas, Lags, rule='ttest', alpha=0.05, n_se=1.0,
                             return_diagnostics=False):
    """Paired analogue of find_2_best_param. bits_LL is (kappa, lag, fold)."""
    kappas, Lags = list(kappas), list(Lags)
    order = np.argsort(Lags)                             # walk the grid in order
    prof = np.nanmean(bits_LL, axis=0)[order]            # (lag, fold), kappa-averaged
    lags_sorted = [Lags[i] for i in order]

    adopted = 0
    increments = {}
    for j in range(1, len(lags_sorted)):
        pays, gain = _increment_pays(prof, adopted, j, rule, alpha, n_se)
        increments[(lags_sorted[adopted], lags_sorted[j])] = (gain, pays)
        if pays:
            adopted = j
    best_lag = lags_sorted[adopted]

    # kappa: best-scoring value at the adopted lag (it barely moves the LL)
    col = np.nanmean(bits_LL[:, order[adopted], :], axis=1)
    best_kappa = kappas[int(np.nanargmax(col))]

    mean_bits_LL = np.nanmean(bits_LL, axis=2)           # (kappa, lag), as before

    if not return_diagnostics:
        return int(best_kappa), int(best_lag), mean_bits_LL

    diag = dict(
        argmax_lag=lags_sorted[int(np.nanargmax(np.nanmean(prof, axis=1)))],
        adopted_lag=best_lag,
        at_ceiling=(best_lag == max(lags_sorted)),
        increments=increments,
        lag_profile=dict(zip(lags_sorted, np.nanmean(prof, axis=1))),
    )
    return int(best_kappa), int(best_lag), mean_bits_LL, diag


def find_1_best_param_paired(bits_LL, kappas, rule='1se', alpha=0.05, n_se=1.0,
                            return_diagnostics=False):
    """Paired analogue of find_1_best_param. bits_LL is (kappa, fold).

    NOTE: for the PoissonHMM every kappa has the same parameter count, so there
    is no complexity ordering to exploit -- a larger kappa is arguably the more
    constrained model, not the less. This returns the least-kappa survivor to
    match the existing convention, but the parsimony argument does not really
    apply on this axis; treat the argmax as the honest answer.
    """
    kappas = list(kappas)
    best, survives, means = _paired_survivors(np.asarray(bits_LL), rule, alpha, n_se)
    chosen = int(np.where(survives)[0][0])
    mean_bits_LL = np.nanmean(bits_LL, axis=1)

    if not return_diagnostics:
        return int(kappas[chosen]), mean_bits_LL
    diag = dict(argmax_kappa=kappas[best], n_survivors=int(survives.sum()),
                n_cells=len(kappas),
                cost_of_parsimony=float(means[best] - means[chosen]))
    return int(kappas[chosen]), mean_bits_LL, diag


def find_best_param_paired(bits_LL, params, param_num, **kw):
    """Signature-compatible with find_best_param(bits_LL, params, param_num)."""
    _, Lags, kappas = params
    if param_num == 1:
        best_kappa, mean_bits_LL = find_1_best_param_paired(bits_LL, kappas, **kw)
        return best_kappa, [], mean_bits_LL
    elif param_num == 2:
        return find_2_best_param_paired(bits_LL, kappas, Lags, **kw)
    raise ValueError(param_num)
