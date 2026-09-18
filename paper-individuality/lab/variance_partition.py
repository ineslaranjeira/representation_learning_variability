"""
PARTITIONING BEHAVIOURAL VARIANCE INTO LAB / MOUSE / SESSION
============================================================
The design is a two-fold NESTED random model -- sessions inside mice inside labs, every
mouse in exactly one lab, everything unbalanced (3-9 mice per lab, 3-16 sessions per
mouse). Crossed-design tools do not apply, and neither does a plain eta^2.

WHY NOT eta^2. The share of variance that falls between labs, computed directly, is biased
UPWARDS, and badly so when the number of groups is small relative to the number of units.
Ten lab means estimated from 58 mice absorb noise: with NO true lab effect at all, this
design returns eta^2(lab) ~ 0.16 on shuffled labels. Any eta^2 must be read against that
floor, which is what `permutation_null_eta2` provides. The variance components below are
unbiased by construction (Henderson I / ANOVA estimators) and are the number to quote.

WHAT THE ESTIMATOR IS. For y_ijk = mu + lab_i + mouse_j(i) + session_k(ij),

    E[MS_session] = s2_session
    E[MS_mouse]   = s2_session + k1 * s2_mouse
    E[MS_lab]     = s2_session + k2 * s2_mouse + k3 * s2_lab

with k1, k2, k3 the standard unbalanced-nested coefficients computed from the session
counts. Solving downwards gives unbiased estimates. They can come out NEGATIVE when the
true component is near zero -- that is information, not a bug, and they are reported raw
as well as truncated.

VALIDATED (see the notebook): on simulated data with the real design and known components
(4.0 / 2.0 / 1.0) it returns 3.90 / 2.00 / 1.00, and per feature it agrees with
statsmodels REML to two decimals.

PRECISION WARNING. s2_lab rests on 10 labs -- 9 degrees of freedom. Per feature it is very
noisy; only the average over features, with a bootstrap over labs, is worth interpreting.
"""
import numpy as np
import pandas as pd


def nested_variance_components(Y, lab, mouse, truncate=False):
    """Unbiased lab / mouse-within-lab / session-within-mouse variance per feature.

    Y : (n_sessions, n_features) array or DataFrame
    lab, mouse : length-n_sessions labels
    truncate : clip negative estimates at 0 (report the raw ones too)

    Returns (DataFrame of per-feature components, dict of design coefficients).
    """
    Yv = np.asarray(Y, float)
    if Yv.ndim == 1:
        Yv = Yv[:, None]
    lab = pd.Series(np.asarray(lab, dtype=object)).reset_index(drop=True)
    mouse = pd.Series(np.asarray(mouse, dtype=object)).reset_index(drop=True)

    N = len(lab)
    idx_l, lab_names = pd.factorize(lab)
    idx_m, mouse_names = pd.factorize(mouse)
    I, M = len(lab_names), len(mouse_names)
    if I < 2 or M <= I:
        raise ValueError(f'need >= 2 labs and more mice than labs (got {I} labs, {M} mice)')

    cnt_m = np.bincount(idx_m, minlength=M).astype(float)
    cnt_l = np.bincount(idx_l, minlength=I).astype(float)
    sum_m = np.zeros((M, Yv.shape[1])); np.add.at(sum_m, idx_m, Yv)
    sum_l = np.zeros((I, Yv.shape[1])); np.add.at(sum_l, idx_l, Yv)
    mean_m, mean_l = sum_m / cnt_m[:, None], sum_l / cnt_l[:, None]
    grand = Yv.mean(axis=0)

    # which lab each mouse belongs to (the nesting; asserted, not assumed)
    lab_of_mouse = pd.DataFrame({'m': idx_m, 'l': idx_l}).drop_duplicates()
    assert lab_of_mouse['m'].is_unique, 'a mouse appears in more than one lab'
    lom = lab_of_mouse.sort_values('m')['l'].to_numpy()

    SSE = ((Yv - mean_m[idx_m]) ** 2).sum(axis=0)              # within mouse
    SSB = (cnt_m[:, None] * (mean_m - mean_l[lom]) ** 2).sum(axis=0)   # mice within lab
    SSA = (cnt_l[:, None] * (mean_l - grand) ** 2).sum(axis=0)         # between labs
    MSE, MSB, MSA = SSE / (N - M), SSB / (M - I), SSA / (I - 1)

    n_ij2_over_ni = sum((cnt_m[lom == i] ** 2).sum() / cnt_l[i] for i in range(I))
    k1 = (N - n_ij2_over_ni) / (M - I)
    k2 = (n_ij2_over_ni - (cnt_m ** 2).sum() / N) / (I - 1)
    k3 = (N - (cnt_l ** 2).sum() / N) / (I - 1)

    s2_session = MSE
    s2_mouse = (MSB - MSE) / k1
    s2_lab = (MSA - MSE - k2 * s2_mouse) / k3
    out = pd.DataFrame({'sigma2_lab': s2_lab, 'sigma2_mouse': s2_mouse,
                        'sigma2_session': s2_session},
                       index=getattr(Y, 'columns', range(Yv.shape[1])))
    if truncate:
        out = out.clip(lower=0)
    return out, dict(k1=k1, k2=k2, k3=k3, N=N, n_mice=M, n_labs=I)


def variance_shares(vc):
    """Per-feature components -> shares of TOTAL variance, pooled over features.

    Pooling is a trace decomposition: sum each component over features, then divide. That
    weights features by how much variance they carry, which is what 'share of behavioural
    variance' should mean; averaging per-feature ratios instead would give a rarely-used
    feature the same vote as a dominant one. Negative estimates are clipped at 0 FOR THE
    SHARES ONLY -- the raw sums are returned alongside so the clipping is visible.
    """
    raw = vc.sum(axis=0)
    pos = vc.clip(lower=0).sum(axis=0)
    shares = pos / pos.sum()
    return pd.DataFrame({'sigma2_sum_raw': raw, 'sigma2_sum_clipped': pos,
                         'share': shares})


def eta2(Y, groups):
    """Naive between-group share of variance, pooled over features (trace). Biased upwards
    -- compare against permutation_null_eta2 before reading anything into it."""
    Yv = np.asarray(Y, float)
    if Yv.ndim == 1:
        Yv = Yv[:, None]
    g = pd.Series(np.asarray(groups, dtype=object)).to_numpy()
    grand = Yv.mean(axis=0)
    ss_tot = ((Yv - grand) ** 2).sum(axis=0).sum()
    ss_bet = 0.0
    for lev in pd.unique(g):
        m = g == lev
        ss_bet += m.sum() * ((Yv[m].mean(axis=0) - grand) ** 2).sum()
    return float(ss_bet / ss_tot)


def permutation_null_eta2(Y, groups, units, n_perm=1000, seed=0):
    """Null distribution of eta2(groups) when the group label is shuffled ACROSS UNITS.

    `units` is the exchangeable unit -- mice, for a lab effect. Shuffling SESSIONS would
    be pseudoreplication: sessions of one mouse are not independent, and the null would be
    far too tight. The label travels with the whole mouse, so the null preserves both the
    nesting and the unequal session counts.
    """
    rng = np.random.default_rng(seed)
    units = pd.Series(np.asarray(units, dtype=object)).reset_index(drop=True)
    groups = pd.Series(np.asarray(groups, dtype=object)).reset_index(drop=True)
    unit_to_group = groups.groupby(units).first()
    obs = eta2(Y, groups)
    null = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = pd.Series(rng.permutation(unit_to_group.to_numpy()),
                             index=unit_to_group.index)
        null[p] = eta2(Y, units.map(shuffled))
    return obs, null, float((null >= obs).mean())


def bootstrap_shares(Y, lab, mouse, n_boot=500, seed=0):
    """Resample LABS with replacement and redo the decomposition, for a CI on the shares.

    Labs are the unit that limits precision here (10 of them, 9 df for the lab component),
    so they are what has to be resampled -- a bootstrap over sessions would report a
    confidence interval an order of magnitude too narrow.
    """
    rng = np.random.default_rng(seed)
    lab = pd.Series(np.asarray(lab, dtype=object)).reset_index(drop=True)
    mouse = pd.Series(np.asarray(mouse, dtype=object)).reset_index(drop=True)
    Yv = np.asarray(Y, float)
    labs = pd.unique(lab)
    rows = []
    for b in range(n_boot):
        pick = rng.choice(labs, len(labs), replace=True)
        idx, new_lab, new_mouse = [], [], []
        for rep, l in enumerate(pick):
            where = np.where((lab == l).to_numpy())[0]
            idx.extend(where)
            # a lab drawn twice becomes two distinct labs with distinct mice, otherwise
            # the duplicate rows would masquerade as a perfectly reproducible lab
            new_lab.extend([f'{l}#{rep}'] * len(where))
            new_mouse.extend([f'{m}#{rep}' for m in mouse.iloc[where]])
        try:
            vc, _ = nested_variance_components(Yv[idx], new_lab, new_mouse)
            rows.append(variance_shares(vc)['share'])
        except (ValueError, AssertionError, ZeroDivisionError):
            continue
    return pd.DataFrame(rows).reset_index(drop=True)
