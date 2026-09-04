"""Does the LDA axis change the SHAPE of a neuron's encoding kernels?

Neurons are the samples here, and the outcome is the kernel itself rather than a scalar
summary of the fit. That is the gap this fills:

  * `encoding_functions.lda1_perm_bf` asks whether LDA1 predicts HOW WELL a neuron is
    explained (cv_r2, dR2_task) -- one number per neuron.
  * `population_encoding` asks whether LDA1 changes the REGION-AVERAGE response.
    Averaging cancels any shape change whose sign differs across neurons, and the
    CA1 result there says the effect is not a uniform gain (the 1-column gain test is
    flat at p~0.68 while the 78-column per-kernel block is weakly positive).
  * This module asks whether LDA1 predicts the SHAPE of the response -- its timing and
    relative weighting. Two neurons whose stimOn kernels peak at 200 ms and 400 ms have
    identical cv_r2 and identical amplitude; every test above is blind to them.

Method (see the docstrings of each function for the algebra)
-----------------------------------------------------------
  0. one ridge coefficient vector per neuron, COMMON alpha across sessions
  1. keep the 42 TASK basis columns (identical in every session, unlike the motor-state
     one-hots whose count varies), normalise to unit norm, centre across neurons
     -> Phi (n_neurons x 42), a pure "shape" matrix
  2. residualise Phi and LDA1 on nuisances (session length, firing rate, yield)
  3. omnibus: multiple correlation R2 between LDA1 and the 42 shape dimensions
  4. null: permute the session -> LDA1 assignment (Phi held fixed); the effective N is
     the number of sessions, not the number of neurons, and the permutation is what
     encodes that
  5. localisation per coefficient with BH-FDR, only if step 3 survives

Because neurons are the samples, there is no MIN_NEURONS filter: every session with at
least one neuron in the region contributes (CA1: ~880 neurons over 65 sessions / 42 mice,
against the 43 sessions the population model could use).
"""
import os
import glob
import pickle

import numpy as np
import pandas as pd

import encoding_functions as ef
import population_encoding as pe

# The 42 task basis columns, in a fixed order. Present in EVERY session, which is what
# makes the coefficient vectors comparable across animals (the motor-state one-hots are
# not: a session with 3 whisker states has more columns than one with 2).
TASK_COLS = [f'{name}_b{k}' for name, spec in ef.DEFAULT_KERNELS.items()
             for k in range(spec['k'])]


# ---------------------------------------------------------------------------
# Step 0: per-neuron coefficients at a common alpha
# ---------------------------------------------------------------------------
def common_alpha(results_dir='encoding_results', region=None):
    """Median of the per-session CV-selected ridge alphas.

    A COMMON alpha is essential: W_s = (X'X + alpha I)^-1 X'y, so alpha sets how hard
    the kernels are shrunk. Let it vary by session and part of any "shape effect" is a
    shrinkage effect that tracks session length rather than biology. Even at fixed alpha
    the shrinkage still depends on X_s'X_s, so `shape_omnibus` should be re-run at
    alpha/10 and alpha*10 as a robustness check.
    """
    d = pd.concat([pd.read_parquet(f) for f in
                   sorted(glob.glob(os.path.join(results_dir, '*.parquet')))],
                  ignore_index=True)
    if region is not None:
        d['beryl'] = d['area'].astype(str).map(pe.beryl_map(d['area'].astype(str).unique()))
        d = d[d['beryl'] == region]
    return float(np.median(d['best_alpha'].dropna()))


def sweep_kernels(sessions, neuron_dir, region, cache_dir, alpha,
                  motor_continuous=False, motor_lags=True, verbose=True):
    """One ridge fit per session (no CV loop, no shuffles -> seconds per session),
    cached as one parquet per session. Returns a DataFrame with ONE ROW PER NEURON and
    the 42 task coefficients as columns.
    """
    os.makedirs(cache_dir, exist_ok=True)
    bmap, out = None, []
    grp = sessions.groupby('session')
    for i, (eid, g) in enumerate(grp, 1):
        cache = os.path.join(cache_dir, f'{eid}.parquet')
        if os.path.exists(cache):
            out.append(pd.read_parquet(cache))
            continue
        try:
            df, cols = pe.load_session(list(g['pid']), neuron_dir, region, bmap)
            if bmap is None:
                raw = [c.split('_neuron_')[0] for c in df.columns
                       if c.endswith('_spike_count')]
                bmap = pe.beryl_map(sorted(set(raw)))
            if not cols:
                continue
            X, groups, trial_ids, keep = ef.build_design_matrix(
                df, motor_continuous=motor_continuous, motor_lags=motor_lags)
            Y = df.loc[keep, cols].values.astype(float)
            W, colnames, ymu = ef.fit_coefficients(X, Y, alpha)
            idx = [colnames.index(c) for c in TASK_COLS]
            r = pd.DataFrame(W[idx].T, columns=TASK_COLS)
            r.insert(0, 'neuron', cols)
            r.insert(1, 'session', eid)
            r.insert(2, 'mouse_name', g['mouse_name'].iloc[0])
            r['n_bins'] = int(keep.sum())
            r['mean_rate'] = Y.mean(axis=0)
            r['n_neurons'] = len(cols)
            r['alpha'] = alpha
            r.to_parquet(cache)
            out.append(r)
            if verbose:
                print(f'[{i}/{len(grp)}] {eid[:8]}  {len(cols):3d} neurons', flush=True)
        except Exception as e:
            print(f'[{i}/{len(grp)}] {eid[:8]} FAILED: {type(e).__name__}: {e}', flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Steps 1-2: shape matrix and nuisance residualisation
# ---------------------------------------------------------------------------
def shape_matrix(K, cols=None, normalize='unit', min_norm=None):
    """Phi: unit-norm, neuron-centred task-coefficient matrix.

    normalize='unit' divides each neuron's vector by its L2 norm, so "this neuron
    responds more" is removed and only direction -- timing and relative weighting --
    survives. Amplitude is already covered by firing_rate/fr_psth_ldabin.ipynb.
    normalize=None keeps amplitude (then the test conflates shape and gain).

    `min_norm` drops neurons whose raw coefficient norm is below a quantile of the
    distribution: after unit-norm scaling an unmodulated neuron becomes a pure noise
    direction, which only dilutes the signal. Pass e.g. 0.25 to drop the bottom quartile.
    """
    cols = cols or TASK_COLS
    P = K[cols].values.astype(float)
    nrm = np.linalg.norm(P, axis=1)
    keep = np.isfinite(nrm) & (nrm > 0)
    if min_norm:
        keep &= nrm >= np.quantile(nrm[keep], min_norm)
    P = P[keep]
    if normalize == 'unit':
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
    P = P - P.mean(axis=0)
    return P, K.loc[keep].reset_index(drop=True), nrm[keep]


def _residualize(Y, Z):
    """Frisch-Waugh-Lovell: strip the column space of Z out of Y."""
    return Y - Z @ (np.linalg.pinv(Z) @ Y)


def _nuisance(K, covars=('n_bins', 'mean_rate', 'n_neurons')):
    cols = [np.ones(len(K))]
    for c in covars:
        v = K[c].values.astype(float)
        if c == 'mean_rate':
            v = np.log1p(v)
        cols.append((v - v.mean()) / (v.std(ddof=1) or 1.0))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Steps 3-4: omnibus statistic + session-permutation null
# ---------------------------------------------------------------------------
def _multiple_r2(Phi, x):
    """R2 of regressing x on Phi.

        R2 = x' Phi (Phi'Phi)^-1 Phi' x / (x'x)

    x is the scalar (LDA1) and Phi the 42 shape dimensions -- this direction gives ONE
    number in which (Phi'Phi)^-1 already accounts for the correlations among the 42
    dimensions. The reverse direction would give 42 slopes needing an arbitrary metric
    to combine. Upward-biased with 42 columns, which is irrelevant: the null below is
    built from the same 42.
    """
    G = Phi.T @ Phi
    v = Phi.T @ x
    return float(v @ np.linalg.solve(G + 1e-10 * np.eye(len(G)), v) / (x @ x))


def shape_omnibus(K, lda1, normalize='unit', min_norm=0.25, cols=None,
                  covars=('n_bins', 'mean_rate', 'n_neurons'),
                  level='session', n_perm=2000, seed=0):
    """One test: does LDA1 depend on kernel shape at all?

    The null permutes the session -> LDA1 assignment, holding Phi completely fixed. That
    preserves the block structure (all neurons of a session share one LDA1) and the
    session sizes, so the reference distribution carries the correct effective N -- the
    number of sessions, not the number of neurons. A parametric F-test here would treat
    ~880 neurons as independent and return a meaningless p-value.

    level='mouse' averages LDA1 within mouse and permutes across mice.
    """
    K = K.merge(lda1, on='session', how='inner').dropna(subset=['lda_1'])
    Phi, K, nrm = shape_matrix(K, cols, normalize, min_norm)
    Z = _nuisance(K, covars)
    x = K['lda_1'].values.astype(float)
    if level == 'mouse':
        x = K.groupby('mouse_name')['lda_1'].transform('mean').values.astype(float)

    Phi_r = _residualize(Phi, Z)
    x_r = _residualize(x[:, None], Z)[:, 0]
    obs = _multiple_r2(Phi_r, x_r)

    unit = 'mouse_name' if level == 'mouse' else 'session'
    key = K[unit].values
    vals = K.groupby(unit)['lda_1'].first()
    if level == 'mouse':
        vals = K.groupby('mouse_name')['lda_1'].mean()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sh = pd.Series(rng.permutation(vals.values), index=vals.index)
        xp = sh.reindex(key).values.astype(float)
        null[i] = _multiple_r2(Phi_r, _residualize(xp[:, None], Z)[:, 0])
    p = float((1 + np.sum(null >= obs)) / (1 + n_perm))
    return dict(r2=obs, null=null, p=p, null_mean=float(null.mean()),
                null_sd=float(null.std(ddof=1)),
                z=float((obs - null.mean()) / null.std(ddof=1)),
                n_neurons=len(K), n_sessions=int(K['session'].nunique()),
                n_mice=int(K['mouse_name'].nunique()), level=level,
                n_dims=Phi.shape[1], normalize=normalize, K=K, Phi=Phi)


# ---------------------------------------------------------------------------
# Step 5: localisation (only if the omnibus survives)
# ---------------------------------------------------------------------------
def shape_localize(res, n_perm=2000, seed=0):
    """Per-coefficient partial slope of shape on LDA1, with the same session
    permutation and BH-FDR across the 42 coefficients.

        beta_j = (x_r . Phi_r[:, j]) / (x_r . x_r)

    Only interpret this if `shape_omnibus` survived -- otherwise it is 42 shots at goal.
    """
    from statsmodels.stats.multitest import multipletests
    K, Phi = res['K'], res['Phi']
    Z = _nuisance(K)
    level = res['level']
    unit = 'mouse_name' if level == 'mouse' else 'session'
    x = (K.groupby('mouse_name')['lda_1'].transform('mean') if level == 'mouse'
         else K['lda_1']).values.astype(float)
    Phi_r = _residualize(Phi, Z)
    x_r = _residualize(x[:, None], Z)[:, 0]
    beta = (Phi_r.T @ x_r) / (x_r @ x_r)

    vals = (K.groupby('mouse_name')['lda_1'].mean() if level == 'mouse'
            else K.groupby('session')['lda_1'].first())
    key = K[unit].values
    rng = np.random.default_rng(seed)
    cnt = np.zeros(Phi.shape[1])
    for _ in range(n_perm):
        sh = pd.Series(rng.permutation(vals.values), index=vals.index)
        xp = _residualize(sh.reindex(key).values.astype(float)[:, None], Z)[:, 0]
        cnt += np.abs((Phi_r.T @ xp) / (xp @ xp)) >= np.abs(beta)
    p = (1 + cnt) / (1 + n_perm)
    out = pd.DataFrame(dict(column=TASK_COLS[:Phi.shape[1]], beta=beta, p=p))
    out['family'] = out['column'].str.replace(r'_b\d+$', '', regex=True)
    out['q'] = multipletests(out['p'], method='fdr_bh')[1]
    return out.sort_values('p').reset_index(drop=True)


def plot_kernels_by_lda(res, family='stimOn', n_bins=3, ax=None):
    """Mean time-domain kernel per LDA1 tercile -- the picture behind the statistic."""
    import matplotlib.pyplot as plt
    K, Phi = res['K'], res['Phi']
    x = K['lda_1'].values
    tercile = pd.qcut(x, n_bins, labels=False, duplicates='drop')
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.8))
    colors = plt.cm.viridis(np.linspace(0, .85, n_bins))
    for b in range(n_bins):
        m = Phi[tercile == b].mean(axis=0)
        t, k = ef.event_kernel(family, m, TASK_COLS[:Phi.shape[1]])
        ax.plot(t, k, color=colors[b], lw=2, label=f'LDA1 tercile {b + 1}')
    ax.axhline(0, color='k', lw=.6)
    ax.set(xlabel='time from event (s)', ylabel='shape-normalised weight',
           title=f'{family} kernel by LDA1 ({res["n_neurons"]} neurons, '
                 f'{res["n_mice"]} mice)')
    ax.legend(fontsize=8, frameon=False)
    return ax
