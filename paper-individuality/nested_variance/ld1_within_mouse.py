"""
IS LD1'S DIRECTION ALSO A DIRECTION OF DAY-TO-DAY VARIATION?
============================================================
Unit: the session, its 360-feature syllable profile (the LDA's design matrix, within_between_models.session_profiles).
Two covariance matrices of those profiles:
    S_W   within-mouse: pooled covariance of each session's deviation from its own mouse's mean (day to day,
          plus the ~2% that is trial sampling noise)
    S_M   between-mouse: covariance of the mouse means with the expected within-mouse share removed,
          (H / (m - 1) - S_W) / n0, n0 the unbalanced-design mean group size (one-way random-effects estimator)

0  LD1 DAY TO DAY. The saved LD1 per session: within-mouse SD, SD of mouse means, ICC. Then LD1's direction w (the
   refit's, unit length, in the 360-feature space): the share of all day-to-day variance that lies along it,
   w'S_W w / tr(S_W), and where that sits in the within-mouse eigenspectrum (wPC1 = 23%).

1  PROPORTIONALITY. If individuality were just "more of the same day-to-day variation", S_M = c S_W and every
   generalised eigenvalue theta of (S_M, S_W) would equal c: there would be no preferred discriminant direction.
   Statistic: theta_1 / mean(theta), scale-free. Null: mouse effects drawn as N(0, c S_W) with c = mean(theta),
   added to the observed within-mouse residuals resampled across sessions, on the real design (who has how many
   sessions); everything recomputed. Done after PCA to K dimensions (S_W has sessions - mice df), for several K.
   The leading generalised eigenvector is unshrunk LDA in K dimensions; its scores are compared with LD1.

2  HELD OUT. The in-sample share of 0 is biased low: LDA chose w to have little within-mouse variance in these very
   sessions. So: split each mouse's sessions at random into a fit half (floor(n/2)) and an evaluation half (the
   rest, >= 2), fit on one, measure on the other, N_SPLIT times. Directions compared, each fitted on the fit half:
       LD1           the embedding's mouse-weighted shrinkage LDA (shrinkage 0.5)
       PC1           first principal component of all sessions (no labels)
       wPC1          first principal component of the within-mouse deviations
       shuffled LD1  the same LDA with sessions shuffled across mice (what LDA finds with no mouse structure)
   Measured on the evaluation half: share of day-to-day variance along the direction, mouse variance along it,
   and the ICC S_M / (S_M + S_W) along it.

Writes figures/ld1_within_mouse.{csv,png,pdf}.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import within_between_models as wb      # noqa: E402  (paths, numpy shim)
import paper_style as ps                # noqa: E402

N_SIM, N_SPLIT, SEED = 2000, 500, 0
KS = [5, 10, 20, 30]
FIGDIR = HERE / 'figures'
CACHE = HERE / 'cache' / 'ld1_within_mouse.pkl'
COLORS = {'LD1': '#E0614A', 'PC1': '#22344A', 'wPC1': '#3E8E54', 'shuffled LD1': '#AAAAAA'}


# ---------------------------------------------------------------------------------------------------------------
# covariances
# ---------------------------------------------------------------------------------------------------------------
def covariances(X, g):
    """-> S_W, S_M, H/(m-1), n0 for rows X grouped by integer codes g (0..m-1)."""
    m = g.max() + 1
    n = np.bincount(g, minlength=m).astype(float)
    means = np.zeros((m, X.shape[1]))
    np.add.at(means, g, X)
    means /= n[:, None]
    R = X - means[g]
    S_W = R.T @ R / (len(X) - m)
    D = (means - X.mean(0)) * np.sqrt(n)[:, None]
    MS_B = D.T @ D / (m - 1)
    n0 = (n.sum() - (n ** 2).sum() / n.sum()) / (m - 1)
    return S_W, (MS_B - S_W) / n0, MS_B, n0


def along(d, S):
    d = d / np.linalg.norm(d)
    return d @ S @ d


def shrinkage_lda(X, g, s=wb.SHRINKAGE):
    """lda_shrinkage's mouse-weighted embedding (as within_between_models.axis_scores' refit) -> unit LD1."""
    m = g.max() + 1
    M = np.stack([X[g == c].mean(0) for c in range(m)])
    Sb = np.cov((M - M.mean(0)).T, bias=True)
    R = X - M[g]
    Sw = R.T @ R / (len(X) - m)
    Sw = (1 - s) * Sw + s * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    ev, V = linalg.eigh(Sb, Sw)
    w = V[:, np.argmax(ev)]
    return w / np.linalg.norm(w)


def top_eigvec(S):
    ev, V = np.linalg.eigh(S)
    return V[:, -1]


# ---------------------------------------------------------------------------------------------------------------
# 0  LD1 day to day
# ---------------------------------------------------------------------------------------------------------------
def ld1_day_to_day(X, g, ld1):
    S_W, S_M, _, _ = covariances(X, g)
    sw, sm, _, _ = covariances(ld1[:, None], g)
    w = shrinkage_lda(X, g)
    ev_W = np.linalg.eigvalsh(S_W)[::-1]
    v = along(w, S_W)
    per_mouse = pd.Series(ld1).groupby(g).std(ddof=1)
    out = dict(ld1_within_sd=np.sqrt(sw[0, 0]), ld1_mouse_sd=np.sqrt(max(sm[0, 0], 0)),
               ld1_icc=sm[0, 0] / (sm[0, 0] + sw[0, 0]),
               ld1_sd_of_mouse_means=pd.Series(ld1).groupby(g).mean().std(ddof=1),
               ld1_range_of_mouse_means=np.ptp(pd.Series(ld1).groupby(g).mean()),
               per_mouse_sd_median=per_mouse.median(), per_mouse_sd_q10=per_mouse.quantile(0.1),
               per_mouse_sd_q90=per_mouse.quantile(0.9),
               share_dtd_along_LD1=v / np.trace(S_W), share_dtd_wPC1=ev_W[0] / np.trace(S_W),
               ld1_dtd_rank=int((ev_W > v).sum()) + 1, dtd_ratio_to_wPC1=v / ev_W[0],
               share_mouse_along_LD1=along(w, S_M) / np.trace(S_M),
               share_mouse_along_mPC1=np.linalg.eigvalsh(S_M)[-1] / np.trace(S_M))
    return out, per_mouse


# ---------------------------------------------------------------------------------------------------------------
# 1  proportionality
# ---------------------------------------------------------------------------------------------------------------
def gen_eig(S_M, S_W):
    th = linalg.eigh(S_M, S_W, eigvals_only=True)[::-1]
    return th


def proportionality(X, g, ld1, rng):
    rows, spectra = [], {}
    m = g.max() + 1
    for K in KS:
        Z = PCA(K).fit_transform(X)
        S_W, S_M, _, _ = covariances(Z, g)
        th = gen_eig(S_M, S_W)
        c = th.mean()
        ev, V = linalg.eigh(S_M, S_W)
        r_ld1 = abs(np.corrcoef(Z @ V[:, -1], ld1)[0, 1])
        # null: S_M = c S_W, with the real within-mouse residuals
        means = np.stack([Z[g == k].mean(0) for k in range(m)])
        n = np.bincount(g)
        R = (Z - means[g]) * np.sqrt(n / (n - 1))[g][:, None]
        L = np.linalg.cholesky(c * S_W) if c > 0 else None
        null = np.empty((N_SIM, K))
        for i in range(N_SIM):
            u = rng.standard_normal((m, K)) @ L.T
            Zs = u[g] + R[rng.integers(0, len(R), len(R))]
            sw, sm, _, _ = covariances(Zs, g)
            null[i] = gen_eig(sm, sw) / gen_eig(sm, sw).mean()
        stat = th[0] / c
        p = (1 + (null[:, 0] >= stat).sum()) / (1 + N_SIM)
        n_above = 0
        for k in range(K):                       # leading run of eigenvalues above their null's 95th percentile
            if th[k] / c > np.quantile(null[:, k], 0.95):
                n_above += 1
            else:
                break
        rows.append(dict(K=K, var_explained=PCA(K).fit(X).explained_variance_ratio_.sum(), mean_theta=c,
                         theta1_over_mean=stat, null_median=np.median(null[:, 0]),
                         null_q95=np.quantile(null[:, 0], 0.95), p=p, n_dims_above_null=n_above,
                         r_top_vector_vs_LD1=r_ld1))
        spectra[K] = (th / c, np.quantile(null, [0.05, 0.5, 0.95], axis=0))
    return pd.DataFrame(rows), spectra


# ---------------------------------------------------------------------------------------------------------------
# 2  held out
# ---------------------------------------------------------------------------------------------------------------
def held_out(X, g, rng):
    m = g.max() + 1
    idx_by_mouse = [np.flatnonzero(g == k) for k in range(m)]
    rows = []
    for split in range(N_SPLIT):
        fit, ev_ = [], []
        for ix in idx_by_mouse:
            ix = rng.permutation(ix)
            k = len(ix) // 2
            fit.append(ix[:k]); ev_.append(ix[k:])
        fit, ev_ = np.concatenate(fit), np.concatenate(ev_)
        Xf, gf, Xe, ge = X[fit], g[fit], X[ev_], g[ev_]
        Sf_W, _, _, _ = covariances(Xf, gf)
        dirs = {'LD1': shrinkage_lda(Xf, gf), 'PC1': top_eigvec(np.cov(Xf.T)), 'wPC1': top_eigvec(Sf_W),
                'shuffled LD1': shrinkage_lda(Xf, rng.permutation(gf))}
        S_W, S_M, _, _ = covariances(Xe, ge)
        lam1 = np.linalg.eigvalsh(S_W)[-1]
        for name, d in dirs.items():
            vw, vm = along(d, S_W), along(d, S_M)
            rows.append(dict(split=split, direction=name, share_dtd=vw / np.trace(S_W), dtd_ratio_to_wPC1=vw / lam1,
                             share_mouse=vm / np.trace(S_M), icc=vm / (vm + vw)))
    return pd.DataFrame(rows)


def in_sample(X, g):
    S_W, S_M, _, _ = covariances(X, g)
    lam1 = np.linalg.eigvalsh(S_W)[-1]
    dirs = {'LD1': shrinkage_lda(X, g), 'PC1': top_eigvec(np.cov(X.T)), 'wPC1': top_eigvec(S_W)}
    return pd.DataFrame([dict(direction=k, share_dtd=along(d, S_W) / np.trace(S_W),
                              dtd_ratio_to_wPC1=along(d, S_W) / lam1, share_mouse=along(d, S_M) / np.trace(S_M),
                              icc=along(d, S_M) / (along(d, S_M) + along(d, S_W))) for k, d in dirs.items()])


# ---------------------------------------------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(SEED)
    X, s_ids, mouse, _ = wb.session_profiles()
    g = pd.factorize(mouse)[0]
    lda = pd.read_pickle(wb.LDA_FILE)
    ld1 = lda.set_index('session').loc[s_ids, 0].to_numpy(float)
    n = np.bincount(g)
    print(f'{len(X)} sessions, {g.max() + 1} mice; sessions per mouse: '
          + ', '.join(f'{k}: {v}' for k, v in sorted(pd.Series(n).value_counts().items())))
    day, per_mouse = ld1_day_to_day(X, g, ld1)
    prop, spectra = proportionality(X, g, ld1, rng)
    ho = held_out(X, g, rng)
    ins = in_sample(X, g)
    pd.to_pickle(dict(day=day, per_mouse=per_mouse, ld1=ld1, mouse=mouse, prop=prop, spectra=spectra, ho=ho,
                      ins=ins), CACHE)
    return day, per_mouse, ld1, mouse, prop, spectra, ho, ins


def report(day, prop, ho, ins):
    print('\n0  LD1 day to day')
    for k, v in day.items():
        print(f'   {k:28s} {v:.4g}')
    print('\n1  proportionality S_M = c S_W')
    with pd.option_context('display.width', 200):
        print(prop.round(3).to_string(index=False))
    print('\n2  held out (mean [2.5, 97.5 %] over splits) vs in sample')
    q = ho.groupby('direction').agg(['mean', lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    for d in COLORS:
        r = q.loc[d]
        txt = '  '.join(f'{c} {r[(c, "mean")]:.3f} [{r[(c, "<lambda_0>")]:.3f}, {r[(c, "<lambda_1>")]:.3f}]'
                        for c in ('share_dtd', 'dtd_ratio_to_wPC1', 'share_mouse', 'icc'))
        print(f'   {d:13s} {txt}')
    with pd.option_context('display.width', 200):
        print(ins.round(3).to_string(index=False))
    frac = (ho.pivot(index='split', columns='direction', values='icc').eval('`LD1` > `PC1`')).mean()
    print(f'   held-out ICC: LD1 > PC1 in {frac:.1%} of splits')


def figure(day, per_mouse, ld1, mouse, prop, spectra, ho):
    ps.use('paper')
    fs = plt.rcParams['font.size']
    fig, axs = plt.subplots(1, 4, figsize=(7.09, 2.2), gridspec_kw={'wspace': 0.55, 'width_ratios': [1.5, 1, 1, 1]})

    # a  LD1 per session, mice ordered by their mean
    ax = axs[0]
    d = pd.DataFrame({'mouse': mouse, 'ld1': ld1})
    order = d.groupby('mouse')['ld1'].mean().sort_values().index
    pos = pd.Series(np.arange(len(order)), index=order)
    ax.scatter(pos[d['mouse']], d['ld1'], s=3, color=COLORS['LD1'], alpha=0.5, lw=0)
    mm = d.groupby('mouse')['ld1'].agg(['min', 'max', 'mean']).loc[order]
    ax.vlines(np.arange(len(order)), mm['min'], mm['max'], color=COLORS['LD1'], lw=0.5, alpha=0.6)
    ax.scatter(np.arange(len(order)), mm['mean'], s=4, color='k', lw=0, zorder=3)
    ax.set_xticks([])
    ax.set_xlabel(f'{len(order)} mice, by mean LD1')
    ax.set_ylabel('LD1 (saved embedding)')
    ax.set_title(f"ICC = {day['ld1_icc']:.2f}; within-mouse SD {day['ld1_within_sd']:.2f}\n"
                 f"vs SD of mouse means {day['ld1_sd_of_mouse_means']:.2f}", fontsize=fs * 0.75)

    # b  generalised eigenvalue spectrum vs proportional null (K = 10)
    ax = axs[1]
    K = 10
    th, (lo, md, hi) = spectra[K]
    x = np.arange(1, K + 1)
    ax.fill_between(x, lo, hi, color='0.85', lw=0, label=r'if $S_M \propto S_W$')
    ax.plot(x, md, color='0.6', lw=0.8)
    ax.plot(x, th, 'o-', color=COLORS['LD1'], ms=2.5, lw=0.8, label='observed')
    r = prop.set_index('K').loc[K]
    ax.set_title(rf"$\theta_1$/mean = {r['theta1_over_mean']:.1f}, p = {r['p']:.3g}", fontsize=fs * 0.75)
    ax.set_xlabel(f'Discriminant dimension ({K} PCs)')
    ax.set_ylabel('Mouse / day-to-day variance\n(relative to mean)')
    ax.legend(frameon=False, fontsize=fs * 0.6, loc='upper right')

    # c, d  held out
    names = list(COLORS)
    for ax, col, lab in ((axs[2], 'share_dtd', 'Share of day-to-day\nvariance along axis'),
                         (axs[3], 'icc', 'ICC along axis\n(held-out sessions)')):
        for i, nm in enumerate(names):
            v = ho.loc[ho.direction == nm, col]
            ax.errorbar(i, v.mean(), yerr=[[v.mean() - v.quantile(0.025)], [v.quantile(0.975) - v.mean()]],
                        fmt='o', color=COLORS[nm], ms=3.5, elinewidth=0.8, capsize=0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([nm.replace('shuffled ', 'shuf.\n') for nm in names], fontsize=fs * 0.7)
        ax.set_ylabel(lab)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(bottom=0)
    for ax, letter in zip(axs, 'abcd'):
        ax.text(-0.3 if letter == 'a' else -0.45, 1.12, letter, transform=ax.transAxes, fontsize=fs + 2,
                fontweight='bold')
    fig.text(0.5, -0.12, 'c, d: axes fitted on half of each mouse\'s sessions, measured on the other half '
                         f'({N_SPLIT} random splits; mean and 95% range).', ha='center', fontsize=fs * 0.65, color='0.35')
    for path in (FIGDIR / 'ld1_within_mouse.png', FIGDIR / 'ld1_within_mouse.pdf'):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', path.relative_to(wb.PI))


if __name__ == '__main__':
    if CACHE.exists() and 'rerun' not in sys.argv:
        c = pd.read_pickle(CACHE)
        day, per_mouse, ld1, mouse, prop, spectra, ho, ins = (c[k] for k in ('day', 'per_mouse', 'ld1', 'mouse',
                                                                            'prop', 'spectra', 'ho', 'ins'))
    else:
        day, per_mouse, ld1, mouse, prop, spectra, ho, ins = run()
    report(day, prop, ho, ins)
    pd.concat([pd.DataFrame([day]).assign(part='0'), prop.assign(part='1'),
               ho.groupby('direction').mean().drop(columns='split').reset_index().assign(part='2 held out'),
               ins.assign(part='2 in sample')]).to_csv(FIGDIR / 'ld1_within_mouse.csv', index=False)
    figure(day, per_mouse, ld1, mouse, prop, spectra, ho)
