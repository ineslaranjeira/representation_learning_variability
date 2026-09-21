"""
ALIGNING TWO LDA RUNS, WITH AND WITHOUT LAB CORRECTION -- the machinery
=======================================================================
Companion module to `lda_lab_correction_alignment.ipynb`, which is where the analysis,
the narrative and the results live. This file holds only the pieces the notebook calls:
loading, the matching, the subspace measures, and the four figures. Nothing here prints
a conclusion.

THE THREE THINGS THAT MAKE THE NAIVE COMPARISON WRONG, all handled below:

  ORDER    LDA components come out ordered by eigenvalue, and the correction changes the
           eigenvalues, so raw LD4 may be corrected LD2. `match()` therefore compares
           every raw axis against every corrected axis and takes the best one-to-one
           assignment (Hungarian on |r|) rather than assuming LD1 goes with LD1.
  SIGN     an LDA axis and its negative are the same axis. Matching is on |r| and the
           sign comes back separately, to be applied before anything is plotted --
           otherwise the movement arrows point backwards for half the axes.
  ROTATION near-tied eigenvalues leave the leading axes free to rotate among themselves,
           so an axis can fail to find a partner while the SUBSPACE it lives in is
           perfectly preserved. `subspace_overlap()` sees that; pairwise |r| cannot.

CHANCE. With 25 axes on 58 mice the best of 25! assignments is a large number even for
unrelated data, so `null_matched_r()` rebuilds the whole assignment on shuffled mice and
gives the matched |r| profile a chance level to be read against.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import orthogonal_procrustes, subspace_angles
from scipy.optimize import linear_sum_assignment

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                                  # .../paper-individuality
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import paper_style as ps                            # noqa: E402

RAW_FILE = 'clustering/data_files/mouse_LDA_5_bins_raw_25_20-09-2026'
LAB_FILE = 'clustering/data_files/mouse_LDA_5_bins_labzscore_25_20-09-2026'
N_DIMS = 25           # axes to align; the files carry 25
N_SHOW = 8            # axes to print and to draw scatter panels for
N_PERM = 2000         # shuffles of the mouse rows, for the chance level of a match
ALPHA = 0.05
FIG_SUBDIR = 'lab_correction'


# ------------------------------------------------------------------ loading
def load(path, n_dims):
    """One embedding, indexed by session, with columns LD1..LDn plus the metadata."""
    d = pd.read_pickle(ROOT / path)
    d = d.rename(columns={i: f'LD{i + 1}' for i in range(n_dims)}).set_index('session')
    return d


def mouse_level(d, lds):
    """Mouse means -- the unit the embedding is actually about, and the one that is not
    pseudo-replicated. Sessions of one mouse are not independent, so a correlation across
    269 sessions would count the same animal several times."""
    m = d.groupby('mouse_name')[lds].mean()
    m['lab'] = d.groupby('mouse_name')['lab'].first()
    return m


def zscore(X):
    """Per-column z-score. The corrected run has visibly smaller spread than the raw one,
    and that global scale difference is not what any of this is asking about."""
    return (X - X.mean(0)) / X.std(0)


# ------------------------------------------------------------------ alignment
def corr_matrix(A, B):
    """corr(A_i, B_j) for every pair, from z-scored columns."""
    Az, Bz = zscore(A), zscore(B)
    return (Az.T @ Bz) / len(Az)


def match(C):
    """Best one-to-one assignment of raw axes to corrected axes, maximising total |r|."""
    rows, cols = linear_sum_assignment(-np.abs(C))
    return rows, cols


def null_matched_r(A, B, n_perm=N_PERM, seed=0):
    """Chance level for the matched |r| profile. Shuffling the ROWS of B destroys the
    correspondence between the two embeddings while leaving both covariance structures
    and the assignment procedure exactly as they are -- so the null answers 'how good a
    match would this algorithm find in data that share nothing?'."""
    rng = np.random.default_rng(seed)
    Az, Bz = zscore(A), zscore(B)
    out = np.empty((n_perm, Az.shape[1]))
    for k in range(n_perm):
        Cp = (Az.T @ Bz[rng.permutation(len(Bz))]) / len(Az)
        r, c = linear_sum_assignment(-np.abs(Cp))
        out[k] = np.sort(np.abs(Cp[r, c]))[::-1]      # sorted best-to-worst
    return out


def subspace_overlap(A, B, ks):
    """For each k: how much of the leading-k raw subspace survives in the leading-k
    corrected one. mean cos^2 of the principal angles, 1 = the same subspace, 0 =
    orthogonal. This is the number that sees a rotation where pairwise |r| cannot."""
    Az, Bz = zscore(A), zscore(B)
    return np.array([float(np.mean(np.cos(subspace_angles(Az[:, :k], Bz[:, :k])) ** 2))
                     for k in ks])


def lab_eta2(M, lds):
    """Fraction of each axis's variance that a lab label explains, across mice. This is
    the quantity the correction is meant to remove, so it is the direct before/after."""
    out = {}
    for ld in lds:
        groups = [v[ld].to_numpy(float) for _, v in M.groupby('lab') if len(v) > 1]
        total = ((M[ld] - M[ld].mean()) ** 2).sum()
        within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        F, p = stats.f_oneway(*groups)
        out[ld] = dict(eta2=1 - within / total, F=F, p=p)
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ figures
def lab_colors(labs):
    cmap = plt.get_cmap('tab10')
    return {l: cmap(i % 10) for i, l in enumerate(sorted(labs))}


def fig_match(C, rows, cols, null, pairs, save):
    """The assignment itself: the full |r| matrix, and the matched values against chance."""
    fig, axes = plt.subplots(1, 2, figsize=(ps.SIZES[ps.MODE]['single'][0] * 2.2,
                                            ps.SIZES[ps.MODE]['single'][1] * 1.1))
    ax = axes[0]
    im = ax.imshow(np.abs(C), cmap=ps.SEQUENTIAL_CMAP, vmin=0, vmax=1,
                   aspect='auto', interpolation='none')
    ax.scatter(cols, rows, s=18, facecolors='none', edgecolors='w', linewidths=1.0)
    ax.set(xlabel='corrected axis', ylabel='raw axis',
           title='|r| between every pair\n(white rings = the chosen assignment)')
    ax.set_xticks(range(0, len(C), 4), [f'LD{i + 1}' for i in range(0, len(C), 4)])
    ax.set_yticks(range(0, len(C), 4), [f'LD{i + 1}' for i in range(0, len(C), 4)])
    fig.colorbar(im, ax=ax, label='|r|', fraction=0.045, pad=0.02)

    ax = axes[1]
    obs = pairs['abs_r'].to_numpy()
    x = np.arange(1, len(obs) + 1)
    lo, hi = np.percentile(null, [5, 95], axis=0)
    ax.fill_between(x, lo, hi, color='0.85', lw=0, label='chance, 5-95%')
    ax.plot(x, np.median(null, axis=0), color='0.55', lw=1.0, ls='--', label='chance median')
    ax.plot(x, obs, 'o-', color=ps.NEUTRAL, lw=1.3,
            ms=plt.rcParams['lines.markersize'] * 0.8, label='observed')
    for i, row in enumerate(pairs.itertuples()):
        if row.p_perm < ALPHA:
            ax.annotate(f'{row.raw}→{row.corrected}', (x[i], obs[i]),
                        textcoords='offset points', xytext=(0, 7), ha='center',
                        fontsize=plt.rcParams['font.size'] * 0.65, rotation=90, color='0.3')
    ax.set(xlabel='matched pair, best to worst', ylabel='|r| of the matched pair',
           title='Matched axes against chance', ylim=(0, 1.15))
    ax.legend(frameon=False, fontsize=plt.rcParams['font.size'] * 0.8, loc='upper right')
    plt.tight_layout()
    if save:
        ps.savefig(fig, 'lab_correction_axis_matching', subdir=FIG_SUBDIR)
    plt.show()


def fig_structure(ks, ov, eta_raw, eta_lab, lds, save):
    """What the correction removed (lab variance) and whether the subspace survived it."""
    fig, axes = plt.subplots(1, 2, figsize=(ps.SIZES[ps.MODE]['single'][0] * 2.2,
                                            ps.SIZES[ps.MODE]['single'][1] * 1.05))
    ax = axes[0]
    ax.plot(ks, ov, 'o-', color=ps.NEUTRAL, lw=1.3,
            ms=plt.rcParams['lines.markersize'] * 0.7)
    ax.axhline(1, color='0.85', lw=0.8, zorder=0)
    ax.set(xlabel='leading k axes', ylabel='mean cos² of the principal angles',
           title='Does the leading SUBSPACE survive?\n1 = same subspace, 0 = orthogonal',
           ylim=(0, 1.05))

    ax = axes[1]
    n = min(len(lds), 12)
    x = np.arange(n)
    ax.bar(x - 0.2, eta_raw.eta2[:n], width=0.4, color='0.35', label='raw')
    ax.bar(x + 0.2, eta_lab.eta2[:n], width=0.4, color=ps.NEUTRAL, alpha=0.75,
           label='lab-corrected')
    ax.set_xticks(x, [l.replace('LD', '') for l in lds[:n]])
    zeroed = np.allclose(eta_lab.eta2.to_numpy(float), 0, atol=1e-9)
    ax.set(xlabel='axis (LD)', ylabel='η² of lab, across mice',
           title='How much of each axis a lab label explains'
                 + ('\n(corrected bars are exactly 0 BY CONSTRUCTION)' if zeroed else ''))
    ax.legend(frameon=False, fontsize=plt.rcParams['font.size'] * 0.8)
    plt.tight_layout()
    if save:
        ps.savefig(fig, 'lab_correction_subspace_and_lab_variance', subdir=FIG_SUBDIR)
    plt.show()


def fig_movement(Mr, Ml, pair, save):
    """THE MOVEMENT FIGURE. Two matched axes, one point per mouse, before and after.

    Both runs are z-scored and the corrected axes are sign-flipped onto their raw
    partners first, so an arrow is a change of POSITION, not a change of units or of an
    arbitrary sign.

    Four panels: where the mice sit before, where they sit after, every mouse's
    displacement, and the lab centroids on their own. The last one exists because the
    mouse arrows are long and mostly report the ROTATION between the two axis systems,
    which would otherwise bury the part that is actually the lab correction -- the
    centroids collapsing onto the origin.
    """
    (ra, ca, sa), (rb, cb, sb) = pair
    x0, y0 = Mr[ra].to_numpy(float), Mr[rb].to_numpy(float)
    x1, y1 = sa * Ml[ca].to_numpy(float), sb * Ml[cb].to_numpy(float)
    labs = Mr['lab'].to_numpy()
    col = lab_colors(set(labs))
    cvec = [col[l] for l in labs]
    order = sorted(set(labs))

    fig, axes = plt.subplots(1, 4, figsize=(ps.SIZES[ps.MODE]['single'][0] * 3.8,
                                            ps.SIZES[ps.MODE]['single'][1] * 1.15))
    lim = 1.05 * max(np.abs(np.r_[x0, x1, y0, y1]))
    for ax, (X, Y, t) in zip(axes[:2], [(x0, y0, f'raw: {ra} x {rb}'),
                                        (x1, y1, f'lab-corrected: {ca} x {cb}')]):
        ax.scatter(X, Y, c=cvec, s=(plt.rcParams['lines.markersize'] * 1.6) ** 2,
                   edgecolors='0.25', linewidths=0.4, zorder=3)
        for l in order:
            k = labs == l
            ax.scatter(X[k].mean(), Y[k].mean(), color=col[l], marker='X', s=110,
                       edgecolors='k', linewidths=0.8, zorder=4)
        ps.zero_line(ax)
        ax.axvline(0, color='0.85', lw=0.8, zorder=0)
        ax.set(title=t, xlim=(-lim, lim), ylim=(-lim, lim),
               xlabel='z-scored axis 1', ylabel='z-scored axis 2' if ax is axes[0] else '')

    ax = axes[2]
    for i in range(len(x0)):
        ax.annotate('', xy=(x1[i], y1[i]), xytext=(x0[i], y0[i]),
                    arrowprops=dict(arrowstyle='->', color=cvec[i], lw=0.6,
                                    alpha=0.35, shrinkA=0, shrinkB=0))
    ps.zero_line(ax)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set(title='every mouse moves\n(mostly the rotation between the two axis systems)',
           xlim=(-lim, lim), ylim=(-lim, lim), xlabel='z-scored axis 1')

    # lab centroids on their own, on their own scale
    ax = axes[3]
    cx0 = np.array([x0[labs == l].mean() for l in order])
    cy0 = np.array([y0[labs == l].mean() for l in order])
    cx1 = np.array([x1[labs == l].mean() for l in order])
    cy1 = np.array([y1[labs == l].mean() for l in order])
    for l, a0, b0, a1, b1 in zip(order, cx0, cy0, cx1, cy1):
        ax.annotate('', xy=(a1, b1), xytext=(a0, b0),
                    arrowprops=dict(arrowstyle='-|>', color=col[l], lw=2.0,
                                    shrinkA=0, shrinkB=0))
        ax.scatter(a0, b0, facecolors='none', edgecolors=col[l], s=90, linewidths=1.5,
                   zorder=4)
        ax.scatter(a1, b1, color=col[l], s=45, edgecolors='k', linewidths=0.5, zorder=5)
    clim = 1.15 * max(np.abs(np.r_[cx0, cy0, cx1, cy1]).max(), 1e-3)
    ps.zero_line(ax)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set(title='lab centroids only\n(ring = before, dot = after)',
           xlim=(-clim, clim), ylim=(-clim, clim), xlabel='z-scored axis 1')

    handles = [plt.Line2D([], [], marker='o', ls='', color=col[l], label=l) for l in order]
    fig.legend(handles=handles, frameon=False, ncol=min(len(order), 5),
               loc='lower center', bbox_to_anchor=(0.5, -0.10),
               fontsize=plt.rcParams['font.size'] * 0.75)
    plt.tight_layout()
    if save:
        ps.savefig(fig, f'lab_correction_movement_{ra}_{rb}', subdir=FIG_SUBDIR)
    plt.show()

    d = np.hypot(x1 - x0, y1 - y0)
    bet0 = float(np.mean(cx0 ** 2 + cy0 ** 2))
    bet1 = float(np.mean(cx1 ** 2 + cy1 ** 2))
    print(f'  mouse displacement in this plane: median {np.median(d):.2f} SD, '
          f'max {d.max():.2f} SD')
    print(f'  mean squared distance of a lab centroid from the origin: '
          f'{bet0:.3f} raw -> {bet1:.3f} corrected')
    print(f'  median displacement of a lab centroid: '
          f'{np.median(np.hypot(cx1 - cx0, cy1 - cy0)):.2f} SD  '
          f'-- compare with the {np.median(d):.2f} SD a typical MOUSE moves: if the mouse\n'
          f'     number is much the larger, most of the movement is the rotation between\n'
          f'     the two axis systems and not the lab correction.')


def fig_pairs(Mr, Ml, pairs, n, save):
    """Each matched pair on its own axes, one point per mouse, coloured by lab."""
    n = min(n, len(pairs))
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(ps.SIZES[ps.MODE]['single'][0] * 0.95 * ncol,
                                      ps.SIZES[ps.MODE]['single'][1] * 0.95 * nrow))
    col = lab_colors(set(Mr['lab']))
    cvec = [col[l] for l in Mr['lab']]
    for ax, row in zip(axes.ravel(), pairs.head(n).itertuples()):
        x = zscore(Mr[[row.raw]].to_numpy(float))[:, 0]
        y = row.sign * zscore(Ml[[row.corrected]].to_numpy(float))[:, 0]
        ax.scatter(x, y, c=cvec, s=(plt.rcParams['lines.markersize'] * 1.4) ** 2,
                   edgecolors='0.25', linewidths=0.4, zorder=3)
        ps.regline(ax, x, y, row.p_perm < ALPHA)
        ps.annotate_corr(ax, abs(row.abs_r), row.p_perm, n=len(x), loc='upper left')
        ax.set(xlabel=f'raw {row.raw}',
               ylabel=f'corrected {row.corrected}' + (' (flipped)' if row.sign < 0 else ''))
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    plt.tight_layout()
    if save:
        ps.savefig(fig, 'lab_correction_matched_pairs', subdir=FIG_SUBDIR)
    plt.show()


# ------------------------------------------------- asking about a SPECIFIC pair
def corr_frame(C, lds):
    """The correlation matrix as a labelled frame, so any one pair is a lookup:

        Cdf.loc['LD3', 'LD2']        raw LD3 against corrected LD2
        Cdf.loc['LD3'].abs().nlargest(5)     what raw LD3 most resembles

    Rows are the RAW axes, columns the CORRECTED ones. That asymmetry matters and is
    easy to lose: Cdf.loc['LD3', 'LD2'] and Cdf.loc['LD2', 'LD3'] are different numbers.
    """
    return pd.DataFrame(C, index=pd.Index(lds, name='raw'),
                        columns=pd.Index(lds, name='corrected'))


def pair_report(C, lds, requests, n_mice):
    """r for named (raw, corrected) pairs, with the context that makes it readable.

    `requests` is a list of (raw, corrected) as either integers (1-based, the way the
    axes are named) or strings ('LD3'). For each pair you get back:

        r, abs_r, sign     the correlation itself
        p                  two-sided, from r and n. HONEST ONLY IF YOU NAMED THE PAIR IN
                           ADVANCE. Pick a pair by looking at the matrix first and this
                           p is the p of a maximum dressed up as the p of one test; the
                           rank columns are there so that is visible rather than hidden.
        rank_in_row        where this corrected axis ranks among all partners for that
                           raw axis (1 = its favourite). Its own answer to 'is this pair
                           special, or just one of many?'
        best_in_row        which corrected axis the raw one actually resembles most,
                           and how strongly
        rank_in_col        the same from the corrected axis's point of view -- a pair can
                           be the raw axis's favourite without being mutual
    """
    idx = {ld: i for i, ld in enumerate(lds)}
    norm = lambda v: v if isinstance(v, str) else f'LD{int(v)}'
    rows = []
    for a, b in requests:
        ra, cb = norm(a), norm(b)
        if ra not in idx or cb not in idx:
            raise KeyError(f'{(ra, cb)} is not a pair of axes in {lds[0]}..{lds[-1]}')
        i, j = idx[ra], idx[cb]
        r = float(C[i, j])
        t = abs(r) * np.sqrt(max(n_mice - 2, 1) / max(1 - r ** 2, 1e-12))
        row_abs, col_abs = np.abs(C[i]), np.abs(C[:, j])
        rows.append(dict(
            raw=ra, corrected=cb, r=r, abs_r=abs(r), sign=int(np.sign(r)),
            p=float(2 * stats.t.sf(t, max(n_mice - 2, 1))),
            rank_in_row=int((row_abs > abs(r)).sum() + 1),
            best_in_row=lds[int(np.argmax(row_abs))], r_best_in_row=float(row_abs.max()),
            rank_in_col=int((col_abs > abs(r)).sum() + 1),
            best_in_col=lds[int(np.argmax(col_abs))]))
    return pd.DataFrame(rows)


def fig_pair_profile(C, lds, raw_axes, assigned=None, mark=None, save=False):
    """For each named RAW axis, its correlation with every corrected axis.

    The question the matching cannot answer: not 'which one partner is best' but 'what
    does this axis look like over there at all'. One tall bar means the direction moved
    across intact; a spread of middling bars means it was smeared over several corrected
    axes, which is what a rotation looks like from one axis's point of view.

    `assigned` is the pairs table, to ring the partner the assignment chose; `mark` is a
    dict {raw: corrected} -- or {raw: [corrected, ...]}, since asking about two different
    partners for the same raw axis is exactly what this panel is for -- outlining the
    pairs you asked about by hand.
    """
    name = lambda a: a if isinstance(a, str) else f'LD{int(a)}'
    # dedupe, keeping order: the same raw axis asked about twice is one panel, not two
    raw_axes = list(dict.fromkeys(name(a) for a in raw_axes))
    idx = {ld: i for i, ld in enumerate(lds)}
    n = len(raw_axes)
    fig, axes = plt.subplots(n, 1, sharex=True, squeeze=False,
                             figsize=(ps.SIZES[ps.MODE]['single'][0] * 1.9,
                                      ps.SIZES[ps.MODE]['single'][1] * 0.62 * n))
    x = np.arange(len(lds))
    for ax, ra in zip(axes[:, 0], raw_axes):
        r = C[idx[ra]]
        ax.bar(x, r, color='0.6', width=0.72)
        if assigned is not None and (assigned.raw == ra).any():
            j = idx[assigned.loc[assigned.raw == ra, 'corrected'].iloc[0]]
            ax.bar(x[j], r[j], color=ps.NEUTRAL, width=0.72, edgecolor='k', lw=1.0,
                   label='assigned partner')
        if mark and ra in mark:
            want = mark[ra]
            want = want if isinstance(want, (list, tuple, set)) else [want]
            for k, w in enumerate(want):
                ax.bar(x[idx[name(w)]], r[idx[name(w)]], color='none', width=0.72,
                       edgecolor='#C51B7D', lw=1.6,
                       label='asked about' if k == 0 else None)
        ps.zero_line(ax)
        ax.set_ylabel(f'raw {ra}')
        ax.set_ylim(-1, 1)
        if ax is axes[0, 0]:
            ax.legend(frameon=False, ncol=2, loc='upper right',
                      fontsize=plt.rcParams['font.size'] * 0.7)
    axes[-1, 0].set_xticks(x[::2], [l.replace('LD', '') for l in lds[::2]])
    axes[-1, 0].set_xlabel('corrected axis (LD)')
    fig.suptitle('r of each raw axis with every corrected axis', y=1.0)
    plt.tight_layout()
    if save:
        ps.savefig(fig, 'lab_correction_pair_profiles', subdir=FIG_SUBDIR)
    plt.show()


def fig_specific_pairs(Mr, Ml, report, n_axes=None, save=False):
    """Scatter for each pair in a `pair_report` table, one point per mouse.

    NOT sign-flipped, unlike the matched-pair panels: you named these axes, so the sign
    of the relation is part of the answer rather than a nuisance to be normalised away.
    """
    n = len(report)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(ps.SIZES[ps.MODE]['single'][0] * 0.95 * ncol,
                                      ps.SIZES[ps.MODE]['single'][1] * 0.95 * nrow))
    col = lab_colors(set(Mr['lab']))
    cvec = [col[l] for l in Mr['lab']]
    for ax, row in zip(axes.ravel(), report.itertuples()):
        x = zscore(Mr[[row.raw]].to_numpy(float))[:, 0]
        y = zscore(Ml[[row.corrected]].to_numpy(float))[:, 0]
        ax.scatter(x, y, c=cvec, s=(plt.rcParams['lines.markersize'] * 1.4) ** 2,
                   edgecolors='0.25', linewidths=0.4, zorder=3)
        ps.regline(ax, x, y, row.p < 0.05)
        ps.annotate_corr(ax, row.r, row.p, n=len(x), loc='upper left')
        # 'of n_axes' is spelled out rather than inferred from the frame's width: Mr
        # carries metadata columns too, and counting them is the kind of coincidence that
        # quietly turns into a wrong label the day another column is added
        of = f' of {n_axes}' if n_axes else ''
        ax.set(xlabel=f'raw {row.raw}', ylabel=f'corrected {row.corrected}',
               title=f'rank {row.rank_in_row}{of} for {row.raw}')
        ax.title.set_fontsize(plt.rcParams['font.size'] * 0.85)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    plt.tight_layout()
    if save:
        ps.savefig(fig, 'lab_correction_specific_pairs', subdir=FIG_SUBDIR)
    plt.show()
