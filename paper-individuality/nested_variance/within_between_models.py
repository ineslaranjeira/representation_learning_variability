"""
TRAIT OR STATE? WITHIN-BETWEEN SLOPES ON LABEL-FREE AXES
========================================================
For every LD1 correlate the paper shows (fig_lab_controls' rows), and for each axis -- the first five principal
components of the session syllable profiles (no labels, all variance) and LD1 for reference -- split each
session's axis score into its mouse's mean and the session's deviation from it, and estimate

    beta_B   between-mouse slope: mouse-mean y on mouse-mean axis score, every mouse counted once
    beta_W   within-mouse slope:  session deviation of y on session deviation of the axis score (fixed effects)
    beta_C = beta_B - beta_W      the contextual effect. Non-zero = the relationship is not the same between mice
                                  as within a mouse from day to day; beta_B != 0 with beta_W ~ 0 = a trait

Both slopes are in SD(y) per SD(axis): y is z-scored across sessions, each axis is scaled to unit SD across all
260 sessions, so slopes compare across correlates and axes.

NOISE IN THE AXIS SCORE attenuates a slope towards 0 by the predictor's reliability, and it does so unequally:
a session's deviation from its mouse is mostly sampling noise, a mouse mean much less. Uncorrected, beta_W would
shrink more than beta_B and fake a trait. So both are disattenuated:
    lambda_W   split-half reliability of the session deviations: every session's score is recomputed from its odd
               and from its even trials, both deviations taken from their own half's mouse means, r between the
               halves, Spearman-Brown to full length: 2r / (1 + r)
    lambda_B   reliability of the mouse means: 1 - (pooled within-mouse variance x mean(1/n_sessions)) / var(means)
    beta = beta_observed / lambda.  The deviation is orthogonal to the mouse means, so each correction is separate.

INFERENCE. Mice resampled with replacement (N_BOOT), every quantity -- slopes and both reliabilities -- recomputed
in each resample from per-mouse sums. CIs are percentile intervals; the p of beta_C is the two-sided bootstrap p.
BH separately per axis, over the correlates.

THE AXES. PCA on the covariance of the 360-feature LDA design matrix (fig_ld1_vs_pc1's PCA; PC1 there is the
whisking/licking axis, r = 0.78 with LD1). Signs are arbitrary and set so each PC correlates positively with LD1;
the sign flips beta_B and beta_W together, so beta_C's test is unaffected. LD1 uses the saved embedding for full
sessions; its half-session scores come from a refit of the same mouse-weighted shrinkage LDA, rescaled onto the
saved values.

CORRELATES. lab_controls.py's own session tables (trial, engage, entropy, neural_mouse), captured once by hooking
its correlate() and cached in cache/correlate_sessions.pqt (delete to recapture; it reruns those sections, ~15 min).
Training time is one value per mouse and has no within-mouse slope, so it is left out.

Writes figures/within_between.{csv,png,pdf}.
"""
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
for p in (PI, HERE, PI / 'figure_generation'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    import numpy._core.numeric  # noqa: F401
except ModuleNotFoundError:     # the LDA file was pickled under numpy 2
    import numpy.core.numeric, numpy.core.multiarray
    sys.modules['numpy._core.numeric'], sys.modules['numpy._core.multiarray'] = np.core.numeric, np.core.multiarray

import nested_variance as nv
import paper_style as ps

N_PCS, N_BOOT, SEED = 5, 2000, 0
SHRINKAGE = 0.5                      # the embedding's
CACHE = HERE / 'cache' / 'correlate_sessions.pqt'
FIGDIR = HERE / 'figures'
LDA_FILE = PI / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
ROWS = [('Task', 'Accuracy (easy trials)', 'Accuracy'),
        ('Task', 'Choice bias', 'Choice bias'),
        ('Task', 'Block bias shift', 'Block bias shift'),
        ('Task', 'Reaction time (log)', 'Reaction time'),
        ('Task', 'Trial elongation (log)', 'Trial elongation'),
        ('Internal states', 'Fraction engaged (hard)', 'Fraction engaged'),
        ('Internal states', 'Engaged log posterior ratio', 'Engaged posterior'),
        ('Internal states', 'State switches (K=2)', 'State switches'),
        ('Internal states', 'State switches | easy accuracy', 'Switches | accuracy'),
        ('Internal states', 'P(stay engaged)', 'P(stay engaged)'),
        ('Internal states', 'P(stay disengaged)', 'P(stay disengaged)'),
        ('Internal states', 'Held-out fit gain K=2->3', 'Fit gain K2→3'),
        ('Internal states', 'Held-out fit gain K=2->4', 'Fit gain K2→4'),
        ('Internal states', 'State switches (K=4)', 'Switches K=4'),
        ('Entropy', 'Trial-mode entropy', 'Trial-mode entropy'),
        ('Entropy', 'Syllable entropy', 'Syllable entropy')] + \
       [('Neural', f'{m}, {t}', f'{s} {lab}') for m, s in [('Firing rate', 'FR'), ('Fano factor', 'FF'),
                                                          ('Noise correlation', 'r_sc')]
        for t, lab in [('Baseline', 'pre'), ('Post-stimulus', 'post'), ('Response magnitude', 'change')]]


# ---------------------------------------------------------------------------------------------------------------
# correlate session tables (lab_controls' own), cached
# ---------------------------------------------------------------------------------------------------------------
def correlate_sessions():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    # by path: nested_variance puts 4_mice/ on sys.path, and 4_mice/ has its own, different lab_controls.py
    import importlib.util
    spec = importlib.util.spec_from_file_location('lab_controls_paper', PI / 'lab_controls.py')
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    lc.METHOD = 'pearson'
    keep, orig = [], lc.correlate

    def hook(sess, y, label, section, rng_, x='LD1', per_session=True, note=''):
        if per_session and 'session' in sess:
            keep.append(sess[['session', 'mouse_name', y]].rename(columns={y: 'y'}).assign(correlate=label))
        return orig(sess, y, label, section, rng_, x=x, per_session=per_session, note=note)

    lc.correlate = hook
    rng = np.random.default_rng(SEED)
    for name in ('trial', 'engage', 'entropy', 'neural_mouse'):
        print(f'--- lab_controls section: {name}')
        lc.SECTIONS[name](rng)
    lc.correlate = orig
    out = pd.concat(keep, ignore_index=True)
    CACHE.parent.mkdir(exist_ok=True)
    out.to_parquet(CACHE)
    return out


# ---------------------------------------------------------------------------------------------------------------
# axis scores: full sessions, odd-trial halves, even-trial halves
# ---------------------------------------------------------------------------------------------------------------
def session_profiles():
    """-> (full [sessions x 360], s_ids, mouse per session, (odd-trial, even-trial) profiles)."""
    codes, meta = nv.load_trials()
    X1 = np.delete(nv.one_hot(codes), 1, axis=2).reshape(len(codes), -1)   # lda_shrinkage.binarize layout (360)
    s_codes, s_ids = pd.factorize(meta['session'], sort=True)
    odd = (meta.groupby('session').cumcount() % 2 == 1).to_numpy()

    def session_means(mask):
        A, _ = nv._indicator(s_codes[mask], mask.sum())
        cnt = np.bincount(s_codes[mask], minlength=len(s_ids))
        return np.asarray(A @ X1[mask]) / cnt[:, None]

    full = session_means(np.ones(len(meta), bool))
    halves = (session_means(odd), session_means(~odd))
    mouse = meta.groupby(s_codes)['mouse_name'].first().to_numpy()
    return full, s_ids, mouse, halves


def axis_scores():
    full, s_ids, mouse, halves = session_profiles()

    lda = pd.read_pickle(LDA_FILE)
    ld1_saved = lda.set_index('session').loc[s_ids, 0].to_numpy(float)
    # refit of lda_shrinkage's mouse-weighted embedding (fig_ld1_vs_pc1), for the half-session scores
    y = pd.factorize(mouse)[0]
    classes = np.unique(y)
    M = np.stack([full[y == c].mean(0) for c in classes])
    Sb = np.cov((M - M.mean(0)).T, bias=True)
    R = full - M[y]
    Sw = R.T @ R / (len(full) - len(classes))
    Sw = (1 - SHRINKAGE) * Sw + SHRINKAGE * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    ev, V = linalg.eigh(Sb, Sw)
    w = V[:, np.argmax(ev)]
    a, b = np.polyfit(full @ w, ld1_saved, 1)              # put the refit on the saved LD1's scale and sign
    r_check = np.corrcoef(full @ w, ld1_saved)[0, 1]
    print(f'LD1 refit vs saved: r = {r_check:+.4f} (trials with a missing bin dropped here, kept there)')
    assert abs(r_check) > 0.99

    pca = PCA(N_PCS).fit(full)
    scores = {}
    for k in range(N_PCS):
        sgn = np.sign(np.corrcoef(pca.transform(full)[:, k], ld1_saved)[0, 1]) or 1.0
        f = lambda Z, k=k, sgn=sgn: sgn * pca.transform(Z)[:, k]
        scores[f'PC{k + 1}'] = (f(full), f(halves[0]), f(halves[1]))
    scores['LD1'] = (ld1_saved, a * (halves[0] @ w) + b, a * (halves[1] @ w) + b)
    share = dict({f'PC{k + 1}': pca.explained_variance_ratio_[k] for k in range(N_PCS)})
    return scores, s_ids, mouse, share


# ---------------------------------------------------------------------------------------------------------------
# the estimator, vectorised over bootstrap resamples of mice
# ---------------------------------------------------------------------------------------------------------------
def per_mouse_sums(mouse, x, xo, xe, y):
    """Per-mouse sufficient statistics (mice with >= 2 sessions)."""
    d = pd.DataFrame({'m': mouse, 'x': x, 'xo': xo, 'xe': xe, 'y': y})
    g = d.groupby('m')
    d = d[g['x'].transform('size') >= 2]
    g = d.groupby('m')
    dx = d['x'] - g['x'].transform('mean')
    dy = d['y'] - g['y'].transform('mean')
    do = d['xo'] - g['xo'].transform('mean')
    de = d['xe'] - g['xe'].transform('mean')
    s = pd.DataFrame({'n': g.size(), 'xbar': g['x'].mean(), 'ybar': g['y'].mean(),
                      'Sxx': (dx * dx).groupby(d['m']).sum(), 'Sxy': (dx * dy).groupby(d['m']).sum(),
                      'Soe': (do * de).groupby(d['m']).sum(), 'Soo': (do * do).groupby(d['m']).sum(),
                      'See': (de * de).groupby(d['m']).sum()})
    return s, len(d)


def estimates(C, s):
    """C: (resamples x mice) draw counts. Returns dict of arrays over resamples."""
    n, xb, yb = s['n'].to_numpy(), s['xbar'].to_numpy(), s['ybar'].to_numpy()
    tot = C.sum(1)
    # within: pooled fixed-effects slope, split-half reliability of the deviations
    beta_w_obs = (C @ s['Sxy'].to_numpy()) / (C @ s['Sxx'].to_numpy())
    r_half = (C @ s['Soe'].to_numpy()) / np.sqrt((C @ s['Soo'].to_numpy()) * (C @ s['See'].to_numpy()))
    lam_w = np.clip(2 * r_half / (1 + r_half), 0.05, 1)
    # between: one point per drawn mouse
    mx, my = (C @ xb) / tot, (C @ yb) / tot
    vxx = (C @ xb ** 2) / tot - mx ** 2
    vxy = (C @ (xb * yb)) / tot - mx * my
    beta_b_obs = vxy / vxx
    within_var = (C @ s['Sxx'].to_numpy()) / (C @ (n - 1))
    noise_mean = within_var * (C @ (1 / n)) / tot
    lam_b = np.clip(1 - noise_mean / vxx, 0.05, 1)
    return dict(beta_b=beta_b_obs / lam_b, beta_w=beta_w_obs / lam_w, beta_b_obs=beta_b_obs,
                beta_w_obs=beta_w_obs, lam_b=lam_b, lam_w=lam_w)


def run():
    ps.use('paper')
    ps.use_paw_states('uniform')
    rng = np.random.default_rng(SEED)
    scores, s_ids, mouse_all, share = axis_scores()
    sess_index = pd.Index(s_ids)
    CS = correlate_sessions()

    # axis variance structure, on all sessions
    print('\naxis       var share  within-mouse share  lambda_W  lambda_B  (all sessions)')
    for ax, (x, xo, xe) in scores.items():
        sd = x.std()
        s, _ = per_mouse_sums(mouse_all, x / sd, xo / sd, xe / sd, np.zeros_like(x))
        e = estimates(np.ones((1, len(s))), s)
        wshare = s['Sxx'].sum() / ((x / sd - (x / sd).mean()) ** 2).sum()
        print(f"{ax:6s}  {share.get(ax, np.nan):9.3f}  {wshare:18.3f}  {e['lam_w'][0]:8.2f}  {e['lam_b'][0]:8.2f}")

    rows = []
    for group, label, short in ROWS:
        c = CS[CS['correlate'] == label]
        if c.empty:
            print(f'  not available: {label}')
            continue
        c = c[c['session'].isin(sess_index)].replace([np.inf, -np.inf], np.nan).dropna(subset=['y'])
        idx = sess_index.get_indexer(c['session'])
        yz = ((c['y'] - c['y'].mean()) / c['y'].std()).to_numpy()
        for ax, (x, xo, xe) in scores.items():
            sd = x.std()                                         # unit SD over all 260 sessions
            s, n_sess = per_mouse_sums(mouse_all[idx], x[idx] / sd, xo[idx] / sd, xe[idx] / sd, yz)
            point = estimates(np.ones((1, len(s))), s)
            C = rng.multinomial(len(s), np.ones(len(s)) / len(s), size=N_BOOT).astype(float)
            boot = estimates(C, s)
            bc = boot['beta_b'] - boot['beta_w']
            p = 2 * min((bc <= 0).mean(), (bc >= 0).mean())
            ci = lambda v: np.nanpercentile(v, [2.5, 97.5])
            rows.append(dict(group=group, correlate=label, label=short, axis=ax, n_sessions=n_sess, n_mice=len(s),
                             beta_b=point['beta_b'][0], beta_b_lo=ci(boot['beta_b'])[0], beta_b_hi=ci(boot['beta_b'])[1],
                             beta_w=point['beta_w'][0], beta_w_lo=ci(boot['beta_w'])[0], beta_w_hi=ci(boot['beta_w'])[1],
                             beta_c=point['beta_b'][0] - point['beta_w'][0], beta_c_lo=ci(bc)[0], beta_c_hi=ci(bc)[1],
                             p_c=max(p, 1 / N_BOOT), beta_b_obs=point['beta_b_obs'][0], beta_w_obs=point['beta_w_obs'][0],
                             lam_b=point['lam_b'][0], lam_w=point['lam_w'][0]))
    RES = pd.DataFrame(rows)
    RES['q_c'] = RES.groupby('axis')['p_c'].transform(lambda p: pd.Series(multipletests(p, method='fdr_bh')[1], index=p.index))
    FIGDIR.mkdir(exist_ok=True)
    RES.to_csv(FIGDIR / 'within_between.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 500):
        for ax in scores:
            t = RES[RES.axis == ax][['label', 'beta_b', 'beta_w', 'beta_c', 'beta_c_lo', 'beta_c_hi', 'q_c', 'lam_w']]
            print(f'\n=== {ax} ===\n' + t.round(2).to_string(index=False))
    figure(RES, list(scores))


def figure(RES, axes_names):
    fig, axs = plt.subplots(2, 3, figsize=(7.09, 5.2), gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
    colors = {'Task': '#3A78B0', 'Internal states': '#7A5C9E', 'Entropy': '#C9A13B', 'Neural': '#3E8E54'}
    lim = 1.6
    for ax, name in zip(axs.ravel(), axes_names):
        d = RES[RES.axis == name]
        ax.plot([-lim, lim], [-lim, lim], color='0.75', lw=0.6, zorder=0)
        ax.axhline(0, color='0.85', lw=0.5, zorder=0); ax.axvline(0, color='0.85', lw=0.5, zorder=0)
        for _, r in d.iterrows():
            col = colors[r['group']]
            sig = r['q_c'] < 0.05
            ax.plot([r['beta_w_lo'], r['beta_w_hi']], [r['beta_b']] * 2, color=col, lw=0.4, alpha=0.5)
            ax.plot([r['beta_w']] * 2, [r['beta_b_lo'], r['beta_b_hi']], color=col, lw=0.4, alpha=0.5)
            ax.plot(r['beta_w'], r['beta_b'], 'o', ms=3.5, color=col, mfc=col if sig else 'white', mew=0.8, zorder=3)
            if sig:
                ax.annotate(r['label'], (r['beta_w'], r['beta_b']), fontsize=4.5, xytext=(2, 2),
                            textcoords='offset points', color='0.25')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_title(name, fontsize=8)
        ax.set_xlabel('β within mouse', fontsize=7); ax.set_ylabel('β between mice', fontsize=7)
        ax.tick_params(labelsize=6)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=g) for g, c in colors.items()]
    handles += [plt.Line2D([], [], marker='o', ls='', color='0.4', mfc='white', label='β_B = β_W not rejected'),
                plt.Line2D([], [], marker='o', ls='', color='0.4', label='β_B ≠ β_W (q < 0.05)')]
    fig.legend(handles=handles, ncol=6, frameon=False, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.965, 'Slopes in SD(y) per SD(axis), corrected for noise in the axis score; diagonal = same '
                         'relationship between mice and within a mouse', ha='center', fontsize=6.5, color='0.35')
    for path in (FIGDIR / 'within_between.png', FIGDIR / 'within_between.pdf'):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', path.relative_to(PI))


if __name__ == '__main__':
    run()
