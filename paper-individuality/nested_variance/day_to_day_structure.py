"""
THE STRUCTURE OF DAY-TO-DAY VARIABILITY, ON ITS OWN TERMS
=========================================================
The within-mouse level of a two-level model of the session syllable profiles (360 features, the LDA's design
matrix): each session's deviation from its own mouse's mean. Four questions about that within-mouse covariance:

A  NOISE-FREE COVARIANCE. The session profile carries trial sampling noise. Odd and even trials of a session share
   the day but not the sampling noise, so the cross-covariance of their within-mouse deviations,
       C = (Do'De + De'Do) / 2 / (sessions - mice),
   estimates the day-to-day covariance with the noise removed. Its eigenvectors are the noise-free wPCs.

B  HOW MANY DIMENSIONS. Parallel analysis on C: the same cross-covariance with the even halves shuffled across
   sessions (no shared day), N_PERM times. Dimensions = the leading run of eigenvalues above the null's 95th
   percentile at their rank. Plus each wPC's split-half reliability when fitted WITHOUT the mouse it is scored on
   (leave-one-mouse-out, sign-aligned to the full fit), Spearman-Brown corrected.

C  SHARED ACROSS MICE? Leave one mouse out, fit the top-k wPCs on the other mice, and ask what fraction of the
   held-out mouse's own day-to-day variance (trace of its deviation scatter) they capture. Against:
       own       the mouse's own top-k axes (in sample: a ceiling; with 3 sessions it has only 2 dimensions)
       PCA       top-k PCs of all other sessions (between + within mice, no labels)
       mouse     top-k axes of the between-mouse covariance of the other mice (S_M, random-effects estimator)
       random    k random directions drawn from the other sessions' total covariance (data-shaped), N_RAND draws
   pooled wPCs >> random = a shared repertoire of day-to-day variation; ~ random = each mouse varies its own way.

D  SCALING. Repeat A and C after
       corr       each feature divided by its SD across sessions
       logratio   per bin, paw occupancies as centred log-ratios (pseudocount EPS), whisking and licking as logits
   and compare the within-mouse session scores on wPC1-3 with the covariance version.

Interpretation: each wPC's session scores correlated with the within-mouse deviations of 40 summaries (4 epochs x
10 channels: 8 paw states in the 'uniform' numbering, whisking, licking).
Writes figures/day_to_day_structure.{csv,png,pdf}.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import linalg, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import within_between_models as wb      # noqa: E402  (paths, numpy shim)
import nested_variance as nv            # noqa: E402
import paper_style as ps                # noqa: E402
from ld1_within_mouse import covariances  # noqa: E402

N_PERM, N_RAND, SEED = 500, 200, 0
K_MAX = 10
KS = [1, 2, 3, 5]
EPS = 0.005
N_BINS, N_CH = 40, nv.N_SYLL - 1          # 9 columns per bin: paw 0, 2..7, whisk, lick (raw paw 1 dropped)
FIGDIR = HERE / 'figures'
CACHE = HERE / 'cache' / 'day_to_day_structure.pkl'
REF = {'own': '#BBBBBB', 'pooled wPCs': '#3E8E54', 'PCA': '#22344A', 'mouse axes': '#E0614A', 'random': '#999999'}


# ---------------------------------------------------------------------------------------------------------------
# data and scalings
# ---------------------------------------------------------------------------------------------------------------
def channels(X):
    """[sessions x 360] -> [sessions x 40 bins x 10] with raw paw 1 restored, paws in the 'uniform' numbering."""
    Y = X.reshape(len(X), N_BINS, N_CH)
    paw = np.concatenate([Y[..., :1], 1 - Y[..., :7].sum(-1, keepdims=True), Y[..., 1:7]], axis=-1)
    m = ps.PAW_STATE_SETS['uniform']['syllable_paw_map']
    out = np.empty_like(paw)
    out[..., m] = paw                                  # syllable digit d is state m[d]
    return np.concatenate([out, Y[..., 7:]], axis=-1)


def logratio(X):
    Y = channels(X)
    lp = np.log(Y[..., :8] + EPS)
    clr = lp - lp.mean(-1, keepdims=True)
    lo = np.log((Y[..., 8:] + EPS) / (1 - Y[..., 8:] + EPS))
    return np.concatenate([clr[..., 1:], lo], axis=-1).reshape(len(X), -1)   # CLR has rank 7: drop one


def scalings(full, halves):
    sd = full.std(0)
    sd[sd == 0] = 1
    return {'covariance': (full, *halves),
            'corr': (full / sd, halves[0] / sd, halves[1] / sd),
            'logratio': (logratio(full), logratio(halves[0]), logratio(halves[1]))}


def deviations(X, g):
    m = g.max() + 1
    M = np.stack([X[g == k].mean(0) for k in range(m)])
    return X - M[g]


def cross_cov(Do, De, df):
    A = Do.T @ De
    return (A + A.T) / 2 / df


def top(S, k):
    ev, V = np.linalg.eigh(S)
    return ev[::-1][:k], V[:, ::-1][:, :k]


# ---------------------------------------------------------------------------------------------------------------
# A, B
# ---------------------------------------------------------------------------------------------------------------
def spectrum(X, Xo, Xe, g, rng):
    df = len(X) - (g.max() + 1)
    D, Do, De = deviations(X, g), deviations(Xo, g), deviations(Xe, g)
    S_raw = D.T @ D / df
    C = cross_cov(Do, De, df)
    ev_C = np.linalg.eigvalsh(C)[::-1]
    null = np.stack([np.linalg.eigvalsh(cross_cov(Do, De[rng.permutation(len(De))], df))[::-1][:K_MAX]
                     for _ in range(N_PERM)])
    q95 = np.quantile(null, 0.95, axis=0)
    above = ev_C[:K_MAX] > q95
    n_dims = K_MAX if above.all() else int(np.argmin(above))
    return dict(S_raw=S_raw, C=C, ev_raw=np.linalg.eigvalsh(S_raw)[::-1], ev_C=ev_C, null=null, n_dims=n_dims,
                noise_share=1 - np.trace(C) / np.trace(S_raw), D=D, Do=Do, De=De, df=df)


def lomo_reliability(sp, g):
    """split-half reliability of wPC1..K_MAX scores, each mouse scored on axes fitted without it."""
    _, V_all = top(sp['C'], K_MAX)
    so, se = np.zeros((len(g), K_MAX)), np.zeros((len(g), K_MAX))
    for j in range(g.max() + 1):
        ix = g == j
        Cj = cross_cov(sp['Do'][ix], sp['De'][ix], 1)
        C_oth = (sp['C'] * sp['df'] - Cj) / (sp['df'] - (ix.sum() - 1))
        _, V = top(C_oth, K_MAX)
        V = V * np.sign((V * V_all).sum(0))
        so[ix], se[ix] = sp['Do'][ix] @ V, sp['De'][ix] @ V
    r = np.array([np.corrcoef(so[:, k], se[:, k])[0, 1] for k in range(K_MAX)])
    return 2 * r / (1 + r)


# ---------------------------------------------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------------------------------------------
def sharedness(X, sp, g, rng):
    rows = []
    kmax = max(KS)
    for j in range(g.max() + 1):
        ix = g == j
        Dj = sp['D'][ix]
        Wj = Dj.T @ Dj
        tot = np.trace(Wj)
        Cj = cross_cov(sp['Do'][ix], sp['De'][ix], 1)
        C_oth = (sp['C'] * sp['df'] - Cj) / (sp['df'] - (ix.sum() - 1))
        Xo, go = X[~ix], pd.factorize(g[~ix])[0]
        S_M = covariances(Xo, go)[1]
        T = np.cov(Xo.T)
        axes = {'pooled wPCs': top(C_oth, kmax)[1], 'PCA': top(T, kmax)[1], 'mouse axes': top(S_M, kmax)[1],
                'own': top(Wj, kmax)[1]}
        evT, VT = np.linalg.eigh(T)
        L = VT * np.sqrt(np.clip(evT, 0, None))
        for k in KS:
            rec = dict(mouse=j, n_sessions=ix.sum(), k=k, day_to_day_var=tot / (ix.sum() - 1))
            for name, V in axes.items():
                rec[name] = np.trace(V[:, :k].T @ Wj @ V[:, :k]) / tot
            f = []
            for _ in range(N_RAND):
                Q = np.linalg.qr(L @ rng.standard_normal((L.shape[1], k)))[0]
                f.append(np.trace(Q.T @ Wj @ Q) / tot)
            rec['random'] = np.mean(f)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# interpretation
# ---------------------------------------------------------------------------------------------------------------
def summaries(full, g):
    Y = channels(full).reshape(len(full), 4, 10, 10).mean(2)        # epochs x channels, averaged over the 10 bins
    return deviations(Y.reshape(len(full), -1), g)


def summary_names():
    ps.use_paw_states('uniform')
    ch = [ps.paw_label(i) for i in range(8)] + ['Whisk', 'Lick']
    return [f'{e} {c}' for e in ps.EPOCH_SHORT for c in ch]


def scores(sp, k=3):
    _, V = top(sp['C'], k)
    return sp['D'] @ V


# ---------------------------------------------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(SEED)
    full, s_ids, mouse, halves = wb.session_profiles()
    g = pd.factorize(mouse)[0]
    out = {}
    for name, (X, Xo, Xe) in scalings(full, halves).items():
        print(f'--- {name}')
        sp = spectrum(X, Xo, Xe, g, rng)
        sp['reliability'] = lomo_reliability(sp, g)
        sh = sharedness(X, sp, g, rng)
        out[name] = dict(ev_raw=sp['ev_raw'], ev_C=sp['ev_C'], null=sp['null'], n_dims=sp['n_dims'],
                         noise_share=sp['noise_share'], reliability=sp['reliability'], shared=sh,
                         scores=scores(sp))
    out['summaries'], out['mouse'], out['sessions'] = summaries(full, g), mouse, s_ids
    pd.to_pickle(out, CACHE)
    return out


def report(out):
    rows = []
    base = out['covariance']['scores']
    for name in ('covariance', 'corr', 'logratio'):
        o = out[name]
        tr = o['ev_C'].clip(0).sum()
        print(f'\n=== {name}: noise share of day-to-day variance {o["noise_share"]:.3f}; '
              f'dimensions above parallel-analysis null: {o["n_dims"]}')
        print('   wPC share (noise-free):', ' '.join(f'{v / tr:.3f}' for v in o['ev_C'][:K_MAX]))
        print('   null 95% share        :', ' '.join(f'{v / tr:.3f}' for v in np.quantile(o['null'], .95, 0)))
        print('   LOMO split-half rel.  :', ' '.join(f'{v:.2f}' for v in o['reliability']))
        sh = o['shared']
        med = sh.groupby('k')[list(REF)].median()
        print('   median fraction of a held-out mouse\'s day-to-day variance captured:')
        print(med.round(3).to_string().replace('\n', '\n      '))
        for k in KS:
            s = sh[sh.k == k]
            p = stats.wilcoxon(s['pooled wPCs'], s['random']).pvalue
            pm = stats.wilcoxon(s['pooled wPCs'], s['mouse axes']).pvalue
            rho = stats.spearmanr(s['pooled wPCs'], s['day_to_day_var'])
            rows.append(dict(scaling=name, k=k, **{f'median_{c}': med.loc[k, c] for c in REF},
                             frac_mice_pooled_gt_random=(s['pooled wPCs'] > s['random']).mean(),
                             p_pooled_vs_random=p, p_pooled_vs_mouse_axes=pm, rho_pooled_vs_dtd_var=rho[0],
                             p_rho=rho[1], n_dims=o['n_dims'], noise_share=o['noise_share']))
            print(f'   k={k}: pooled > random in {(s["pooled wPCs"] > s["random"]).mean():.0%} of mice, '
                  f'p = {p:.2g}; pooled vs mouse axes p = {pm:.2g}; '
                  f'captured vs mouse\'s day-to-day variance rho = {rho[0]:+.2f} (p = {rho[1]:.2g})')
        if name != 'covariance':
            r = np.corrcoef(base.T, o['scores'].T)[:3, 3:]
            print('   |r| of wPC scores with the covariance version (rows: cov wPC1-3, cols: this wPC1-3):')
            print('  ', np.abs(r).round(2).tolist())
            cc = np.sqrt(np.clip(linalg.svdvals(np.linalg.qr(base)[0].T @ np.linalg.qr(o['scores'])[0]), 0, 1) ** 2)
            print('   canonical correlations of the wPC1-3 score subspaces:', cc.round(2).tolist())
    S, names = out['summaries'], summary_names()
    for name in ('covariance', 'logratio'):
        sc = out[name]['scores']
        print(f'\n=== {name}: top within-mouse correlates of wPC1-3 (sign oriented so the top |r| is positive)')
        for k in range(3):
            r = np.array([np.corrcoef(sc[:, k], S[:, i])[0, 1] if S[:, i].std() > 0 else 0 for i in range(S.shape[1])])
            r *= np.sign(r[np.argmax(np.abs(r))])
            o = np.argsort(r)
            print(f'   wPC{k + 1}  + ' + ', '.join(f'{names[i]} {r[i]:+.2f}' for i in o[::-1][:5]))
            print(f'          - ' + ', '.join(f'{names[i]} {r[i]:+.2f}' for i in o[:5]))
    return pd.DataFrame(rows)


def figure(out):
    ps.use('paper')
    fs = plt.rcParams['font.size']
    fig, axs = plt.subplots(1, 4, figsize=(7.09, 2.2), gridspec_kw={'wspace': 0.6})
    o = out['covariance']
    tr = o['ev_C'].clip(0).sum()
    x = np.arange(1, K_MAX + 1)

    ax = axs[0]
    ax.plot(x, o['ev_C'][:K_MAX] / tr, 'o-', color=REF['pooled wPCs'], ms=2.5, lw=0.8, label='day-to-day (noise-free)')
    lo, hi = np.quantile(o['null'], [0.05, 0.95], 0) / tr
    ax.fill_between(x, lo, hi, color='0.85', lw=0, label='trial-sampling null')
    ax.set_yscale('log')
    ax.set_xlabel('Within-mouse PC')
    ax.set_ylabel('Share of day-to-day variance')
    ax.set_title(f"{o['n_dims']} dimensions above noise", fontsize=fs * 0.75)
    ax.legend(frameon=False, fontsize=fs * 0.6, loc='lower left')

    ax = axs[1]
    ax.plot(x, o['reliability'], 'o-', color=REF['pooled wPCs'], ms=2.5, lw=0.8)
    ax.axhline(0, color='0.8', lw=0.5)
    ax.set_xlabel('Within-mouse PC')
    ax.set_ylabel('Split-half reliability\n(axis fitted without the mouse)')
    ax.set_ylim(min(0, o['reliability'].min() - 0.05), 1.02)

    ax = axs[2]
    med = o['shared'].groupby('k')[list(REF)].median()
    for c in REF:
        ax.plot(med.index, med[c], 'o-' if c != 'own' else 'o--', color=REF[c], ms=2.5, lw=0.8, label=c)
    ax.set_xlabel('Number of axes (k)')
    ax.set_ylabel("Held-out mouse's day-to-day\nvariance captured (median)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=fs * 0.55, loc='lower right')

    ax = axs[3]
    sh = o['shared'][o['shared'].k == 3]
    for i, c in enumerate(['pooled wPCs', 'random']):
        ax.scatter(np.full(len(sh), i) + rng_jit(len(sh)), sh[c], s=4, color=REF[c], lw=0, alpha=0.7)
    ax.plot([0, 1], sh[['pooled wPCs', 'random']].T.values, color='0.85', lw=0.3, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['pooled\nwPC1-3', 'random\n3 axes'])
    ax.set_ylabel('Captured, per mouse (k = 3)')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    for ax, letter in zip(axs, 'abcd'):
        ax.text(-0.45, 1.1, letter, transform=ax.transAxes, fontsize=fs + 2, fontweight='bold')
    for path in (FIGDIR / 'day_to_day_structure.png', FIGDIR / 'day_to_day_structure.pdf'):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', path.relative_to(wb.PI))


def rng_jit(n, s=0.06):
    return np.random.default_rng(1).uniform(-s, s, n)


if __name__ == '__main__':
    out = pd.read_pickle(CACHE) if CACHE.exists() and 'rerun' not in sys.argv else run()
    report(out).to_csv(FIGDIR / 'day_to_day_structure.csv', index=False)
    figure(out)
