"""
LD1 (TRAIT, ACROSS MICE) vs WITHIN-MOUSE PC1 (STATE, ACROSS SESSIONS OF A MOUSE)
================================================================================
fig_lab_controls' compact figure, with its columns replaced: every LD1 correlate the paper shows, tested

    LD1, across mice           session values averaged within mouse, Pearson r with the mouse's LD1 -- the
                               `rho` column of lab_controls.py, recomputed here in the same call
    within-mouse PC1           Pearson r between each session's deviation from its mouse's mean and that
                               session's score on within-mouse PC1 (mice with >= 2 sessions); p from shuffling
                               sessions within mouse, CI from mice resampled with replacement

The session values are lab_controls.py's own: its sections are run with `correlate()` hooked, so each row sees
exactly the table the paper's numbers come from. Neural rows use its `neural_mouse` section (metric adjusted for
region and trial-count covariates, averaged per session), so the neural rows are computed like the behavioural ones.

WITHIN-MOUSE PC1 is the first principal component of the session syllable profiles (400 features) after each
mouse's mean profile is subtracted (see within_mouse_pca in this folder's notes). Its sign is set so that + = a
session with more trials than that mouse usually runs.

FDR: Benjamini-Hochberg separately per column, over every row drawn (behaviour and neural together).
Writes figures/ld1_vs_wpc1_correlates.{csv,png,pdf}.
"""
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
for p in (PI, HERE, PI / 'figure_generation'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import lab_controls as lc             # loads the LDA file (and its numpy-2 shim) on import
import nested_variance as nv
import paper_style as ps
import fig_layout as fl

lc.METHOD = 'pearson'                 # what the paper's figures report
N_PERM, N_BOOT, SEED = 5000, 2000, 0
rng = np.random.default_rng(SEED)
FIGDIR = HERE / 'figures'

# rows as fig_lab_controls' compact panels
ROWS_BEH = [('Task', 'Accuracy (easy trials)', 'Accuracy'),
            ('Task', 'Choice bias', 'Choice bias'),
            ('Task', 'Block bias shift', 'Block bias shift'),
            ('Task', 'Reaction time (log)', 'Reaction time'),
            ('Task', 'Trial elongation (log)', 'Trial elongation'),
            ('Task', 'Training time (log days)', 'Training time*'),
            ('Internal states', 'Fraction engaged (hard)', 'Fraction engaged'),
            ('Internal states', 'Engaged log posterior ratio', 'Engaged posterior'),
            ('Internal states', 'State switches (K=2)', 'State switches'),
            ('Internal states', 'State switches | easy accuracy', 'Switches | accuracy'),
            ('Internal states', 'P(stay engaged)', 'P(stay engaged)'),
            ('Internal states', 'P(stay disengaged)', 'P(stay disengaged)'),
            ('Internal states', 'Held-out fit gain K=2->3', 'Fit gain, K = 2 → 3'),
            ('Internal states', 'Held-out fit gain K=2->4', 'Fit gain, K = 2 → 4'),
            ('Internal states', 'State switches (K=4)', 'Switches, K = 4'),
            ('Entropy', 'Trial-mode entropy', 'Trial-mode entropy'),
            ('Entropy', 'Syllable entropy', 'Syllable entropy')]
ROWS_NEU = [(g, f'{m}, {t}', f'{short}, {lab}') for m, g, short in
            [('Firing rate', 'Firing rate', 'FR'), ('Fano factor', 'Fano factor', 'FF'),
             ('Noise correlation', 'Noise corr.', 'r$_{SC}$')]
            for t, lab in [('Baseline', 'pre'), ('Post-stimulus', 'post'), ('Response magnitude', 'change')]]


# ---------------------------------------------------------------------------------------------------------------
# within-mouse PC1, per session
# ---------------------------------------------------------------------------------------------------------------
def within_mouse_pc1():
    codes, meta = nv.load_trials()
    Y = nv.one_hot(ps.paw_codes(codes)).reshape(len(codes), -1)
    s_codes, s_ids = pd.factorize(meta['session'], sort=True)
    A, _ = nv._indicator(s_codes, len(meta))
    SM = np.asarray(A @ Y) / np.bincount(s_codes)[:, None]
    mouse = meta.groupby(s_codes)['mouse_name'].first().to_numpy()
    D = SM - pd.DataFrame(SM).groupby(mouse).transform('mean').to_numpy()
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    w = D @ Vt[0]
    n = np.bincount(s_codes).astype(float)
    n_dev = n - pd.Series(n).groupby(mouse).transform('mean').to_numpy()
    w *= np.sign(np.corrcoef(w, n_dev)[0, 1])           # + = more trials than the mouse usually runs
    return pd.Series(w, index=s_ids, name='wPC1')


ps.use('paper')
ps.use_paw_states('uniform')
WPC1 = within_mouse_pc1()


# ---------------------------------------------------------------------------------------------------------------
# run lab_controls' sections with correlate() hooked
# ---------------------------------------------------------------------------------------------------------------
CAPTURED = {}
_orig = lc.correlate


def _hook(sess, y, label, section, rng_, x='LD1', per_session=True, note=''):
    row = _orig(sess, y, label, section, rng_, x=x, per_session=per_session, note=note)
    CAPTURED[label] = dict(sess=sess.copy(), y=y, per_session=per_session, row=row)
    return row


lc.correlate = _hook
lc_rng = np.random.default_rng(SEED)
for name in ('trial', 'engage', 'entropy', 'neural_mouse'):
    print(f'--- lab_controls section: {name}')
    lc.SECTIONS[name](lc_rng)
lc.correlate = _orig


# ---------------------------------------------------------------------------------------------------------------
# within-mouse PC1 test on the same session tables
# ---------------------------------------------------------------------------------------------------------------
def wpc1_test(sess, y):
    s = sess.copy()
    s['wPC1'] = s['session'].map(WPC1)
    s = s[['mouse_name', 'wPC1', y]].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s.groupby('mouse_name')['wPC1'].transform('size') >= 2].reset_index(drop=True)
    dx = (s['wPC1'] - s.groupby('mouse_name')['wPC1'].transform('mean')).to_numpy()
    dy = (s[y] - s.groupby('mouse_name')[y].transform('mean')).to_numpy()
    r = np.corrcoef(dx, dy)[0, 1]
    groups = [np.flatnonzero(s['mouse_name'].to_numpy() == m) for m in s['mouse_name'].unique()]
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        px = dx.copy()
        for g in groups:
            px[g] = dx[rng.permutation(g)]
        null[k] = np.corrcoef(px, dy)[0, 1]
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot[b] = np.corrcoef(dx[idx], dy[idx])[0, 1]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(r_wpc1=r, p_wpc1=(1 + np.sum(np.abs(null) >= abs(r))) / (1 + N_PERM),
                r_wpc1_lo=lo, r_wpc1_hi=hi, n_sessions_wpc1=len(s), n_mice_wpc1=len(groups))


rows = []
for panel, spec in (('behaviour', ROWS_BEH), ('neural', ROWS_NEU)):
    for group, label, short in spec:
        if label not in CAPTURED:
            print(f'  not computed on this machine: {label}')
            continue
        c = CAPTURED[label]
        r = c['row']
        out = dict(panel=panel, group=group, correlate=label, label=short,
                   r_ld1=r['rho'], p_ld1=r['p'], r_ld1_lo=r['rho_lo'], r_ld1_hi=r['rho_hi'], n_mice_ld1=r['n_mice'])
        if c['per_session'] and 'session' in c['sess']:
            out.update(wpc1_test(c['sess'], c['y']))
        rows.append(out)
RES = pd.DataFrame(rows)
for col in ('ld1', 'wpc1'):
    ok = RES[f'p_{col}'].notna()
    RES[f'q_{col}'] = np.nan
    RES.loc[ok, f'q_{col}'] = multipletests(RES.loc[ok, f'p_{col}'], method='fdr_bh')[1]
FIGDIR.mkdir(exist_ok=True)
RES.to_csv(FIGDIR / 'ld1_vs_wpc1_correlates.csv', index=False)
with pd.option_context('display.width', 200):
    print(RES[['panel', 'label', 'r_ld1', 'q_ld1', 'n_mice_ld1', 'r_wpc1', 'q_wpc1', 'n_sessions_wpc1', 'n_mice_wpc1']]
          .round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------
# figure, in the compact figure's style: one column per axis, filled where q < 0.05
# ---------------------------------------------------------------------------------------------------------------
COLS = [('ld1', 'LD1\nacross mice', '#3A78B0'),
        ('wpc1', 'Within-mouse PC1\nacross sessions', '#7A5C9E')]


def columns(fig, spec, d, xlim, xticks, xlabel):
    ys = np.arange(len(d))
    g = spec.subgridspec(1, len(COLS), wspace=0.12)
    axes = []
    for j, (key, title, col) in enumerate(COLS):
        ax = fig.add_subplot(g[j], sharey=axes[0] if axes else None)
        for y, (_, r) in zip(ys, d.iterrows()):
            v = r[f'r_{key}']
            if not np.isfinite(v):
                continue
            sig = r[f'q_{key}'] < 0.05
            ax.plot([r[f'r_{key}_lo'], r[f'r_{key}_hi']], [y, y], color=col, lw=0.8, solid_capstyle='butt')
            ax.plot(v, y, 'o', ms=3.4, color=col, mfc=col if sig else 'white', mew=0.8, zorder=3)
        ax.axvline(0, color='0.6', lw=0.5, zorder=0)
        grp = d['group'].to_numpy()
        for i in range(1, len(grp)):
            if grp[i] != grp[i - 1]:
                ax.axhline(i - 0.5, color='0.85', lw=0.5, zorder=0)
        ax.set_xlim(*xlim); ax.set_xticks(xticks)
        ax.set_title(title, color=col, fontsize=plt.rcParams['font.size'] * 0.9, pad=3)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        if j:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(d['label'])
    axes[0].set_ylim(len(d) - 0.4, -0.6)
    fig.text(np.mean([a.get_position().x0 for a in axes] + [axes[-1].get_position().x1]), 0, '')
    axes[0].set_xlabel(xlabel, x=1.06)
    return axes


def group_labels(fig, ax, d, x):
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    grp = d['group'].to_numpy()
    for gname in dict.fromkeys(grp):
        idx = np.flatnonzero(grp == gname)
        ax.text(x, (idx.min() + idx.max()) / 2, gname, transform=tr, rotation=90, ha='center', va='center',
                color='0.4', fontsize=plt.rcParams['font.size'] * 0.85, clip_on=False)
        ax.plot([x + 0.008] * 2, [idx.min() - 0.3, idx.max() + 0.3], transform=tr, color='0.6', lw=0.6, clip_on=False)


fig = plt.figure(figsize=(fl.FIG_WIDTH, 3.9))
gc = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.85], wspace=0.6, left=0.17, right=0.985, top=0.9, bottom=0.12)
gr = gc[1].subgridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.75)
beh, neu = RES[RES.panel == 'behaviour'].reset_index(drop=True), RES[RES.panel == 'neural'].reset_index(drop=True)
pa = columns(fig, gc[0], beh, (-0.8, 0.8), [-0.5, 0, 0.5], 'Pearson r')
group_labels(fig, pa[0], beh, 0.012)
pb = columns(fig, gr[0], neu, (-0.6, 0.6), [-0.5, 0, 0.5], 'Pearson r (covariate-adjusted)')
key = [Line2D([], [], marker='o', ls='-', lw=0.8, ms=3.4, color='0.35', label='q < 0.05'),
       Line2D([], [], marker='o', ls='-', lw=0.8, ms=3.4, color='0.35', mfc='white', label='n.s.')]
fig.legend(handles=key, frameon=False, loc='upper left', bbox_to_anchor=(0.035, 0.995), ncol=2, handlelength=1.4,
           columnspacing=0.8)
fig.text(0.62, 0.02, '* one value per mouse: no within-mouse test.  Within-mouse PC1: + = session with more trials '
                     'than usual.', fontsize=plt.rcParams['font.size'] * 0.75, color='0.4')
fl.place_letters(fig, {'a': pa, 'b': pb}, rows=['ab'], left='a')
for path in [FIGDIR / 'ld1_vs_wpc1_correlates.png', FIGDIR / 'ld1_vs_wpc1_correlates.pdf']:
    fig.savefig(path, dpi=300)
    print('wrote', path.relative_to(PI))
