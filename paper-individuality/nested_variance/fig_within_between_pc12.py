"""
Within-between slopes for PC1 and PC2, drawn so the data behind each number are visible.

Rows a-b: example correlates. For each, two panels on the same axes (z units):
    between mice   one point per mouse: its mean PC score (x) and mean correlate (y); line = beta_B
    within mice    one point per session: its deviation from its own mouse's mean on both; line = beta_W
  Same slope in both panels = the same relationship between mice and from day to day (a shared axis);
  a slope only on the left = a trait-specific association.
Row c: every correlate, beta_B and beta_W side by side with 95% CIs (mice resampled), for PC1 and PC2.

The unit is the SESSION: its syllable profile projected on the PC, and its value of the correlate. Lines in a-b are
the observed least-squares slopes; c shows the slopes corrected for noise in the PC score (split-half reliability
0.98-0.99 for these two axes, so the two barely differ). Numbers from within_between_models.py.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import within_between_models as wb          # noqa: E402  (sets up paths and the numpy shim)
import paper_style as ps                    # noqa: E402

ps.use('paper')
ps.use_paw_states('uniform')
AXES = ['PC1', 'PC2']
EXAMPLES = [('PC1', 'State switches (K=2)', 'State switches'),
            ('PC1', 'Reaction time (log)', 'Reaction time (log)'),
            ('PC1', 'Fano factor, Response magnitude', 'Fano factor change'),
            ('PC2', 'P(stay disengaged)', 'P(stay disengaged)'),
            ('PC2', 'Held-out fit gain K=2->4', 'Fit gain K2→4'),
            ('PC2', 'Reaction time (log)', 'Reaction time (log)')]
C_B, C_W = '#22344A', '#3E8E54'
AXIS_NAME = {'PC1': 'PC1 (whisking/licking, 28%)', 'PC2': 'PC2 (16%)'}

scores, s_ids, mouse_all, share = wb.axis_scores()
CS = wb.correlate_sessions()
RES = pd.read_csv(wb.FIGDIR / 'within_between.csv')
sess_index = pd.Index(s_ids)
HEADERS = []


def session_table(axis, label):
    c = CS[(CS['correlate'] == label) & CS['session'].isin(sess_index)].replace([np.inf, -np.inf], np.nan).dropna(subset=['y'])
    idx = sess_index.get_indexer(c['session'])
    x = scores[axis][0]
    d = pd.DataFrame({'mouse': mouse_all[idx], 'x': x[idx] / x.std(),
                      'y': ((c['y'] - c['y'].mean()) / c['y'].std()).to_numpy()})
    return d[d.groupby('mouse')['x'].transform('size') >= 2]


def slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.polyfit(x, y, 1)


def example(fig, spec, axis, label, title):
    d = session_table(axis, label)
    m = d.groupby('mouse')[['x', 'y']].mean()
    dev = d[['x', 'y']] - d.groupby('mouse')[['x', 'y']].transform('mean')
    r = RES[(RES.axis == axis) & (RES.correlate == label)].iloc[0]
    g = spec.subgridspec(1, 2, wspace=0.08)
    axs = [fig.add_subplot(g[0]), None]
    axs[1] = fig.add_subplot(g[1], sharey=axs[0])
    lim_x = max(np.abs(m['x']).max(), np.abs(dev['x']).max()) * 1.1
    for ax, data, col, beta, lo, hi, name, n in (
            (axs[0], m, C_B, r['beta_b'], r['beta_b_lo'], r['beta_b_hi'], 'between mice', f'{len(m)} mice'),
            (axs[1], dev, C_W, r['beta_w'], r['beta_w_lo'], r['beta_w_hi'], 'within mice', f'{len(dev)} sessions')):
        ax.scatter(data['x'], data['y'], s=6 if ax is axs[0] else 3, color=col, alpha=0.6 if ax is axs[0] else 0.35, lw=0)
        b = slope(data['x'], data['y'])
        xx = np.array([-lim_x, lim_x])
        ax.plot(xx, np.polyval(b, xx), color=col, lw=1.2)
        ax.axhline(0, color='0.85', lw=0.5, zorder=0); ax.axvline(0, color='0.85', lw=0.5, zorder=0)
        ax.set_xlim(-lim_x, lim_x)
        ax.set_title(f'{name} ({n})\nβ = {beta:+.2f} [{lo:+.2f}, {hi:+.2f}]', fontsize=5.2, color=col, pad=2)
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel(f'{axis} (z)' if ax is axs[0] else f'{axis}, session − mouse mean', fontsize=6)
    axs[1].tick_params(labelleft=False)
    axs[0].set_ylabel(f'{title} (z)', fontsize=6)
    q = r['q_c']
    verdict = ('slopes differ' if q < 0.05 else 'CI excludes 0, n.s. after FDR'
               if (r['beta_c_lo'] > 0 or r['beta_c_hi'] < 0) else 'slopes not distinguishable')
    HEADERS.append((axs, f'{title} vs {axis}',
                    f'β_B − β_W = {r["beta_c"]:+.2f} [{r["beta_c_lo"]:+.2f}, {r["beta_c_hi"]:+.2f}], q = {q:.2f}\n{verdict}'))
    return axs


def forest(fig, spec, axis):
    ax = fig.add_subplot(spec)
    d = RES[RES.axis == axis].reset_index(drop=True)
    y = np.arange(len(d))
    for off, col, key, lab in ((-0.17, C_B, 'beta_b', 'β between mice'), (0.17, C_W, 'beta_w', 'β within mice')):
        ax.errorbar(d[key], y + off, xerr=[d[key] - d[f'{key}_lo'], d[f'{key}_hi'] - d[key]], fmt='o', ms=2.6,
                    color=col, elinewidth=0.7, capsize=0, label=lab)
    for i, r in d.iterrows():
        if r['beta_c_lo'] > 0 or r['beta_c_hi'] < 0:
            ax.text(1.02, i, '*' if r['q_c'] < 0.05 else '†', transform=ax.get_yaxis_transform(), va='center',
                    fontsize=7, color='0.2')
    grp = d['group'].to_numpy()
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axhline(i - 0.5, color='0.85', lw=0.5)
    ax.axvline(0, color='0.6', lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(d['label'], fontsize=5.5)
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlabel('slope, SD(correlate) per SD(axis)', fontsize=6)
    ax.set_title(AXIS_NAME[axis], fontsize=7)
    return ax


fig = plt.figure(figsize=(7.09, 9.6))
G = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2.3], hspace=0.75, left=0.1, right=0.96, top=0.9, bottom=0.05)
top = G[0].subgridspec(1, 3, wspace=0.3)
mid = G[1].subgridspec(1, 3, wspace=0.3)
for spec, (axis, label, title) in zip(list(top) + list(mid), EXAMPLES):
    example(fig, spec, axis, label, title)
bot = G[2].subgridspec(1, 2, wspace=0.75)
f1 = forest(fig, bot[0], 'PC1')
forest(fig, bot[1], 'PC2')
f1.legend(frameon=False, fontsize=6, loc='lower left', bbox_to_anchor=(0, 1.06), ncol=2)
# example headers above each pair of panels, placed from the axes' final positions
for axs, head, sub in HEADERS:
    b0, b1 = axs[0].get_position(), axs[1].get_position()
    fig.text(b0.x0, b0.y1 + 0.043, head, fontsize=6.5, fontweight='bold')
    fig.text(b0.x0, b0.y1 + 0.024, sub, fontsize=5.4, color='0.3', va='bottom', linespacing=1.15)
for letter, axs_, txt in (('a', HEADERS[0][0][0], 'PC1 examples'), ('b', HEADERS[3][0][0], 'PC2 examples'),
                          ('c', f1, 'all correlates: between-mouse vs within-mouse slope')):
    b = axs_.get_position()
    fig.text(0.01, b.y1 + (0.075 if letter != 'c' else 0.05), f'{letter}  {txt}', fontsize=8, fontweight='bold')
fig.text(0.1, 0.012, 'Unit: the session (its syllable profile projected on the PC, and its value of the correlate). '
                     '* β_B ≠ β_W, FDR q < 0.05;  † 95% CI of β_B − β_W excludes 0 but q ≥ 0.05.',
         fontsize=5.8, color='0.35')
for path in (wb.FIGDIR / 'within_between_pc12.png', wb.FIGDIR / 'within_between_pc12.pdf'):
    fig.savefig(path, dpi=300)
    print('wrote', path.relative_to(wb.PI))
