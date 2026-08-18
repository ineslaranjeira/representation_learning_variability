"""2-state Gaussian HMM vs AR-HMM: what the segmentation looks like.

One row per session: the whisker ME trace, the Gaussian states as the shaded band and
lower ribbon, the AR states as the upper ribbon, and the frames where they disagree in
orange. Titles carry the numbers that matter -- syllable duration against the model-free
changepoint anchor, and frame agreement.

Run from this directory, after gaussian_vs_ar.py.
"""
import os, sys, pickle
sys.path.insert(0, '..')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmm_dynamic_functions import load_fit_variable, prepare_batches, dwell_times

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
S1, S2 = '#2a78d6', '#eb6834'
BAND = '#cde2fb'
AR = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
GA = '../../data/hmm/grid_search_gaussian/5_kmeans_em_zsc_True_gaussian/'
DM = '../../data/design_matrices/'
PRE = 'best_results_whisker_me_'
FPS, WIN_S = 60.0, 15
ANCHOR, A_LO, A_HI = 450., 379., 550.


def runs_of(m):
    e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int), [0]))))
    return list(zip(e[::2], e[1::2]))


t = pd.read_csv('gaussian_vs_ar.csv')
rows = []
for _, r in t.iterrows():
    hits = [n for n in os.listdir(GA) if n.startswith(PRE) and n[-36:].startswith(r.eid)]
    if not hits:
        continue
    name = hits[0]; mouse, eid = name[len(PRE):-36], name[-36:]
    g = pickle.load(open(GA + name, 'rb'))
    a = pickle.load(open(AR + name, 'rb'))
    sa = np.asarray(a['most_likely_states'])
    sg = np.asarray(g['most_likely_states'])[:len(sa)]
    sig = np.asarray(prepare_batches(
        load_fit_variable(DM, eid, mouse, ['whisker_me'], True), 5)[0])[:len(sa), 0]
    rows.append((mouse, r.eid, sig, sg, sa, a['best_lag'], g, a))

n = len(rows)
H = 2.9 * n + 1.6
fig, axes = plt.subplots(n, 1, figsize=(13, H))
fig.patch.set_facecolor(SURFACE)
for ax, (mouse, e8, sig, sg, sa, lag, g, a) in zip(np.atleast_1d(axes), rows):
    win = int(WIN_S * FPS)
    # centre on the largest disagreement -- the honest place to look
    d = sg != sa
    rr = runs_of(d)
    st = ((sum(max(rr, key=lambda x: x[1] - x[0])) // 2) - win // 2) if rr else 0
    st = int(np.clip(st, 0, max(len(sa) - win, 0)))
    tt = np.arange(win) / FPS
    L, G_, A_, D = sig[st:st+win], sg[st:st+win], sa[st:st+win], d[st:st+win]
    lo, hi = np.nanmin(L) - 0.85, np.nanmax(L) + 0.5

    ax.set_facecolor(SURFACE)
    for p, q in runs_of(G_ == 1):
        ax.axvspan(tt[p], tt[min(q, win-1)], color=BAND, lw=0, zorder=0)
    for p, q in runs_of(D):
        ax.axvspan(tt[p], tt[min(q, win-1)], color=S2, alpha=0.38, lw=0, zorder=1)
    for k, (seq, lab) in enumerate([(G_, 'Gaussian'), (A_, f'AR lag {lag}')]):
        y = lo + 0.06 + k * 0.24
        for p, q in runs_of(seq == 1):
            ax.fill_between([tt[p], tt[min(q, win-1)]], y, y + 0.16, color=S1, lw=0, zorder=4)
        ax.annotate(lab, xy=(-0.006, y + 0.08), xycoords=('axes fraction', 'data'),
                    fontsize=7.5, color=INK_2, ha='right', va='center')
    ax.plot(tt, L, color=INK, lw=0.9, zorder=2)
    ax.set_xlim(0, tt[-1]); ax.set_ylim(lo, hi)
    ax.set_ylabel('whisker ME (z)', fontsize=8.5, color=INK_2)
    dg, da = dwell_times(sg), dwell_times(sa)
    mg, ma = np.median(dg)*1000/FPS, np.median(da)*1000/FPS
    flag = ''
    if not (A_LO <= ma <= A_HI): flag += '  AR off anchor'
    if not (A_LO <= mg <= A_HI): flag += '  Gaussian off anchor'
    ax.set_title(f"{mouse}  {e8}   ·   dwell: AR {ma:.0f} ms → Gaussian {mg:.0f} ms   ·   "
                 f"{np.mean(sg == sa):.1%} of frames agree{flag}",
                 fontsize=9.5, color=INK, loc='left', pad=8)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
np.atleast_1d(axes)[-1].set_xlabel('time (s)', fontsize=8.5, color=INK_2)

fig.suptitle('2-state Gaussian HMM vs AR-HMM — same data, same folds, no lag to choose',
             fontsize=13, color=INK, x=0.006, ha='left', y=1 - 0.25/H)
fig.text(0.006, 1 - 0.56/H,
         'pale blue = Gaussian high state · ribbons = the two segmentations · orange = disagreement · '
         f'window centred on the largest one · anchor = model-free changepoint duration {ANCHOR:.0f} ms '
         f'(IQR {A_LO:.0f}-{A_HI:.0f})',
         fontsize=8.3, color=INK_MUTED, ha='left', va='top')
fig.subplots_adjust(top=1 - 1.05/H, bottom=0.62/H, left=0.085, right=0.99, hspace=0.52)
fig.savefig('gaussian_vs_ar.png', dpi=155, facecolor=SURFACE)
print('wrote gaussian_vs_ar.png')
