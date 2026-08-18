"""Lick state labels at different kappa -- judge the difference by eye.

Top panel:    CSHL053 810b1e07, kappa=0 vs kappa=100. This is a session where a paired
              test on raw held-out LL says kappa=100 is *significantly* better
              (+0.0000079 bits). Is that visible?
Bottom panel: NYU-12 a8a8af78, kappa=0 vs kappa=5e5 -- 500x beyond the searched grid and
              past the point where the prior outweighs the data. This is what a real
              kappa effect looks like, for contrast.

Run from this directory.
"""
import sys, os, re, pickle
SEG = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.insert(0, SEG); os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np, pandas as pd
import jax.numpy as jnp, jax.random as jr
from dynamax.hidden_markov_model import PoissonHMM
from segmentation_functions import cross_validate_poismodel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
BAND, RIBBON, DIFF = '#cde2fb', '#2a78d6', '#eb6834'
FPS, NB = 60.0, 5
WIN = int(30 * FPS)
F, DM = '../data/hmm/grid_search/5_prior_em_zsc_False', '../data/design_matrices/'
dmi = {f.split('_')[2]: DM + f for f in os.listdir(DM)
       if f.startswith('design_matrix') and 'standardized' not in f}

CASES = [
    dict(eid='810b1e07-009e-4ebe-930a-915e4cd8ece4', kappas=[0, 100], refit=[],
         note='paired test on raw held-out LL prefers κ=100 (+0.0000079 bits) — significant, and within the searched grid'),
    dict(eid='a8a8af78-16de-4841-ab07-fde4b5281a03', kappas=[0, 500000], refit=[500000],
         note='κ=5×10⁵ — 500× beyond the searched grid, and 59× this session’s rare-state evidence'),
]


def runs_of(m):
    e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int), [0]))))
    return list(zip(e[::2], e[1::2]))


def decode(short, ed, kap, fitp, fold):
    md = PoissonHMM(2, ed, transition_matrix_stickiness=kap)
    p, _ = md.initialize(key=jr.PRNGKey(0), method='prior',
                         initial_probs=fitp[0].probs[fold],
                         transition_matrix=np.asarray(fitp[1].transition_matrix)[fold],
                         emission_rates=fitp[2].rates[fold])
    s = np.asarray(md.most_likely_states(p, short))
    r = np.asarray(fitp[2].rates)[fold].ravel()
    return 1 - s if r[1] < r[0] else s


prepared = []
for c in CASES:
    eid = c['eid']
    pkl = [f for f in os.listdir(F) if f.endswith(eid) and f.startswith('best_results_Lick count_')][0]
    mouse = pkl[len('best_results_Lick count_'):-36]
    all_lls, all_base, _, fitp_all, dmx, params = pickle.load(open(os.path.join(F, pkl), 'rb'))
    _, _, ks = params; ks = list(ks)
    x = pd.read_parquet(dmi[eid])[['Lick count']].dropna().values
    nt, ed = x.shape
    short = np.array(x[:(nt // NB) * NB])
    tr = jnp.stack(jnp.split(short, NB)); fl = len(short) / NB
    seqs = {}
    for kap in c['kappas']:
        if kap in c['refit']:
            m = PoissonHMM(2, ed, transition_matrix_stickiness=kap)
            vll, fp, _, bll = cross_validate_poismodel(m, jr.PRNGKey(0), tr, NB, 'em')
            fold = int(np.nanargmax(np.asarray(vll)))
            seqs[kap] = decode(short, ed, kap, fp, fold)
        else:
            raw = np.asarray(all_lls[kap]) / fl
            fold = int(np.nanargmax(raw))
            seqs[kap] = decode(short, ed, kap, fitp_all[kap], fold)
    a, b = seqs[c['kappas'][0]], seqs[c['kappas'][1]]
    diff = a != b
    prepared.append(dict(mouse=mouse, eid=eid, kappas=c['kappas'], note=c['note'],
                         licks=short[:, 0], sa=a, sb=b, diff=diff))
    print(f"{mouse} {eid[:8]}  κ={c['kappas'][0]} vs {c['kappas'][1]}: "
          f"{diff.sum():,} of {len(diff):,} frames differ ({diff.mean():.4%}), "
          f"{len(runs_of(diff))} runs", flush=True)

# ---- figure ----
H = 3.3 * len(prepared) + 1.6
fig, axes = plt.subplots(len(prepared), 1, figsize=(12, H))
fig.patch.set_facecolor(SURFACE)
for ax, P in zip(np.atleast_1d(axes), prepared):
    lk, a, b, diff = P['licks'], P['sa'], P['sb'], P['diff']
    # window: containing disagreement if any, else the densest licking
    if diff.any():
        r = max(runs_of(diff), key=lambda t: t[1] - t[0])
        st = int(np.clip((r[0] + r[1]) // 2 - WIN // 2, 0, len(lk) - WIN))
    else:
        dens = pd.Series(lk).rolling(WIN).sum().values
        st = int(np.clip(np.nanargmax(dens) - WIN, 0, len(lk) - WIN))
    t = np.arange(WIN) / FPS
    L, A, B, D = lk[st:st + WIN], a[st:st + WIN], b[st:st + WIN], diff[st:st + WIN]
    ax.set_facecolor(SURFACE)
    for p, q in runs_of(B == 1):
        ax.axvspan(t[p], t[min(q, WIN - 1)], color=BAND, lw=0, zorder=0)
    for p, q in runs_of(D):
        ax.axvspan(t[p], t[min(q, WIN - 1)], color=DIFF, lw=0, alpha=0.45, zorder=1)
    ax.vlines(t[L > 0], 0.42, 0.92, color=INK, lw=0.8, zorder=3)
    for k, (seq, kap) in enumerate(zip((A, B), P['kappas'])):
        y = 0.06 + k * 0.15
        for p, q in runs_of(seq == 1):
            ax.add_patch(Rectangle((t[p], y), t[min(q, WIN - 1)] - t[p], 0.10,
                                   color=RIBBON, lw=0, zorder=4))
        ax.annotate(f'κ={kap:,.0f}', xy=(-0.008, y + 0.05), xycoords=('axes fraction', 'data'),
                    fontsize=8, color=INK_2, ha='right', va='center')
    ax.annotate('licks', xy=(-0.008, 0.67), xycoords=('axes fraction', 'data'),
                fontsize=8, color=INK_2, ha='right', va='center')
    ax.set_ylim(0, 1.0); ax.set_xlim(0, t[-1]); ax.set_yticks([])
    ax.set_title(f"{P['mouse']}  {P['eid'][:8]}   ·   κ={P['kappas'][0]:,} vs κ={P['kappas'][1]:,}   ·   "
                 f"{P['diff'].sum():,}/{len(P['diff']):,} frames differ ({P['diff'].mean():.3%})",
                 fontsize=10, color=INK, loc='left', pad=16)
    ax.annotate(P['note'], xy=(0, 1.02), xycoords='axes fraction', fontsize=8,
                color=INK_MUTED, ha='left')
    ax.set_xlabel('time (s)', fontsize=8.5, color=INK_2)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(INK_MUTED); ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)

fig.suptitle('Do different κ give different lick states?', fontsize=13, color=INK,
             x=0.006, ha='left', y=1 - 0.26 / H)
fig.text(0.006, 1 - 0.60 / H, 'black ticks = lick frames · pale blue = lick state (lower κ) · '
         'blue ribbons = the two state sequences · orange = frames where they disagree',
         fontsize=8.5, color=INK_MUTED, ha='left', va='top')
fig.subplots_adjust(top=1 - 1.05 / H, bottom=0.75 / H, left=0.075, right=0.985, hspace=0.62)
fig.savefig('lick_kappa_snippet.png', dpi=165, facecolor=SURFACE)
print('\nwrote lick_kappa_snippet.png')
