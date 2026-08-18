"""Whisker trace with HMM state overlay, for one session under three model settings.

Shows why the current hyperparameter selection produces a degenerate segmentation
on NYU-37 / 7af49c00, and what the corrected lag and a properly-scaled kappa do.
"""
import sys, os, re, pickle
SEG = '/home/ines/repositories/representation_learning_variability/paper-individuality/segmentation'
sys.path.insert(0, SEG)
os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np, pandas as pd
from scipy.stats import zscore
import jax.numpy as jnp, jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM
from segmentation_functions import get_bits_LL, compute_inputs, cross_validate_armodel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- palette (reference instance, light surface) ---
SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
INK_MUTED = '#87867f'
BAND      = '#cde2fb'   # sequential blue step 100 -- recessive region fill
RIBBON    = '#2a78d6'   # categorical slot 1 -- carries state identity at full contrast

EID = '7af49c00-63dd-4fed-b2e0-1b3bd945b20b'
MOUSE = 'NYU-37'
FPS = 60.0
NB = 5
WIN = int(15 * FPS)          # 15 s snippet
OUT = os.environ['OUTPNG']

F = '../data/hmm/grid_search/5_prior_em_zsc_True'
DM = '../data/design_matrices/'

arr = zscore(np.array(pd.read_parquet(DM + f'design_matrix_{EID}_{MOUSE}')[['whisker_me']].dropna()),
             axis=0, nan_policy='omit')
nt, ed = arr.shape
short = np.array(arr[:(nt // NB) * NB])
train_em = jnp.stack(jnp.split(short, NB))
fold_len = len(short) / NB

all_lls, all_base, _, allfit, _, params = pickle.load(
    open(os.path.join(F, f'best_results_whisker_me_{MOUSE}{EID}'), 'rb'))
_, Lags, kappas = params
bits_grid, _, bestfold = get_bits_LL(all_lls, all_base, arr, NB, params, 2)


def decode(lag, kappa, fitp, fold):
    inp = compute_inputs(short, lag, ed)
    m = LinearAutoregressiveHMM(2, ed, num_lags=lag, transition_matrix_stickiness=kappa)
    p, _ = m.initialize(key=jr.PRNGKey(0), method='prior',
                        initial_probs=fitp[0].probs[fold],
                        transition_matrix=np.asarray(fitp[1].transition_matrix)[fold],
                        emission_weights=fitp[2].weights[fold],
                        emission_biases=fitp[2].biases[fold],
                        emission_covariances=fitp[2].covs[fold],
                        emissions=short)
    A = np.asarray(fitp[1].transition_matrix)[fold]
    return np.asarray(m.most_likely_states(p, short, inp)), float(np.mean(np.diag(A)))


panels = []
# (1) what the current rule selects, and (2) the corrected lag -- both already fitted
for lag, kap, label in [(1, 0, 'Current rule'), (10, 0, 'Corrected lag')]:
    fold = int(bestfold[list(kappas).index(kap), list(Lags).index(lag)])
    s, dg = decode(lag, kap, allfit[lag][kap], fold)
    b = float(np.nanmean(bits_grid[list(kappas).index(kap), list(Lags).index(lag)]))
    panels.append(dict(lag=lag, kappa=kap, label=label, states=s, diag=dg, bits=b))

# (3) properly-scaled kappa -- needs a fresh fit
lag = 10
kap = 20000.0
inp = compute_inputs(short, lag, ed)
train_in = jnp.stack(jnp.split(inp, NB))
m = LinearAutoregressiveHMM(2, ed, num_lags=lag, transition_matrix_stickiness=kap)
vll, fitp, _, bll = cross_validate_armodel(m, jr.PRNGKey(0), train_em, train_in, 'prior', NB, 'em')
bits = (np.asarray(vll) - np.asarray(bll)) / fold_len * np.log(2)
fold = int(np.nanargmax(bits))
s, dg = decode(lag, kap, fitp, fold)
panels.append(dict(lag=lag, kappa=kap, label='Corrected lag + rescaled $\\kappa$',
                   states=s, diag=dg, bits=float(np.nanmean(bits))))

# orient every panel so state 1 == the higher-motion state, and collect stats
for p in panels:
    s = p['states']
    if short[s == 1, 0].mean() < short[s == 0, 0].mean():
        s = 1 - s
        p['states'] = s
    ch = np.where(np.diff(s) != 0)[0]
    runs = np.diff(np.concatenate(([0], ch + 1, [len(s)])))
    p['med_dwell'] = float(np.median(runs))
    p['n_seg'] = len(runs)

# pick a snippet where the corrected-lag model actually alternates: balanced
# occupancy and a handful of transitions, so both states are visible on screen
ref = panels[1]['states']
best, best_score = 0, -1e9
for cand in range(0, len(ref) - WIN, int(FPS * 2)):
    seg = ref[cand:cand + WIN]
    nch = int(np.sum(np.diff(seg) != 0))
    occ = float(seg.mean())
    score = -abs(occ - 0.5) * 12 - abs(nch - 6) * 0.5
    if score > best_score:
        best_score, best = score, cand
st = best
print(f'window occ={ref[st:st+WIN].mean():.2f} '
      f'changes={int(np.sum(np.diff(ref[st:st+WIN])!=0))}')
t = np.arange(WIN) / FPS
sig = short[st:st + WIN, 0]

fig, axes = plt.subplots(len(panels), 1, figsize=(11, 7.2), sharex=True,
                         gridspec_kw=dict(hspace=0.42))
fig.patch.set_facecolor(SURFACE)

for ax, p in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    s = p['states'][st:st + WIN]
    lo, hi = sig.min() - 0.35, sig.max() + 0.55
    # state bands
    edges = np.flatnonzero(np.diff(np.concatenate(([0], s, [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, WIN - 1)], color=BAND, lw=0, zorder=0)
    # state ribbon -- keeps fast flicker visible when the bands merge
    ribbon_y, ribbon_h = lo + 0.02, 0.16
    for a, b in zip(edges[::2], edges[1::2]):
        ax.add_patch(Rectangle((t[a], ribbon_y), t[min(b, WIN - 1)] - t[a], ribbon_h,
                               color=RIBBON, lw=0, zorder=3))
    ax.plot(t, sig, color=INK, lw=0.9, zorder=2)
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, t[-1])
    kt = '0' if p['kappa'] == 0 else f"{p['kappa']:,.0f}"
    ax.set_title(f"{p['label']}   ·   lag {p['lag']}, $\\kappa$ = {kt}   ·   "
                 f"median dwell {p['med_dwell']:.0f} frames "
                 f"({p['med_dwell']/FPS*1000:.0f} ms)   ·   {p['n_seg']:,} segments   ·   "
                 f"held-out {p['bits']:.3f} bits",
                 fontsize=9.5, color=INK, loc='left', pad=6)
    ax.set_ylabel('whisker ME (z)', fontsize=8.5, color=INK_2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    ax.grid(axis='y', color=INK_MUTED, lw=0.4, alpha=0.28, zorder=0)

axes[-1].set_xlabel('time (s)', fontsize=8.5, color=INK_2)
fig.suptitle(f'Same {int(WIN/FPS)} s of whisking, three hyperparameter settings — {MOUSE}  {EID[:8]}',
             fontsize=12.5, color=INK, x=0.006, ha='left', y=0.988)
fig.text(0.006, 0.945, 'shaded band and ribbon = high-whisking state; unshaded = low-whisking state',
         fontsize=8.5, color=RIBBON, ha='left', va='top')
fig.text(0.006, 0.012,
         'AR-HMM, 2 states, whisker motion energy z-scored per session. '
         'Same session, same data, same fold-selection procedure — only lag and $\\kappa$ differ.',
         fontsize=8, color=INK_MUTED, ha='left')
fig.subplots_adjust(top=0.885, bottom=0.10, left=0.075, right=0.985, hspace=0.52)
fig.savefig(OUT, dpi=170, facecolor=SURFACE)
print('wrote', OUT)
for p in panels:
    print(f"  lag {p['lag']:2d} k={p['kappa']:>8,.0f}  dwell {p['med_dwell']:5.1f}  "
          f"nseg {p['n_seg']:6,}  diag {p['diag']:.6f}  bits {p['bits']:.4f}")
print('snippet start frame', st, f'= {st/FPS:.0f}s into session')
