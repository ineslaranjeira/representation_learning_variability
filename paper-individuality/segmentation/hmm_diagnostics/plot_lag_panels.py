"""Whisker trace + HMM state overlay: lag selected by the current rule vs the
corrected paired-increment rule, for the sessions flagged as problematic.

kappa is held at 0 throughout -- this isolates the effect of the lag.
No refitting: every (lag, kappa=0) cell is already in the stored grid-search pickle.

When the corrected rule picks the SAME lag as the current one, a third reference
panel at the longest lag in the grid is added, to show whether the session is
fixable by lag selection at all.
"""
import sys, os, re, pickle
SEG = '/home/ines/repositories/representation_learning_variability/paper-individuality/segmentation'
sys.path.insert(0, SEG)
os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np, pandas as pd
from scipy.stats import zscore
import jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM
from segmentation_functions import get_bits_LL, compute_inputs
from paired_selection import find_best_param_paired
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
BAND, RIBBON = '#cde2fb', '#2a78d6'
FPS = 60.0
NB = 5
WIN = int(15 * FPS)
F, DM = '../data/hmm/grid_search/5_prior_em_zsc_True', '../data/design_matrices/'
OUTDIR = os.environ['OUTDIR']
os.makedirs(OUTDIR, exist_ok=True)

# eid prefix -> why it is on the list
FLAGGED = {
    '0deb75fb': 'your bad-fit list; lowest bits_LL of 321',
    'a71175be': 'your bad-fit list; 4th-lowest bits_LL',
    '02fbb6da': 'your bad-fit list; 3rd-lowest bits_LL',
    '510b1a50': 'your bad-fit list; 2nd-lowest bits_LL',
    '49368f16': 'your bad-fit list; largest lag-1 penalty in the cohort',
    '7f6b86f9': 'your bad-fit list; 2 of 5 folds failed in every grid cell',
    'ee212778': 'my flag: bits_LL 0.336, below your 0.35 threshold',
    '87ad026d': 'my flag: bits_LL 0.345, below your 0.35 threshold',
    'a8a8af78': 'my flag: median dwell 1 frame (already in sessions_to_exclude)',
    '8c33abef': 'my flag: median dwell 7 frames (already in sessions_to_exclude)',
    'd0ea3148': 'my flag: median dwell 7 frames, but high bits_LL',
    '9b5a1754': 'my flag: median dwell 8 frames, but high bits_LL',
    '09b2c4d1': 'my flag: median dwell 9 frames, but high bits_LL',
    '63f3dbc1': 'my flag: median dwell 9 frames, but high bits_LL',
    'c4432264': 'my flag: median dwell 10 frames, but high bits_LL',
    '88224abb': 'my flag: median dwell 10 frames, but high bits_LL',
}

files = {}
for f in os.listdir(F):
    if not f.startswith('best_results_whisker_me_'):
        continue
    m = re.search(r'([0-9a-f-]{36})$', f)
    if m and m.group(1)[:8] in FLAGGED:
        files[m.group(1)] = (f, f[len('best_results_whisker_me_'):-36])

summary = []
for eid, (pkl, mouse) in sorted(files.items(), key=lambda kv: kv[1][1]):
    dmf = DM + f'design_matrix_{eid}_{mouse}'
    if not os.path.exists(dmf):
        print(f'SKIP {mouse} {eid[:8]}: no design matrix'); continue
    arr = zscore(np.array(pd.read_parquet(dmf)[['whisker_me']].dropna()),
                 axis=0, nan_policy='omit')
    nt, ed = arr.shape
    short = np.array(arr[:(nt // NB) * NB])

    all_lls, all_base, _, allfit, _, params = pickle.load(open(os.path.join(F, pkl), 'rb'))
    _, Lags, kappas = params
    bits, _, bestfold = get_bits_LL(all_lls, all_base, arr, NB, params, 2)
    ik0 = list(kappas).index(0)

    # lag chosen by the current rule vs the corrected sequential paired rule
    from segmentation_functions import find_best_param
    _, cur_lag, _ = find_best_param(bits, params, 2)
    _, new_lag, _ = find_best_param_paired(bits, params, 2, rule='ttest')

    wanted = [(cur_lag, 'Current rule')]
    if new_lag != cur_lag:
        wanted.append((new_lag, 'Corrected rule'))
    else:
        wanted.append((max(Lags), 'Longest lag in grid (reference)'))

    panels = []
    for lag, label in wanted:
        il = list(Lags).index(lag)
        bf = bestfold[ik0, il]
        if np.isnan(bf):
            continue
        fold = int(bf)
        fitp = allfit[lag][0]
        inp = compute_inputs(short, lag, ed)
        m = LinearAutoregressiveHMM(2, ed, num_lags=lag, transition_matrix_stickiness=0)
        A = np.asarray(fitp[1].transition_matrix)[fold]
        p, _ = m.initialize(key=jr.PRNGKey(0), method='prior',
                            initial_probs=fitp[0].probs[fold], transition_matrix=A,
                            emission_weights=fitp[2].weights[fold],
                            emission_biases=fitp[2].biases[fold],
                            emission_covariances=fitp[2].covs[fold], emissions=short)
        s = np.asarray(m.most_likely_states(p, short, inp))
        if short[s == 1, 0].mean() < short[s == 0, 0].mean():
            s = 1 - s
        ch = np.where(np.diff(s) != 0)[0]
        runs = np.diff(np.concatenate(([0], ch + 1, [len(s)])))
        panels.append(dict(lag=lag, label=label, states=s,
                           med=float(np.median(runs)), n=len(runs),
                           bits=float(np.nanmean(bits[ik0, il])),
                           diag=float(np.mean(np.diag(A)))))
    if len(panels) < 2:
        print(f'SKIP {mouse} {eid[:8]}: fewer than 2 usable panels'); continue

    # window: balanced occupancy + a few transitions under the better model
    ref = panels[-1]['states']
    best, bs = 0, -1e9
    for c in range(0, len(ref) - WIN, int(FPS * 2)):
        seg = ref[c:c + WIN]
        sc = -abs(float(seg.mean()) - 0.5) * 12 - abs(int(np.sum(np.diff(seg) != 0)) - 6) * 0.5
        if sc > bs:
            bs, best = sc, c
    st = best
    t = np.arange(WIN) / FPS
    sig = short[st:st + WIN, 0]

    H = 2.55 * len(panels) + 1.9
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, H), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        s = p['states'][st:st + WIN]
        lo, hi = sig.min() - 0.35, sig.max() + 0.55
        edges = np.flatnonzero(np.diff(np.concatenate(([0], s, [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[min(b, WIN - 1)], color=BAND, lw=0, zorder=0)
        for a, b in zip(edges[::2], edges[1::2]):
            ax.add_patch(Rectangle((t[a], lo + 0.02), t[min(b, WIN - 1)] - t[a], 0.16,
                                   color=RIBBON, lw=0, zorder=3))
        ax.plot(t, sig, color=INK, lw=0.9, zorder=2)
        ax.set_ylim(lo, hi); ax.set_xlim(0, t[-1])
        ax.set_title(f"{p['label']}   ·   lag {p['lag']}, $\\kappa$ = 0   ·   "
                     f"median dwell {p['med']:.0f} frames ({p['med']/FPS*1000:.0f} ms)   ·   "
                     f"{p['n']:,} segments   ·   held-out {p['bits']:.3f} bits",
                     fontsize=9.5, color=INK, loc='left', pad=6)
        ax.set_ylabel('whisker ME (z)', fontsize=8.5, color=INK_2)
        for sd in ('top', 'right'):
            ax.spines[sd].set_visible(False)
        for sd in ('left', 'bottom'):
            ax.spines[sd].set_color(INK_MUTED); ax.spines[sd].set_linewidth(0.6)
        ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
        ax.grid(axis='y', color=INK_MUTED, lw=0.4, alpha=0.28, zorder=0)
    axes[-1].set_xlabel('time (s)', fontsize=8.5, color=INK_2)

    # header laid out in absolute inches so it never collides with the first panel title
    fig.suptitle(f'{mouse}   {eid[:8]}  —  {int(WIN/FPS)} s of whisking, lag effect only',
                 fontsize=12.5, color=INK, x=0.006, ha='left', y=1 - 0.28 / H)
    fig.text(0.006, 1 - 0.62 / H, FLAGGED[eid[:8]],
             fontsize=8.5, color=INK_MUTED, ha='left', va='top')
    fig.text(0.006, 1 - 0.95 / H,
             'shaded band and ribbon = high-whisking state; unshaded = low-whisking state',
             fontsize=8.5, color=RIBBON, ha='left', va='top')
    fig.subplots_adjust(top=1 - 1.5 / H, bottom=0.9 / H, left=0.075, right=0.985, hspace=0.5)
    out = os.path.join(OUTDIR, f'lag_{mouse}_{eid[:8]}.png')
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    a, b = panels[0], panels[-1]
    summary.append(dict(mouse=mouse, eid=eid[:8], reason=FLAGGED[eid[:8]],
                        lag_a=a['lag'], lag_b=b['lag'], same=(a['lag'] == b['lag']),
                        dwell_a=a['med'], dwell_b=b['med'], nseg_a=a['n'], nseg_b=b['n'],
                        bits_a=round(a['bits'], 3), bits_b=round(b['bits'], 3),
                        d_bits=round(b['bits'] - a['bits'], 3)))
    print(f"{mouse:14s} {eid[:8]}  lag {a['lag']:2d}->{b['lag']:2d}  "
          f"dwell {a['med']:5.0f}->{b['med']:5.0f}  bits {a['bits']:.3f}->{b['bits']:.3f}", flush=True)

s = pd.DataFrame(summary)
s.to_csv(os.path.join(OUTDIR, '_summary.csv'), index=False)
print(f'\nwrote {len(s)} figures to {OUTDIR}')
