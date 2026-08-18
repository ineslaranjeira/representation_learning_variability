"""Fit sheet for the dynamic search: one row per session, plus the lag-selection profiles.

Reads the rich pickles written by hmm_dynamic_functions.run_session -- no refitting, the
state sequence is already stored.

Env: RESULTS_DIR (contains best_results_*), DM_DIR, OUTPREFIX, FPS.
"""
import os, re, glob, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
BAND, RIBBON, ACC = '#cde2fb', '#2a78d6', '#eb6834'

RES = os.environ['RESULTS_DIR']
DM = os.environ.get('DM_DIR', '../data/design_matrices/')
OUT = os.environ.get('OUTPREFIX', 'dynamic_fits')
FPS = float(os.environ.get('FPS', '60'))
WIN = int(15 * FPS)


def runs_of(m):
    e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int), [0]))))
    return list(zip(e[::2], e[1::2]))


files = sorted(glob.glob(os.path.join(RES, 'best_results_*')))
sessions = []
for f in files:
    m = re.search(r'([0-9a-f-]{36})$', f)
    if not m:
        continue
    eid = m.group(1)
    mouse = os.path.basename(f).split('_results_')[1][:-36]
    for _v in ('whisker_me_', 'Lick count_', 'avg_wheel_vel_'):
        if mouse.startswith(_v):
            mouse = mouse[len(_v):]
    d = pickle.load(open(f, 'rb'))
    var = d['config']['var_interest'][0]
    dmf = [p for p in os.listdir(DM) if p.startswith(f'design_matrix_{eid}')]
    if not dmf:
        continue
    sig = pd.read_parquet(os.path.join(DM, dmf[0]), columns=[var])[var].dropna().values
    sig = (sig - np.nanmean(sig)) / np.nanstd(sig)
    s = np.asarray(d['most_likely_states'])
    sessions.append(dict(eid=eid, mouse=mouse, d=d, sig=sig[:len(s)], states=s))
print(f'{len(sessions)} sessions loaded')

# ---------------- figure 1: snippet per session ----------------
n = len(sessions)
H = 2.35 * n + 1.7
fig, axes = plt.subplots(n, 1, figsize=(12, H))
fig.patch.set_facecolor(SURFACE)
for ax, S in zip(np.atleast_1d(axes), sessions):
    s, sig, d = S['states'], S['sig'], S['d']
    # window with balanced occupancy and a handful of transitions
    best, bs = 0, -1e9
    for c in range(0, max(len(s) - WIN, 1), int(FPS * 2)):
        seg = s[c:c + WIN]
        if len(seg) < WIN:
            break
        sc = -abs(float(seg.mean()) - 0.5) * 12 - abs(int(np.sum(np.diff(seg) != 0)) - 7) * 0.4
        if sc > bs:
            bs, best = sc, c
    st = best
    t = np.arange(WIN) / FPS
    L, A = sig[st:st + WIN], s[st:st + WIN]
    lo, hi = np.nanmin(L) - 0.35, np.nanmax(L) + 0.5
    ax.set_facecolor(SURFACE)
    for p, q in runs_of(A == 1):
        ax.axvspan(t[p], t[min(q, WIN - 1)], color=BAND, lw=0, zorder=0)
        ax.add_patch(Rectangle((t[p], lo + 0.02), t[min(q, WIN - 1)] - t[p], 0.14,
                               color=RIBBON, lw=0, zorder=3))
    ax.plot(t, L, color=INK, lw=0.9, zorder=2)
    ax.set_ylim(lo, hi); ax.set_xlim(0, t[-1])
    a = d['assessment']
    ax.set_title(f"{S['mouse']} {S['eid'][:8]}   ·   τ={d['tau']:.0f} → cap {d['cap']}   ·   "
                 f"lag {d['best_lag']}{' (AT CAP)' if d['at_cap'] else ''}   ·   "
                 f"dwell {a['median_dwell_ms']:.0f} ms   ·   {a['n_segments']:,} segments   ·   "
                 f"{'OK' if a['fit_ok'] else 'FLAGGED'}",
                 fontsize=9.5, color=INK if a['fit_ok'] else ACC, loc='left', pad=6)
    ax.set_ylabel('signal (z)', fontsize=8.5, color=INK_2)
    for sd in ('top', 'right'):
        ax.spines[sd].set_visible(False)
    for sd in ('left', 'bottom'):
        ax.spines[sd].set_color(INK_MUTED); ax.spines[sd].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
np.atleast_1d(axes)[-1].set_xlabel('time (s)', fontsize=8.5, color=INK_2)
fig.suptitle('Dynamic search — fitted segmentations', fontsize=13, color=INK,
             x=0.006, ha='left', y=1 - 0.26 / H)
fig.text(0.006, 1 - 0.58 / H, 'pale blue band and ribbon = high state; unshaded = low state · '
         '15 s window chosen for balanced occupancy',
         fontsize=8.5, color=INK_MUTED, ha='left', va='top')
fig.subplots_adjust(top=1 - 1.0 / H, bottom=0.7 / H, left=0.075, right=0.985, hspace=0.55)
fig.savefig(f'{OUT}_snippets.png', dpi=160, facecolor=SURFACE)
print(f'wrote {OUT}_snippets.png')

# ---------------- figure 2: the profile AND the paired comparison ----------------
# Row 1: absolute mean held-out LL per lag -- context only.
# Row 2: the quantity the rule ACTUALLY tests -- the PAIRED per-fold difference against
#        the currently adopted lag, with its 95% CI (t_crit * paired SE, df = n_folds-1).
#        The absolute LL's across-fold SD is NOT shown because it is not what is tested:
#        for ff96bfe1 it is 0.34 while the paired SE is 0.026, a 13x difference. That gap
#        is the whole point of the paired test.
from scipy.stats import t as _t

fig, axes = plt.subplots(2, n, figsize=(2.6 * n + 1.2, 6.4), squeeze=False)
fig.patch.set_facecolor(SURFACE)
for col, S in enumerate(sessions):
    d = S['d']
    lags = sorted(d['lag_profile'])

    # --- row 1: absolute LL ---
    ax = axes[0][col]
    y = [d['lag_profile'][l] for l in lags]
    ax.set_facecolor(SURFACE)
    ax.plot(lags, y, color=INK_MUTED, lw=1.4, zorder=2)
    ax.scatter(lags, y, s=18, color=INK_MUTED, zorder=3)
    ax.scatter([d['best_lag']], [d['lag_profile'][d['best_lag']]], s=90, facecolor='none',
               edgecolor=ACC, lw=2, zorder=4)
    ax.axvline(d['cap'], color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_title(f"{S['mouse']}\nadopted lag {d['best_lag']}", fontsize=9, color=INK, pad=6)
    if col == 0:
        ax.set_ylabel('mean held-out LL / frame\n(context)', fontsize=8.5, color=INK_2)

    # --- row 2: paired gains with 95% CI ---
    ax = axes[1][col]
    ax.set_facecolor(SURFACE)
    xs, mus, cis, pays = [], [], [], []
    for stp in d['selection_steps']:
        lo = np.asarray(d['all_lls'][int(stp['lag_from'])])
        hi = np.asarray(d['all_lls'][int(stp['lag_to'])])
        dif = hi - lo
        ok = np.isfinite(dif)
        nf = int(ok.sum())
        mu = dif[ok].mean()
        se = dif[ok].std(ddof=1) / np.sqrt(nf) if nf > 1 else np.nan
        xs.append(stp['lag_to']); mus.append(mu); pays.append(stp['pays'])
        cis.append(_t.ppf(0.975, nf - 1) * se if nf > 1 else np.nan)
    xs, mus, cis = np.array(xs, float), np.array(mus), np.array(cis)
    pays = np.array(pays, bool)
    ax.axhline(0, color=INK, lw=0.9, zorder=1)
    ax.errorbar(xs, mus, yerr=cis, fmt='none', ecolor=INK_2, elinewidth=1.1,
                capsize=3, zorder=2)
    ax.scatter(xs[pays], mus[pays], s=44, color=RIBBON, zorder=3, label='adopted')
    ax.scatter(xs[~pays], mus[~pays], s=52, facecolor=SURFACE, edgecolor=ACC, lw=1.6,
               zorder=3, label='rejected')
    ax.set_xlabel('lag being tested', fontsize=8.5, color=INK_2)
    if col == 0:
        ax.set_ylabel('paired gain vs adopted lag\n(95% CI)', fontsize=8.5, color=INK_2)
        ax.legend(frameon=False, fontsize=7.5, loc='upper left', labelcolor=INK_2)

    for ax in (axes[0][col], axes[1][col]):
        ax.set_xscale('log', base=2)
        ax.set_xticks(lags); ax.set_xticklabels([str(l) for l in lags], fontsize=7)
        for sd in ('top', 'right'):
            ax.spines[sd].set_visible(False)
        for sd in ('left', 'bottom'):
            ax.spines[sd].set_color(INK_MUTED); ax.spines[sd].set_linewidth(0.6)
        ax.tick_params(colors=INK_2, labelsize=7.5, width=0.6)
        ax.grid(color=INK_MUTED, lw=0.4, alpha=0.25, zorder=0)

fig.suptitle('Lag selection — top: absolute LL (context) · bottom: the paired comparison the rule tests',
             fontsize=11.5, color=INK, x=0.006, ha='left', y=0.985)
fig.text(0.006, 0.955, 'orange ring / hollow = rejected · dashed line = cap · '
         'a point is adopted when its 95% CI excludes zero',
         fontsize=8, color=INK_MUTED, ha='left', va='top')
fig.subplots_adjust(top=0.86, bottom=0.09, left=0.075, right=0.99, wspace=0.38, hspace=0.42)
fig.savefig(f'{OUT}_profiles.png', dpi=160, facecolor=SURFACE)
print(f'wrote {OUT}_profiles.png')

# ---------------- the selection record ----------------
print('\nselection steps per session:')
for S in sessions:
    print(f"  {S['mouse']} {S['eid'][:8]}  (cap {S['d']['cap']}, adopted {S['d']['best_lag']})")
    for stp in S['d']['selection_steps']:
        print(f"     {stp['lag_from']:>3} -> {stp['lag_to']:<3}  gain {stp['gain']:+.6f}"
              f"  se {stp['se']:.6f}  {'ADOPT' if stp['pays'] else 'reject'}")
