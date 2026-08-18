"""Two kappa figures: the MoSeq-style scan curve, and a snippet comparison at kappa 0 vs high.

Env: NPZ (example dump from kappa_scan.py).  Run from this directory.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
BAND, RIBBON, ACC = '#cde2fb', '#2a78d6', '#eb6834'
FPS = 60.0
TARGET, TLO, THI = 450., 379., 550.      # model-free changepoint duration, median + IQR

d = pd.read_csv('kappa_scan.csv')
d = d[pd.to_numeric(d.bits_LL, errors='coerce').notna()].copy()
for c in ['kappa', 'bits_LL', 'med_dwell_ms', 'n_seg']:
    d[c] = pd.to_numeric(d[c])

# ---------- figure 1: the kappa scan ----------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
fig.patch.set_facecolor(SURFACE)
for a in (ax, ax2):
    a.set_facecolor(SURFACE)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK_MUTED); a.spines[s].set_linewidth(0.6)
    a.tick_params(colors=INK_2, labelsize=8, width=0.6)
    a.grid(color=INK_MUTED, lw=0.4, alpha=0.25, zorder=0)

x = sorted(d.kappa.unique())
xs = [max(k, 300) for k in x]                       # 0 plotted at the left edge of a log axis
ax.axhspan(TLO, THI, color=ACC, alpha=0.10, lw=0, zorder=0)
ax.axhline(TARGET, color=ACC, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.annotate('model-free changepoint duration\n450 ms (IQR 379–550)', xy=(xs[0], THI + 12),
            fontsize=8, color=ACC, va='bottom', ha='left')
for m, g in d.groupby('mouse'):
    g = g.sort_values('kappa')
    ax.plot([max(k, 300) for k in g.kappa], g.med_dwell_ms, color=RIBBON, lw=1, alpha=0.45, zorder=2)
med = d.groupby('kappa').med_dwell_ms.median().reindex(x)
ax.plot(xs, med.values, color=INK, lw=2, zorder=4)
ax.scatter(xs, med.values, s=30, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.5)
ax.set_xscale('log')
ax.set_xticks(xs); ax.set_xticklabels(['0', '10³', '10⁴', '5×10⁴', '10⁵', '2×10⁵'])
ax.set_xlabel('κ  (transition_matrix_stickiness)', fontsize=9, color=INK_2)
ax.set_ylabel('median syllable duration (ms)', fontsize=9, color=INK_2)
ax.set_title('κ scan: duration vs stickiness', fontsize=11, color=INK, loc='left', pad=24)
ax.annotate('6 typical sessions, lag 10 · thin lines = sessions, black = median',
            xy=(0, 1.015), xycoords='axes fraction', fontsize=8, color=INK_MUTED, ha='left')

b = d.groupby('kappa').bits_LL.median().reindex(x)
ax2.plot(xs, b.values, color=INK, lw=2, zorder=4)
ax2.scatter(xs, b.values, s=30, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.5)
ax2.set_xscale('log')
ax2.set_xticks(xs); ax2.set_xticklabels(['0', '10³', '10⁴', '5×10⁴', '10⁵', '2×10⁵'])
ax2.set_xlabel('κ', fontsize=9, color=INK_2)
ax2.set_ylabel('held-out bits / frame', fontsize=9, color=INK_2)
ax2.set_title('what it costs', fontsize=11, color=INK, loc='left', pad=24)
ax2.annotate(f'−{(b.iloc[0]-b.iloc[-1])*1000:.0f} millibits from κ=0 to κ=2×10⁵',
             xy=(0, 1.015), xycoords='axes fraction', fontsize=8, color=INK_MUTED, ha='left')

fig.text(0.006, 0.015,
         'κ buys duration monotonically and pays for it monotonically in likelihood. '
         'At the corrected lag the κ=0 duration sits just below the target band.',
         fontsize=8, color=INK_MUTED)
fig.subplots_adjust(top=0.84, bottom=0.19, left=0.075, right=0.985, wspace=0.28)
fig.savefig('kappa_scan.png', dpi=165, facecolor=SURFACE)
print('wrote kappa_scan.png')

# ---------- figure 2: snippet at kappa 0 vs high ----------
NPZ = os.environ.get('NPZ', '')
if NPZ and os.path.exists(NPZ):
    z = np.load(NPZ, allow_pickle=True)
    sig = z['signal']
    ks = sorted([k for k in z.files if k.startswith('states_k')],
                key=lambda s: float(s.split('k')[1]))
    mouse, eid = str(z['mouse']), str(z['eid'])
    WIN = int(15 * FPS)

    def runs_of(m):
        e = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int), [0]))))
        return list(zip(e[::2], e[1::2]))

    ref = z[ks[0]]
    best, bs = 0, -1e9
    for c in range(0, len(ref) - WIN, int(FPS * 2)):
        seg = ref[c:c + WIN]
        sc = -abs(float(seg.mean()) - 0.45) * 12 - abs(int(np.sum(np.diff(seg) != 0)) - 8) * 0.4
        if sc > bs: bs, best = sc, c
    st = best
    t = np.arange(WIN) / FPS
    s_sig = sig[st:st + WIN]

    H = 2.55 * len(ks) + 1.9
    fig, axes = plt.subplots(len(ks), 1, figsize=(11, H), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for a, key in zip(np.atleast_1d(axes), ks):
        kv = float(key.split('k')[1])
        s = z[key][st:st + WIN]
        d_ = np.diff(np.concatenate(([0], np.where(np.diff(z[key]) != 0)[0] + 1, [len(z[key])])))
        lo, hi = s_sig.min() - 0.35, s_sig.max() + 0.5
        a.set_facecolor(SURFACE)
        for p, q in runs_of(s == 1):
            a.axvspan(t[p], t[min(q, WIN - 1)], color=BAND, lw=0, zorder=0)
            a.add_patch(Rectangle((t[p], lo + 0.02), t[min(q, WIN - 1)] - t[p], 0.16,
                                  color=RIBBON, lw=0, zorder=3))
        a.plot(t, s_sig, color=INK, lw=0.9, zorder=2)
        a.set_ylim(lo, hi); a.set_xlim(0, t[-1])
        a.set_title(f"κ = {kv:,.0f}   ·   median dwell {np.median(d_):.0f} frames "
                    f"({np.median(d_)/FPS*1000:.0f} ms)   ·   {len(d_):,} segments   ·   "
                    f"held-out {float(z['bits_'+key.split('states_')[1]]):.4f} bits",
                    fontsize=9.5, color=INK, loc='left', pad=6)
        a.set_ylabel('whisker ME (z)', fontsize=8.5, color=INK_2)
        for sd in ('top', 'right'): a.spines[sd].set_visible(False)
        for sd in ('left', 'bottom'): a.spines[sd].set_color(INK_MUTED); a.spines[sd].set_linewidth(0.6)
        a.tick_params(colors=INK_2, labelsize=8, width=0.6)
    np.atleast_1d(axes)[-1].set_xlabel('time (s)', fontsize=8.5, color=INK_2)
    fig.suptitle(f'What stickiness costs — {mouse} {eid[:8]}, lag 10', fontsize=12.5,
                 color=INK, x=0.006, ha='left', y=1 - 0.28 / H)
    fig.text(0.006, 1 - 0.62 / H,
             'shaded band and ribbon = high-whisking state; unshaded = low-whisking state',
             fontsize=8.5, color=RIBBON, ha='left', va='top')
    fig.text(0.006, 0.012,
             'A large κ merges neighbouring bouts and swallows short quiet gaps — longer syllables, '
             'but not better ones.', fontsize=8, color=INK_MUTED, ha='left')
    fig.subplots_adjust(top=1 - 1.15 / H, bottom=1.0 / H, left=0.075, right=0.985, hspace=0.45)
    fig.savefig('kappa_snippet.png', dpi=165, facecolor=SURFACE)
    print('wrote kappa_snippet.png')
else:
    print('no NPZ dump found - skipped the snippet figure')
