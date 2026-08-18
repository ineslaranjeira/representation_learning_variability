"""Why not just raise the cap until nothing sits at it?

Three panels from lag_ceiling_effect.csv (24 sessions decoded at every lag in their grid,
using the stored fit params -- no refitting):

  A  held-out likelihood keeps improving, but saturates
  B  the SEGMENTATION converges: agreement stops changing past lag ~16
  C  syllable duration drifts monotonically away from the model-free changepoint anchor

Run from this directory.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
S1, S2 = '#2a78d6', '#eb6834'
CHANGEPOINT, CP_LO, CP_HI = 450., 379., 550.     # model-free changepoint duration + IQR

r = pd.read_csv('lag_ceiling_effect.csv')
lags = sorted(r.lag.unique())
g = r.groupby('lag')

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
fig.patch.set_facecolor(SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    ax.grid(color=INK_MUTED, lw=0.4, alpha=0.25); ax.set_axisbelow(True)
    ax.set_xscale('log', base=2); ax.set_xticks(lags)
    ax.set_xticklabels([str(l) for l in lags], fontsize=7.5)
    ax.set_xlabel('AR lag (frames)', fontsize=8.5, color=INK_2)

# A -- likelihood
ax = axes[0]
for e, s in r.groupby('eid'):
    s = s.sort_values('lag')
    ax.plot(s.lag, s.raw_ll, color=INK_MUTED, lw=0.8, alpha=0.35, zorder=2)
m = g.raw_ll.median()
ax.plot(m.index, m.values, color=S1, lw=2.2, zorder=4)
ax.set_ylabel('held-out LL / frame', fontsize=8.5, color=INK_2)
ax.set_title('A · the likelihood does keep improving', fontsize=10, color=INK, loc='left', pad=16)
ax.annotate('but the gain from 128→256 is 0.004, vs 0.035 for 1→8',
            xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

# B -- segmentation convergence
ax = axes[1]
for e, s in r.groupby('eid'):
    s = s.sort_values('lag')
    ax.plot(s.lag, s.agree_vs_shortest, color=INK_MUTED, lw=0.8, alpha=0.35, zorder=2)
m = g.agree_vs_shortest.median()
ax.plot(m.index, m.values, color=S1, lw=2.2, zorder=4)
ax.axhspan(0.955, 0.97, color=S2, alpha=0.10, lw=0, zorder=1)
ax.annotate('plateau: 0.960–0.964 from lag 16 to 256', xy=(0.5, 0.16),
            xycoords='axes fraction', ha='center', fontsize=8, color=S2)
ax.set_ylabel('frame agreement with the lag-1 segmentation', fontsize=8.5, color=INK_2)
ax.set_title('B · the segmentation stops changing', fontsize=10, color=INK, loc='left', pad=16)
ax.annotate('this is the panel that answers "why not raise the cap?"',
            xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

# C -- duration vs the external anchor
ax = axes[2]
ax.axhspan(CP_LO, CP_HI, color=S2, alpha=0.12, lw=0, zorder=1)
ax.axhline(CHANGEPOINT, color=S2, lw=1.2, ls=(0, (4, 3)), zorder=3)
ax.annotate('model-free changepoint duration\n450 ms (IQR 379–550)', xy=(0.97, CHANGEPOINT),
            xycoords=('axes fraction', 'data'), xytext=(0, 8), textcoords='offset points',
            ha='right', fontsize=7.5, color=S2)
for e, s in r.groupby('eid'):
    s = s.sort_values('lag')
    ax.plot(s.lag, s.med_dwell_ms, color=INK_MUTED, lw=0.8, alpha=0.35, zorder=2)
m = g.med_dwell_ms.median()
ax.plot(m.index, m.values, color=S1, lw=2.2, zorder=4)
ax.set_yscale('log')
ax.set_ylabel('median syllable duration (ms)', fontsize=8.5, color=INK_2)
ax.set_title('C · but duration drifts away from the anchor', fontsize=10, color=INK,
             loc='left', pad=16)
ax.annotate('median 433 ms at lag 1 → 267 ms at lag 256',
            xy=(0, 1.015), xycoords='axes fraction', fontsize=7.5, color=INK_MUTED)

fig.text(0.005, 0.015, '24 sessions whose grid reaches 128 (8 reach 256), decoded at every '
         'fitted lag with the stored parameters — no refitting. Thin grey = one session; '
         'blue = median.', fontsize=7.8, color=INK_MUTED)
fig.subplots_adjust(top=0.84, bottom=0.17, left=0.055, right=0.99, wspace=0.28)
fig.savefig('lag_ceiling_effect.png', dpi=155, facecolor=SURFACE)
print('wrote lag_ceiling_effect.png')
