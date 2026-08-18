"""Is the segmentation insensitive to kappa?

Panel A: how much kappa moves the median syllable duration (fold change vs kappa=0)
Panel B: frame-wise agreement against the kappa=0 segmentation
Panel C: kappa-sensitivity tracks how many state transitions the session contains

Writes two versions:
  kappa_insensitivity_60Hz.png  -- whisker + 60 Hz lick (the paper cohort)
  kappa_insensitivity_all.png   -- adds the 30 Hz training lick sessions

Run from this directory.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK_2, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#87867f'
C = {'whisker': '#2a78d6', 'lick 60 Hz': '#eb6834', 'lick 30 Hz': '#1baf7a'}   # slots 1-3
ACC = '#e34948'

a = pd.read_csv('kappa_insensitivity.csv')
b = pd.read_csv('kappa_lick60.csv')
a['set'] = np.where(a.modality == 'whisker', 'whisker', 'lick 30 Hz')
b['set'] = 'lick 60 Hz'
dall = pd.concat([a, b], ignore_index=True)
dall['agree_vs_k0'] = pd.to_numeric(dall.agree_vs_k0, errors='coerce')
base = dall[dall.kappa == 0].set_index('eid')
dall['dur_fold'] = dall.med_dwell_ms / dall.eid.map(base.med_dwell_ms)
dall['nseg0'] = dall.eid.map(base.n_seg)
dall['collapsed'] = dall.n_seg <= 1


def make(d, out, subtitle):
    order = [k for k in C if k in set(d['set'])]
    has_collapse = bool(d.collapsed.any())
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
        ax.grid(color=INK_MUTED, lw=0.4, alpha=0.25, zorder=0)

    # A -- duration fold change
    ax = axes[0]
    for st in order:
        for e, s in d[d['set'] == st].groupby('eid'):
            s = s.sort_values('kappa')
            ax.plot([max(k, 30) for k in s.kappa], s.dur_fold, color=C[st], lw=1.2,
                    alpha=0.65, zorder=2)
    ax.axhline(1, color=INK_MUTED, lw=0.8, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    lo = max(0.45, float(d.dur_fold.min()) * 0.8) if not has_collapse else 0.45
    hi = min(20., float(d.dur_fold.max()) * 1.3) if not has_collapse else 4.0
    ax.set_ylim(lo, hi)
    ax.set_yticks([0.5, 1, 2, 5, 10]); ax.set_yticklabels(['0.5×', '1×', '2×', '5×', '10×'])
    ax.set_xlabel('κ  (0 plotted at the left edge)', fontsize=9, color=INK_2)
    ax.set_ylabel('median duration ÷ duration at κ=0', fontsize=9, color=INK_2)
    ax.set_title('A · how much κ moves duration', fontsize=10.5, color=INK, loc='left', pad=22)
    ax.annotate('one line per session' + (' · collapsed fits run off-scale' if has_collapse else ''),
                xy=(0, 1.015), xycoords='axes fraction', fontsize=8, color=INK_MUTED, ha='left')

    # B -- agreement
    ax = axes[1]
    for st in order:
        for e, s in d[d['set'] == st].groupby('eid'):
            s = s[s.kappa > 0].sort_values('kappa')
            ax.plot(s.kappa, s.agree_vs_k0, color=C[st], lw=1.2, alpha=0.65, zorder=2)
    ax.axhline(0.95, color=ACC, lw=1, ls=(0, (4, 3)), zorder=1)
    ymin = min(0.9, float(d.agree_vs_k0.min()) - 0.03)
    ax.set_xscale('log'); ax.set_ylim(ymin, 1.006)
    ax.annotate('95% of frames unchanged', xy=(ax.get_xlim()[0] * 1.15, 0.952),
                fontsize=8, color=ACC, va='bottom')
    ax.set_xlabel('κ', fontsize=9, color=INK_2)
    ax.set_ylabel('frames agreeing with the κ=0 segmentation', fontsize=9, color=INK_2)
    ax.set_title('B · does the segmentation change?', fontsize=10.5, color=INK, loc='left', pad=22)
    ax.annotate('flat near 1 = κ is irrelevant to the state labels',
                xy=(0, 1.015), xycoords='axes fraction', fontsize=8, color=INK_MUTED, ha='left')
    ax.legend(handles=[Line2D([], [], color=C[k], lw=2, label=k) for k in order],
              frameon=False, fontsize=8, loc='lower left', labelcolor=INK_2)

    # C -- sensitivity vs number of transitions
    ax = axes[2]
    worst = (d[d.kappa > 0].groupby(['set', 'eid'])
             .agg(agree=('agree_vs_k0', 'min'), nseg0=('nseg0', 'first'),
                  collapsed=('collapsed', 'any')).reset_index())
    for st in order:
        g = worst[worst['set'] == st]
        ok, bad = g[~g.collapsed], g[g.collapsed]
        ax.scatter(ok.nseg0.clip(lower=1), ok.agree, s=46, color=C[st], lw=0, alpha=0.85, zorder=3)
        ax.scatter(bad.nseg0.clip(lower=1), bad.agree, s=64, facecolor='none',
                   edgecolor=C[st], lw=1.6, zorder=4)
    ax.axhline(0.95, color=ACC, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xscale('log'); ax.set_ylim(ymin, 1.02)
    ax.set_xlabel('segments at κ=0  (≈ transitions the data pins down)', fontsize=9, color=INK_2)
    ax.set_ylabel('worst agreement across the κ grid', fontsize=9, color=INK_2)
    ax.set_title('C · κ only bites when events are scarce', fontsize=10.5, color=INK,
                 loc='left', pad=22)
    ax.annotate('hollow = collapsed to one state at some κ' if has_collapse
                else 'no fit collapsed anywhere in the grid',
                xy=(0, 1.015), xycoords='axes fraction', fontsize=8, color=INK_MUTED, ha='left')

    fig.text(0.005, 0.055, subtitle, fontsize=7.8, color=INK_MUTED)
    fig.text(0.005, 0.015, 'Held-out bits is deliberately not plotted: the CV baseline is itself '
             'κ-dependent — initialize(method="prior") samples from a prior containing κ — so bits '
             'is not comparable across κ.', fontsize=7.8, color=INK_MUTED)
    fig.subplots_adjust(top=0.83, bottom=0.23, left=0.055, right=0.99, wspace=0.30)
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out}')
    return worst


sub = dall[dall['set'] != 'lick 30 Hz']
w60 = make(sub, 'kappa_insensitivity_60Hz.png',
           'Whisker AR-HMM (lag 10) and lick Poisson-HMM, 60 Hz two-camera cohort. '
           'κ grids scaled per modality to span "nothing" to roughly double the dwell.')
wall = make(dall, 'kappa_insensitivity_all.png',
            'As left, plus the 30 Hz training lick sessions, where fits are unstable and several '
            'collapse to a single state.')

print('\n=== 60 Hz cohort only ===')
print(w60.groupby('set').agree.describe()[['count', 'min', '50%', 'max']].to_string(float_format=lambda x: f'{x:.4f}'))
print('\nsessions below 0.95 agreement:')
w = w60[w60.agree < 0.95]
print(w[['set', 'eid', 'nseg0', 'agree']].to_string(index=False) if len(w) else '  none')
print('\nmax duration fold-change per set (60 Hz cohort):')
print(sub.groupby('set').dur_fold.max().to_string(float_format=lambda x: f'{x:.2f}'))
