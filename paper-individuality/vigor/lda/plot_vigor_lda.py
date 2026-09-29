"""FIGURE -- what LD1 tracks: wheel vigor, not paw vigor, and the task gate most of all."""
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import paper_style as ps                                            # noqa: E402
import vigor_vs_lda as V                                            # noqa: E402

ps.use('paper')
os.environ['EMBEDDING'] = 'mouse_LDA_5_bins_raw_shrink0.5_360_28-09-2026'


def main(save=False):
    import pandas as pd
    S, _ = V.load()
    num = S.select_dtypes(include=[np.number]).columns
    M = S.groupby('mouse_name')[list(num)].mean()
    M['gate'] = np.log(M['wheel_speed_Pre-quiescence'] / M['wheel_speed_Quiescence'])

    panels = [('wheel_mean_speed_moving', 'Wheel speed while moving\n(whole session)'),
              ('paw_mean_speed_moving', 'Paw speed while moving\n(whole session)'),
              ('wheel_speed_Pre-quiescence', 'Wheel speed,\npre-quiescence epoch'),
              ('gate', 'Wheel gating,\nlog(pre-q / quiescence)')]
    fig, axes = plt.subplots(1, 4, figsize=(ps.SIZES[ps.MODE]['double'][0] * 1.1,
                                            ps.SIZES[ps.MODE]['double'][1] * 0.8))
    colors = ps.ld1_colors(M['LD1'].values)
    for ax, (col, lab) in zip(axes, panels):
        r, p = spearmanr(M[col], M['LD1'], nan_policy='omit')
        ax.scatter(M[col], M['LD1'], c=colors, s=14, linewidths=0, alpha=0.85)
        ok = np.isfinite(M[col]) & np.isfinite(M['LD1'])
        b = np.polyfit(M[col][ok], M['LD1'][ok], 1)
        xs = np.linspace(M[col][ok].min(), M[col][ok].max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=ps.NEUTRAL, lw=1)
        ax.set_xlabel(lab)
        ax.set_title(rf'$\rho$ = {r:+.2f}, p = {p:.3f}', fontsize=plt.rcParams['font.size'])
    axes[0].set_ylabel(f'LD1 (mouse mean, n = {len(M)})')
    fig.tight_layout()
    ps.savefig(fig, 'lda1_vs_vigor', svg=True)
    plt.show()
    return fig


if __name__ == '__main__':
    ps.saving('--save' in sys.argv)
    main()
