"""
How crisp is the GLM-HMM state along LD1?
=========================================
For every trial, the posterior of its most likely state, max_k p(z_t = k | all choices of the
session). The figure shows the fraction of trials at or above each value of that max posterior,
one curve per LD1 bin. Each mouse gets its own curve, and the bin's curve is the mean of its
mice +- SEM across mice, so mice with many trials don't dominate. A curve that stays high means
the state is unambiguous on most trials; one that drops early means many trials sit between
states.

The right panel shows, per mouse, the fraction of trials with max posterior > CRISP vs the
mouse's mean LD1 (Spearman).

LD1 BINS are per MOUSE (mean LD1 over its sessions in the LDA file, then quantiles), as in
transition_prediction/tp_models.ld1_bins, so no mouse spans two bins. Each mouse contributes
all of its GLM-HMM trials, because the fits are per mouse with sessions pooled, so the posterior
of every session comes from the same model. The sign of LD1 is arbitrary between LDA files, so
the script prints each bin's state switch rate (argmax flips per trial) to say which end is
the switchy one.

SOURCES (transition_prediction/tp_data.py loaders)
  --source k2_original   merged_behavioral_and_states.pqt, the K=2 fit of load_states.ipynb
  --source all_k --K 3   the K=2/3/4 refits in k2_k3_pilot/all_k_posteriors.parquet (mice with
                         >= 3 sessions); max is taken over all K states

Usage (iblenv):
    python max_posterior_by_ld1.py
    python max_posterior_by_ld1.py --n-bins 5
    python max_posterior_by_ld1.py --source all_k --K 3
Outputs in GLM-HMM/max_posterior_by_ld1/ (figure + per-mouse table).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'transition_prediction'))
sys.path.insert(0, str(HERE.parent))
import tp_data  # noqa: E402
import paper_style as ps  # noqa: E402

OUT_DIR = HERE / 'max_posterior_by_ld1'
CRISP = 0.8          # threshold for the per-mouse summary in the right panel


def trial_max_posterior(source, K):
    """One row per trial: eid, mouse_name, trial_id, p_max, state (argmax)."""
    if source == 'k2_original':
        s = tp_data.load_states('k2_original')
        s['p_max'] = np.maximum(s['p_engaged'], 1 - s['p_engaged'])
        return s[['eid', 'mouse_name', 'trial_id', 'p_max', 'state']]
    post = pd.read_parquet(tp_data.ALLK_POSTERIORS, filters=[('K', '==', K)])
    wide = post.pivot_table(index=['mouse_name', 'eid', 'trial_idx'], columns='state',
                            values='posterior').reset_index()
    P = wide[list(range(K))].to_numpy()
    wide['p_max'] = P.max(axis=1)
    wide['state'] = P.argmax(axis=1)
    return wide.rename(columns={'trial_idx': 'trial_id'})[['eid', 'mouse_name', 'trial_id', 'p_max', 'state']]


def mouse_bins(lda, n_bins):
    """Mean LD1 per mouse and its quantile bin (0 = lowest LD1)."""
    m = lda.groupby('mouse_name')['lda_1'].mean().rename('ld1').to_frame()
    m['bin'] = pd.qcut(m['ld1'], n_bins, labels=False)
    return m


def survival(p, grid):
    """Fraction of trials with p_max >= each grid value."""
    p = np.sort(p)
    return 1 - np.searchsorted(p, grid, side='left') / len(p)


def bin_colors(n_bins):
    """Positions on the LD1 colormap (blue low, red high). A bin that lands on the white
    centre is drawn grey so it stays visible."""
    pos = np.linspace(0.05, 0.95, n_bins)
    return ['#8C8C8C' if abs(x - 0.5) < 0.15 else ps.LD1_CMAP(x) for x in pos]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='k2_original', choices=['k2_original', 'all_k'])
    ap.add_argument('--K', type=int, default=2, help='number of states (all_k only)')
    ap.add_argument('--n-bins', type=int, default=3)
    ap.add_argument('--lda-file', type=Path, default=tp_data.LDA_FILE)
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args(argv)
    K = 2 if args.source == 'k2_original' else args.K

    ps.use('paper')
    trials = trial_max_posterior(args.source, K)
    lda = tp_data.load_lda(args.lda_file)
    dropped = set(lda['mouse_name']) - set(trials['mouse_name'])
    lda = lda[~lda['mouse_name'].isin(dropped)]          # bins over the mice that have a fit
    mice = mouse_bins(lda, args.n_bins)
    trials = trials[trials['mouse_name'].isin(mice.index)].sort_values(['eid', 'trial_id'])

    # per-mouse summaries
    flips = (trials.groupby('eid')['state'].diff().fillna(0) != 0)
    trials['flip'] = flips & (trials.groupby('eid').cumcount() > 0)
    per = trials.groupby('mouse_name').agg(n_trials=('p_max', 'size'), n_sessions=('eid', 'nunique'),
                                           median_p_max=('p_max', 'median'),
                                           frac_crisp=('p_max', lambda x: (x > CRISP).mean()),
                                           switch_rate=('flip', 'mean'))
    per = mice.join(per, how='inner')

    grid = np.linspace(1 / K, 1, 201)
    curves = {m: survival(g['p_max'].to_numpy(), grid) for m, g in trials.groupby('mouse_name')}

    colors = bin_colors(args.n_bins)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), width_ratios=[1.3, 1])
    ax = axes[0]
    for b in range(args.n_bins):
        members = per.index[per['bin'] == b]
        C = np.vstack([curves[m] for m in members])
        mean, sem = C.mean(0), C.std(0, ddof=1) / np.sqrt(len(C))
        lo, hi = per.loc[members, 'ld1'].min(), per.loc[members, 'ld1'].max()
        ax.fill_between(grid, mean - sem, mean + sem, color=colors[b], alpha=0.25, lw=0)
        ax.plot(grid, mean, color=colors[b], lw=1.5,
                label=f'LD1 bin {b + 1} [{lo:.2f}, {hi:.2f}], {len(members)} mice')
    ax.set(xlabel='max posterior on a trial', ylabel='fraction of trials ≥ x', xlim=(1 / K, 1), ylim=(0, 1.02))
    ax.axvline(CRISP, color='0.85', lw=0.8, ls='--', zorder=0)
    ax.legend(fontsize=6, frameon=False, loc='lower left')
    ax.set_title(f'GLM-HMM K={K} ({args.source})', fontsize=7)

    ax = axes[1]
    ax.scatter(per['ld1'], per['frac_crisp'], c=ps.ld1_colors(per['ld1']), s=18,
               edgecolor='0.3', linewidth=0.4)
    r = spearmanr(per['ld1'], per['frac_crisp'])
    ax.set(xlabel='mouse mean LD1', ylabel=f'fraction of trials with max posterior > {CRISP}')
    ax.set_title(f'Spearman ρ = {r.statistic:.2f}, p = {r.pvalue:.3f}, n = {len(per)} mice', fontsize=7)
    plt.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    stem = f'max_posterior_by_ld1_{args.source}_K{K}_{args.n_bins}bins_{args.lda_file.name}'
    fig.savefig(OUT_DIR / f'{stem}.png', dpi=300, bbox_inches='tight')
    per.to_csv(OUT_DIR / f'{stem}_per_mouse.csv')

    print(f'{len(per)} mice, {per["n_trials"].sum():,} trials; LDA mice without GLM-HMM trials: {sorted(dropped)}')
    summary = per.groupby('bin').agg(mice=('ld1', 'size'), ld1=('ld1', 'mean'),
                                     median_p_max=('median_p_max', 'mean'),
                                     frac_crisp=('frac_crisp', 'mean'), switch_rate=('switch_rate', 'mean'))
    print(summary.round(3).to_string())
    rs = spearmanr(per['ld1'], per['switch_rate'])
    print(f'switch rate vs LD1: rho = {rs.statistic:.2f} (p = {rs.pvalue:.3f}) -> '
          f'the switchy end is {"HIGH" if rs.statistic > 0 else "LOW"} LD1 in this file')
    print(f'frac max posterior > {CRISP} vs LD1: rho = {r.statistic:.2f} (p = {r.pvalue:.3f})')
    print('wrote', OUT_DIR / f'{stem}.png')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
