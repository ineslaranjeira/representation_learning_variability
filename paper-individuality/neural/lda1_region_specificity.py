"""
CAN LD1'S NEURAL CORRELATES BE TESTED PER BRAIN REGION, AND HOW HETEROGENEOUS ARE THEY?
=======================================================================================
fig_neural's tests are global: every neuron (or, for noise correlations, every session x region pair), region as a
covariate, LD1 permuted across sessions (lda1_neural_metrics.age_effect_test, the age paper's test -- Zang et al. 2026,
Nat Commun 17:8156 -- with LD1 in place of age). As in that paper, regions are looked at only for the measures with a
global effect: Fano factor post-stimulus and change, and the three noise-correlation tests.

The age paper's 16 regions were chosen for its own sampling; in the LD1 cohort several have 0-5 sessions or almost no
LD1 spread (ACB: 3 sessions, LD1 -1.9..-1.1). So regions here are DATA-DRIVEN Beryl areas, admitted by a rule fixed in
advance, per measure:

    >= MIN_SESSIONS sessions, >= MIN_MICE mice, and SD of LD1 across its sessions >= MIN_SPREAD x the SD across all
    sessions of that measure (a region recorded only in similar mice cannot show an LD1 slope)

For each measure:
  region-wise     age_effect_test within each admitted region: partial r, permutation p (LD1 shuffled across
                  sessions), BH q across the admitted regions of that measure
  heterogeneity   one model on the admitted regions' rows: y ~ region + covariates + LD1 + LD1 x region. Statistic =
                  the F of the LD1 x region block given the LD1 main effect (do the regions' LD1 slopes differ?);
                  p from the same session permutation of LD1. Reported with the number of regions it rests on --
                  with only a few regions it says little either way.
  leave-one-out   the global test (all rows) re-run without each admitted region: does any single region carry it?

Writes results/lda1_region_specificity_<date>.csv (one row per measure x region / test) and, with --figure, Swanson
flatmaps of the region-wise r (iblatlas.plots.plot_swanson_vector, offline) to figures/ and figure_generation/ciss_figures/.
"""
import re
import sys
import pathlib
import warnings
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
for p in [str(PI), str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)
import lda1_cortex_subcortex as cs          # load(): same tables, cohort and LD1 as fig_neural (and its numpy shim)
nm = cs.nm

MEASURES = [('Fano factor', 'Post-stimulus'), ('Fano factor', 'Response magnitude'),
            ('Noise correlation', 'Baseline'), ('Noise correlation', 'Post-stimulus'),
            ('Noise correlation', 'Response magnitude')]
MIN_SESSIONS, MIN_MICE, MIN_SPREAD = 10, 8, 0.5
N_PERM = 2000
SEED = 0
NOT_A_REGION = {'root', 'void', 'fiber tracts', 'VS'}
LDA_FILE = PI / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
DATE = date.today().strftime('%d-%m-%Y')


def mouse_of_session():
    lda = pd.read_pickle(LDA_FILE)
    return lda.drop_duplicates('session').set_index('session')['mouse_name']


def admitted(df, region_col, mouse):
    """Regions passing the inclusion rule, with their coverage."""
    d = df[~df[region_col].isin(NOT_A_REGION)]
    sd_all = d.groupby('session')['lda_1'].first().std()
    rows = []
    for reg, g in d.groupby(region_col):
        s = g.groupby('session')['lda_1'].first()
        rows.append(dict(region=reg, sessions=len(s), mice=s.index.map(mouse).nunique(), rows=len(g),
                         ld1_sd_ratio=s.std() / sd_all if len(s) > 1 else 0.0))
    cov = pd.DataFrame(rows)
    cov['admitted'] = (cov['sessions'] >= MIN_SESSIONS) & (cov['mice'] >= MIN_MICE) & (cov['ld1_sd_ratio'] >= MIN_SPREAD)
    return cov.sort_values('sessions', ascending=False)


def heterogeneity_test(df, y_col, region_col, covars, rng):
    """F of the LD1 x region block given LD1, in y ~ region + covars + LD1 + LD1 x region; LD1 permuted across sessions."""
    d = df.dropna(subset=[y_col, 'lda_1', region_col] + list(covars)).reset_index(drop=True)
    y = d[y_col].to_numpy(float)
    R = pd.get_dummies(d[region_col]).to_numpy(float)                        # one column per region
    Z = np.column_stack([R, d[list(covars)].to_numpy(float)])                 # region intercepts + covariates
    Zp = np.linalg.pinv(Z)
    yz = y - Z @ (Zp @ y)
    sess = d.groupby('session')['lda_1'].first()
    row = d['session'].map({s: i for i, s in enumerate(sess.index)}).to_numpy()
    k = R.shape[1]

    def F(xs):
        x = xs[row]
        X1 = x[:, None]                                                      # common LD1 slope
        X2 = x[:, None] * R[:, 1:]                                           # region-specific deviations
        X1z = X1 - Z @ (Zp @ X1)
        X2z = X2 - Z @ (Zp @ X2)
        b1 = np.linalg.lstsq(X1z, yz, rcond=None)[0]
        rss1 = np.sum((yz - X1z @ b1) ** 2)
        Xf = np.column_stack([X1z, X2z])
        bf = np.linalg.lstsq(Xf, yz, rcond=None)[0]
        rss2 = np.sum((yz - Xf @ bf) ** 2)
        return ((rss1 - rss2) / (k - 1)) / (rss2 / (len(y) - Z.shape[1] - k))

    obs = F(sess.to_numpy())
    null = np.array([F(rng.permutation(sess.to_numpy())) for _ in range(N_PERM)])
    return dict(F=obs, p=(1 + np.sum(null >= obs)) / (1 + N_PERM), n_regions=k, sessions=len(sess), rows=len(d))


def main():
    warnings.filterwarnings('ignore')
    T, _ = cs.load()
    mouse = mouse_of_session()
    rng = np.random.default_rng(SEED)
    out, coverage = [], []
    for metric, test in MEASURES:
        tbl, y_col, _, _, region_col, covars = nm.GRID[(metric, test)]
        df = T[tbl]
        cov = admitted(df, region_col, mouse).assign(metric=metric, test=test)
        coverage.append(cov)
        regs = cov.loc[cov['admitted'], 'region'].tolist()
        print(f'\n{metric} / {test}: {len(regs)} regions admitted: {", ".join(regs)}')
        g = nm.age_effect_test(df, y_col, predictor='lda_1', region_col=region_col, extra_covars=covars,
                               n_perm=N_PERM, seed=SEED)
        out.append(dict(metric=metric, test=test, kind='global', region='all', r=g['r_partial'], p=g['p_perm'],
                        sessions=g['n_sessions'], rows=g['n']))
        for reg in regs:                                                      # region-wise
            r = nm.age_effect_test(df[df[region_col] == reg], y_col, predictor='lda_1', region_col=region_col,
                                   extra_covars=covars, n_perm=N_PERM, seed=SEED)
            c = cov.set_index('region').loc[reg]
            out.append(dict(metric=metric, test=test, kind='region', region=reg, r=r['r_partial'], p=r['p_perm'],
                            sessions=r['n_sessions'], rows=r['n'], mice=c['mice'], ld1_sd_ratio=c['ld1_sd_ratio']))
            print(f'   {reg:<8} r = {r["r_partial"]:+.3f}  p = {r["p_perm"]:.4f}  ({r["n_sessions"]} sessions, '
                  f'{c["mice"]} mice)')
        if len(regs) >= 2:                                                    # heterogeneity
            h = heterogeneity_test(df[df[region_col].isin(regs)], y_col, region_col, covars, rng)
            out.append(dict(metric=metric, test=test, kind='heterogeneity', region=f'{h["n_regions"]} regions',
                            F=h['F'], p=h['p'], sessions=h['sessions'], rows=h['rows']))
            print(f'   heterogeneity across {h["n_regions"]} regions: F = {h["F"]:.2f}, p = {h["p"]:.4f}')
        for reg in regs:                                                      # leave one region out
            r = nm.age_effect_test(df[df[region_col] != reg], y_col, predictor='lda_1', region_col=region_col,
                                   extra_covars=covars, n_perm=N_PERM, seed=SEED)
            out.append(dict(metric=metric, test=test, kind='leave_one_out', region=reg, r=r['r_partial'],
                            p=r['p_perm'], sessions=r['n_sessions'], rows=r['n']))
        lo = [o for o in out if o['metric'] == metric and o['test'] == test and o['kind'] == 'leave_one_out']
        if lo:
            print(f'   global without each region: r {min(o["r"] for o in lo):+.3f} .. {max(o["r"] for o in lo):+.3f}, '
                  f'max p = {max(o["p"] for o in lo):.4f}')
    t = pd.DataFrame(out)
    t['q'] = np.nan
    m = t['kind'] == 'region'
    t.loc[m, 'q'] = t[m].groupby(['metric', 'test'])['p'].transform(lambda s: multipletests(s, method='fdr_bh')[1])
    path = PI / 'results' / f'lda1_region_specificity_{DATE}.csv'
    t.to_csv(path, index=False)
    pd.concat(coverage).to_csv(PI / 'results' / f'lda1_region_coverage_{DATE}.csv', index=False)
    print(f'\nwrote {path.relative_to(PI)} and lda1_region_coverage_{DATE}.csv')
    return t


def figure(t=None):
    """Swanson flatmaps: region-wise r per measure; admitted regions coloured, others silver; q < 0.05 outlined."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from iblatlas.plots import plot_swanson_vector
    from iblatlas.regions import BrainRegions
    import paper_style as ps
    ps.use('paper')
    if t is None:
        t = pd.read_csv(sorted((PI / 'results').glob('lda1_region_specificity_*.csv'),
                               key=lambda p: pd.to_datetime(p.stem[-10:], dayfirst=True))[-1])
    br = BrainRegions()
    R = t[t['kind'] == 'region']
    meas = [m for m in MEASURES if ((R['metric'] == m[0]) & (R['test'] == m[1])).any()]
    ncol = 3
    nrow = int(np.ceil((len(meas) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 1.55 * nrow), gridspec_kw=dict(wspace=0.05, hspace=0.45))
    axes = axes.ravel()
    for ax in axes[len(meas):]:
        ax.set_axis_off()
    vmax = 0.4
    for ax, (metric, test) in zip(axes, meas):
        d = R[(R['metric'] == metric) & (R['test'] == test)]
        # tested regions coloured by r and labelled; '*' marks q < 0.05 (BH across that measure's regions)
        plot_swanson_vector(d['region'].to_numpy(), d['r'].to_numpy(), ax=ax, br=br, cmap=ps.LD1_CMAP,
                            vmin=-vmax, vmax=vmax, empty_color='#E6E6E6', linewidth=0.15, edgecolor='0.6',
                            annotate=True, annotate_list=d['region'].tolist(), fontsize=4)
        for txt in ax.texts:
            if txt.get_text() in set(d.loc[d['q'] < 0.05, 'region']):
                txt.set_text(txt.get_text() + '*'); txt.set_fontweight('bold')
        h = t[(t['kind'] == 'heterogeneity') & (t['metric'] == metric) & (t['test'] == test)]
        sub = f'{len(d)} regions' + (f'; heterogeneity p = {h["p"].iloc[0]:.2f}' if len(h) else '')
        ax.set_title(f"{metric}, {test.lower().replace('response magnitude', 'change')}", fontsize=plt.rcParams['font.size'], fontweight='bold')
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha='center', va='top',
                fontsize=plt.rcParams['font.size'] * 0.8, color='0.35')
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(plt.Normalize(-vmax, vmax), ps.LD1_CMAP)
    cax = axes[-1].inset_axes([0.15, 0.7, 0.6, 0.07])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[-vmax, 0, vmax])
    axes[-1].text(0.45, 0.3, 'grey: too little data\n* q < 0.05 (BH across regions)', transform=axes[-1].transAxes,
                  ha='center', va='top', fontsize=plt.rcParams['font.size'] * 0.8, color='0.35')
    cb.set_label('partial r with LD1'); cb.outline.set_visible(False)
    for d_ in (PI / 'figures', PI / 'figure_generation' / 'ciss_figures'):
        for ext, kw in (('svg', {}), ('png', {'dpi': 400})):
            fig.savefig(d_ / f'lda1_region_swanson.{ext}', bbox_inches='tight', transparent=True, **kw)
    print('wrote figures/lda1_region_swanson.{svg,png} (and a copy in ciss_figures/)')
    return fig


# --- confidence intervals and dot plot --------------------------------------------------------------------------------
N_BOOT = 1000
ORDER = ['MOs', 'MOp', 'CA1', 'CP', 'LSr', 'LP', 'PO', 'VPM', 'APN', 'MRN']     # cortex, hippocampus, striatum,
GROUP = {'MOs': 'Cortex', 'MOp': 'Cortex', 'CA1': 'Hippocampus', 'CP': 'Striatum', 'LSr': 'Septum',  # septum,
         'LP': 'Thalamus', 'PO': 'Thalamus', 'VPM': 'Thalamus', 'APN': 'Midbrain', 'MRN': 'Midbrain'}  # thalamus, midbrain


def partial_r(d, y_col, region_col, covars):
    """age_effect_test's statistic: r of y and LD1, both residualised on region dummies + covariates."""
    Z = nm._design_matrix(d, region_col, covars)
    Zp = np.linalg.pinv(Z)
    y, x = d[y_col].to_numpy(float), d['lda_1'].to_numpy(float)
    return np.corrcoef(y - Z @ (Zp @ y), x - Z @ (Zp @ x))[0, 1]


def session_bootstrap(df, y_col, region_col, covars, rng, n_boot=N_BOOT):
    """95% CI of the partial r by resampling SESSIONS (a session's rows stay together), as LD1 varies by session."""
    d = df.dropna(subset=[y_col, 'lda_1', region_col] + list(covars)).reset_index(drop=True)
    groups = {s: g for s, g in d.groupby('session')}
    keys = list(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        b = pd.concat([groups[keys[i]].assign(session=f'{keys[i]}_{j}') for j, i in enumerate(pick)])
        try:
            boots.append(partial_r(b, y_col, region_col, covars))
        except Exception:
            pass
    return np.nanpercentile(boots, [2.5, 97.5])


def add_cis(t=None):
    """Session-bootstrap CIs for every region-wise and global r; writes results/lda1_region_ci_<date>.csv."""
    warnings.filterwarnings('ignore')
    if t is None:
        t = latest_table()
    T, _ = cs.load()
    rng = np.random.default_rng(SEED)
    rows = []
    for (metric, test), g in t[t['kind'].isin(['region', 'global'])].groupby(['metric', 'test'], sort=False):
        tbl, y_col, _, _, region_col, covars = nm.GRID[(metric, test)]
        df = T[tbl]
        for r in g.itertuples():
            part = df if r.kind == 'global' else df[df[region_col] == r.region]
            lo, hi = session_bootstrap(part, y_col, region_col, covars, rng, N_BOOT if r.kind == 'region' else 300)
            rows.append(dict(metric=metric, test=test, region=r.region, r_lo=lo, r_hi=hi))
        print('CI done:', metric, test, flush=True)
    ci = t.merge(pd.DataFrame(rows), on=['metric', 'test', 'region'], how='left')
    ci.to_csv(PI / 'results' / f'lda1_region_ci_{DATE}.csv', index=False)
    return ci


def latest_table(stem='lda1_region_specificity'):
    return pd.read_csv(sorted((PI / 'results').glob(f'{stem}_*.csv'),
                              key=lambda p: pd.to_datetime(p.stem[-10:], dayfirst=True))[-1])


def dotplot(ci=None):
    """One panel per measure: region-wise partial r with session-bootstrap 95% CI, regions grouped by brain division,
    the global estimate on top; filled = q < 0.05 (BH across regions), heterogeneity p in the panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.use('paper')
    if ci is None:
        ci = latest_table('lda1_region_ci')
    fs = plt.rcParams['font.size']
    meas = [m for m in MEASURES if ((ci['metric'] == m[0]) & (ci['test'] == m[1])).any()]
    regs = [r for r in ORDER if r in set(ci.loc[ci['kind'] == 'region', 'region'])]
    ypos, y, prev = {}, 0.0, None
    ypos['all'] = y; y += 1.4
    for r in regs:
        if prev is not None and GROUP[r] != GROUP[prev]:
            y += 0.45
        ypos[r] = y; y += 1; prev = r
    fig, axes = plt.subplots(1, len(meas), figsize=(7.4, 2.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, (metric, test) in zip(axes, meas):
        d = ci[(ci['metric'] == metric) & (ci['test'] == test)]
        ax.axvline(0, color='0.6', lw=0.6, zorder=0)
        for row in d.itertuples():
            if row.kind not in ('region', 'global'):
                continue
            yy = ypos['all' if row.kind == 'global' else row.region]
            sig = (row.q < 0.05) if row.kind == 'region' else (row.p < 0.05)
            col = 'k' if row.kind == 'global' else '0.25'
            ax.plot([row.r_lo, row.r_hi], [yy, yy], color=col, lw=1.0 if row.kind == 'global' else 0.8,
                    solid_capstyle='butt')
            ax.plot(row.r, yy, 'D' if row.kind == 'global' else 'o', ms=4.2 if row.kind == 'global' else 3.6,
                    mfc=col if sig else 'white', mec=col, mew=0.9, zorder=3)
        h = d[d['kind'] == 'heterogeneity']
        if len(h):
            ax.text(0.5, -0.17, f'heterogeneity p = {h["p"].iloc[0]:.2f}', transform=ax.transAxes, ha='center',
                    va='top', fontsize=fs * 0.8, color='0.3')
        ax.set_title(f"{metric.replace('Noise correlation', 'Noise corr.')}\n"
                     f"{test.lower().replace('response magnitude', 'change')}", fontsize=fs * 0.9, fontweight='bold')
        ax.set_xlim(-1.05, 1.05); ax.set_xticks([-1, 0, 1])
        ax.set_xlabel('partial r with LD1', fontsize=fs * 0.85)
        ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks([ypos['all']] + [ypos[r] for r in regs])
    axes[0].set_yticklabels(['All regions'] + regs)
    axes[0].get_yticklabels()[0].set_fontweight('bold')
    axes[0].set_ylim(y - 0.4, -0.8)
    for r in regs:                                                            # division names, left of the region names
        first = [x for x in regs if GROUP[x] == GROUP[r]][0]
        if r == first:
            grp = [x for x in regs if GROUP[x] == GROUP[r]]
            axes[0].text(-0.62, np.mean([ypos[x] for x in grp]), GROUP[r], transform=axes[0].get_yaxis_transform(),
                         ha='right', va='center', fontsize=fs * 0.75, color='0.4')
    fig.text(0.99, -0.1, 'mean ± 95% CI (bootstrap over sessions); filled: q < 0.05 (regions, BH) / p < 0.05 (all)',
             ha='right', va='bottom', fontsize=fs * 0.7, color='0.45')
    for d_ in (PI / 'figures', PI / 'figure_generation' / 'ciss_figures'):
        for ext, kw in (('svg', {}), ('png', {'dpi': 400})):
            fig.savefig(d_ / f'lda1_region_dots.{ext}', bbox_inches='tight', transparent=True, **kw)
    print('wrote figures/lda1_region_dots.{svg,png} (and a copy in ciss_figures/)')
    return fig


if __name__ == '__main__':
    tab = None if any(a in sys.argv for a in ('--figure-only', '--dots', '--dots-only')) else main()
    if '--figure' in sys.argv or '--figure-only' in sys.argv:
        figure(tab)
    if '--dots' in sys.argv:
        dotplot(add_cis(tab))
    if '--dots-only' in sys.argv:
        dotplot()
