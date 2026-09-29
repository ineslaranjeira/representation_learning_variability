"""Figures for the left/right forepaw bias analysis."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_context('paper', font_scale=1.0)
CL, CR = '#3B6FB6', '#C4562F'      # left paw, right paw


def _mice_ge2(df):
    return df[df.mouse.isin(df.mouse.value_counts()[lambda s: s >= 2].index)]


def fig_confound(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].scatter(df.jitter_L, df.jitter_R, s=12, alpha=.6, color='k')
    lim = [0, np.nanpercentile(df[['jitter_L', 'jitter_R']].values, 99.5)]
    ax[0].plot(lim, lim, 'k--', lw=.8)
    ax[0].set(xlabel='left-paw jitter (px)', ylabel='right-paw jitter (px)',
              title='tracking noise is larger\non the right (low-res) camera', xlim=lim, ylim=lim)

    for a, (c, t) in zip(ax[1:], [('li_spmed', 'frame-difference speed'),
                                  ('li_bandpow', '0.5-8 Hz band power')]):
        m = df[[c, 'li_jitter']].dropna()
        a.scatter(m.li_jitter, m[c], s=12, alpha=.6, color='k')
        r = stats.pearsonr(m.li_jitter, m[c])
        b = np.polyfit(m.li_jitter, m[c], 1)
        xs = np.linspace(m.li_jitter.min(), m.li_jitter.max(), 20)
        a.plot(xs, np.polyval(b, xs), color='crimson', lw=1.5)
        a.axhline(0, color='grey', lw=.6)
        a.set(xlabel='LI jitter', ylabel=f'LI {t}',
              title=f'{t}\nr={r[0]:+.2f}, p={r[1]:.1e}')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_population(df, path):
    metrics = [('li_bandpow', 'amplitude\n(0.5-8 Hz power)'),
               ('li_spmed', 'amplitude\n(median speed)'),
               ('li_boutrate', 'bout rate\n(matched duty cycle)'),
               ('li_boutdur', 'bout duration\n(matched duty cycle)')]
    pm = df.groupby('mouse').mean(numeric_only=True)
    fig, ax = plt.subplots(1, len(metrics) + 1, figsize=(3 * (len(metrics) + 1), 3.0))
    for a, (c, t) in zip(ax, metrics):
        v = pm[c].dropna()
        a.hist(v, bins=22, color='0.6', edgecolor='w')
        a.axvline(0, color='k', lw=1)
        a.axvline(v.mean(), color='crimson', lw=1.5)
        p = stats.ttest_1samp(v, 0).pvalue
        a.set(xlabel=f'LI  (+ = left paw)', title=f'{t}\nmean={v.mean():+.3f}, p={p:.1e}')
    a = ax[-1]
    v = pm['lag_s'].dropna() * 1000
    a.hist(v, bins=22, color='0.6', edgecolor='w')
    a.axvline(0, color='k', lw=1)
    a.axvline(v.mean(), color='crimson', lw=1.5)
    a.set(xlabel='xcorr lag (ms, + = left leads)', title='temporal precedence')
    ax[0].set_ylabel('# mice')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_individuality(df, path, icc_txt=None):
    d = _mice_ge2(df[['mouse', 'li_bandpow', 'li_boutrate', 'lag_s']].dropna(
        subset=['li_bandpow']))
    order = d.groupby('mouse').li_bandpow.mean().sort_values().index
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2),
                           gridspec_kw={'width_ratios': [2.1, 1, 1]})
    a = ax[0]
    for i, m in enumerate(order):
        v = d[d.mouse == m].li_bandpow
        a.scatter(np.full(len(v), i), v, s=16, color='0.5', alpha=.8, zorder=2)
        a.scatter([i], [v.mean()], s=45, color='crimson', marker='_', zorder=3)
    a.axhline(0, color='k', lw=.8)
    a.set(xlabel=f'mouse (n={len(order)}, ranked by mean)',
          ylabel='LI amplitude (+ = left paw)',
          title='per-session paw bias, grouped by mouse' +
                (f'\n{icc_txt}' if icc_txt else ''))
    a.set_xticks([])

    # split-half reliability
    a = ax[1]
    m = df[['li_bandpow_h1', 'li_bandpow_h2']].dropna()
    a.scatter(m.li_bandpow_h1, m.li_bandpow_h2, s=12, alpha=.6, color='k')
    r = stats.pearsonr(m.li_bandpow_h1, m.li_bandpow_h2)
    lim = [m.values.min(), m.values.max()]
    a.plot(lim, lim, 'k--', lw=.8)
    a.set(xlabel='LI, first half of session', ylabel='LI, second half',
          title=f'within-session reliability\nr={r[0]:.3f} (n={len(m)})')

    # session-to-session within mouse: first vs second session
    a = ax[2]
    pairs = []
    for mo, g in d.groupby('mouse'):
        v = g.li_bandpow.to_numpy()
        if len(v) >= 2:
            pairs.append((v[0], v[1]))
    pairs = np.array(pairs)
    a.scatter(pairs[:, 0], pairs[:, 1], s=18, alpha=.75, color='k')
    r = stats.pearsonr(pairs[:, 0], pairs[:, 1])
    lim = [pairs.min(), pairs.max()]
    a.plot(lim, lim, 'k--', lw=.8)
    a.set(xlabel='LI, session 1', ylabel='LI, session 2',
          title=f'across-session reliability\nr={r[0]:.3f}, p={r[1]:.3f} (n={len(pairs)} mice)')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_spectrum(df, path):
    freqs = ['0.5', '1.0', '2.0', '4.0', '8.0']
    pm = df.groupby('mouse').mean(numeric_only=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    # mean spectral shape per paw
    a = ax[0]
    for tag, col, lab in [('L', CL, 'left paw'), ('R', CR, 'right paw')]:
        v = np.array([pm[f'shape_{tag}_{f}'] for f in freqs])
        a.errorbar(range(len(freqs)), v.mean(1), v.std(1) / np.sqrt(v.shape[1]),
                   color=col, marker='o', label=lab, capsize=2)
    a.set(xticks=range(len(freqs)), xticklabels=freqs, xlabel='wavelet frequency (Hz)',
          ylabel='fraction of 0.5-8 Hz power', title='spectral shape per paw')
    a.legend(frameon=False)
    # per-frequency LI of raw power and of shape
    for a, pref, t in [(ax[1], 'li_pow_', 'LI of band power'),
                       (ax[2], 'li_shape_', 'LI of spectral shape (scale-free)')]:
        v = np.array([pm[pref + f].dropna().to_numpy() for f in freqs])
        a.errorbar(range(len(freqs)), v.mean(1), v.std(1) / np.sqrt(v.shape[1]),
                   color='k', marker='o', capsize=2)
        a.axhline(0, color='grey', lw=.8)
        for i, f in enumerate(freqs):
            p = stats.ttest_1samp(v[i], 0).pvalue
            if p < .05:
                a.text(i, v[i].mean(), '*', ha='center', va='bottom', fontsize=13)
        a.set(xticks=range(len(freqs)), xticklabels=freqs,
              xlabel='wavelet frequency (Hz)', ylabel='LI (+ = left)', title=t)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_task(df, path):
    pairs = [('li_bandpow', 'frac_right_zero', 'P(right choice), 0% contrast'),
             ('li_bandpow', 'frac_right_choice', 'P(right choice), all trials'),
             ('li_bandpow', 'perf', 'performance'),
             ('xcorr_peak', 'perf', 'performance')]
    fig, ax = plt.subplots(1, len(pairs), figsize=(3.2 * len(pairs), 3.2))
    for a, (x, y, ylab) in zip(ax, pairs):
        d = df[[x, y, 'mouse']].dropna()
        pm = d.groupby('mouse').mean(numeric_only=True)
        a.scatter(d[x], d[y], s=10, alpha=.35, color='0.6', label='sessions')
        a.scatter(pm[x], pm[y], s=26, color='k', label='mouse means')
        r = stats.pearsonr(pm[x], pm[y])
        b = np.polyfit(pm[x], pm[y], 1)
        xs = np.linspace(pm[x].min(), pm[x].max(), 20)
        a.plot(xs, np.polyval(b, xs), color='crimson', lw=1.5)
        a.set(xlabel=x, ylabel=ylab,
              title=f'mouse-level r={r[0]:+.2f}\np={r[1]:.3f} (n={len(pm)})')
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(csv, outdir, icc_txt=None):
    df = pd.read_csv(csv)
    if 'err' in df:
        df = df[df.err.isna()]
    df = df[df.n_both.notna() & (df.n_both > 6000)].copy()
    for name, fn in [('fig1_confound', fig_confound), ('fig2_population', fig_population),
                     ('fig3_individuality', fig_individuality), ('fig4_spectrum', fig_spectrum),
                     ('fig5_task', fig_task)]:
        p = f'{outdir}/{name}.png'
        try:
            fn(df, p, icc_txt) if name == 'fig3_individuality' else fn(df, p)
            print('wrote', p)
        except Exception as e:
            print('FAILED', name, type(e).__name__, e)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
