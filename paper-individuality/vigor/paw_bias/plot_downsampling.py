"""Figure: what each candidate fix does to the left/right indices."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy import stats

FRQ = ['0.5', '1.0', '2.0', '4.0', '8.0']
C = {'raw': '#4A4E57', 'quant': '#2F5E9E', 'match': '#B54A28', 'whiten': '#8A6516'}
LBL = {'raw': 'RAW', 'quant': 'QUANT', 'match': 'MATCH', 'whiten': 'WHITEN'}


def main(csv, out):
    df = pd.read_csv(csv)
    if 'err' in df:
        df = df[df.err.isna()]
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.8))

    # (1) noise floor
    a = ax[0]
    for i, v in enumerate(['raw', 'quant', 'match']):
        s = df[f'{v}_li_noisepow']
        a.scatter(np.full(len(s), i) + np.random.uniform(-.13, .13, len(s)), s,
                  s=9, alpha=.5, color=C[v])
        a.plot([i - .28, i + .28], [s.mean()] * 2, color='k', lw=2, zorder=5)
    a.axhline(0, color='crimson', ls='--', lw=1)
    a.set(xticks=range(3), xticklabels=[LBL[v] for v in ['raw', 'quant', 'match']],
          ylabel='LI 32 Hz power (noise floor)',
          title='C1  noise floors equalised?\n(0 = matched)')

    # (2) does the band-power metric move at all?
    a = ax[1]
    for v in ['quant', 'match']:
        a.scatter(df.raw_li_bandpow, df[f'{v}_li_bandpow'], s=11, alpha=.6,
                  color=C[v], label=f'{LBL[v]}  r={stats.pearsonr(df.raw_li_bandpow, df[f"{v}_li_bandpow"])[0]:.4f}')
    lim = [df.raw_li_bandpow.min() - .03, df.raw_li_bandpow.max() + .03]
    a.plot(lim, lim, 'k--', lw=.8)
    a.set(xlabel='LI 0.5-8 Hz power, RAW', ylabel='LI 0.5-8 Hz power, after fix',
          title='the metric the clustering uses\nis essentially unchanged', xlim=lim, ylim=lim)
    a.legend(frameon=False, fontsize=9, loc='upper left')

    # (3) contrast: the speed metric, which IS contaminated
    a = ax[2]
    for v in ['quant', 'match']:
        a.scatter(df.raw_li_spmed, df[f'{v}_li_spmed'], s=11, alpha=.6, color=C[v],
                  label=f'{LBL[v]}  r={stats.pearsonr(df.raw_li_spmed, df[f"{v}_li_spmed"])[0]:.3f}')
    lim = [min(df.raw_li_spmed.min(), df.match_li_spmed.min()) - .03,
           max(df.raw_li_spmed.max(), df.match_li_spmed.max()) + .03]
    a.plot(lim, lim, 'k--', lw=.8)
    a.set(xlabel='LI median speed, RAW', ylabel='LI median speed, after fix',
          title='by contrast, the speed measure\ndoes shift', xlim=lim, ylim=lim)
    a.legend(frameon=False, fontsize=9, loc='upper left')

    # (4) per-frequency profile
    a = ax[3]
    for v in ['raw', 'quant', 'match']:
        vals = [df[f'{v}_li_pow_{float(f)}'].mean() for f in FRQ]
        se = [df[f'{v}_li_pow_{float(f)}'].sem() for f in FRQ]
        a.errorbar(range(len(FRQ)), vals, se, color=C[v], marker='o', capsize=2, label=LBL[v])
    a.axhline(0, color='grey', lw=.8)
    a.set(xticks=range(len(FRQ)), xticklabels=FRQ, xlabel='wavelet frequency (Hz)',
          ylabel='LI power (+ = left)', title='C3  per-frequency profile')
    a.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
