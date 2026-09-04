"""Separate genuine left/right movement bias from the right-camera noise artefact.

The first pass showed LI[0.5-8 Hz power] correlates r=+0.90 with LI[high-frequency residual].
That residual is not a clean noise measure though: a mouse that genuinely moves a paw faster also
has more real high-frequency content. So this script uses two cleaner handles:

  * 32 Hz wavelet power as a near-pure noise reference. A mouse forepaw cannot oscillate at 32 Hz,
    so this band is essentially all tracking error.
  * a GAIN-INVARIANT per-paw signal-to-noise ratio, SNR = P(0.5-8 Hz) / P(32 Hz). Any per-camera
    multiplicative pixel-gain error cancels in the ratio, so LI[SNR] cannot be produced by the
    resolution correction at all.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '.')
from analyse_paw_bias import icc_and_perm, mouse_level_corr, sign_consistency
from extract_paw_bias import li


def partial_corr(pm, a, b, ctrl):
    d = pm[[a, b, ctrl]].dropna()
    ra = d[a] - np.polyval(np.polyfit(d[ctrl], d[a], 1), d[ctrl])
    rb = d[b] - np.polyval(np.polyfit(d[ctrl], d[b], 1), d[ctrl])
    r, p = stats.pearsonr(ra, rb)
    return r, p, len(d)


def main(csv):
    df = pd.read_csv(csv)
    df = df[df.n_both.notna() & (df.n_both > 6000)].copy()

    # gain-invariant per-paw SNR and its laterality index
    for t in 'LR':
        df[f'snr_{t}'] = df[f'bandpow_{t}'] / df[f'noisepow_{t}']
    df['li_snr'] = [li(a, b) for a, b in zip(df.snr_L, df.snr_R)]

    print('== how contaminated is each metric? (session level, r with the 32 Hz noise LI) ==')
    for c in ['li_bandpow', 'li_spmed', 'li_jitter', 'li_boutrate', 'lag_s',
              'xcorr_peak', 'li_snr']:
        d = df[[c, 'li_noisepow']].dropna()
        r, p = stats.pearsonr(d[c], d.li_noisepow)
        print(f'  {c:14s} r={r:+.3f} p={p:.2e}   mean={df[c].mean():+.4f}')
    print('\n  32 Hz noise LI itself: mean=%.4f  (strongly right-shifted = right camera noisier)'
          % df.li_noisepow.mean())
    print('  per-paw SNR: left median=%.1f  right median=%.1f'
          % (df.snr_L.median(), df.snr_R.median()))

    print('\n== population-level bias on the gain-invariant SNR metric (mouse means) ==')
    pm = df.groupby('mouse').mean(numeric_only=True)
    v = pm.li_snr.dropna()
    t = stats.ttest_1samp(v, 0)
    w = stats.wilcoxon(v)
    print(f'  LI[SNR] mean={v.mean():+.4f} sd={v.std():.4f} n={len(v)} mice  '
          f't={t.statistic:+.2f} p={t.pvalue:.2e}  wilcoxon p={w.pvalue:.2e}')
    print(f'  mice with left-biased SNR: {(v > 0).sum()}/{len(v)}')

    print('\n== is the SNR bias an individual trait? ==')
    r = icc_and_perm(df, 'li_snr')
    print(f'  ICC={r["icc"]:.3f} p_perm={r["p_perm"]:.4f} '
          f'({r["n_sess"]} sessions, {r["n_mice"]} mice, null95={r["null_95"]:.3f})')
    sc = sign_consistency(df, 'li_snr')
    print(f'  sign consistency: {sc["obs"]*100:.1f}% of {sc["n_mice"]} mice '
          f'(chance {sc["null_mean"]*100:.1f}%) p_perm={sc["p_perm"]:.4f}')
    m = df[['li_bandpow_h1', 'li_bandpow_h2']].dropna()
    print(f'  (within-session split-half of li_bandpow, reliability ceiling: '
          f'r={stats.pearsonr(m.li_bandpow_h1, m.li_bandpow_h2)[0]:.3f})')

    print('\n== task relations, contamination-controlled (mouse level, n=%d) ==' % len(pm))
    for a in ['li_snr', 'li_bandpow', 'lag_s', 'xcorr_peak']:
        for b in ['wheel_bias', 'frac_right_choice', 'frac_right_zero', 'perf']:
            res = mouse_level_corr(df, a, b, n_perm=5000)
            extra = ''
            if a != 'li_snr':
                pr, pp, n = partial_corr(pm, a, b, 'li_noisepow')
                extra = f' | partial on noise: r={pr:+.3f} p={pp:.3f}'
            print(f'  {a:12s} vs {b:18s} r={res["r"]:+.3f} rho={res["rho"]:+.3f} '
                  f'p_perm={res["p_perm"]:.4f}{extra}')

    print('\n== is the noise LI itself related to the task? (it should not be, if the task '
          'relations are real) ==')
    for b in ['wheel_bias', 'frac_right_choice', 'perf']:
        res = mouse_level_corr(df, 'li_noisepow', b, n_perm=5000)
        print(f'  li_noisepow  vs {b:18s} r={res["r"]:+.3f} p_perm={res["p_perm"]:.4f}')

    df.to_csv(csv.replace('.csv', '_snr.csv'), index=False)
    print('\nwrote', csv.replace('.csv', '_snr.csv'))


if __name__ == '__main__':
    main(sys.argv[1])
