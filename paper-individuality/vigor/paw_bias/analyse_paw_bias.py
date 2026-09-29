"""Left/right forepaw movement bias across animals.

Reads the per-session table from extract_paw_bias.py and asks, in order:
  0. Confound check   - how much of the L/R difference is right-camera pixel noise?
  1. Population level - is there a group-level paw preference? (mostly uninterpretable, see below)
  2. Individuality    - is paw bias a stable property of a MOUSE? (ICC + permutation + split-half)
  3. Lab              - is the between-mouse structure just between-lab structure?
  4. Frequency        - which movement timescales carry the bias?
  5. Task             - does paw bias relate to choice side bias or performance?

Because halving the left camera also halves its tracking noise, the population MEAN laterality
index is contaminated by a camera artefact of unknown size. Individual differences around that
mean are the interpretable quantity, so the jitter covariate is checked explicitly at step 0/2.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(2024)
N_PERM = 10000


def wprint(*a):
    print(*a)


def icc_and_perm(df, col, mouse='mouse', n_perm=N_PERM, rng=RNG):
    """One-way random-effects ICC(1) on sessions nested in mice, plus a permutation test that
    shuffles session -> mouse assignment (preserving the number of sessions per mouse).
    Only mice with >=2 sessions contribute; ICC(1) is reported with the standard
    unequal-group-size correction."""
    d = df[[mouse, col]].dropna()
    d = d[d[mouse].isin(d[mouse].value_counts()[lambda s: s >= 2].index)]
    if d[mouse].nunique() < 5:
        return None

    def _icc(vals, groups):
        g = pd.DataFrame({'v': vals, 'g': groups})
        k = g.groupby('g')['v'].size()
        n_g, N = len(k), len(g)
        gm = g['v'].mean()
        means = g.groupby('g')['v'].mean()
        ms_b = (k * (means - gm) ** 2).sum() / (n_g - 1)
        ss_w = g.groupby('g')['v'].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
        ms_w = ss_w / (N - n_g)
        k0 = (N - (k ** 2).sum() / N) / (n_g - 1)      # mean group size correction
        return (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w), ms_b / ms_w

    obs, f_obs = _icc(d[col].to_numpy(), d[mouse].to_numpy())
    g = d[mouse].to_numpy()
    v = d[col].to_numpy()
    null = np.array([_icc(v, rng.permutation(g))[0] for _ in range(n_perm)])
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    return dict(metric=col, n_sess=len(d), n_mice=d[mouse].nunique(), icc=obs,
                f_ratio=f_obs, p_perm=p, null_95=np.quantile(null, 0.95))


def partial_r(x, y, z):
    """Pearson r between x and y after linearly removing z from both."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)


def mouse_level_corr(df, a, b, n_perm=N_PERM, rng=RNG):
    """Correlation between two metrics at the level of MICE, not sessions: repeated sessions from
    one animal are averaged first, so nothing here can be manufactured by pseudo-replication.
    p is a permutation p (mouse labels of b shuffled); Spearman is reported as a robustness check."""
    pm = df.groupby('mouse')[[a, b]].mean().dropna()
    if len(pm) < 10:
        return None
    x, y = pm[a].to_numpy(), pm[b].to_numpy()
    r, p_param = stats.pearsonr(x, y)
    rho = stats.spearmanr(x, y)[0]
    null = np.array([abs(stats.pearsonr(x, rng.permutation(y))[0]) for _ in range(n_perm)])
    return dict(r=r, p_param=p_param, rho=rho,
                p_perm=(1 + (null >= abs(r)).sum()) / (1 + n_perm), n_mice=len(pm))


def sign_consistency(df, col, min_sess=3, rng=RNG, n_perm=N_PERM):
    """Of the mice with >=min_sess sessions, how many have EVERY session on the same side of the
    population median? Compared against shuffling sessions across mice, which is the honest null
    (a simple binomial would ignore that the metric's spread differs between animals)."""
    d = df[['mouse', col]].dropna()
    keep = d.mouse.value_counts()[lambda s: s >= min_sess].index
    d = d[d.mouse.isin(keep)]
    if len(keep) < 5:
        return None
    med = d[col].median()

    def frac(vals, groups):
        g = pd.DataFrame({'v': vals > med, 'g': groups})
        return g.groupby('g')['v'].apply(lambda s: s.all() or (~s).all()).mean()

    obs = frac(d[col].to_numpy(), d.mouse.to_numpy())
    gg = d.mouse.to_numpy()
    v = d[col].to_numpy()
    null = np.array([frac(v, rng.permutation(gg)) for _ in range(n_perm)])
    return dict(n_mice=len(keep), obs=obs, null_mean=null.mean(),
                p_perm=(1 + (null >= obs).sum()) / (1 + n_perm))


def main(csv):
    df = pd.read_csv(csv)
    if 'err' in df:
        df = df[df.err.isna()].drop(columns=['err'])
    df = df[df.n_both.notna() & (df.n_both > 6000)].copy()
    wprint(f'== dataset ==\n{len(df)} sessions, {df.mouse.nunique()} mice, '
           f'{df.lab.nunique() if "lab" in df else "?"} labs, '
           f'median {df.dur_min.median():.0f} min/session')
    ge2 = df.mouse.value_counts()[lambda s: s >= 2]
    wprint(f'{len(ge2)} mice with >=2 sessions ({ge2.sum()} sessions)')

    # ---------------------------------------------------------------- 0. confound
    wprint('\n== 0. camera-noise confound ==')
    wprint('jitter (high-frequency tracking residual, common spatial units):')
    wprint(f'  left  paw: {df.jitter_L.median():.3f}   right paw: {df.jitter_R.median():.3f}   '
           f'ratio R/L = {(df.jitter_R / df.jitter_L).median():.2f}')
    for raw, lab in [('li_spmed', 'frame-difference speed (8 Hz low-passed)'),
                     ('li_bandpow', '0.5-8 Hz wavelet band power'),
                     ('li_noisepow', '32 Hz power (pure noise proxy)')]:
        if raw in df:
            r = stats.pearsonr(*[v for v in np.array(
                df[[raw, 'li_jitter']].dropna()).T][:2])
            wprint(f'  LI[{lab:38s}] mean={df[raw].mean():+.4f}  '
                   f'r with LI[jitter] = {r[0]:+.3f} (p={r[1]:.1e})')

    # ---------------------------------------------------------------- 1. population
    wprint('\n== 1. population-level bias (mouse-averaged, so each animal counts once) ==')
    per_mouse = df.groupby('mouse').mean(numeric_only=True)
    for c, lab in [('li_bandpow', 'amplitude: 0.5-8 Hz band power'),
                   ('li_spmed', 'amplitude: median speed'),
                   ('li_boutrate', 'scale-free: bout rate at matched duty cycle'),
                   ('li_boutdur', 'scale-free: bout duration at matched duty cycle'),
                   ('lag_s', 'temporal: cross-correlation lag (s, + = left leads)'),
                   ('xcorr_peak', 'bilateral coupling (peak xcorr)')]:
        if c not in per_mouse:
            continue
        v = per_mouse[c].dropna()
        t = stats.ttest_1samp(v, 0)
        w = stats.wilcoxon(v) if len(v) > 10 else None
        wprint(f'  {lab:52s} mean={v.mean():+.4f} sd={v.std():.4f} n={len(v)}  '
               f't={t.statistic:+.2f} p={t.pvalue:.2e}'
               + (f'  wilcoxon p={w.pvalue:.2e}' if w else ''))
    wprint('  NB: a nonzero mean here is NOT evidence of biology -- see step 0.')

    # ---------------------------------------------------------------- 2. individuality
    wprint('\n== 2. is paw bias a stable individual trait? ==')
    wprint('split-half reliability within session (ceiling on any across-session correlation):')
    if 'li_bandpow_h1' in df:
        m = df[['li_bandpow_h1', 'li_bandpow_h2']].dropna()
        r = stats.pearsonr(m.li_bandpow_h1, m.li_bandpow_h2)
        wprint(f'  li_bandpow first vs second half: r={r[0]:.3f} (p={r[1]:.1e}, n={len(m)})')
    wprint('\nICC(1): between-mouse variance / total, sessions nested in mice')
    wprint(f'{"metric":22s} {"n_sess":>6s} {"n_mice":>6s} {"ICC":>7s} {"F":>6s} {"p_perm":>8s} {"null95":>7s}')
    rows = []
    for c in ['li_bandpow', 'li_spmed', 'li_boutrate', 'li_boutdur', 'lag_s',
              'xcorr_peak', 'li_jitter']:
        if c not in df:
            continue
        r = icc_and_perm(df, c)
        if r:
            rows.append(r)
            wprint(f'{c:22s} {r["n_sess"]:6d} {r["n_mice"]:6d} {r["icc"]:7.3f} '
                   f'{r["f_ratio"]:6.2f} {r["p_perm"]:8.4f} {r["null_95"]:7.3f}')
    wprint('\nsign consistency: mice whose every session falls on the same side of the '
           'population median')
    for c in ['li_bandpow', 'li_boutrate', 'lag_s']:
        if c not in df:
            continue
        r = sign_consistency(df, c)
        if r:
            wprint(f'  {c:14s} {r["obs"]*100:5.1f}% of {r["n_mice"]} mice '
                   f'(chance {r["null_mean"]*100:.1f}%)  p_perm={r["p_perm"]:.4f}')

    wprint('\njitter-corrected amplitude bias (li_bandpow with li_jitter regressed out):')
    d = df[['li_bandpow', 'li_jitter', 'mouse']].dropna()
    d['li_bandpow_adj'] = (d.li_bandpow -
                           np.polyval(np.polyfit(d.li_jitter, d.li_bandpow, 1), d.li_jitter))
    r = icc_and_perm(d, 'li_bandpow_adj')
    if r:
        wprint(f'  ICC={r["icc"]:.3f}  p_perm={r["p_perm"]:.4f}  '
               f'(n={r["n_sess"]} sessions, {r["n_mice"]} mice)')

    # ---------------------------------------------------------------- 3. lab
    if 'lab' in df:
        wprint('\n== 3. lab as an alternative explanation ==')
        for c in ['li_bandpow', 'li_boutrate', 'lag_s']:
            if c not in df:
                continue
            d = df[[c, 'lab', 'mouse']].dropna()
            groups = [g[c].to_numpy() for _, g in d.groupby('lab') if len(g) >= 3]
            if len(groups) >= 3:
                f = stats.f_oneway(*groups)
                # between-lab variance on mouse means, to avoid pseudo-replication
                pm = d.groupby(['lab', 'mouse'])[c].mean().reset_index()
                gm = [g[c].to_numpy() for _, g in pm.groupby('lab') if len(g) >= 3]
                f2 = stats.f_oneway(*gm) if len(gm) >= 3 else None
                wprint(f'  {c:14s} by-session F={f.statistic:.2f} p={f.pvalue:.2e} '
                       f'({len(groups)} labs)' +
                       (f' | by-mouse F={f2.statistic:.2f} p={f2.pvalue:.2e}' if f2 else ''))
        wprint('  mouse-level ICC after centring each metric within its lab '
               '(removes any between-lab offset):')
        for c in ['li_bandpow', 'li_boutrate', 'lag_s']:
            if c not in df:
                continue
            d = df[[c, 'lab', 'mouse']].dropna().copy()
            d[c + '_wl'] = d[c] - d.groupby('lab')[c].transform('mean')
            r = icc_and_perm(d, c + '_wl')
            if r:
                wprint(f'    {c:14s} ICC={r["icc"]:6.3f}  p_perm={r["p_perm"]:.4f}  '
                       f'({r["n_sess"]} sessions, {r["n_mice"]} mice)')

    # ---------------------------------------------------------------- 4. frequency
    wprint('\n== 4. which timescales carry the bias? (mouse means) ==')
    wprint(f'{"freq":>6s} {"LI power":>18s} {"LI spectral shape":>22s} {"ICC(shape)":>11s}')
    for f in ['0.5', '1.0', '2.0', '4.0', '8.0']:
        cp, cs = f'li_pow_{f}', f'li_shape_{f}'
        if cp not in df:
            continue
        vp = per_mouse[cp].dropna()
        vs = per_mouse[cs].dropna()
        tp, ts = stats.ttest_1samp(vp, 0), stats.ttest_1samp(vs, 0)
        ic = icc_and_perm(df, cs, n_perm=2000)
        wprint(f'{f:>6s} {vp.mean():+8.4f} p={tp.pvalue:7.1e} '
               f'{vs.mean():+10.4f} p={ts.pvalue:7.1e} '
               + (f'{ic["icc"]:8.3f} p={ic["p_perm"]:.4f}' if ic else ''))

    # ---------------------------------------------------------------- 5. task
    wprint('\n== 5. does paw bias relate to the task? (mouse level) ==')
    pairs = [('li_bandpow', 'frac_right_choice'), ('li_bandpow', 'frac_right_zero'),
             ('li_bandpow', 'wheel_bias'), ('li_bandpow', 'perf'),
             ('li_boutrate', 'frac_right_zero'), ('lag_s', 'frac_right_zero'),
             ('xcorr_peak', 'perf'), ('li_spmed_wheel', 'frac_right_zero'),
             ('li_bandpow', 'li_jitter')]
    for a, b in pairs:
        if a not in df or b not in df:
            continue
        r = mouse_level_corr(df, a, b, n_perm=5000)
        if r:
            wprint(f'  {a:16s} vs {b:18s} r={r["r"]:+.3f} rho={r["rho"]:+.3f} '
                   f'p_perm={r["p_perm"]:.4f} (n={r["n_mice"]} mice)')

    df.to_csv(csv.replace('.csv', '_analysed.csv'), index=False)
    per_mouse.to_csv(csv.replace('.csv', '_per_mouse.csv'))
    wprint('\nwrote per-mouse table:', csv.replace('.csv', '_per_mouse.csv'))


if __name__ == '__main__':
    main(sys.argv[1])
