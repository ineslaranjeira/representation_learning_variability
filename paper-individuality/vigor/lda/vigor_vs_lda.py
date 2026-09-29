"""
DOES LD1 TRACK WHEEL VIGOR, OR PAW VIGOR?
==========================================
Mouse-level, on the current states_files (332 sessions, 101 mice) and the shrinkage
LDA embedding regenerated locally by make_embedding.py (260 sessions, 56 mice).

WHAT IS AND IS NOT INDEPENDENT OF THE LDA
    The LDA features are per-epoch occupancies of syllables built from PAW velocity
    wavelets (+ whisk, lick). So:
      * WHEEL vigor is an independent signal -- nothing about the wheel enters the
        syllables, so a correlation is a real external correlate of LD1.
      * PAW vigor is PARTLY CIRCULAR -- the HMM states are clusters of paw velocity,
        so "LD1 tracks paw amplitude" is close to restating what the states are. The
        informative question there is WHICH aspect: overall amplitude, or the
        temporal structure at matched duty cycle (the *_p75 metrics, which are
        invariant to any per-session multiplicative gain).

EVERY CORRELATION IS REPORTED THREE WAYS
    raw            Spearman across mice
    lab-centred    each variable centred within its lab first. LD1 has a lab effect
                   (Kruskal p = 0.0014 in the earlier check), and mice are nested in
                   labs, so this is the column that is about mice.
    partial        controlling the paw high-frequency noise index, which is the
                   camera-gain / tracking-quality axis segmentation/paw_bias
                   identified; and separately controlling session duration.

LD1's SIGN IS ARBITRARY. An eigenvector is defined up to sign, so only the RELATIVE
signs within a run mean anything. They are reported as-is and the text says so.
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, kruskal

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EMBEDDING = os.environ.get('EMBEDDING', 'mouse_LDA_5_bins_raw_shrink0.5_360_28-09-2026')
VIGOR_CSV = HERE / 'vigor_sessions_states.csv'

WHEEL_METRICS = ['wheel_frac_moving', 'wheel_mean_speed_moving', 'wheel_mean_speed',
                 'wheel_median_speed', 'wheel_p95_speed', 'wheel_sd_speed',
                 'wheel_bout_rate_per_min', 'wheel_mean_bout_s',
                 'wheel_distance_per_min', 'wheel_mean_vel_signed', 'wheel_turn_bias',
                 'wheel_bout_rate_p75', 'wheel_mean_bout_p75_s']
PAW_METRICS = ['paw_frac_moving', 'paw_mean_speed_moving', 'paw_mean_speed',
               'paw_median_speed', 'paw_p95_speed', 'paw_sd_speed',
               'paw_bout_rate_per_min', 'paw_mean_bout_s', 'paw_distance_per_min',
               'paw_band_0.5_8', 'paw_bout_rate_p75', 'paw_mean_bout_p75_s']
EPOCH_METRICS = [f'{s}_speed_{e}' for s in ('wheel', 'paw')
                 for e in ('Pre-quiescence', 'Quiescence', 'Choice', 'ITI')]
CO_PRIMARY = {'wheel': ['wheel_frac_moving', 'wheel_mean_speed_moving'],
              'paw': ['paw_frac_moving', 'paw_mean_speed_moving']}


# ------------------------------------------------------------------ small helpers
def bh_fdr(p):
    p = np.asarray(p, float)
    n, order = len(p), np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def center_within(v, groups):
    """Subtract each group's mean. NaNs stay NaN."""
    s = pd.Series(np.asarray(v, float))
    return (s - s.groupby(np.asarray(groups)).transform('mean')).to_numpy()


def partial_spearman(x, y, z):
    """Spearman partial: rank everything, then residualise both on z."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 8:
        return np.nan, np.nan, int(m.sum())
    rx, ry, rz = (rankdata(a[m]) for a in (x, y, z))
    Z = np.c_[np.ones(m.sum()), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = spearmanr(ex, ey)
    return r, p, int(m.sum())


def perm_p(x, y, n=10000, seed=0):
    """Two-tailed permutation p for Spearman, shuffling one side."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan
    a, b = np.asarray(x)[m], np.asarray(y)[m]
    obs = abs(spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    null = np.array([abs(spearmanr(a, rng.permutation(b))[0]) for _ in range(n)])
    return float((1 + (null >= obs).sum()) / (1 + n))


def split_half_reliability(df, col, key='mouse_name', min_n=4, n_rep=300, seed=0):
    """Across-SESSION split half, Spearman-Brown corrected to the full session count.

    Halves share no session, so session-level nuisance counts as noise -- the same
    estimator wheel_vigor_first_vs_proficient.ipynb uses for its ceiling.
    """
    rng = np.random.default_rng(seed)
    counts = df.groupby(key)[col].count()
    mice = counts[counts >= min_n].index
    if len(mice) < 8:
        return np.nan, 0
    vals = {m: df.loc[df[key] == m, col].dropna().to_numpy() for m in mice}
    acc = []
    for _ in range(n_rep):
        A, B = [], []
        for m in mice:
            v = vals[m]
            idx = rng.permutation(len(v))
            h = len(v) // 2
            A.append(v[idx[:h]].mean())
            B.append(v[idx[h:2 * h]].mean())
        r = spearmanr(A, B)[0]
        if np.isfinite(r):
            acc.append(r)
    if not acc:
        return np.nan, len(mice)
    r = float(np.mean(acc))
    return (2 * r / (1 + r) if r > -1 else np.nan), len(mice)


# ------------------------------------------------------------------------- load
def load():
    V = pd.read_csv(VIGOR_CSV)
    E = pd.read_pickle(ROOT / 'clustering' / 'data_files' / EMBEDDING)
    E = E.rename(columns={i: f'LD{i + 1}' for i in range(8)})
    lds = [f'LD{i}' for i in range(1, 9)]
    E = E[['session', 'mouse_name', 'lab'] + lds]
    print(f'vigor: {len(V)} sessions, {V.mouse_name.nunique()} mice')
    print(f'embedding {EMBEDDING}: {len(E)} sessions, {E.mouse_name.nunique()} mice, '
          f'{E.lab.nunique()} labs')
    S = E.merge(V.drop(columns=['mouse_name']), on='session', how='inner')
    missing = set(E.session) - set(V.session)
    print(f'merged: {len(S)} sessions, {S.mouse_name.nunique()} mice'
          + (f'  ({len(missing)} embedding sessions had no states file)' if missing else ''))
    return S, lds


def main():
    S, lds = load()
    num = S.select_dtypes(include=[np.number]).columns
    M = S.groupby('mouse_name')[list(num)].mean()
    M['lab'] = S.groupby('mouse_name')['lab'].first()
    M['n_sessions'] = S.groupby('mouse_name').size()
    print(f'\nmouse level: {len(M)} mice, {M.n_sessions.sum()} sessions '
          f'(median {M.n_sessions.median():.0f}/mouse)')

    h, p = kruskal(*[g['LD1'].to_numpy() for _, g in M.groupby('lab')])
    print(f'LD1 ~ lab across mice: Kruskal H={h:.1f}, p={p:.4f}')

    # ------------------------------------------------ ceilings
    print('\n' + '=' * 78)
    print('RELIABILITY (split-half across sessions, Spearman-Brown) AND THE CEILING')
    print('=' * 78)
    rel_ld1, n_ld1 = split_half_reliability(S, 'LD1')
    print(f'LD1 reliability {rel_ld1:.3f}  (n={n_ld1} mice with >=4 sessions)')
    rel = {}
    for m in WHEEL_METRICS + PAW_METRICS:
        rel[m], _ = split_half_reliability(S, m)
    print(f'{"metric":26s} {"reliability":>11s} {"ceiling":>8s}')
    for m in WHEEL_METRICS + PAW_METRICS:
        print(f'{m:26s} {rel[m]:11.3f} {np.sqrt(max(rel[m], 0) * rel_ld1):8.3f}')

    # ------------------------------------------------ the tests
    noise = M['paw_hf_noise'].to_numpy()
    dur = M['duration_min'].to_numpy()
    lab = M['lab'].to_numpy()

    for fam, metrics in [('WHEEL  (independent of the LDA features)', WHEEL_METRICS),
                         ('PAW    (partly circular -- see header)', PAW_METRICS),
                         ('BY EPOCH', EPOCH_METRICS)]:
        print('\n' + '=' * 78)
        print(f'{fam}  vs LD1   (n = {len(M)} mice)')
        print('=' * 78)
        ld1 = M['LD1'].to_numpy()
        ld1_c = center_within(ld1, lab)
        rows = []
        for m in metrics:
            x = M[m].to_numpy()
            r, p = spearmanr(x, ld1, nan_policy='omit')
            rc, pc = spearmanr(center_within(x, lab), ld1_c, nan_policy='omit')
            rn, pn, _ = partial_spearman(x, ld1, noise)
            rd, pd_, _ = partial_spearman(x, ld1, dur)
            rows.append(dict(metric=m, rho=r, p=p, rho_lab=rc, p_lab=pc,
                             rho_noise=rn, p_noise=pn, rho_dur=rd,
                             ceiling=np.sqrt(max(rel.get(m, np.nan), 0) * rel_ld1)
                             if m in rel else np.nan))
        R = pd.DataFrame(rows)
        R['q'] = bh_fdr(R['p'])
        R['q_lab'] = bh_fdr(R['p_lab'])
        print(f'{"metric":26s} {"rho":>7s} {"p":>9s} {"q":>7s} | '
              f'{"rho|lab":>8s} {"p":>9s} {"q":>7s} | {"rho|noise":>9s} '
              f'{"rho|dur":>8s} | {"ceil":>5s} {"r/ceil":>6s}')
        for _, r in R.iterrows():
            star = '*' if r['q'] < 0.05 else ' '
            starl = '*' if r['q_lab'] < 0.05 else ' '
            rc = (f'{r["rho"] / r["ceiling"]:+.2f}'
                  if np.isfinite(r['ceiling']) and r['ceiling'] > 0 else '   -')
            ceil = f'{r["ceiling"]:.2f}' if np.isfinite(r['ceiling']) else '   -'
            print(f'{r["metric"]:26s} {r["rho"]:+7.3f} {r["p"]:9.4g} {r["q"]:7.3g}{star}'
                  f' | {r["rho_lab"]:+8.3f} {r["p_lab"]:9.4g} {r["q_lab"]:7.3g}{starl}'
                  f' | {r["rho_noise"]:+9.3f} {r["rho_dur"]:+8.3f} | {ceil:>5s} {rc:>6s}')
        if fam.startswith('WHEEL') or fam.startswith('PAW'):
            key = 'wheel' if fam.startswith('WHEEL') else 'paw'
            print('  co-primaries, permutation p (10k, mouse labels shuffled):')
            for m in CO_PRIMARY[key]:
                x = M[m].to_numpy()
                print(f'    {m:26s} raw p_perm={perm_p(x, M["LD1"].to_numpy()):.4f}   '
                      f'lab-centred p_perm='
                      f'{perm_p(center_within(x, lab), center_within(M["LD1"].to_numpy(), lab)):.4f}')

    # ------------------------------------------------ other LDs, for context
    print('\n' + '=' * 78)
    print('THE CO-PRIMARIES AGAINST LD1..LD8  (rho; * = p < 0.05 uncorrected)')
    print('=' * 78)
    picks = CO_PRIMARY['wheel'] + CO_PRIMARY['paw'] + ['paw_band_0.5_8', 'paw_LI_band']
    print(f'{"metric":26s} ' + ' '.join(f'{l:>9s}' for l in lds))
    for m in picks:
        cells = []
        for l in lds:
            r, p = spearmanr(M[m], M[l], nan_policy='omit')
            cells.append(f'{r:+.3f}{"*" if p < 0.05 else " "}')
        print(f'{m:26s} ' + ' '.join(f'{c:>9s}' for c in cells))

    M.to_csv(HERE / 'vigor_mouse_level.csv')
    print(f'\nwrote {HERE / "vigor_mouse_level.csv"}')


if __name__ == '__main__':
    main()
