"""
Does the NM signal during a syllable predict how often that syllable is used afterwards (Markowitz et al. 2023)?

    python neuromodulators/syllable_reinforcement.py fit [NMs, default VTA-DA,SNc-DA] [n_future=10] [n_jobs=12]
    python neuromodulators/syllable_reinforcement.py summary [n_future=10]

Per fiber-session, on nm_syllables' masked windows (stimulus and feedback, ±1 s):
  e(t, s)   NM surprise during syllable s on trial t: photometry minus that sample's mean over the session's trials
            with the same outcome and contrast (removes the stimulus / reward responses), averaged over the samples
            where the trial is in s, z-scored within the session.
  u(t, s)   use of s on trial t: fraction of the trial's valid window samples in s.
  Δ(t, s')  change in use of s': mean u over the next n_future trials minus the mean over the previous n_future,
            z-scored per syllable within the session.
For every trial t, every syllable s present on t and every common syllable s', one row; the regression
    Δ(t, s') ~ e(t, s) + e(t, s) x same + u(t, s') + reward(t) + trial position + syllable s' (fixed effect)
gives β_cross (e's effect on the other syllables' use: a global state shift) and β_self = β_cross + β_same (on the
syllable it accompanied). β_same > 0 is syllable-specific reinforcement. Null: e(t, s) shuffled across trials
within each syllable (null='shuffle') or rotated (null='circular', default: keeps slow drift), n_shuffles times.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from Functions import nm_syllables as ns  # noqa: E402

OUT = ns.DATA / 'syllable_reinforcement'
OUT.mkdir(exist_ok=True)
MIN_USE = 0.01        # a syllable is "common" in a session above this fraction of samples
MIN_PRESENT = 20      # and present on at least this many trials (needed for e(t, s) and its shuffle)
N_SHUFFLES = 100


def session_tables(fiber, windows=ns.WINDOWS):
    """(e: trials x syllables NM surprise, u: trials x syllables use, meta, syllables). windows: where e is taken
    (use is always over both windows); ('stim',) keeps only the pre-feedback window, i.e. before the outcome."""
    meta = fiber['meta']
    resid, codes = [], []
    for w in ns.WINDOWS:
        P = fiber[w]['photometry'].copy()
        for _, idx in meta.groupby(['correct', 'contrast']).indices.items():
            P[idx] -= np.nanmean(P[idx], 0)
        if w not in windows:
            P[:] = np.nan
        resid.append(P)
        codes.append(fiber[w]['code'])
    R, C = np.hstack(resid), np.hstack(codes)
    R[np.isnan(C)] = np.nan
    C_e = np.where(np.isnan(R), np.nan, C)   # samples where e can be read
    valid = ~np.isnan(C)
    syl = np.unique(C[valid]).astype(int)
    use_total = np.array([(C == s).sum() for s in syl]) / valid.sum()
    present = np.array([((C_e == s).any(1)).sum() for s in syl])
    syl = syl[(use_total >= MIN_USE) & (present >= MIN_PRESENT)]
    u = np.stack([(C == s).sum(1) / np.maximum(valid.sum(1), 1) for s in syl], 1)
    u[valid.sum(1) == 0] = np.nan
    with np.errstate(invalid='ignore'):
        e = np.stack([np.nanmean(np.where(C_e == s, R, np.nan), 1) for s in syl], 1)
    e = (e - np.nanmean(e)) / np.nanstd(e)
    return e, u, meta, syl


def change_in_use(u, n):
    """Δ(t, s): mean use over trials t+1..t+n minus over t-n..t-1, z-scored per syllable; NaN at the edges."""
    c = np.vstack([np.zeros((1, u.shape[1])), np.nancumsum(u, 0)])
    T = len(u)
    d = np.full(u.shape, np.nan)
    for t in range(n, T - n):
        d[t] = (c[t + n + 1] - c[t + 1]) / n - (c[t] - c[t - n]) / n
    return (d - np.nanmean(d, 0)) / np.nanstd(d, 0)


def design(e, u, d, meta):
    """Long table: one row per (trial t, syllable s present on t, target syllable s')."""
    T, S = e.shape
    t, s = np.nonzero(~np.isnan(e))
    tt, ss = np.repeat(t, S), np.repeat(s, S)
    sp = np.tile(np.arange(S), len(t))
    return pd.DataFrame({'t': tt, 's': ss, 'sp': sp, 'e': e[tt, ss], 'same': (ss == sp).astype(float),
                         'u_target': u[tt, sp], 'delta': d[tt, sp],
                         'reward': meta['correct'].to_numpy()[tt], 'trial_frac': meta['trial_frac'].to_numpy()[tt]})


def fit_betas(X):
    """OLS of delta on e, e x same and controls (+ target-syllable dummies): (β_cross, β_same)."""
    X = X.dropna()
    D = pd.get_dummies(X['sp'], prefix='sp', drop_first=True, dtype=float)
    A = np.column_stack([np.ones(len(X)), X['e'], X['e'] * X['same'], X['same'], X['u_target'], X['reward'],
                         X['trial_frac'], D.to_numpy()])
    b = np.linalg.lstsq(A, X['delta'].to_numpy(), rcond=None)[0]
    return b[1], b[2]


def one_fiber(fiber, n_future, null='circular', windows=ns.WINDOWS, seed=0):
    """null: 'shuffle' permutes e(., s) across the trials where s occurred; 'circular' rotates that sequence by a
    random offset, which keeps its slow drift and autocorrelation (stricter: slow co-drift of the NM signal and of
    syllable use is then in the null too)."""
    e, u, meta, syl = session_tables(fiber, windows)
    if len(syl) < 3:
        return None
    d = change_in_use(u, n_future)
    b_cross, b_same = fit_betas(design(e, u, d, meta))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N_SHUFFLES):
        es = e.copy()
        for k in range(e.shape[1]):   # shuffle e(., s) across the trials where s occurred
            idx = np.nonzero(~np.isnan(e[:, k]))[0]
            src = rng.permutation(idx) if null == 'shuffle' else np.roll(idx, rng.integers(1, len(idx)))
            es[idx, k] = e[src, k]
        null.append(fit_betas(design(es, u, d, meta)))
    null = np.array(null)
    return {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'subject': meta['subject'].iat[0],
            'target_NM': meta['target_NM'].iat[0], 'n_trials': len(meta), 'n_syllables': len(syl),
            'beta_cross': b_cross, 'beta_same': b_same, 'beta_self': b_cross + b_same,
            'null_cross': null[:, 0], 'null_same': null[:, 1], 'null_self': null.sum(1)}


def one_session(eid, n_future, null, windows):
    warnings.filterwarnings('ignore')
    try:
        return [r for f in ns.load_session(eid, movement=False)
                if (r := one_fiber(f, n_future, null, windows)) is not None]
    except Exception as ex:
        print(f'{eid}: {type(ex).__name__}: {ex}', flush=True)
        return []


def tag(n_future, null, windows):
    return f"n{n_future}_{null}_{'+'.join(windows)}"


def fit(nms, n_future=10, n_jobs=12, null='circular', windows=ns.WINDOWS):
    from joblib import Parallel, delayed
    rec = ns.load_recordings()
    eids = rec.loc[rec['target_NM'].isin(nms), 'eid'].unique()
    print(f'{len(eids)} sessions of {nms}, n_future = {n_future}', flush=True)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_session)(e, n_future, null, windows) for e in eids)
    table = pd.DataFrame([r for s in res for r in s])
    path = OUT / f'betas_{tag(n_future, null, windows)}_{ns.RUN}.pkl'
    if path.exists():   # keep the other NMs already fitted
        old = pd.read_pickle(path)
        table = pd.concat([old[~old['target_NM'].isin(nms)], table], ignore_index=True)
    table.to_pickle(path)
    print(table.groupby('target_NM').size().to_dict())


def summary(n_future=10, null='circular', windows=ns.WINDOWS, show=True):
    from scipy.stats import ttest_1samp
    tab = pd.read_pickle(OUT / f'betas_{tag(n_future, null, windows)}_{ns.RUN}.pkl')
    for k in ('cross', 'same', 'self'):   # each session's β as a z against its own shuffle null
        tab[f'z_{k}'] = (tab[f'beta_{k}'] - tab[f'null_{k}'].apply(np.mean)) / tab[f'null_{k}'].apply(np.std)
    rows = []
    for nm, g in tab.groupby('target_NM'):
        mice = g.groupby('subject')[[f'{p}_{k}' for p in ('beta', 'z') for k in ('self', 'cross', 'same')]].mean()
        row = {'NM': nm, 'mice': len(mice), 'sessions': len(g)}
        for k in ('self', 'cross', 'same'):
            b, z = mice[f'beta_{k}'], mice[f'z_{k}']
            row[f'β_{k}'] = f'{b.mean():+.4f} ({(b > 0).sum()}/{len(b)}) p={ttest_1samp(b, 0).pvalue:.3f}'
            row[f'z_{k}'] = f'{z.mean():+.2f}'
        row['sessions |z_same| > 1.96'] = f"{(g['z_same'] > 1.96).sum()} up / {(g['z_same'] < -1.96).sum()} down"
        rows.append(row)
    out = pd.DataFrame(rows).set_index('NM').T
    if show:
        pd.set_option('display.width', 250)
        print(f'{tag(n_future, null, windows)}: per mouse mean over sessions; p = t-test across mice; '
              f'z = β vs its own null')
        print(out.to_string())
    return out


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    # fit [NMs] [n_future] [n_jobs] [null] [windows, e.g. stim or stim,fb]; summary [n_future] [null] [windows]
    a = sys.argv
    if stage == 'fit':
        fit(a[2].split(',') if len(a) > 2 else ['VTA-DA', 'SNc-DA'], int(a[3]) if len(a) > 3 else 10,
            int(a[4]) if len(a) > 4 else 12, a[5] if len(a) > 5 else 'circular',
            tuple(a[6].split(',')) if len(a) > 6 else ns.WINDOWS)
    else:
        summary(int(a[2]) if len(a) > 2 else 10, a[3] if len(a) > 3 else 'circular',
                tuple(a[4].split(',')) if len(a) > 4 else ns.WINDOWS)
