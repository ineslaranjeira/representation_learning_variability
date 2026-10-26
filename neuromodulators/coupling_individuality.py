"""
Is NM-movement coupling an individual trait? (separate project from the individuality paper)
Functions only; the analysis and figures are in neuromodulators/9_coupling_individuality.ipynb.

Steps:
1. Coupling per fiber-session, from the caches of run_nm_syllable_effects.py: partial β (task covariates held
   fixed, both outcomes averaged) of still paw / whisking / licking, and the unique CV R² of all vigor and of whisker
   ME beyond the task variables (task_vs_vigor), each averaged over three epochs.
2. Trait-ness per NM, mice with >= MIN_SESSIONS sessions: ICC(1) (between-mouse share of the session variance) and
   split-half reliability of the mouse means (Spearman-Brown); null = session labels shuffled across mice.
3. Confound: recording day. Coupling can drift with training stage, and mice were recorded at different stages.
   Correlation of each metric with day_n within mice, and ICC again after removing each mouse-independent day trend.
   Signal quality: ICC after partialling a session-level proxy (task-variable CV R² after feedback).
   Fiber vs mouse: in mice with two fibers in the same sessions, between-fiber vs between-mouse variance.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from Functions import nm_syllables as ns  # noqa: E402

CACHE = ns.DATA / 'syllables_photometry'
OUT = ns.DATA / 'coupling_individuality'
OUT.mkdir(exist_ok=True)
NM_ORDER = ['DR-5HT', 'VTA-DA', 'SNc-DA', 'LC-NE', 'NBM-ACh']
TIME = np.arange(-30, 30) / 30
EPOCHS = {'pre-stim': ('stim', (-0.5, 0)), 'post-stim': ('stim', (0, 0.5)), 'post-fb': ('fb', (0, 1))}
FEATURES = ['paw 0', 'whisking', 'licking']
MIN_SESSIONS = 4
N_PERM = 1000
N_SPLITS = 200


def coupling_table():
    """One row per fiber-session, one column per metric."""
    eff = pd.read_pickle(CACHE / f'effects_task_{ns.RUN}.pkl')
    names = eff.attrs['features']
    rows = {}
    for _, r in eff.iterrows():
        key = (r['eid'], r['region'])
        d = rows.setdefault(key, {'eid': r['eid'], 'region': r['region'], 'subject': r['subject'],
                                  'target_NM': r['target_NM'], '_n': 0})
        d['_n'] += 1
        for ep, (w, (lo, hi)) in EPOCHS.items():
            m = (TIME >= lo) & (TIME < hi)
            for f in FEATURES:
                v = np.nanmean(r[f'beta_{w}'][names.index(f), m])
                d.setdefault(f'β {f} · {ep}', []).append(v)
    for d in rows.values():   # mean over the two outcomes
        for k in [k for k in d if k.startswith('β')]:
            d[k] = np.nanmean(d[k])
        del d['_n']
    tvv = pd.read_pickle(CACHE / f'task_vs_vigor_{ns.RUN}.pkl')
    M = tvv.attrs['models']
    for _, r in tvv.iterrows():
        d = rows[(r['eid'], r['region'])]
        for ep, (w, (lo, hi)) in EPOCHS.items():
            m = (TIME >= lo) & (TIME < hi)
            R = np.nanmean(r[f'r2_{w}'][:, m], 1)
            d[f'R² vigor · {ep}'] = R[2] - R[0]
            d[f'R² whisker ME · {ep}'] = R[2] - R[M.index('without whisker ME')]
    tab = pd.DataFrame(rows.values())
    rec = ns.load_recordings()[['eid', 'brain_region', 'day_n', 'session_n', 'session_type']]
    return tab.merge(rec.rename(columns={'brain_region': 'region'}), on=['eid', 'region'], how='left')


def icc1(values, groups):
    """One-way random-effects ICC(1) for unbalanced groups."""
    df = pd.DataFrame({'v': values, 'g': groups}).dropna()
    k = df.groupby('g').size()
    n, N = len(k), len(df)
    if n < 3:
        return np.nan
    grand = df['v'].mean()
    msb = (k * (df.groupby('g')['v'].mean() - grand) ** 2).sum() / (n - 1)
    msw = ((df['v'] - df.groupby('g')['v'].transform('mean')) ** 2).sum() / (N - n)
    k0 = (N - (k ** 2).sum() / N) / (n - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def split_half(values, groups, rng, n_splits=N_SPLITS):
    """Mean over random splits of corr(mouse means of half A, of half B), Spearman-Brown corrected."""
    df = pd.DataFrame({'v': values, 'g': groups}).dropna()
    rs = []
    for _ in range(n_splits):
        a, b = [], []
        for _, g in df.groupby('g'):
            idx = rng.permutation(len(g))
            h = len(g) // 2
            a.append(g['v'].iloc[idx[:h]].mean()); b.append(g['v'].iloc[idx[h:2 * h]].mean())
        r = np.corrcoef(a, b)[0, 1]
        rs.append(2 * r / (1 + r))
    return np.nanmean(rs)


def trait_stats(tab, metrics, detrend_day=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for nm in NM_ORDER:
        g = tab[tab['target_NM'] == nm]
        g = g[g.groupby('subject')['eid'].transform('size') >= MIN_SESSIONS]
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            if detrend_day:   # remove a common linear trend in recording day (fitted within mice)
                dm = g['day_n'] - g.groupby('subject')['day_n'].transform('mean')
                vm = g[m] - g.groupby('subject')[m].transform('mean')
                ok = np.isfinite(dm) & np.isfinite(vm)
                slope = np.polyfit(dm[ok], vm[ok], 1)[0] if ok.sum() > 5 else 0.
                v = v - slope * g['day_n'].to_numpy()
            icc = icc1(v, g['subject'].to_numpy())
            null = np.array([icc1(v, rng.permutation(g['subject'].to_numpy())) for _ in range(N_PERM)])
            rows.append({'NM': nm, 'metric': m, 'mice': g['subject'].nunique(), 'sessions': len(g), 'ICC': icc,
                         'p (perm)': (np.sum(null >= icc) + 1) / (N_PERM + 1),
                         'split-half': split_half(v, g['subject'].to_numpy(), rng)})
    return pd.DataFrame(rows)


def day_drift(tab, metrics):
    """Within-mouse Spearman r of each metric with recording day, mean over mice (≥ MIN_SESSIONS sessions)."""
    from scipy.stats import spearmanr, ttest_1samp
    rows = []
    for nm in NM_ORDER:
        g = tab[tab['target_NM'] == nm]
        g = g[g.groupby('subject')['eid'].transform('size') >= MIN_SESSIONS]
        for m in metrics:
            r = g.groupby('subject').apply(
                lambda x: spearmanr(x['day_n'], x[m], nan_policy='omit')[0] if x[m].notna().sum() >= 3 else np.nan).dropna()
            rows.append({'NM': nm, 'metric': m, 'r with day': r.mean(),
                         'p': ttest_1samp(r, 0).pvalue if len(r) > 2 else np.nan})
    return pd.DataFrame(rows)


def add_task_r2(tab):
    """Adds the task-variable CV R² per epoch ('R² task · <epoch>'), a session-level signal-quality proxy."""
    tvv = pd.read_pickle(CACHE / f'task_vs_vigor_{ns.RUN}.pkl')
    vals = {}
    for _, r in tvv.iterrows():
        for ep, (w, (lo, hi)) in EPOCHS.items():
            m = (TIME >= lo) & (TIME < hi)
            vals.setdefault(f'R² task · {ep}', {})[(r['eid'], r['region'])] = np.nanmean(r[f'r2_{w}'][0, m])
    key = list(zip(tab['eid'], tab['region']))
    return tab.assign(**{c: [v.get(k, np.nan) for k in key] for c, v in vals.items()})


def partial_out(tab, metrics, proxy, nm_col='target_NM'):
    """Each metric minus its linear fit on the proxy, across sessions within each NM."""
    out = tab.copy()
    for nm, g in tab.groupby(nm_col):
        for m in metrics:
            ok = g[m].notna() & g[proxy].notna()
            if ok.sum() > 5:
                out.loc[g.index[ok], m] = g.loc[ok, m] - np.polyval(np.polyfit(g.loc[ok, proxy], g.loc[ok, m], 1),
                                                                  g.loc[ok, proxy])
    return out


def fiber_vs_mouse(tab, metrics):
    """Mice with two fibers in the same sessions: per metric, the fiber means and the variance between mice
    (of the mouse means) vs between the two fibers of a mouse (mean over mice of their half squared difference)."""
    two = tab.groupby('eid')['region'].transform('nunique') > 1
    g = tab[two]
    rows = []
    for m in metrics:
        fm = g.groupby(['subject', 'region'])[m].mean()
        per_mouse = fm.groupby('subject')
        rows.append({'metric': m, 'between mice': per_mouse.mean().var(),
                     'between fibers of a mouse': per_mouse.apply(lambda x: x.var()).mean()})
    return g, pd.DataFrame(rows)
