"""
Neural correlates of the BWM LD1 within neuromodulator mice, session to session.
=================================================================================
LD1 per session = the session's projection into the BWM proficient 1-camera embedding (nm_projection). The neural
measures are per FIBER-session (a mouse can carry two fibers):

  photometry (z-scored GCaMP snippets of the export, nm_syllables.load_session; masks as there)
    baseline      mean in [-0.5, 0) s before stimulus onset
    stim          mean in [0.1, 0.4) s after stimulus onset minus the trial's baseline; all go trials
    stim_slow     the same on trials whose first movement comes after the window (rt > 0.4 s), so movement-locked
                  DA cannot fill it (>= 10 such trials)
    reward        mean in [0.1, 0.6) s after feedback minus [-0.2, 0) before it; correct trials
    error         the same on error trials
    reward_hard   reward on |contrast| <= 0.125 minus reward on |contrast| >= 0.5 (a prediction-error signature:
                  reward is less expected on hard trials)
                  CAVEAT: reward / error subtract [-0.2, 0) s before feedback; signals that already rise before
                  feedback (VTA DA, LC NE) then change sign / size with that rise -- check the traces
  movement coupling (coupling_individuality.coupling_table, from the photometry-syllable caches; NM's own
  syllables): partial beta of still paw / whisking / licking and unique vigor R², per epoch

within_fiber(): per measure, the correlation of LD1 with it after removing each fiber's mean (optionally also a
linear recording-day trend and the protocol, within fiber); p from permuting LD1 among the sessions of each fiber
(after the same residualisation, Freedman-Lane). Fibers with >= MIN_SESSIONS sessions.
"""
import sys
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in [REPO, REPO / 'neuromodulators']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from Functions import nm_syllables as ns          # noqa: E402

import nm_lda as n                                # noqa: E402

OUT = n.NM_DATA / 'ld1_neural'
MIN_SESSIONS = 5
WIN = dict(baseline=('stim', (-0.5, 0.0)), stim=('stim', (0.1, 0.4)), fb=('fb', (0.1, 0.6)), pre_fb=('fb', (-0.2, 0.0)))


def _mean(P, time, lo, hi):
    m = (time >= lo) & (time < hi)
    with np.errstate(invalid='ignore'):
        return np.nanmean(P[:, m], axis=1)


def fiber_session_measures(eid):
    """One row per fiber of a session."""
    rows = []
    for f in ns.load_session(eid, movement=False):
        t, meta = f['time'], f['meta']
        base = _mean(f['stim']['photometry'], t, *WIN['baseline'][1])
        stim = _mean(f['stim']['photometry'], t, *WIN['stim'][1]) - base
        fb = _mean(f['fb']['photometry'], t, *WIN['fb'][1]) - _mean(f['fb']['photometry'], t, *WIN['pre_fb'][1])
        cor = meta['correct'].to_numpy() == 1
        c = meta['contrast'].abs().to_numpy()
        rows.append(dict(eid=eid, region=f['region'], subject=meta['subject'].iloc[0],
                         target_NM=meta['target_NM'].iloc[0], n_trials=len(meta),
                         baseline=np.nanmean(base), stim=np.nanmean(stim),
                         stim_slow=np.nanmean(stim[slow]) if (slow := (meta['rt'].to_numpy() > 0.4)).sum() >= 10 else np.nan,
                         frac_slow=float((meta['rt'].to_numpy() > 0.4).mean()),
                         reward=np.nanmean(fb[cor]), error=np.nanmean(fb[~cor]) if (~cor).sum() >= 5 else np.nan,
                         reward_hard=(np.nanmean(fb[cor & (c <= 0.125)]) - np.nanmean(fb[cor & (c >= 0.5)])
                                      if (cor & (c <= 0.125)).sum() >= 5 and (cor & (c >= 0.5)).sum() >= 5 else np.nan)))
    return rows


def photometry_measures(eids, name='photometry_measures', cache=True):
    """fiber_session_measures for every eid; the cache grows (sessions already in it are not recomputed)."""
    f = OUT / f'{name}.csv'
    have = pd.read_csv(f) if (cache and f.exists()) else pd.DataFrame(columns=['eid'])
    rows = []
    for e in [e for e in eids if e not in set(have['eid'])]:
        try:
            rows.extend(fiber_session_measures(e))
        except Exception as ex:                       # noqa: BLE001
            print(f'  {e}: {type(ex).__name__}: {ex}')
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        OUT.mkdir(parents=True, exist_ok=True)
        have.to_csv(f, index=False)
    return have[have['eid'].isin(eids)].reset_index(drop=True)


def coupling_measures():
    import coupling_individuality as ci
    return ci.coupling_table()


def session_table(A, targets=('VTA-DA', 'SNc-DA')):  # noqa: C901
    """Fiber-session rows of the given targets: LD1-3 of the session, recording day, protocol, the photometry and
    coupling measures, and the session's behavioural measures (nm_lda.session_measures) for the sanity check."""
    N = A[A['group'] == 'NM']
    rec = pd.read_csv(ns.ARCHIVE / 'recordings.csv').drop_duplicates(['eid', 'brain_region'])
    rec = rec[rec['target_NM'].isin(targets) & rec['eid'].isin(N.index)]
    ph = photometry_measures(sorted(rec['eid'].unique()))
    cp = coupling_measures().drop(columns=['subject', 'target_NM', 'session_type'], errors='ignore')
    T = (ph.merge(cp, on=['eid', 'region'], how='left', suffixes=('', '_cp'))
           .merge(N[['LD1', 'LD2', 'LD3', 'protocol']], left_on='eid', right_index=True))
    T = T.merge(rec[['eid', 'brain_region', 'day_n']].rename(columns={'brain_region': 'region'}),
                on=['eid', 'region'], how='left', suffixes=('_cp', ''))
    T['fiber'] = T['subject'] + ' ' + T['region']
    beh = n.session_measures(T['eid'].unique())[['log_reaction', 'wheel_gate', 'paw_gate', 'wheel_speed_Quiescence',
                                                 'switches', 'correct']]
    T = T.join(beh, on='eid')
    counts = T.groupby('fiber')['eid'].transform('size')
    return T[counts >= MIN_SESSIONS].reset_index(drop=True)


def _resid(v, D):
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]


def within_fiber(T, measures, x='LD1', controls=(), n_perm=5000, seed=0, unit='fiber'):
    """Correlation of x with each measure within fibers. controls: 'day' (linear day trend per fiber), 'protocol'
    (fixed effect), and/or any column of T (a session covariate, e.g. 'log_reaction', 'correct'; one slope pooled
    over fibers, after the fiber means). -> one row per measure: r, p (within-fiber permutation), n.
    unit='mouse': for mice with two fibers in the same sessions (bilateral), the residuals are averaged over a
    mouse's fibers per session and LD1 is permuted within mouse, so a session is not counted twice."""
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(seed)
    rows = []
    for y in measures:
        covs = [c for c in controls if c not in ('day', 'protocol')]
        d = T[['fiber', 'subject', x, y, 'day_n', 'protocol'] + covs].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d.groupby('fiber')[x].transform('size') >= MIN_SESSIONS]
        if d['fiber'].nunique() < 2:
            continue
        F = pd.get_dummies(d['fiber']).to_numpy(float)
        cols = [F]
        if 'day' in controls:
            cols.append(F * (d['day_n'].to_numpy(float) - d.groupby('fiber')['day_n'].transform('mean').to_numpy())[:, None])
        if 'protocol' in controls and d['protocol'].nunique() > 1:
            cols.append((d['protocol'] == 'ephys').to_numpy(float)[:, None])
        cols += [d[c].to_numpy(float)[:, None] for c in covs]
        D = np.column_stack(cols)
        ex, ey = _resid(d[x].to_numpy(float), D), _resid(d[y].to_numpy(float), D)
        g_col = 'fiber'
        if unit == 'mouse':
            m = pd.DataFrame({'subject': d['subject'].to_numpy(), 'eid': T.loc[d.index, 'eid'].to_numpy(), 'ex': ex, 'ey': ey})
            m = m.groupby(['subject', 'eid'], sort=False)[['ex', 'ey']].mean().reset_index()
            ex, ey = m['ex'].to_numpy(), m['ey'].to_numpy()
            d_units = m['subject'].to_numpy()
        else:
            d_units = d[g_col].to_numpy()
        r = np.corrcoef(ex, ey)[0, 1]
        groups = [np.flatnonzero(d_units == f) for f in pd.unique(d_units)]
        null = np.empty(n_perm)
        for k in range(n_perm):
            e = ex.copy()
            for g in groups:
                e[g] = rng.permutation(e[g])
            null[k] = np.corrcoef(e, ey)[0, 1]
        rows.append(dict(measure=y, r=r, p=(1 + np.sum(np.abs(null) >= abs(r))) / (1 + n_perm),
                         n_sessions=len(ex), n_fibers=d['fiber'].nunique(), n_mice=d['subject'].nunique()))
    t = pd.DataFrame(rows)
    if len(t):
        t['q'] = multipletests(t['p'], method='fdr_bh')[1]
    return t


# ------------------------------------------------------------------------------------------------
# Mean traces per fiber-session, for the LD1-binned figures
# ------------------------------------------------------------------------------------------------
TRACE_MOVEMENT = ['wheel speed', 'paw speed', 'whisker ME']


def fiber_session_traces(eid):
    """Per fiber: mean photometry around stimulus (all trials; rt > 0.4 s; rt <= 0.4 s) and feedback (correct;
    error), and mean movement around the stimulus. Masked samples (the neighbouring epoch) are NaN, as in
    nm_syllables.load_session."""
    out = {}
    for f in ns.load_session(eid, movement=True):
        meta = f['meta']
        rt = meta['rt'].to_numpy()
        cor = meta['correct'].to_numpy() == 1
        S, Fb = f['stim']['photometry'], f['fb']['photometry']
        with np.errstate(invalid='ignore'):
            d = {'time': f['time'], 'stim_all': np.nanmean(S, 0),
                 'stim_slow': np.nanmean(S[rt > 0.4], 0) if (rt > 0.4).sum() >= 10 else np.full(len(f['time']), np.nan),
                 'stim_fast': np.nanmean(S[rt <= 0.4], 0) if (rt <= 0.4).sum() >= 10 else np.full(len(f['time']), np.nan),
                 'fb_correct': np.nanmean(Fb[cor], 0),
                 'fb_error': np.nanmean(Fb[~cor], 0) if (~cor).sum() >= 5 else np.full(len(f['time']), np.nan)}
            for m in TRACE_MOVEMENT:
                d[f'stim {m}'] = np.nanmean(np.abs(f['stim']['movement'][m]), 0)
        out[(eid, f['region'])] = d
    return out


def traces(eids, name='traces', cache=True):
    import pickle
    f = OUT / f'{name}.pkl'
    have = pickle.load(open(f, 'rb')) if (cache and f.exists()) else {}
    todo = [e for e in eids if not any(k[0] == e for k in have)]
    for e in todo:
        try:
            have.update(fiber_session_traces(e))
        except Exception as ex:                       # noqa: BLE001
            print(f'  {e}: {type(ex).__name__}: {ex}')
    if todo:
        OUT.mkdir(parents=True, exist_ok=True)
        pickle.dump(have, open(f, 'wb'))
    return have


def ld1_bins(T, n_bins=3, x='LD1'):
    """Within-fiber LD1 bin of every fiber-session (0 = lowest third of that fiber's sessions)."""
    return T.groupby('fiber')[x].transform(lambda s: pd.qcut(s.rank(method='first'), n_bins, labels=False))


def binned_traces(T, tr, key, n_bins=3, unit='fiber'):
    """(time, mean, sem) per LD1 bin: sessions averaged within fiber and bin, then units averaged (unit = 'fiber',
    or 'subject' to average a bilateral mouse's fibers first, so it counts once)."""
    T = T.assign(bin=ld1_bins(T, n_bins))
    per_fiber = {}
    for (fib, b), g in T.groupby([unit, 'bin']):
        X = np.stack([tr[(e, r)][key] for e, r in zip(g['eid'], g['region']) if (e, r) in tr])
        per_fiber[(fib, b)] = np.nanmean(X, 0)
    time = next(iter(tr.values()))['time']
    out = {}
    for b in range(n_bins):
        X = np.stack([v for (f, bb), v in per_fiber.items() if bb == b])
        out[b] = (np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0).clip(1)))
    return time, out


def fiber_slopes(T, measures, controls=('day', 'protocol', 'log_reaction', 'correct')):
    """Per fiber: the within-fiber regression slope of each measure on LD1 (both standardised within fiber), after
    the controls -- one number per fiber, so targets can be compared with fibers as units."""
    rows = []
    for fib, g in T.groupby('fiber'):
        for y in measures:
            covs = [c for c in controls if c not in ('day', 'protocol')]
            d = g[['LD1', y, 'day_n', 'protocol'] + covs].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < MIN_SESSIONS:
                continue
            D = [np.ones(len(d))]
            if 'day' in controls:
                D.append(d['day_n'].to_numpy(float))
            if 'protocol' in controls and d['protocol'].nunique() > 1:
                D.append((d['protocol'] == 'ephys').to_numpy(float))
            D = np.column_stack(D + [d[c].to_numpy(float) for c in covs])
            if D.shape[1] >= len(d) - 1:
                continue
            ex, ey = _resid(d['LD1'].to_numpy(float), D), _resid(d[y].to_numpy(float), D)
            rows.append(dict(fiber=fib, target_NM=g['target_NM'].iloc[0], measure=y,
                             r=np.corrcoef(ex, ey)[0, 1] if ex.std() > 0 and ey.std() > 0 else np.nan, n=len(d)))
    return pd.DataFrame(rows)


def target_difference(slopes, a='VTA-DA', b='SNc-DA'):
    """Per measure: mean per-fiber r in each target and the difference, p exact over every split of the fibers
    into two groups of the observed sizes (fibers are the units)."""
    from itertools import combinations
    rows = []
    for y, g in slopes.dropna(subset=['r']).groupby('measure', sort=False):
        r, t = g['r'].to_numpy(), g['target_NM'].to_numpy()
        na = int((t == a).sum())
        if na < 2 or (t == b).sum() < 2:
            continue
        obs = r[t == a].mean() - r[t == b].mean()
        null = [r[list(c)].mean() - np.delete(r, list(c)).mean() for c in combinations(range(len(r)), na)]
        rows.append(dict(measure=y, **{f'r {a}': r[t == a].mean(), f'r {b}': r[t == b].mean()}, diff=obs,
                         p=float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), n_fibers=f'{na} vs {(t == b).sum()}'))
    return pd.DataFrame(rows)


def target_interaction(T, measures, a='VTA-DA', controls=('day', 'protocol', 'log_reaction', 'correct'), n_perm=5000,
                       seed=0):
    """Does the within-fiber LD1 slope differ between targets? After within_fiber's residualisation (fiber means,
    per-fiber day trend, protocol and covariate slopes PER TARGET), regress the measure on LD1 and LD1 x [target == a];
    p for the interaction from permuting LD1 within fibers. Slopes are per within-fiber SD of LD1 and the measure."""
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(seed)
    rows = []
    for y in measures:
        covs = [c for c in controls if c not in ('day', 'protocol')]
        d = T[['fiber', 'target_NM', 'LD1', y, 'day_n', 'protocol'] + covs].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d.groupby('fiber')['LD1'].transform('size') >= MIN_SESSIONS]
        F = pd.get_dummies(d['fiber']).to_numpy(float)
        cols = [F]
        if 'day' in controls:
            cols.append(F * (d['day_n'] - d.groupby('fiber')['day_n'].transform('mean')).to_numpy(float)[:, None])
        if 'protocol' in controls and d['protocol'].nunique() > 1:
            cols.append((d['protocol'] == 'ephys').to_numpy(float)[:, None])
        isa = (d['target_NM'] == a).to_numpy(float)
        # covariate slopes per target (a pooled slope, estimated mostly from the larger target, would leave the
        # smaller one mis-residualised); protocol per target too
        D = np.column_stack(cols + [d[c].to_numpy(float)[:, None] * m[:, None] for c in covs for m in (isa, 1 - isa)]
                            + ([(d['protocol'] == 'ephys').to_numpy(float)[:, None] * isa[:, None]]
                               if 'protocol' in controls and d['protocol'].nunique() > 1 else []))
        ex, ey = _resid(d['LD1'].to_numpy(float), D), _resid(d[y].to_numpy(float), D)
        ex, ey = ex / ex.std(), ey / ey.std()

        def fit(x):
            X = np.column_stack([x, x * isa])
            return np.linalg.lstsq(X, ey, rcond=None)[0]
        b = fit(ex)
        groups = [np.flatnonzero(d['fiber'].to_numpy() == f) for f in d['fiber'].unique()]
        null = np.empty(n_perm)
        for k in range(n_perm):
            e = ex.copy()
            for g in groups:
                e[g] = rng.permutation(e[g])
            null[k] = fit(e)[1]
        rows.append(dict(measure=y, slope_other=b[0], slope_a=b[0] + b[1], interaction=b[1],
                         p=(1 + np.sum(np.abs(null) >= abs(b[1]))) / (1 + n_perm), n_sessions=len(d)))
    t = pd.DataFrame(rows).rename(columns={'slope_a': f'slope {a}', 'slope_other': 'slope other'})
    t['q'] = multipletests(t['p'], method='fdr_bh')[1]
    return t


# ------------------------------------------------------------------------------------------------
# How much of the neural (trace) variability does LD1 explain?
# ------------------------------------------------------------------------------------------------
NEURAL_KEYS = ['stim_all', 'fb_correct', 'fb_error']
MOVEMENT_KEYS = ['stim wheel speed', 'stim whisker ME']


def trace_matrix(T, tr, keys, level='mouse', predictors=('LD1',)):
    """Rows = units, columns = the keys' mean traces concatenated in time.
    level 'mouse': sessions averaged per fiber, fibers per mouse; predictors = the mouse means of the session values.
    level 'session': one row per fiber-session.
    -> (Y [units x samples], meta frame with unit, target_NM, fiber, the predictors)."""
    T = T[[(e, r) in tr for e, r in zip(T['eid'], T['region'])]]
    vec = np.stack([np.concatenate([tr[(e, r)][k] for k in keys]) for e, r in zip(T['eid'], T['region'])])
    meta = T[['subject', 'fiber', 'target_NM', 'eid'] + list(predictors)].reset_index(drop=True)
    if level == 'session':
        return vec, meta.rename(columns={'subject': 'unit'})
    fib = pd.DataFrame(vec).groupby(meta['fiber'].to_numpy()).mean()
    fmeta = meta.groupby('fiber').agg(subject=('subject', 'first'), target_NM=('target_NM', 'first'),
                                      **{p: (p, 'mean') for p in predictors})
    Y = fib.groupby(fmeta.loc[fib.index, 'subject'].to_numpy()).mean()
    m = fmeta.groupby('subject').agg(target_NM=('target_NM', 'first'), **{p: (p, 'mean') for p in predictors})
    return Y.to_numpy(), m.loc[Y.index].reset_index().rename(columns={'subject': 'unit'})


def explained(Y, x, groups, n_perm=5000, seed=0):
    """Share of the within-group variance of Y (units x samples) explained by x with a separate slope per group
    (Y and x centred within group). Columns with any NaN are dropped. Null: x permuted within group.
    -> dict: share, null mean / 95th pct, p, per-sample R² (curve), per-group share."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(Y).all(0)
    Y = Y[:, ok]
    groups = np.asarray(groups)
    idx = [np.flatnonzero(groups == g) for g in pd.unique(groups)]
    Yc = Y.copy()
    for i in idx:
        Yc[i] -= Yc[i].mean(0)
    sst_t = (Yc ** 2).sum(0)

    def ss_exp(xv):
        out = np.zeros(Y.shape[1])
        for i in idx:
            xc = xv[i] - xv[i].mean()
            if xc @ xc > 0:
                out += (xc @ Yc[i]) ** 2 / (xc @ xc)
        return out
    obs_t = ss_exp(np.asarray(x, float))
    share = obs_t.sum() / sst_t.sum()
    null = np.empty(n_perm)
    for k in range(n_perm):
        xp = np.asarray(x, float).copy()
        for i in idx:
            xp[i] = rng.permutation(xp[i])
        null[k] = ss_exp(xp).sum() / sst_t.sum()
    per_group = {}
    for g, i in zip(pd.unique(groups), idx):
        xc = np.asarray(x, float)[i] - np.asarray(x, float)[i].mean()
        per_group[g] = float(((xc @ Yc[i]) ** 2 / (xc @ xc)).sum() / (Yc[i] ** 2).sum()) if xc @ xc > 0 else np.nan
    curve = np.full(len(ok), np.nan)
    curve[ok] = obs_t / sst_t
    return dict(share=share, null_mean=null.mean(), null_95=np.percentile(null, 95),
                p=(1 + np.sum(null >= share)) / (1 + n_perm), curve=curve, per_group=per_group)


def unique_explained(Y, X, x, groups, n_perm=2000, seed=0):
    """Unique share of x beyond the columns of X (per-group OLS, everything centred within group): share(X + x) -
    share(X). Null: x permuted within group, X kept. -> dict like explained (no curve)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(Y).all(0)
    Y = Y[:, ok]
    groups = np.asarray(groups)
    idx = [np.flatnonzero(groups == g) for g in pd.unique(groups)]
    Yc, Xc = Y.copy(), np.asarray(X, float).copy()
    for i in idx:
        Yc[i] -= Yc[i].mean(0)
        Xc[i] -= Xc[i].mean(0)
    sst = (Yc ** 2).sum()

    def ss(Z):
        out = 0.0
        for i in idx:
            if len(i) <= Z.shape[1] + 1:
                continue
            B = np.linalg.lstsq(Z[i], Yc[i], rcond=None)[0]
            out += ((Z[i] @ B) ** 2).sum()
        return out
    base = ss(Xc)

    def full(xv):
        xv = xv.copy()
        for i in idx:
            xv[i] -= xv[i].mean()
        return ss(np.column_stack([Xc, xv])) - base
    obs = full(np.asarray(x, float)) / sst
    null = np.empty(n_perm)
    for k in range(n_perm):
        xp = np.asarray(x, float).copy()
        for i in idx:
            xp[i] = rng.permutation(xp[i])
        null[k] = full(xp) / sst
    return dict(share=obs, null_mean=null.mean(), null_95=np.percentile(null, 95),
                p=(1 + np.sum(null >= obs)) / (1 + n_perm), base_share=base / sst)
