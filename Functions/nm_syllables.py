"""
NM SYLLABLES: neuromodulator photometry and behaviour syllables on one trial grid
=================================================================================
Replaces the uncommitted Functions/event_psth.py + syllable_photometry.py for the syllable x neuromodulator
analyses. Reads only two sources, both under paper-individuality/data/neuromodulators/:

    pose_qc_hmm_k4/            photometry export (davcrom/ibl_neuromodulators): per session {eid}.h5 with the
                               trials table and -1..+1 s snippets of z-scored GCaMP around stimOnTrigger_times and
                               feedback_times at 30 Hz; recordings.csv = one row per fiber
    states_files_<RUN>/        per session parquet at 30 Hz on the same clock: syllable code (most_likely_states)
                               plus the continuous signals it was built from (wheel, whisker ME, paw x/y, licks)

    load_session(eid)  -> one dict per fiber: meta (one row per go trial) and (n_trials x T) arrays per window,
                          photometry, syllable code and continuous movement, all masked the same way
    factored_features / syllable_features -> 0/1 indicators per feature, (F x n_trials x T)
    correlation_maps   -> Fisher z of r(photometry, indicator) across trials at each sample
    effect_maps        -> per feature and sample: base rate, r, Δz, d and the partial β (Δz with covariates held)
    joint_maps         -> all features in one regression: each feature's unique β
    variance_partition -> cross-validated R² of covariates / + continuous vigor / + syllables / + both
    summarise          -> mice as the unit: mean of per-mouse means, t-test across mice, FDR

USE IT (from any folder in the repo):

    import sys, pathlib
    _p = pathlib.Path.cwd().resolve()
    while not (_p / 'Functions' / 'nm_syllables.py').exists() and _p != _p.parent:
        _p = _p.parent
    sys.path.insert(0, str(_p))
    from Functions import nm_syllables as ns

    fibers = ns.load_session(eid)

Masks, per trial (as notebooks 4-6): in the stimulus window, samples before the trial start (intervals_0) and from
feedback on are blanked; in the feedback window, samples before stimulus onset and after the trial end
(intervals_1). No sample shows a neighbouring trial, and the two windows never share a sample.

Syllable codes are read at the nearest 30 Hz bin of each photometry sample, no time-warping:
code = paw + n_paw * whisking + 2 * n_paw * licking. Checked against the cached notebook-6 maps
(corr_factored_raw, session 813e8682): identical z at every sample.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / 'paper-individuality' / 'data' / 'neuromodulators'
ARCHIVE = DATA / 'pose_qc_hmm_k4'
RUN = 'left_paw_wheel_k8_own_lick-100ms'
STATES_DIR = DATA / f'states_files_{RUN}'
PALETTE_FILE = DATA / 'state_palette.csv'
N_PAW = 8
FS = 30.
WINDOWS = ('stim', 'fb')
EVENTS = {'stim': 'stimOnTrigger_times', 'fb': 'feedback_times'}
# name -> states-file column; 'paw speed' and 'lick rate' are derived in load_session
MOVEMENT = {'paw speed': 'paw speed', 'wheel speed': 'avg_wheel_vel', 'whisker ME': 'whisker_me',
            'lick count': 'Lick count', 'lick rate': 'lick rate'}
LICK_WINDOW = 7  # bins (~233 ms, centred) over which lick events are counted for the lick rate

sys.path.insert(0, str(ARCHIVE))
import h5_loaders  # noqa: E402  (ships with the archive)


def load_recordings(with_states=True):
    """recordings.csv, optionally only fibers whose session has a states file."""
    rec = pd.read_csv(ARCHIVE / 'recordings.csv')
    if with_states:
        rec = rec[rec['eid'].isin(states_files())]
    return rec.reset_index(drop=True)


def states_files():
    """eid -> path of its states file."""
    return {p.name.split('_')[3]: p for p in STATES_DIR.glob('*_states_file_*')}


def load_palette():
    """Paw-state palette (colour, vigor) indexed by state; None if the file is not there."""
    return pd.read_csv(PALETTE_FILE, index_col='state') if PALETTE_FILE.exists() else None


def run_config():
    cfg = json.loads((STATES_DIR / '_config.json').read_text())
    assert cfg['relabel'] is None, 'paw digits are relabelled in this run: map them first'
    return cfg


def trial_table(eid):
    """IBL trials table of the h5, go trials only, with the task variables the analyses use."""
    tr = h5_loaders.load_trials(eid)
    tr = tr[tr['choice'] != 0].copy()
    tr['correct'] = (tr['feedbackType'] == 1).astype(float)
    tr['rt'] = tr['firstMovement_times'] - tr['stimOnTrigger_times']
    tr['response_rt'] = tr['response_times'] - tr['stimOnTrigger_times']
    tr['trial_frac'] = tr['trial'] / tr['trial'].max()
    tr['prev_correct'] = tr['feedbackType'].shift(1).eq(1).astype(float)
    tr.loc[tr['trial'].diff() != 1, 'prev_correct'] = np.nan  # previous trial was a no-go or missing
    return tr.set_index('trial', drop=False)


def window_masks(tr, time):
    """(n_trials x T) boolean keep-masks for the stimulus and feedback windows."""
    stim = tr['stimOnTrigger_times'].to_numpy()[:, None]
    fb = tr['feedback_times'].to_numpy()[:, None]
    t0, t1 = tr['intervals_0'].to_numpy()[:, None], tr['intervals_1'].to_numpy()[:, None]
    ts, tf = stim + time, fb + time
    return {'stim': (ts >= t0) & (ts < fb), 'fb': (tf >= stim) & (tf <= t1)}


def on_grid(states, columns, event_times, time):
    """Values of states[columns] at the bin nearest event_times + time; NaN outside the file."""
    b = states['Bin'].to_numpy()
    t = event_times[:, None] + time
    i = np.round((t - b[0]) * FS)
    outside = np.isnan(i) | (i < 0) | (i > len(b) - 1)
    i = np.where(outside, 0, i).astype(int)
    out = {}
    for c in columns:
        x = states[c].to_numpy(dtype=float)[i]
        x[outside] = np.nan
        out[c] = x
    return out


def paw_speed(states):
    """Left-paw speed (px/s) per bin from the pose x/y (the bins are contiguous at 30 Hz)."""
    v = np.hypot(np.gradient(states['l_paw_x'].to_numpy()), np.gradient(states['l_paw_y'].to_numpy())) * FS
    return pd.Series(v, index=states.index)


def lick_rate(states, window=LICK_WINDOW):
    """Licks/s per bin: 'Lick count' is a 0/1 lick event per bin, counted over a centred window."""
    return states['Lick count'].rolling(window, center=True, min_periods=1).mean() * FS


def load_session(eid, movement=True):
    """One dict per fiber of the session: region, meta, time, and per window photometry / code / movement.

    meta: one row per go trial that has a photometry snippet, with fiber and task columns. Arrays are
    (n_trials x T) and row-aligned with meta. Masked samples are NaN in every array.
    """
    rec = pd.read_csv(ARCHIVE / 'recordings.csv')
    rec = rec[rec['eid'] == eid]
    responses = h5_loaders.load_responses(eid)
    tr = trial_table(eid)
    states = pd.read_parquet(states_files()[eid])
    states = states.assign(**{'paw speed': paw_speed(states), 'lick rate': lick_rate(states),
                              'avg_wheel_vel': states['avg_wheel_vel'].abs()})
    fibers = []
    for _, r in rec.iterrows():
        R = responses[r['brain_region']]
        time = R['time'].values
        trials = R['trial'].values[np.isin(R['trial'].values, tr.index)]
        t = tr.loc[trials]
        rows = pd.Index(R['trial'].values).get_indexer(trials)
        keep = window_masks(t, time)
        meta = t.reset_index(drop=True).assign(eid=eid, subject=r['subject'], region=r['brain_region'],
                                               target_NM=r['target_NM'], hemisphere=r['hemisphere'])
        out = {'region': r['brain_region'], 'meta': meta, 'time': time}
        cols = ['most_likely_states'] + (list(MOVEMENT.values()) if movement else [])
        for w in WINDOWS:
            P = R.sel(event=EVENTS[w]).values[rows]
            g = on_grid(states, cols, t[EVENTS[w]].to_numpy(), time)
            m = keep[w]
            out[w] = {'photometry': np.where(m, P, np.nan), 'code': np.where(m, g['most_likely_states'], np.nan)}
            if movement:
                out[w]['movement'] = {name: np.where(m, g[c], np.nan) for name, c in MOVEMENT.items()}
        fibers.append(out)
    return fibers


def decode(code, n_paw=N_PAW):
    """code -> (paw, whisking, licking), NaN where code is NaN."""
    return code % n_paw, (code // n_paw) % 2, code // (2 * n_paw)


def factored_features(code, n_paw=N_PAW):
    """(names, F x ... indicators): one per paw state, whisking, licking."""
    paw, whisk, lick = decode(code, n_paw)
    names = [f'paw {k}' for k in range(n_paw)] + ['whisking', 'licking']
    X = np.stack([np.where(np.isnan(code), np.nan, paw == k) for k in range(n_paw)] + [whisk, lick]).astype(float)
    return names, X


def syllable_name(c, n_paw=N_PAW):
    return f'paw {c % n_paw}' + (' +W' if (c // n_paw) % 2 else '') + (' +L' if c // (2 * n_paw) else '')


def syllable_features(code, codes, n_paw=N_PAW):
    """(names, F x ... indicators): one per full syllable code in `codes`."""
    names = [syllable_name(c, n_paw) for c in codes]
    return names, np.stack([np.where(np.isnan(code), np.nan, code == c) for c in codes]).astype(float)


def nan_corr(X, Y, min_trials=20, min_events=5):
    """Column-wise Pearson r over rows where both are finite. NaN with fewer than min_trials rows, or (Y 0/1)
    fewer than min_events trials on either side of the indicator."""
    M = ~np.isnan(X) & ~np.isnan(Y)
    n = M.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx = np.where(M, X, 0).sum(0) / n
        my = np.where(M, Y, 0).sum(0) / n
        dx, dy = np.where(M, X - mx, 0), np.where(M, Y - my, 0)
        r = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
        ones = np.where(M, Y, 0).sum(0)
    r[(n < min_trials) | (ones < min_events) | (n - ones < min_events)] = np.nan
    return r


def correlation_maps(fiber, features=factored_features, min_trials=20, min_events=5):
    """Raw r maps of one fiber, per outcome: rows like the cached notebook-6 table (z_stim, z_fb = F x T Fisher z).

    `features` maps a code array to (names, F x n x T indicators).
    """
    meta, rows = fiber['meta'], []
    for outcome, g in meta.groupby('correct'):
        idx = g.index.to_numpy()
        z = {}
        for w in WINDOWS:
            names, F = features(fiber[w]['code'][idx])
            r = np.vstack([nan_corr(fiber[w]['photometry'][idx], f, min_trials, min_events) for f in F])
            z[w] = np.arctanh(np.clip(r, -0.999, 0.999))
        rows.append({'eid': meta['eid'].iat[0], 'region': fiber['region'], 'outcome': outcome,
                     'subject': meta['subject'].iat[0], 'target_NM': meta['target_NM'].iat[0],
                     'n_trials': len(g), 'z_stim': z['stim'], 'z_fb': z['fb']})
    out = pd.DataFrame(rows)
    out.attrs['features'] = names
    return out


# ------------------------------------------------------------------------------------------------
# Effect sizes
# ------------------------------------------------------------------------------------------------
# Covariates held fixed in the partial β. 'task' is notebook 6's set (it reproduces the cached partial maps
# to r ~ 0.997); 'extended' adds what dopamine is expected to track besides the movement.
COVARIATES = {'task': ['log response RT', '|contrast| level', 'trial position', 'trial position²'],
              'extended': ['log response RT', '|contrast| level', 'trial position', 'trial position²',
                           'previous outcome', 'choice side', 'block congruence']}


def covariate_matrix(meta, kind='task'):
    """(n_trials x k) covariates, without intercept; rows with any NaN are dropped by the fits."""
    cols = {}
    for c in COVARIATES[kind]:
        if c == 'log response RT':
            cols[c] = np.log(meta['response_rt'].where(meta['response_rt'] > 0))
        elif c == '|contrast| level':
            for v in sorted(meta['contrast'].unique())[1:]:
                cols[f'contrast {v:g}'] = (meta['contrast'] == v).astype(float)
        elif c == 'trial position':
            cols[c] = meta['trial_frac']
        elif c == 'trial position²':
            cols[c] = meta['trial_frac'] ** 2
        elif c == 'previous outcome':
            cols[c] = meta['prev_correct']
        elif c == 'choice side':
            cols[c] = meta['choice'].astype(float)
        elif c == 'block congruence':
            # +1 stimulus on the side the block favours, -1 against it, 0 for 50:50 blocks or 0% contrast
            side = np.sign(meta['signed_contrast'])
            favoured = np.sign(0.5 - meta['probabilityLeft'])  # pLeft 0.8 -> left (-1) favoured
            cols[c] = (side * favoured).fillna(0.)
    return pd.DataFrame(cols, index=meta.index).to_numpy(dtype=float)


def _ols(X, y):
    """β and its standard error for the first column of X (intercept added)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    dof = len(y) - rank
    if dof < 3:
        return np.nan, np.nan
    s2 = np.sum((y - A @ beta) ** 2) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    return beta[1], np.sqrt(cov[1, 1])


def effect_maps(fiber, features=factored_features, covariates='task', min_trials=20, min_events=5):
    """Per outcome, window, feature and sample: how the photometry differs with vs without the feature.

    Columns of the returned table hold (F x T) arrays per window (suffix _stim / _fb):
      n     trials with both values          p     base rate of the feature
      r     point-biserial r (raw)           dz    mean(z | feature) - mean(z | no feature), in z units
      d     dz / sd(z): r with the base rate taken out, r / sqrt(p(1-p))
      beta  dz with the covariates held fixed (OLS), and its t = beta / se
    r depends on p (it is capped by sqrt(p(1-p))); dz, d and beta do not, so they compare across epochs.
    """
    meta, rows = fiber['meta'], []
    C = covariate_matrix(meta, covariates)
    for outcome, g in meta.groupby('correct'):
        idx = g.index.to_numpy()
        row = {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'outcome': outcome,
               'subject': meta['subject'].iat[0], 'target_NM': meta['target_NM'].iat[0],
               'hemisphere': meta['hemisphere'].iat[0], 'n_trials': len(g)}
        for w in WINDOWS:
            P = fiber[w]['photometry'][idx]
            names, F = features(fiber[w]['code'][idx])
            out = {k: np.full((len(F), P.shape[1]), np.nan) for k in ('n', 'p', 'r', 'dz', 'd', 'beta', 't')}
            for i, x in enumerate(F):
                for j in range(P.shape[1]):
                    ok = ~np.isnan(P[:, j]) & ~np.isnan(x[:, j])
                    n, ones = ok.sum(), x[ok, j].sum()
                    out['n'][i, j], out['p'][i, j] = n, ones / n if n else np.nan
                    if n < min_trials or ones < min_events or n - ones < min_events:
                        continue
                    y, xi = P[ok, j], x[ok, j]
                    out['dz'][i, j] = y[xi == 1].mean() - y[xi == 0].mean()
                    out['d'][i, j] = out['dz'][i, j] / y.std()
                    out['r'][i, j] = np.corrcoef(y, xi)[0, 1]
                    full = ok & np.isfinite(C[idx]).all(1)
                    if full.sum() >= min_trials and min(x[full, j].sum(), (1 - x[full, j]).sum()) >= min_events:
                        b, se = _ols(np.column_stack([x[full, j], C[idx][full]]), P[full, j])
                        out['beta'][i, j], out['t'][i, j] = b, b / se
            for k, v in out.items():
                row[f'{k}_{w}'] = v
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs.update(features=names, covariates=covariates)
    return res


def joint_maps(fiber, covariates='task', min_trials=20, min_events=5, n_paw=N_PAW, ref_paw=0):
    """All factored features in one regression per sample: paw states (vs ref_paw, the stillest), whisking,
    licking, plus covariates. beta_<w> is (F x T) with NaN for the reference and for features too rare at that
    sample (fewer than min_events trials), which are left out of that sample's model."""
    meta, rows = fiber['meta'], []
    C = covariate_matrix(meta, covariates)
    names = [f'paw {k}' for k in range(n_paw)] + ['whisking', 'licking']
    for outcome, g in meta.groupby('correct'):
        idx = g.index.to_numpy()
        row = {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'outcome': outcome,
               'subject': meta['subject'].iat[0], 'target_NM': meta['target_NM'].iat[0], 'n_trials': len(g)}
        for w in WINDOWS:
            P = fiber[w]['photometry'][idx]
            _, F = factored_features(fiber[w]['code'][idx], n_paw)
            B, T = np.full((len(names), P.shape[1]), np.nan), np.full((len(names), P.shape[1]), np.nan)
            for j in range(P.shape[1]):
                ok = ~np.isnan(P[:, j]) & ~np.isnan(F[0, :, j]) & np.isfinite(C[idx]).all(1)
                if ok.sum() < min_trials:
                    continue
                X = F[:, ok, j].T
                use = [i for i in range(len(names)) if i != ref_paw
                       and min(X[:, i].sum(), (1 - X[:, i]).sum()) >= min_events]
                if not use:
                    continue
                A = np.column_stack([np.ones(ok.sum()), X[:, use], C[idx][ok]])
                beta, _, rank, _ = np.linalg.lstsq(A, P[ok, j], rcond=None)
                dof = ok.sum() - rank
                if dof < 3:
                    continue
                se = np.sqrt(np.sum((P[ok, j] - A @ beta) ** 2) / dof * np.diag(np.linalg.pinv(A.T @ A)))
                B[use, j], T[use, j] = beta[1:len(use) + 1], beta[1:len(use) + 1] / se[1:len(use) + 1]
            row[f'beta_{w}'], row[f't_{w}'] = B, T
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs.update(features=names, covariates=covariates, reference=names[ref_paw])
    return res


# ------------------------------------------------------------------------------------------------
# Syllable identity vs continuous vigor
# ------------------------------------------------------------------------------------------------
VIGOR = ['paw speed', 'wheel speed', 'whisker ME', 'lick rate']


def _ridge_cv_r2(X, y, folds, alpha):
    """Cross-validated R² of a ridge fit (standardised predictors, unpenalised intercept)."""
    pred = np.empty_like(y)
    for test in folds:
        train = np.ones(len(y), bool); train[test] = False
        mu, sd = X[train].mean(0), X[train].std(0)
        sd[sd == 0] = 1
        Xt, Xs = (X[train] - mu) / sd, (X[test] - mu) / sd
        ym = y[train].mean()
        b = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(X.shape[1]), Xt.T @ (y[train] - ym))
        pred[test] = ym + Xs @ b
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def variance_partition(fiber, covariates='task', n_folds=5, alpha=1., min_trials=30, n_paw=N_PAW, seed=0):
    """Per outcome, window and sample: cross-validated R² of four nested models of the photometry,
    covariates only / + vigor / + syllables / + vigor + syllables.

    vigor = log(1 + paw speed, wheel speed, whisker ME, lick rate) at that sample; syllables = the factored
    one-hots (paw states without paw 0, whisking, licking). Unique syllable variance = R²(all) - R²(cov + vigor):
    what the discrete syllable adds beyond how much the mouse moves; unique vigor = R²(all) - R²(cov + syllables).
    """
    meta, rows = fiber['meta'], []
    C = covariate_matrix(meta, covariates)
    rng = np.random.default_rng(seed)
    models = ['covariates', '+ vigor', '+ syllables', '+ both']
    for outcome, g in meta.groupby('correct'):
        idx = g.index.to_numpy()
        row = {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'outcome': outcome,
               'subject': meta['subject'].iat[0], 'target_NM': meta['target_NM'].iat[0], 'n_trials': len(g)}
        for w in WINDOWS:
            P = fiber[w]['photometry'][idx]
            _, F = factored_features(fiber[w]['code'][idx], n_paw)
            V = np.stack([np.log1p(np.abs(fiber[w]['movement'][v][idx])) for v in VIGOR])
            R2 = np.full((len(models), P.shape[1]), np.nan)
            for j in range(P.shape[1]):
                ok = (~np.isnan(P[:, j]) & ~np.isnan(F[0, :, j]) & ~np.isnan(V[:, :, j]).any(0)
                      & np.isfinite(C[idx]).all(1))
                n = ok.sum()
                if n < min_trials:
                    continue
                folds = np.array_split(rng.permutation(n), n_folds)
                cov, vig, syl = C[idx][ok], V[:, ok, j].T, np.delete(F[:, ok, j].T, 0, axis=1)
                for k, X in enumerate([cov, np.hstack([cov, vig]), np.hstack([cov, syl]),
                                       np.hstack([cov, vig, syl])]):
                    R2[k, j] = _ridge_cv_r2(X, P[ok, j], folds, alpha)
            row[f'r2_{w}'] = R2
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs.update(models=models, covariates=covariates, vigor=VIGOR)
    return res


# ------------------------------------------------------------------------------------------------
# Across sessions and mice
# ------------------------------------------------------------------------------------------------
def summarise(table, column, fisher=False, min_mice=3, alpha=0.05):
    """Mice as the unit, per (target_NM, outcome): mean over each mouse's sessions, then over mice.

    Returns {(nm, outcome): dict(mean, mice (n_mice x ...), t, q, n_mice, n_sessions)}. With fisher=True the
    column holds Fisher z and the mean is returned as r. t-test across mice and FDR (Benjamini-Hochberg, over
    all cells of the map) only with at least min_mice mice; otherwise t and q are NaN.
    """
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    out = {}
    for (nm, outcome), g in table.groupby(['target_NM', 'outcome']):
        mice = np.stack([np.nanmean(np.stack(gm[column].to_list()), 0) for _, gm in g.groupby('subject')])
        mean = np.nanmean(mice, 0)
        t = q = np.full(mean.shape, np.nan)
        if len(mice) >= min_mice:
            t, p = ttest_1samp(mice, 0, axis=0, nan_policy='omit')
            t, p = np.asarray(t), np.asarray(p)
            q = np.full(p.shape, np.nan)
            ok = np.isfinite(p)
            if ok.any():
                q[ok] = multipletests(p[ok], alpha=alpha, method='fdr_bh')[1]
        out[(nm, outcome)] = {'mean': np.tanh(mean) if fisher else mean, 'mice': mice, 't': t, 'q': q,
                              'n_mice': len(mice), 'n_sessions': len(g)}
    return out
