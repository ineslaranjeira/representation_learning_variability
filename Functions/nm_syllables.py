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
code = paw + n_paw * whisking + 2 * n_paw * licking. Masks and nearest-bin ties follow event_psth.py exactly, so
correlation_maps reproduces the cached notebook-6 maps (corr_factored_raw) on every session checked.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / 'paper-individuality' / 'data' / 'neuromodulators'
# the export sits under data/ on the laptop and under neuromodulators/ on the analysis machine (as in ddm_hmm.py)
ARCHIVE = next((p for p in (DATA / 'pose_qc_hmm_k4', REPO / 'neuromodulators' / 'pose_qc_hmm_k4') if p.exists()),
               DATA / 'pose_qc_hmm_k4')
RUN = 'left_paw_wheel_k8_own_lick-100ms'
STATES_DIR = DATA / f'states_files_{RUN}'
PALETTE_FILE = next((p for p in (DATA / 'state_palette.csv',
                                 DATA / 'left_paw_wheel_states_uniform_raw_sessionz' / 'state_palette.csv') if p.exists()),
                    DATA / 'state_palette.csv')
MIN_COVERAGE = 0.25  # fraction of the session's trials inside the states file; below it the fiber is dropped
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


def load_recordings(with_states=True, min_coverage=MIN_COVERAGE):
    """recordings.csv, optionally only fibers whose session has a states file covering at least min_coverage of
    its trials (see states_coverage). Adds the column 'coverage'."""
    rec = pd.read_csv(ARCHIVE / 'recordings.csv')
    if with_states:
        rec = rec[rec['eid'].isin(states_files())]
        cov = states_coverage()
        rec = rec.assign(coverage=rec['eid'].map(cov['coverage']))
        rec = rec[rec['coverage'] >= min_coverage]
    return rec.reset_index(drop=True)


def states_coverage(cache=True):
    """Per eid: how much of the session the states file spans. One row per session with the file's first and last
    bin, the session's trial span, and coverage = fraction of trials (intervals_0..intervals_1) inside the file.

    A file stops early when the video does: the design matrix is cut to the shortest of camera, wheel and licks.
    Known cases (2026-10-26): 813e8682 (ZFM-08828) recorded only 11142 frames (6 of 83 min) -> coverage .07;
    e637617d (ZFM-04392) has a full-length video (155709 frames) but camera times, pose and motion energy were only
    extracted for the first 61901 frames (camera_times QC: 93808 missing) -> coverage .40, recoverable upstream.
    """
    path = STATES_DIR / '_coverage.csv'
    files = states_files()
    if cache and path.exists():
        cov = pd.read_csv(path, index_col='eid')
        if set(cov.index) == set(files):
            return cov
    rows = []
    for eid, f in files.items():
        b = pd.read_parquet(f, columns=['Bin'])['Bin']
        tr = h5_loaders.load_trials(eid) if (ARCHIVE / 'sessions' / f'{eid}.h5').exists() else None
        row = {'eid': eid, 'subject': f.name.split('_', 4)[4], 'first_bin': b.min(), 'last_bin': b.max()}
        if tr is not None:
            inside = (tr['intervals_0'] >= b.min()) & (tr['intervals_1'] <= b.max())
            row.update(session_start=tr['intervals_0'].min(), session_end=tr['intervals_1'].max(),
                       n_trials=len(tr), coverage=inside.mean())
        rows.append(row)
    cov = pd.DataFrame(rows).set_index('eid')
    cov.to_csv(path)
    return cov


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
    """(n_trials x T) boolean keep-masks for the stimulus and feedback windows.

    Written as event_psth.window_masks (offsets relative to the event, a NaN bound masks nothing) so both give the
    same samples, including trials without feedback_times and samples exactly on a bound."""
    stim = tr['stimOnTrigger_times'].to_numpy()[:, None]
    fb = tr['feedback_times'].to_numpy()[:, None]
    t0, t1 = tr['intervals_0'].to_numpy()[:, None], tr['intervals_1'].to_numpy()[:, None]
    t = time[None]
    return {'stim': ~((t >= fb - stim) | (t < t0 - stim)), 'fb': ~((t < stim - fb) | (t > t1 - fb))}


def on_grid(states, columns, event_times, time):
    """Values of states[columns] at the bin nearest event_times + time; NaN outside the file.

    Nearest bin as in event_psth.syllable_windows (searchsorted on the bin times; an exact tie goes to the later
    bin): in some sessions the photometry grid sits half a bin off the states grid, so ties do occur."""
    b = states['Bin'].to_numpy()
    t = event_times[:, None] + time
    j = np.clip(np.searchsorted(b, t), 1, len(b) - 1)
    i = np.where(np.abs(b[j - 1] - t) < np.abs(b[j] - t), j - 1, j)
    outside = np.isnan(t) | (np.abs(b[i] - t) > 0.5 / FS)
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


def load_session(eid, movement=True, mask=True):
    """One dict per fiber of the session: region, meta, time, and per window photometry / code / movement.

    meta: one row per go trial that has a photometry snippet, with fiber and task columns. Arrays are
    (n_trials x T) and row-aligned with meta. Masked samples are NaN in every array; mask=False keeps the full
    -1..+1 s snippets (the trial-shape analyses, which need every trial on the same samples).
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
            m = keep[w] if mask else np.ones_like(keep[w])
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
                           'previous outcome', 'choice side', 'block congruence'],
              'full': ['log response RT', '|contrast| level', 'trial position', 'trial position²',
                       'previous outcome', 'choice side', 'block congruence', 'outcome']}
# covariate -> group, for the unique variance of each task variable (task_vs_movement)
TASK_GROUPS = {'log response RT': 'RT', '|contrast| level': 'contrast', 'trial position': 'trial position',
               'trial position²': 'trial position', 'previous outcome': 'previous outcome', 'choice side': 'choice',
               'block congruence': 'block congruence', 'outcome': 'outcome'}


def covariate_matrix(meta, kind='task', groups=False):
    """(n_trials x k) covariates, without intercept; rows with any NaN are dropped by the fits.
    groups=True also returns the TASK_GROUPS name of each column."""
    cols, group = {}, []
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
        elif c == 'outcome':
            cols[c] = meta['correct']
        group += [TASK_GROUPS[c]] * (len(cols) - len(group))
    X = pd.DataFrame(cols, index=meta.index).to_numpy(dtype=float)
    return (X, group) if groups else X


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


ALPHAS = np.logspace(-1, 5, 13)  # ridge penalties searched on each training fold (standardised predictors)
# CV fits need min_trials = 60 at a sample: with 30, the few long-RT trials left early in the feedback window gave
# single sessions an R² of -1000 % or worse, which dragged whole mouse means down


def _ridge_cv_r2(X, y, folds, alpha):
    """Cross-validated R² of a ridge fit (standardised predictors, unpenalised intercept).

    alpha: a number, or 'auto' to pick it per training fold from ALPHAS by leave-one-out error (closed form from
    the SVD), so a model with nothing to add shrinks to the intercept instead of paying for overfitting."""
    pred = np.empty_like(y)
    for test in folds:
        train = np.ones(len(y), bool); train[test] = False
        mu, sd = X[train].mean(0), X[train].std(0)
        sd[sd == 0] = 1
        Xt, Xs = (X[train] - mu) / sd, (X[test] - mu) / sd
        ym = y[train].mean()
        yc = y[train] - ym
        U, s, Vt = np.linalg.svd(Xt, full_matrices=False)
        Uy = U.T @ yc
        if alpha == 'auto':
            shrink = s ** 2 / (s ** 2 + ALPHAS[:, None])                 # (n_alpha x p)
            fit = (U[None] * (shrink * Uy)[:, None, :]).sum(-1)           # (n_alpha x n_train)
            hat = (U[None] ** 2 * shrink[:, None, :]).sum(-1) + 1 / len(yc)
            loo = (((yc - fit) / (1 - hat)) ** 2).mean(1)
            a = ALPHAS[np.argmin(loo)]
        else:
            a = alpha
        b = Vt.T @ (s / (s ** 2 + a) * Uy)
        pred[test] = ym + Xs @ b
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def variance_partition(fiber, covariates='task', n_folds=5, alpha='auto', min_trials=60, n_paw=N_PAW, seed=0):
    """Per outcome, window and sample: cross-validated R² of four nested models of the photometry,
    covariates only / + vigor / + syllables / + vigor + syllables.

    vigor = log(1 + paw speed, wheel speed, whisker ME, lick rate) at that sample; syllables = the factored
    one-hots (paw states without paw 0, whisking, licking). Unique syllable variance = R²(all) - R²(cov + vigor):
    what the discrete syllable adds beyond how much the mouse moves; unique vigor = R²(all) - R²(cov + syllables).
    """
    meta, rows = fiber['meta'], []
    C = covariate_matrix(meta, covariates)
    rng = np.random.default_rng(seed)
    models = ['task variables', '+ vigor', '+ syllables', '+ both']
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


def task_vs_movement(fiber, covariates='full', movement='vigor', pool_outcomes=True, n_folds=5, alpha='auto',
                     min_trials=60, n_paw=N_PAW, seed=0):
    """Cross-validated R² per window and sample of: task only, movement only, task + movement, and task + movement
    without each task group in turn, then without each movement component in turn (their unique variance).
    movement = 'vigor' (the continuous signals, components = VIGOR; the syllables add nothing beyond them, see
    variance_partition) or 'vigor + syllables' (components = vigor, paw state, whisking, licking).

    pool_outcomes=True fits rewarded and error trials together (one row per fiber, outcome = 'all'), so the outcome
    can be a task variable; otherwise one row per outcome (and 'outcome' is dropped as it is constant).
    Columns r2_<w>: (n_models x T), model names in attrs['models']. Like variance_partition, this is variance across
    trials at a fixed sample: the mean time course (the PSTH) is not in it.
    """
    meta, rows = fiber['meta'], []
    C, group = covariate_matrix(meta, covariates, groups=True)
    group = np.array(group)
    names = list(dict.fromkeys(group))
    rng = np.random.default_rng(seed)
    splits = [('all', meta.index.to_numpy())] if pool_outcomes else \
        [(o, g.index.to_numpy()) for o, g in meta.groupby('correct')]
    parts = list(VIGOR) if movement == 'vigor' else ['vigor', 'paw state', 'whisking', 'licking']
    models = ['task', 'movement', 'task + movement'] + [f'without {n}' for n in names + parts]
    for outcome, idx in splits:
        keep_cols = np.ones(len(group), bool) if pool_outcomes else group != 'outcome'
        row = {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'outcome': outcome,
               'subject': meta['subject'].iat[0], 'target_NM': meta['target_NM'].iat[0], 'n_trials': len(idx)}
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
                task = C[idx][ok][:, keep_cols]
                vig, syl = V[:, ok, j].T, np.delete(F[:, ok, j].T, 0, axis=1)   # syllables: paw 1..7, whisk, lick
                if movement == 'vigor':
                    move, part = vig, np.arange(len(VIGOR))
                else:
                    move = np.hstack([vig, syl])
                    part = np.r_[np.zeros(len(VIGOR)), np.ones(n_paw - 1), 2, 3].astype(int)
                Xs = [task, move, np.hstack([task, move])]
                Xs += [np.hstack([task[:, group[keep_cols] != g], move]) if (group[keep_cols] == g).any() else None
                       for g in names]
                Xs += [np.hstack([task, move[:, part != k]]) for k in range(len(parts))]
                for k, X in enumerate(Xs):
                    if X is not None:
                        R2[k, j] = _ridge_cv_r2(X, P[ok, j], folds, alpha)
            row[f'r2_{w}'] = R2
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs.update(models=models, groups=names, parts=parts, covariates=covariates, movement=movement)
    return res


# ------------------------------------------------------------------------------------------------
# Trial shapes: CCA and reduced-rank regression between the vigor and photometry time courses
# ------------------------------------------------------------------------------------------------
# Unmasked windows (s) and downsampling: every trial is one vector of photometry samples and one of vigor samples
SHAPE_WINDOWS = {'stim': (-0.5, 0.5), 'fb': (-0.5, 1.0)}
SHAPE_BIN = 3  # 30 Hz samples per bin -> 10 Hz


def trial_shapes(fiber, windows=SHAPE_WINDOWS, bin=SHAPE_BIN):
    """Per trial, the photometry and the four vigor signals over the shape windows, binned.

    fiber must come from load_session(..., mask=False). Returns (Y, X, meta, labels): Y (n x q) photometry,
    X (n x 4q) log(1 + vigor), trials with any NaN dropped; labels give (window, signal, bin centre) per column.
    """
    T = fiber['time']
    Ys, Xs, ylab, xlab = [], [], [], []
    for w, (a, b) in windows.items():
        m = np.nonzero((T >= a - 1e-9) & (T < b - 1e-9))[0]
        m = m[:len(m) // bin * bin]
        centres = T[m].reshape(-1, bin).mean(1)
        binned = lambda A: A[:, m].reshape(len(A), -1, bin).mean(2)
        Ys.append(binned(fiber[w]['photometry']))
        ylab += [(w, 'photometry', c) for c in centres]
        for v in VIGOR:
            Xs.append(binned(np.log1p(np.abs(fiber[w]['movement'][v]))))
            xlab += [(w, v, c) for c in centres]
    Y, X = np.hstack(Ys), np.hstack(Xs)
    ok = np.isfinite(Y).all(1) & np.isfinite(X).all(1)
    return Y[ok], X[ok], fiber['meta'][ok].reset_index(drop=True), (ylab, xlab)


def _pca(A, k):
    """Centre, then the top-k principal axes (k x p) and whitened scores' scale."""
    mu = A.mean(0)
    U, s, Vt = np.linalg.svd(A - mu, full_matrices=False)
    k = min(k, (s > s[0] * 1e-8).sum())
    return mu, Vt[:k], s[:k] / np.sqrt(len(A) - 1)


def _cca_fit(X, Y, kx, ky):
    """CCA after PCA to kx / ky dims (the PCA is the regularisation). Returns a projector per side,
    x -> canonical variates (n x d) and y -> (n x d), with d = min(kx, ky)."""
    mx, Vx, sx = _pca(X, kx)
    my, Vy, sy = _pca(Y, ky)
    Zx = (X - mx) @ Vx.T / sx
    Zy = (Y - my) @ Vy.T / sy
    U, rho, Wt = np.linalg.svd(Zx.T @ Zy / (len(X) - 1), full_matrices=False)
    ax = Vx.T @ (U / sx[:, None])                      # p_x x d
    ay = Vy.T @ (Wt.T / sy[:, None])                   # p_y x d
    return (mx, ax), (my, ay), rho


def cca_cv(X, Y, kx=10, ky=10, n_folds=5, n_shuffles=20, seed=0):
    """Cross-validated canonical correlations: fit on k-1 folds, correlate the held-out variates.

    Returns dict(r = (d,) held-out canonical correlations, r_null = (n_shuffles x d) the same with the trial
    pairing of X and Y shuffled, x_load / y_load = structure coefficients on all trials (corr of each original
    column with each canonical variate; sign set so the photometry loading has a positive sum)."""
    rng = np.random.default_rng(seed)

    def held_out(X, Y):
        folds = np.array_split(rng.permutation(len(X)), n_folds)
        d = min(kx, ky)
        U, V = np.full((len(X), d), np.nan), np.full((len(X), d), np.nan)
        for test in folds:
            train = np.ones(len(X), bool); train[test] = False
            (mx, ax), (my, ay), _ = _cca_fit(X[train], Y[train], kx, ky)
            U[test, :ax.shape[1]], V[test, :ay.shape[1]] = (X[test] - mx) @ ax, (Y[test] - my) @ ay
        return np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(d)])

    r = held_out(X, Y)
    r_null = np.stack([held_out(X[rng.permutation(len(X))], Y) for _ in range(n_shuffles)])
    (mx, ax), (my, ay), rho = _cca_fit(X, Y, kx, ky)
    U, V = (X - mx) @ ax, (Y - my) @ ay
    corr = lambda A, B: ((A - A.mean(0)) / A.std(0)).T @ ((B - B.mean(0)) / B.std(0)) / len(A)
    x_load, y_load = np.nan_to_num(corr(X, U)), np.nan_to_num(corr(Y, V))  # a constant column (no licks) -> 0
    sign = np.sign(y_load.sum(0)); sign[sign == 0] = 1
    return {'r': r, 'r_null': r_null, 'rho_train': rho, 'x_load': x_load * sign, 'y_load': y_load * sign}


def rrr_cv(X, Y, kx=10, max_rank=5, n_folds=5, n_shuffles=20, seed=0):
    """Reduced-rank regression of the photometry shape (Y) on the vigor shape (X, PCA to kx dims).

    Rank-r prediction = OLS prediction projected on its top r principal axes. Returns dict(r2 = (max_rank,)
    held-out R² of Y (summed over its columns, so the whole trace), r2_null = (n_shuffles x max_rank) with the
    trial pairing shuffled, y_axes = (max_rank x q) the photometry patterns X predicts, fitted on all trials,
    x_weights = (q_x x max_rank) the vigor pattern that drives each)."""
    rng = np.random.default_rng(seed)

    def fit(X, Y):
        mx, Vx, sx = _pca(X, kx)
        Z = (X - mx) @ Vx.T / sx
        my = Y.mean(0)
        B = np.linalg.lstsq(Z, Y - my, rcond=None)[0]       # kx x q
        _, _, At = np.linalg.svd(Z @ B, full_matrices=False)
        return mx, Vx, sx, my, B, At[:max_rank]

    def held_out(X, Y):
        folds = np.array_split(rng.permutation(len(X)), n_folds)
        sse, sst = np.zeros(max_rank), 0.
        for test in folds:
            train = np.ones(len(X), bool); train[test] = False
            mx, Vx, sx, my, B, A = fit(X[train], Y[train])
            pred_full = ((X[test] - mx) @ Vx.T / sx) @ B
            for r in range(max_rank):
                P = A[:r + 1]
                sse[r] += np.sum((Y[test] - my - pred_full @ P.T @ P) ** 2)
            sst += np.sum((Y[test] - Y[train].mean(0)) ** 2)
        return 1 - sse / sst

    r2 = held_out(X, Y)
    r2_null = np.stack([held_out(X[rng.permutation(len(X))], Y) for _ in range(n_shuffles)])
    mx, Vx, sx, my, B, A = fit(X, Y)
    A = A * np.sign(A.sum(1, keepdims=True))
    x_weights = Vx.T @ ((B @ A.T) / sx[:, None])
    return {'r2': r2, 'r2_null': r2_null, 'y_axes': A, 'x_weights': x_weights}


def shape_analyses(fiber, **kw):
    """cca_cv and rrr_cv of one fiber (from load_session(..., mask=False)), one row."""
    Y, X, meta, (ylab, xlab) = trial_shapes(fiber)
    row = {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'subject': meta['subject'].iat[0],
           'target_NM': meta['target_NM'].iat[0], 'n_trials': len(Y)}
    if len(Y) < 100:
        return None
    c, r = cca_cv(X, Y, **kw), rrr_cv(X, Y, **kw.get('rrr', {}))
    row.update({f'cca_{k}': v for k, v in c.items()})
    row.update({f'rrr_{k}': v for k, v in r.items()})
    return row, (ylab, xlab)


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
