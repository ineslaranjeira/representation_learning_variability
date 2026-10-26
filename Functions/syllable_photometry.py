"""
SYLLABLES x PHOTOMETRY: trial-to-trial correlations on the PSTH grid
====================================================================
Used by neuromodulators/6_syllables_photometry.ipynb. Works on every session of the photometry export
(neuromodulators/pose_qc_hmm_k4) that also has a syllable states file; no DDM-HMM needed.

At each sample of the stimulus- and feedback-aligned windows (Functions/event_psth.py), across the trials of
one fiber-session and one outcome, correlate the photometry value with a syllable indicator (1 = the trial is
in that syllable at that sample). Fixing the sample removes the shared trial time course. Optionally both
are first residualised on trial covariates (log RT, |contrast| level, trial position) -- a partial
correlation -- fitted separately at every sample, on the trials valid there.

    from Functions import syllable_photometry as sp
    trials = sp.session_trials(rec['eid'].unique())
    res = sp.correlations(meta, STIM, FB, SYL_S, SYL_F, sp.factored_features(8))
"""
import warnings

import numpy as np
import pandas as pd

from Functions import ddm_hmm as dh

CONTRASTS = [0.0, 6.25, 12.5, 25.0, 100.0]


def session_trials(eids):
    """Go trials of each session from the export's trials table, with the columns the analyses use."""
    L = dh._loaders()
    rows = []
    for eid in eids:
        t = L.load_trials(eid)
        t = t[t['choice'] != 0].copy()
        t['eid'] = eid
        t['correct'] = (t['feedbackType'] == 1).astype(float)
        t['rt'] = t['response_times'] - t['stimOnTrigger_times']
        span = max(t['trial'].max() - t['trial'].min(), 1)
        t['trial_frac'] = (t['trial'] - t['trial'].min()) / span
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def covariates(t):
    """Design matrix (n x p): intercept, z(log RT), |contrast| dummies (0% reference), trial position, its
    square. Columns with no variance in this subset are dropped."""
    lrt = np.log(np.clip(t['rt'].to_numpy(float), 0.05, None))
    cols = [np.ones(len(t)), (lrt - lrt.mean()) / (lrt.std() + 1e-9)]
    c = t['contrast'].to_numpy(float)
    cols += [(np.isclose(c, lvl)).astype(float) for lvl in CONTRASTS[1:]]
    tf = t['trial_frac'].to_numpy(float)
    cols += [tf - tf.mean(), (tf - tf.mean()) ** 2]
    X = np.column_stack(cols)
    keep = np.r_[True, X[:, 1:].std(0) > 0]
    return X[:, keep]


def residualise(Y, X, ridge=1e-6):
    """Residuals of every column of Y (n x T, NaN = missing) on X (n x p), fitted per column on the rows
    valid in that column (batched normal equations)."""
    M = ~np.isnan(Y)
    Y0 = np.where(M, Y, 0.0)
    XtX = np.einsum('nt,np,nq->tpq', M.astype(float), X, X) + ridge * np.eye(X.shape[1])[None]
    XtY = np.einsum('nt,np,nt->tp', M.astype(float), X, Y0)
    beta = np.linalg.solve(XtX, XtY[..., None])[..., 0]          # (T, p)
    R = Y - X @ beta.T
    R[~M] = np.nan
    return R


def nan_corr(X, Y, valid_y=None, min_n=20, min_events=5):
    """Column-wise Pearson r over rows where both are finite. `valid_y` (binary indicator before any
    residualisation) gates samples with too few events on either side."""
    M = ~np.isnan(X) & ~np.isnan(Y)
    n = M.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx = np.where(M, X, 0).sum(0) / n
        my = np.where(M, Y, 0).sum(0) / n
        dx = np.where(M, X - mx, 0); dy = np.where(M, Y - my, 0)
        r = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
    bad = n < min_n
    if valid_y is not None:
        ones = np.where(M, valid_y, 0).sum(0)
        bad |= (ones < min_events) | (n - ones < min_events)
    r[bad] = np.nan
    return r


def factored_features(n_paw=8):
    """name -> function(codes) giving the binary indicator (NaN where no code)."""
    feats = {f'paw {k}': (lambda c, k=k: np.where(np.isnan(c), np.nan, (c % n_paw) == k)) for k in range(n_paw)}
    feats['whisking'] = lambda c: np.where(np.isnan(c), np.nan, (c // n_paw) % 2 == 1)
    feats['licking'] = lambda c: np.where(np.isnan(c), np.nan, c // (2 * n_paw) == 1)
    return feats


def syllable_features(codes_in_use, n_paw=8):
    """name -> indicator function, one per full syllable code (paw + n_paw*whisk + 2*n_paw*lick)."""
    feats = {}
    for c in codes_in_use:
        name = f'paw {c % n_paw}' + (' +W' if (c // n_paw) % 2 else '') + (' +L' if c // (2 * n_paw) else '')
        feats[name] = lambda x, c=c: np.where(np.isnan(x), np.nan, x == c)
    return feats


def correlations(meta, STIM, FB, SYL_S, SYL_F, features, partial=True, min_trials=20, min_events=5):
    """Fisher-z correlation maps per (fiber-session, outcome).

    Returns a DataFrame with one row per (eid, region, outcome) and columns subject, target_NM, n_trials,
    z_stim, z_fb (arrays of shape n_features x T). `features` is an ordered dict name -> indicator fn."""
    names = list(features)
    out = []
    for (eid, region, outcome), g in meta.groupby(['eid', 'region', 'correct']):
        if len(g) < min_trials:
            continue
        idx = g.index.to_numpy()
        X = covariates(g) if partial else None
        row = dict(eid=eid, region=region, outcome=outcome, subject=g['subject'].iloc[0],
                   target_NM=g['target_NM'].iloc[0], n_trials=len(g))
        for w, (P, C) in enumerate([(STIM, SYL_S), (FB, SYL_F)]):
            p = P[idx].astype(float)
            c = C[idx].astype(float)
            p = np.where(np.isnan(c), np.nan, p)          # same rows for every feature
            if partial:
                p = residualise(p, X)
            Z = np.full((len(names), P.shape[1]), np.nan)
            for i, f in enumerate(names):
                y = features[f](c).astype(float)
                yr = residualise(y, X) if partial else y
                r = nan_corr(p, yr, valid_y=y, min_n=min_trials, min_events=min_events)
                Z[i] = np.arctanh(np.clip(r, -0.999, 0.999))
            row['z_stim' if w == 0 else 'z_fb'] = Z
        out.append(row)
    res = pd.DataFrame(out)
    res.attrs['features'] = names
    return res


def summarise(res, by='target_NM'):
    """Per group and outcome: mouse-mean z maps, mean over mice, t-test across mice (FDR within the map),
    and the fraction of mice sharing the mean's sign.

    Returns dict (group, outcome) -> dict(r=(stim, fb), q=(stim, fb), agree=(stim, fb), mice=(stim, fb)
    per-mouse r maps, n_mice, n_sessions)."""
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    out = {}
    for (grp, outcome), g in res.groupby([by, 'outcome']):
        entry = dict(n_mice=g['subject'].nunique(), n_sessions=len(g), r=[], q=[], agree=[], mice=[])
        for col in ('z_stim', 'z_fb'):
            per_mouse = np.stack([np.nanmean(np.stack(gm[col].to_list()), 0) for _, gm in g.groupby('subject')])
            mean = np.nanmean(per_mouse, 0)
            q = np.full(mean.shape, np.nan)
            if len(per_mouse) >= 3:
                with np.errstate(all='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _, p = ttest_1samp(per_mouse, 0, axis=0, nan_policy='omit')
                p = np.asarray(p, float)
                ok = ~np.isnan(p)
                q[ok] = multipletests(p[ok], method='fdr_bh')[1]
            agree = (np.sign(per_mouse) == np.sign(mean)[None]).mean(0)
            entry['r'].append(np.tanh(mean)); entry['q'].append(q); entry['agree'].append(agree)
            entry['mice'].append(np.tanh(per_mouse))
        out[(grp, outcome)] = entry
    return out


def pooled_correlations(meta, STIM, FB, SYL_S, SYL_F, features, centre=False, min_samples=200, min_events=20):
    """Correlation between the photometry value and each syllable indicator over ALL samples of a
    fiber-session, wherever in the PSTH they come from (both windows, every go trial, both outcomes).

    centre=True first subtracts, within the fiber-session, the mean photometry at each (outcome, window,
    sample) and the mean indicator likewise -- i.e. it removes the PSTH time course, leaving the
    trial-to-trial part only. Returns one row per fiber-session: eid, region, subject, target_NM,
    n_samples, r (array over features)."""
    names = list(features)
    rows = []
    for (eid, region), g in meta.groupby(['eid', 'region']):
        idx = g.index.to_numpy()
        P = np.concatenate([STIM[idx], FB[idx]], axis=1).astype(float)
        C = np.concatenate([SYL_S[idx], SYL_F[idx]], axis=1).astype(float)
        ok = ~np.isnan(P) & ~np.isnan(C)
        if ok.sum() < min_samples:
            continue
        P = np.where(ok, P, np.nan)
        outcome = g['correct'].to_numpy()
        r = np.full(len(names), np.nan)
        Pc = P.copy()
        if centre:
            for o in np.unique(outcome):
                m = outcome == o
                Pc[m] = P[m] - np.nanmean(P[m], axis=0, keepdims=True)
        for i, f in enumerate(names):
            Y = np.where(ok, features[f](C).astype(float), np.nan)
            n1 = np.nansum(Y)
            if n1 < min_events or ok.sum() - n1 < min_events:
                continue
            Yc = Y.copy()
            if centre:
                for o in np.unique(outcome):
                    m = outcome == o
                    Yc[m] = Y[m] - np.nanmean(Y[m], axis=0, keepdims=True)
            a, b = Pc[ok], Yc[ok]
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            r[i] = (a * b).sum() / den if den > 0 else np.nan
        rows.append(dict(eid=eid, region=region, subject=g['subject'].iloc[0], target_NM=g['target_NM'].iloc[0],
                         n_samples=int(ok.sum()), r=r))
    res = pd.DataFrame(rows)
    res.attrs['features'] = names
    return res


def summarise_pooled(res, by='target_NM'):
    """Per group: mouse-mean Fisher z per feature, mean over mice (as r), t-test across mice with FDR over
    features, fraction of mice with the mean's sign. Returns (summary DataFrame long, per-mouse r DataFrame)."""
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    names = res.attrs['features']
    z = np.arctanh(np.clip(np.stack(res['r'].to_list()), -0.999, 0.999))
    zdf = pd.DataFrame(z, columns=names).assign(subject=res['subject'].values, group=res[by].values)
    per_mouse = zdf.groupby(['group', 'subject']).mean()
    rows = []
    for grp, pm in per_mouse.groupby(level='group'):
        mean = pm.mean()
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, p = ttest_1samp(pm.to_numpy(), 0, axis=0, nan_policy='omit')
        p = np.asarray(p, float)
        q = np.full(p.shape, np.nan)
        ok = ~np.isnan(p)
        if ok.any():
            q[ok] = multipletests(p[ok], method='fdr_bh')[1]
        agree = (np.sign(pm) == np.sign(mean)).mean()
        for i, f in enumerate(names):
            rows.append({by: grp, 'feature': f, 'r': np.tanh(mean[f]), 'q': q[i], 'mice same sign': agree[f],
                         'n mice': int(pm[f].notna().sum())})
    return pd.DataFrame(rows), np.tanh(per_mouse)
