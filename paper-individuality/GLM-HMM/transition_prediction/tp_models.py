"""
Models and tests for transition prediction (see tp_data.py for the data and the target).

THE COMPARISON. Three nested logistic regressions, all scored out of fold:
    base   task history only (rewards, contrasts, block-congruent choices, session position,
           dwell time)
    syll   syllables only
    full   base + syllables
"How predictable are transitions from syllables" is answered by full vs base -- what the
syllables add beyond what the task history already says -- not by syll alone, because
syllables carry task history too (a rewarded trial is followed by licking).

CROSS-VALIDATION. Folds hold out whole MICE (GroupKFold on mouse_name) by default, so a score is
"predicts transitions in an animal the model has never seen". cv='session' holds out sessions
instead (the model has seen other sessions of the same mouse). The ridge strength is picked
inside each training fold on the same grouping.

SCORES (pooled over all held-out trials, plus per mouse / per session):
    auc        ROC AUC
    ap         average precision; compare with the base rate, its chance level
    ll_skill   1 - logloss / logloss of a constant (training base rate); 0 = no better than
               the base rate, > 0 better. The headline number: it is calibrated and additive
               across trials, so a full-minus-base difference is meaningful
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CS = np.logspace(-4, 0, 9)


def _loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(y, p, p_null):
    y = np.asarray(y)
    out = dict(n=len(y), events=int(y.sum()), base_rate=float(y.mean()))
    if 0 < y.sum() < len(y):
        out.update(auc=roc_auc_score(y, p), ap=average_precision_score(y, p))
    else:
        out.update(auc=np.nan, ap=np.nan)
    ll, ll0 = _loss(y, p).mean(), _loss(y, p_null).mean()
    out.update(logloss=ll, ll_skill=1 - ll / ll0)
    return out


def _splits(groups, n_splits):
    n = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups))


def fit_one(X, y, groups, inner_splits=3, Cs=CS):
    """Standardise, then L2 logistic regression with C chosen by grouped inner CV on log loss."""
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    if len(Cs) == 1:
        clf = LogisticRegression(C=Cs[0], max_iter=2000).fit(Z, y)
    else:
        clf = LogisticRegressionCV(Cs=Cs, cv=_splits(groups, inner_splits), scoring='neg_log_loss',
                                   max_iter=2000, n_jobs=-1).fit(Z, y)
    return scaler, clf


def cross_validate(X, y, meta, cv='mouse', n_splits=5, Cs=CS):
    """Out-of-fold probabilities and per-fold models. Returns (p_oof, p_null_oof, folds) where
    folds is a list of dicts(train, test, scaler, clf)."""
    X = np.asarray(X, dtype=float)
    groups = meta['mouse_name' if cv == 'mouse' else 'eid'].to_numpy()
    p, p0, folds = np.zeros(len(y)), np.zeros(len(y)), []
    for tr, te in _splits(groups, n_splits):
        scaler, clf = fit_one(X[tr], y[tr], groups[tr], Cs=Cs)
        p[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        p0[te] = y[tr].mean()
        folds.append(dict(train=tr, test=te, scaler=scaler, clf=clf))
    return p, p0, folds


def compare_models(Xb, Xs, y, meta, cv='mouse', n_splits=5, models=('base', 'syll', 'full')):
    """Fit the nested models; returns (summary DataFrame, dict of oof predictions, dict of folds)."""
    mats = {'base': Xb, 'syll': Xs, 'full': pd.concat([Xb, Xs], axis=1)}
    rows, preds, folds = [], {}, {}
    for m in models:
        p, p0, f = cross_validate(mats[m], y, meta, cv, n_splits)
        preds[m], folds[m] = p, f
        rows.append(dict(model=m, n_features=mats[m].shape[1], **scores(y, p, p0)))
        preds['null'] = p0
    return pd.DataFrame(rows), preds, folds


def lag_sweep(df, target, lags, cv='mouse', n_splits=5, **design_kw):
    """compare_models for every lag; long DataFrame with one row per (lag, model)."""
    import tp_data as td
    out = []
    for lag in lags:
        Xb, Xs, y, meta = td.design(df, target, lag=lag, **design_kw)
        summ, _, _ = compare_models(Xb, Xs, y, meta, cv, n_splits)
        summ.insert(0, 'lag', lag)
        out.append(summ)
        print(f'lag {lag}: ' + ', '.join(f"{r.model} skill={r.ll_skill:.4f} auc={r.auc:.3f}"
                                         for r in summ.itertuples()), flush=True)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# what the prediction relies on
# ---------------------------------------------------------------------------------------------
def coefficients(folds, columns):
    """Standardised coefficients, one column per fold, plus their mean and the fraction of
    folds that agree in sign with the mean."""
    W = pd.DataFrame({i: f['clf'].coef_.ravel() for i, f in enumerate(folds)}, index=columns)
    out = pd.DataFrame({'mean': W.mean(axis=1), 'sd': W.std(axis=1)})
    out['sign_agreement'] = (np.sign(W).eq(np.sign(out['mean']), axis=0)).mean(axis=1)
    return out


def group_importance(X, y, meta, folds, groups, n_repeats=5, seed=0):
    """Grouped permutation importance on held-out trials: increase in mean log loss when every
    column of a group is shuffled together. Rows are permuted WITHIN session, so session-level
    differences stay put and only the trial-to-trial link to the target is broken.
    Returns one row per group with the mean increase (x 1e3) and its spread across repeats."""
    rng = np.random.default_rng(seed)
    X = X.reset_index(drop=True)
    cols = list(X.columns)
    eids = meta['eid'].to_numpy()
    rows = []
    for name, gcols in groups.items():
        idx = [cols.index(c) for c in gcols]
        deltas = []
        for _ in range(n_repeats):
            total = 0.0
            for f in folds:
                te = f['test']
                Xt = X.iloc[te].to_numpy(dtype=float)
                base = _loss(y[te], f['clf'].predict_proba(f['scaler'].transform(Xt))[:, 1]).sum()
                Xp = Xt.copy()
                e = eids[te]
                for s in np.unique(e):
                    r = np.where(e == s)[0]
                    Xp[np.ix_(r, idx)] = Xt[np.ix_(rng.permutation(r), idx)]
                pert = _loss(y[te], f['clf'].predict_proba(f['scaler'].transform(Xp))[:, 1]).sum()
                total += pert - base
            deltas.append(total / len(y) * 1e3)
        rows.append(dict(group=name, n_cols=len(gcols), d_logloss_x1e3=np.mean(deltas),
                         sd=np.std(deltas)))
    return pd.DataFrame(rows).sort_values('d_logloss_x1e3', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# does it change along LD1?
# ---------------------------------------------------------------------------------------------
def per_unit_scores(y, preds, meta, unit='mouse_name', min_events=5):
    """Held-out scores per mouse (or session) for every model in `preds`, with that unit's LD1.
    Units with fewer than `min_events` events are dropped (AUC is meaningless there)."""
    rows = []
    for u, idx in meta.groupby(unit).indices.items():
        if y[idx].sum() < min_events:
            continue
        r = {unit: u, 'lda_1': meta['lda_1'].iloc[idx].mean(), 'lab': meta['lab'].iloc[idx[0]]}
        for m, p in preds.items():
            if m == 'null':
                continue
            s = scores(y[idx], p[idx], preds['null'][idx])
            r[f'{m}_auc'], r[f'{m}_skill'] = s['auc'], s['ll_skill']
        r['events'] = int(y[idx].sum())
        r['n'] = len(idx)
        rows.append(r)
    out = pd.DataFrame(rows)
    if {'full_skill', 'base_skill'} <= set(out.columns):
        out['gain_skill'] = out['full_skill'] - out['base_skill']
        out['gain_auc'] = out['full_auc'] - out['base_auc']
    return out


def ld1_interaction(Xb, Xs, y, meta, cv='mouse', n_splits=5, ld='lda_1'):
    """Does the syllable -> transition mapping change with LD1?

    Compares, out of fold, the full model against the full model plus (syllable x LD1) and LD1
    main-effect terms (LD1 z-scored across mice). Mice are held out, so the interaction must
    generalise to animals at LD1 values it was not fitted on. The test is a per-mouse paired
    comparison of held-out log loss (Wilcoxon across mice), so mice are the unit.

    Returns (summary dict, per-mouse DataFrame, interaction coefficients DataFrame)."""
    mouse_mean = meta.groupby('mouse_name')[ld].mean()
    z = ((meta[ld] - mouse_mean.mean()) / mouse_mean.std()).to_numpy()[:, None]
    full = pd.concat([Xb, Xs], axis=1).reset_index(drop=True)
    inter = pd.DataFrame(Xs.to_numpy() * z, columns=[f'{c} x LD1' for c in Xs.columns])
    Xi = pd.concat([full, inter, pd.DataFrame({'LD1': z.ravel()})], axis=1)

    p_f, p0, _ = cross_validate(full, y, meta, cv, n_splits)
    p_i, _, folds_i = cross_validate(Xi, y, meta, cv, n_splits)
    lf, li = _loss(y, p_f), _loss(y, p_i)
    per_mouse = pd.DataFrame({'mouse_name': meta['mouse_name'], 'lf': lf, 'li': li, 'y': y,
                              ld: meta[ld]}).groupby('mouse_name').agg(
        d_logloss=('li', 'mean'), lf=('lf', 'mean'), events=('y', 'sum'), lda=(ld, 'mean'))
    per_mouse['d_logloss'] = per_mouse['d_logloss'] - per_mouse['lf']
    stat = wilcoxon(per_mouse['d_logloss'])
    summary = dict(full=scores(y, p_f, p0), with_ld1_interaction=scores(y, p_i, p0),
                   median_mouse_d_logloss=per_mouse['d_logloss'].median(),
                   frac_mice_improved=(per_mouse['d_logloss'] < 0).mean(),
                   wilcoxon_p=stat.pvalue)
    coefs = coefficients(folds_i, Xi.columns)
    coefs = coefs[coefs.index.str.endswith(' x LD1')]
    return summary, per_mouse.drop(columns='lf').reset_index(), coefs


def ld1_bins(meta, n_bins=3, ld='lda_1'):
    """Assign each MOUSE (not session) to an LD1 bin by its mean LD1, so bins share no mice."""
    m = meta.groupby('mouse_name')[ld].mean()
    b = pd.qcut(m, n_bins, labels=False)
    return meta['mouse_name'].map(b).to_numpy()


def transfer_matrix(Xb, Xs, y, meta, n_bins=3, n_splits=5, model='full', ld='lda_1'):
    """Train in one LD1 bin, test in another. Diagonal cells use mouse-grouped CV within the
    bin; off-diagonal cells train on all of bin i and test on all of bin j (disjoint mice).
    If the mapping is shared along LD1 the off-diagonal skill matches the diagonal; if it
    changes, models travel worse the further apart the bins are.

    Returns (skill matrix, auc matrix, per-bin coefficient table)."""
    X = {'base': Xb, 'syll': Xs, 'full': pd.concat([Xb, Xs], axis=1)}[model].reset_index(drop=True)
    bins = ld1_bins(meta, n_bins, ld)
    skill = np.full((n_bins, n_bins), np.nan)
    auc = np.full((n_bins, n_bins), np.nan)
    coefs = {}
    for i in range(n_bins):
        tr = np.where(bins == i)[0]
        mi = meta.iloc[tr].reset_index(drop=True)
        p, p0, folds = cross_validate(X.iloc[tr], y[tr], mi, 'mouse', n_splits)
        s = scores(y[tr], p, p0)
        skill[i, i], auc[i, i] = s['ll_skill'], s['auc']
        scaler, clf = fit_one(X.iloc[tr].to_numpy(float), y[tr], mi['mouse_name'].to_numpy())
        coefs[f'LD1 bin {i}'] = clf.coef_.ravel()
        for j in range(n_bins):
            if j == i:
                continue
            te = np.where(bins == j)[0]
            pj = clf.predict_proba(scaler.transform(X.iloc[te].to_numpy(float)))[:, 1]
            s = scores(y[te], pj, np.full(len(te), y[tr].mean()))
            skill[i, j], auc[i, j] = s['ll_skill'], s['auc']
    lab = [f'bin {i}' for i in range(n_bins)]
    return (pd.DataFrame(skill, index=[f'train {l}' for l in lab], columns=[f'test {l}' for l in lab]),
            pd.DataFrame(auc, index=[f'train {l}' for l in lab], columns=[f'test {l}' for l in lab]),
            pd.DataFrame(coefs, index=X.columns))


def ld1_correlation(per_unit, col, ld='lda_1'):
    d = per_unit[[ld, col]].dropna()
    r = spearmanr(d[ld], d[col])
    return dict(metric=col, n=len(d), rho=r.statistic, p=r.pvalue)
