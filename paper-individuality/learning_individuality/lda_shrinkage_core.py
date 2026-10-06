"""
The shrinkage LDA scheme of 4_mice/lda_shrinkage.ipynb, as functions the learning notebooks share.
====================================================================================================
Why (see lda_shrinkage.ipynb's header for the measurements): LDA whitens by the within-mouse
scatter S_W. With more features than within-class degrees of freedom S_W is singular and the
answer depends on the basis -- which is what the PCA step used to hide, and why its component
count changed the score. Shrinkage mixes S_W with a scaled identity,

    S_W -> (1 - s) S_W + s (tr S_W / p) I,

so the solution is unique at any feature count. Hence:
  * NO PCA and NO StandardScaler: the LDA sees the session x feature matrix itself.
  * a FIXED shrinkage chosen by cross-validation (never sklearn's 'auto', whose Ledoit-Wolf
    intensity is basis dependent), and never on the score that is reported;
  * scores: LDA (solver='eigen', uniform priors) trained on at most N_PER_MOUSE sessions per mouse;
  * embeddings: every mouse counted once in the between-mouse scatter (fit_mouse_weighted), the
    shrinkage picked by the one-standard-error rule, coordinates centred on the mean of the
    mouse means so LD = 0 is the average mouse.

The functions are copied from lda_shrinkage.ipynb with its constants as arguments; the defaults are
that notebook's values. proficient_design() reproduces its design matrix (QC sheet, missing-data
screen, >= 3 sessions per mouse).
"""
import warnings

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# EXPECTED, as in lda_shrinkage.ipynb: a mouse left with a single training session (an inner fold,
# or a 1-session timepoint) makes sklearn warn when it forms that class's covariance. The class adds
# a zero matrix to the pooled within-class scatter, which the shrinkage term fills in.
warnings.filterwarnings('ignore', message='Only one sample available', category=UserWarning)

SHRINKAGE_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)
N_PER_MOUSE = 3
INNER_FOLDS = 3
EMBED_FOLDS = 20
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']


# ------------------------------------------------------------------------------------------------
# Scores
# ------------------------------------------------------------------------------------------------
def balanced_draw(y_sub, rng, cap=N_PER_MOUSE):
    """Indices into y_sub taking at most `cap` sessions per mouse. A MAXIMUM, not a minimum:
    every mouse stays a class and sparse ones contribute what they have."""
    idx = []
    for m in np.unique(y_sub):
        m_idx = np.where(y_sub == m)[0]
        idx.extend(rng.choice(m_idx, min(cap, len(m_idx)), replace=False))
    return np.array(idx)


def fit_lda(Xt, yt, shrinkage, n_components=None):
    """solver='eigen' (identical predictions to 'lsqr', and it exposes scalings_), uniform priors."""
    k = len(np.unique(yt))
    return LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage, priors=np.ones(k) / k,
                                      n_components=n_components).fit(Xt, yt)


def select_shrinkage(Xt, yt, rng, grid=SHRINKAGE_GRID, n_folds=INNER_FOLDS, cap=N_PER_MOUSE):
    """Inner CV on TRAINING sessions only: each fold holds out one session per mouse and trains
    on a balanced draw of the rest, the same shape as the outer protocol."""
    wins = np.zeros(len(grid))
    for _ in range(n_folds):
        held = np.array([rng.choice(np.where(yt == m)[0]) for m in np.unique(yt)])
        rest = np.setdiff1d(np.arange(len(yt)), held)
        tr = rest[balanced_draw(yt[rest], rng, cap)]
        for gi, s in enumerate(grid):
            wins[gi] += np.mean(fit_lda(Xt[tr], yt[tr], s).predict(Xt[held]) == yt[held])
    return grid[int(np.argmax(wins))]


def loso_score(X, y, rng, n_repeats=1, grid=SHRINKAGE_GRID, inner_folds=INNER_FOLDS, cap=N_PER_MOUSE,
               verbose=True):
    """Leave-one-session-out mouse identification with the shrinkage chosen by nested CV in every
    outer fold (lda_shrinkage.ipynb's score). Returns per-session hits (repeats x sessions) for
    the true and the shuffled labels, and the shrinkage chosen in each fold."""
    X, y = np.asarray(X, float), np.asarray(y)
    n = len(X)
    hit = np.zeros((n_repeats, n), bool)
    hit_shuf = np.zeros((n_repeats, n), bool)
    chosen = []
    for r in range(n_repeats):
        for t in range(n):
            tr_all = np.setdiff1d(np.arange(n), t)
            s = select_shrinkage(X[tr_all], y[tr_all], rng, grid, inner_folds, cap)
            chosen.append(s)
            sel = tr_all[balanced_draw(y[tr_all], rng, cap)]
            if y[t] not in y[sel]:
                raise ValueError('the held-out mouse is not a training class: every mouse needs >= 2 sessions')
            hit[r, t] = fit_lda(X[sel], y[sel], s).predict(X[t:t + 1])[0] == y[t]
            hit_shuf[r, t] = fit_lda(X[sel], rng.permutation(y[sel]), s).predict(X[t:t + 1])[0] == y[t]
        if verbose:
            print(f'  repeat {r + 1}/{n_repeats}: {hit[r].mean():.3f} (shuffled {hit_shuf[r].mean():.3f})')
    return dict(hit=hit, hit_shuffled=hit_shuf, chosen=np.array(chosen),
                score=hit.mean(), shuffled=hit_shuf.mean(), n_mice=len(np.unique(y)))


def mouse_ci(hit_per_session, mice, n_boot=2000, seed=0):
    """95% CI of the accuracy by resampling MICE (a mouse may contribute several sessions)."""
    hit_per_session, mice = np.asarray(hit_per_session, float), np.asarray(mice)
    um = np.unique(mice)
    of = {m: np.where(mice == m)[0] for m in um}
    rng = np.random.default_rng(seed)
    boot = [hit_per_session[np.concatenate([of[m] for m in rng.choice(um, len(um))])].mean()
            for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


# ------------------------------------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------------------------------------
def fit_mouse_weighted(Xt, yt, shrinkage):
    """Equal-mouse-weight LDA -> (scalings [features x C-1], grand mean, eigenvalues). S_B is the
    covariance of the mouse means (each counted once); S_W the within-mouse scatter pooled over
    sessions, shrunk as sklearn does."""
    classes = np.unique(yt)
    M = np.stack([Xt[yt == c].mean(axis=0) for c in classes])
    grand = M.mean(axis=0)
    Sb = np.cov((M - grand).T, bias=True)
    R = Xt - M[np.searchsorted(classes, yt)]
    Sw = R.T @ R / (len(Xt) - len(classes))
    Sw = (1 - shrinkage) * Sw + shrinkage * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    evals, evecs = linalg.eigh(Sb, Sw)
    order = np.argsort(evals)[::-1][:len(classes) - 1]
    return evecs[:, order], grand, evals[order]


def select_shrinkage_embedding(Xt, yt, rng, grid=SHRINKAGE_GRID, n_folds=EMBED_FOLDS):
    """CV for the embedding's shrinkage: hold out one session per mouse, fit the mouse-weighted
    LDA on the rest, assign each held-out session to the nearest mouse centroid. ONE-SE RULE:
    the most regularised value within one SE of the best (stable axes among equally accurate fits)."""
    grid = np.sort(np.asarray(grid, float))
    acc = np.zeros((n_folds, len(grid)))
    for f in range(n_folds):
        held = np.array([rng.choice(np.where(yt == m)[0]) for m in np.unique(yt)])
        rest = np.setdiff1d(np.arange(len(yt)), held)
        for gi, s in enumerate(grid):
            W, _, _ = fit_mouse_weighted(Xt[rest], yt[rest], s)
            Zr = Xt[rest] @ W
            cls = np.unique(yt[rest])
            M = np.stack([Zr[yt[rest] == c].mean(axis=0) for c in cls])
            d2 = ((Xt[held] @ W)[:, None, :] - M[None]) ** 2
            acc[f, gi] = np.mean(cls[d2.sum(axis=2).argmin(axis=1)] == yt[held])
    mean, se = acc.mean(axis=0), acc.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best = int(np.argmax(mean))
    pick = int(np.max(np.where(mean >= mean[best] - se[best])[0]))
    return float(grid[pick]), mean, se


class MouseEmbedding:
    """The mouse-weighted shrinkage LDA embedding, fit on (X, y); .transform(X) = (X - grand) @ W."""

    def __init__(self, X, y, shrinkage=None, rng=None, grid=SHRINKAGE_GRID, n_folds=EMBED_FOLDS):
        X, y = np.asarray(X, float), np.asarray(y)
        yi = pd.factorize(y)[0] if y.dtype.kind not in 'iu' else y
        if shrinkage is None:
            shrinkage, self.cv_mean, self.cv_se = select_shrinkage_embedding(
                X, yi, rng if rng is not None else np.random.default_rng(0), grid, n_folds)
            self.grid = np.sort(np.asarray(grid, float))
        self.shrinkage = float(shrinkage)
        self.W, self.grand, self.evals = fit_mouse_weighted(X, yi, self.shrinkage)
        Z = (X - self.grand) @ self.W
        between = pd.DataFrame(Z).groupby(yi).mean().var(axis=0, ddof=0).to_numpy()
        self.axis_share = between / between.sum()     # share of between-mouse variance per axis
        self.n_mice = len(np.unique(yi))

    def transform(self, X, n_components=None):
        Z = (np.asarray(X, float) - self.grand) @ self.W
        return Z if n_components is None else Z[:, :n_components]

    def describe_cv(self):
        if not hasattr(self, 'cv_mean'):
            return f'shrinkage {self.shrinkage} (given)'
        return ('shrinkage CV (mean +/- SE): '
                + ', '.join(f'{s} {a:.3f}+/-{e:.3f}' for s, a, e in zip(self.grid, self.cv_mean, self.cv_se))
                + f'  -> {self.shrinkage} (most regularised within 1 SE of the best)')


# ------------------------------------------------------------------------------------------------
# The proficient design matrix of lda_shrinkage.ipynb
# ------------------------------------------------------------------------------------------------
def proficient_design(syllable_file, prob_sessions, n_paw_states=8, n_wheel_states=0,
                      max_session_missing=0.1, min_sessions_per_mouse=3, verbose=True):
    """QC sheet -> missing-data screen -> >= min_sessions_per_mouse -> binarize -> session means.
    Returns (session x feature DataFrame indexed by session, mouse per session)."""
    from syllable_data import binarize
    seq = pd.read_parquet(syllable_file)
    seq['session'] = seq['sample'].str[:36]
    seq = seq.loc[~seq['session'].isin(prob_sessions)].reset_index(drop=True)
    if max_session_missing is not None:
        nan_frac = np.isnan(np.stack(seq['binned_sequence'].to_numpy())).mean(axis=1)
        by_session = pd.Series(nan_frac, index=seq['session'].to_numpy()).groupby(level=0).mean()
        drop = by_session[by_session > max_session_missing]
        seq = seq.loc[~seq['session'].isin(drop.index)].reset_index(drop=True)
    counts = seq[['mouse_name', 'session']].drop_duplicates().groupby('mouse_name')['session'].count()
    seq = seq.loc[seq['mouse_name'].isin(counts[counts >= min_sessions_per_mouse].index)].reset_index(drop=True)
    trials = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'], columns=['broader_label'],
                        values='binned_sequence').reset_index().dropna().sort_values(by='session'))
    use = np.vstack(trials[EPOCHS].apply(lambda r: np.hstack(r), axis=1))
    feats = binarize(use, n_paw_states, n_wheel_states)
    frame = pd.DataFrame(feats)
    frame['session'] = trials['session'].values
    session_feats = frame.groupby('session', sort=False)[np.arange(feats.shape[1])].mean()
    mapping = trials[['session', 'mouse_name']].drop_duplicates().set_index('session')['mouse_name']
    if verbose:
        print(f'proficient design: {len(session_feats)} sessions x {session_feats.shape[1]} features, '
              f'{mapping.nunique()} mice')
    return session_feats, mapping.reindex(session_feats.index)
