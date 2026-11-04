"""
SYLLABLE REPRESENTATION SWEEPS -- is the representation an arbitrary choice?
============================================================================
Mouse identification (lda_shrinkage_figures' leave-one-session-out protocol: same session
screens, same nested shrinkage, same per-mouse training cap) as one choice of the syllable
representation is varied, everything else fixed at the paper's values:

  'bins'  bins per epoch B (time-warping resolution), paw states fixed at 8. The syllables are
          re-binned from the SAME states files (data/states_files, the production run behind
          8_k_10_bin_syllables_02-10-2026) with syllable_pipeline's own binning (mode syllable
          per chunk). B = 10 reproduces the production file exactly.
              4 epochs x B bins x 9 columns (7 paw one-hots, whisk, lick) = 36 B features
  'k'     number of paw states K, 10 bins per epoch. The syllables of segmentation/paw_k_sweep.py
          (3.3's production clustering refitted at K); K = 8 is the production file.
              4 epochs x 10 bins x (K + 1) columns = 40 (K + 1) features

Only the LDA and the syllable-space (z-scored, nearest centroid) scores are computed per fold,
not the shuffled-axis / decorrelation / PCA extras of the full notebook.

Cached per value in data/syllable_sweeps/<sweep>/, so downstream analyses (e.g. LD1 agreement
across values) can be run later without regenerating anything:
    syllables_<v>.pqt   (bins only) the re-binned syllables, same columns as the production file
    features_<v>.pqt    session x feature matrix after the screens, + 'mouse_name' column
    loso_<v>.pkl        per-fold results: repeat, test_idx, shrink, pred, pred_syl, correct, correct_syl

    python syllable_sweeps.py bins            # every value, skipping what is cached
    python syllable_sweeps.py k 4 6           # some values
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import glob
import pickle
import pathlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
sys.path.insert(0, str(PI / 'segmentation'))
sys.path.insert(0, str(PI / 'learning_individuality'))
import syllable_pipeline as sp                                   # noqa: E402
from session_filters import exclusions_by_timepoint             # noqa: E402

DATA = PI / 'data'
OUT = DATA / 'syllable_sweeps'
REFERENCE = DATA / '8_k_10_bin_syllables_02-10-2026'             # the paper's syllables; its sessions
STATES_DIR = DATA / 'states_files'
SWEEPS = {'bins': [1, 2, 5, 7, 10, 15, 20], 'k': [4, 6, 8, 10, 12, 16]}

# ---- lda_shrinkage_figures' protocol (its defaults; see that notebook for the reasoning) ----
QC_STRICTNESS = 'filtered_out'
MAX_SESSION_MISSING = 0.1
MIN_SESSIONS_PER_MOUSE = 3
N_PAW_STATES = 8                  # the paper's; the 'k' sweep passes its own
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_PER_MOUSE = 3
SHRINKAGE_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)
INNER_FOLDS = 3
N_REPEATS = 3                     # the notebook uses 20 for panel b's spread; 3 is enough for a mean
N_JOBS = -1


# ------------------------------------------------------------------------------------------------
# Syllables at B bins per epoch
# ------------------------------------------------------------------------------------------------
def rebinned_syllables(B, n_jobs=8):
    path = OUT / 'bins' / f'syllables_{B}.pqt'
    if path.exists():
        return pd.read_parquet(path)
    ref = pd.read_parquet(REFERENCE, columns=['sample', 'mouse_name'])
    sessions = (ref.assign(eid=ref['sample'].str[:36])[['eid', 'mouse_name']].drop_duplicates()
                .itertuples(index=False, name=None))
    paths = [STATES_DIR / f'8_states_file_{e}_{m}' for e, m in sessions]
    missing = [p for p in paths if not p.exists()]
    assert not missing, f'{len(missing)} states files missing, e.g. {missing[0]}'
    parts = Parallel(n_jobs=n_jobs)(delayed(sp._syllables_job)(str(p), B) for p in paths)
    seq = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq.to_parquet(path)
    return seq


def k_syllables(k):
    """The paw_k_sweep syllables at K paw states (the production file for K = 8)."""
    if k == N_PAW_STATES:
        return pd.read_parquet(REFERENCE)
    files = sorted(glob.glob(sp.syllables_glob('proficient', 'both_paws', 'own', 'standard', k=k, target_length=10)))
    assert files, f'no syllables for K = {k}: run segmentation/paw_k_sweep.py first'
    return pd.read_parquet(files[-1])


def syllables(sweep, v):
    """-> (syllable table, number of paw states) for one value of one sweep."""
    return (rebinned_syllables(v), N_PAW_STATES) if sweep == 'bins' else (k_syllables(v), v)


# ------------------------------------------------------------------------------------------------
# Session x feature matrix (lda_shrinkage_figures' binarize / filter_sequences / build_design_matrix)
# ------------------------------------------------------------------------------------------------
def binarize(use_sequences, n_paw_states=N_PAW_STATES):
    n_trials, timesteps = use_sequences.shape
    per_step = n_paw_states + 2
    out = np.zeros((n_trials, timesteps * per_step))
    for t in range(timesteps):
        vals = use_sequences[:, t]
        valid = ~np.isnan(vals)
        labels = vals[valid].astype(int)
        start = t * per_step
        if len(labels):
            rows = np.arange(n_trials)[valid]
            out[rows, start + labels % n_paw_states] = 1
            out[valid, start + n_paw_states] = (labels // n_paw_states) % 2
            out[valid, start + n_paw_states + 1] = labels // (n_paw_states * 2)
        if (~valid).any():
            out[~valid, start:start + per_step] = np.nan
    return np.delete(out, [t * per_step + 1 for t in range(timesteps)], axis=1)


def filter_sequences(seq, prob_sessions):
    seq = seq.loc[~seq['session'].isin(prob_sessions)].reset_index(drop=True)
    nan_frac = np.isnan(np.stack(seq['binned_sequence'].to_numpy())).mean(axis=1)
    by_session = pd.Series(nan_frac, index=seq['session'].to_numpy()).groupby(level=0).mean()
    seq = seq.loc[~seq['session'].isin(by_session[by_session > MAX_SESSION_MISSING].index)].reset_index(drop=True)
    counts = seq[['mouse_name', 'session']].drop_duplicates().groupby('mouse_name')['session'].count()
    return seq.loc[seq['mouse_name'].isin(counts[counts >= MIN_SESSIONS_PER_MOUSE].index)].reset_index(drop=True)


def features(sweep, v):
    path = OUT / sweep / f'features_{v}.pqt'
    if path.exists():
        f = pd.read_parquet(path)
        return f.drop(columns='mouse_name'), f['mouse_name']
    seq, n_paw = syllables(sweep, v)
    seq = seq.copy()
    seq['session'] = seq['sample'].str[:36]
    seq = filter_sequences(seq, sorted(exclusions_by_timepoint(QC_STRICTNESS)['Proficient']))
    trials = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'],
                        columns=['broader_label'], values='binned_sequence')
              .reset_index().dropna().sort_values(by='session'))
    feats = binarize(np.vstack(trials[EPOCHS].apply(lambda r: np.hstack(r), axis=1)), n_paw)
    frame = pd.DataFrame(feats)
    frame['session'] = trials['session'].values
    sf = frame.groupby('session', sort=False)[np.arange(feats.shape[1])].mean()
    sf.columns = sf.columns.astype(str)
    mouse = trials[['session', 'mouse_name']].drop_duplicates().set_index('session')['mouse_name'].reindex(sf.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.assign(mouse_name=mouse).to_parquet(path)
    return sf, mouse


# ------------------------------------------------------------------------------------------------
# Leave-one-session-out (lda_shrinkage_figures' loso_fold, LDA + syllable space only)
# ------------------------------------------------------------------------------------------------
def fold_rng(*key):
    return np.random.default_rng([2026, *key])


def balanced_draw(y_sub, rng, cap=N_PER_MOUSE):
    idx = []
    for m in np.unique(y_sub):
        m_idx = np.where(y_sub == m)[0]
        idx.extend(rng.choice(m_idx, min(cap, len(m_idx)), replace=False))
    return np.array(idx)


def fit_lda(Xt, yt, shrinkage):
    k = len(np.unique(yt))
    return LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage, priors=np.ones(k) / k).fit(Xt, yt)


def select_shrinkage(Xt, yt, rng):
    wins = np.zeros(len(SHRINKAGE_GRID))
    for _ in range(INNER_FOLDS):
        held = np.array([rng.choice(np.where(yt == m)[0]) for m in np.unique(yt)])
        rest = np.setdiff1d(np.arange(len(yt)), held)
        tr = rest[balanced_draw(yt[rest], rng)]
        for gi, s in enumerate(SHRINKAGE_GRID):
            wins[gi] += np.mean(fit_lda(Xt[tr], yt[tr], s).predict(Xt[held]) == yt[held])
    return SHRINKAGE_GRID[int(np.argmax(wins))]


def loso_fold(X, y, repeat, test_idx):
    warnings.filterwarnings('ignore', message='Only one sample available', category=UserWarning)
    rng = fold_rng(0, repeat, test_idx)
    train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
    X_tr_full, y_tr_full = X[train_idx], y[train_idx]
    shrink = select_shrinkage(X_tr_full, y_tr_full, rng)
    sel = balanced_draw(y_tr_full, rng)
    X_tr, y_tr = X_tr_full[sel], y_tr_full[sel]
    x_te = X[test_idx][None]
    pred = fit_lda(X_tr, y_tr, shrink).predict(x_te)[0]
    mu, sd = X_tr_full.mean(axis=0), X_tr_full.std(axis=0) + 1e-12
    classes = np.unique(y_tr)
    M = np.stack([((X_tr[y_tr == c] - mu) / sd).mean(axis=0) for c in classes])
    pred_syl = classes[np.argmin(((((x_te - mu) / sd) - M) ** 2).sum(axis=1))]
    return dict(repeat=repeat, test_idx=test_idx, shrink=shrink, pred=pred, pred_syl=pred_syl)


def loso(sweep, v):
    path = OUT / sweep / f'loso_{v}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    sf, mouse = features(sweep, v)
    X = sf.to_numpy(float)
    y, mice = pd.factorize(mouse.to_numpy())
    t0 = datetime.now()
    rows = Parallel(n_jobs=N_JOBS)(delayed(loso_fold)(X, y, r, t)
                                   for r in range(N_REPEATS) for t in range(len(X)))
    res = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    res['correct'] = res['pred'] == y[res['test_idx']]
    res['correct_syl'] = res['pred_syl'] == y[res['test_idx']]
    res.update(sweep=sweep, value=v, sessions=sf.index.to_numpy(), mouse_of_session=mouse.to_numpy(), mice=np.asarray(mice),
               n_features=X.shape[1], n_repeats=N_REPEATS, seconds=(datetime.now() - t0).seconds)
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    return res


if __name__ == '__main__':
    sweep = sys.argv[1]
    for v in ([int(a) for a in sys.argv[2:]] or SWEEPS[sweep]):
        r = loso(sweep, v)
        print(f"{sweep} = {v:2d}: {r['n_features']:3d} features, {len(r['sessions'])} sessions, "
              f"{len(r['mice'])} mice | LDA {r['correct'].mean():.3f}  syllable space {r['correct_syl'].mean():.3f} "
              f"| shrinkage {dict(pd.Series(r['shrink']).value_counts().sort_index())} | {r['seconds']} s", flush=True)
