"""
Syllables vs continuous movement as predictors of the NM photometry, before stimulus onset only.
Functions and cache runner; the results are in neuromodulators/10_syllables_vs_movement.ipynb.

    python neuromodulators/syllables_vs_movement.py [n_jobs]

Data, per fiber-session: the stimulus-aligned photometry snippet from -1 to 0 s, unmasked (the previous feedback is
always >= ~1.2 s before stimulus onset, so the whole second is pre-stimulus), averaged into 100 ms bins (10 per
trial). Predictors are read from the 30 Hz states file at any time, so they can be lagged beyond the snippet:
  movement   log(1 + ·) of paw speed, |wheel velocity|, whisker ME, lick rate   (4 signals, ns.VIGOR)
  syllables  paw states 1..7 (one-hot against paw 0), whisking, licking        (9 indicators; bin mean = fraction)
  lags       predictor at t - lag, lag in 100 ms steps from -0.5 to +0.5 s (+ = movement before the photometry),
             or lag 0 only. Each lag of each signal is its own column.
  baseline   what is known before the stimulus: trial position, previous outcome, block (pLeft 0.2 / 0.8 vs 0.5);
             the encoding model also gets one-hots of the time bin (the mean pre-stimulus time course).

Two framings, both banded ridge (one penalty per block: baseline, movement, syllables) with cross-validated R²:
  encoding   all bins of all trials in one regression, kernels shared across time; 5 folds of whole trials; the
             penalties picked inside each training set by 3-fold CV over trials. R² is also split into between-trial
             (trial means) and within-trial (deviations from the trial mean) parts. Models: baseline, + movement,
             + syllables, + both, and + both without each movement signal / each syllable group (unique shares).
  timestep   one regression across trials per 100 ms bin (notebook 7's design, pre-stim only); baseline + lagged
             predictors around that bin; penalties by leave-one-out (trials are independent there).
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from Functions import nm_syllables as ns  # noqa: E402

OUT = ns.DATA / 'syllables_vs_movement'
OUT.mkdir(exist_ok=True)
ENCODING_FILE = OUT / f'encoding_{ns.RUN}.pkl'
TIMESTEP_FILE = OUT / f'timestep_{ns.RUN}.pkl'

WINDOW = (-1.0, 0.0)
BIN = 3                       # 30 Hz samples per bin -> 100 ms bins
LAGS = np.arange(-5, 6)       # in bins: -0.5 .. +0.5 s
N_BINS = 10
SYLLABLE_GROUPS = ['paw state', 'whisking', 'licking']
MIN_TRIALS = 60


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
def load_prestim(eid, lags=LAGS):
    """One dict per fiber: meta (go trials), y (n x N_BINS photometry), and the binned predictors on an extended
    bin grid, mov (4 x n x E) and syl (9 x n x E), where bin b at lag L is column b + pad - L."""
    rec = pd.read_csv(ns.ARCHIVE / 'recordings.csv')
    rec = rec[rec['eid'] == eid]
    responses = ns.h5_loaders.load_responses(eid)
    tr = ns.trial_table(eid)
    states = pd.read_parquet(ns.states_files()[eid])
    states = states.assign(**{'paw speed': ns.paw_speed(states), 'lick rate': ns.lick_rate(states),
                              'avg_wheel_vel': states['avg_wheel_vel'].abs()})
    pad = int(max(abs(lags.min()), abs(lags.max())))
    out = []
    for _, r in rec.iterrows():
        R = responses[r['brain_region']]
        time = R['time'].values
        sel = np.flatnonzero((time >= WINDOW[0] - 1e-9) & (time < WINDOW[1] - 1e-9))[:N_BINS * BIN]
        trials = R['trial'].values[np.isin(R['trial'].values, tr.index)]
        t = tr.loc[trials]
        rows = pd.Index(R['trial'].values).get_indexer(trials)
        P = R.sel(event='stimOnTrigger_times').values[rows][:, sel]
        y = P.reshape(len(P), N_BINS, BIN).mean(2)
        dt = time[1] - time[0]
        fine = time[sel[0]] + dt * np.arange(-pad * BIN, (N_BINS + pad) * BIN)
        g = ns.on_grid(states, list(ns.MOVEMENT.values()) + ['most_likely_states'],
                       t['stimOnTrigger_times'].to_numpy(), fine)
        n, E = len(t), N_BINS + 2 * pad
        binned = lambda a: a.reshape(n, E, BIN).mean(2)   # NaN if any sample in the bin is NaN
        mov = np.stack([binned(np.log1p(np.abs(g[ns.MOVEMENT[v]]))) for v in ns.VIGOR])
        _, F = ns.factored_features(g['most_likely_states'])
        syl = np.stack([binned(f) for f in F[1:]])         # paw 1..7, whisking, licking
        meta = t.reset_index(drop=True).assign(eid=eid, subject=r['subject'], region=r['brain_region'],
                                               target_NM=r['target_NM'])
        out.append({'meta': meta, 'y': y, 'mov': mov, 'syl': syl, 'pad': pad})
    return out


def baseline_trial(meta):
    """(n x k) trial-level covariates known before the stimulus."""
    p = meta['probabilityLeft'].round(1)
    return np.column_stack([meta['trial_frac'], meta['prev_correct'], p.eq(0.2), p.eq(0.8)]).astype(float)


def lagged(A, pad, lags, b):
    """(n x signals*lags) predictors for bin b from A (signals x n x E)."""
    return np.concatenate([A[:, :, b + pad - L].T for L in lags], 1)


# ------------------------------------------------------------------------------------------------
# Banded ridge: one penalty per feature space, chosen inside the training set
# ------------------------------------------------------------------------------------------------
# With a single penalty, adding a block of many weak columns (99 lagged syllable columns) forces a compromise
# penalty that also shrinks the useful block, so R²(both) can fall below R²(movement) and a "unique" share goes
# negative for no reason but the fit. Banded ridge (Nunez-Elizalde et al. 2019) scales each block by its own factor
# before one shared penalty, i.e. penalty alpha / scale² per block; scale and alpha are picked together by CV.
SCALES = np.array([1 / 3, 1., 3.])


def _combos(n_bands):
    """Scale factor per band: the first band (baseline) fixed at 1, the others from SCALES."""
    if n_bands == 1:
        return [np.ones(1)]
    grids = np.meshgrid(*[SCALES] * (n_bands - 1), indexing='ij')
    return [np.r_[1., c] for c in np.stack([g.ravel() for g in grids], 1)]


def _scaled(Z, band, scale):
    return Z * scale[band]


def ridge_grouped(Xtr, ytr, gtr, Xte, band, alphas=ns.ALPHAS, n_inner=3, seed=0):
    """Banded ridge: fit on (Xtr, ytr), band = block index of each column; scales and alpha chosen by n_inner-fold
    CV over the groups gtr (trials); predict Xte."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Z, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    ug = np.unique(gtr)
    fold_of = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug)) % n_inner))
    f = np.array([fold_of[x] for x in gtr])
    combos = _combos(band.max() + 1)
    err = np.zeros((len(combos), len(alphas)))
    for c, sc in enumerate(combos):
        Zs = _scaled(Z, band, sc)
        for k in range(n_inner):
            tr, te = f != k, f == k
            m = ytr[tr].mean()
            U, s, Vt = np.linalg.svd(Zs[tr], full_matrices=False)
            Uy = U.T @ (ytr[tr] - m)
            P = Zs[te] @ Vt.T                                      # test rows in the SVD basis
            shrink = s / (s[None] ** 2 + alphas[:, None])          # alphas x rank
            err[c] += (((ytr[te] - m)[None] - (shrink * Uy) @ P.T) ** 2).sum(1)
    c, a = np.unravel_index(np.argmin(err), err.shape)
    Zs, Zst = _scaled(Z, band, combos[c]), _scaled(Zte, band, combos[c])
    U, s, Vt = np.linalg.svd(Zs, full_matrices=False)
    return ym + Zst @ (Vt.T @ (s / (s ** 2 + alphas[a]) * (U.T @ (ytr - ym))))


def ridge_loo_r2(X, y, band, folds, alphas=ns.ALPHAS):
    """CV R² of a banded ridge, scales and alpha picked per training fold by leave-one-out (independent rows)."""
    pred = np.empty_like(y)
    combos = _combos(band.max() + 1)
    for test in folds:
        train = np.ones(len(y), bool); train[test] = False
        mu, sd = X[train].mean(0), X[train].std(0)
        sd[sd == 0] = 1
        Z, Zt = (X[train] - mu) / sd, (X[test] - mu) / sd
        ym = y[train].mean(); yc = y[train] - ym
        best = (np.inf, None)
        for sc in combos:
            U, s, Vt = np.linalg.svd(_scaled(Z, band, sc), full_matrices=False)
            Uy = U.T @ yc
            shrink = s ** 2 / (s ** 2 + alphas[:, None])
            fit = (U[None] * (shrink * Uy)[:, None, :]).sum(-1)
            hat = (U[None] ** 2 * shrink[:, None, :]).sum(-1) + 1 / len(yc)
            loo = (((yc - fit) / (1 - hat)) ** 2).mean(1)
            i = np.argmin(loo)
            if loo[i] < best[0]:
                best = (loo[i], (sc, alphas[i]))
        sc, a = best[1]
        U, s, Vt = np.linalg.svd(_scaled(Z, band, sc), full_matrices=False)
        pred[test] = ym + _scaled(Zt, band, sc) @ (Vt.T @ (s / (s ** 2 + a) * (U.T @ yc)))
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


# ------------------------------------------------------------------------------------------------
# Framing 1: encoding model over all pre-stimulus bins
# ------------------------------------------------------------------------------------------------
def encoding_models(fib, lags=LAGS, n_folds=5, seed=0):
    """CV R² (total / between-trial / within-trial) of each model for one fiber. Returns a dict of arrays."""
    meta, y, mov, syl, pad = fib['meta'], fib['y'], fib['mov'], fib['syl'], fib['pad']
    base_t = baseline_trial(meta)
    n = len(meta)
    # per bin: baseline, movement and syllable columns; keep a trial only if every bin is complete
    B = np.stack([np.column_stack([base_t, np.eye(N_BINS)[np.full(n, b)][:, 1:]]) for b in range(N_BINS)], 1)
    M = np.stack([lagged(mov, pad, lags, b) for b in range(N_BINS)], 1)          # n x bins x (4*L)
    S = np.stack([lagged(syl, pad, lags, b) for b in range(N_BINS)], 1)          # n x bins x (9*L)
    ok = (np.isfinite(y).all(1) & np.isfinite(B).all((1, 2)) & np.isfinite(M).all((1, 2))
          & np.isfinite(S).all((1, 2)))
    if ok.sum() < MIN_TRIALS:
        return None
    nt = ok.sum()
    flat = lambda A: A[ok].reshape(nt * N_BINS, -1)
    B, M, S, Y = flat(B), flat(M), flat(S), y[ok].ravel()
    trial = np.repeat(np.arange(nt), N_BINS)
    nl = len(lags)
    mov_part = np.repeat(np.arange(len(ns.VIGOR)), 1)[None].repeat(nl, 0).ravel()   # column -> signal (lag-major)
    syl_part = np.r_[np.zeros(7), 1, 2].astype(int)[None].repeat(nl, 0).ravel()       # column -> paw/whisk/lick
    bands = lambda *w: np.concatenate([np.full(x, i) for i, x in enumerate(w)])
    nb, nm, ns_ = B.shape[1], M.shape[1], S.shape[1]
    models = {'baseline': (B, bands(nb)), '+ movement': (np.hstack([B, M]), bands(nb, nm)),
              '+ syllables': (np.hstack([B, S]), bands(nb, ns_)), '+ both': (np.hstack([B, M, S]), bands(nb, nm, ns_))}
    for k, v in enumerate(ns.VIGOR):
        keep = mov_part != k
        models[f'both without {v}'] = (np.hstack([B, M[:, keep], S]), bands(nb, keep.sum(), ns_))
    for k, g in enumerate(SYLLABLE_GROUPS):
        keep = syl_part != k
        models[f'both without {g}'] = (np.hstack([B, M, S[:, keep]]), bands(nb, nm, keep.sum()))
    fold = np.random.default_rng(seed).permutation(nt) % n_folds
    res = {}
    for name, (X, band) in models.items():
        pred = np.empty_like(Y)
        for k in range(n_folds):
            te = fold[trial] == k
            pred[te] = ridge_grouped(X[~te], Y[~te], trial[~te], X[te], band, seed=seed)
        yt, pt = Y.reshape(nt, N_BINS), pred.reshape(nt, N_BINS)
        res[name] = (r2(Y, pred), r2(yt.mean(1), pt.mean(1)),
                     r2((yt - yt.mean(1, keepdims=True)).ravel(), (pt - pt.mean(1, keepdims=True)).ravel()))
    return {'models': list(res), 'r2': np.array([res[m] for m in res]), 'n_trials': int(nt)}


# ------------------------------------------------------------------------------------------------
# Framing 2: one model per time bin, across trials
# ------------------------------------------------------------------------------------------------
TIMESTEP_MODELS = ['baseline', '+ movement', '+ syllables', '+ both']


def timestep_models(fib, lags=LAGS, n_folds=5, seed=0):
    """(4 models x N_BINS) CV R² across trials, per bin."""
    meta, y, mov, syl, pad = fib['meta'], fib['y'], fib['mov'], fib['syl'], fib['pad']
    base = baseline_trial(meta)
    rng = np.random.default_rng(seed)
    R2 = np.full((len(TIMESTEP_MODELS), N_BINS), np.nan)
    for b in range(N_BINS):
        M, S = lagged(mov, pad, lags, b), lagged(syl, pad, lags, b)
        ok = np.isfinite(y[:, b]) & np.isfinite(base).all(1) & np.isfinite(M).all(1) & np.isfinite(S).all(1)
        if ok.sum() < MIN_TRIALS:
            continue
        folds = np.array_split(rng.permutation(ok.sum()), n_folds)
        nb, nm, ns_ = base.shape[1], M.shape[1], S.shape[1]
        for k, (X, band) in enumerate([
                (base[ok], np.zeros(nb, int)),
                (np.hstack([base[ok], M[ok]]), np.r_[np.zeros(nb), np.ones(nm)].astype(int)),
                (np.hstack([base[ok], S[ok]]), np.r_[np.zeros(nb), np.ones(ns_)].astype(int)),
                (np.hstack([base[ok], M[ok], S[ok]]), np.r_[np.zeros(nb), np.ones(nm), 2 * np.ones(ns_)].astype(int))]):
            R2[k, b] = ridge_loo_r2(X, y[ok, b], band, folds)
    return R2


LAG_SETS = {'lag 0': np.array([0]), 'lags ±0.5 s': LAGS}


def one_session(eid):
    warnings.filterwarnings('ignore')
    enc, ts = [], []
    try:
        for fib in load_prestim(eid):
            m = fib['meta']
            key = {'eid': eid, 'region': m['region'].iat[0], 'subject': m['subject'].iat[0],
                   'target_NM': m['target_NM'].iat[0]}
            for lag_name, lags in LAG_SETS.items():
                e = encoding_models(fib, lags)
                if e is not None:
                    enc.append({**key, 'lags': lag_name, 'n_trials': e['n_trials'], 'r2': e['r2'],
                                'models': e['models']})
                ts.append({**key, 'lags': lag_name, 'r2': timestep_models(fib, lags)})
    except Exception as ex:  # report, keep going
        print(f'{eid}: {type(ex).__name__}: {ex}', flush=True)
    return enc, ts


if __name__ == '__main__':
    from joblib import Parallel, delayed
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    eids = ns.load_recordings()['eid'].unique()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_session)(e) for e in eids)
    enc = pd.DataFrame([r for e, _ in res for r in e])
    enc.attrs.update(models=enc['models'].iat[0], parts=['total', 'between trials', 'within trials'])
    enc.drop(columns='models').to_pickle(ENCODING_FILE)
    ts = pd.DataFrame([r for _, t in res for r in t])
    ts.attrs.update(models=TIMESTEP_MODELS, bin_centres=WINDOW[0] + (np.arange(N_BINS) + 0.5) * BIN / ns.FS)
    ts.to_pickle(TIMESTEP_FILE)
    print('encoding', enc.shape, 'timestep', ts.shape, flush=True)
