"""
CONFIDENCE INTERVALS FOR SYLLABLES vs RAW (per-session z), DIMENSION-MATCHED
===========================================================================
raw_vs_syllables.py / raw_velocity_vs_syllables.py report point estimates only. This
recomputes the matched pairs keeping every session's held-out prediction, then bootstraps
OVER MICE (the unit that is sampled: sessions of one mouse are not independent), 2,000
resamples, percentile 95% CIs. The two blocks of a pair are resampled TOGETHER, so the CI
of their difference is paired -- same sessions, same resample.

  channel        syllables                     raw, per-session z (dimension-matched)
  paw            paw syllables          280    speed per paw          80
  whisk + lick   whisk / lick states     80    whisker ME + lick       80
  all            the 360 LDA features   360    speed per paw + w + l  160

Metrics (as in zscore_cost.py; every block's columns standardised across sessions first):
  mouse       leave-one-session-out mouse ID, <=3 sessions per mouse, 3 repeats
  mouse|lab   the same after centring every feature within its lab
  lab LOMO    lab ID with every session of the test mouse held out
  lab eta2    mean over features of the lab share of variance

CIs: accuracies (mouse, mouse|lab, lab LOMO) use the mouse bootstrap above. LAB ETA2 DOES NOT:
resampling mice with replacement duplicates animals, leaves fewer distinct mice per lab, and
inflates eta2 in every resample -- the percentile interval then sits entirely ABOVE the
estimate (paw syllables: 0.131, "CI" [0.156, 0.244]). So eta2 and its paired difference get a
leave-one-mouse-out JACKKNIFE interval, estimate +- 1.96 SE, SE^2 = (n-1)/n * sum (theta_-i - mean)^2.

Reads the feature caches of those two scripts and compare_pipelines.py.
Output: ci_segmentation_vs_raw.csv (estimates + CIs + paired differences).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import raw_vs_syllables as RS
from raw_vs_syllables import Z, me, lab_labels, HERE

N_BOOT, SEED = 2000, 0
PAIRS = [('paw', 'syllables', 'raw, speed per paw'),
         ('whisk + lick', 'syllables', 'raw, whisker ME + lick'),
         ('all', 'syllables', 'raw, speed per paw + whisk + lick')]


def loso_hits(X, y, n_repeats=Z.N_REPEATS, seed=0):
    """zscore_cost.loso_score, keeping each session's hit rate over the repeats."""
    X = np.nan_to_num(np.asarray(X, float))
    hits = np.zeros(len(X))
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        for i in range(len(X)):
            tr = np.setdiff1d(np.arange(len(X)), [i])
            ys = y[tr]
            idx = np.concatenate([rng.choice(np.where(ys == m)[0], min(Z.N_PER_MOUSE, (ys == m).sum()),
                                             replace=False) for m in np.unique(ys)])
            hits[i] += Z._fit(X[tr][idx], ys[idx]).predict(X[i:i + 1])[0] == y[i]
    return hits / n_repeats


def lomo_hits(X, labs, mice):
    X = np.nan_to_num(np.asarray(X, float))
    y = pd.factorize(labs)[0]
    hits = np.zeros(len(X))
    for m in np.unique(mice):
        te = mice == m
        hits[te] = Z._fit(X[~te], y[~te]).predict(X[te]) == y[te]
    return hits


def main():
    raw = np.load(HERE / 'raw_vs_syllables_features.npz', allow_pickle=True)['raw'].item()
    vel = np.load(HERE / 'raw_velocity_features.npz', allow_pickle=True)['raw'].item()
    syl, mouse_of = me.build_design_matrix(RS.SYLLABLE_FILE)
    idx = [s for s in syl.index if all(s in raw[k] for k in raw) and all(s in vel[k] for k in vel)]
    mice = mouse_of.loc[idx].to_numpy()
    labs = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx], verbose=False)))
    y = pd.factorize(mice)[0]
    print(f'syllables: {RS.SYLLABLE_FILE}')
    print(f'cohort: {len(idx)} sessions, {len(set(mice))} mice, {len(set(labs))} labs', flush=True)

    S360 = syl.loc[idx].to_numpy(float)
    stack = lambda d, k: np.vstack([d[k][s] for s in idx])          # noqa: E731
    BLOCKS = {
        ('paw', 'syllables'): RS.paw_only(S360),
        ('paw', 'raw, speed per paw'): stack(vel, 'spd_sessz'),
        ('whisk + lick', 'syllables'): RS.whisk_lick_only(S360),
        ('whisk + lick', 'raw, whisker ME + lick'): stack(raw, 'wl_sessz'),
        ('all', 'syllables'): S360,
        ('all', 'raw, speed per paw + whisk + lick'): stack(vel, 'spd_wl_sessz'),
    }
    BLOCKS = {k: RS.zs(v) for k, v in BLOCKS.items()}

    def hits(key):
        X = BLOCKS[key]
        return key, dict(mouse=loso_hits(X, y), mouse_lab=loso_hits(Z.lab_center(X, labs), y),
                         lab_lomo=lomo_hits(X, labs, mice))

    hits_file = HERE / 'ci_segmentation_vs_raw_hits.npz'
    if hits_file.exists():                       # the expensive part: reuse (delete to refit)
        H = np.load(hits_file, allow_pickle=True)['hits'].item()
        print(f'held-out predictions read from {hits_file.name}', flush=True)
    else:
        H = dict(Parallel(n_jobs=18)(delayed(hits)(k) for k in BLOCKS))
        np.savez(hits_file, hits=H, mice=mice, labs=labs)

    # ---- bootstrap over mice, pairs resampled together
    rng = np.random.default_rng(SEED)
    u = np.unique(mice)
    rows_of = {m: np.where(mice == m)[0] for m in u}
    draws = [np.concatenate([rows_of[m] for m in rng.choice(u, len(u), replace=True)])
             for _ in range(N_BOOT)]

    def metric(key, name, r):
        if name == 'lab_eta2':
            return Z.lab_eta2(BLOCKS[key][r], labs[r])
        return H[key][name][r].mean()

    def jackknife(f):
        """Leave-one-mouse-out jackknife: (estimate, lo, hi) of f(rows)."""
        full = f(np.arange(len(idx)))
        th = np.array([f(np.where(mice != m)[0]) for m in u])
        se = np.sqrt((len(u) - 1) / len(u) * ((th - th.mean()) ** 2).sum())
        return full, full - 1.96 * se, full + 1.96 * se

    out = []
    full = np.arange(len(idx))
    for ch, a, b in PAIRS:
        for name in ('mouse', 'mouse_lab', 'lab_lomo', 'lab_eta2'):
            if name == 'lab_eta2':
                fa = lambda r: Z.lab_eta2(BLOCKS[(ch, a)][r], labs[r])       # noqa: E731
                fb = lambda r: Z.lab_eta2(BLOCKS[(ch, b)][r], labs[r])       # noqa: E731
                res = [(a,) + jackknife(fa), (b,) + jackknife(fb),
                       ('difference (syllables - raw)',) + jackknife(lambda r: fa(r) - fb(r))]
                method = 'jackknife over mice'
            else:
                ba = np.array([H[(ch, a)][name][r].mean() for r in draws])
                bb = np.array([H[(ch, b)][name][r].mean() for r in draws])
                ea, eb = H[(ch, a)][name].mean(), H[(ch, b)][name].mean()
                res = [(rep, est) + tuple(np.percentile(bs, [2.5, 97.5]))
                       for rep, est, bs in [(a, ea, ba), (b, eb, bb),
                                            ('difference (syllables - raw)', ea - eb, ba - bb)]]
                method = 'bootstrap over mice'
            for rep, est, lo, hi in res:
                out.append(dict(channel=ch, metric=name, rep=rep, estimate=est, lo=lo, hi=hi,
                                ci=method, syllable_file=pathlib.Path(RS.SYLLABLE_FILE).name,
                                dims=BLOCKS[(ch, rep)].shape[1] if (ch, rep) in BLOCKS else np.nan))
        print(f'  {ch} done', flush=True)
    R = pd.DataFrame(out)
    R.to_csv(HERE / 'ci_segmentation_vs_raw.csv', index=False)
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
