"""
WHEEL MOVEMENT STATES IN THE LDA -- does adding the wheel (3.4's movement clusters) change mouse identification?
================================================================================================================
Exactly syllable_sweeps' protocol (same session screens, nested shrinkage, per-mouse training cap,
leave-one-session-out, LDA + syllable-space scores); only the feature matrix changes. The wheel comes
from syllable files made with WHEEL_SEGMENTATION = 'movement', WHEEL_BINNING = 'separate': their
`binned_sequence` is the production paw / whisker / lick code (byte-identical) and `wheel_binned_sequence`
is the wheel state per bin (0 = still, 1.. = movement state). The wheel is one-hot coded per bin with
state 0 (still) dropped as the reference level, like paw state 1 in binarize.

  variant          features per bin                           total (4 epochs x 10 bins)
  base             7 paw one-hots, whisk, lick                     360   (= the production features)
  wheel<K>         base + (K - 1) wheel one-hots                   360 + 40 (K - 1)
  wheel<K>_only    (K - 1) wheel one-hots                          40 (K - 1)

Cached like the sweeps, in data/syllable_sweeps/wheel/ (features_<variant>.pqt, loso_<variant>.pkl).

    python wheel_lda_test.py base wheel10 wheel10_only        # any variants, skipping what is cached
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import re
import sys
import glob
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import syllable_sweeps as ss
from syllable_sweeps import sp, exclusions_by_timepoint

OUT = ss.OUT / 'wheel'


def wheel_syllables(k_wheel):
    files = sorted(glob.glob(sp.syllables_glob('proficient', 'both_paws', 'own', 'standard', k=8, wheel=True,
                                               k_wheel=k_wheel, target_length=10, wheel_segmentation='movement',
                                               wheel_binning='separate')))
    assert files, f'no movement-wheel syllables with K_WHEEL = {k_wheel}: run 5_syllable_generation_fast first'
    return pd.read_parquet(files[-1])


def binarize_wheel(seq, k_wheel):
    """(trials, timesteps) wheel states -> one-hot, state 0 (still) dropped; NaN bins stay NaN."""
    n, T = seq.shape
    per = k_wheel - 1
    out = np.zeros((n, T * per))
    for t in range(T):
        v = seq[:, t]
        ok = ~np.isnan(v)
        s = v[ok].astype(int)
        rows = np.arange(n)[ok][s > 0]
        out[rows, t * per + s[s > 0] - 1] = 1
        out[~ok, t * per:(t + 1) * per] = np.nan
    return out


def features(variant):
    path = OUT / f'features_{variant}.pqt'
    if path.exists():
        f = pd.read_parquet(path)
        return f.drop(columns='mouse_name'), f['mouse_name']
    if variant == 'base':
        sf, mouse = ss.features('bins', 10)          # the production features, already cached by the sweep
    else:
        k_wheel = int(re.match(r'wheel(\d+)', variant).group(1))
        seq = wheel_syllables(k_wheel)
        seq['session'] = seq['sample'].str[:36]
        seq = ss.filter_sequences(seq, sorted(exclusions_by_timepoint(ss.QC_STRICTNESS)['Proficient']))
        trials = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'], columns=['broader_label'],
                            values=['binned_sequence', 'wheel_binned_sequence'])
                  .dropna().reset_index().sort_values(by='session'))
        stack = lambda col: np.vstack(trials[[(col, e) for e in ss.EPOCHS]].apply(lambda r: np.hstack(r), axis=1))
        blocks = [binarize_wheel(stack('wheel_binned_sequence'), k_wheel)]
        if not variant.endswith('_only'):
            blocks.insert(0, ss.binarize(stack('binned_sequence'), ss.N_PAW_STATES))
        feats = np.hstack(blocks)
        frame = pd.DataFrame(feats)
        frame['session'] = trials['session'].values
        sf = frame.groupby('session', sort=False)[np.arange(feats.shape[1])].mean()
        sf.columns = sf.columns.astype(str)
        mouse = (trials[['session', 'mouse_name']].droplevel(1, axis=1).drop_duplicates()
                 .set_index('session')['mouse_name'].reindex(sf.index))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.assign(mouse_name=mouse).to_parquet(path)
    return sf, mouse


def loso(variant):
    if variant == 'base':                            # the same features and seeds as the sweep's bins = 10 run
        return dict(ss.loso('bins', 10), variant='base')
    path = OUT / f'loso_{variant}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    sf, mouse = features(variant)
    X = sf.to_numpy(float)
    y, mice = pd.factorize(mouse.to_numpy())
    t0 = datetime.now()
    rows = Parallel(n_jobs=ss.N_JOBS)(delayed(ss.loso_fold)(X, y, r, t)
                                      for r in range(ss.N_REPEATS) for t in range(len(X)))
    res = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    res['correct'] = res['pred'] == y[res['test_idx']]
    res['correct_syl'] = res['pred_syl'] == y[res['test_idx']]
    res.update(variant=variant, sessions=sf.index.to_numpy(), mouse_of_session=mouse.to_numpy(), mice=np.asarray(mice),
               n_features=X.shape[1], n_repeats=ss.N_REPEATS, seconds=(datetime.now() - t0).seconds)
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    return res


if __name__ == '__main__':
    for v in sys.argv[1:]:
        r = loso(v)
        print(f"{v:14s}: {r['n_features']:4d} features, {len(r['sessions'])} sessions, {len(r['mice'])} mice | "
              f"LDA {r['correct'].mean():.3f}  syllable space {r['correct_syl'].mean():.3f} | "
              f"shrinkage {dict(pd.Series(r['shrink']).value_counts().sort_index())} | {r['seconds']} s", flush=True)
