"""
MOUSE AND LAB IDENTIFICATION FROM TRIAL MEASURES, ON THE SYLLABLES-vs-RAW PROTOCOL
==================================================================================
ci_segmentation_vs_raw.py scores syllables and the raw signal; this scores the session-level task measures
(accuracy, reaction time, elongation, biases, engagement...) the SAME way, so the three can sit in one panel
(cosyne_figure panel e):

  mouse      leave-one-session-out mouse ID, shrinkage LDA (zscore_cost: shrinkage 0.5, <= 3 sessions per mouse,
             3 repeats)
  lab_lomo   lab ID with every session of the test mouse held out
  CIs        bootstrap over mice, 2,000 resamples, percentile 95 %

Features: the task measures of non_emerging/nonemerging_session_metrics_<newest>.csv with <= 5 % missing sessions
(as fig_ld1 panel E), z-scored across sessions; sessions with all of them.
Output: ci_trial_measures.csv (rows as ci_segmentation_vs_raw.csv: channel='all', rep='trial measures').
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import re
import sys
import glob
import pathlib
import warnings
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'lda'))
sys.path.insert(0, str(PI / '4_mice'))
sys.path.insert(0, str(PI / 'learning_individuality'))   # session_filters, used by lab_labels
import zscore_cost as Z                                     # noqa: E402  the LDA protocol
from functions import lab_labels                            # noqa: E402

warnings.filterwarnings('ignore')
N_BOOT, SEED, MAX_MISSING = 2000, 0, 0.05
TASK = ['correct', 'log_reaction', 'log_elongation', 'bias', 'signed_choice_bias', 'choice_bias',
        'p_engaged', 'p_switch', 'p_eng_to_dis', 'p_dis_to_eng', 'trial_count']


def loso_hits(X, y, n_repeats=Z.N_REPEATS, seed=0):
    hits = np.zeros(len(X))
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        for i in range(len(X)):
            tr = np.setdiff1d(np.arange(len(X)), [i])
            ys = y[tr]
            idx = np.concatenate([rng.choice(np.where(ys == m)[0], min(Z.N_PER_MOUSE, (ys == m).sum()), replace=False)
                                  for m in np.unique(ys)])
            hits[i] += Z._fit(X[tr][idx], ys[idx]).predict(X[i:i + 1])[0] == y[i]
    return hits / n_repeats


def lomo_hits(X, labs, mice):
    y = pd.factorize(labs)[0]
    hits = np.zeros(len(X))
    for m in np.unique(mice):
        te = mice == m
        hits[te] = Z._fit(X[~te], y[~te]).predict(X[te]) == y[te]
    return hits


def main():
    f = sorted(glob.glob(str(PI / 'non_emerging' / 'nonemerging_session_metrics_*.csv')),
               key=lambda p: pd.to_datetime(re.search(r'(\d\d-\d\d-\d{4})', p).group(1), dayfirst=True))[-1]
    M = pd.read_csv(f).set_index('session')
    feats = [c for c in TASK if c in M and M[c].isna().mean() <= MAX_MISSING]
    d = M[feats + ['mouse_name']].dropna()
    X = ((d[feats] - d[feats].mean()) / d[feats].std()).to_numpy(float)
    mice = d['mouse_name'].to_numpy()
    labs = np.array(list(lab_labels(d.index, mouse_names=d['mouse_name'], verbose=False)))
    y = pd.factorize(mice)[0]
    print(f'{pathlib.Path(f).name}: {len(d)} sessions, {len(set(mice))} mice, {len(set(labs))} labs, {len(feats)} measures')
    H = dict(mouse=loso_hits(X, y), lab_lomo=lomo_hits(X, labs, mice))
    rng = np.random.default_rng(SEED)
    u = np.unique(mice)
    rows_of = {m: np.where(mice == m)[0] for m in u}
    draws = [np.concatenate([rows_of[m] for m in rng.choice(u, len(u), replace=True)]) for _ in range(N_BOOT)]
    out = []
    for name, h in H.items():
        bs = np.array([h[r].mean() for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(dict(channel='all', metric=name, rep='trial measures', estimate=h.mean(), lo=lo, hi=hi,
                        ci='bootstrap over mice', n_sessions=len(d), n_mice=len(u), n_labs=len(set(labs)),
                        measures=' '.join(feats), source=pathlib.Path(f).name))
        print(f'{name:9s} {h.mean():.3f} [{lo:.3f}, {hi:.3f}]')
    pd.DataFrame(out).to_csv(HERE / 'ci_trial_measures.csv', index=False)
    print('wrote ci_trial_measures.csv')


if __name__ == '__main__':
    main()
