"""
REGENERATE THE SHRINKAGE LDA EMBEDDING LOCALLY
===============================================
`4_mice/lda_shrinkage.ipynb` wrote `mouse_LDA_5_bins_raw_shrink0.5_360_27-09-2026`
on the mac; that file is not on this machine. This reproduces it from the same
inputs with the same knobs, so the LD1 used here is the LD1 that notebook's
premise-check cell used.

The nested CV that PICKS the shrinkage is skipped -- the notebook records its result
(modal 0.5 over 5 repeats x 260 sessions) and re-running it would take ~18 min to
land on the same value. Everything downstream of that choice is reproduced exactly,
including the by-hand grand-mean centring that solver='eigen' needs.
"""
import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

_root = pathlib.Path(__file__).resolve().parents[2]          # paper-individuality/
for p in (str(_root), str(_root / 'learning_individuality'), str(_root / '4_mice')):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings('ignore', message='Only one sample available', category=UserWarning)

from session_filters import exclusions_by_timepoint, find_csv          # noqa: E402
from functions import lab_labels, lab_variance_explained               # noqa: E402

QC_STRICTNESS = 'filtered_out'
MAX_SESSION_MISSING = 0.1
MIN_SESSIONS_PER_MOUSE = 3
N_PAW_STATES = 8
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
BEST_SHRINKAGE = 0.5
SYLLABLE_FILE = str(_root / 'data' / '8_k_10_bin_syllables_19-08-2026')


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
    print(f'{seq.mouse_name.nunique()} mice, {seq.session.nunique()} sessions in total')
    seq = seq.loc[~seq['session'].isin(prob_sessions)].reset_index(drop=True)
    print(f'{seq["session"].nunique()} sessions after the QC sheet ({QC_STRICTNESS})')
    nan_frac = np.isnan(np.stack(seq['binned_sequence'].to_numpy())).mean(axis=1)
    by_session = pd.Series(nan_frac, index=seq['session'].to_numpy()).groupby(level=0).mean()
    drop = by_session[by_session > MAX_SESSION_MISSING]
    seq = seq.loc[~seq['session'].isin(drop.index)].reset_index(drop=True)
    print(f'{seq["session"].nunique()} sessions after dropping {len(drop)} with > '
          f'{MAX_SESSION_MISSING:.1%} missing bins')
    counts = (seq[['mouse_name', 'session']].drop_duplicates()
              .groupby('mouse_name')['session'].count())
    keep = counts[counts >= MIN_SESSIONS_PER_MOUSE].index
    seq = seq.loc[seq['mouse_name'].isin(keep)].reset_index(drop=True)
    print(f'{len(keep)} mice with at least {MIN_SESSIONS_PER_MOUSE} sessions, '
          f'{seq["session"].nunique()} remaining sessions')
    return seq


def build_design_matrix(filename):
    seq = pd.read_parquet(filename)
    seq['session'] = seq['sample'].str[:36]
    prob = sorted(exclusions_by_timepoint(QC_STRICTNESS)['Proficient'])
    print(f'QC sheet: {find_csv()}  ->  {len(prob)} proficient sessions dropped')
    seq = filter_sequences(seq, prob)
    trials = (seq.pivot(index=['mouse_name', 'session', 'sample', 'trial_type'],
                        columns=['broader_label'], values='binned_sequence')
              .reset_index().dropna().sort_values(by='session'))
    print(f'{len(trials)} trials, {trials["mouse_name"].nunique()} mice, '
          f'{trials["session"].nunique()} sessions')
    use_sequences = np.vstack(trials[EPOCHS].apply(lambda r: np.hstack(r), axis=1))
    feats = binarize(use_sequences)
    frame = pd.DataFrame(feats)
    frame['session'] = trials['session'].values
    session_feats = frame.groupby('session', sort=False)[np.arange(feats.shape[1])].mean()
    mapping = trials[['session', 'mouse_name']].drop_duplicates().set_index('session')['mouse_name']
    print(f'{len(session_feats)} sessions x {session_feats.shape[1]} features, '
          f'{mapping.nunique()} mice')
    return session_feats, mapping.reindex(session_feats.index)


def main():
    session_feats, mouse_of_session = build_design_matrix(SYLLABLE_FILE)
    lab_of_session = lab_labels(session_feats.index, mouse_names=mouse_of_session)
    print('lab share of feature variance: '
          f'{lab_variance_explained(session_feats, lab_of_session):.3f}')

    X = np.asarray(session_feats, dtype=float)
    mouse_names = pd.Series(mouse_of_session.to_numpy(), name='mouse_name')
    y = pd.factorize(mouse_names)[0]
    n_mice = len(np.unique(y))

    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=BEST_SHRINKAGE,
                                     priors=np.ones(n_mice) / n_mice,
                                     n_components=n_mice - 1).fit(X, y)
    grand = lda.means_.mean(axis=0)
    embedding = ((X - grand) @ lda.scalings_)[:, :n_mice - 1]

    mm = pd.DataFrame(embedding).groupby(y).mean()
    bv = mm.var(axis=0, ddof=0).to_numpy()
    print('between-mouse variance share (top 6):', np.round(bv / bv.sum(), 3))

    clustered = pd.DataFrame(embedding)
    clustered['mouse_name'] = mouse_names.to_numpy()
    clustered['mouse_number'] = pd.factorize(clustered['mouse_name'])[0]
    clustered['lab'] = list(lab_of_session)
    clustered['lab_number'] = pd.factorize(clustered['lab'])[0]
    clustered['session'] = session_feats.index.to_numpy()
    for b in range(5):
        clustered[f'binned{b + 1}'] = pd.cut(clustered[b], 5)

    out = (_root / 'clustering' / 'data_files' /
           f'mouse_LDA_5_bins_raw_shrink{BEST_SHRINKAGE}_{X.shape[1]}'
           f'_{datetime.now().strftime("%d-%m-%Y")}')
    clustered.to_pickle(out)
    print(f'saved {out}\n  {clustered.shape[0]} rows x {clustered.shape[1]} cols')


if __name__ == '__main__':
    main()
