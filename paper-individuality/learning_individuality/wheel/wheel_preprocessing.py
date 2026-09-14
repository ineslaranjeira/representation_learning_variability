"""
WHEEL VELOCITY PREPROCESSING: design matrix -> time-warped vector per session
=============================================================================
One place for the design-matrix -> epoch-labelled -> time-warped chain, so the
notebooks in this folder can import it instead of copying cells around.

The chain is the same one wheel_velocity_timepoints.ipynb already implements; this
module changes three things and nothing else:

  1  BOTH DATASETS. Training sessions and proficient sessions live in different
     folders with different columns (training design matrices carry only
     `Bin` and `avg_wheel_vel`; proficient ones also carry lick, whisker and paw).
     Only `avg_wheel_vel` is read, so the two are handled by the same code.

  2  SIGN IS A PARAMETER. The notebook hardcodes np.abs(), i.e. speed. A mouse with a
     consistent turning bias is exactly the kind of individual trait worth looking
     for, and abs() destroys it -- so `signed=True` keeps direction.

  3  NOTHING IS AVERAGED ACROSS SESSIONS. The unit of output is the SESSION. Whether
     several sessions of a mouse get pooled, and which ones count as "early", is a
     downstream decision -- making it here would bake a timepoint definition into
     the cache and force a rebuild every time it changed.

Output geometry matches the syllable files on purpose: 4 epochs x TARGET_LENGTH bins
= 40 numbers per trial, so a wheel vector drops straight into the analysis machinery
written for syllables.
"""
import os
import pathlib
import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# PATHS -- first candidate that exists wins, so the same file runs on the mac
# (data on Google Drive) and on the linux box (data local).
# --------------------------------------------------------------------------
# The repo's own data directory, derived from THIS file's location, so the same
# candidate works on the mac and on the linux box without either path being
# hardcoded. It is tried FIRST: a local copy is ~500x faster to read than the Google
# Drive mount (0.01 s/file vs 5.2 s/file), which is the difference between a 2-minute
# build and a 40-minute one.
_REPO_DATA = pathlib.Path(__file__).resolve().parents[2] / 'data'

TRAINING_CANDIDATES = [
    str(_REPO_DATA / 'training' / 'training_design_matrices') + '/',
    '/home/ines/repositories/representation_learning_variability/paper-individuality/'
    'data/training/training_design_matrices/',
    '/Users/ineslaranjeira/Google Drive/O meu disco/CCU/PhD Project/paper-individuality/'
    'data/training/training_design_matrices/',
]
PROFICIENT_CANDIDATES = [
    str(_REPO_DATA / 'newly_generated' / 'segmentation' / 'design_matrices') + '/',
    '/home/ines/repositories/representation_learning_variability/paper-individuality/'
    'data/newly_generated/segmentation/design_matrices/',
    '/Users/ineslaranjeira/Google Drive/O meu disco/CCU/PhD Project/paper-individuality/'
    'data/newly_generated/segmentation/design_matrices/',
]
# per-mouse trials tables carrying `session` (eid) and `session_date`: the training
# history, used to order a mouse's sessions and so to define "early" downstream.
# The " (1)" spelling is a Google Drive duplicate-name artefact; a local copy should
# be renamed to plain `training_data`.
LEARNING_TABLE_CANDIDATES = [
    str(_REPO_DATA / 'training' / 'training_data') + '/',
    '/home/ines/repositories/representation_learning_variability/paper-individuality/'
    'data/training/training_data/',
    '/Users/ineslaranjeira/Google Drive/O meu disco/CCU/PhD Project/paper-individuality/'
    'data/training/training_data (1)/',
]
SEGMENTATION_CANDIDATES = [
    str(pathlib.Path(__file__).resolve().parents[2] / 'segmentation' / '1_camera_setup') + '/',
    '/home/ines/repositories/representation_learning_variability/paper-individuality/'
    'segmentation/1_camera_setup/',
    os.path.expanduser('~/Documents/Repositories/representation_learning_variability/'
                       'paper-individuality/segmentation/1_camera_setup/'),
]


def _pick(cands, what):
    hit = next((c for c in cands if os.path.isdir(c)), None)
    assert hit is not None, f'no {what} directory found; tried:\n  ' + '\n  '.join(cands)
    return hit


TRAINING_DIR = _pick(TRAINING_CANDIDATES, 'training design-matrix')
PROFICIENT_DIR = _pick(PROFICIENT_CANDIDATES, 'proficient design-matrix')
LEARNING_TABLE_DIR = _pick(LEARNING_TABLE_CANDIDATES, 'training-history table')
SOURCE_DIR = {'training': TRAINING_DIR, 'proficient': PROFICIENT_DIR}

_seg = _pick(SEGMENTATION_CANDIDATES, 'segmentation_functions')
if _seg not in sys.path:
    sys.path.insert(0, _seg)
from segmentation_functions import (          # noqa: E402
    align_bin_design_matrix, states_per_trial_phase, broader_label,
    define_trial_types, rescale_sequence,
)

# --------------------------------------------------------------------------
# PARAMETERS -- the alignment window and warping are the notebook's, unchanged,
# so wheel vectors stay comparable with the syllable build.
# --------------------------------------------------------------------------
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
TARGET_LENGTH = 10
ESTIMATOR = 'mean'                # continuous velocity -> mean within each warped bin
EVENT_TYPE_LIST = ['goCueTrigger_times']
MULTIPLIER = 1
INIT, END = -1.0 * MULTIPLIER, 1.5 * MULTIPLIER
TRIAL_TYPE_AGG = ['correct_str', 'contrast_str', 'block_str', 'choice']

KEEP_COLS = ['Bin', 'avg_wheel_vel', 'correct', 'choice', 'contrast', 'block',
             'reaction', 'response', 'elongation', 'wsls', 'trial_id',
             'goCueTrigger_times', 'label', 'broader_label', 'mouse_name',
             'session', 'source']

import re as _re
_FNAME = _re.compile(r'^design_matrix_([0-9a-f-]{36})_(.+)$')


def list_sessions(source=None):
    """Every session with BOTH a design matrix and a trials file, per source."""
    frames = []
    for src in ([source] if source else list(SOURCE_DIR)):
        d = SOURCE_DIR[src]
        files = os.listdir(d)
        dm = {m.group(1): m.group(2) for m in map(_FNAME.match, files) if m}
        ok = {eid for eid, mouse in dm.items()
              if f'session_trials_{eid}_{mouse}' in set(files)}
        frames.append(pd.DataFrame([{'session': e, 'mouse_name': dm[e], 'source': src}
                                    for e in sorted(ok)]))
    return pd.concat(frames, ignore_index=True)


def training_history(cache='training_history.parquet', refresh=False):
    """(mouse, session, session_date, ordinal) from the per-mouse trials tables.

    `ordinal` is 1 for a mouse's first training session. It is what lets a caller say
    "the first session ever" without this module deciding what early means.

    Two reasons this is not a naive read-everything loop: the tables live on Google
    Drive (every byte is streamed, not read from a local disk), and each one is a
    full trials table of which exactly TWO columns matter. Parquet is columnar, so
    `columns=` skips the rest on disk rather than after loading. The result is then
    cached locally, because even the cheap version is not worth repeating.
    """
    if cache and os.path.exists(cache) and not refresh:
        return pd.read_parquet(cache)
    rows = []
    for f in sorted(os.listdir(LEARNING_TABLE_DIR)):
        if f.startswith('.'):
            continue
        mouse = f.replace('training_data_trials_', '')
        try:
            t = pd.read_parquet(os.path.join(LEARNING_TABLE_DIR, f),
                                columns=['session', 'session_date'])
        except Exception:
            continue                      # not a trials table, or lacks the columns
        u = (t.drop_duplicates().sort_values('session_date').reset_index(drop=True))
        u['mouse_name'] = mouse
        u['ordinal'] = np.arange(1, len(u) + 1)
        rows.append(u)
    out = pd.concat(rows, ignore_index=True)
    if cache:
        out.to_parquet(cache)
    return out


def first_session_table(min_proficient=1):
    """The cohort for the first-session-ever analysis: each mouse's ordinal-1
    training session, plus every proficient session it has.

    A mouse is kept only if its first session actually has a design matrix on disk
    AND it has at least `min_proficient` proficient sessions -- otherwise it cannot
    contribute to a comparison and would only inflate the cohort count.
    """
    S = list_sessions()
    hist = training_history()
    first = hist.loc[hist['ordinal'] == 1, ['mouse_name', 'session', 'session_date']]
    tr = S.loc[S['source'] == 'training', ['mouse_name', 'session']]
    first = first.merge(tr, on=['mouse_name', 'session'], how='inner')
    first['source'] = 'training'

    prof = S.loc[S['source'] == 'proficient']
    n_prof = prof.groupby('mouse_name').size()
    keep = sorted(set(first['mouse_name']) & set(n_prof[n_prof >= min_proficient].index))

    first = first.loc[first['mouse_name'].isin(keep)]
    prof = prof.loc[prof['mouse_name'].isin(keep)].copy()
    prof['session_date'] = pd.NaT
    cols = ['mouse_name', 'session', 'source', 'session_date']
    return pd.concat([first[cols], prof[cols]], ignore_index=True)


def label_session(mouse_name, session, source, fast=True):
    """Design matrix -> one row per (bin, trial) with an epoch label.

    align_bin_design_matrix wants a `most_likely_states` argument this analysis has
    no use for; a zero placeholder goes in and is dropped immediately, exactly as in
    wheel_velocity_timepoints.
    """
    d = SOURCE_DIR[source]
    dm_file = os.path.join(d, f'design_matrix_{session}_{mouse_name}')
    tr_file = os.path.join(d, f'session_trials_{session}_{mouse_name}')
    for f in (dm_file, tr_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    design_matrix = pd.read_parquet(dm_file)[['Bin', 'avg_wheel_vel']].copy()
    session_trials = pd.read_parquet(tr_file, engine='pyarrow').reset_index()

    aligned = align_bin_design_matrix(INIT, END, EVENT_TYPE_LIST, session_trials,
                                      design_matrix, np.zeros(len(design_matrix)),
                                      MULTIPLIER)
    aligned = aligned.drop(columns=['new_bin', 'most_likely_states'])

    labeller = states_per_trial_phase_fast if fast else states_per_trial_phase
    per_trial = labeller(aligned, session_trials, MULTIPLIER)
    per_trial = broader_label(per_trial)
    per_trial['mouse_name'] = mouse_name
    per_trial['session'] = session
    per_trial['source'] = source
    missing = [c for c in KEEP_COLS if c not in per_trial.columns]
    assert not missing, f'{mouse_name} {session}: missing columns {missing}'
    return per_trial[KEEP_COLS]


def warp_trials(labelled, signed=False, target_length=TARGET_LENGTH,
                estimator=ESTIMATOR, trial_type_agg=TRIAL_TYPE_AGG):
    """One row per (trial, epoch) carrying a fixed-length warped velocity sequence.

    signed=False -> |velocity| (speed, the notebook's behaviour)
    signed=True  -> velocity keeps its sign, so a consistent turning direction
                    survives into the vector instead of being folded away.

    trial_type is computed either way; it is metadata here, and whether anything is
    split by it is a downstream choice.
    """
    df = define_trial_types(labelled, trial_type_agg)
    df = df.dropna(subset=['avg_wheel_vel']).copy()
    if not signed:
        df['avg_wheel_vel'] = np.abs(df['avg_wheel_vel'])
    g = (df.groupby(['sample', 'trial_type', 'broader_label', 'mouse_name',
                     'session', 'source'])['avg_wheel_vel']
         .apply(list).reset_index().rename(columns={'avg_wheel_vel': 'seq'}))
    g['binned'] = g['seq'].apply(lambda s: rescale_sequence(s, target_length, estimator))
    return g.drop(columns=['seq'])


def session_vector(warped, by_trial_type=False, epochs=EPOCHS,
                   target_length=TARGET_LENGTH):
    """(n_units, 4 * target_length) -- the wheel vector, averaged over TRIALS ONLY.

    by_trial_type=False -> one vector per session   (index: mouse, session, source)
    by_trial_type=True  -> one per (session, trial_type)

    Sessions are never pooled here. A trial is kept only if all four epochs are
    present, matching how the syllable design matrix is built (pivot + dropna).
    """
    idx = ['mouse_name', 'session', 'source'] + (['trial_type'] if by_trial_type else [])
    wide = (warped.pivot_table(index=['sample'] + idx, columns='broader_label',
                               values='binned', aggfunc='first')
            .reset_index())
    have = [e for e in epochs if e in wide.columns]
    if len(have) < len(epochs):
        return pd.DataFrame(columns=idx + list(range(len(epochs) * target_length)))
    wide = wide.dropna(subset=epochs)
    if not len(wide):
        return pd.DataFrame(columns=idx + list(range(len(epochs) * target_length)))
    M = np.vstack(wide[epochs].apply(lambda r: np.hstack(r.values), axis=1))
    out = pd.DataFrame(M, columns=range(M.shape[1]))
    for c in idx:
        out[c] = wide[c].values
    out['n_trials'] = 1
    agg = {**{c: 'mean' for c in range(M.shape[1])}, 'n_trials': 'sum'}
    return out.groupby(idx, as_index=False).agg(agg)


def build_session(mouse_name, session, source, signed=False, fast=True, **kw):
    """label -> warp -> vector, for one session. Returns the session-level frame."""
    return session_vector(
        warp_trials(label_session(mouse_name, session, source, fast=fast),
                    signed=signed), **kw)


# --------------------------------------------------------------------------
# FAST EPOCH LABELLING
# --------------------------------------------------------------------------
def states_per_trial_phase_fast(reduced_design_matrix, session_trials, multiplier):
    """Drop-in replacement for segmentation_functions.states_per_trial_phase.

    Identical output, ~100x faster. Two things make the original slow, and neither
    is intrinsic to what it computes:

      * `prepro(session_trials)` is called THREE times inside the per-trial loop, so
        a 1000-trial session re-preprocesses the whole trials table ~3000 times. It
        is loop-invariant and is hoisted out here.
      * every phase assignment is a boolean mask over the ENTIRE design matrix
        (~150k rows x 5 phases x n_trials full scans). `Bin` is sorted, so the same
        rows can be found with two binary searches instead.

    Semantics are preserved exactly, including the bit that matters: the original
    loops TRIAL-major and lets later writes overwrite earlier ones, so the phase
    order within a trial, and the trial order itself, both change the answer when
    intervals touch. This keeps that order rather than vectorising across trials.
    """
    from segmentation_functions import prepro

    use_data = reduced_design_matrix.copy()
    bins = use_data['Bin'].to_numpy()
    assert np.all(np.diff(bins) >= 0), 'Bin must be sorted for the fast labeller'
    labels = np.full(len(use_data), None, dtype=object)

    st = session_trials
    sc = prepro(st)['signed_contrast'].to_numpy()          # hoisted out of the loop
    fb = st['feedbackType'].to_numpy()
    ch = st['choice'].to_numpy()

    def span(a, b):
        """rows with a*multiplier < Bin <= b*multiplier"""
        if not (np.isfinite(a) and np.isfinite(b)):
            return 0, 0
        return (np.searchsorted(bins, a * multiplier, 'right'),
                np.searchsorted(bins, b * multiplier, 'right'))

    pre_i = st['intervals_0'].to_numpy()
    pre_e = (st['goCueTrigger_times'] - st['quiescencePeriod']).to_numpy()
    qui_e = st['goCueTrigger_times'].to_numpy()
    iti_i = st['feedback_times'].to_numpy()
    iti_c = st['intervals_1'].to_numpy()
    rt_i = st['goCueTrigger_times'].to_numpy()
    rt_e = st['firstMovement_times'].to_numpy()
    mv_e = st['feedback_times'].to_numpy()

    for t in range(len(st)):
        lo, hi = span(pre_i[t], pre_e[t]);  labels[lo:hi] = 'Pre-quiescence'
        lo, hi = span(pre_e[t], qui_e[t]);  labels[lo:hi] = 'Quiescence'
        if fb[t] == -1.:
            lo, hi = span(iti_i[t], iti_c[t] - 1); labels[lo:hi] = 'ITI'
        elif fb[t] == 1.:
            lo, hi = span(iti_i[t], iti_c[t]);     labels[lo:hi] = 'ITI'
        if ch[t] == -1:
            lo, hi = span(rt_e[t], mv_e[t]); labels[lo:hi] = 'Left choice'
        elif ch[t] == 1.:
            lo, hi = span(rt_e[t], mv_e[t]); labels[lo:hi] = 'Right choice'
        elif ch[t] == 0:
            lo, hi = span(rt_e[t], mv_e[t]); labels[lo:hi] = 'No go'
        if sc[t] < 0:
            lo, hi = span(rt_i[t], rt_e[t]); labels[lo:hi] = 'Stimulus left'
        elif sc[t] > 0:
            lo, hi = span(rt_i[t], rt_e[t]); labels[lo:hi] = 'Stimulus right'
        elif sc[t] == 0:
            lo, hi = span(rt_i[t], rt_e[t]); labels[lo:hi] = 'Stimulus zero'

    use_data['label'] = pd.Series(labels, index=use_data.index).astype(object)
    use_data.loc[use_data['label'].isna(), 'label'] = np.nan
    return use_data
