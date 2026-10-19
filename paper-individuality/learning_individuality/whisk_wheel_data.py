"""
First-session whisking (HMM) and wheel movement on the SAME time-warped trial grid.
==================================================================================
Used by whisk_wheel_engagement.ipynb.

WHISKING comes from the session-1 (Early) or last-training (Late) syllable file, whose codes pack
    code = paw_wheel_state + 8 * (whisk + 2 * lick)
so the whisk HMM state (1 = whisking) is (code // 8) % 2. Each trial phase is already
warped to 10 bins (mode of the 0/1 state per bin).

WHEEL comes from the raw training design matrix (`avg_wheel_vel`, ~30 Hz). It is cut
into the same four trial phases with the same boundaries the syllable pipeline uses
(segmentation_functions.states_per_trial_phase), and each phase is warped to 10 bins
with the same np.array_split rule (mean of |velocity| per bin). So bin b of trial t means
the same stretch of time in both arrays, and they can be correlated bin by bin.

Two deliberate deviations from the pipeline, both small:
  * the pre-quiescence / quiescence boundary uses goCue_times, not goCueTrigger_times
    (the training trial tables do not carry the trigger times; they differ by ms);
  * segmentation_functions is not imported, because it imports jax, which crashes on
    the Mac. The few lines needed are re-implemented here.
"""
import os
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PI = os.path.dirname(HERE)
DATA = os.path.join(PI, 'data')

# legacy left-paw+wheel syllables (k = 8, 100 ms lick HMM), one session per mouse
SYL_FILES = {'Early': os.path.join(DATA, 'training', 'paw_wheel_session_1_8_k_10_bin_syllables_04-09-2026'),
             'Late': os.path.join(DATA, 'training', 'paw_wheel_last_training_8_k_10_bin_syllables_04-09-2026')}
DM_DIR = os.path.join(DATA, 'training', 'training_design_matrices')
TRIALS_DIR = os.path.join(DATA, 'training', 'training_data')

PHASES = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_PER_PHASE = 10
K_PAW = 8


def rescale(seq, target_length=N_PER_PHASE):
    """segmentation_functions.rescale_sequence with estimator='mean'."""
    seq = np.asarray(seq, float)
    n = len(seq)
    if n == 0:
        return np.full(target_length, np.nan)
    if n == target_length:
        return seq
    if target_length < n:
        return np.array([b.mean() for b in np.array_split(seq, target_length)])
    return seq[np.floor(np.linspace(0, n - 1, target_length)).astype(int)]


def load_whisk(timepoint='Early'):
    """One row per trial: mouse, session, trial_id, trial_type, whisk (n_trials x 40, NaN = no phase)."""
    s = pd.read_parquet(SYL_FILES[timepoint])
    s['session'] = s['sample'].str[:36]
    s['trial_id'] = s['sample'].str[37:].astype(float).astype(int)
    codes = np.vstack([np.asarray(x, float) for x in s['binned_sequence']])
    whisk = np.where(np.isnan(codes), np.nan, (np.nan_to_num(codes).astype(int) // K_PAW) % 2)
    s['w'] = list(whisk)
    trials = s[['mouse_name', 'session', 'trial_id', 'trial_type']].drop_duplicates(['session', 'trial_id'])
    trials = trials.sort_values(['mouse_name', 'trial_id']).reset_index(drop=True)
    W = np.full((len(trials), len(PHASES) * N_PER_PHASE), np.nan)
    pos = {(a, b): i for i, (a, b) in enumerate(zip(trials['session'], trials['trial_id']))}
    for (sess, tid, ph, w) in zip(s['session'], s['trial_id'], s['broader_label'], s['w']):
        k = PHASES.index(ph)
        W[pos[(sess, tid)], k * N_PER_PHASE:(k + 1) * N_PER_PHASE] = w
    return trials, W


def phase_bounds(tt):
    """(n_trials, 4, 2) start/end times of each phase, as states_per_trial_phase draws them."""
    go = tt['goCue_times'].to_numpy(float)
    q = tt['quiescence'].to_numpy(float)
    fb = tt['feedback_times'].to_numpy(float)
    i0 = tt['intervals_0'].to_numpy(float)
    i1 = tt['intervals_1'].to_numpy(float)
    iti_end = np.where(tt['feedbackType'].to_numpy(float) == -1, i1 - 1, i1)
    return np.stack([np.c_[i0, go - q], np.c_[go - q, go], np.c_[go, fb], np.c_[fb, iti_end]], axis=1)


def session_trials(mouse, session):
    f = os.path.join(TRIALS_DIR, f'training_data_trials_{mouse}')
    if not os.path.exists(f):
        return pd.DataFrame(columns=['session', 'feedbackType'])
    t = pd.read_parquet(f)
    return t[t['session'] == session].reset_index(drop=True)


def warp_wheel(mouse, session, trial_ids):
    """|wheel velocity| on the 40-bin grid for the given trial ids (row index in the trial table),
    plus the trial table rows so the caller can check the alignment."""
    out = np.full((len(trial_ids), len(PHASES) * N_PER_PHASE), np.nan)
    tt = session_trials(mouse, session)
    f = os.path.join(DM_DIR, f'design_matrix_{session}_{mouse}')
    if not os.path.exists(f) or not len(tt):  # no wheel or no trial table for this session -> all NaN
        return out, tt
    dm = pd.read_parquet(f, columns=['Bin', 'avg_wheel_vel'])
    dm = dm.dropna().sort_values('Bin')
    b = dm['Bin'].to_numpy(float)
    v = np.abs(dm['avg_wheel_vel'].to_numpy(float))
    bounds = phase_bounds(tt)
    for r, tid in enumerate(trial_ids):
        if tid >= len(tt):
            continue
        for k in range(len(PHASES)):
            lo, hi = bounds[tid, k]
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                continue
            i, j = np.searchsorted(b, lo, 'right'), np.searchsorted(b, hi, 'right')   # (lo, hi]
            if j > i:
                out[r, k * N_PER_PHASE:(k + 1) * N_PER_PHASE] = rescale(v[i:j])
    return out, tt


def build(timepoint='Early', verbose=True):
    """Trial table + aligned whisk / wheel matrices for every mouse at `timepoint` (Early / Late).
    `agree` is the fraction of trials whose correct/incorrect label in the syllable file
    matches feedbackType at that row of the trial table -- the alignment check."""
    trials, W = load_whisk(timepoint)
    V = np.full_like(W, np.nan)
    agree = {}
    for (mouse, session), idx in trials.groupby(['mouse_name', 'session']).groups.items():
        idx = np.asarray(idx)
        tids = trials.loc[idx, 'trial_id'].to_numpy()
        V[idx], tt = warp_wheel(mouse, session, tids)
        lab = trials.loc[idx, 'trial_type'].str.split().str[0].map({'correct': 1.0, 'incorrect': -1.0}).to_numpy()
        fb = tt['feedbackType'].reindex(tids).to_numpy(float) if len(tt) else np.full(len(tids), np.nan)
        ok = np.isfinite(lab)
        ok &= np.isfinite(fb)
        agree[mouse] = float(np.mean(lab[ok] == fb[ok])) if ok.any() else np.nan
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # trials with no whisk bins
        trials['whisk_frac'] = np.nanmean(W, axis=1)
        trials['wheel_speed'] = np.nanmean(V, axis=1)
    if verbose:
        a = pd.Series(agree)
        n_wheel = trials.groupby('session')['wheel_speed'].apply(lambda x: x.notna().any()).sum()
        print(f'{timepoint}: {len(trials)} trials, {trials.session.nunique()} sessions '
              f'({n_wheel} with wheel); label agreement min {a.min():.3f}, median {a.median():.3f}')
    return trials, W, V, pd.Series(agree, name='label_agreement')
