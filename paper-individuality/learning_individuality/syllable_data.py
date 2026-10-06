"""
Which syllable files the learning analyses read, and how a syllable code becomes features.
=========================================================================================
One place for both, so lda_pred, lda_sweep_timepoints and lda_trajectories always agree on
(a) which files a timepoint means and (b) how a code is decoded.

FILES. 5_syllable_generation_fast names its outputs after every choice that made them
(movement, k, wheel, state space, lick HMM, dataset). `syllable_file` rebuilds that name with
segmentation/syllable_pipeline.py and returns the newest date on disk, so no notebook hardcodes
a date. version='legacy' returns the older paw_wheel_* files (04-09 / 28-08) instead.

CODES. A syllable code counts through its digits with the movement fastest:
    code = movement + k * (whisk + 2 * (lick + 2 * wheel))
(no wheel term unless the syllables were made with WHEEL = True). `binarize` turns codes into
per-bin indicators: the movement one-hot WITHOUT state `ref_state` (the redundant reference),
whisk, lick, and the wheel one-hot without its `ref_state`.
"""
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'segmentation'))
import syllable_pipeline as sp  # noqa: E402

DATA = sp.DATA

# analysis timepoint -> syllable_pipeline dataset
TIMEPOINT_DATASET = {'Early': 'session_1', 'Late': 'last_training', 'Pre-rec': 'biased',
                     'Proficient': 'proficient', 'Neuromodulators': 'neuromodulators'}

# THE DEFAULT: one left paw + wheel state space fitted on all four stages (3.3 FIT_NAME='stages'),
# vigor-numbered (0 = stillest), 10 bins per trial phase. Copy and edit in a notebook.
SYLLABLES = dict(version='new', movement='left_paw_wheel', state_space='stages', k=8,
                 wheel=False, k_wheel=8, lick_hmm='auto', target_length=10)

# the older files: proficient-fitted left paw + wheel states, 100 ms lick HMM (training),
# states relabelled with PAW_WHEEL_RELABEL_SESSIONZ -- so their paw numbers are NOT vigor order
_LEGACY = {'Early': 'training/paw_wheel_session_1_8_k_{L}_bin_syllables_04-09-2026',
           'Late': 'training/paw_wheel_last_training_8_k_{L}_bin_syllables_04-09-2026',
           'Pre-rec': 'training/paw_wheel_biased_8_k_{L}_bin_syllables_04-09-2026',
           'Proficient': 'paw_wheel_8_k_{L}_bin_syllables_28-08-2026'}
_ARCHIVES = ['outdated?/', 'maybe old?/']    # subfolders (of data/ or data/training/) old files were moved to


def _date(path):
    m = re.search(r'(\d{2}-\d{2}-\d{4})$', path)
    return datetime.strptime(m.group(1), '%d-%m-%Y') if m else datetime.min


def syllable_file(timepoint, version='new', movement='left_paw_wheel', state_space='stages', k=8,
                  wheel=False, k_wheel=8, lick_hmm='auto', target_length=10, date=None, missing_ok=False):
    """Path of one timepoint's syllable file. `date` ('dd-mm-yyyy') pins a version; by default
    the newest on disk. missing_ok: return the (non-existent) search pattern instead of raising."""
    if timepoint not in TIMEPOINT_DATASET:
        raise ValueError(f'timepoint must be one of {list(TIMEPOINT_DATASET)}, got {timepoint!r}')
    if version == 'legacy':
        if timepoint not in _LEGACY or k != 8 or wheel:
            raise ValueError('the legacy files are k = 8, no wheel, for Early / Late / Pre-rec / Proficient')
        rel = _LEGACY[timepoint].format(L=target_length)
        # also where old files get archived
        cands = [DATA + rel] + [DATA + os.path.dirname(rel) + ('/' if os.path.dirname(rel) else '') + d
                                + os.path.basename(rel) for d in _ARCHIVES]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None and not missing_ok:
            raise FileNotFoundError('legacy file not found in any of:\n  ' + '\n  '.join(cands))
        return path or cands[0]
    if version != 'new':
        raise ValueError(f"version must be 'new' or 'legacy', got {version!r}")
    pattern = sp.syllables_glob(TIMEPOINT_DATASET[timepoint], movement, state_space, lick_hmm, k, wheel,
                                k_wheel, target_length)
    found = sorted(glob.glob(pattern), key=_date)
    if date is not None:
        found = [f for f in found if f.endswith(date)]
    if not found and missing_ok:
        return pattern + (date or '')
    if not found:
        raise FileNotFoundError(
            f'no syllables for {timepoint} matching\n  {pattern}' + (f' dated {date}' if date else '') +
            f'\nMake them with 5_syllable_generation_fast: DATASET={TIMEPOINT_DATASET[timepoint]!r}, '
            f'MOVEMENT={movement!r}, K={k}, WHEEL={wheel}, K_WHEEL={k_wheel}, STATE_SPACE={state_space!r}, '
            f'LICK_HMM={lick_hmm!r}, TARGET_LENGTH={target_length}.')
    return found[-1]


def syllable_files(timepoints, **syllables):
    return {t: syllable_file(t, **syllables) for t in timepoints}


def n_states(k=8, wheel=False, k_wheel=8, **_):
    """(movement states, wheel states or 0) -- what binarize needs."""
    return int(k), (int(k_wheel) if wheel else 0)


def describe(**syllables):
    s = {**SYLLABLES, **syllables}
    if s['version'] == 'legacy':
        return 'legacy paw_wheel files (proficient-fitted states, relabelled; k = 8)'
    return (f"{s['movement']} k={s['k']}" + (f" + wheel k={s['k_wheel']}" if s['wheel'] else '') +
            f", state space {s['state_space']!r}, lick HMM {s['lick_hmm']!r}, {s['target_length']} bins/phase")


# ------------------------------------------------------------------------------------------------
# Codes -> features
# ------------------------------------------------------------------------------------------------
def _layout(n_paw_states, n_wheel_states, keep_paw, keep_whisk, keep_lick, keep_wheel, ref_state):
    """Raw columns per bin (paw one-hot, whisk, lick, wheel one-hot) and which of them are kept."""
    n_raw = n_paw_states + 2 + n_wheel_states
    cols = []
    if keep_paw:
        cols += [i for i in range(n_paw_states) if i != ref_state]
    if keep_whisk:
        cols += [n_paw_states]
    if keep_lick:
        cols += [n_paw_states + 1]
    if keep_wheel and n_wheel_states:
        cols += [n_paw_states + 2 + i for i in range(n_wheel_states) if i != ref_state]
    return n_raw, cols


def binarize(use_sequences, n_paw_states=8, n_wheel_states=0, keep_paw=True, keep_whisk=True,
             keep_lick=True, keep_wheel=True, ref_state=1):
    """
    (n_trials, timesteps) integer codes (NaN = no state) -> (n_trials, timesteps * features).
    With n_wheel_states = 0 this is exactly the binarize the notebooks used before.
    """
    if not (keep_paw or keep_whisk or keep_lick or (keep_wheel and n_wheel_states)):
        raise ValueError('At least one feature type must be kept.')
    use_sequences = np.asarray(use_sequences, dtype=float)
    k = n_paw_states
    n_raw, cols = _layout(k, n_wheel_states, keep_paw, keep_whisk, keep_lick, keep_wheel, ref_state)
    n_trials, timesteps = use_sequences.shape
    valid_codes = use_sequences[~np.isnan(use_sequences)]
    top = k * 4 * max(n_wheel_states, 1)
    if valid_codes.size and (valid_codes.min() < 0 or valid_codes.max() >= top):
        raise ValueError(f'codes span {valid_codes.min():.0f}..{valid_codes.max():.0f}, but k = {k}'
                         + (f', k_wheel = {n_wheel_states}' if n_wheel_states else ', no wheel')
                         + f' allows 0..{top - 1}: do the settings match the file?')
    out = np.zeros((n_trials, timesteps * n_raw))
    for t in range(timesteps):
        vals = use_sequences[:, t]
        nan_mask = np.isnan(vals)
        valid = ~nan_mask
        labels = vals[valid].astype(int)
        start = t * n_raw
        if len(labels):
            rows = np.flatnonzero(valid)
            out[rows, start + labels % k] = 1
            out[valid, start + k] = (labels // k) % 2
            out[valid, start + k + 1] = (labels // (2 * k)) % 2
            if n_wheel_states:
                out[rows, start + k + 2 + labels // (4 * k)] = 1
        if nan_mask.any():
            out[nan_mask, start:start + n_raw] = np.nan
    return out[:, [t * n_raw + c for t in range(timesteps) for c in cols]]


def feature_names(n_paw_states=8, n_wheel_states=0, prefix='Paw'):
    """Names of the FULL per-bin feature set that `reconstruct` returns."""
    return ([f'{prefix} {i}' for i in range(n_paw_states)] + ['Whisk', 'Lick']
            + [f'Wheel {i}' for i in range(n_wheel_states)])


def _reconstruct(Xmat, n_paw_states, n_wheel_states, ref_state, total):
    """Restore each dropped reference state as total - (sum of its group's kept columns), summed
    in binarize's column order (so results match the notebooks' old code to the last bit)."""
    Xmat = np.asarray(Xmat, dtype=float)
    n_raw, cols = _layout(n_paw_states, n_wheel_states, True, True, True, True, ref_state)
    npf = len(cols)
    assert Xmat.shape[1] % npf == 0, (f'{Xmat.shape[1]} features is not a multiple of {npf}; were all '
                                      'feature types kept and the settings the same as for binarize?')
    T = Xmat.shape[1] // npf
    R = Xmat.reshape(-1, T, npf)
    full = np.zeros((Xmat.shape[0], T, n_raw))
    full[:, :, cols] = R
    n_paw_kept = sum(c < n_paw_states for c in cols)
    if n_paw_kept < n_paw_states:
        full[:, :, ref_state] = total - R[:, :, :n_paw_kept].sum(axis=2)
    if n_wheel_states:
        w0 = n_paw_kept + 2
        n_wheel_kept = npf - w0
        if n_wheel_kept < n_wheel_states:
            full[:, :, n_paw_states + 2 + ref_state] = total - R[:, :, w0:w0 + n_wheel_kept].sum(axis=2)
    return full


def reconstruct(Xmat, n_paw_states=8, n_wheel_states=0, ref_state=1):
    """
    Session-averaged binarize() output (all feature types kept) -> (n_sessions, n_bins, full
    features), with each dropped reference state restored as 1 - (sum of its group's others).
    Feature order: paw 0..k-1, whisk, lick, wheel 0..k_wheel-1 (see feature_names).
    """
    return _reconstruct(Xmat, n_paw_states, n_wheel_states, ref_state, 1.0)


def binarized_names(n_paw_states=8, n_wheel_states=0, keep_paw=True, keep_whisk=True, keep_lick=True,
                    keep_wheel=True, ref_state=1, prefix='Paw'):
    """Names of the per-bin columns binarize() returns, in its order (reference states left out)."""
    full = feature_names(n_paw_states, n_wheel_states, prefix)
    _, cols = _layout(n_paw_states, n_wheel_states, keep_paw, keep_whisk, keep_lick, keep_wheel, ref_state)
    return [full[c] for c in cols]


def modalities(n_paw_states=8, n_wheel_states=0):
    """Feature indices (in reconstruct's order) per modality."""
    out = {'Paw': list(range(n_paw_states)), 'Whisk': [n_paw_states], 'Lick': [n_paw_states + 1]}
    if n_wheel_states:
        out['Wheel'] = list(range(n_paw_states + 2, n_paw_states + 2 + n_wheel_states))
    return out


def reconstruct_diff(Xmat, n_paw_states=8, n_wheel_states=0, ref_state=1):
    """reconstruct() for a DIFFERENCE of profiles: a dropped reference state is -sum(others), not
    1 - sum(others), because the constant cancels in a contrast."""
    return _reconstruct(Xmat, n_paw_states, n_wheel_states, ref_state, 0.0)


def _palette(var, state_space, k, datasets=('proficient', 'session_1', 'neuromodulators')):
    """Colours of a 3.3 state set, from the state_palette.csv it wrote next to its states
    (lightness = vigor, hue = which source leads); None if absent or written for another k."""
    import pandas as pd
    for d in datasets:
        f = sp.DATASETS[d]['root'] + sp.STATE_SPACES[state_space].format(var=var) + 'state_palette.csv'
        if os.path.exists(f):
            pal = pd.read_csv(f)
            if len(pal) == k and list(pal['state']) == list(range(k)):
                return list(pal['colour'])
    return None


def feature_colors(version='new', movement='left_paw_wheel', state_space='stages', k=8, wheel=False,
                   k_wheel=8, legacy_colors=None, **_):
    """One colour per feature_names() entry, and where the movement colours came from.
    New syllables: 3.3's state_palette.csv for that state set (falls back to a categorical
    palette). Legacy: `legacy_colors` (pass paper_style.SYLLABLE_COLORS to keep old figures)."""
    import matplotlib.pyplot as plt
    greys = ['#b8b8b8', '#484949']                     # whisk, lick (paper_style's)
    if version == 'legacy' and legacy_colors is not None:
        return list(legacy_colors), 'paper_style.SYLLABLE_COLORS (legacy)'
    mov = _palette(sp.MOVEMENTS[movement], state_space, k)
    src = f'state_palette.csv of {sp.MOVEMENTS[movement]} / {state_space}'
    if mov is None:
        mov = [plt.cm.Set3(i % 12) for i in range(k)]
        src = 'Set3 (no matching state_palette.csv)'
    whl = []
    if wheel:
        whl = _palette(sp.WHEEL_VAR, state_space, k_wheel) or [plt.cm.Blues(0.3 + 0.6 * i / max(k_wheel - 1, 1))
                                                               for i in range(k_wheel)]
    return list(mov) + greys + list(whl), src
