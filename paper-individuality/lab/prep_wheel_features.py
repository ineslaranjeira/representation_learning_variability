"""
WHEEL SPEED IN THE SAME 10-BIN-PER-EPOCH FORMAT AS THE SYLLABLES
================================================================
Follows learning_individuality/wheel/wheel_velocity_timepoints.ipynb (cells 18 and 22)
exactly: per-bin wheel velocity, labelled with trial and epoch, grouped by
(sample, trial_type, broader_label, mouse_name), each sequence rescaled to 10 bins with
`rescale_sequence(..., 'mean')` on |avg_wheel_vel|.

ONE DEPARTURE, AND IT IS THE POINT. That notebook rebuilds the alignment from
`design_matrix_*` + `session_trials_*` via align_bin_design_matrix -> states_per_trial_phase
-> broader_label. `data/states_files/8_states_file_<eid>_<mouse>` IS the output of those three
steps, carries `avg_wheel_vel` alongside `trial_id` and `broader_label`, and exists for all
269 sessions. Reading it instead skips ~an hour of re-derivation AND guarantees the wheel
bins are the SAME bins the syllable features were built from -- same trial ids, same epoch
boundaries -- which is what makes the three-way comparison exact rather than approximate.

WHY WHEEL MATTERS HERE: it comes from the rotary encoder, in physical units, with no camera
in the path. A lab effect that shows up in the syllables but not in the wheel cannot be the
video pipeline's doing.
"""
import os
import sys
import glob
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'segmentation'),
           str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions
# the 3-argument rescale_sequence (mean / mode) lives in segmentation_functions and is the
# one the wheel notebook uses; 4_mice/functions.py still carries an older 2-argument copy
# that is mode-only, so importing from the wrong module silently loses the estimator
from segmentation_functions import define_trial_types, rescale_sequence

STATES_DIR = ROOT / 'data' / 'states_files'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
OUT = HERE / 'wheel_10bin_sequences.pqt'
TRIAL_TYPE_AGG = ['correct_str', 'contrast_str', 'block_str', 'choice']
TARGET_LENGTH = 10
GROUP = ['sample', 'trial_type', 'broader_label', 'mouse_name']


def session_sequences(eid, mouse):
    f = glob.glob(str(STATES_DIR / f'8_states_file_{eid}_{mouse}'))
    if not f:
        return None
    d = pd.read_parquet(f[0])
    d = d.dropna(subset=['broader_label', 'trial_id', 'avg_wheel_vel']).copy()
    if len(d) == 0:
        return None
    d = define_trial_types(d, TRIAL_TYPE_AGG)
    # speed, not signed velocity: direction is choice-dependent and is a different feature
    d['avg_wheel_vel'] = np.abs(d['avg_wheel_vel'])
    g = (d.groupby(GROUP)['avg_wheel_vel'].apply(list).reset_index()
         .rename(columns={'avg_wheel_vel': 'sequence'}))
    g['binned_sequence'] = g['sequence'].apply(
        lambda s: rescale_sequence(s, TARGET_LENGTH, 'mean'))
    g['session'] = eid
    return g.drop(columns=['sequence'])


def main():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = dd[['mouse_name', 'session']].drop_duplicates().set_index('session')['mouse_name']
    out, missing = [], []
    for i, eid in enumerate(ss.index, 1):
        r = session_sequences(eid, mouse[eid])
        if r is None:
            missing.append(eid)
        else:
            out.append(r)
        if i % 40 == 0:
            print(f'  {i}/{len(ss)}', flush=True)
    allseq = pd.concat(out, ignore_index=True)
    allseq.to_parquet(OUT)
    print(f'\n{len(allseq)} trial-epoch rows, {allseq.session.nunique()} sessions, '
          f'{allseq.mouse_name.nunique()} mice -> {OUT}')
    if missing:
        print(f'{len(missing)} sessions had no states file')
    print(allseq.groupby('broader_label').size().to_string())


if __name__ == '__main__':
    main()
