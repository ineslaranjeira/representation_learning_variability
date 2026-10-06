"""
Trials of every habituation session, saved only to validate the trial-free session window of
habituation_design_matrix.ipynb (its last section). Nothing in the design matrix uses them.

ALF trials where registered (69 of the 136 sessions), otherwise extracted from
_iblrig_taskData.raw.jsonable with HabituationTrials. On a session that has both, the extraction
reproduces the ALF stimOn/feedback/goCueTrigger times exactly; only `intervals` differs, by up to
2.4 s, because the trial-end definition changed in ibllib.

Writes session_trials_<eid>_<mouse> to data/individuality/habituation/trial_window_reference/.

@author: Ines
"""
#%%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader
from ibllib.io.extractors.habituation_trials import HabituationTrials
from one.api import ONE

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUT = HERE.parent / 'data' / 'individuality' / 'habituation' / 'trial_window_reference'
RAW_TASK_DATASETS = ['_iblrig_taskData.raw.jsonable', '_iblrig_taskSettings.raw.json',
                     '_iblrig_encoderEvents.raw.ssv', '_iblrig_encoderTrialInfo.raw.ssv']

one = ONE(mode='remote')
inventory = pd.read_csv(HERE / 'habituation_lp_sessions.csv')
usable = inventory[inventory['lp_left'] & inventory['cam_times_left'] & inventory['me_left']]


def load_trials(session):
    """ ALF trials if registered, otherwise extracted from the raw task data, in SessionLoader's
    format (intervals split into intervals_0 / intervals_1). """
    sl = SessionLoader(eid=session, one=one)
    try:
        sl.load_trials()
        return sl.trials, 'alf'
    except Exception:
        pass
    for dataset in RAW_TASK_DATASETS:
        try:
            one.load_dataset(session, dataset, download_only=True)
        except Exception:
            pass  # encoder files are not on every session; the extractor fails loudly if it needs them
    extracted, _ = HabituationTrials(one.eid2path(session)).extract(save=False)
    trials = pd.DataFrame({k: v for k, v in extracted.items() if np.ndim(v) == 1})
    trials['intervals_0'] = extracted['intervals'][:, 0]
    trials['intervals_1'] = extracted['intervals'][:, 1]
    return trials, 'raw_extracted'


#%%
OUT.mkdir(parents=True, exist_ok=True)
for eid, mouse in zip(usable['eid'], usable['mouse_name']):
    f = OUT / f'session_trials_{eid}_{mouse}'
    if f.exists():
        continue
    try:
        trials, source = load_trials(eid)
        trials.to_parquet(f, compression='gzip')
        print(f'{eid} {mouse}: {len(trials)} trials ({source})', flush=True)
    except Exception as e:
        print(f'{eid} {mouse}: FAILED {e!r}', flush=True)

print(f'{len(list(OUT.glob("session_trials_*")))} of {len(usable)} sessions have trials')

# %%
