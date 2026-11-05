"""
Wheel movement states (the proficient model, segmentation/wheel_movement_clusters.py) in the learning timepoints.

Timepoints = the QC sheet's Early (first training session), Late (last training session), Pre-rec (up to 3 biased
sessions before recording) and Proficient (the LDA cohort), 'filtered out' sessions dropped.

The proficient model is fitted on the 60 Hz design-matrix wheel velocity (`avg_wheel_vel`). The training / biased
design matrices are 30 Hz, so here the wheel comes from the 200 Hz wheel-only design matrices instead
(representation_learning/wheel_design_matrices.py: brainbox 20 Hz zero-phase low-pass of the 1 kHz interpolated
position), averaged into 1/60 s bins to match the proficient input. Movements = IBL's alf `wheelMoves` intervals,
as for the proficient sessions.

Caches (new files only):
    data/training/training_wheel_200Hz/     existing 200 Hz training sessions (Early / Late are there)
    data/training/stages_wheel_200Hz/       200 Hz design matrices + trials of the Pre-rec (biased) sessions
    data/training/wheel_moves/              intervals_<eid>_<mouse>.npy (alf wheelMoves)

    python wheel_states_stages.py fetch      # download / build what is missing
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parents[1]
REPO = PI.parent
for p in (PI / 'segmentation', PI / 'learning_individuality', REPO / 'representation_learning'):
    sys.path.insert(0, str(p))
import wheel_movement_clusters as wmc

TRAIN_DIR = PI / 'data' / 'training'
DM_TRAINING = TRAIN_DIR / 'training_wheel_200Hz'
DM_STAGES = TRAIN_DIR / 'stages_wheel_200Hz'
MOVES = TRAIN_DIR / 'wheel_moves'
TIMEPOINTS = ('Early', 'Late', 'Pre-rec', 'Proficient')
FS_OUT = 60.0


def sessions():
    """(timepoint, mouse, eid) of Early / Late / Pre-rec, QC 'filtered out' dropped; Proficient = the LDA cohort."""
    import session_filters as sf
    t = sf.load_session_table()
    t = t[t.timepoint.isin(['Early', 'Late', 'Pre-rec']) & (t['Used in paper'].str.strip() != 'filtered out')]
    s = t[['timepoint', 'mouse_name', 'eid']].rename(columns={'mouse_name': 'mouse'})
    prof = wmc.fit_cohort().assign(timepoint='Proficient')
    return pd.concat([s, prof[['timepoint', 'mouse', 'eid']]], ignore_index=True)


def _dm_path(mouse, eid):
    for d in (DM_TRAINING, DM_STAGES):
        if (d / f'design_matrix_{eid}_{mouse}').exists():
            return d
    return None


def _fetch_one(mouse, eid):
    out = []
    if _dm_path(mouse, eid) is None:
        import wheel_design_matrices as wdm
        wdm.OUT_DIR = DM_STAGES                 # set inside the worker (joblib re-imports modules)
        out.append(wdm.process(mouse, eid)['status'])
    f = MOVES / f'intervals_{eid}_{mouse}.npy'
    if not f.exists():
        from one.api import ONE
        try:
            np.save(f, ONE(mode='remote').load_object(eid, 'wheelMoves', collection='alf')['intervals'])
            out.append('moves ok')
        except Exception as e:
            out.append(f'moves error: {type(e).__name__}: {e}'[:200])
    return ';'.join(out) or 'cached'


def fetch(n_jobs=4):
    DM_STAGES.mkdir(parents=True, exist_ok=True)
    MOVES.mkdir(parents=True, exist_ok=True)
    s = sessions()
    s = s[s.timepoint != 'Proficient']
    res = Parallel(n_jobs=n_jobs)(delayed(_fetch_one)(m, e) for m, e in zip(s.mouse, s.eid))
    return pd.Series(res, index=pd.MultiIndex.from_frame(s[['timepoint', 'mouse', 'eid']]))


def load_stage_session(mouse, eid):
    """60 Hz bin-averaged velocity (times, rad/s), movement intervals, trials of a training / biased session."""
    d = _dm_path(mouse, eid)
    dm = pd.read_parquet(d / f'design_matrix_{eid}_{mouse}', columns=['times', 'velocity'])
    tr = pd.read_parquet(d / f'trials_{eid}_{mouse}')
    t, v = dm.times.to_numpy(), dm.velocity.to_numpy(np.float64)
    edges = np.arange(t[0], t[-1], 1 / FS_OUT)
    idx = np.searchsorted(edges, t, side='right') - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    v60 = np.bincount(idx[ok], v[ok], minlength=len(edges) - 1) / np.maximum(np.bincount(idx[ok], minlength=len(edges) - 1), 1)
    t60 = edges[:-1] + 0.5 / FS_OUT
    iv = np.load(MOVES / f'intervals_{eid}_{mouse}.npy')
    return t60, v60, iv, tr


def nosplit_model(k=7):
    with open(wmc.states_dir('proficient') + f'model_k{k}_nosplit.pkl', 'rb') as f:
        return pickle.load(f)


def label_session(timepoint, mouse, eid):
    """One row per movement: timepoint, mouse, eid, onset, duration, peak, epoch, trial, balance, c20, s8, u6."""
    if timepoint == 'Proficient':
        S, info = wmc.session_shapes('proficient', mouse, eid)
        n_trials = len(pd.read_parquet(wmc.moves_dir('proficient') + f'trials_{eid}_{mouse}'))
    else:
        if _dm_path(mouse, eid) is None or not (MOVES / f'intervals_{eid}_{mouse}.npy').exists():
            return None
        t, v, iv, tr = load_stage_session(mouse, eid)
        S, info = wmc.shapes(t, v, iv)
        if len(info):
            ep = [wmc.epoch_of(o, tr) for o in info.onset]
            info['epoch'] = [a for a, _ in ep]
            info['trial'] = [b for _, b in ep]
        n_trials = len(tr)
    if not len(info):
        return None
    paper = wmc.load_model('proficient', 21)
    m8 = wmc.state_map(paper, 'small_side_vigor')
    info = info[['onset', 'duration', 'peak', 'epoch', 'trial']].copy()
    info['c20'] = wmc.predict(paper, S, info.duration)
    info['s8'] = [m8[c] for c in info.c20]                       # 1..7 (0 = still is not a movement)
    info['u6'] = wmc.predict(nosplit_model(), S, info.duration)
    info['balance'] = wmc.balance(S)
    info['dir_half'] = np.where(S[:, :wmc.NT // 2].sum(1) >= 0, 1, -1)
    return info.assign(timepoint=timepoint, mouse=mouse, eid=eid, n_trials=n_trials)


def label_all(n_jobs=8):
    s = sessions()
    out = Parallel(n_jobs=n_jobs)(delayed(label_session)(*r) for r in s[['timepoint', 'mouse', 'eid']].itertuples(index=False))
    return pd.concat([o for o in out if o is not None], ignore_index=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['fetch']:
        r = fetch()
        print(r.str.split(':').str[0].groupby(level=0).value_counts())
        print(r[r.str.contains('error')])
