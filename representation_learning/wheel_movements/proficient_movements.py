"""
Wheel-movement shapes for the PROFICIENT (BWM) sessions of the individuality LDA cohort.

Cohort = the sessions of the current proficient syllables (8_k_10_bin_syllables_02-10-2026) after
the QC sheet's 'filtered out' exclusions, keeping mice with >= 3 sessions -- the selection of
4_mice.functions.filter_sequences / syllable_sweeps (QC_STRICTNESS = 'filtered_out').

Velocity: the 60 Hz `avg_wheel_vel` of the BWM design matrices (data/design_matrices/).
Movements: IBL's alf/_ibl_wheelMoves.intervals, cached here with the trials table:
    paper-individuality/data/wheel_movements_proficient/{intervals,trials}_{eid}_{mouse}

Run as a script to fill the cache:  python proficient_movements.py
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent.parent / 'paper-individuality'
DM_DIR = PI / 'data' / 'design_matrices'
CACHE = PI / 'data' / 'wheel_movements_proficient'
SYLLABLES = PI / 'data' / '8_k_10_bin_syllables_02-10-2026'
FS = 60
PAD_S = 0.05
NT = 200


def cohort():
    """(mouse, eid) of the LDA cohort."""
    sys.path.insert(0, str(PI / '4_mice'))
    from functions import qc_exclusions
    d = pd.read_parquet(SYLLABLES, columns=['mouse_name', 'sample'])
    s = pd.DataFrame({'mouse': d.mouse_name, 'eid': d['sample'].str[:36]}).drop_duplicates()
    s = s[~s.eid.isin(qc_exclusions('Proficient', 'filtered_out', verbose=False))]
    s = s[s.groupby('mouse').eid.transform('size') >= 3]
    return s.sort_values(['mouse', 'eid']).reset_index(drop=True)


def fetch(eid, mouse):
    f = CACHE / f'intervals_{eid}_{mouse}.npy'
    if f.exists():
        return 'exists'
    from one.api import ONE
    one = ONE(mode='remote')
    try:
        iv = one.load_object(eid, 'wheelMoves', collection='alf')['intervals']
        tr = one.load_object(eid, 'trials', collection='alf')
        n = len(tr['intervals'])
        tr = type(tr)({k: v for k, v in tr.items() if len(v) == n}).to_df()
        tr.to_parquet(CACHE / f'trials_{eid}_{mouse}')
        np.save(f, iv)
        return 'ok'
    except Exception as e:
        return f'error: {type(e).__name__}: {e}'[:200]


def load_session(eid, mouse):
    """60 Hz times + wheel velocity, movement intervals, trials."""
    dm = pd.read_parquet(DM_DIR / f'design_matrix_{eid}_{mouse}', columns=['Bin', 'avg_wheel_vel'])
    iv = np.load(CACHE / f'intervals_{eid}_{mouse}.npy')
    tr = pd.read_parquet(CACHE / f'trials_{eid}_{mouse}')
    return dm.Bin.values, dm.avg_wheel_vel.values.astype(float), iv, tr


def shapes(t, v, intervals, pad=PAD_S, nt=NT, kind='linear', normalise=True):
    """Signed, time-normalised shape of every movement inside the design matrix's time range,
    divided by its peak speed (normalise=True) or in rad/s (normalise=False). Returns (shapes (n, nt), table with onset/offset/duration/peak)."""
    from scipy.interpolate import interp1d
    S, rows = [], []
    for k, (on, off) in enumerate(intervals):
        if on - pad < t[0] or off + pad > t[-1]:
            continue        # the design matrix only covers the cameras' overlap
        i0, i1 = np.searchsorted(t, [on - pad, off + pad])
        vv = v[i0:i1]
        if len(vv) < 4 or not np.isfinite(vv).all():
            continue
        peak = np.abs(vv).max()
        x = np.linspace(0, 1, len(vv))
        S.append(interp1d(x, vv, kind=kind)(np.linspace(0, 1, nt)) / (peak if normalise else 1))
        rows.append(dict(movement=k, onset=on, offset=off, duration=off - on, peak=peak))
    return np.array(S, np.float32), pd.DataFrame(rows)


if __name__ == '__main__':
    CACHE.mkdir(exist_ok=True)
    c = cohort()
    print(len(c), 'sessions,', c.mouse.nunique(), 'mice')
    res = Parallel(n_jobs=8, verbose=5)(delayed(fetch)(e, m) for m, e in zip(c.mouse, c.eid))
    print(pd.Series([r.split(':')[0] for r in res]).value_counts())
    for (m, e), r in zip(zip(c.mouse, c.eid), res):
        if r.startswith('error'):
            print(m, e, r)


# ---------------------------------------------------------------------------------- fitted models
def model_path(variant):
    return CACHE / f'model_{variant}.pkl'


def load_model(variant):
    """The global fit saved by 5_proficient_shapes_fit Part A ('normalised') / Part B ('absolute')."""
    import pickle
    with open(model_path(variant), 'rb') as f:
        return pickle.load(f)


def side_of(S, split):
    pos = np.clip(S, 0, None).sum(1)
    neg = np.clip(-S, 0, None).sum(1)
    si = (pos - neg) / (pos + neg)
    return np.where(si > split, 2, np.where(si < -split, 0, 1))


def predict(model, S, info):
    """Display cluster (0..K-1, ordered left -> both -> right, by vigor) of each movement."""
    F = np.column_stack([model['pca'].transform(S)[:, :model['n_pc']], np.log10(info.duration)])
    F = (F - model['MU']) / model['SD']
    if model['split_by_side']:
        g_ = side_of(S, model['SPLIT'])
        raw = np.empty(len(S), int)
        for g in range(3):
            if (g_ == g).any():
                raw[g_ == g] = model['GM'][g].predict(F[g_ == g]) + model['OFF'][g]
    else:
        raw = model['gmm'].predict(F)
    return np.array([model['REL'][r] for r in raw])


def session_labels(model, eid, mouse):
    """60 Hz times, velocity, the wheel cluster of every frame (-1 = not in a movement), trials."""
    t, v, iv, tr = load_session(eid, mouse)
    S, info = shapes(t, v, iv, normalise=model['normalise'])
    frame = np.full(len(t), -1)
    if len(info):
        for (_, q), c in zip(info.iterrows(), predict(model, S, info)):
            i0, i1 = np.searchsorted(t, [q.onset, q.offset])
            frame[i0:i1] = c
    return t, v, frame, tr
