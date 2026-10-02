"""
RAW vs SYLLABLES, WITH PAW VELOCITY INSTEAD OF ITS WAVELETS
===========================================================
raw_vs_syllables.py gave the raw side the paw WAVELET features (20 per bin -> 800 dims),
so it had 2.4x the syllables' dimensions, and dimensions alone help any decoder. Here the
raw paw signal is the velocity the wavelets are computed from, as 3.1.1_paw_wavelets
defines it (diff(position / camera resolution) * 60, per axis; left camera resolution 2,
right 1), taken as SPEED |v| per axis because signed velocity cancels in a trial average.
Per frame log(|v| + 1 px/s), then the same 4 epochs x 10 bins, equal weight per trial.

  |v| per axis   l/r paw x/y            4 per bin  + whisk + lick  = 6 x 40 = 240 dims
  speed per paw  hypot(vx, vy), l/r      2 per bin  + whisk + lick  = 4 x 40 = 160 dims

against the syllables' 9 per bin (360). Whisk and lick are as in raw_vs_syllables.py.
Same cohort, protocol and column standardisation. Output: results_raw_velocity_vs_syllables.txt
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import raw_vs_syllables as RS                     # time_resolved, zs, cohort helpers, paths
from raw_vs_syllables import Z, me, lab_labels, lomo_lab_acc, DATA, HERE

V_EPS = 1.0                                       # px/s; median paw speed is ~2-12 px/s
RES = {'l_paw': 2, 'r_paw': 1}                    # get_speed's RESOLUTION: left / right camera
AXES = [f'{p}_{a}' for p in ('l_paw', 'r_paw') for a in 'xy']
CACHE = HERE / 'raw_velocity_features.npz'


def session_block(eid, mouse):
    sf = DATA / 'states_files' / f'8_states_file_{eid}_{mouse}'
    if not sf.exists():
        return None
    S = pd.read_parquet(sf, columns=['Bin', 'trial_id', 'broader_label', 'whisker_me',
                                     'Lick count'] + AXES).sort_values('Bin').reset_index(drop=True)
    V = {}
    for c in AXES:
        v = np.diff(S[c].to_numpy(float) / RES[c[:5]]) * 60
        V[c] = np.r_[np.nan, v]                                     # velocity at frame t: t-1 -> t
    cols = {}
    for c in AXES:
        cols[f'v_{c}'] = np.log(np.abs(V[c]) + V_EPS)
    for p in ('l_paw', 'r_paw'):
        cols[f's_{p}'] = np.log(np.hypot(V[f'{p}_x'], V[f'{p}_y']) + V_EPS)
    cols['whisk'] = np.log(S['whisker_me'].to_numpy(float))
    cols['lick'] = S['Lick count'].to_numpy(float)
    for c in list(cols):                                            # session z: full-session stats
        x = cols[c]
        cols[c + '_z'] = (x - np.nanmean(x)) / np.nanstd(x)
    D = pd.concat([S[['trial_id', 'broader_label']], pd.DataFrame(cols)], axis=1)
    vel = [f'v_{c}' for c in AXES]
    spd = ['s_l_paw', 's_r_paw']
    wl = ['whisk', 'lick']
    z = lambda cs: [c + '_z' for c in cs]                           # noqa: E731
    return {
        'vel_wl_global': RS.time_resolved(D, vel + wl).ravel(),
        'vel_wl_sessz': RS.time_resolved(D, z(vel + wl)).ravel(),
        'vel_global': RS.time_resolved(D, vel).ravel(),
        'vel_sessz': RS.time_resolved(D, z(vel)).ravel(),
        'spd_wl_global': RS.time_resolved(D, spd + wl).ravel(),
        'spd_wl_sessz': RS.time_resolved(D, z(spd + wl)).ravel(),
        'spd_global': RS.time_resolved(D, spd).ravel(),
        'spd_sessz': RS.time_resolved(D, z(spd)).ravel(),
    }


def main():
    tags = sorted(f[len('paw_vel_wavelets_'):] for f in os.listdir(DATA / 'paw_wavelets')
                  if f.startswith('paw_vel_wavelets_'))
    sessions = [(t[:36], t[37:]) for t in tags]
    if CACHE.exists():
        raw = np.load(CACHE, allow_pickle=True)['raw'].item()
    else:
        res = Parallel(n_jobs=18)(delayed(session_block)(e, m) for e, m in sessions)
        raw = {}
        for (eid, _), r in zip(sessions, res):
            for k, v in (r or {}).items():
                raw.setdefault(k, {})[eid] = v
        np.savez(CACHE, raw=raw)
        print(f'cached {CACHE}')

    syl, mouse_of = me.build_design_matrix(RS.SYLLABLE_FILE)
    idx = [s for s in syl.index if all(s in raw[k] for k in raw)]
    mice = mouse_of.loc[idx].to_numpy()
    labs = np.array(list(lab_labels(pd.Index(idx), mouse_names=mouse_of.loc[idx], verbose=False)))
    y = pd.factorize(mice)[0]
    lab_y = pd.factorize(labs)[0]
    print(f'\nsyllables: {RS.SYLLABLE_FILE}')
    print(f'cohort: {len(idx)} sessions, {len(set(mice))} mice, {len(set(labs))} labs')

    S360 = syl.loc[idx].to_numpy(float)
    R = {k: np.vstack([raw[k][s] for s in idx]) for k in raw}
    BLOCKS = [
        ('SYLLABLES', None),
        ('  paw + whisk + lick (the 360)', S360),
        ('  paw only', RS.paw_only(S360)),
        ('RAW VELOCITY, |v| per axis (4 per bin)', None),
        ('  paw + whisk + lick, global', R['vel_wl_global']),
        ('  paw + whisk + lick, session z', R['vel_wl_sessz']),
        ('  paw only, global', R['vel_global']),
        ('  paw only, session z', R['vel_sessz']),
        ('RAW SPEED PER PAW (2 per bin)', None),
        ('  paw + whisk + lick, global', R['spd_wl_global']),
        ('  paw + whisk + lick, session z', R['spd_wl_sessz']),
        ('  paw only, global', R['spd_global']),
        ('  paw only, session z', R['spd_sessz']),
    ]

    def score(name, X):
        X = RS.zs(X)
        acc, _ = Z.loso_score(X, y)
        accc, _ = Z.loso_score(Z.lab_center(X, labs), y)
        lacc, _ = Z.loso_score(X, lab_y, n_repeats=1)
        return (f'{name:44s} {X.shape[1]:5d} {acc:6.3f} {accc:9.3f} {Z.lab_eta2(X, labs):8.3f} '
                f'{lacc:8.3f} {lomo_lab_acc(X, labs, mice):8.3f} {Z.icc1(X, mice):6.3f}')

    rows = iter(Parallel(n_jobs=18)(delayed(score)(n, X) for n, X in BLOCKS if X is not None))
    print('\n' + '=' * 98)
    print(f'MOUSE vs LAB, SYLLABLES vs RAW VELOCITY   (LOSO; chance: mouse {1 / len(set(mice)):.3f}, '
          f'lab ~0.100)')
    print('=' * 98)
    print(f'{"block":44s} {"dims":>5s} {"mouse":>6s} {"mouse|lab":>9s} {"lab eta2":>8s} '
          f'{"lab LOSO":>8s} {"lab LOMO":>8s} {"ICC":>6s}')
    for name, X in BLOCKS:
        print(name if X is None else next(rows))


if __name__ == '__main__':
    main()
