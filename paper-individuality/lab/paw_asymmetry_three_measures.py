"""
PAW ASYMMETRY FROM THREE INDEPENDENT MEASURES, PER SESSION
==========================================================
Is there a LAB-dependent paw asymmetry? Three measures, deliberately chosen because they
fail in different ways:

  LI_raw   pixel speed of l_paw vs r_paw, from `data/states_files`. Those two columns come
           from DIFFERENT CAMERAS -- the design-matrix pipeline takes `paw_r` (the near,
           well-resolved paw) from each camera and calls them l_paw / r_paw. Both are on the
           same 1/60 s grid, so frame rate is already equalised; the left camera's 2x higher
           resolution is corrected by the /2 the pipeline applies to l_paw. What remains is
           any residual per-camera gain, which is a RIG property -- so a lab effect here is
           ambiguous between animal and apparatus by construction.

  LI_syll  occupancy-weighted asymmetry of the 8-state paw alphabet. The states are fitted on
           BOTH paws jointly, so no single state is "left" or "right" a priori. Each state is
           therefore CHARACTERISED GLOBALLY first -- pooled over every session, the mean
           l_paw and r_paw speed while that state is active -- giving each state a fixed
           asymmetry score. A session's index is then its occupancy-weighted average of those
           scores: does this mouse spend more time in the states that are left-dominant?
           It inherits the same two-camera comparison as LI_raw, one level removed.

  LI_LA    lightningAction, averaging the two WITHIN-camera indices. Within a camera both
           paws share resolution, frame rate and classifier, so camera gain cancels in the
           ratio; the near/far term flips sign between cameras, so averaging cancels that
           too. This is the only one of the three with no cross-camera term -- it is the
           arbiter when the other two disagree.

SIGN: positive = LEFT paw more active, verified against the raw point clouds (SWC_054 is
left-dominant, NYU-47 right-dominant) rather than taken from the file naming, which is
view-relative and carries no anatomical side.
"""
import os
import sys
import glob
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions

STATES_DIR = ROOT / 'data' / 'states_files'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
N_PAW = 8
LEFT_SCALE = 2.0          # left camera is 1280x1024, right 640x512 -- as in the pipeline
OUT = HERE / 'paw_asymmetry_three_measures.csv'


def li(a, b):
    return (a - b) / (a + b) if (a + b) > 0 else np.nan


def session_speeds(eid, mouse):
    """per-bin speed of each paw, plus the paw state, from the aligned states file."""
    f = glob.glob(str(STATES_DIR / f'8_states_file_{eid}_{mouse}'))
    if not f:
        return None
    d = pd.read_parquet(f[0], columns=['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y',
                                       'most_likely_states'])
    lx, ly = d['l_paw_x'].to_numpy() / LEFT_SCALE, d['l_paw_y'].to_numpy() / LEFT_SCALE
    rx, ry = d['r_paw_x'].to_numpy(), d['r_paw_y'].to_numpy()
    sl = np.r_[np.nan, np.hypot(np.diff(lx), np.diff(ly))]
    sr = np.r_[np.nan, np.hypot(np.diff(rx), np.diff(ry))]
    paw_state = np.mod(d['most_likely_states'].to_numpy(dtype=float), N_PAW)
    ok = np.isfinite(sl) & np.isfinite(sr) & np.isfinite(paw_state)
    return sl[ok], sr[ok], paw_state[ok].astype(int)


def main():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = dd[['mouse_name', 'session']].drop_duplicates().set_index('session')['mouse_name']

    # ---- pass 1: per-session speeds, and the global state characterisation
    per_session, tot_l, tot_r, n_state = {}, np.zeros(N_PAW), np.zeros(N_PAW), np.zeros(N_PAW)
    for i, eid in enumerate(ss.index, 1):
        r = session_speeds(eid, mouse[eid])
        if r is None:
            continue
        sl, sr, st = r
        per_session[eid] = (sl, sr, st)
        for s in range(N_PAW):
            m = st == s
            if m.any():
                tot_l[s] += sl[m].sum()
                tot_r[s] += sr[m].sum()
                n_state[s] += m.sum()
        if i % 50 == 0:
            print(f'  {i}/{len(ss)}', flush=True)

    state_li = np.array([li(tot_l[s] / max(n_state[s], 1), tot_r[s] / max(n_state[s], 1))
                         for s in range(N_PAW)])
    print('\nglobal paw-state characterisation (positive = left-dominant state):')
    for s in range(N_PAW):
        print(f'  state {s}: LI {state_li[s]:+.3f}   mean speed L {tot_l[s]/max(n_state[s],1):.3f} '
              f'R {tot_r[s]/max(n_state[s],1):.3f}   {100*n_state[s]/n_state.sum():5.1f}% of bins')

    # ---- pass 2: the two states-file indices per session
    rows = []
    for eid, (sl, sr, st) in per_session.items():
        occ = np.bincount(st, minlength=N_PAW) / len(st)
        rows.append(dict(session=eid, mouse=mouse[eid],
                         li_raw=li(np.nanmean(sl), np.nanmean(sr)),
                         li_syll=float(np.dot(occ, state_li)),
                         **{f'occ_{s}': occ[s] for s in range(N_PAW)}))
    R = pd.DataFrame(rows).set_index('session')

    # ---- the lightningAction index, averaged over the two within-camera views
    la = pd.read_csv(HERE / 'paw_asymmetry.csv').set_index('session')
    R['li_LA'] = (la['li_left_cam'] + la['li_right_cam']) / 2
    R['lab'] = functions.lab_labels(R.index, mouse_names=R['mouse'], verbose=False)
    R.to_csv(OUT)
    print(f'\n{len(R)} sessions, {R.mouse.nunique()} mice, {R.lab.nunique()} labs '
          f'({R.li_LA.notna().sum()} with lightningAction) -> {OUT}')
    print('\nper-measure means:')
    print(R[['li_raw', 'li_syll', 'li_LA']].describe().loc[['mean', 'std']].round(4).to_string())


if __name__ == '__main__':
    main()
