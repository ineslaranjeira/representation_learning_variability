"""
IS PAW VIGOR LAB-DEPENDENT -- OR IS IT THE ZOOM?
=================================================
Asymmetry was safe to test naively because it is a RATIO: the camera's scale divides out.
Vigor is a MAGNITUDE, so it is exposed to every optical term the ratio cancelled. A mouse
filmed from closer produces larger pixel excursions for identical movement, and how far the
camera sits is a lab property -- `nose_x` is 74% between labs, `tube_len_px` 42%. So a lab
difference in pixel speed is close to guaranteed and means nothing on its own.

FOUR MEASURES, ordered by how much optics they can carry:

  vigor_px    mean paw speed in PIXELS per 1/60 s bin, from `data/states_files`
              (left-camera paw scaled by 1/2 for its 2x resolution, as the pipeline does).
              Fully exposed to zoom.
  vigor_tube  speed divided by `tube_len_px`. KEPT ONLY AS A COUNTER-EXAMPLE: the lick tube
              is NOT fixed hardware -- it can be cut longer or shorter between rigs -- so its
              apparent length confounds zoom with spout trimming. It is 56% session variance
              and correlates with paw speed at r = +0.07, so dividing by it injects the lab
              structure it carries instead of removing magnification. See zoom_and_lab_effects.py.
  vigor_pupil the same speed divided by `nose_to_pupil_px` -- nose tip to pupil centre, a
              distance ON THE ANIMAL at roughly the depth the paws move in. 88.5% lab and
              only 2.5% session (a rig property, as a zoom measure should be) and it does
              predict pixel speed, r = +0.278. THIS is the zoom correction to use: it puts
              the lab share of vigor at 0.000 (p = 0.70), with or without angelakilab.
  vigor_syll  occupancy-weighted mean speed of the 8 paw states, where each state's speed is
              a GLOBAL constant fitted across all sessions. A session only moves this by
              which states it occupies, so per-session zoom cannot enter.
  vigor_LA    fraction of frames lightningAction calls move or wheel_turn, near views only.
              A RATE, not an amplitude -- dimensionless, so optics cannot scale it at all.

READ THE ROWS AGAINST EACH OTHER. If lab survives from vigor_px down to vigor_LA, the animals
differ. If it collapses as the optical terms are removed, it was the rigs.

Per epoch as well as overall, because task-driven vigor (Choice) and spontaneous vigor (ITI)
need not behave alike.
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
import variance_partition as vp

STATES_DIR = ROOT / 'data' / 'states_files'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']
N_PAW = 8
LEFT_SCALE = 2.0
OUT = HERE / 'paw_vigor.csv'


def session_rows(eid, mouse):
    f = glob.glob(str(STATES_DIR / f'8_states_file_{eid}_{mouse}'))
    if not f:
        return None
    d = pd.read_parquet(f[0], columns=['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y',
                                       'most_likely_states', 'broader_label'])
    lx, ly = d['l_paw_x'].to_numpy() / LEFT_SCALE, d['l_paw_y'].to_numpy() / LEFT_SCALE
    rx, ry = d['r_paw_x'].to_numpy(), d['r_paw_y'].to_numpy()
    sl = np.r_[np.nan, np.hypot(np.diff(lx), np.diff(ly))]
    sr = np.r_[np.nan, np.hypot(np.diff(rx), np.diff(ry))]
    speed = np.nanmean(np.vstack([sl, sr]), axis=0)            # both paws, mean
    st = np.mod(d['most_likely_states'].to_numpy(dtype=float), N_PAW)
    return speed, st, d['broader_label'].to_numpy()


def main():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = dd[['mouse_name', 'session']].drop_duplicates().set_index('session')['mouse_name']

    cache, tot, n = {}, np.zeros(N_PAW), np.zeros(N_PAW)
    for i, eid in enumerate(ss.index, 1):
        r = session_rows(eid, mouse[eid])
        if r is None:
            continue
        speed, st, ep = r
        cache[eid] = r
        ok = np.isfinite(speed) & np.isfinite(st)
        for s in range(N_PAW):
            m = ok & (st == s)
            if m.any():
                tot[s] += speed[m].sum()
                n[s] += m.sum()
        if i % 50 == 0:
            print(f'  {i}/{len(ss)}', flush=True)
    state_speed = tot / np.maximum(n, 1)
    print('\nglobal paw-state speeds (px per 1/60 s bin):')
    for s in range(N_PAW):
        print(f'  state {s}: {state_speed[s]:7.3f}   {100*n[s]/n.sum():5.1f}% of bins')

    geo = pd.read_csv(HERE / 'rig_geometry.csv', index_col=0)
    la = pd.read_csv(HERE / 'paw_asymmetry.csv').set_index('session')

    rows = []
    for eid, (speed, st, ep) in cache.items():
        ok = np.isfinite(speed) & np.isfinite(st)
        occ = np.bincount(st[ok].astype(int), minlength=N_PAW) / max(ok.sum(), 1)
        rec = dict(session=eid, mouse=mouse[eid],
                   vigor_px=float(np.nanmean(speed[ok])),
                   vigor_syll=float(np.dot(occ, state_speed)))
        tube = geo['tube_len_px'].get(eid, np.nan)
        rec['tube_len_px'] = tube
        rec['vigor_tube'] = rec['vigor_px'] / tube if np.isfinite(tube) and tube > 0 else np.nan
        pup = geo['nose_to_pupil_px'].get(eid, np.nan)
        rec['nose_to_pupil_px'] = pup
        rec['vigor_pupil'] = rec['vigor_px'] / pup if np.isfinite(pup) and pup > 0 else np.nan
        if eid in la.index:
            rec['vigor_LA'] = float((la.loc[eid, 'act_left_near'] + la.loc[eid, 'act_right_near']) / 2)
        for e in EPOCHS:
            m = ok & (ep == e)
            rec[f'px_{e}'] = float(np.nanmean(speed[m])) if m.any() else np.nan
        rows.append(rec)
    R = pd.DataFrame(rows).set_index('session')
    R['lab'] = functions.lab_labels(R.index, mouse_names=R['mouse'], verbose=False)
    R.to_csv(OUT)

    MEAS = {'vigor_px': 'pixel speed (raw)',
            'vigor_pupil': 'speed / nose-to-pupil (zoom-corrected)',
            'vigor_tube': 'speed / tube length (BAD ruler, kept as a counter-example)',
            'vigor_syll': 'state-occupancy vigor (no per-session scale)',
            'vigor_LA': 'lightningAction active fraction (a rate)'}
    print(f'\n{len(R)} sessions, {R.mouse.nunique()} mice, {R.lab.nunique()} labs\n')
    print('IS PAW VIGOR LAB-DEPENDENT?')
    print(f'  {"measure":46s} {"n":>4s} {"lab":>7s} {"mouse":>7s} {"session":>8s}     p')
    for c, lab in MEAS.items():
        D = R[[c]].dropna()
        if len(D) < 30:
            continue
        lb, ms = R.loc[D.index, 'lab'], R.loc[D.index, 'mouse']
        vc, _ = vp.nested_variance_components(D, lb, ms)
        s = vc.clip(lower=0).iloc[0]; s = s / s.sum()
        _, _, pv = vp.permutation_null_eta2(D, lb, ms, n_perm=5000, seed=0)
        print(f'  {lab:46s} {len(D):4d} {s["sigma2_lab"]:7.3f} {s["sigma2_mouse"]:7.3f} '
              f'{s["sigma2_session"]:8.3f}  {pv:.4f}')

    print('\nTHE MECHANISM: does pixel vigor track the zoom?')
    d = R[['vigor_px', 'tube_len_px']].dropna()
    from scipy import stats as st_
    r = st_.pearsonr(d['vigor_px'], d['tube_len_px'])
    print(f'  vigor_px vs tube_len_px across sessions: r = {r[0]:+.3f} (p = {r[1]:.1g}, n = {len(d)})')
    pm = R.groupby('mouse')[['vigor_px', 'tube_len_px']].mean().dropna()
    print(f'  per mouse: r = {st_.pearsonr(pm["vigor_px"], pm["tube_len_px"])[0]:+.3f}')

    print('\nPER EPOCH (raw pixel speed)')
    print(f'  {"epoch":16s} {"lab":>7s} {"mouse":>7s} {"session":>8s}     p    mean px/bin')
    for e in EPOCHS:
        D = R[[f'px_{e}']].dropna()
        lb, ms = R.loc[D.index, 'lab'], R.loc[D.index, 'mouse']
        vc, _ = vp.nested_variance_components(D, lb, ms)
        s = vc.clip(lower=0).iloc[0]; s = s / s.sum()
        _, _, pv = vp.permutation_null_eta2(D, lb, ms, n_perm=2000, seed=0)
        print(f'  {e:16s} {s["sigma2_lab"]:7.3f} {s["sigma2_mouse"]:7.3f} {s["sigma2_session"]:8.3f}'
              f'  {pv:.4f}   {D.mean().iloc[0]:8.3f}')

    print('\nPER-LAB MEANS (mouse-level)')
    t = pd.DataFrame({m: R.groupby(['lab', 'mouse'])[m].mean().groupby('lab').mean()
                      for m in MEAS if m in R})
    t['n_mice'] = R.groupby(['lab', 'mouse']).size().groupby('lab').size()
    print(t.sort_values('vigor_px').round(4).to_string())
    print(f'\nsaved {OUT}')


if __name__ == '__main__':
    main()
