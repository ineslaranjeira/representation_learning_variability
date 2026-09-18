"""
IS THE LAB EFFECT ALSO PRESENT IN AN INDEPENDENT ACTION CLASSIFIER?
==================================================================
The syllable features carry a lab signature that measured rig geometry mostly explains
(1_lab_variance.ipynb section 4). A sharper test of the same claim: IBL's lightningAction
paw-state classifier is trained across labs, is ensembled, and is not built from the
pixel-scale wavelets our own HMM consumes. If the lab effect is our measurement pipeline
rather than the animals, it should be weaker or absent there.

DATA  alf/lightningaction/_ibl_{left,right}Camera.pawstates.pqt -- per-frame posteriors over
      five paw actions (background, still, move, wheel_turn, groom) for paw_r and paw_l,
      plus per-state ensemble variances. Summarised per session as argmax occupancy.

COVERAGE IS THE LIMIT. Only 15 of the 269 LDA sessions have these files cached locally.
Widening to every cached session of an LDA mouse gives 45 sessions / 27 mice / 10 labs, but
18 of those mice have a single session -- so the MOUSE is the unit here, and the design
supports a lab test on mouse means and nothing finer.

TWO GOTCHAS, both checked in the code below:
  * PAW NAMING IS MIRRORED BETWEEN THE CAMERA FILES. left-camera `paw_r` and right-camera
    `paw_r` are DIFFERENT physical paws: cross-pairing correlates at r = 0.90-0.96 while
    same-name pairing gives only 0.53-0.72. Anyone comparing the two cameras by column name
    is comparing two paws and will conclude the classifier is view-dependent when it is not.
  * eta^2 has a large floor with 27 mice and 10 labs (null ~0.35), so a raw eta^2 says
    nothing on its own; every test here is against a mouse-level permutation null, and the
    SAME test is run on the syllable features in the SAME mice, which is what makes a null
    result interpretable rather than just underpowered.
"""
import os
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions
import variance_partition as vp

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
OUT = HERE / 'lightningaction_states.csv'
STATES = ['background', 'still', 'move', 'wheel_turn', 'groom']
PAWS = ['paw_r', 'paw_l']
OCC = [f'{p}_{s}_occ' for p in PAWS for s in STATES]


def extract():
    """One row per (session, camera): argmax occupancy of each state, per paw."""
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    lda_mice = set(dd['mouse_name'].unique())
    rows = []
    for d in sorted(glob.glob(CACHE + '*/Subjects/*/*/*/alf/lightningaction')):
        q = d.split('/')
        if q[-5] not in lda_mice:
            continue
        for cam, fn in [('left', '_ibl_leftCamera.pawstates.pqt'),
                        ('right', '_ibl_rightCamera.pawstates.pqt')]:
            f = os.path.join(d, fn)
            if not os.path.exists(f):
                continue
            df = pd.read_parquet(f, columns=[f'{p}_{s}' for p in PAWS for s in STATES])
            rec = dict(mouse=q[-5], pathlab=q[-7], date=q[-4], num=q[-3],
                       camera=cam, n_frames=len(df))
            for p in PAWS:
                am = np.argmax(df[[f'{p}_{s}' for s in STATES]].to_numpy(), axis=1)
                for j, s in enumerate(STATES):
                    rec[f'{p}_{s}_occ'] = float((am == j).mean())
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'{len(out)} rows -> {OUT}')
    return out


def main():
    S = pd.read_csv(OUT) if OUT.exists() else extract()
    L = S[S.camera == 'left']
    M = L.groupby('mouse').agg({**{c: 'mean' for c in OCC}, 'pathlab': 'first'})
    print(f'\n{len(L)} sessions, {L.mouse.nunique()} mice, {L.pathlab.nunique()} labs '
          f'(left camera)')

    # --- 1. lab bias in the action states, mouse level
    o, null, pv = vp.permutation_null_eta2(M[OCC], M['pathlab'], M.index, n_perm=5000)
    print(f'\nlab effect on the 10 occupancies: eta^2 {o:.3f}  null {null.mean():.3f} '
          f'[{np.percentile(null, 2.5):.3f}, {np.percentile(null, 97.5):.3f}]  p = {pv:.4f}')
    for c in OCC:
        oc, nc, pc = vp.permutation_null_eta2(M[[c]], M['pathlab'], M.index, n_perm=5000)
        flag = '  <-' if pc < 0.05 else ''
        print(f'    {c:22s} eta^2 {oc:.3f}  null {nc.mean():.3f}  p {pc:.3f}{flag}')

    # --- 2. the control that makes a null interpretable: same mice, our own features
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = pd.Series(ss.index.map(dd[['mouse_name', 'session']].drop_duplicates()
                                   .set_index('session')['mouse_name']), index=ss.index)
    lab = functions.lab_labels(ss.index, mouse_names=mouse, verbose=False)
    syl = ss.groupby(mouse.values).mean()
    labm = lab.groupby(mouse.values).first()
    common = [m for m in M.index if m in syl.index]
    print(f'\nSAME {len(common)} MICE, same test:')
    for nm, D, g in [('lightningAction (10 occupancies)', M.loc[common, OCC], M.loc[common, 'pathlab']),
                     ('syllable features (360)', syl.loc[common], labm.loc[common])]:
        o, null, pv = vp.permutation_null_eta2(D, g, D.index, n_perm=5000)
        print(f'  {nm:34s} eta^2 {o:.3f}  null {null.mean():.3f}  p = {pv:.4f}')

    # --- 3. per-lab means, and the one artifact that does show up
    print('\nper-lab mean occupancy (mouse-level means):')
    print(M.groupby('pathlab')[OCC].mean().round(4).to_string())

    # --- 4. cross-camera agreement, with the mirrored pairing checked, not assumed
    W = S.pivot_table(index=['mouse', 'date', 'num'], columns='camera', values=OCC)
    print(f'\ncross-camera check on {len(W)} sessions with both cameras:')
    for s in ['still', 'move', 'wheel_turn', 'groom']:
        same = np.mean([stats.pearsonr(W[(f'paw_{p}_{s}_occ', 'left')],
                                       W[(f'paw_{p}_{s}_occ', 'right')])[0] for p in 'rl'])
        cross = np.mean([stats.pearsonr(W[(f'paw_{a}_{s}_occ', 'left')],
                                        W[(f'paw_{b}_{s}_occ', 'right')])[0]
                         for a, b in [('r', 'l'), ('l', 'r')]])
        l, r = W[(f'paw_l_{s}_occ', 'left')], W[(f'paw_r_{s}_occ', 'right')]
        print(f'  {s:11s} same-name r {same:+.3f} | MIRRORED r {cross:+.3f} | '
              f'far-paw bias {(l - r).mean():+.4f} (Wilcoxon p {stats.wilcoxon(l, r).pvalue:.1g})')


if __name__ == '__main__':
    main()
