"""
DOES CAMERA ZOOM EXPLAIN THE LAB EFFECT IN WHISKING AND LICKING?
================================================================
Whisk and lick carry most of the lab signal in the syllable features (0.208 and 0.345 of
their variance). Both are video measurements, so the obvious suspect is magnification: a
camera closer to the animal makes a whisker excursion bigger in pixels and may change how
reliably a lick is detected.

PICKING A ZOOM RULER. The first attempt divided by `tube_len_px`, which was wrong: the lick
tube is NOT fixed hardware -- it can be cut longer or shorter between rigs -- so its apparent
length confounds magnification with how someone trimmed the spout. Measured on this dataset,
that shows up immediately in the variance structure and in whether the ruler predicts anything:

    ruler               lab    mouse  session   correlation with paw speed
    tube_len_px        0.420   0.018   0.562    r = +0.072  (p = 0.31)   <- mostly noise
    nose_to_pupil_px   0.885   0.090   0.025    r = +0.278  (p = 0.0001) <- a rig property
    pupil_diam_px      0.260   0.212   0.528    r = +0.197               <- moves with arousal

Nose tip to pupil centre is a distance ON THE ANIMAL at roughly the depth the paws and
whiskers move in, it is stable within a rig (2.5% session variance), and it predicts pixel
speed. It is the ruler used here.

THE TRAP THIS SCRIPT IS BUILT TO AVOID. A good zoom ruler is 88.5% lab-structured, precisely
because zoom is a rig property -- so "regressing out zoom" is very nearly "regressing out
lab", and will remove lab variance whether or not zoom causes anything. Three checks that do
not have that problem:

  1. does zoom predict the feature at all?  If not, it cannot explain anything.
  2. lab effect WITHIN a zoom group -- zoom held roughly fixed, lab varying. Usable because
     6 of 9 labs span more than one zoom tertile.
  3. regress out only the WITHIN-LAB component of zoom, which is orthogonal to lab by
     construction, and compare against regressing out zoom as measured.

Plus a leave-one-lab-out sweep, to check no single lab is carrying the effect -- the failure
mode that turned out to explain the tube-corrected vigor result.
"""
import sys
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

SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
RAW = str(ROOT / 'data' / '10_bin_raw_10-09-2026')
ZOOM = 'nose_to_pupil_px'
N_GROUPS = 3


def load():
    syl, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = pd.Series(syl.index.map(dd[['mouse_name', 'session']].drop_duplicates()
                                    .set_index('session')['mouse_name']), index=syl.index)
    raw, _ = functions.build_design_matrix(RAW, verbose=False)
    geo = pd.read_csv(HERE / 'rig_geometry.csv', index_col=0)
    z = geo[ZOOM].dropna()
    order = [s for s in syl.index if s in set(raw.index) and s in z.index]
    w = np.arange(syl.shape[1])
    ch = np.arange(raw.shape[1]) // 40
    blocks = {'syllable WHISK': syl.loc[order].iloc[:, (w % 9) == 7],
              'syllable LICK': syl.loc[order].iloc[:, (w % 9) == 8],
              'raw whisker_me': raw.loc[order].iloc[:, ch == 1],
              'raw Lick count': raw.loc[order].iloc[:, ch == 0]}
    lab = functions.lab_labels(pd.Index(order), mouse_names=mouse[order], verbose=False)
    return blocks, lab, mouse[order], z[order], geo


def lab_share(D, lb, ms, n_perm=2000):
    vc, _ = vp.nested_variance_components(D, lb, ms)
    s = vc.clip(lower=0).sum()
    s = s / s.sum()
    _, _, pv = vp.permutation_null_eta2(D, lb, ms, n_perm=n_perm, seed=0)
    return s['sigma2_lab'], pv


def main():
    B, lab, ms, z, geo = load()
    print(f'{len(z)} sessions, {ms.nunique()} mice, {lab.nunique()} labs; zoom = {ZOOM}\n')

    print('0. THE RULERS THEMSELVES')
    for c in ['tube_len_px', ZOOM, 'pupil_diam_px']:
        v = geo[c].dropna()
        idx = [s for s in v.index if s in ms.index]
        lb2 = functions.lab_labels(pd.Index(idx), mouse_names=ms[idx], verbose=False)
        vc, _ = vp.nested_variance_components(v[idx].to_frame(), lb2, ms[idx])
        s = vc.clip(lower=0).iloc[0]; s = s / s.sum()
        print(f'  {c:18s} n={len(idx):3d}  lab {s["sigma2_lab"]:.3f}  mouse {s["sigma2_mouse"]:.3f}'
              f'  session {s["sigma2_session"]:.3f}')

    print('\n1. DOES ZOOM CLUSTER CUT ACROSS LABS?')
    grp = pd.qcut(z, N_GROUPS, labels=['far (small px)', 'mid', 'near (large px)'])
    ct = pd.crosstab(grp, lab)
    print(ct.to_string())
    print(f'  labs spanning >1 zoom group: {int((ct > 0).sum(0).gt(1).sum())} of {ct.shape[1]}')

    print('\n2. DOES ZOOM PREDICT THE FEATURE AT ALL?')
    for nm, D in B.items():
        m = D.mean(axis=1)
        r = stats.pearsonr(m, z)
        pm = pd.DataFrame({'m': m, 'z': z, 'mouse': ms}).groupby('mouse').mean()
        print(f'  {nm:16s} r = {r[0]:+.3f} (p={r[1]:.2g}) per session | '
              f'{stats.pearsonr(pm["m"], pm["z"])[0]:+.3f} per mouse')

    print('\n3. LAB EFFECT WITHIN EACH ZOOM GROUP')
    print(f'  {"zoom group":18s} {"n":>4s} {"labs":>5s} ' + ' '.join(f'{k:>20s}' for k in B))
    for g in ct.index:
        idx = [o for o in z.index if grp[o] == g]
        nl = lab[idx].nunique()
        line = [f'{lab_share(D.loc[idx], lab[idx], ms[idx])[0]:.3f} '
                f'(p={lab_share(D.loc[idx], lab[idx], ms[idx])[1]:.3f})' if nl >= 3 else '--'
                for D in B.values()]
        print(f'  {str(g):18s} {len(idx):4d} {nl:5d} ' + ' '.join(f'{v:>20s}' for v in line))

    print('\n4. REGRESS OUT ZOOM -- as measured, versus its WITHIN-LAB part only')
    zc = z - z.groupby(lab.values).transform('mean')
    print(f'  {"feature":16s} {"as-is":>8s} {"minus zoom":>12s} {"minus within-lab zoom":>23s}')
    for nm, D in B.items():
        out = [lab_share(D, lab, ms)[0]]
        for zv in (z, zc):
            X = np.column_stack([np.ones(len(zv)), (zv - zv.mean()) / zv.std()])
            Bc, *_ = np.linalg.lstsq(X, np.asarray(D, float), rcond=None)
            R = pd.DataFrame(np.asarray(D, float) - X @ Bc, index=D.index, columns=D.columns)
            out.append(lab_share(R, lab, ms)[0])
        print(f'  {nm:16s} {out[0]:8.3f} {out[1]:12.3f} {out[2]:23.3f}')

    print('\n5. LEAVE-ONE-LAB-OUT -- is any single lab carrying it?')
    print(f'  {"dropped":20s} ' + ' '.join(f'{k:>20s}' for k in B))
    for drop in ['(none)'] + sorted(lab.unique()):
        idx = list(z.index) if drop == '(none)' else [s for s in z.index if lab[s] != drop]
        line = []
        for D in B.values():
            l, pv = lab_share(D.loc[idx], lab[idx], ms[idx], n_perm=1500)
            line.append(f'{l:.3f} (p={pv:.3f})')
        print(f'  {drop:20s} ' + ' '.join(f'{v:>20s}' for v in line))


if __name__ == '__main__':
    main()
