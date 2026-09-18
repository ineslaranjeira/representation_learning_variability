"""
WHAT IS MOUSE IDENTITY MADE OF?  COMMONALITY ANALYSIS ON PREDICTING THE MOUSE
=============================================================================
Predict WHICH MOUSE a session came from, out of several predictor blocks, and ask what each
block contributes that the others do not.

  B  behaviour   the 30 PCs of the 360 syllable features -- exactly what the LDA consumes
  G  rig geometry the 12 measured camera/tracking terms (nose and tube position, animal-to-
                 tube distance, tracker confidence): nuisance by construction
  L  lab         one-hot lab

TARGET. Mouse identity one-hot encoded, so "R^2" is the share of the identity indicators'
variance a block predicts. It is the regression form of the same question the LDA answers
with accuracy, and unlike accuracy it decomposes additively, which is what makes ΔR^2 mean
something. Accuracy is reported next to it so the numbers stay legible.

ΔR^2 AND THE SHARED PARTS. Fitting all seven subsets gives the full commonality
decomposition: each block's UNIQUE part (what it adds to the other two) plus every SHARED
part (what two or three blocks explain equally well and cannot be credited to either). The
shared parts are the interesting ones here -- behaviour and geometry overlap precisely to the
extent that our behavioural features are reading the camera.

CROSS-VALIDATION IS LEAVE-ONE-SESSION-OUT, and it has to be: a mouse held out entirely can
never be predicted, since its class is not in the training set. That means mouse-stable
nuisance (each mouse sits on one rig, so geometry barely moves across its sessions) is free
to identify the mouse. That is not leakage, it IS the finding -- it measures how much of our
identification would survive if the animal were moved to another rig, which is none of it.

IN-SAMPLE R^2 IS REPORTED TOO, and it is much higher: with ~200 sessions, 51 predictors and
46 indicator columns, everything fits in sample. Only the CV column should be read.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / '4_mice'), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import functions

SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
CORE = ['nose_x', 'nose_y', 'tube_x', 'tube_y', 'pawNear_x', 'pawNear_y', 'pawFar_x',
        'pawFar_y', 'nose_to_tube_px', 'conf_nose', 'conf_pawNear', 'conf_pawFar']


def r2_and_acc(X, Y, loo=True):
    """Leave-one-session-out R^2 pooled over the indicator columns, plus argmax accuracy."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    Y = np.asarray(Y, float)
    pred = np.zeros_like(Y)
    if loo:
        for t in range(len(X)):
            tr = np.setdiff1d(np.arange(len(X)), t)
            B, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
            pred[t] = X[t] @ B
    else:
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ B
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot, float((pred.argmax(1) == Y.argmax(1)).mean())


def main():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = pd.Series(ss.index.map(dd[['mouse_name', 'session']].drop_duplicates()
                                   .set_index('session')['mouse_name']), index=ss.index)
    lab = functions.lab_labels(ss.index, mouse_names=mouse, verbose=False)
    geo = pd.read_csv(HERE / 'rig_geometry.csv', index_col=0)

    k = geo[CORE].dropna().index.intersection(ss.index)
    cnt = mouse[k].value_counts()
    k = [s for s in k if cnt[mouse[s]] >= 3]          # a class needs sessions on both sides
    X_all, mo, lb, ge = ss.loc[k], mouse[k], lab[k], geo.loc[k, CORE]
    print(f'{len(k)} sessions, {mo.nunique()} mice, {lb.nunique()} labs\n')

    Y = pd.get_dummies(mo).astype(float).to_numpy()
    blocks = {
        'B behaviour': StandardScaler().fit_transform(
            PCA(min(np.asarray(X_all).shape)).fit_transform(np.asarray(X_all))[:, :30]),
        'G geometry': StandardScaler().fit_transform(np.asarray(ge, float)),
        'L lab': pd.get_dummies(lb, drop_first=True).astype(float).to_numpy(),
    }
    names = list(blocks)

    subsets = {}
    for r in range(1, 4):
        for c in combinations(names, r):
            Xc = np.column_stack([blocks[b] for b in c])
            cv, acc = r2_and_acc(Xc, Y)
            ins, _ = r2_and_acc(Xc, Y, loo=False)
            subsets[c] = dict(cv=cv, acc=acc, ins=ins, k=Xc.shape[1])
    full = subsets[tuple(names)]

    print('MODEL                          k    R2 in-sample   R2 cross-val   accuracy')
    for c, v in subsets.items():
        print(f"  {' + '.join(c):28s} {v['k']:3d}   {v['ins']:+.3f}         {v['cv']:+.3f}"
              f"         {v['acc']:.3f}")
    print(f"\nchance accuracy = 1/{mo.nunique()} = {1 / mo.nunique():.3f}")

    print('\nUNIQUE CONTRIBUTION (delta R2 when the block is added to the other two):')
    for b in names:
        others = tuple(n for n in names if n != b)
        print(f'  {b:14s} {full["cv"] - subsets[others]["cv"]:+.3f} R2   '
              f'({full["acc"] - subsets[others]["acc"]:+.3f} accuracy)')

    print('\nCOMMONALITY DECOMPOSITION of the cross-validated R2:')
    # Inclusion-exclusion for THREE predictor sets. The two-set formula
    # (R(A)+R(B)-R(AB)) is wrong here and would not sum back to the total.
    r = {c: v['cv'] for c, v in subsets.items()}
    B, G, L = names
    tot = r[(B, G, L)]
    U = {B: tot - r[(G, L)], G: tot - r[(B, L)], L: tot - r[(B, G)]}
    C = {f'{B} & {G}': tot - r[(L,)] - U[B] - U[G],
         f'{B} & {L}': tot - r[(G,)] - U[B] - U[L],
         f'{G} & {L}': tot - r[(B,)] - U[G] - U[L]}
    C['all three'] = tot - sum(U.values()) - sum(C.values())
    for b in names:
        print(f'  unique {b:16s} {U[b]:+.3f}')
    for kk, v in C.items():
        print(f'  shared {kk:16s} {v:+.3f}')
    print(f'  {"TOTAL":23s} {tot:+.3f}   (components sum to {sum(U.values()) + sum(C.values()):+.3f})')
    print('  NEGATIVE components are normal with cross-validated R2: a block can make the')
    print('  out-of-sample fit worse, and suppression can push a shared part below zero.')

    print('\nBEHAVIOUR WITH THE GEOMETRY TAKEN OUT (the quantity the paper needs):')
    D = np.column_stack([np.ones(len(ge)), StandardScaler().fit_transform(np.asarray(ge, float))])
    Bc, *_ = np.linalg.lstsq(D, np.asarray(X_all, float), rcond=None)
    resid = np.asarray(X_all, float) - D @ Bc
    Br = StandardScaler().fit_transform(PCA(min(resid.shape)).fit_transform(resid)[:, :30])
    cv, acc = r2_and_acc(Br, Y)
    print(f'  residualised behaviour alone: R2 {cv:+.3f}  accuracy {acc:.3f}  '
          f'(raw behaviour alone: R2 {r[(B,)]:+.3f}  accuracy {subsets[(B,)]["acc"]:.3f})')


if __name__ == '__main__':
    main()
