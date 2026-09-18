"""
IS THERE A POPULATION PAW BIAS?  BOTH CAMERAS, MEASUREMENT BIAS CANCELLED
=========================================================================
lightningAction gives a five-way action state per paw per frame, from each camera. Two facts
make a naive laterality index untrustworthy, and together they also give the fix.

1. THE PAW NAMING IS MIRRORED BETWEEN THE CAMERA FILES. left-camera `paw_r` and
   right-camera `paw_r` are DIFFERENT physical paws (measured: cross-pairing correlates
   r = 0.90-0.96 across cameras, same-name pairing only 0.53-0.72). Following the
   convention in lda_rig_geometry_validation.py, for the LEFT camera `paw_r` is the
   animal's LEFT forepaw -- the NEAR one -- and `paw_l` is the RIGHT forepaw, far away.
   By mirror symmetry the RIGHT camera has it the other way round. VERIFIED rather than
   assumed: on the left camera `paw_r` has higher DLC likelihood (0.996 vs 0.938), more
   usable frames (99.2% vs 90.3%), sits lower in the image (median y 626 vs 591) and moves
   faster (0.172 vs 0.151) -- all four the signature of the near paw. lightningAction
   inherits DLC's naming: LA `paw_r` tracks DLC `paw_r` speed at +0.213 against +0.153 for
   the cross pairing.

2. THE FAR PAW IS MEASURED DIFFERENTLY -- AND IN THE DIRECTION YOU WOULD NOT GUESS. It is
   classified as ACTIVE MORE OFTEN, not less, even though it genuinely moves less. Measured
   on the left camera: near paw active 0.2529, far paw 0.2654, while DLC speed for the same
   paws runs the other way (0.172 near, 0.151 far). The cause is tracking noise, not motion
   -- 10% of far-paw frames fall below likelihood 0.9 against 0.8% for the near paw, and the
   jitter reads as movement. Any index built from ONE camera therefore mixes real laterality
   with a near/far term that favours whichever paw is WORSE tracked.

THE FIX, which is why both cameras matter:

   LI_near  = (left paw seen by the LEFT camera) vs (right paw seen by the RIGHT camera)
              -- each paw measured from its OWN near view, so the near/far term enters both
              sides identically and cancels. This is the honest laterality estimate.
   LI_left  = both paws from the left camera   \\  each carries the near/far term with
   LI_right = both paws from the right camera  /   OPPOSITE sign, so their disagreement
                                                   measures the artifact directly.

If a population bias is real it appears in LI_near and has the SAME sign in LI_left and
LI_right. If it is the camera, LI_left and LI_right are biased in opposite directions and
LI_near sits at zero.

UNIT OF ANALYSIS IS THE MOUSE. Sessions within an animal are not independent, so the
population test is a one-sample test over mouse means, with a bootstrap over mice.
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
import paper_style as ps
import matplotlib.pyplot as plt

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
SYLLABLES = str(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026')
STATES = ['background', 'still', 'move', 'wheel_turn', 'groom']
ACTIVE = ['move', 'wheel_turn']          # "using the paw", as against still/background/groom
OUT_CSV = HERE / 'paw_asymmetry.csv'

# physical paw -> (column name) in each camera's file. NEAR paw first in each pair.
PHYS = {'left':  {'left': 'paw_r', 'right': 'paw_l'},    # left camera: paw_r = animal's LEFT
        'right': {'left': 'paw_l', 'right': 'paw_r'}}    # right camera: mirrored


def occupancies(d, cam):
    f = os.path.join(d, f'pawstates_argmax_{cam}.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    out = {}
    for phys, col in PHYS[cam].items():
        if col not in z:
            return None
        s = z[col]
        c = np.bincount(s.astype(int), minlength=len(STATES)) / max(len(s), 1)
        out[phys] = {st: float(c[i]) for i, st in enumerate(STATES)}
    return out


def li(a, b):
    """(a - b) / (a + b); 0 = symmetric, positive = more left."""
    return (a - b) / (a + b) if (a + b) > 0 else np.nan


def collect():
    ss, dd = functions.build_design_matrix(SYLLABLES, n_paw_states=8, verbose=False)
    mouse = dd[['mouse_name', 'session']].drop_duplicates().set_index('session')['mouse_name']
    from one.api import ONE
    one = ONE(mode='local')
    rows = []
    for eid in ss.index:
        p = one.eid2path(eid)
        if p is None:
            continue
        q = str(p).split('/')
        d = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf', 'lightningaction')
        L, R = occupancies(d, 'left'), occupancies(d, 'right')
        if L is None or R is None:
            continue
        act = lambda o, paw: sum(o[paw][s] for s in ACTIVE)
        rows.append(dict(
            session=eid, mouse=mouse[eid],
            # each paw from its OWN near camera -- the measurement term cancels
            li_near=li(act(L, 'left'), act(R, 'right')),
            # one camera at a time: both carry the near/far term, with opposite signs
            li_left_cam=li(act(L, 'left'), act(L, 'right')),
            li_right_cam=li(act(R, 'left'), act(R, 'right')),
            act_left_near=act(L, 'left'), act_right_near=act(R, 'right'),
            act_left_far=act(R, 'left'), act_right_far=act(L, 'right')))
    return pd.DataFrame(rows)


def main():
    df = collect()
    df.to_csv(OUT_CSV, index=False)
    per_mouse = df.groupby('mouse')[['li_near', 'li_left_cam', 'li_right_cam']].mean()
    n_sess = df.groupby('mouse').size()
    print(f'{len(df)} sessions with BOTH cameras, {len(per_mouse)} mice\n')

    print('POPULATION TEST -- one row per mouse, positive = left paw used more')
    rng = np.random.default_rng(0)
    for c, label in [('li_near', 'LI near (measurement cancelled)'),
                     ('li_left_cam', 'LI from the left camera only'),
                     ('li_right_cam', 'LI from the right camera only')]:
        v = per_mouse[c].dropna().to_numpy()
        boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
        t = stats.ttest_1samp(v, 0)
        w = stats.wilcoxon(v)
        print(f'  {label:34s} mean {v.mean():+.4f}  95% CI [{np.percentile(boot,2.5):+.4f},'
              f' {np.percentile(boot,97.5):+.4f}]  t p={t.pvalue:.2g}  wilcoxon p={w.pvalue:.2g}')

    print(f"\n  near/far measurement term (half the left-right camera disagreement): "
          f"{(per_mouse['li_left_cam'].mean() - per_mouse['li_right_cam'].mean()) / 2:+.4f}")
    print(f"  mice with li_near > 0: {int((per_mouse['li_near'] > 0).sum())} of {len(per_mouse)}")

    # ---------------- figure ----------------
    order = per_mouse['li_near'].sort_values().index
    y = np.arange(len(order))
    sem = (df.groupby('mouse')['li_near'].std() / np.sqrt(n_sess)).reindex(order)
    fig, axs = plt.subplots(1, 2, figsize=(11, max(5, 0.19 * len(order))),
                            gridspec_kw={'width_ratios': [2.1, 1]})

    ax = axs[0]
    ax.axvline(0, color='0.55', lw=1, ls='--', zorder=0)
    vals = per_mouse['li_near'].reindex(order).to_numpy()
    cols = np.where(vals > 0, '#B3472A', '#2F6B8F')
    ax.errorbar(vals, y, xerr=sem.to_numpy(), fmt='none', ecolor='0.72', lw=1, zorder=1)
    ax.scatter(vals, y, s=26, c=cols, zorder=2, linewidths=0)
    m = np.nanmean(vals)
    ax.axvline(m, color='k', lw=1.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=plt.rcParams['font.size'] * 0.42)
    ax.set_ylim(-1, len(order))
    ax.set_xlabel('Paw-use asymmetry   (left − right) / (left + right)')
    ax.set_title(f'Each mouse, both cameras, near view only\npopulation mean {m:+.3f}',
                 fontsize=plt.rcParams['font.size'] * 0.8)
    ax.text(m, len(order) - 0.5, ' population', fontsize=plt.rcParams['font.size'] * 0.6,
            va='top', ha='left')

    ax = axs[1]
    for i, (c, lab, colr) in enumerate([
            ('li_left_cam', 'left camera\nonly', '#8C8C8C'),
            ('li_near', 'both cameras\nnear view', '#B3472A'),
            ('li_right_cam', 'right camera\nonly', '#8C8C8C')]):
        v = per_mouse[c].dropna().to_numpy()
        ax.scatter(np.full(len(v), i) + rng.normal(0, .06, len(v)), v, s=12,
                   color=colr, alpha=.45, linewidths=0)
        ax.plot([i - .25, i + .25], [v.mean()] * 2, color='k', lw=2, zorder=3)
    ax.axhline(0, color='0.55', lw=1, ls='--', zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['left camera\nonly', 'both cameras\nnear view', 'right camera\nonly'],
                       fontsize=plt.rcParams['font.size'] * 0.62)
    ax.set_ylabel('Paw-use asymmetry')
    ax.set_title('Single-camera views carry a near/far\nterm with opposite signs',
                 fontsize=plt.rcParams['font.size'] * 0.8)
    fig.tight_layout()
    ps.savefig(fig, 'paw_asymmetry_lightningaction', svg=True)
    plt.show()
    print(f'\nsaved {OUT_CSV}')


if __name__ == '__main__':
    main()
