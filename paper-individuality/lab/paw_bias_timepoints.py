"""
PAW SIDE BIAS AT FOUR TIMEPOINTS -- does an animal keep it across a change of rig?
==================================================================================
Early and Late are on BEHAVIOUR rigs, Pre-rec on the biased-protocol rigs, Proficient on
EPHYS rigs. The timepoints therefore span a change of apparatus, which is what turns this
from a description into a test.

EVERYTHING FROM lightningPose, ONE CAMERA, ONE CONVENTION. An earlier version mixed trackers
-- DLC for the proficient timepoint, LP everywhere else -- and mixed conventions: the
proficient design matrices take the near paw from EACH camera, while training has no right
camera at all, so its two paw columns are both from the left one. Those are different
measurements and their population means even have opposite signs, so comparing them directly
confounds the rig change with a change of method.

Here every timepoint uses the LEFT camera's `paw_r` (near) against `paw_l` (far):
  Early, Late   data/training/states_files/{session_1,last_training}/  -- VERIFIED that the
                states file's l_paw is the left camera's LP paw_r and r_paw is paw_l
                (medians agree to <1 px)
  Pre-rec       the cached biased-protocol LP files
  Proficient    lp_session_metrics.csv, built by lp_extract.py from LP

NO DOWNLOADS. The paw coordinates are already in the design matrices and states files; only
the geometry landmarks (nose, tube, pupil) would need the pose files, and those are not used
here.

SIGN CAVEAT. Positive means the NEAR paw moved more, and the near paw is measured better --
the far paw is both harder to track and, in lightningAction, classified active more often.
So the population offset (~+0.1) is a measurement property, not a claim about handedness; the
two-camera estimate that cancels it put the population at +0.010, CI [-0.037, +0.058]. What
is interpretable here is each mouse's position relative to that offset, and whether it holds
across timepoints.
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

import paper_style as ps
import matplotlib.pyplot as plt
from session_filters import find_csv

CACHE = '/home/ines/Downloads/ONE/alyx.internationalbrainlab.org/'
LP = '_ibl_leftCamera.lightningPose.pqt'
TP = ['Early', 'Late', 'Pre-rec', 'Proficient']
OUT = HERE / 'paw_bias_timepoints.csv'


def li(a, b):
    return (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) and (a + b) > 0 else np.nan


def speed(x, y, ok=None):
    v = np.hypot(np.diff(np.asarray(x, float)), np.diff(np.asarray(y, float)))
    if ok is not None:
        v = v[ok[1:]]
    return float(np.nanmean(v)) if len(v) else np.nan


def from_states(f):
    """training: l_paw = left-camera paw_r (near), r_paw = paw_l (far). Verified."""
    d = pd.read_parquet(f, columns=['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y'])
    return li(speed(d.l_paw_x, d.l_paw_y), speed(d.r_paw_x, d.r_paw_y))


def from_lp(f):
    d = pd.read_parquet(f, columns=['paw_r_x', 'paw_r_y', 'paw_r_likelihood',
                                    'paw_l_x', 'paw_l_y', 'paw_l_likelihood'])
    s = {}
    for paw in ['paw_r', 'paw_l']:
        ok = d[f'{paw}_likelihood'].to_numpy() > 0.9
        s[paw] = speed(d[f'{paw}_x'], d[f'{paw}_y'], ok)
    return li(s['paw_r'], s['paw_l'])


def collect():
    rows = []
    for tp, pat in [('Early', str(ROOT / 'data/training/states_files/session_1/*')),
                    ('Late', str(ROOT / 'data/training/states_files/last_training/*'))]:
        for f in glob.glob(pat):
            p = os.path.basename(f).split('_states_file_')
            if len(p) != 2:
                continue
            try:
                rows.append(dict(tp=tp, session=p[1][:36], mouse=p[1][37:], li=from_states(f)))
            except Exception:
                pass

    # Pre-rec: the biased-protocol LP files already on disk
    from one.api import ONE
    one = ONE(mode='local')
    c = pd.read_csv(find_csv(), header=1)
    c = c[c['Used in paper'].astype(str).str.lower() != 'filtered out']
    for r in c[c.task_protocol == 'biased'].itertuples():
        p = one.eid2path(r.eid)
        if p is None:
            continue
        q = str(p).split('/')
        f = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf', LP)
        if not os.path.exists(f):
            continue
        try:
            rows.append(dict(tp='Pre-rec', session=r.eid, mouse=r.mouse_name, li=from_lp(f)))
        except Exception:
            pass

    # Proficient: already summarised from LP by lp_extract.py
    m = pd.read_csv(HERE / 'lp_session_metrics.csv')
    m = m[m.timepoint == 'Proficient']
    for r in m.itertuples():
        rows.append(dict(tp='Proficient', session=r.session, mouse=r.mouse, li=r.li_paw))
    return pd.DataFrame(rows).dropna()


def main():
    R = collect()
    R.to_csv(OUT, index=False)
    print(R.groupby('tp').agg(sessions=('session', 'nunique'), mice=('mouse', 'nunique'),
                              mean=('li', 'mean')).reindex(TP).round(3).to_string())
    PM = R.groupby(['tp', 'mouse'])['li'].mean().unstack(0).reindex(columns=TP)

    ps.use('poster')
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.4),
                            gridspec_kw={'width_ratios': [1.7, 1, 1, 1]})
    ax = axs[0]
    full = PM.dropna()
    x = np.arange(len(TP))
    for _, row in full.iterrows():
        v = row[TP].to_numpy(float)
        ax.plot(x, v, color=('#B3472A' if v[0] > 0 else '#2F6B8F'), alpha=.45, lw=1.1,
                marker='o', ms=3.5)
    ax.plot(x, PM[TP].mean().to_numpy(), color='k', lw=2.6, marker='o', ms=7, zorder=5,
            label='population mean (all mice)')
    ax.axhline(0, color='0.5', ls='--', lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(['Early\n(1st session)', 'Late\n(end training)',
                        'Pre-rec\n(biased)', 'Proficient\n(ephys)'],
                       fontsize=plt.rcParams['font.size'] * .55)
    ax.set_ylabel('paw bias   (near − far) / (near + far)')
    ax.set_title(f'Each line is a mouse ({len(full)} with all four)\n'
                 'lightningPose, left camera, one convention',
                 fontsize=plt.rcParams['font.size'] * .66)
    ax.legend(frameon=False, fontsize=plt.rcParams['font.size'] * .5)

    for ax, (a, b) in zip(axs[1:], [('Early', 'Late'), ('Late', 'Pre-rec'),
                                    ('Pre-rec', 'Proficient')]):
        d = PM[[a, b]].dropna()
        r = stats.pearsonr(d[a], d[b]); rs = stats.spearmanr(d[a], d[b])
        same = int(np.sign(d[a]).eq(np.sign(d[b])).sum())
        ax.axhline(0, color='0.85', lw=.8, zorder=0); ax.axvline(0, color='0.85', lw=.8, zorder=0)
        ax.scatter(d[a], d[b], s=28, color='#4A4C47', alpha=.75, linewidths=0)
        lo, hi = d[a].min(), d[a].max()
        m_, c_ = np.polyfit(d[a], d[b], 1)
        ax.plot([lo, hi], [m_ * lo + c_, m_ * hi + c_], color='#B3472A', lw=1.5)
        ax.set_xlabel(a); ax.set_ylabel(b)
        ax.set_title(f'{a} → {b}   n = {len(d)}\nr = {r[0]:+.3f} (p = {r[1]:.3f})  '
                     f'rho = {rs[0]:+.2f}\nsame side: {same}/{len(d)}',
                     fontsize=plt.rcParams['font.size'] * .58)
    fig.suptitle('Paw bias across four timepoints — rigs change between Pre-rec and Proficient',
                 fontsize=plt.rcParams['font.size'] * .74, y=1.02)
    fig.tight_layout()
    ps.savefig(fig, 'paw_bias_four_timepoints', svg=True)
    plt.show()

    print('\nconsecutive and across-rig agreement')
    for a, b in [('Early', 'Late'), ('Late', 'Pre-rec'), ('Pre-rec', 'Proficient'),
                 ('Early', 'Proficient')]:
        d = PM[[a, b]].dropna()
        if len(d) < 8:
            print(f'  {a} -> {b}: only {len(d)} mice'); continue
        r = stats.pearsonr(d[a], d[b]); rs = stats.spearmanr(d[a], d[b])
        s = int(np.sign(d[a]).eq(np.sign(d[b])).sum())
        print(f'  {a:10s} -> {b:11s} n={len(d):3d}  r={r[0]:+.3f} (p={r[1]:.3f})  '
              f'rho={rs[0]:+.3f}  same side {s}/{len(d)} ({s/len(d):.0%})')


if __name__ == '__main__':
    main()
