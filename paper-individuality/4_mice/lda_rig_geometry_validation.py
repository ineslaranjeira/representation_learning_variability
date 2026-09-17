"""
IS LD1 BEHAVIOUR, OR IS IT RIG GEOMETRY?
========================================
The worry this answers: the design matrix is built from pixel measurements, and
pixels depend on where the camera sits, how far it is zoomed, how the mouse is
positioned, and which frame rate the camera runs at. Each mouse is typically
recorded on ONE rig, so any of those would be mouse-stable -- exactly the
signature the LDA is asked to find. An LDA that separates mice by rig would look
identical, from the inside, to one that separates them by behaviour.

Everything below is measured from the LEFT CAMERA ALONE. That matters: the
design matrix takes l_paw from the left camera and r_paw from the right, so its
laterality index carries the camera-gain term gamma (paw_bias section 5, gamma =
+0.089, the whole of the apparent population left bias). Both paws from ONE
camera share a frame rate, an exposure, a smoother and a lens, so gamma cancels
exactly, with no correction constant.

MEASURES (leftCamera.dlc.pqt, likelihood > 0.9)
  zoom          tube_len_px   lick tube is rigid rig hardware of FIXED physical
                              size, so its apparent length is px/mm directly
                pupil_diam_px bilaterally symmetric, viewed near face-on
  positioning   nose_x/y, tube_top_x/y, nose_to_tube_px, paw medians
  laterality    LI_within = LI(paw_r [LEFT forepaw, near], paw_l [RIGHT forepaw,
                              far]), 0.5-8 Hz velocity band power. gamma-free.
                              Still carries near/far foreshortening f, which is
                              itself a positioning term -- so it is reported as a
                              nuisance measure too, not only as behaviour.
  timing        fps from leftCamera.times.npy

NOTE ON TRACKER. The design matrices were built from lightningPose; the landmarks
needed here exist only in the DLC files (LP carries paws and pupil, no tube or
nose). For the geometry terms that is irrelevant. For LI_within it makes the
estimate INDEPENDENT of the tracker the LDA saw, which is a feature here.

THE TWO QUESTIONS
  1. Is each nuisance term mouse-stable? (ICC, sessions nested in mice.) A term
     that is not mouse-stable cannot masquerade as individuality, whatever it
     correlates with.
  2. Does it predict LD1 across mice? Mouse-stable AND predictive = contamination.
     Mouse-stable but not predictive = LD1 is clean on that axis.
"""
import os
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import signal, stats

warnings.filterwarnings('ignore')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = '/Users/ineslaranjeira/Downloads/FlatIron'
LIK = 0.9
BAND = (0.5, 8.0)
MIN_SEG = 512
OUT = HERE / 'lda_rig_geometry_validation.csv'

KP = ['nose_tip', 'tube_top', 'tube_bottom', 'paw_l', 'paw_r',
      'pupil_top_r', 'pupil_bottom_r', 'pupil_left_r', 'pupil_right_r']
COLS = [f'{k}_{s}' for k in KP for s in ('x', 'y', 'likelihood')]


def conf(d, kp):
    c = f'{kp}_likelihood'
    m = np.isfinite(d[f'{kp}_x']) & np.isfinite(d[f'{kp}_y'])
    if c in d:
        m &= d[c].to_numpy() > LIK
    return m


def med_dist(d, a, b):
    m = conf(d, a) & conf(d, b)
    if m.sum() < 500:
        return np.nan
    return float(np.median(np.hypot(d[f'{a}_x'][m] - d[f'{b}_x'][m],
                                    d[f'{a}_y'][m] - d[f'{b}_y'][m])))


def band_power(v, fs):
    ok = np.isfinite(v)
    if ok.sum() < MIN_SEG:
        return np.nan
    edges = np.flatnonzero(np.diff(ok.astype(np.int8)))
    bounds = np.r_[0, edges + 1, len(ok)]
    psds, w = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not ok[a] or (b - a) < MIN_SEG:
            continue
        f, p = signal.welch(v[a:b], fs=fs, nperseg=min(512, b - a))
        m = (f >= BAND[0]) & (f <= BAND[1])
        psds.append(np.trapz(p[m], f[m]))
        w.append(b - a)
    return float(np.average(psds, weights=w)) if psds else np.nan


def session_geometry(alf):
    f = os.path.join(alf, '_ibl_leftCamera.dlc.pqt')
    if not os.path.exists(f):
        return None
    have = pd.read_parquet(f, columns=None).columns
    d = pd.read_parquet(f, columns=[c for c in COLS if c in have])

    tpath = os.path.join(alf, '_ibl_leftCamera.times.npy')
    fps = np.nan
    if os.path.exists(tpath):
        t = np.load(tpath)
        t = t[np.isfinite(t)]
        if len(t) > 100:
            fps = float(1.0 / np.median(np.diff(t)))
    fs = fps if np.isfinite(fps) else 60.0

    rec = dict(fps=fps, n_frames=len(d))
    rec['tube_len_px'] = med_dist(d, 'tube_top', 'tube_bottom')
    rec['nose_to_tube_px'] = med_dist(d, 'nose_tip', 'tube_top')
    pv = med_dist(d, 'pupil_top_r', 'pupil_bottom_r')
    ph = med_dist(d, 'pupil_left_r', 'pupil_right_r')
    rec['pupil_diam_px'] = np.nanmean([pv, ph])
    for kp, short in [('nose_tip', 'nose'), ('tube_top', 'tube'),
                      ('paw_r', 'pawNear'), ('paw_l', 'pawFar')]:
        m = conf(d, kp)
        rec[f'{short}_x'] = float(np.median(d[f'{kp}_x'][m])) if m.sum() > 500 else np.nan
        rec[f'{short}_y'] = float(np.median(d[f'{kp}_y'][m])) if m.sum() > 500 else np.nan

    # gamma-free within-camera laterality. paw_r = LEFT forepaw (near),
    # paw_l = RIGHT forepaw (far), per paw_bias/qc_paw_identity.py.
    pw = {}
    for kp, name in [('paw_r', 'left'), ('paw_l', 'right')]:
        m = conf(d, kp)
        tot = 0.0
        for ax in ('x', 'y'):
            v = d[f'{kp}_{ax}'].to_numpy(float).copy()
            v[~m] = np.nan
            bp = band_power(np.diff(v, prepend=np.nan) * fs, fs)
            if not np.isfinite(bp):
                tot = np.nan
                break
            tot += bp
        pw[name] = tot
    L, R = pw.get('left', np.nan), pw.get('right', np.nan)
    rec['LI_within'] = (L - R) / (L + R) if np.isfinite(L) and np.isfinite(R) else np.nan
    rec['P_within_total'] = L + R
    return rec


def icc1(values, groups):
    """ICC(1): between-group variance share, sessions nested in mice."""
    df = pd.DataFrame(dict(v=values, g=groups)).dropna()
    if df.g.nunique() < 3 or len(df) < 6:
        return np.nan
    k = df.groupby('g')['v'].size()
    gm = df.v.mean()
    msb = (k * (df.groupby('g')['v'].mean() - gm) ** 2).sum() / (df.g.nunique() - 1)
    msw = df.groupby('g')['v'].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / \
        max(len(df) - df.g.nunique(), 1)
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (df.g.nunique() - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw)) if (msb + (k0 - 1) * msw) else np.nan


def main():
    import lda_allsessions_heldout as L
    clustered = L.main()
    lda = clustered.set_index('session')[[0, 1, 2]].rename(
        columns={0: 'LD1', 1: 'LD2', 2: 'LD3'})
    lda['mouse_name'] = clustered.set_index('session')['mouse_name'].values

    from one.api import ONE
    one = ONE(base_url='https://openalyx.internationalbrainlab.org',
              password='international', silent=True)

    print("\n" + "=" * 78)
    print("PER-SESSION RIG GEOMETRY FROM leftCamera.dlc")
    print("=" * 78)
    rows = []
    for i, (eid, r) in enumerate(lda.iterrows(), 1):
        p = one.eid2path(eid)
        if p is None:
            continue
        q = p.parts
        alf = os.path.join(CACHE, q[-5], 'Subjects', q[-3], q[-2], q[-1], 'alf')
        try:
            g = session_geometry(alf)
        except Exception as e:
            print(f'  !! {eid[:8]}: {type(e).__name__}: {e}')
            continue
        if g is None:
            continue
        g.update(session=eid, mouse_name=r.mouse_name, lab=q[-5],
                 LD1=r.LD1, LD2=r.LD2, LD3=r.LD3)
        rows.append(g)
        if i % 50 == 0:
            print(f'  ... {i}/{len(lda)} scanned, {len(rows)} measured')
    df = pd.DataFrame(rows).set_index('session')
    df.to_csv(OUT)
    print(f"✓ {len(df)} sessions, {df.mouse_name.nunique()} mice, {df.lab.nunique()} labs")
    print(f"saved {OUT}")

    NUIS = ['tube_len_px', 'pupil_diam_px', 'nose_to_tube_px', 'nose_x', 'nose_y',
            'tube_x', 'tube_y', 'pawNear_x', 'pawNear_y', 'pawFar_x', 'pawFar_y',
            'fps', 'LI_within', 'P_within_total']

    print("\n" + "=" * 78)
    print("Q1  IS EACH TERM MOUSE-STABLE?   ICC(1), sessions nested in mice")
    print("    (a term that is not mouse-stable cannot masquerade as individuality)")
    print("=" * 78)
    icc = {c: icc1(df[c].values, df.mouse_name.values) for c in NUIS}
    for c, v in sorted(icc.items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -9)):
        print(f"  {c:18s} ICC = {v:+.3f}   n = {df[c].notna().sum()}")

    mouse = df.groupby('mouse_name').agg({**{c: 'mean' for c in NUIS + ['LD1', 'LD2', 'LD3']},
                                          'lab': 'first'})
    print("\n" + "=" * 78)
    print(f"Q2  DOES IT PREDICT LD1?   mouse level, n = {len(mouse)}")
    print("=" * 78)
    res = []
    for c in NUIS:
        s = mouse[[c, 'LD1']].dropna()
        if len(s) < 10:
            continue
        r, p = stats.pearsonr(s[c], s.LD1)
        rho, prho = stats.spearmanr(s[c], s.LD1)
        z = np.arctanh(r); se = 1 / np.sqrt(len(s) - 3)
        res.append(dict(term=c, n=len(s), r=r, lo=np.tanh(z - 1.96 * se),
                        hi=np.tanh(z + 1.96 * se), p=p, rho=rho, p_rho=prho,
                        ICC=icc[c]))
    res = pd.DataFrame(res).sort_values('p')
    res['p_holm'] = np.minimum.accumulate(
        (res.p.values * (len(res) - np.arange(len(res))))[::-1])[::-1].clip(max=1.0)
    print(res.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))

    print("\n" + "=" * 78)
    print("HOW MUCH OF LD1 DO ALL THE GEOMETRY TERMS TOGETHER EXPLAIN?")
    print("=" * 78)
    geo = [c for c in NUIS if c not in ('LI_within', 'P_within_total')]
    s = mouse[geo + ['LD1']].dropna()
    X = np.column_stack([np.ones(len(s)), stats.zscore(s[geo].values, axis=0)])
    beta, *_ = np.linalg.lstsq(X, s.LD1.values, rcond=None)
    r2 = 1 - ((s.LD1.values - X @ beta).var() / s.LD1.values.var())
    k = X.shape[1] - 1
    r2adj = 1 - (1 - r2) * (len(s) - 1) / (len(s) - k - 1)
    F = (r2 / k) / ((1 - r2) / (len(s) - k - 1))
    pF = 1 - stats.f.cdf(F, k, len(s) - k - 1)
    print(f"  {len(geo)} geometry terms, n = {len(s)} mice")
    print(f"  R2 = {r2:.3f}   adjusted R2 = {r2adj:.3f}   F({k},{len(s)-k-1}) = {F:.2f}, p = {pF:.4f}")

    labs = mouse.dropna(subset=['LD1']).groupby('lab')['LD1'].apply(list)
    labs = [v for v in labs if len(v) >= 3]
    if len(labs) >= 3:
        F2, p2 = stats.f_oneway(*labs)
        print(f"\n  LD1 by LAB (mouse as unit, {len(labs)} labs): F = {F2:.2f}, p = {p2:.3f}")

    print("\n" + "=" * 78)
    print("CROSS-CHECK  gamma-free vs cross-camera laterality")
    print("=" * 78)
    try:
        prev = pd.read_csv(HERE / 'lda1_vs_raw_paw_laterality.csv', index_col=0)
        j = df.join(prev[['LI']], how='inner').dropna(subset=['LI', 'LI_within'])
        mj = j.groupby('mouse_name')[['LI', 'LI_within', 'LD1']].mean()
        r1, p1 = stats.pearsonr(mj.LI, mj.LI_within)
        print(f"  cross-camera LI vs within-camera LI : r = {r1:+.3f}, p = {p1:.2g}, n = {len(mj)}")
        print(f"  mean cross-camera LI = {mj.LI.mean():+.3f}   "
              f"mean within-camera LI = {mj.LI_within.mean():+.3f}")
        r2_, p2_ = stats.pearsonr(mj.LD1, mj.LI_within)
        z = np.arctanh(r2_); se = 1 / np.sqrt(len(mj) - 3)
        print(f"  LD1 vs WITHIN-camera LI (gamma-free): r = {r2_:+.3f} "
              f"95% CI [{np.tanh(z-1.96*se):+.3f}, {np.tanh(z+1.96*se):+.3f}], p = {p2_:.2g}")
    except FileNotFoundError:
        print("  (run lda1_vs_raw_paw_laterality.py first for the cross-camera comparison)")

    return df, mouse, res


if __name__ == '__main__':
    main()
