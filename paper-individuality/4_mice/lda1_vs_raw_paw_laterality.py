"""
IS LD1 A LEFT/RIGHT PAW AXIS?  (raw paw movement, not syllables)
================================================================
LD1 comes from LDA_analyses_pipeline_ALLSESSIONS.ipynb, whose features are the
occupancy of 8 HMM paw states. Four of those states are themselves lateralised
(segmentation/paw_bias/state_description.txt: state 3 LI=+0.63 and state 5
LI=+0.31 are left-paw; state 4 LI=-0.81 and state 6 LI=-0.41 are right-paw). So
this asks the question in a DIFFERENT REPRESENTATION of the same signal -- raw
forepaw position from the per-session design matrices -- not in an independent
one. Read a positive result as "here is the shape and sign of the relationship",
not as "paw laterality is independently predicted by LD1".

METRIC -- follows segmentation/paw_bias, which validated it:
  * l_paw comes from the LEFT camera, r_paw from the RIGHT; get_speed's
    RESOLUTION divides left-camera pixels by 2, so that is applied here too.
  * velocity = diff(position) * 60 Hz, then Welch power integrated over 0.5-8 Hz,
    x and y summed per paw.
  * LI = (P_left - P_right) / (P_left + P_right);  positive = left paw.
  * band power, NOT speed: paw_bias section 4 shows the speed index is dominated
    by the right paw's larger tracking-noise floor and flips sign under
    noise-matching, while the band-power index is invariant (r = 0.9999).

THE CAVEAT THAT GOVERNS INTERPRETATION. l_paw and r_paw are measured by DIFFERENT
CAMERAS running at different frame rates, and the resulting camera-gain term
gamma = +0.089 is the whole of the apparent population-level left bias
(paw_bias section 5). So the MEAN of LI here is instrumental and means nothing.
gamma is uncorrelated with true laterality (r = 0.035, p = 0.77), i.e. it is an
additive offset, and an additive offset does not change a correlation -- which is
why between-mouse comparisons, the only thing done below, survive it.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import signal, stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / 'learning_individuality')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FS = 60.0
BAND = (0.5, 8.0)
NPERSEG = 512
MIN_SEG = 512
DM_DIR = ROOT / 'data' / 'design_matrices'
OUT = HERE / 'lda1_vs_raw_paw_laterality.csv'
RESOLUTION = {'l_paw': 2.0, 'r_paw': 1.0}   # left camera is 1280x1024, right 640x512


def band_power(v):
    """Welch power in BAND over the NaN-free runs of a velocity trace.

    Segments rather than interpolation: paw_bias drops tracking NaNs and never
    fills them, and filling would inject spurious low-frequency power exactly in
    the band being measured.
    """
    ok = np.isfinite(v)
    if ok.sum() < MIN_SEG:
        return np.nan
    edges = np.flatnonzero(np.diff(ok.astype(np.int8)))
    bounds = np.r_[0, edges + 1, len(ok)]
    psds, weights = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not ok[a] or (b - a) < MIN_SEG:
            continue
        f, p = signal.welch(v[a:b], fs=FS, nperseg=min(NPERSEG, b - a))
        m = (f >= BAND[0]) & (f <= BAND[1])
        psds.append(np.trapz(p[m], f[m]))
        weights.append(b - a)
    if not psds:
        return np.nan
    return float(np.average(psds, weights=weights))


def session_metrics(path):
    cols = ['Bin', 'l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y']
    d = pd.read_parquet(path, columns=cols)
    out = {}
    for paw in ('l_paw', 'r_paw'):
        tot = 0.0
        for ax in ('x', 'y'):
            pos = d[f'{paw}_{ax}'].to_numpy(float) / RESOLUTION[paw]
            vel = np.diff(pos, prepend=np.nan) * FS
            bp = band_power(vel)
            if not np.isfinite(bp):
                return None
            tot += bp
        out[paw] = tot
    L, R = out['l_paw'], out['r_paw']
    return dict(P_left=L, P_right=R, LI=(L - R) / (L + R), P_total=L + R,
                n_bins=len(d), dur_min=(d['Bin'].iloc[-1] - d['Bin'].iloc[0]) / 60.0)


def paw_state_occupancy():
    """Per-session occupancy of the 8 HMM paw states, from the same syllable file
    the LDA uses. label = paw_state + 8*whisk + 16*lick, so paw_state = label % 8."""
    from session_filters import exclusions_by_timepoint
    prob = set(exclusions_by_timepoint('filtered_out')['Proficient'])
    d = pd.read_parquet(ROOT / 'data' / '8_k_10_bin_syllables_19-08-2026',
                        columns=['sample', 'mouse_name', 'binned_sequence'])
    d['session'] = d['sample'].str[:36]
    d = d[~d.session.isin(prob)]
    rows = {}
    for sess, grp in d.groupby('session'):
        v = np.concatenate([np.asarray(s, float) for s in grp['binned_sequence'].values])
        v = v[np.isfinite(v)]
        st = (v.astype(int) % 8)
        occ = np.bincount(st, minlength=8) / max(len(st), 1)
        rows[sess] = occ
    occ = pd.DataFrame(rows).T
    occ.columns = [f'state{i}' for i in range(8)]
    # states 3, 5 are left-lateralised; 4, 6 right-lateralised
    occ['state_LI'] = ((occ.state3 + occ.state5) - (occ.state4 + occ.state6)) / \
                      ((occ.state3 + occ.state5) + (occ.state4 + occ.state6))
    return occ


def report(x, y, label, n_perm=10000, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    r, pr = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    null = np.array([abs(np.corrcoef(rng.permutation(x), y)[0, 1]) for _ in range(n_perm)])
    p_perm = (np.sum(null >= abs(r)) + 1) / (n_perm + 1)
    print(f"  {label:36s} n={len(x):4d}  r={r:+.3f} (p={pr:.2g})  "
          f"rho={rho:+.3f} (p={prho:.2g})  p_perm={p_perm:.4f}")
    return dict(label=label, n=len(x), r=r, p=pr, rho=rho, p_spearman=prho, p_perm=p_perm)


def main():
    import lda_allsessions_heldout as L
    clustered = L.main()                      # LD1-LD3 per session, all 330
    lda = clustered.set_index('session')[[0, 1, 2]].rename(
        columns={0: 'LD1', 1: 'LD2', 2: 'LD3'})
    lda['mouse_name'] = clustered.set_index('session')['mouse_name']
    lda['cohort'] = clustered.set_index('session')['cohort']

    print("\n" + "=" * 72)
    print("RAW PAW BAND POWER (0.5-8 Hz) PER SESSION")
    print("=" * 72)
    by_eid = {p.name.split('_')[2]: p for p in DM_DIR.glob('design_matrix_*')}
    todo = [s for s in lda.index if s in by_eid]
    print(f"{len(todo)} of {len(lda)} LDA sessions have a design matrix on disk")

    rows = []
    for i, sess in enumerate(todo, 1):
        try:
            m = session_metrics(by_eid[sess])
        except Exception as e:
            print(f"  !! {sess[:8]}: {type(e).__name__}: {e}")
            continue
        if m is None:
            print(f"  !! {sess[:8]}: too few NaN-free samples")
            continue
        m['session'] = sess
        rows.append(m)
        if i % 25 == 0:
            print(f"  ... {i}/{len(todo)}")
    paw = pd.DataFrame(rows).set_index('session')
    print(f"✓ {len(paw)} sessions measured")

    df = lda.join(paw, how='inner').join(paw_state_occupancy(), how='left')
    df.to_csv(OUT)
    print(f"saved {OUT}")

    print(f"\nLI: mean {df.LI.mean():+.3f}, sd {df.LI.std():.3f}  "
          f"<- the MEAN is instrumental (camera gain), only the spread is usable")

    mouse = df.groupby('mouse_name').agg(
        LD1=('LD1', 'mean'), LD2=('LD2', 'mean'), LI=('LI', 'mean'),
        P_total=('P_total', 'mean'), state_LI=('state_LI', 'mean'),
        n_sessions=('LD1', 'size'))

    print("\n" + "=" * 72)
    print("MOUSE LEVEL  (one point per mouse -- the unit that is not pseudo-replicated)")
    print("=" * 72)
    res = [report(mouse.LD1, mouse.LI, 'LD1 vs raw paw LI'),
           report(mouse.LD2, mouse.LI, 'LD2 vs raw paw LI'),
           report(mouse.LD1, mouse.state_LI, 'LD1 vs HMM-state LI (in-pipeline)'),
           report(mouse.state_LI, mouse.LI, 'HMM-state LI vs raw paw LI')]

    print("\n" + "=" * 72)
    print("SESSION LEVEL  (pseudo-replicated: sessions of one mouse are not independent)")
    print("=" * 72)
    report(df.LD1, df.LI, 'LD1 vs raw paw LI')
    report(df.LD2, df.LI, 'LD2 vs raw paw LI')

    print("\nis the measured subset representative of the full LD1 range?")
    full = clustered.groupby('mouse_name')[0].mean()
    sub = full.loc[mouse.index]
    print(f"  all 101 mice : LD1 mean {full.mean():+.3f} sd {full.std():.3f} "
          f"range [{full.min():+.2f}, {full.max():+.2f}]")
    print(f"  the {len(sub)} here : LD1 mean {sub.mean():+.3f} sd {sub.std():.3f} "
          f"range [{sub.min():+.2f}, {sub.max():+.2f}]")
    print(f"  Levene equal-variance p = {stats.levene(full.values, sub.values).pvalue:.3f}")

    return df, mouse, res


if __name__ == '__main__':
    main()
