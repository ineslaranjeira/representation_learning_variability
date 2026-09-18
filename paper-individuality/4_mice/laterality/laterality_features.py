"""
PER-SESSION PAW LATERALITY FEATURES -- position, wavelets, syllable use
=======================================================================
Feature extraction for `lda3_paw_laterality.ipynb`, which asks whether LD3 of the
mouse LDA embedding is a left/right forepaw axis. Kept out of the notebook because
reading 269 wavelet files takes a few minutes and the notebook should be re-runnable
in seconds; `paw_features()` caches to CSV next to this file.

THREE VIEWS OF THE SAME BEHAVIOUR, deliberately at different removes from the LDA
---------------------------------------------------------------------------------
  syllable_features()  occupancy of the 8 HMM paw states, from the very file the LDA
                       is built on. This is NOT independent evidence -- it says what
                       inside the design matrix LD3 is reading -- but four of the
                       states are themselves lateralised (state_profiles below), so
                       it is the mechanism.
  paw_features()       raw forepaw POSITION and the paw velocity WAVELETS, per
                       session, straight from data/paw_wavelets/. A different
                       representation of the same cameras: no HMM, no binning into
                       trials, no epoch structure.
  segmentation/paw_bias/paw_bias_sessions.csv   the validated metrics from the paw
                       bias analysis, loaded for cross-checking and for the
                       gain-invariant measures (bout rate at matched duty cycle,
                       spectral shape, lead-lag) that a camera-gain error cannot
                       touch.

CONVENTIONS, taken from segmentation/paw_bias (which established them)
----------------------------------------------------------------------
  * l_paw_* = leftCamera 'paw_r' = the mouse's LEFT forepaw; r_paw_* = right forepaw.
    Both are the near, well-resolved view. Verified in qc_paw_identity.py.
  * LI = (left - right) / (left + right); POSITIVE = LEFT paw.
  * POSITION is resolution-corrected: the left camera is 1280x1024 against the
    right's 640x512, so left-camera pixels are halved (what get_speed does).
  * WAVELET band power is left UNCORRECTED, exactly as extract_paw_bias.py computes
    it, so the numbers here can be validated against paw_bias_sessions.csv. The
    correction is a constant per paw and cannot change a correlation across sessions.
  * 32 Hz wavelet power is the TRACKING-NOISE proxy: a forepaw cannot oscillate
    there. li_noisepow is the confound every result has to be partialled on -- raw
    amplitude LI correlates r = +0.8 with it (paw_bias section 0).

THE POPULATION MEAN OF ANY LI HERE IS INSTRUMENTAL, not biology: the two paws are
measured by two different cameras and the gain difference is the whole of the
apparent group-level left bias. Only the SPREAD across mice is interpretable.
"""
import os
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy import signal, stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent                     # .../paper-individuality
DATA = ROOT / 'data'
WV_DIR = DATA / 'paw_wavelets'
PAW_BIAS_CSV = ROOT / 'segmentation' / 'paw_bias' / 'paw_bias_sessions.csv'
STATE_PROFILES = ROOT / 'segmentation' / 'paw_bias' / 'state_profiles_19Ago2026.csv'
CACHE = HERE / 'paw_laterality_sessions.csv'

FS = 60.0                                     # design-matrix bin rate
BAND = ['0.5', '1.0', '2.0', '4.0', '8.0']    # real forepaw movement
NOISE = '32.0'                                # tracking noise proxy
RESOLUTION = {'l': 2.0, 'r': 1.0}             # left camera pixels are 2x
N_PAW_STATES = 8
MIN_FRAMES = 6000                             # paw_bias's floor: 100 s of paired data


# --------------------------------------------------------------------------- utils
def li(left, right):
    """Laterality index in [-1, 1]; positive = left paw. NaN if the sum is degenerate."""
    tot = left + right
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    return float((left - right) / tot)


def _wavelet_cols():
    return [f'{p}_paw_{ax}{f}' for p in 'lr' for ax in 'xy' for f in BAND + [NOISE]]


# ------------------------------------------------------- per-session paw features
def session_paw_features(path):
    """Position and wavelet laterality for one session's paw_vel_wavelets file.

    Everything is computed on the frames where BOTH paws are tracked, so the two
    sides are paired within the session and a dropout on one camera cannot shift the
    index (see the note in MEMORY about NaN bins being right-paw dropouts). NaNs are
    dropped, never interpolated.
    """
    cols = ['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y'] + _wavelet_cols()
    d = pd.read_parquet(path, columns=cols)
    out = {'n_frames': len(d)}
    for p in 'lr':
        out[f'nan_{p}'] = float(d[f'{p}_paw_x'].isna().mean())

    pos_ok = np.isfinite(d[['l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y']].to_numpy(float)).all(axis=1)
    out['n_pos'] = int(pos_ok.sum())
    if pos_ok.sum() >= MIN_FRAMES:
        for p in 'lr':
            x = d[f'{p}_paw_x'].to_numpy(float)[pos_ok] / RESOLUTION[p]
            y = d[f'{p}_paw_y'].to_numpy(float)[pos_ok] / RESOLUTION[p]
            # POSTURAL SPREAD: how far this paw ranges around its own resting place.
            # Centred on the median rather than the mean so a tracking excursion does
            # not move the origin; robust scale (IQR) reported alongside the SD because
            # the SD of a position trace is sensitive to the same outliers.
            out[f'possd_{p}'] = float(np.hypot(x.std(), y.std()))
            out[f'posiqr_{p}'] = float(np.mean([stats.iqr(x), stats.iqr(y)]))
            out[f'exc_{p}'] = float(np.mean(np.hypot(x - np.median(x), y - np.median(y))))
            out[f'x_med_{p}'], out[f'y_med_{p}'] = float(np.median(x)), float(np.median(y))
        for k in ('possd', 'posiqr', 'exc'):
            out[f'li_{k}'] = li(out[f'{k}_l'], out[f'{k}_r'])

    wv_ok = d[_wavelet_cols()].notna().all(axis=1).to_numpy()
    out['n_wv'] = int(wv_ok.sum())
    if wv_ok.sum() >= MIN_FRAMES:
        w = d.loc[wv_ok]
        pw = {(p, f): float(np.mean(np.hypot(w[f'{p}_paw_x{f}'], w[f'{p}_paw_y{f}'])))
              for p in 'lr' for f in BAND + [NOISE]}
        for p in 'lr':
            out[f'bandpow_{p}'] = sum(pw[(p, f)] for f in BAND)
            out[f'noisepow_{p}'] = pw[(p, NOISE)]
        out['li_bandpow'] = li(out['bandpow_l'], out['bandpow_r'])
        out['li_noisepow'] = li(out['noisepow_l'], out['noisepow_r'])
        for f in BAND:
            out[f'li_pow_{f}'] = li(pw[('l', f)], pw[('r', f)])
        # SNR-normalised amplitude: dividing each paw's band power by its own 32 Hz
        # floor removes a multiplicative per-camera gain exactly. paw_bias section 4
        # shows it OVER-corrects (the noise floor is not purely multiplicative), so it
        # is a bound on the other side of li_bandpow rather than a better estimate.
        out['li_snr'] = li(out['bandpow_l'] / out['noisepow_l'],
                           out['bandpow_r'] / out['noisepow_r'])
        # split halves, for the measurement-noise ceiling any correlation is judged against
        h = np.flatnonzero(wv_ok)
        for tag, idx in [('h1', h[:len(h) // 2]), ('h2', h[len(h) // 2:])]:
            ww = d.iloc[idx]
            bp = {p: sum(float(np.mean(np.hypot(ww[f'{p}_paw_x{f}'], ww[f'{p}_paw_y{f}'])))
                         for f in BAND) for p in 'lr'}
            out[f'li_bandpow_{tag}'] = li(bp['l'], bp['r'])
    return out


def paw_features(sessions=None, cache=True, verbose=True):
    """Position + wavelet laterality for every session with a wavelet file on disk.

    `sessions` restricts the run to an eid list (the LDA cohort, normally). A cached
    CSV is reused only when it already covers every requested session, so adding
    sessions re-runs rather than silently returning a subset.
    """
    want = None if sessions is None else set(sessions)
    if cache and CACHE.exists():
        got = pd.read_csv(CACHE).set_index('session')
        if want is None or want <= set(got.index):
            if verbose:
                print(f'cached: {CACHE.name} ({len(got)} sessions)')
            return got if want is None else got.loc[sorted(want)]

    files = {f[17:53]: WV_DIR / f for f in os.listdir(WV_DIR)
             if f.startswith('paw_vel_wavelets_')}
    todo = sorted(files) if want is None else sorted(want & set(files))
    if verbose:
        missing = 0 if want is None else len(want - set(files))
        print(f'{len(todo)} sessions to extract' + (f' ({missing} have no wavelet file)' if missing else ''))

    rows = []
    for i, eid in enumerate(todo, 1):
        try:
            r = session_paw_features(files[eid])
        except Exception as e:                       # a corrupt file must not kill the run
            print(f'  !! {eid[:8]}: {type(e).__name__}: {e}')
            continue
        r['session'] = eid
        rows.append(r)
        if verbose and i % 50 == 0:
            print(f'  ... {i}/{len(todo)}')
    got = pd.DataFrame(rows).set_index('session')
    if cache:
        got.to_csv(CACHE)
        if verbose:
            print(f'wrote {CACHE} {got.shape}')
    return got


# ------------------------------------------------------------- syllable occupancy
def state_laterality():
    """Per-HMM-state laterality, recomputed from the state wavelet profiles rather
    than hardcoded, so a refit of the clustering changes it here automatically.

    state_profiles_19Ago2026.csv holds the mean session-z-scored wavelet power of
    every (paw, axis, frequency) in each state. `LI` below averages the 10 left
    columns against the 10 right ones -- the same summary describe_states.py prints.
    """
    m = pd.read_csv(STATE_PROFILES).set_index('state')
    left = m[[c for c in m.columns if c.startswith('l_paw')]].mean(axis=1)
    right = m[[c for c in m.columns if c.startswith('r_paw')]].mean(axis=1)
    return pd.DataFrame({'left': left, 'right': right,
                         'overall': m.mean(axis=1),
                         'LI': (left - right) / (left.abs() + right.abs())})


def syllable_features(syllable_file, sessions=None, state_li=None):
    """Per-session occupancy of the 8 paw states, whisk and lick, plus two syllable
    laterality indices.

    The sequence values are `identifiable_states`: state = paw + 8*whisk + 16*lick,
    so paw = value % 8 (the decomposition the LDA's own binarize() uses).

      state_LI     contrast of the two clearly-left states against the two clearly-
                   right ones, the index lab/lda1_vs_raw_paw_laterality.py uses.
      occ_LI       every state weighted by its own profile LI, so states 1, 2 and 7
                   contribute in proportion to how lateralised they actually are
                   instead of being thrown away.
    """
    d = pd.read_parquet(syllable_file, columns=['sample', 'binned_sequence'])
    d['session'] = d['sample'].str[:36]
    if sessions is not None:
        d = d[d.session.isin(set(sessions))]
    rows = {}
    for sess, g in d.groupby('session'):
        v = np.concatenate([np.asarray(s, float) for s in g['binned_sequence'].values])
        v = v[np.isfinite(v)].astype(int)
        if len(v) == 0:
            continue
        rows[sess] = np.r_[np.bincount(v % N_PAW_STATES, minlength=N_PAW_STATES) / len(v),
                           ((v // N_PAW_STATES) % 2).mean(), (v // (2 * N_PAW_STATES)).mean(),
                           len(v)]
    occ = pd.DataFrame(rows).T
    occ.columns = [f'state{i}' for i in range(N_PAW_STATES)] + ['whisk', 'lick', 'n_bins']
    occ.index.name = 'session'

    sli = state_laterality() if state_li is None else state_li
    lefts = sli.index[sli.LI > 0.25].tolist()
    rights = sli.index[sli.LI < -0.25].tolist()
    L = occ[[f'state{i}' for i in lefts]].sum(axis=1)
    R = occ[[f'state{i}' for i in rights]].sum(axis=1)
    occ['state_LI'] = (L - R) / (L + R)
    # weighted version: sum_s occ_s * LI_s, renormalised by the occupancy that carries
    # any laterality at all, so it stays on the same [-1, 1] scale as state_LI
    w = sli.LI.reindex(range(N_PAW_STATES)).to_numpy()
    O = occ[[f'state{i}' for i in range(N_PAW_STATES)]].to_numpy()
    occ['occ_LI'] = (O * w).sum(axis=1) / (O * np.abs(w)).sum(axis=1)
    return occ, sli, lefts, rights


def paw_bias_metrics(sessions=None):
    """The validated per-session metrics from segmentation/paw_bias, for cross-checking
    and for the gain-invariant measures. `lab` and `mouse` are dropped so the frame can
    be joined onto the LDA table without column collisions."""
    pb = pd.read_csv(PAW_BIAS_CSV)
    pb = pb.drop(columns=[c for c in ('lab', 'mouse') if c in pb.columns])
    pb = pb.rename(columns={'eid': 'session'}).set_index('session')
    pb = pb.add_suffix('_pb')
    return pb if sessions is None else pb.reindex(sorted(set(sessions) & set(pb.index)))


if __name__ == '__main__':
    n = paw_features(cache=True)
    print(n.describe().T.to_string())
