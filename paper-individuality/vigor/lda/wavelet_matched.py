"""
PAW VIGOR IN THE PIPELINE'S OWN UNITS -- and why the paw null is structural
===========================================================================
Two things about `states_vigor.py`'s paw metric do not match how the syllables are
actually built, and both are checked here.

 1  SIGNED PER-AXIS VELOCITY, NOT UNSIGNED 2D SPEED.
    `states_vigor.py` takes hypot(dx, dy) * fs -- the magnitude of the 2D velocity.
    The segmentation pipeline calls `get_speed(..., split=True)` and runs the Morlet
    transform on the SIGNED x and y components SEPARATELY
    (3.1.1_paw_wavelets.ipynb, cell 8), giving 20 features:
        {l,r}_paw_{x,y} at {0.5, 1, 2, 4, 8} Hz          (3.3_wavelet_clusters, cell 3)
    Wavelet amplitude is itself unsigned, so the difference is not signed-vs-unsigned
    at the feature level -- it is BAND-LIMITED PER-AXIS amplitude against BROADBAND
    2D speed. This script reads the actual wavelet files and re-asks the question in
    those 20 features, before any standardisation.

 2  THE FEATURES ARE Z-SCORED WITHIN SESSION -- TWICE.
    3.2_wavelet_subsample.ipynb builds the supersession as
        zscore(resampled_data, axis=0)   per session, then vstack
    and 3.3_wavelet_clusters.ipynb cell 20 assigns every bin of every session with
        mouse_data = zscore(var_array[not_nan, :], axis=0)   # within session
        mouse_data = (mouse_data - global_mean) / global_std # then globally
        -> PCA -> nearest supersession centroid
    So each session's 20 wavelet features are centred and scaled by THAT SESSION's
    own mean and SD before a state is ever assigned. A mouse's absolute movement
    amplitude is removed by construction; what reaches the HMM is the shape of the
    20-d distribution and its temporal structure.

    That makes "LD1 does not track paw amplitude" a PREDICTION of the pipeline, not a
    surprising null -- and it makes it testable directly: if amplitude is really gone,
    it should not be recoverable from the 360 syllable features at all. Section 3
    runs that regression.
"""
import os
import re
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import vigor_vs_lda as V

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WAVELET_DIR = ROOT / 'data' / 'paw_wavelets'
CACHE = HERE / 'wavelet_amplitude_sessions.csv'

FREQS = ['0.5', '1.0', '2.0', '4.0', '8.0']
VAR_INTEREST = [f'{p}_{c}{f}' for p in ('l_paw', 'r_paw') for c in ('x', 'y')
                for f in FREQS]


def build():
    """Session-level mean amplitude in each of the 20 clustering features.

    The mean over time of the RAW (pre-z-score) wavelet amplitude is exactly the
    quantity the per-session z-score throws away.
    """
    pat = re.compile(r'^paw_vel_wavelets_([0-9a-f\-]{36})_(.+)$')
    rows = []
    for f in sorted(os.listdir(WAVELET_DIR)):
        m = pat.match(f)
        if not m or not os.path.isfile(WAVELET_DIR / f):
            continue
        eid, mouse = m.groups()
        d = pd.read_parquet(WAVELET_DIR / f, columns=VAR_INTEREST)
        r = dict(session=eid, mouse_name=mouse)
        mu = d.mean()
        for c in VAR_INTEREST:
            r[f'wav_{c}'] = float(mu[c])
        # the summaries the analysis actually uses
        r['wav_all'] = float(mu.mean())
        r['wav_l'] = float(mu[[c for c in VAR_INTEREST if c.startswith('l_')]].mean())
        r['wav_r'] = float(mu[[c for c in VAR_INTEREST if c.startswith('r_')]].mean())
        for fq in FREQS:
            r[f'wav_f{fq}'] = float(mu[[c for c in VAR_INTEREST if c.endswith(fq)]].mean())
        # SHAPE, not level: the ratio of high to low band, which a per-session z-score
        # of each column separately does NOT remove (it rescales each column, so the
        # BETWEEN-column ratio of means is gone, but the within-column distribution
        # shape survives). Carried to see whether LD1 lives there instead.
        r['wav_8_over_0.5'] = r['wav_f8.0'] / r['wav_f0.5']
        r['wav_cv'] = float((d.std() / d.mean()).mean())    # per-column CV, then mean
        rows.append(r)
    W = pd.DataFrame(rows)
    W.to_csv(CACHE, index=False)
    print(f'wrote {CACHE}: {W.shape}')
    return W


def syllable_features():
    """The 360-column session x feature matrix the LDA is fit on."""
    import make_embedding as me
    return me.build_design_matrix(me.SYLLABLE_FILE)


def main():
    W = pd.read_csv(CACHE) if CACHE.exists() else build()
    os.environ['EMBEDDING'] = 'mouse_LDA_5_bins_raw_shrink0.5_360_28-09-2026'
    S, _ = V.load()
    S = S.merge(W.drop(columns=['mouse_name']), on='session', how='left')
    print(f'sessions with wavelet files: {S["wav_all"].notna().sum()}/{len(S)}')
    num = S.select_dtypes(include=[np.number]).columns
    M = S.groupby('mouse_name')[list(num)].mean()
    M['lab'] = S.groupby('mouse_name')['lab'].first()
    lab, ld1 = M['lab'].to_numpy(), M['LD1'].to_numpy()

    print('\n' + '=' * 78)
    print('1. IS THE hypot() METRIC THE SAME THING AS THE PIPELINE FEATURES?')
    print('=' * 78)
    for a, b in [('paw_mean_speed', 'wav_all'), ('paw_band_0.5_8', 'wav_all'),
                 ('paw_mean_speed_moving', 'wav_all'),
                 ('l_paw_mean_speed', 'wav_l'), ('r_paw_mean_speed', 'wav_r')]:
        print(f'  r({a:24s}, {b:8s}) = {spearmanr(M[a], M[b], nan_policy="omit")[0]:+.3f}')

    print('\n' + '=' * 78)
    print('2. THE 20 CLUSTERING FEATURES vs LD1  (pre-z-score session means)')
    print('=' * 78)
    cols = (['wav_all', 'wav_l', 'wav_r'] + [f'wav_f{f}' for f in FREQS]
            + ['wav_8_over_0.5', 'wav_cv'])
    print(f'{"feature":18s} {"rho":>7s} {"p":>9s} | {"rho|lab":>8s} {"p":>9s}')
    for c in cols:
        r, p = spearmanr(M[c], ld1, nan_policy='omit')
        rc, pc = spearmanr(V.center_within(M[c].to_numpy(), lab),
                           V.center_within(ld1, lab), nan_policy='omit')
        print(f'{c:18s} {r:+7.3f} {p:9.4g} | {rc:+8.3f} {pc:9.4g}')
    print('  per-feature, all 20:')
    rr = [(c, spearmanr(M[f'wav_{c}'], ld1, nan_policy='omit')[0]) for c in VAR_INTEREST]
    print('   max |rho| = '
          f'{max(abs(r) for _, r in rr):.3f} ({max(rr, key=lambda t: abs(t[1]))[0]}); '
          f'{sum(1 for _, r in rr if abs(r) > 0.264)}/20 reach p < 0.05 uncorrected')

    print('\n' + '=' * 78)
    print('3. IS SESSION-LEVEL AMPLITUDE RECOVERABLE FROM THE 360 SYLLABLE FEATURES?')
    print('=' * 78)
    print('   Ridge, grouped 5-fold by MOUSE, so a fold never shares a mouse. If the')
    print('   per-session z-score really removes amplitude, R2 should sit at or below 0.')
    feats, mouse_of = syllable_features()
    idx = feats.index
    targets = {'paw amplitude (wav_all)': 'wav_all',
               'paw amplitude, left (wav_l)': 'wav_l',
               'wheel speed while moving': 'wheel_mean_speed_moving',
               'wheel speed, pre-quiescence': 'wheel_speed_Pre-quiescence',
               'LD1 itself (positive control)': 'LD1'}
    sess = S.set_index('session')
    X = np.asarray(feats.loc[idx], float)
    groups = mouse_of.loc[idx].to_numpy()
    for name, col in targets.items():
        y = sess.loc[idx, col].to_numpy(float)
        ok = np.isfinite(y)
        Xo, yo, go = X[ok], y[ok], groups[ok]
        yo = np.log(yo) if col.startswith('wav') else yo
        pred = np.full(len(yo), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(Xo, yo, go):
            mdl = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(Xo[tr], yo[tr])
            pred[te] = mdl.predict(Xo[te])
        r2 = 1 - np.sum((yo - pred) ** 2) / np.sum((yo - yo.mean()) ** 2)
        print(f'   {name:32s} n={len(yo):3d}  cross-validated R2 = {r2:+.3f}   '
              f'r = {spearmanr(yo, pred)[0]:+.3f}')


if __name__ == '__main__':
    main()
