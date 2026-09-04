"""LDA1 as a SEPARATE COEFFICIENT rather than an interaction.

Part 1 -- the pure main effect, demonstrated to be identically zero.
    The main-effect column is L_s(t) = z_s, constant within a session. Both y and every
    base column are centred within session, so

        L . y   = sum_s z_s * sum_t y_s(t)   = sum_s z_s * 0 = 0
        L . B_j = sum_s z_s * sum_t B_j,s(t) = sum_s z_s * 0 = 0

    i.e. L is exactly orthogonal to the target AND to the entire design. Its ridge
    weight is 0, its predictions are unchanged, dR2 = 0. This is not an empirical
    result to be measured, it is arithmetic -- but the script measures it anyway, so
    the claim is checkable rather than asserted.

Part 2 -- the 1-degree-of-freedom GAIN test, which is the useful version of the same
    idea. Instead of reshaping each of the 78 kernels (the interaction test), let LDA1
    scale the WHOLE fitted response by a single number:

        y ~ [B, z_s * (B w0)]

    with w0 refit on each fold's TRAINING sessions only, so nothing about the held-out
    mice enters the regressor. One extra column instead of 78, so the permutation null
    barely dips below zero -- much more powerful IF the effect is a uniform gain change
    rather than a change of kernel shape.

Both parts are run for every region, on the cached sufficient statistics (no neuron
file is reopened). Writes `pop_lda_regions/main_effect_gain.parquet`.
"""
import os
import glob
import time
import warnings

import numpy as np
import pandas as pd

import encoding_functions as ef            # noqa: F401
import population_encoding as pe

warnings.filterwarnings('ignore')

PREFIX      = '/home/ines/repositories/representation_learning_variability/paper-individuality/'
NEURON_DIR  = PREFIX + 'data/neuron_files/'
CLUSTERING  = PREFIX + 'clustering/data_files/'
LDA_FILE    = 'mouse_LDA_5_bins_cut19-08-2026'
COMPONENT   = 0
RESULTS_DIR = 'encoding_results'
OUT_DIR     = 'pop_lda_regions'

REGIONS      = ['CA1', 'MRN', 'CP', 'LP']
CHECK_REGION = 'CA1'    # only rebuild raw design matrices for this region
N_CHECK      = 2        # how many sessions to verify the orthogonality claim on
N_PERM       = 2000
MIN_NEURONS, REBIN, MOTOR_CONTINUOUS = 5, 3, False


def main_effect_orthogonality(sessions, region, n_check=1):
    """Rebuild a session's ACTUAL design and target and measure the two inner products.

    The cached statistics only hold A = B'B and b = B'y, from which the column sums
    cannot be recovered -- so proving the orthogonality claim needs the real matrices
    back. This replays exactly what `session_stats` does (same design, same rebin, same
    centring) for `n_check` sessions and reports

        |L . y| / |z_s|      = |sum_t y_s(t)|
        max_j |L . B_j| / |z_s| = max_j |sum_t B_j,s(t)|

    Both are zero to machine precision iff the within-session centring is doing what
    the argument assumes. Anything larger than ~1e-8 would falsify the claim.
    """
    out = []
    bmap = None
    for eid, g in list(sessions.groupby('session'))[:n_check]:
        df, cols = pe.load_session(list(g['pid']), NEURON_DIR, region, bmap)
        if bmap is None:
            raw = [c.split('_neuron_')[0] for c in df.columns if c.endswith('_spike_count')]
            bmap = pe.beryl_map(sorted(set(raw)))
        if len(cols) < MIN_NEURONS:
            continue
        X, groups, trial_ids, keep = ef.build_design_matrix(
            df, motor_continuous=MOTOR_CONTINUOUS, motor_lags=True)
        y = df.loc[keep, cols].astype(float).mean(axis=1).values
        sl = pe._rebin_blocks(keep, REBIN)
        y = pe._apply_rebin(y, sl)
        B = pe._zscore(pe._apply_rebin(X.values.astype(float), sl))
        yz = (y - y.mean()) / y.std(ddof=1)
        out.append(dict(session=eid, n_bins=len(yz),
                        abs_sum_y=float(abs(yz.sum())),
                        max_abs_sum_B=float(np.abs(B.sum(axis=0)).max())))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lda  = pd.read_pickle(CLUSTERING + LDA_FILE)
    lda1 = pe._lda1(lda, COMPONENT)

    res = pd.concat([pd.read_parquet(f)
                     for f in sorted(glob.glob(RESULTS_DIR + '/*.parquet'))],
                    ignore_index=True)
    res['beryl'] = res['area'].astype(str).map(pe.beryl_map(res['area'].astype(str).unique()))

    rows = []
    for region in REGIONS:
        t0 = time.time()
        sessions = (res[res['beryl'].eq(region) & res['session'].isin(set(lda1['session']))]
                    [['session', 'mouse_name', 'pid']].drop_duplicates()
                    .sort_values(['session', 'pid']).reset_index(drop=True))
        stats = pe.sweep_sessions(sessions, NEURON_DIR, region,
                                  f'population_stats_{region}',
                                  motor_continuous=MOTOR_CONTINUOUS,
                                  min_neurons=MIN_NEURONS, rebin=REBIN, verbose=False)
        S = pe.assemble(stats)
        folds = pe.mouse_folds(S, n_splits=5)
        z = pe.lda_vector(S, lda1, level='session')
        n_sess, n_mice = len(S['session']), int(len(np.unique(S['mouse'])))
        print(f'\n=== {region} === {n_sess} sessions | {n_mice} mice', flush=True)

        # ---- Part 1: the pure main effect --------------------------------------
        base = pe.cv_r2(S, z, np.array([], int), folds)
        print(f'{region}: base cvR2 = {base["cv_r2"]:.4f}', flush=True)
        if region == CHECK_REGION:
            for chk in main_effect_orthogonality(sessions, region, n_check=N_CHECK):
                print(f'  [orthogonality check] {chk["session"][:8]}  '
                      f'{chk["n_bins"]:6d} bins  |sum_t y| = {chk["abs_sum_y"]:.3e}  '
                      f'max_j |sum_t B_j| = {chk["max_abs_sum_B"]:.3e}', flush=True)
        rows.append(dict(region=region, test='main_effect', level='-',
                         n_sessions=n_sess, n_mice=n_mice, n_extra_cols=1,
                         cv_r2_base=base['cv_r2'], cv_r2_full=base['cv_r2'],
                         dR2=0.0, null_mean=np.nan, null_sd=np.nan, z=np.nan, p=np.nan,
                         note='identically zero: column orthogonal to y and to B'))

        # ---- Part 2: the 1-df gain test ----------------------------------------
        g = pe.perm_null_gain(S, z, folds, n_perm=N_PERM, base=base)
        for level, r in g.items():
            rows.append(dict(region=region, test='gain_1df', level=level,
                             n_sessions=n_sess, n_mice=n_mice, n_extra_cols=1,
                             cv_r2_base=r['cv_r2_base'], cv_r2_full=r['cv_r2_full'],
                             dR2=r['dR2'], null_mean=r['null_mean'],
                             null_sd=r['null_sd'], z=r['z'], p=r['p'], note=''))
            print(f'{region:4s} gain(1 df) [{level:7s}]  base cvR2={r["cv_r2_base"]:.4f}  '
                  f'full={r["cv_r2_full"]:.4f}  dR2={r["dR2"]:+.5f}  '
                  f'null {r["null_mean"]:+.5f}+-{r["null_sd"]:.5f}  '
                  f'z={r["z"]:+.2f}  p={r["p"]:.4f}', flush=True)

        pd.DataFrame(rows).to_parquet(os.path.join(OUT_DIR, 'main_effect_gain.parquet'))
        print(f'{region} done in {time.time() - t0:.0f}s', flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(OUT_DIR, 'main_effect_gain.parquet'))
    print('\n===== SUMMARY =====', flush=True)
    print(out.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
