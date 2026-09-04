"""Population encoding + LDA1 interaction test, run separately for the best-covered
brain regions (the notebook `population_encoding_lda.ipynb` does this for one region).

Same model as the notebook, region by region:
    target  y_s(t) = region-average spike count in session s, z-scored within session
    base B  Musall design (task kernels + motor HMM states), z-scored within session
    LDA L   z_s * B  -- LDA1 rescales the kernels (a main effect is identically 0
            because z_s is constant within a session and y is centred within session)
    score   dR2 = cvR2(B + L) - cvR2(B), folds held out by MOUSE
    null    shuffle the session -> LDA1 assignment (session level and mouse level)

Writes one row per (region, block, null level) to `pop_lda_regions/summary.parquet`
and the raw null distributions to `pop_lda_regions/nulls_<REGION>.npz`.
"""
import os
import glob
import json
import time
import warnings

import numpy as np
import pandas as pd

import encoding_functions as ef            # noqa: F401  (imported by population_encoding)
import population_encoding as pe

warnings.filterwarnings('ignore')

PREFIX      = '/home/ines/repositories/representation_learning_variability/paper-individuality/'
NEURON_DIR  = PREFIX + 'data/neuron_files/'
CLUSTERING  = PREFIX + 'clustering/data_files/'
LDA_FILE    = 'mouse_LDA_5_bins_cut19-08-2026'
COMPONENT   = 0
RESULTS_DIR = 'encoding_results'           # cached per-neuron fits -> region membership
OUT_DIR     = 'pop_lda_regions'

# The four best-covered Beryl regions (see region_coverage). CP is included despite its
# yield<->LDA1 confound (rho = -0.42, p = 0.004) so the number exists, but every CP row
# is flagged: there, neurons-per-session is itself a function of the predictor.
REGIONS     = ['CA1', 'MRN', 'CP', 'LP']
YIELD_TRAP  = {'CP'}

MIN_NEURONS      = 5
REBIN            = 3
MOTOR_CONTINUOUS = False
N_PERM           = 2000
BLOCKS           = ['all', 'task', 'motor']
LEVELS           = ['session', 'mouse']


def region_sessions(res, lda1, region):
    """Probes containing `region` (Beryl) in sessions that have an LDA score."""
    return (res[res['beryl'].eq(region) & res['session'].isin(set(lda1['session']))]
            [['session', 'mouse_name', 'pid']].drop_duplicates()
            .sort_values(['session', 'pid']).reset_index(drop=True))


def check_cache(stats, region):
    """The cache does not record its build parameters beyond `rebin`; refuse to mix."""
    bad = [s['session'] for s in stats if s.get('rebin') != REBIN]
    if bad:
        raise ValueError(f'{region}: {len(bad)} cached sessions have rebin != {REBIN} '
                         f'(first: {bad[0]}). Delete the cache dir and rerun.')
    if MOTOR_CONTINUOUS is False:
        cont = [s['session'] for s in stats if 'motor_continuous' in set(s['col_group'])]
        if cont:
            raise ValueError(f'{region}: {len(cont)} cached sessions contain '
                             f'motor_continuous columns but MOTOR_CONTINUOUS=False.')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lda = pd.read_pickle(CLUSTERING + LDA_FILE)
    lda1 = pe._lda1(lda, COMPONENT)

    res = pd.concat([pd.read_parquet(f)
                     for f in sorted(glob.glob(RESULTS_DIR + '/*.parquet'))],
                    ignore_index=True)
    res['beryl'] = res['area'].astype(str).map(pe.beryl_map(res['area'].astype(str).unique()))
    print(f'{len(lda1)} sessions with an LDA score (component {COMPONENT})', flush=True)

    rows, weights = [], []
    for region in REGIONS:
        t0 = time.time()
        sessions = region_sessions(res, lda1, region)
        print(f'\n=== {region} === {sessions["session"].nunique()} sessions, '
              f'{sessions["mouse_name"].nunique()} mice, {len(sessions)} probes', flush=True)

        stats = pe.sweep_sessions(sessions, NEURON_DIR, region,
                                  f'population_stats_{region}',
                                  motor_continuous=MOTOR_CONTINUOUS,
                                  min_neurons=MIN_NEURONS, rebin=REBIN, verbose=True)
        check_cache(stats, region)
        S = pe.assemble(stats)
        n_sess, n_mice = len(S['session']), int(len(np.unique(S['mouse'])))
        print(f'{region}: {n_sess} sessions | {n_mice} mice | {S["n"].sum():,} bins | '
              f'{len(S["cols"])} base cols | neurons/session median '
              f'{np.median(S["n_neurons"]):.0f} '
              f'(min {S["n_neurons"].min()}, max {S["n_neurons"].max()}) '
              f'[{time.time() - t0:.0f}s]', flush=True)

        folds = pe.mouse_folds(S, n_splits=5)
        z = pe.lda_vector(S, lda1, level='session')
        base = pe.cv_r2(S, z, np.array([], int), folds)
        print(f'{region}: base cvR2 = {base["cv_r2"]:.4f}  '
              f'per-fold {np.round(base["r2_folds"], 3)}', flush=True)

        nulls = {}
        for which in BLOCKS:
            for level in LEVELS:
                zl = pe.lda_vector(S, lda1, level=level)
                obs = pe.delta_r2(S, zl, folds, which=which, base=base)
                r = pe.perm_null(S, zl, folds, which=which, level=level,
                                 n_perm=N_PERM, observed=obs)
                nulls[f'{which}_{level}'] = r['null']
                rows.append(dict(region=region, block=which, level=level,
                                 n_sessions=n_sess, n_mice=n_mice,
                                 median_neurons=float(np.median(S['n_neurons'])),
                                 n_interactions=obs['n_interactions'],
                                 cv_r2_base=r['cv_r2_base'], cv_r2_full=r['cv_r2_full'],
                                 dR2=r['dR2'], null_mean=r['null_mean'],
                                 null_sd=r['null_sd'], z=r['z'], p=r['p'],
                                 yield_trap=region in YIELD_TRAP))
                print(f'{region:4s} {which:6s} x LDA1 [{level:7s}] '
                      f'dR2={r["dR2"]:+.5f}  null {r["null_mean"]:+.5f}+-{r["null_sd"]:.5f}  '
                      f'z={r["z"]:+.2f}  p={r["p"]:.4f}', flush=True)

        np.savez_compressed(os.path.join(OUT_DIR, f'nulls_{region}.npz'), **nulls)

        W = pe.interaction_weights(S, z, which='all', folds=folds)
        W.insert(0, 'region', region)
        weights.append(W)

        pd.DataFrame(rows).to_parquet(os.path.join(OUT_DIR, 'summary.parquet'))
        print(f'{region} done in {time.time() - t0:.0f}s', flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(OUT_DIR, 'summary.parquet'))
    pd.concat(weights, ignore_index=True).to_parquet(
        os.path.join(OUT_DIR, 'interaction_weights.parquet'))
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(dict(lda_file=LDA_FILE, component=COMPONENT, regions=REGIONS,
                       min_neurons=MIN_NEURONS, rebin=REBIN,
                       motor_continuous=MOTOR_CONTINUOUS, n_perm=N_PERM,
                       blocks=BLOCKS, levels=LEVELS), f, indent=2)

    print('\n===== SUMMARY =====', flush=True)
    print(out.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
