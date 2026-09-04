"""What does the LDA1 interaction effect need in the BASE model?

`_run_pop_lda_regions.py` fits the full base (task kernels + motor HMM states) and puts
the LDA1 interactions on a subset of those columns. This script instead varies the BASE:

    full    B = [task, motor_states]     (what the notebook does)
    task    B = [task]                   -- motor coefficients REMOVED from the model
    motor   B = [motor_states]           -- the complement, for symmetry

and in each case the LDA block is `z_s * B`, i.e. LDA1 rescales whatever is in the base.

Why this is exact and cheap: the cached per-session sufficient statistics are
A = B'B and b = B'y over the *full* column set, so a task-only model is just the
submatrix A[task, task], b[task]. No neuron file is reopened; `syy` and `n` are
unchanged, so the cvR2 denominators stay comparable across the three variants.

What it tells you: CA1's full-base effect sits in the motor interactions (p ~ 0.042)
rather than the task ones (p ~ 0.09). Two very different readings of that:
  * motor states are the carrier -- with them gone, the effect should vanish;
  * motor states were *absorbing* shared variance -- with them gone, the task kernels
    inherit it and the task effect should get STRONGER.
Removing them is what separates the two.

Writes `pop_lda_regions/base_variants.parquet`.
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

REGIONS   = ['CA1', 'MRN', 'CP', 'LP']
VARIANTS  = {'full':  ('task', 'motor_states'),
             'task':  ('task',),
             'motor': ('motor_states',)}
N_PERM    = 2000
LEVELS    = ['session', 'mouse']
MIN_NEURONS, REBIN, MOTOR_CONTINUOUS = 5, 3, False


def subset_base(S, keep_groups):
    """S restricted to the base columns whose group is in `keep_groups`.

    A[np.ix_(j, j)] and b[j] are exactly the sufficient statistics of the reduced
    design -- dropping a column from B'B is the same as never having built it.
    `syy` and `n` are properties of the target, so they carry over untouched and the
    cvR2 of the variants are measured against the same denominator.
    """
    j = np.flatnonzero(np.isin(S['groups'], list(keep_groups)))
    if not len(j):
        raise ValueError(f'no columns in groups {keep_groups}')
    out = dict(S)
    out['cols']   = [S['cols'][i] for i in j]
    out['groups'] = S['groups'][j]
    out['A']      = S['A'][:, j][:, :, j]
    out['b']      = S['b'][:, j]
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
        S_full = pe.assemble(stats)
        print(f'\n=== {region} === {len(S_full["session"])} sessions | '
              f'{len(np.unique(S_full["mouse"]))} mice | {len(S_full["cols"])} base cols '
              f'({(S_full["groups"] == "task").sum()} task, '
              f'{(S_full["groups"] == "motor_states").sum()} motor)', flush=True)

        folds = pe.mouse_folds(S_full, n_splits=5)
        for variant, groups in VARIANTS.items():
            S = subset_base(S_full, groups)
            z = pe.lda_vector(S, lda1, level='session')
            base = pe.cv_r2(S, z, np.array([], int), folds)
            for level in LEVELS:
                zl  = pe.lda_vector(S, lda1, level=level)
                obs = pe.delta_r2(S, zl, folds, which='all', base=base)
                r   = pe.perm_null(S, zl, folds, which='all', level=level,
                                   n_perm=N_PERM, observed=obs)
                rows.append(dict(region=region, base=variant, level=level,
                                 n_base_cols=len(S['cols']),
                                 n_sessions=len(S['session']),
                                 n_mice=int(len(np.unique(S['mouse']))),
                                 cv_r2_base=r['cv_r2_base'], cv_r2_full=r['cv_r2_full'],
                                 dR2=r['dR2'], null_mean=r['null_mean'],
                                 null_sd=r['null_sd'], z=r['z'], p=r['p']))
                print(f'{region:4s} base={variant:5s} ({len(S["cols"]):2d} cols) '
                      f'[{level:7s}]  base cvR2={r["cv_r2_base"]:.4f}  '
                      f'dR2={r["dR2"]:+.5f}  null {r["null_mean"]:+.5f}+-{r["null_sd"]:.5f}  '
                      f'z={r["z"]:+.2f}  p={r["p"]:.4f}', flush=True)
            pd.DataFrame(rows).to_parquet(os.path.join(OUT_DIR, 'base_variants.parquet'))
        print(f'{region} done in {time.time() - t0:.0f}s', flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(OUT_DIR, 'base_variants.parquet'))
    print('\n===== SUMMARY =====', flush=True)
    print(out.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
