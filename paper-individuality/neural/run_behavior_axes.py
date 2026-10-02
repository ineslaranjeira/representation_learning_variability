"""
BEHAVIOURAL AXES vs THE THREE NEURAL METRICS -- the mechanism check for the LDA-1 result
========================================================================================
`session_axes.py` builds per-session behavioural columns (reaction time, engagement,
performance) as alternatives to LDA 1 on the x axis. This script runs them through the
SAME neural grid the LDA-1 figure uses, so the numbers are comparable cell by cell:

  * the same cached metric tables  (cfg below == the config cell of lda1_neural_summary,
    i.e. cache tag lda1_neural_pr90_fr10_sess),
  * the same cohort               (apply_cohort(..., 'common')),
  * the same ten tests            (nm.GRID),
  * the same statistic            (nm.age_effect_test: GLM slope, region + n_trials
    covariates, 2000 session-stratified permutations, partial-correlation BF10),
  * the same correction family    (cfg['FDR_FAMILY'] = 'metric').

WHAT IS AND IS NOT CORRECTED. Each predictor's ten tests are corrected among themselves
exactly as the LDA-1 grid is. Nothing corrects ACROSS the eight predictors -- that is 80
tests -- so a q here answers "among this axis's tests", not "among everything tried".
The screen in step 1 is what keeps that honest: an axis uncorrelated with LDA 1 cannot be
what LDA 1 was measuring, whatever it does on its own.

THREE QUESTIONS, THREE OUTPUTS:
  1. screen   -- does the axis correlate with LDA 1 at all? (session_axes.axis_vs_lda)
  2. grid     -- the axis in place of LDA 1, ten tests each
  3. mediation-- LDA 1 with the axis added as a covariate, and the axis with LDA 1 added.
                 If the axis is the mechanism, LDA 1's r_partial collapses when it is
                 controlled and the axis survives control for LDA 1. Run only for the
                 cells where LDA 1 itself is significant, because there is nothing to
                 mediate anywhere else.

    python run_behavior_axes.py            # all axes
    python run_behavior_axes.py rt_median switch_rate
"""
import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import lda1_neural_metrics as nm
import session_axes as sa

# Behavioural axes to test. n_trials_beh is excluded: n_trials is already a covariate in
# every model in the grid, so it is not an independent predictor here.
DEFAULT_AXES = ['rt_median', 'rt_mean', 'rt_sd', 'p_engaged', 'frac_engaged',
                'switch_rate', 'performance']
DATE = pd.Timestamp.today().strftime('%d-%m-%Y')


def config():
    """cfg == the config cell of lda1_neural_summary.ipynb. Kept in one function so a
    change there is a one-line change here and the two cannot quietly diverge."""
    cfg = nm.default_config()
    cfg['MIN_FR_HZ'] = 1.0
    cfg['FR_FLOOR_REQUIRE'] = 'both'
    cfg['FR_FLOOR_SOURCE'] = 'session'
    cfg['UNIFORM_FR_FLOOR'] = True
    cfg['MIN_PRESENCE_RATIO'] = 0.90        # USE_PRESENCE_RATIO = True
    cfg['MIN_NEURONS'] = 15
    cfg['MIN_NEURONS_RSC'] = None
    return cfg


def mediation(tables, cfg, axis, res_lda, alpha=None):
    """For each grid cell where LDA 1 is significant, refit twice with both terms in.

    Returns one row per cell: LDA 1's partial r alone and with the axis controlled, and
    the axis's partial r alone and with LDA 1 controlled. Mediation looks like the first
    pair shrinking toward zero while the second pair does not.
    """
    alpha = alpha if alpha is not None else cfg['ALPHA']
    rows = []
    for _, r in res_lda[res_lda['p_perm'] < alpha].iterrows():
        tbl, y_col, _, _, region_col, covars = nm.GRID[(r['metric'], r['test'])]
        df = tables[tbl]
        both = nm.age_effect_test(df, y_col, predictor='lda_1', region_col=region_col,
                                  extra_covars=tuple(covars) + (axis,),
                                  n_perm=cfg['N_PERM'], seed=cfg['SEED'])
        flip = nm.age_effect_test(df, y_col, predictor=axis, region_col=region_col,
                                  extra_covars=tuple(covars) + ('lda_1',),
                                  n_perm=cfg['N_PERM'], seed=cfg['SEED'])
        rows.append(dict(
            axis=axis, metric=r['metric'], test=r['test'], y_col=y_col, n=both['n'],
            r_lda_alone=r['r_partial'], p_lda_alone=r['p_perm'],
            r_lda_given_axis=both['r_partial'], p_lda_given_axis=both['p_perm'],
            r_axis_given_lda=flip['r_partial'], p_axis_given_lda=flip['p_perm'],
            shrinkage=1 - abs(both['r_partial']) / abs(r['r_partial'])))
    return pd.DataFrame(rows)


def main(axes=None):
    axes = axes or DEFAULT_AXES
    cfg = config()
    print(f'cache tag: {nm.cache_tag(cfg)} | {cfg["N_PERM"]} permutations | '
          f'FDR family: {cfg["FDR_FAMILY"]}')

    tables_raw = nm.load_tables(cfg=cfg)
    tables = nm.apply_cohort(tables_raw, cfg, 'common')
    ax = sa.session_axes()
    tables = sa.attach_axes(tables, ax)

    # ---- 1. the screen -----------------------------------------------------------
    lda_file = nm._first_existing(cfg['LDA_FILES'], 'LDA file')
    screen = sa.axis_vs_lda(lda_file, ax)
    print('\n--- each axis vs LDA 1 (the screen) ---')
    print(screen.to_string(index=False))
    screen.to_csv(ROOT / 'lda1_tables' / f'behavior_axes_screen_{DATE}.csv', index=False)

    # ---- 2. the grid, LDA 1 first as the reference -------------------------------
    all_res, res_lda = [], None
    for pred in ['lda_1'] + list(axes):
        print(f'\n=== predictor: {pred} ===')
        res = nm.run_tests(tables, cfg, verbose=False, predictor=pred)
        print(nm.results_table(res))
        if pred == 'lda_1':
            res_lda = res
        all_res.append(res)
    res_all = pd.concat(all_res, ignore_index=True)
    cols = ['predictor', 'metric', 'test', 'unit', 'n', 'n_sessions', 'y_col',
            'region_col', 'covars', 'slope', 'r_partial', 'bf10', 'evidence', 'p_perm',
            'q_fdr_metric', 'q_holm_metric', 'q_fdr_all', 'q_holm_all', 'family',
            'n_family', 'fdr_sig', 'significant']
    p = ROOT / 'lda1_tables' / f'behavior_axes_grid_{DATE}.csv'
    res_all[cols].to_csv(p, index=False)
    print(f'\nwrote {p.name}')

    # ---- 3. mediation, only where LDA 1 is significant ---------------------------
    med = pd.concat([mediation(tables, cfg, a, res_lda) for a in axes], ignore_index=True)
    print('\n--- mediation: does the axis account for the LDA-1 effect? ---')
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(med.round(4).to_string(index=False))
    p = ROOT / 'lda1_tables' / f'behavior_axes_mediation_{DATE}.csv'
    med.to_csv(p, index=False)
    print(f'\nwrote {p.name}')
    return screen, res_all, med


if __name__ == '__main__':
    main(sys.argv[1:] or None)
