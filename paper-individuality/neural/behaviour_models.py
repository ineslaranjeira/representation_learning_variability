"""
SESSION-LEVEL BEHAVIOUR -> NEURAL MODELS, the compute behind fig_neural's behaviour row
=======================================================================================
The one-predictor-at-a-time analysis of `behaviour_models_neural.ipynb` (section 6b), as functions, so a figure
can draw it without re-running that notebook. Same definitions throughout:

  outcomes   FF quench, FF quench onset, r_SC quench, r_SC baseline: each row (neuron / session x region) adjusted
             for region and trial count (+ neuron count for r_SC) as the LD1 grid does, residual + grand mean
             averaged per session
  predictors over each session's first TRUNC trials (the trials the neural metrics use):
             rt_median    log1p of the median reaction time
             performance  fraction correct
             elongation   median log quiescence elongation
             H_modes      trial-mode entropy (equal-count subsamples, sessions with >= MIN_MODE_TRIALS moded trials)
             learning_speed  log sessions to proficiency (per mouse)
  per outcome x predictor:
             r2_alone     R² of that predictor alone
             dr2_unique   R² lost when it is dropped from the five-predictor model
             p / q        its coefficient's p (plain SEs: sessions as units), BH-FDR within outcome (q) and across
                          all outcomes x predictors (q_all)
  plus LD1's R² alone per outcome, for reference.

    import behaviour_models as bm
    one, ld1_r2 = bm.one_at_a_time(cfg)      # cached in neural/lda1_tables/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
CACHE = ROOT / 'lda1_tables'

OUTCOMES = [('Fano factor', 'Response magnitude'), ('Fano factor', 'Response onset'),
            ('Noise correlation', 'Response magnitude'), ('Noise correlation', 'Baseline')]
OUT_LABEL = {('Fano factor', 'Response magnitude'): 'FF quench', ('Fano factor', 'Response onset'): 'FF quench onset',
             ('Noise correlation', 'Response magnitude'): 'r_SC quench', ('Noise correlation', 'Baseline'): 'r_SC baseline'}
PREDICTORS = ['rt_median', 'performance', 'elongation', 'H_modes', 'learning_speed']
NICE = {'rt_median': 'RT (median)', 'performance': 'Accuracy', 'elongation': 'Quiescence elongation',
        'H_modes': 'Trial-mode entropy', 'learning_speed': 'Sessions to\nproficiency (log)', 'lda_1': 'LD1'}
TRUNC = 400
MIN_MODE_TRIALS = 300
N_SUB = 200
SEED = 0
TRIAL_META = PAPER / 'data' / 'session_trial_meta_10-07-2026'
TRIAL_MODES = PAPER / '3_trial_modes' / 'trial_modes_8_k_10_bin_syllables_02-10-2026' / 'trials.pqt'
LEARNING_SPEED = PAPER / 'learning_individuality' / 'wheel' / 'learning_speed.parquet'


def _numpy2_pickles():
    # the LDA file was pickled under numpy 2; under numpy 1 its module paths need aliasing to unpickle
    try:
        import numpy._core.numeric  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.numeric
        import numpy.core.multiarray
        sys.modules['numpy._core.numeric'] = np.core.numeric
        sys.modules['numpy._core.multiarray'] = np.core.multiarray


def session_outcomes(tables):
    """Region- and trial-count-adjusted neural metric, averaged per session; one column per outcome."""
    import lda1_neural_metrics as nm
    cols = {}
    for cell in OUTCOMES:
        tbl, y_col, _, _, region_col, covars = nm.GRID[cell]
        d = tables[tbl].dropna(subset=[y_col, region_col, *covars])
        Z = nm._design_matrix(d, region_col, covars)
        y = d[y_col].to_numpy(float)
        resid = y - Z @ (np.linalg.pinv(Z) @ y)
        cols[OUT_LABEL[cell]] = pd.Series(resid + y.mean(), index=d.index).groupby(d['session']).mean()
    return pd.DataFrame(cols)


def session_predictors(sessions, mouse_of):
    """The five behavioural predictors per session, over the first TRUNC trials."""
    tm = pd.read_parquet(TRIAL_META).sort_values(['session', 'trial_id'])
    tm = tm[tm['session'].isin(set(sessions))]
    tm = tm[tm.groupby('session').cumcount() < TRUNC]
    tm['correct'] = ((tm['feedback'] == 'correct') if tm['feedback'].dtype == object else tm['feedback']).astype(float)
    g = tm.groupby('session')
    out = pd.DataFrame({
        'rt_median': np.log1p(g['reaction'].median()),
        'performance': g['correct'].mean(),
        'elongation': np.log(tm['elongation'].clip(lower=1e-3)).groupby(tm['session']).median(),
    })
    mt = pd.read_parquet(TRIAL_MODES, columns=['session', 'trial_id', 'trial_cluster'])
    mt = mt[mt['session'].isin(set(sessions)) & (mt['trial_id'] < TRUNC)].dropna(subset=['trial_cluster'])
    cnt = mt.groupby('session').size()
    mt = mt[mt['session'].isin(cnt.index[cnt >= MIN_MODE_TRIALS])]
    lab = mt['trial_cluster'].astype(int)
    K, n = lab.nunique(), int(mt.groupby('session').size().min())
    rng = np.random.default_rng(SEED)

    def ent(c):
        p = c[c > 0] / c.sum()
        return -(p * np.log(p)).sum()
    out['H_modes'] = pd.Series({s: np.mean([ent(np.bincount(rng.choice(v, n, replace=False), minlength=lab.max() + 1))
                                            for _ in range(N_SUB)]) / np.log(K)
                                for s, v in lab.groupby(mt['session']).apply(lambda x: x.to_numpy()).items()})
    speed = pd.read_parquet(LEARNING_SPEED)['n_sessions']
    out['learning_speed'] = np.log(pd.Series(mouse_of).reindex(out.index).map(speed).astype(float))
    return out


def _fit(d, cols, y):
    import statsmodels.api as sm
    Zx = (d[cols] - d[cols].mean()) / d[cols].std()
    zy = (d[y] - d[y].mean()) / d[y].std()
    return sm.OLS(zy.to_numpy(float), sm.add_constant(Zx)).fit()


def one_at_a_time(cfg, cache=True, verbose=True):
    """(table of r2_alone / dr2_unique with p and q per outcome x predictor, LD1's R² alone per outcome)."""
    from statsmodels.stats.multitest import multipletests
    path = CACHE / f'behaviour_one_at_a_time_{TRIAL_MODES.parent.name}.pqt'
    if cache and path.exists():
        one = pd.read_parquet(path)
        ld1 = one.attrs.get('ld1_r2') or one[one['predictor'] == 'lda_1'].set_index('outcome')['r2_alone'].to_dict()
        return one[one['predictor'] != 'lda_1'].reset_index(drop=True), ld1
    _numpy2_pickles()
    import lda1_neural_metrics as nm
    tables = nm.apply_cohort(nm.load_tables(cfg=cfg, verbose=verbose), cfg, 'common', verbose=verbose)
    cur = nm.load_lda_frame(cfg)
    Y = session_outcomes(tables)
    mouse_of = tables['sessions'].set_index('session')['mouse_name']
    X = session_predictors(Y.index, mouse_of)
    X['lda_1'] = X.index.map(dict(zip(cur['session'], cur['lda_1'])))    # the current LDA file
    D = X.join(Y).assign(mouse_name=mouse_of).dropna(subset=PREDICTORS + ['lda_1'])
    rows = []
    for y in Y.columns:
        d = D.dropna(subset=[y])
        full = _fit(d, PREDICTORS, y)
        for c in PREDICTORS:
            alone = _fit(d, [c], y)
            reduced = _fit(d, [x for x in PREDICTORS if x != c], y)
            rows.append(dict(outcome=y, predictor=c, n=len(d), r2_alone=alone.rsquared, p_alone=alone.pvalues[c],
                             dr2_unique=full.rsquared - reduced.rsquared, p_unique=full.pvalues[c]))
        rows.append(dict(outcome=y, predictor='lda_1', n=len(d), r2_alone=_fit(d, ['lda_1'], y).rsquared))
    one = pd.DataFrame(rows)
    beh = one['predictor'] != 'lda_1'
    for p_col in ['p_alone', 'p_unique']:
        q = p_col.replace('p_', 'q_')
        one.loc[beh, q] = one[beh].groupby('outcome')[p_col].transform(lambda p: multipletests(p, method='fdr_bh')[1])
        one.loc[beh, q + '_all'] = multipletests(one.loc[beh, p_col], method='fdr_bh')[1]
    if verbose:
        print(f"behaviour models: {D.index.nunique()} sessions, {D['mouse_name'].nunique()} mice")
    if cache:
        one.to_parquet(path, index=False)
    ld1 = one[~beh].set_index('outcome')['r2_alone'].to_dict()
    return one[beh].reset_index(drop=True), ld1
