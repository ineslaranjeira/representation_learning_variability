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
             learning_speed  log training days (per mouse)
             choice_bias  |P(right) - P(right expected)| on non-zero-contrast trials (fig_ld1 panel C's definition)
             bias_shift   P(right | 0.8 block) - P(right | 0.2 block) on zero-contrast trials (fig_ld1 panel C)
  per outcome x predictor:
             r2_alone     R² of that predictor alone
             dr2_unique   R² lost when it is dropped from the all-predictor model
             beta_alone / beta_unique  standardised coefficient alone / in the all-predictor model (its sign)
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
_SHORT = {'Firing rate': 'FR', 'Fano factor': 'FF', 'Noise correlation': 'r_SC'}


def out_label(cell):
    # the historical names above, else '<metric> <test>' (e.g. 'FR Response onset')
    return OUT_LABEL.get(tuple(cell), f'{_SHORT[cell[0]]} {cell[1]}')
PREDICTORS = ['rt_median', 'performance', 'elongation', 'H_modes', 'learning_speed']
# also computed by session_predictors, left out of the models (fig_neural row d): add them to PREDICTORS to include
EXTRA_PREDICTORS = ['choice_bias', 'bias_shift']
NICE = {'rt_median': 'RT (median)', 'performance': 'Accuracy', 'elongation': 'Quiescence elongation',
        'H_modes': 'Trial-mode entropy', 'learning_speed': 'Training days\n(log)', 'choice_bias': '|Choice bias|',
        'bias_shift': 'Block bias shift', 'lda_1': 'LD1'}
TRUNC = 400
MIN_MODE_TRIALS = 300
N_SUB = 200
SEED = 0
# the newest per-trial meta table (session_trial_meta_10-07-2026 is no longer on disk; 19-08-2026 has the same columns)
TRIAL_META = max((PAPER / 'data').glob('session_trial_meta_*'), key=lambda f: pd.to_datetime(f.name[-10:], dayfirst=True))
TRIAL_MODES = PAPER / '3_trial_modes' / 'trial_modes_8_k_10_bin_syllables_02-10-2026' / 'trials.pqt'
# learning_individuality/wheel/learning_speed.parquet no longer exists; the paper's learning speed is training_days in
# learning_prediction/training_time_05-05-2026 (as loaded in learning_individuality/syllable_correlations.ipynb)
LEARNING_SPEED = PAPER.parent / 'learning_prediction' / 'training_time_05-05-2026'


def _numpy2_pickles():
    # the LDA file was pickled under numpy 2; under numpy 1 its module paths need aliasing to unpickle
    try:
        import numpy._core.numeric  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.numeric
        import numpy.core.multiarray
        sys.modules['numpy._core.numeric'] = np.core.numeric
        sys.modules['numpy._core.multiarray'] = np.core.multiarray


def session_outcomes(tables, outcomes=None):
    """Region- and trial-count-adjusted neural metric, averaged per session; one column per outcome."""
    import lda1_neural_metrics as nm
    cols = {}
    for cell in (outcomes or OUTCOMES):
        tbl, y_col, _, _, region_col, covars = nm.GRID[cell]
        d = tables[tbl].dropna(subset=[y_col, region_col, *covars])
        Z = nm._design_matrix(d, region_col, covars)
        y = d[y_col].to_numpy(float)
        resid = y - Z @ (np.linalg.pinv(Z) @ y)
        cols[out_label(cell)] = pd.Series(resid + y.mean(), index=d.index).groupby(d['session']).mean()
    return pd.DataFrame(cols)


def session_predictors(sessions, mouse_of):
    """The behavioural predictors per session, over the first TRUNC trials."""
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
    # fig_ld1 panel C's definitions, over the same trials
    ch = tm['choice'].map({'left': 0.0, 'right': 1.0})            # 'no_go' -> NaN
    nz = tm['contrast'] > 0
    out['choice_bias'] = (ch[nz].groupby(tm.loc[nz, 'session']).mean()
                          - (tm.loc[nz, 'signed_contrast'] < 0).astype(float).groupby(tm.loc[nz, 'session']).mean()).abs()
    z = tm['contrast'] == 0
    zc = ch[z].groupby([tm.loc[z, 'session'], tm.loc[z, 'block']]).mean().unstack()
    out['bias_shift'] = zc[0.8] - zc[0.2]
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
    speed = pd.read_parquet(LEARNING_SPEED).set_index('mouse_name')['training_days']
    log_speed = np.log(pd.Series(mouse_of).reindex(out.index).map(speed).astype(float))
    out['learning_speed'] = log_speed.where(~log_speed.between(0, 1))   # as syllable_correlations: drops 1-2 days
    return out


def _fit(d, cols, y):
    import statsmodels.api as sm
    Zx = (d[cols] - d[cols].mean()) / d[cols].std()
    zy = (d[y] - d[y].mean()) / d[y].std()
    return sm.OLS(zy.to_numpy(float), sm.add_constant(Zx)).fit()


def cache_path(outcomes=None):
    """The cache file of one_at_a_time for these outcomes and the current PREDICTORS."""
    import hashlib
    outcomes = [tuple(c) for c in (outcomes or OUTCOMES)]
    key = repr(outcomes) + ('' if len(PREDICTORS) == 5 else repr(PREDICTORS))   # 5 = the original predictor set
    tag = '' if (outcomes == OUTCOMES and len(PREDICTORS) == 5) else '_' + hashlib.md5(key.encode()).hexdigest()[:8]
    return CACHE / f'behaviour_one_at_a_time_{TRIAL_MODES.parent.name}{tag}.pqt'


def one_at_a_time(cfg, cache=True, verbose=True, outcomes=None):
    """(table of r2_alone / dr2_unique with p and q per outcome x predictor, LD1's R² alone per outcome).
    outcomes: list of (metric, test) cells; default OUTCOMES."""
    from statsmodels.stats.multitest import multipletests
    outcomes = [tuple(c) for c in (outcomes or OUTCOMES)]
    path = cache_path(outcomes)
    if cache and path.exists():
        one = pd.read_parquet(path)
        ld1 = one.attrs.get('ld1_r2') or one[one['predictor'] == 'lda_1'].set_index('outcome')['r2_alone'].to_dict()
        return one[one['predictor'] != 'lda_1'].reset_index(drop=True), ld1
    _numpy2_pickles()
    import lda1_neural_metrics as nm
    tables = nm.apply_cohort(nm.load_tables(cfg=cfg, verbose=verbose), cfg, 'common', verbose=verbose)
    cur = nm.load_lda_frame(cfg)
    Y = session_outcomes(tables, outcomes)
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
                             beta_alone=alone.params[c], dr2_unique=full.rsquared - reduced.rsquared,
                             p_unique=full.pvalues[c], beta_unique=full.params[c], r2_full=full.rsquared))
        ld = _fit(d, ['lda_1'], y)
        rows.append(dict(outcome=y, predictor='lda_1', n=len(d), r2_alone=ld.rsquared, beta_alone=ld.params['lda_1'],
                         p_alone=ld.pvalues['lda_1']))
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


# ---------------------------------------------------------------------------------------------------------------
# The same question at the level of rows a-c: each behaviour as the predictor of nm.age_effect_test (neuron or
# session x region rows, region + trial-count covariates, the predictor shuffled across SESSIONS), alone and with
# the other behaviours as covariates (its unique effect). LD1 alone is rerun on the same sessions, for reference.
# ---------------------------------------------------------------------------------------------------------------
def perm_cache_path(outcomes):
    import hashlib
    key = repr([tuple(c) for c in outcomes]) + repr(PREDICTORS)
    return CACHE / f'behaviour_perm_{TRIAL_MODES.parent.name}_{hashlib.md5(key.encode()).hexdigest()[:8]}.pqt'


def permutation_grid(cfg, outcomes, cache=True, verbose=True):
    """One row per outcome x predictor (plus lda_1): r_partial and p_perm alone and unique; q within outcome
    (over the behaviours) and across all; LD1's q over the outcomes."""
    from statsmodels.stats.multitest import multipletests
    path = perm_cache_path(outcomes)
    if cache and path.exists():
        return pd.read_parquet(path)
    _numpy2_pickles()
    import lda1_neural_metrics as nm
    tables = nm.apply_cohort(nm.load_tables(cfg=cfg, verbose=verbose), cfg, 'common', verbose=verbose)
    mouse_of = tables['sessions'].set_index('session')['mouse_name']
    X = session_predictors(sorted(mouse_of.index), mouse_of).dropna(subset=PREDICTORS)
    keep = set(X.index)                          # one session set for every model (sessions with all predictors)
    rows = []
    for cell in outcomes:
        tbl, y_col, _, _, region_col, covars = nm.GRID[tuple(cell)]
        df = tables[tbl][tables[tbl]['session'].isin(keep)].copy()
        for c in PREDICTORS:
            df[c] = df['session'].map(X[c])
        for c in PREDICTORS + ['lda_1']:
            kw = dict(region_col=region_col, n_perm=cfg['N_PERM'], seed=cfg['SEED'])
            alone = nm.age_effect_test(df, y_col, predictor=c, extra_covars=tuple(covars), **kw)
            row = dict(outcome=out_label(cell), metric=cell[0], test=cell[1], predictor=c, n=alone['n'],
                       n_sessions=alone['n_sessions'], r_alone=alone['r_partial'], p_alone=alone['p_perm'])
            if c != 'lda_1':
                others = tuple(x for x in PREDICTORS if x != c)
                uniq = nm.age_effect_test(df, y_col, predictor=c, extra_covars=tuple(covars) + others, **kw)
                row.update(r_unique=uniq['r_partial'], p_unique=uniq['p_perm'])
            rows.append(row)
        if verbose:
            print('done', cell, flush=True)
    res = pd.DataFrame(rows)
    beh = res['predictor'] != 'lda_1'
    for p_col in ['p_alone', 'p_unique']:
        q = p_col.replace('p_', 'q_')
        res.loc[beh, q] = res[beh].groupby('outcome')[p_col].transform(lambda p: multipletests(p, method='fdr_bh')[1])
        res.loc[beh, q + '_all'] = multipletests(res.loc[beh, p_col], method='fdr_bh')[1]
    res.loc[~beh, 'q_alone'] = multipletests(res.loc[~beh, 'p_alone'], method='fdr_bh')[1]
    if cache:
        res.to_parquet(path, index=False)
    return res
