"""
Is the slow within-session change of the NM signal ("trial position" in notebook 7) technical or engagement?

    python neuromodulators/trial_position_engagement.py glmhmm      # 1. fit the GLM-HMM per mouse, cache posteriors
    python neuromodulators/trial_position_engagement.py fit [n_jobs] # 2. per-fiber analyses, cached
    python neuromodulators/trial_position_engagement.py summary      # 3. across-mice tables

1. GLM-HMM: the paper's K=2 model (paper-individuality/GLM-HMM, numpy reimplementation of ssm in
   session_glmhmm_em.py), fitted per mouse on all its sessions in the photometry export, with the paper's inputs
   (stimulus z-scored over the mouse's pooled sessions, previous choice, win-stay/lose-shift, bias) and no-go
   trials masked. The engaged state is the one with the steeper stimulus weight. -> p_engaged per trial.
2. Per fiber, on the masked windows of nm_syllables (both outcomes pooled):
   a. shape: per sample, OLS β of session time (z-scored) with the task variables held fixed. A technical additive
      drift gives a flat β(t); a loss of sensor gain gives β(t) proportional to the PSTH.
   b. clocks: per sample, CV R² that each slow regressor adds to the task variables (without trial position):
      session time (bleaching / light exposure), trial count, cumulative reward (satiety), p_engaged.
   c. beyond the trend: CV R² that p_engaged adds when a smooth function of session time (cubic B-spline, 5 df)
      is already in, i.e. whether engagement explains the signal at a fixed point in the session.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'paper-individuality' / 'GLM-HMM'))
from Functions import nm_syllables as ns  # noqa: E402
import session_glmhmm_em as em  # noqa: E402

OUT = ns.DATA / 'trial_position_engagement'
OUT.mkdir(exist_ok=True)
GLMHMM_FILE = OUT / 'glmhmm_k2_per_mouse.pqt'
FIT_FILE = OUT / f'per_fiber_{ns.RUN}.pkl'
CLOCKS = ['session time', 'trial count', 'cumulative reward', 'p_engaged']
SPLINE_DF = 5


# ------------------------------------------------------------------------------------------------
# 1. GLM-HMM per mouse
# ------------------------------------------------------------------------------------------------
def glmhmm_inputs(tr, rng):
    """The paper's design (engaged.py process_bwm_mouse) for one session's full trials table, unscaled stimulus.
    y = 1 for IBL choice -1, as in the paper; no-go (choice 0) -> masked."""
    stim = tr['contrastRight'].fillna(0).to_numpy() - tr['contrastLeft'].fillna(0).to_numpy()
    y = tr['choice'].map({1: 0, -1: 1, 0: -1}).to_numpy().astype(int)
    prev = np.r_[y[0], y[:-1]]
    prev[prev == -1] = rng.choice([0, 1], (prev == -1).sum())
    prev_bin = 2 * prev - 1
    reward = tr['feedbackType'].to_numpy()
    wsls = np.r_[reward[0], reward[:-1]] * prev_bin
    wsls[wsls == 0] = -1
    X = np.column_stack([stim, prev_bin, wsls, np.ones(len(tr))]).astype(float)
    return X, np.where(y == -1, 1, y), (y != -1).astype(float)


def fit_glmhmm_all():
    rec = ns.load_recordings()
    rng = np.random.default_rng(0)
    rows = []
    for subject, eids in rec.groupby('subject')['eid'].unique().items():
        trials = [ns.h5_loaders.load_trials(e) for e in eids]
        parts = [glmhmm_inputs(t, rng) for t in trials]
        stim = np.concatenate([p[0][:, 0] for p in parts])
        mu, sd = stim.mean(), stim.std()
        inputs = [np.column_stack([(X[:, 0] - mu) / sd, X[:, 1:]]) for X, _, _ in parts]
        fit = em.fit_glmhmm(inputs, [p[1] for p in parts], num_states=2, masks=[p[2] for p in parts])
        engaged = int(np.argmin(fit['W'][:, 0]))  # p(right) = sigmoid(-x W): the most negative stim weight
        for e, t, g in zip(eids, trials, fit['posteriors']):
            rows.append(pd.DataFrame({'eid': e, 'subject': subject, 'trial': t['trial'].to_numpy(),
                                      'p_engaged': g[:, engaged]}))
        W = fit['W']
        print(f"{subject}: {len(eids)} sessions, converged={fit['converged']} ({fit['n_iter']} it), "
              f"stim w engaged {W[engaged, 0]:+.2f} / other {W[1 - engaged, 0]:+.2f}, "
              f"P(stay) {np.diag(fit['trans']).round(3)}, mean p_engaged {np.concatenate(fit['posteriors'])[:, engaged].mean():.2f}",
              flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(GLMHMM_FILE)


# ------------------------------------------------------------------------------------------------
# 2. Per fiber
# ------------------------------------------------------------------------------------------------
def slow_regressors(meta, eid, post):
    """Per go trial (rows of meta): the candidate clocks, z-scored within the session."""
    tr = ns.h5_loaders.load_trials(eid).set_index('trial')
    cum_reward = tr['rewardVolume'].fillna(0).cumsum().shift(1, fill_value=0)   # before this trial
    t0 = tr['intervals_0'].min()
    p = post.set_index('trial')['p_engaged']
    out = pd.DataFrame({'session time': (meta['stimOnTrigger_times'] - t0).to_numpy(),
                        'trial count': meta['trial'].to_numpy().astype(float),
                        'cumulative reward': cum_reward.reindex(meta['trial']).to_numpy(),
                        'p_engaged': p.reindex(meta['trial']).to_numpy()})
    return (out - out.mean()) / out.std()


def bspline(x, df=SPLINE_DF):
    """Cubic B-spline basis (n x df, no intercept) of x with df - 3 quantile knots."""
    from scipy.interpolate import BSpline
    inner = np.quantile(x, np.linspace(0, 1, df - 2)[1:-1])
    lo, hi = x.min() - 1e-9, x.max() + 1e-9
    knots = np.r_[[lo] * 4, inner, [hi] * 4]
    B = np.stack([BSpline(knots, np.eye(len(knots) - 4)[i], 3)(x) for i in range(len(knots) - 4)], 1)
    return B[:, 1:]


def one_fiber(fiber, post, n_folds=5, min_trials=60, seed=0):
    meta = fiber['meta']
    task, groups = ns.covariate_matrix(meta, 'full', groups=True)
    task = task[:, np.array(groups) != 'trial position']
    S = slow_regressors(meta, meta['eid'].iat[0], post)
    spline = bspline(S['session time'].to_numpy())
    rng = np.random.default_rng(seed)
    row = {'eid': meta['eid'].iat[0], 'region': fiber['region'], 'subject': meta['subject'].iat[0],
           'target_NM': meta['target_NM'].iat[0], 'n_trials': len(meta),
           'p_engaged_mean': np.nanmean(post['p_engaged']),
           'r_engaged_time': S[['p_engaged', 'session time']].corr().iat[0, 1],
           'r_clocks': S.corr().to_numpy()}
    models = ['task'] + [f'task + {c}' for c in CLOCKS] + ['task + spline(time)', 'task + spline(time) + p_engaged']
    for w in ns.WINDOWS:
        P = fiber[w]['photometry']
        T = P.shape[1]
        beta, tval = np.full(T, np.nan), np.full(T, np.nan)
        R2 = np.full((len(models), T), np.nan)
        for j in range(T):
            ok = ~np.isnan(P[:, j]) & np.isfinite(task).all(1) & S.notna().all(1).to_numpy()
            n = ok.sum()
            if n < min_trials:
                continue
            y, C = P[ok, j], task[ok]
            b, se = ns._ols(np.column_stack([S['session time'].to_numpy()[ok], C]), y)
            beta[j], tval[j] = b, b / se
            folds = np.array_split(rng.permutation(n), n_folds)
            Xs = [C] + [np.column_stack([C, S[c].to_numpy()[ok], S[c].to_numpy()[ok] ** 2]) if c != 'p_engaged'
                        else np.column_stack([C, S[c].to_numpy()[ok]]) for c in CLOCKS]
            Xs += [np.column_stack([C, spline[ok]]), np.column_stack([C, spline[ok], S['p_engaged'].to_numpy()[ok]])]
            for k, X in enumerate(Xs):
                R2[k, j] = ns._ridge_cv_r2(X, y, folds, 'auto')
        row[f'beta_{w}'], row[f't_{w}'], row[f'r2_{w}'] = beta, tval, R2
        # PSTH of this fiber (both outcomes), for the template comparison
        row[f'psth_{w}'] = np.nanmean(P, 0)
    return row, models


def one_session(eid, posts):
    warnings.filterwarnings('ignore')
    try:
        return [one_fiber(f, posts[posts['eid'] == eid]) for f in ns.load_session(eid, movement=False)]
    except Exception as e:
        print(f'{eid}: {type(e).__name__}: {e}', flush=True)
        return []


def fit_all(n_jobs=12):
    from joblib import Parallel, delayed
    posts = pd.read_parquet(GLMHMM_FILE)
    eids = ns.load_recordings()['eid'].unique()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_session)(e, posts) for e in eids)
    rows = [r for s in res for r, _ in s]
    table = pd.DataFrame(rows)
    table.attrs['models'] = next(m for s in res for _, m in s)
    table.to_pickle(FIT_FILE)
    print(table.shape)


# ------------------------------------------------------------------------------------------------
# 3. Summary
# ------------------------------------------------------------------------------------------------
EPOCHS = {'pre-stim': ('stim', (-0.5, 0)), 'post-stim': ('stim', (0, 0.5)), 'post-fb': ('fb', (0, 1))}
NM_ORDER = ['DR-5HT', 'VTA-DA', 'SNc-DA', 'LC-NE', 'NBM-ACh']


def template_fit(beta, psth):
    """Per fiber and window: R² of β(t) by a flat template, and by a + b·PSTH(t). Returns (r2_flat, r2_psth, b)."""
    ok = np.isfinite(beta) & np.isfinite(psth)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    y, x = beta[ok], psth[ok]
    sst = np.sum((y - y.mean()) ** 2)
    A = np.column_stack([np.ones(ok.sum()), x])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return 0., 1 - np.sum((y - A @ coef) ** 2) / sst, coef[1]


def summary():
    from scipy.stats import ttest_1samp, wilcoxon
    pd.set_option('display.width', 250)
    tab = pd.read_pickle(FIT_FILE)
    models = tab.attrs['models']
    T = np.arange(-30, 30) / 30
    print('GLM-HMM: mean p_engaged per NM', tab.groupby('target_NM')['p_engaged_mean'].mean().round(2).to_dict())
    print('corr(p_engaged, session time) per NM, mouse means:',
          tab.groupby(['target_NM', 'subject'])['r_engaged_time'].mean().groupby('target_NM').mean().round(2).to_dict())

    # a. shape of β(t)
    rows = []
    for _, r in tab.iterrows():
        for w in ns.WINDOWS:
            _, r2p, b = template_fit(r[f'beta_{w}'], r[f'psth_{w}'])
            rows.append({'nm': r['target_NM'], 'subject': r['subject'], 'window': w, 'R2 PSTH template': r2p,
                         'gain slope': b, 'mean beta': np.nanmean(r[f'beta_{w}']), 'sd beta over time': np.nanstd(r[f'beta_{w}'])})
    a = pd.DataFrame(rows).groupby(['nm', 'window', 'subject']).mean().groupby(['nm', 'window']).mean().round(3)
    print('\n# a. shape of the session-time β(t): flat offset (mean β) vs ∝ PSTH (R² of a + b·PSTH, gain slope b)')
    print(a.to_string())

    # b, c. CV R² increments per epoch
    rows = []
    for _, r in tab.iterrows():
        for ep, (w, (lo, hi)) in EPOCHS.items():
            m = (T >= lo) & (T < hi)
            R = np.nanmean(r[f'r2_{w}'][:, m], 1)
            d = {'nm': r['target_NM'], 'subject': r['subject'], 'epoch': ep}
            d.update({c: R[1 + k] - R[0] for k, c in enumerate(CLOCKS)})
            d['spline(time)'] = R[-2] - R[0]
            d['p_engaged beyond spline'] = R[-1] - R[-2]
            rows.append(d)
    mice = pd.DataFrame(rows).groupby(['nm', 'epoch', 'subject']).mean()
    cols = CLOCKS + ['spline(time)', 'p_engaged beyond spline']

    def cell(x):
        x = x.dropna()
        p = ttest_1samp(x, 0).pvalue if len(x) > 2 else np.nan
        return f"{100 * x.mean():+.2f} ({(x > 0).sum()}/{len(x)}){'*' if p < .05 else ''}"
    print('\n# b, c. CV R² (%) each regressor adds to the task variables; n mice > 0; * p < .05 across mice')
    print(mice.groupby(['epoch', 'nm'])[cols].agg(cell).reindex(
        pd.MultiIndex.from_product([list(EPOCHS), NM_ORDER])).to_string())
    best = mice[['session time', 'trial count', 'cumulative reward']].idxmax(axis=1)
    print('\nbest clock per mouse (counts):')
    print(best.groupby(['epoch', 'nm']).value_counts().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    if stage == 'glmhmm':
        fit_glmhmm_all()
    elif stage == 'fit':
        fit_all(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        summary()
