"""
Analysis 1: does what the mouse does before the stimulus predict the upcoming trial, and does the NM add to it?
Functions and the cache runner; the analysis and figures are in 2_prestim_prediction.ipynb.

    python neuromodulators/behavioural_states/prestim.py [n_jobs]          # builds the cache (all sessions)
    python neuromodulators/behavioural_states/prestim.py [n_jobs] split    # same, pre-stimulus split into Q / pre-Q

Per trial (go trials with a photometry snippet), over the pre-stimulus window PRE, clipped at trial start:
  motor   mean log(1 + |x|) of paw speed, wheel speed, whisker ME, lick rate, plus the time the trial waited beyond
          its sampled quiescence period (log; > 0 when wheel movement reset the quiescence timer)
  NM      baseline = mean photometry over the same samples; stimulus response = mean over RESP (s after stim on)
  task    |contrast| level (one-hot), block congruence, previous outcome, previous-choice congruence
          (stimulus on the side chosen last trial), trial position and its square

Three questions, each as cross-validated nested models within session (fiber for the NM), with a circular-shift
null (the added predictors rotated across trials, which keeps their slow drift):
  a. behaviour -> outcome: accuracy (logistic, Δ held-out log-likelihood in bits/trial) and log RT (ridge, ΔR²)
     from task -> task + motor.
  b. NM -> outcome beyond behaviour: task + motor -> task + motor + NM baseline (and task -> task + NM baseline).
  c. motor -> NM stimulus response: task (+ log RT, choice side, NM baseline) -> + motor (ΔR²).
Coefficients of the full models (standardised predictors, all trials) are kept to read the direction.
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from Functions import nm_syllables as ns  # noqa: E402

OUT = ns.DATA / 'prestim_prediction'
PRE = (-1.0, 0.0)        # s before stimulus onset (also clipped at trial start by the masks)
RESP = (0.1, 0.4)        # s after stimulus onset: NM stimulus response
MIN_PRE = 6              # samples (200 ms) of pre-stimulus behaviour needed for a trial
MIN_TRIALS = 150
MIN_ERRORS = 20
N_FOLDS = 5
N_NULL = 50
C_LOGIT = 0.1            # L2 strength of the logistic fits (standardised predictors)
MOTOR = ns.VIGOR + ['quiescence resets']
MOTOR_Q = [f'{v} (Q)' for v in ns.VIGOR]
MOTOR_PRE = [f'{v} (pre-Q)' for v in ns.VIGOR] + ['quiescence resets']


def _means(A, keep):
    """Per-trial mean of A over the samples where keep is True, and the number of finite samples."""
    x = np.where(keep, A, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(x, 1), np.isfinite(x).sum(1)


def pre_windows(meta, t, split=False):
    """{name: (n_trials x T) keep-mask} of the pre-stimulus window(s).
    split=False: PRE (the fiber's masks already clip it at trial start).
    split=True (fiber loaded with mask=False): 'Q' = the trial's final quiescence period, [-quiescencePeriod, 0),
    during which the wheel was held still; 'pre-Q' = the rest of PRE before it, [-1 s, -quiescencePeriod). pre-Q is
    0.3-0.6 s and is either the end of the ITI (no reset: the trial started when quiescence began) or restless time
    in the trial (resets); quiescence resets tells the two apart."""
    if not split:
        return {'': np.broadcast_to((t >= PRE[0]) & (t < PRE[1]), (len(meta), len(t)))}
    qp = meta['quiescencePeriod'].to_numpy()[:, None]
    return {' (Q)': (t[None] >= -qp) & (t[None] < 0), ' (pre-Q)': (t[None] >= PRE[0]) & (t[None] < -qp)}


def trial_features(fiber, split=False):
    """One row per trial of the fiber: task columns, motor features, NM baseline and response.
    split=True: motor and NM baseline per pre-stimulus window (column suffixes ' (Q)' / ' (pre-Q)'); the fiber must
    be loaded with mask=False (the response is then masked here as the fiber's masks would)."""
    meta, t = fiber['meta'], fiber['time']
    S = fiber['stim']
    d = meta[['eid', 'subject', 'region', 'target_NM', 'trial', 'correct', 'contrast', 'signed_contrast',
              'probabilityLeft', 'prev_correct', 'choice', 'rt', 'response_rt', 'trial_frac']].copy()
    ok = np.ones(len(meta), bool)
    for suffix, keep in pre_windows(meta, t, split).items():
        d[f'NM baseline{suffix}'], n = _means(S['photometry'], keep)
        ok &= n >= MIN_PRE
        for v in ns.VIGOR:
            d[f'{v}{suffix}'], n = _means(np.log1p(np.abs(S['movement'][v])), keep)
            ok &= n >= MIN_PRE
    wait = meta['stimOnTrigger_times'] - meta['intervals_0'] - meta['quiescencePeriod']
    d['quiescence resets'] = np.log1p(wait.clip(lower=0))
    resp = (t >= RESP[0]) & (t < RESP[1])
    if split:
        resp = resp[None] & ns.window_masks(meta, t)['stim']
    d['NM response'], _ = _means(S['photometry'], np.broadcast_to(resp, S['photometry'].shape))
    prev_choice = meta['choice'].shift(1).where(meta['trial'].diff() == 1)
    side = np.sign(meta['signed_contrast'])
    d['prev choice congruence'] = (-prev_choice * side).fillna(0.)   # IBL choice -1 = right; side +1 = right
    favoured = np.sign(0.5 - meta['probabilityLeft'])
    d['block congruence'] = (side * favoured).fillna(0.)
    d['log RT'] = np.log(meta['rt'].where(meta['rt'] > 0.05))
    d['log response RT'] = np.log(meta['response_rt'].where(meta['response_rt'] > 0))
    d['choice side'] = meta['choice'].astype(float)
    return d[ok].reset_index(drop=True)


def task_matrix(d, extra=()):
    cols = {f'contrast {v:g}': (d['contrast'] == v).astype(float) for v in sorted(d['contrast'].unique())[1:]}
    cols.update({'block congruence': d['block congruence'], 'prev correct': d['prev_correct'],
                 'prev choice congruence': d['prev choice congruence'], 'trial_frac': d['trial_frac'],
                 'trial_frac²': d['trial_frac'] ** 2})
    cols.update({c: d[c] for c in extra})
    return pd.DataFrame(cols).to_numpy(float)


def _logit_cv_bits(X, y, folds):
    """Held-out log-likelihood per trial (bits) of an L2 logistic fit on standardised predictors."""
    from sklearn.linear_model import LogisticRegression
    ll = np.empty(len(y))
    for test in folds:
        train = np.ones(len(y), bool); train[test] = False
        mu, sd = X[train].mean(0), X[train].std(0)
        sd[sd == 0] = 1
        m = LogisticRegression(C=C_LOGIT, max_iter=1000).fit((X[train] - mu) / sd, y[train])
        p = np.clip(m.predict_proba((X[test] - mu) / sd)[:, 1], 1e-6, 1 - 1e-6)
        ll[test] = np.where(y[test] == 1, np.log2(p), np.log2(1 - p))
    return ll.mean()


def _std_coefs(X, y, names, kind):
    from sklearn.linear_model import LogisticRegression
    Z = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    if kind == 'logit':
        b = LogisticRegression(C=C_LOGIT, max_iter=1000).fit(Z, y).coef_[0]
    else:
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), y - y.mean(), rcond=None)[0][1:]
    return dict(zip(names, b))


def _rotate(A, rng):
    return np.roll(A, rng.integers(len(A) // 10, len(A) - len(A) // 10), axis=0)


def nested_gain(base, add, y, kind, rng, n_null=N_NULL):
    """Δ held-out score of [base, add] over base, and the same with `add` circularly shifted n_null times."""
    folds = np.array_split(rng.permutation(len(y)), N_FOLDS)
    score = (lambda X: _logit_cv_bits(X, y, folds)) if kind == 'logit' else \
        (lambda X: ns._ridge_cv_r2(X, y, folds, 'auto'))
    s0 = score(base)
    gain = score(np.hstack([base, add])) - s0
    null = np.array([score(np.hstack([base, _rotate(add, rng)])) - s0 for _ in range(n_null)])
    return gain, null


def analyse_fiber(d, seed=0, behaviour=True):
    """Rows of (question, outcome, gain, null) and the standardised coefficients of the full models.
    behaviour=False skips question a (it doesn't depend on the fiber, so it is run once per session)."""
    rng = np.random.default_rng(seed)
    rows, coefs = [], []
    M = d[MOTOR].to_numpy(float)
    B = d[['NM baseline']].to_numpy(float)
    key = {k: d[k].iat[0] for k in ['eid', 'subject', 'region', 'target_NM']}
    for outcome, kind in [('correct', 'logit'), ('log RT', 'ridge')]:
        ok = d[outcome].notna().to_numpy() & np.isfinite(task_matrix(d)).all(1)
        y = d.loc[ok, outcome].to_numpy(float)
        if ok.sum() < MIN_TRIALS or (kind == 'logit' and min(y.sum(), len(y) - y.sum()) < MIN_ERRORS):
            continue
        T = task_matrix(d[ok])
        tests = [('b. NM | task + motor', np.hstack([T, M[ok]]), B[ok]), ('b. NM | task', T, B[ok])]
        if behaviour:
            tests = [('a. motor | task', T, M[ok])] + tests
        for q, base, add in tests:
            g, null = nested_gain(base, add, y, kind, rng)
            rows.append({**key, 'question': q, 'outcome': outcome, 'n': len(y), 'gain': g, 'null': null})
        names = list(pd.Index(['task'] * T.shape[1])) + MOTOR + ['NM baseline']
        c = _std_coefs(np.hstack([T, M[ok], B[ok]]), y, names, kind)
        coefs.append({**key, 'outcome': outcome, **{k: v for k, v in c.items() if k != 'task'}})
    # c. motor -> NM stimulus response
    T = task_matrix(d, extra=['log response RT', 'choice side', 'NM baseline'])
    ok = np.isfinite(T).all(1) & d['NM response'].notna().to_numpy()
    if ok.sum() >= MIN_TRIALS:
        y = d.loc[ok, 'NM response'].to_numpy(float)
        g, null = nested_gain(T[ok], M[ok], y, 'ridge', rng)
        rows.append({**key, 'question': 'c. motor | task', 'outcome': 'NM response', 'n': len(y), 'gain': g,
                     'null': null})
        c = _std_coefs(np.hstack([T[ok], M[ok]]), y, ['task'] * T.shape[1] + MOTOR, 'ridge')
        coefs.append({**key, 'outcome': 'NM response', **{k: v for k, v in c.items() if k != 'task'}})
    return rows, coefs


def analyse_fiber_split(d, seed=0, behaviour=True):
    """As analyse_fiber, with the pre-stimulus window split into the final quiescence (Q) and the time before it
    (pre-Q). Each window's motor (or NM baseline) is tested alone over task, and beyond the other window."""
    rng = np.random.default_rng(seed)
    rows, coefs = [], []
    Mq, Mp = d[MOTOR_Q].to_numpy(float), d[MOTOR_PRE].to_numpy(float)
    Bq, Bp = d[['NM baseline (Q)']].to_numpy(float), d[['NM baseline (pre-Q)']].to_numpy(float)
    key = {k: d[k].iat[0] for k in ['eid', 'subject', 'region', 'target_NM']}
    names_m = MOTOR_Q + MOTOR_PRE
    for outcome, kind in [('correct', 'logit'), ('log RT', 'ridge')]:
        ok = d[outcome].notna().to_numpy() & np.isfinite(task_matrix(d)).all(1)
        y = d.loc[ok, outcome].to_numpy(float)
        if ok.sum() < MIN_TRIALS or (kind == 'logit' and min(y.sum(), len(y) - y.sum()) < MIN_ERRORS):
            continue
        T, mq, mp, bq, bp = task_matrix(d[ok]), Mq[ok], Mp[ok], Bq[ok], Bp[ok]
        TM = np.hstack([T, mq, mp])
        tests = [('b. NM pre-Q | task + motor', TM, bp), ('b. NM Q | task + motor', TM, bq),
                 ('b. NM Q | task + motor + NM pre-Q', np.hstack([TM, bp]), bq),
                 ('b. NM pre-Q | task + motor + NM Q', np.hstack([TM, bq]), bp)]
        if behaviour:
            tests = [('a. motor pre-Q | task', T, mp), ('a. motor Q | task', T, mq),
                     ('a. motor pre-Q | task + motor Q', np.hstack([T, mq]), mp),
                     ('a. motor Q | task + motor pre-Q', np.hstack([T, mp]), mq)] + tests
        for q, base, add in tests:
            g, null = nested_gain(base, add, y, kind, rng)
            rows.append({**key, 'question': q, 'outcome': outcome, 'n': len(y), 'gain': g, 'null': null})
        names = ['task'] * T.shape[1] + names_m + ['NM baseline (Q)', 'NM baseline (pre-Q)']
        c = _std_coefs(np.hstack([TM, bq, bp]), y, names, kind)
        coefs.append({**key, 'outcome': outcome, **{k: v for k, v in c.items() if k != 'task'}})
    # c. motor -> NM stimulus response, both baselines held fixed
    T = task_matrix(d, extra=['log response RT', 'choice side', 'NM baseline (Q)', 'NM baseline (pre-Q)'])
    ok = np.isfinite(T).all(1) & d['NM response'].notna().to_numpy()
    if ok.sum() >= MIN_TRIALS:
        y = d.loc[ok, 'NM response'].to_numpy(float)
        T, mq, mp = T[ok], Mq[ok], Mp[ok]
        for q, base, add in [('c. motor | task', T, np.hstack([mq, mp])),
                             ('c. motor Q | task + motor pre-Q', np.hstack([T, mp]), mq),
                             ('c. motor pre-Q | task + motor Q', np.hstack([T, mq]), mp)]:
            g, null = nested_gain(base, add, y, 'ridge', rng)
            rows.append({**key, 'question': q, 'outcome': 'NM response', 'n': len(y), 'gain': g, 'null': null})
        c = _std_coefs(np.hstack([T, mq, mp]), y, ['task'] * T.shape[1] + names_m, 'ridge')
        coefs.append({**key, 'outcome': 'NM response', **{k: v for k, v in c.items() if k != 'task'}})
    return rows, coefs


def analyse_session(eid, split=False):
    warnings.filterwarnings('ignore')
    try:
        fibers = ns.load_session(eid, mask=not split)
        fit = analyse_fiber_split if split else analyse_fiber
        rows, coefs = [], []
        for i, f in enumerate(fibers):
            r, c = fit(trial_features(f, split), behaviour=(i == 0))
            rows += r; coefs += c
        return rows, coefs
    except Exception as ex:
        print(f'{eid}: {type(ex).__name__}: {ex}', flush=True)
        return [], []


def tag(split):
    return f"{'split_' if split else ''}{ns.RUN}"


def run(n_jobs=12, split=False):
    from joblib import Parallel, delayed
    OUT.mkdir(exist_ok=True)
    eids = ns.load_recordings()['eid'].unique()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(analyse_session)(e, split) for e in eids)
    gains = pd.DataFrame([r for s in res for r in s[0]])
    coefs = pd.DataFrame([c for s in res for c in s[1]])
    gains.to_pickle(OUT / f'gains_{tag(split)}.pkl')
    coefs.to_pickle(OUT / f'coefs_{tag(split)}.pkl')
    return gains, coefs


def load(split=False):
    gains = pd.read_pickle(OUT / f'gains_{tag(split)}.pkl')
    coefs = pd.read_pickle(OUT / f'coefs_{tag(split)}.pkl')
    gains['z'] = (gains['gain'] - gains['null'].apply(np.mean)) / gains['null'].apply(np.std)
    gains['excess'] = gains['gain'] - gains['null'].apply(np.mean)   # gain beyond what slow co-drift gives
    return gains, coefs


def by_mouse(gains, value='excess', group=('question', 'outcome')):
    """Mouse means (over sessions / fibers), then across mice: mean, n positive, one-sample t-test p."""
    from scipy.stats import ttest_1samp
    keys = list(dict.fromkeys([*group, 'target_NM', 'subject']))
    m = gains.groupby(keys)[value].mean().reset_index()
    rows = []
    for k, g in m.groupby(list(group)):
        rows.append({**dict(zip(group, k)), 'mice': len(g), f'mean {value}': g[value].mean(),
                     'positive': f"{(g[value] > 0).sum()}/{len(g)}",
                     'p': ttest_1samp(g[value], 0).pvalue if len(g) > 2 else np.nan})
    return pd.DataFrame(rows), m


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 12, split=len(sys.argv) > 2 and sys.argv[2] == 'split')
