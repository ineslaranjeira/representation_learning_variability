"""
Analysis 2: does the task context change how the NM signal couples to movement?
Functions and the cache runner; the analysis and figures are in 3_coupling_by_context.ipynb.

    python neuromodulators/behavioural_states/context_coupling.py [n_jobs]            # all contexts and epochs
    python neuromodulators/behavioural_states/context_coupling.py [n_jobs] controls   # reward-consumption control

Per trial and epoch, the mean NM signal and the mean log(1 + |x|) of each vigor signal (paw speed, wheel speed,
whisker ME, lick rate), from the unmasked snippets with the masks applied here:
  pre-Q      [-1 s, -quiescencePeriod) before stimulus onset (end of ITI or restless time in the trial)
  Q          [-quiescencePeriod, 0): final quiescence, wheel held still
  post-stim  [0, 0.5) s after stimulus onset, up to feedback
  post-fb    [0, 1) s after feedback, up to trial end
Vigor is z-scored over the session (not within context, so a context with less movement keeps its smaller range),
clipped at ±Z_CLIP; a signal enters an epoch's fit only if it moves (> ACTIVE) on >= MIN_ACTIVE of the trials.

For each fiber, epoch and binary context c, one OLS fit
    NM ~ task + c + sum_v beta_v z_v + sum_v gamma_v z_v * c
task = |contrast| level, block congruence, previous outcome, trial position (+ squared), log response RT, choice
side (+ outcome after feedback); the context's own column is dropped from task. gamma_v = how much the coupling to
vigor signal v changes in context c (beta_v is the coupling in c = 0, beta_v + gamma_v in c = 1). Also the
cross-validated R² that the interactions add (ridge, circular-shift null = context labels rotated over trials), and
the SD of each vigor signal per context (range check).
Contexts:
  prev outcome   previous trial rewarded (1) vs error (0)
  late session   second half of the session's trials
  outcome        this trial rewarded vs error (post-fb only: the mouse knows it then)
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from Functions import nm_syllables as ns  # noqa: E402
import prestim as pp  # noqa: E402

OUT = ns.DATA / 'context_coupling'
EPOCHS = ['pre-Q', 'Q', 'post-stim', 'post-fb']
CONTEXTS = {'prev outcome': EPOCHS, 'late session': EPOCHS, 'outcome': ['post-fb']}
MIN_TRIALS = 100
MIN_PER_CONTEXT = 30
MIN_SAMPLES = 6
N_NULL = 50
ACTIVE = 0.01      # log(1 + |x|) above which a trial counts as moving in that signal
MIN_ACTIVE = 0.2   # fraction of trials that must move for a vigor signal to enter an epoch's fit
Z_CLIP = 5


def epoch_masks(meta, t):
    """{epoch: (window, (n_trials x T) keep-mask)} on the unmasked snippets."""
    pre = pp.pre_windows(meta, t, split=True)
    m = ns.window_masks(meta, t)
    return {'pre-Q': ('stim', pre[' (pre-Q)']), 'Q': ('stim', pre[' (Q)']),
            'post-stim': ('stim', m['stim'] & (t[None] >= 0) & (t[None] < 0.5)),
            'post-fb': ('fb', m['fb'] & (t[None] >= 0) & (t[None] < 1.0))}


def epoch_table(fiber):
    """Long table: one row per (trial, epoch) with NM, vigor (session z-scored, per epoch) and task columns."""
    meta, t = fiber['meta'], fiber['time']
    base = meta[['eid', 'subject', 'region', 'target_NM', 'trial', 'correct', 'contrast', 'signed_contrast',
                 'probabilityLeft', 'prev_correct', 'choice', 'response_rt', 'trial_frac']].copy()
    side = np.sign(meta['signed_contrast'])
    base['block congruence'] = (side * np.sign(0.5 - meta['probabilityLeft'])).fillna(0.)
    base['log response RT'] = np.log(meta['response_rt'].where(meta['response_rt'] > 0))
    base['choice side'] = meta['choice'].astype(float)
    base['late session'] = (meta['trial_frac'] > 0.5).astype(float)
    base['prev outcome'] = meta['prev_correct']
    base['outcome'] = meta['correct']
    base['in trial'] = (meta['stimOnTrigger_times'] - meta['intervals_0']) >= -pp.PRE[0]   # pre-Q inside the trial
    out = []
    for ep, (w, keep) in epoch_masks(meta, t).items():
        d = base.copy()
        d['epoch'] = ep
        d['NM'], n = pp._means(fiber[w]['photometry'], keep)
        ok = n >= MIN_SAMPLES
        for v in ns.VIGOR:
            d[v], n = pp._means(np.log1p(np.abs(fiber[w]['movement'][v])), keep)
            ok &= n >= MIN_SAMPLES
        out.append(d[ok])
    return pd.concat(out, ignore_index=True)


def task_columns(d, context, epoch):
    cols = {f'contrast {v:g}': (d['contrast'] == v).astype(float) for v in sorted(d['contrast'].unique())[1:]}
    for c in ['block congruence', 'prev outcome', 'log response RT', 'choice side']:
        cols[c] = d[c]
    cols['trial_frac'] = d['trial_frac']
    cols['trial_frac²'] = d['trial_frac'] ** 2
    if epoch == 'post-fb':
        cols['outcome'] = d['outcome']
    cols.pop(context, None)   # 'late session' keeps trial position: the context is then a step on top of the trend
    return pd.DataFrame(cols).to_numpy(float)


def fit_context(d, context, epoch, rng):
    """One fiber, epoch and context: betas, gammas, interaction ΔR² and its null, vigor SD per context."""
    d = d[d['epoch'] == epoch]
    T = task_columns(d, context, epoch)
    c = d[context].to_numpy(float)
    V = d[ns.VIGOR].to_numpy(float)
    y = d['NM'].to_numpy(float)
    ok = np.isfinite(T).all(1) & np.isfinite(c) & np.isfinite(V).all(1) & np.isfinite(y)
    T, c, V, y = T[ok], c[ok], V[ok], y[ok]
    if len(y) < MIN_TRIALS or min(c.sum(), len(c) - c.sum()) < MIN_PER_CONTEXT:
        return None
    sd = V.std(0)
    # a signal that is ~0 on most trials of this epoch (wheel or licking during quiescence) gives a few huge z
    # values that dominate the fit: keep a signal only if it moves on >= MIN_ACTIVE of the trials, and clip z
    live = (sd > 0) & ((V > ACTIVE) .mean(0) >= MIN_ACTIVE)
    Z = np.clip((V - V.mean(0)) / np.where(sd == 0, 1, sd), -Z_CLIP, Z_CLIP)[:, live]
    A = np.column_stack([np.ones(len(y)), T, c, Z, Z * c[:, None]])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    k = Z.shape[1]
    beta, gamma = np.full(len(ns.VIGOR), np.nan), np.full(len(ns.VIGOR), np.nan)
    beta[live], gamma[live] = b[-2 * k:-k], b[-k:]
    folds = np.array_split(rng.permutation(len(y)), pp.N_FOLDS)
    add = np.column_stack([T, c, Z])
    r0 = ns._ridge_cv_r2(add, y, folds, 'auto')
    gain = ns._ridge_cv_r2(np.column_stack([add, Z * c[:, None]]), y, folds, 'auto') - r0
    null = []
    for _ in range(N_NULL):
        cs = pp._rotate(c, rng)
        X0 = np.column_stack([T, cs, Z])
        null.append(ns._ridge_cv_r2(np.column_stack([X0, Z * cs[:, None]]), y, folds, 'auto')
                    - ns._ridge_cv_r2(X0, y, folds, 'auto'))
    row = {k_: d[k_].iat[0] for k_ in ['eid', 'subject', 'region', 'target_NM']}
    row.update(context=context, epoch=epoch, n=len(y), n_context=int(c.sum()), gain=gain, null=np.array(null))
    for i, v in enumerate(ns.VIGOR):
        row[f'beta {v}'] = beta[i]
        row[f'gamma {v}'] = gamma[i]
        for lvl in (0, 1):
            row[f'sd {v} | c={lvl}'] = V[c == lvl, i].std() / sd[i] if live[i] else np.nan
    return row


def analyse_session(eid, seed=0):
    warnings.filterwarnings('ignore')
    rng = np.random.default_rng(seed)
    rows = []
    try:
        for f in ns.load_session(eid, mask=False):
            d = epoch_table(f)
            for context, epochs in CONTEXTS.items():
                for ep in epochs:
                    r = fit_context(d, context, ep, rng)
                    if r is not None:
                        rows.append(r)
    except Exception as ex:
        print(f'{eid}: {type(ex).__name__}: {ex}', flush=True)
    return rows


# ---- reward-consumption control for the previous-outcome effect ------------------------------------------------
CONTROLS = {'all trials': lambda d: np.ones(len(d), bool),
            'no licking': lambda d: d['no lick'].to_numpy(),
            'pre-Q inside trial': lambda d: d['in trial'].to_numpy()}


def control_session(eid, seed=0):
    """prev outcome x {pre-Q, Q}, refitted on subsets of trials that exclude the end of reward consumption:
    'no licking' = no licking in pre-Q nor Q; 'pre-Q inside trial' = the trial started >= 1 s before the stimulus
    (wheel resets), so pre-Q is restless time inside the trial, >= 1.5 s + 1 s after the previous feedback."""
    warnings.filterwarnings('ignore')
    rng = np.random.default_rng(seed)
    rows = []
    try:
        for f in ns.load_session(eid, mask=False):
            d = epoch_table(f)
            pre = d[d['epoch'].isin(['pre-Q', 'Q'])]
            licked = pre.groupby('trial')['lick rate'].max() > ACTIVE
            d['no lick'] = ~d['trial'].map(licked).fillna(True).astype(bool)
            for name, keep in CONTROLS.items():
                for ep in ['pre-Q', 'Q']:
                    r = fit_context(d[keep(d)], 'prev outcome', ep, rng)
                    if r is not None:
                        rows.append({**r, 'control': name})
    except Exception as ex:
        print(f'{eid}: {type(ex).__name__}: {ex}', flush=True)
    return rows


def run_controls(n_jobs=12):
    from joblib import Parallel, delayed
    eids = ns.load_recordings()['eid'].unique()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(control_session)(e) for e in eids)
    tab = pd.DataFrame([r for s in res for r in s])
    tab.to_pickle(OUT / f'prev_outcome_controls_{ns.RUN}.pkl')
    return tab


def run(n_jobs=12):
    from joblib import Parallel, delayed
    OUT.mkdir(exist_ok=True)
    eids = ns.load_recordings()['eid'].unique()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(analyse_session)(e) for e in eids)
    tab = pd.DataFrame([r for s in res for r in s])
    tab.to_pickle(OUT / f'context_coupling_{ns.RUN}.pkl')
    return tab


def load():
    tab = pd.read_pickle(OUT / f'context_coupling_{ns.RUN}.pkl')
    tab['excess'] = tab['gain'] - tab['null'].apply(np.mean)
    return tab


def across_mice(tab, column, by=('context', 'epoch', 'target_NM')):
    """Mouse means of a column, then per group: mean, sem, n positive, one-sample t-test p."""
    from scipy.stats import ttest_1samp
    m = tab.groupby([*by, 'subject'])[column].mean().dropna().reset_index()
    rows = []
    for k, g in m.groupby(list(by)):
        x = g[column]
        rows.append({**dict(zip(by, k)), 'mice': len(x), 'mean': x.mean(), 'sem': x.sem(),
                     'positive': f'{(x > 0).sum()}/{len(x)}', 'p': ttest_1samp(x, 0).pvalue if len(x) > 2 else np.nan})
    return pd.DataFrame(rows), m


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    run_controls(n) if len(sys.argv) > 2 and sys.argv[2] == 'controls' else run(n)
