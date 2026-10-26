"""
Does movement during the quiescence period predict reaction time? The same per-trial measures for the
neuromodulator (NM) sessions and the brain-wide-map (BWM) sessions of the individuality paper.
Functions only; the analysis is in neuromodulators/behavioural_states/5_quiescence_movement_rt.ipynb.

Per trial, over the final quiescence period (goCue/stimOn - quiescencePeriod .. goCue/stimOn; wheel held still):
  whisking  mean log(1 + whisker ME)
  paw       mean log(1 + left-paw speed)
  wheel     mean log(1 + |wheel velocity|)
  lick      mean lick count per bin
  RT        firstMovement - goCue (BWM 'reaction') / firstMovement - stimOnTrigger (NM); log, RT > MIN_RT only
Per session, OLS of log RT on the four signals (z-scored within session) with |contrast| level, previous outcome and
trial position (+ squared) held fixed ('joint'), and on each signal alone with the same covariates ('alone').
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / 'paper-individuality' / 'data'
BWM_STATES = DATA / 'states_files'
CACHE = DATA / 'quiescence_rt'
SIGNALS = ['whisking', 'paw', 'wheel', 'lick']
MIN_RT = 0.05
MIN_TRIALS = 100


# ---- BWM ---------------------------------------------------------------------------------------------------
def bwm_session(path):
    """Per-trial quiescence means and task columns of one BWM states file (60 Hz bins, epoch labels)."""
    warnings.filterwarnings('ignore')
    cols = ['Bin', 'Lick count', 'avg_wheel_vel', 'whisker_me', 'l_paw_x', 'l_paw_y', 'trial_id', 'label',
            'correct', 'contrast', 'reaction', 'mouse_name', 'session']
    d = pd.read_parquet(path, columns=cols)
    fs = 1 / np.median(np.diff(d['Bin'].to_numpy()))
    d['paw'] = np.hypot(np.gradient(d['l_paw_x'].to_numpy()), np.gradient(d['l_paw_y'].to_numpy())) * fs
    d['whisking'] = np.log1p(d['whisker_me'].abs())
    d['paw'] = np.log1p(d['paw'])
    d['wheel'] = np.log1p(d['avg_wheel_vel'].abs())
    d['lick'] = d['Lick count']
    q = d[d['label'] == 'Quiescence'].dropna(subset=['trial_id'])
    per = q.groupby('trial_id')[SIGNALS].mean()   # NaN bins (tracking dropouts) are skipped, not interpolated
    task = d.dropna(subset=['trial_id']).groupby('trial_id')[['correct', 'contrast', 'reaction']].first()
    out = task.join(per, how='inner').reset_index()
    out['rt'] = out['reaction']
    out['subject'], out['eid'] = d['mouse_name'].iat[0], d['session'].iat[0]
    return out.drop(columns='reaction')


def bwm_table(n_jobs=12, cache=True):
    path = CACHE / 'bwm_trials.pqt'
    if cache and path.exists():
        return pd.read_parquet(path)
    from joblib import Parallel, delayed
    files = sorted(BWM_STATES.glob('*_states_file_*'))
    T = pd.concat(Parallel(n_jobs=n_jobs)(delayed(bwm_session)(f) for f in files), ignore_index=True)
    T = T.drop_duplicates(['eid', 'trial_id'])
    CACHE.mkdir(exist_ok=True)
    T.to_parquet(path)
    return T


EPOCH_NAMES = {'Stimulus right': 'stimulus', 'Stimulus left': 'stimulus', 'Stimulus zero': 'stimulus',
               'Right choice': 'choice', 'Left choice': 'choice', 'Quiescence': 'quiescence',
               'Pre-quiescence': 'pre-quiescence', 'ITI': 'ITI'}


def bwm_whisk_epochs_session(path):
    """Per trial: fraction of bins in the whisking syllable (2nd digit of identifiable_states = 1) per epoch and over
    the whole trial, lick-state fraction over the trial, and RT."""
    warnings.filterwarnings('ignore')
    d = pd.read_parquet(path, columns=['identifiable_states', 'label', 'trial_id', 'reaction', 'mouse_name', 'session'])
    d = d.dropna(subset=['trial_id'])
    d = d[d['label'].isin(EPOCH_NAMES)]
    s = d['identifiable_states'].astype(str)
    d['whisk'] = pd.to_numeric(s.str[1], errors='coerce')
    d['lick state'] = pd.to_numeric(s.str[2], errors='coerce')
    d['epoch'] = d['label'].map(EPOCH_NAMES)
    per = d.pivot_table(index='trial_id', columns='epoch', values='whisk', aggfunc='mean')
    per['whole trial'] = d.groupby('trial_id')['whisk'].mean()
    per['licking, whole trial'] = d.groupby('trial_id')['lick state'].mean()
    per['rt'] = d.groupby('trial_id')['reaction'].first()
    per['subject'], per['eid'] = d['mouse_name'].iat[0], d['session'].iat[0]
    return per.reset_index()


def bwm_whisk_epochs(n_jobs=12, cache=True):
    path = CACHE / 'bwm_whisk_epochs.pqt'
    if cache and path.exists():
        return pd.read_parquet(path)
    from joblib import Parallel, delayed
    files = sorted(BWM_STATES.glob('*_states_file_*'))
    T = pd.concat(Parallel(n_jobs=n_jobs)(delayed(bwm_whisk_epochs_session)(f) for f in files), ignore_index=True)
    T = T.drop_duplicates(['eid', 'trial_id'])
    T.to_parquet(path)
    return T


def levels(T, columns):
    """Spearman r of each column with log RT at three levels: between mice (mouse means), across sessions within a
    mouse (mice with >= 3 sessions, mean r), across trials within a session (mouse mean of session r)."""
    from scipy.stats import spearmanr, ttest_1samp
    T = T[T['rt'] > MIN_RT].assign(logrt=lambda d: np.log(d['rt']))
    S = T.groupby(['subject', 'eid'])[columns + ['logrt']].mean().reset_index()
    M = S.groupby('subject')[columns + ['logrt']].mean()
    rows = []
    for c in columns:
        bm = spearmanr(M[c], M['logrt'], nan_policy='omit')
        wm = S.groupby('subject').apply(lambda g: spearmanr(g[c], g['logrt'])[0] if len(g) >= 3 else np.nan).dropna()
        ws = T.groupby('eid').apply(lambda g: spearmanr(g[c], g['logrt'], nan_policy='omit')[0]
                                    if g[c].notna().sum() > 50 else np.nan).dropna()
        ws = ws.groupby(T.groupby('eid')['subject'].first()).mean().dropna()
        rows.append({'whisking during': c, 'between mice': bm[0], 'p (mice)': bm[1],
                     'sessions within mouse': wm.mean(), 'p (sessions)': ttest_1samp(wm, 0).pvalue,
                     'trials within session': ws.mean(), 'p (trials)': ttest_1samp(ws, 0).pvalue})
    return pd.DataFrame(rows).set_index('whisking during')


# ---- NM ----------------------------------------------------------------------------------------------------
def nm_session(eid):
    """Per-trial quiescence means of the NM session (first fiber's trials), from the photometry-aligned snippets."""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(REPO / 'neuromodulators' / 'behavioural_states'))
    import prestim as pp
    from Functions import nm_syllables as ns
    f = ns.load_session(eid, mask=False)[0]
    d = pp.trial_features(f, split=True)
    names = {'whisker ME (Q)': 'whisking', 'paw speed (Q)': 'paw', 'wheel speed (Q)': 'wheel', 'lick rate (Q)': 'lick'}
    out = d[['eid', 'subject', 'target_NM', 'trial', 'correct', 'contrast', 'rt', *names]].rename(columns=names)
    out['contrast'] = out['contrast'] / 100
    return out.rename(columns={'trial': 'trial_id'})


def nm_table(n_jobs=12, cache=True):
    path = CACHE / 'nm_trials.pqt'
    if cache and path.exists():
        return pd.read_parquet(path)
    from joblib import Parallel, delayed
    sys.path.insert(0, str(REPO))
    from Functions import nm_syllables as ns
    eids = ns.load_recordings()['eid'].unique()
    T = pd.concat(Parallel(n_jobs=n_jobs)(delayed(nm_session)(e) for e in eids), ignore_index=True)
    CACHE.mkdir(exist_ok=True)
    T.to_parquet(path)
    return T


# ---- model -------------------------------------------------------------------------------------------------
def session_betas(g):
    """Standardised β of each signal on log RT, joint and alone, with the task covariates held fixed."""
    g = g.sort_values('trial_id')
    g = g.assign(prev_correct=g['correct'].shift(1), trial_frac=g['trial_id'] / g['trial_id'].max())
    g = g[(g['rt'] > MIN_RT)].dropna(subset=['rt', 'prev_correct'])
    live = [s for s in SIGNALS if g[s].notna().all() and g[s].std() > 0]
    if len(g) < MIN_TRIALS or 'whisking' not in live:
        return None
    y = np.log(g['rt'].to_numpy())
    C = pd.get_dummies(g['contrast'].round(4), drop_first=True, dtype=float)
    C['prev'] = g['prev_correct'].to_numpy()
    C['pos'] = g['trial_frac'].to_numpy()
    C['pos2'] = C['pos'] ** 2
    Z = (g[live] - g[live].mean()) / g[live].std()

    def ols(X):
        A = np.column_stack([np.ones(len(y)), X])
        return np.linalg.lstsq(A, (y - y.mean()) / y.std(), rcond=None)[0][1:]
    row = {'subject': g['subject'].iat[0], 'eid': g['eid'].iat[0], 'n': len(g)}
    bj = ols(np.column_stack([C.to_numpy(), Z.to_numpy()]))[-len(live):]
    for s, b in zip(live, bj):
        row[f'joint {s}'] = b
    for s in live:
        row[f'alone {s}'] = ols(np.column_stack([C.to_numpy(), Z[s].to_numpy()]))[-1]
    # quintiles of whisking: mean residual log RT (task covariates removed)
    r = y - np.column_stack([np.ones(len(y)), C.to_numpy()]) @ np.linalg.lstsq(
        np.column_stack([np.ones(len(y)), C.to_numpy()]), y, rcond=None)[0]
    qbin = pd.qcut(g['whisking'].rank(method='first'), 5, labels=False).to_numpy()
    row['whisking quintile RT'] = np.array([r[qbin == k].mean() for k in range(5)])
    return row


def betas(T):
    rows = [session_betas(g) for _, g in T.groupby('eid')]
    return pd.DataFrame([r for r in rows if r is not None])


def across_mice(B):
    """Mouse means of each β, then mean, n positive and t-test across mice."""
    from scipy.stats import ttest_1samp
    cols = [c for c in B.columns if c.startswith(('joint', 'alone'))]
    M = B.groupby('subject')[cols].mean()
    return pd.DataFrame({c: {'mean β': M[c].mean(), 'positive': f'{(M[c] > 0).sum()}/{M[c].notna().sum()}',
                             'p': ttest_1samp(M[c].dropna(), 0).pvalue} for c in cols}).T
