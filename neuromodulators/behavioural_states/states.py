"""
Latent-state models of the NM sessions on one trial table: GLM-HMM (choice only) and DDM-HMM (choice + RT).
Functions only; analyses are in the notebooks of this folder (see README.md).

    import sys, pathlib
    _p = pathlib.Path.cwd().resolve()
    while not (_p / 'Functions' / 'nm_syllables.py').exists() and _p != _p.parent:
        _p = _p.parent
    sys.path.insert(0, str(_p)); sys.path.insert(0, str(_p / 'neuromodulators' / 'behavioural_states'))
    import states as st

    T = st.aligned()      # one row per trial of the DDM-HMM sessions: GLM-HMM + DDM-HMM state + task columns

GLM-HMM: the paper's K=2 model fitted per mouse on the export's sessions (neuromodulators/
trial_position_engagement.py glmhmm), p_engaged per trial, every trial (no-gos masked in the fit, posterior kept).
DDM-HMM: the collaborator's K=4 fits (Functions/ddm_hmm.py), states named by dh.name_state, one row per fitted
trial; trials in the omission state get the label 'omission'.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from Functions import ddm_hmm as dh  # noqa: E402
from Functions import nm_syllables as ns  # noqa: E402

GLMHMM_FILE = ns.DATA / 'trial_position_engagement' / 'glmhmm_k2_per_mouse.pqt'
KIND_ORDER = ['engaged', 'slow', 'impulsive', 'biased-L', 'biased-R', 'disengaged', 'omission']
KIND_COLORS = {**dh.LABEL_COLORS, 'omission': '#000000'}


def load_glmhmm():
    """eid, subject, trial, p_engaged for every trial of every export session."""
    return pd.read_parquet(GLMHMM_FILE)


def load_ddm(k=4):
    """(trials with 'kind' per trial, state table from dh.state_labels; transition matrices in S.attrs['trans'])."""
    trials, params = dh.load_all(k)
    S = dh.state_labels(trials, params)
    S.attrs['trans'] = params.attrs['trans']   # per mouse, rows = from, cols = to (omission state included)
    kind = S.set_index(['subject', 'state'])['label']
    trials['kind'] = [kind.get((s, x), 'omission') for s, x in zip(trials['subject'], trials['state'])]
    trials.loc[trials['is_omission'], 'kind'] = 'omission'
    return trials, S


def aligned(k=4, threshold=0.5):
    """DDM-HMM trials joined to the GLM-HMM posterior on (eid, trial). glm_state = 'engaged' / 'lapse'."""
    trials, _ = load_ddm(k)
    g = load_glmhmm()[['eid', 'trial', 'p_engaged']]
    T = trials.merge(g, on=['eid', 'trial'], how='inner')
    T['glm_state'] = np.where(T['p_engaged'] >= threshold, 'engaged', 'lapse')
    T['repeat'] = T.groupby('eid')['choice'].shift(1).eq(T['choice']).astype(float)
    T.loc[T['choice'] == 0, 'repeat'] = np.nan
    return T


def crosstab(T, by='subject'):
    """Per mouse: P(DDM kind | GLM state) and P(GLM state | DDM kind), long tables."""
    a = (T.groupby([by, 'glm_state'])['kind'].value_counts(normalize=True).rename('p').reset_index())
    b = (T.groupby([by, 'kind'])['glm_state'].value_counts(normalize=True).rename('p').reset_index())
    return a, b


def conditional_entropy(T, given, target, by='subject'):
    """Per mouse: H(target), H(target | given) in bits, and the fraction of H(target) left."""
    def H(p):
        p = p[p > 0]
        return -(p * np.log2(p)).sum()
    rows = []
    for m, g in T.groupby(by):
        h = H(g[target].value_counts(normalize=True).to_numpy())
        hc = sum(len(x) / len(g) * H(x[target].value_counts(normalize=True).to_numpy()) for _, x in g.groupby(given))
        rows.append({by: m, f'H({target})': h, f'H({target} | {given})': hc, 'fraction left': hc / h if h else np.nan})
    return pd.DataFrame(rows)


def acf(x, eids, lags):
    """Autocorrelation of x at each lag, from pairs of trials within the same session (NaNs skipped)."""
    x = x - np.nanmean(x)
    v = np.nanvar(x)
    out = []
    for lag in lags:
        num = n = 0
        for e in np.unique(eids):
            y = x[eids == e]
            if len(y) > lag:
                p = y[:-lag] * y[lag:]
                num += np.nansum(p)
                n += np.isfinite(p).sum()
        out.append(num / n / v)
    return np.array(out)


def implied_acf(trans, f, var_total, lags):
    """Autocorrelation of a trial variable that the Markov chain alone implies: its state means f vary along the
    chain and trials are independent given the state, so Cov(lag) = sum_ij pi_i (T^lag)_ij f_i f_j - (pi . f)^2."""
    T = np.asarray(trans)
    w, V = np.linalg.eig(T.T)
    pi = np.real(V[:, np.argmax(np.real(w))])
    pi /= pi.sum()
    f = np.asarray(f) - pi @ np.asarray(f)
    return np.array([(pi[:, None] * np.linalg.matrix_power(T, lag) * np.outer(f, f)).sum() for lag in lags]) / var_total


def temporal_check(trials, trans, lags, var='logRT'):
    """Per mouse and lag: autocorrelation of a trial variable (log RT or correct, contrast means removed; go trials
    in DDM states) in the data, implied by the fitted chain, and left after subtracting the posterior-expected
    state mean. If the chain captures the temporal structure, data = implied and the residual is ~0."""
    rows = []
    for m, g in trials.groupby('subject'):
        g = g.sort_values(['session_idx', 'trial'])
        states = trans[m].index.to_numpy()
        go = ~g['is_omission'] & (g['choice'] != 0) & (g['rt'] > 0)
        y = (np.log(g['rt']) if var == 'logRT' else g['correct']).where(go)
        y = (y - y.groupby(g['contrast'].abs()).transform('mean')).to_numpy(dtype=float)
        z = g['state'].to_numpy()
        f = np.array([np.nanmean(y[z == s]) if np.isfinite(y[z == s]).any() else np.nanmean(y) for s in states])
        f -= np.nanmean(y)
        post = np.nansum(g[[f'p_state_{s}' for s in states]].to_numpy() * f, 1)
        eids = g['eid'].to_numpy()
        for lag, a, b, c in zip(lags, acf(y, eids, lags), implied_acf(trans[m], f, np.nanvar(y), lags),
                                acf(y - post, eids, lags)):
            rows.append({'subject': m, 'lag': lag, 'data': a, 'implied by chain': b, 'left after states': c})
    return pd.DataFrame(rows)
