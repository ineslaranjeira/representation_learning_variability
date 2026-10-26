"""
Is trial-mode entropy just a consequence of where a mouse sits on LD1?
=====================================================================
Mode-use entropy tracks LD1 across mice (trial_mode_entropy.ipynb). LD1 is a linear read-out of mean
syllable usage, and the modes are built from the same syllables, so the entropy effect might follow
from the shift in AVERAGE mode usage along LD1, with nothing extra. Test: predict each mouse's
mode-usage profile from its LD1 alone (leave that mouse out), take the entropy of the predicted
profile, and compare with the observed entropy. If the mean shift is the whole story, the predicted
entropy reproduces the entropy-LD1 slope and the residual (observed - predicted) is flat along LD1.

Estimators, so observed and predicted are on the same footing:
  * observed H_modes: as trial_mode_entropy / fig_engagement_syllables G: per session, plug-in entropy
    of N_TRIALS trials subsampled without replacement (N_SUB draws), / log K; mean over the mouse's sessions.
  * predicted: plug-in entropy expected from N_TRIALS multinomial draws of the predicted profile
    (simulated), / log K, so it carries the same small-sample bias.
  * H_pooled: the same for the mouse's own mean profile (sessions pooled). H_pooled - H_modes is the
    part of the entropy that comes from sessions of one mouse using DIFFERENT modes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'GLM-HMM' / 'transition_prediction'))
import tp_data as td   # noqa: E402

N_SUB = 200       # subsamples per session (as trial_mode_entropy)
N_SIM = 200       # multinomial draws per predicted profile
EPS = 1e-4        # floor for the log-linear model's log(usage)
SEED = 0


def norm_entropy(p, k):
    p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(k)


def load_usage(lda=None, trials_file=None):
    """(U, meta, n_trials): per-session mode usage fractions (sessions x modes; trials without a
    mode left out), LDA rows for those sessions, and the smallest per-session trial count."""
    lda = td.load_lda() if lda is None else lda
    t = pd.read_parquet(trials_file or td.TRIAL_MODES_FILE, columns=['session', 'trial_cluster'])
    t = t.dropna(subset=['trial_cluster'])
    t = t[t['session'].isin(lda['session'])]
    U = pd.crosstab(t['session'], t['trial_cluster'], normalize='index')
    meta = lda.set_index('session').loc[U.index]
    return U, meta, int(t.groupby('session').size().min()), t


def observed_entropy(trials, k, n_trials, n_sub=N_SUB, seed=SEED):
    """Per session: mean plug-in entropy over n_sub subsamples of n_trials trials (no replacement)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, lab in trials.groupby('session')['trial_cluster']:
        lab = lab.to_numpy()
        out[s] = np.mean([norm_entropy(np.unique(rng.choice(lab, n_trials, replace=False), return_counts=True)[1]
                                       / n_trials, k) for _ in range(n_sub)])
    return pd.Series(out, name='H_sess')


def expected_entropy(p, k, n_trials, n_sim=N_SIM, rng=None):
    """Plug-in entropy expected from n_trials multinomial draws of profile p."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    return np.mean([norm_entropy(c / n_trials, k) for c in rng.multinomial(n_trials, p / p.sum(), size=n_sim)])


def predict_profiles(P, X, model='log'):
    """Leave-one-mouse-out prediction of usage profiles P (mice x modes) from covariates X (mice x d).
    'linear': each mode's usage linear in X, clipped at 0. 'log': log usage linear in X (a log-linear /
    softmax-like model, always positive). Rows renormalised to sum to 1."""
    n = len(P)
    Xc = np.column_stack([np.ones(n), X])
    out = np.zeros_like(P)
    for i in range(n):
        tr = np.arange(n) != i
        if model == 'linear':
            p = np.clip(Xc[i] @ np.linalg.lstsq(Xc[tr], P[tr], rcond=None)[0], 1e-6, None)
        else:
            p = np.exp(Xc[i] @ np.linalg.lstsq(Xc[tr], np.log(P[tr] + EPS), rcond=None)[0])
        out[i] = p / p.sum()
    return out


def profile_r2(P, P_hat):
    return 1 - ((P - P_hat) ** 2).sum() / ((P - P.mean(0)) ** 2).sum()
