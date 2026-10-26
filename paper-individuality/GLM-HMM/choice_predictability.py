"""
How predictable is each choice, per session? (no hidden states)
================================================================
The K=2 GLM-HMM switch count tracks LD1, but that count is set by scattered single-trial
deviations, not by their order (the count survives shuffling trial order within a session). This
module measures the same thing without an HMM: the cross-entropy of a session's choices, in
bits/trial, under a logistic GLM on the GLM-HMM's own inputs (stimulus, previous choice, WSLS, bias).

    ce_engaged   under the mouse's ENGAGED-state weights (glmhmm_evidence/mouse_params_*.csv): how far
                 the session's choices stray from the mouse's own engaged policy. Those weights were
                 fit on all of the mouse's sessions, this one included (4 weights on ~3000 trials).
    ce_glm       under a plain GLM fit on the mouse's OTHER sessions (leave one session out): how
                 unpredictable the choices are given the mouse's average policy. Held out.

Each is also split by trial type: easy (|contrast| >= 25 %), hard (6.25 / 12.5 %) and zero contrast.
On easy trials the cross-entropy is mostly lapses; on hard trials it also carries psychometric slope.

For the link to syllable use, two session-level entropies with the estimators of
fig_engagement_syllables.ipynb (panels G and I): trial-mode entropy H_modes and pooled syllable
entropy H_syl, both normalised and averaged over equal-trial subsamples.

Inputs as fig_engagement: merged_behavioral_and_states.pqt (choice reconstructed from side and reward,
exact; see session_glmhmm_em / lda_predicts_session_glmhmm), each session's first N_TRUNC trials, the
stimulus z-scored over the mouse's pooled sessions. Sessions with an LD1 in tp_data.LDA_FILE.

Plus, per mouse, a psychometric fit with separate lapse and slope parameters (psychometric_table), and,
per session, the mean log reaction time (rt_table), to ask whether the choice effect is lapses rather
than sensitivity, and whether it is separate from the RT level.

Usage (iblenv):  python choice_predictability.py      # writes choice_predictability/*.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'transition_prediction'))
sys.path.insert(0, str(HERE.parents[1] / 'Functions'))
from session_glmhmm_em import log_likelihoods, PRIOR_SIGMA, INPUT_NAMES          # noqa: E402
from lda_predicts_session_glmhmm import build_session_covariates, raw_signed_contrast  # noqa: E402
import tp_data as td                                                                 # noqa: E402

OUT = HERE / 'choice_predictability'
STATES_PATH = HERE / 'merged_behavioral_and_states.pqt'
MOUSE_PARAMS = HERE / 'glmhmm_evidence' / 'mouse_params_lda_syllable_mice.csv'
N_TRUNC = 400                 # as fig_engagement: every session's first 400 trials
TRIAL_TYPES = {'easy': lambda c: c >= 0.25, 'hard': lambda c: (c > 0) & (c < 0.25), 'zero': lambda c: c == 0}
N_SUB, N_SUB_SYL = 200, 50    # entropy subsamples per session (as fig_engagement_syllables)
SEED = 0


# ------------------------------------------------------------------ GLM --------

def fit_glm(X, y, prior_sigma=PRIOR_SIGMA):
    """MAP logistic regression in the GLM-HMM's convention, p(y=1) = sigmoid(-x @ w)."""
    def nll(w):
        ll = log_likelihoods(X, y, w[None, :])[:, 0]
        return -ll.sum() + 0.5 * (w ** 2).sum() / prior_sigma ** 2
    return minimize(nll, np.zeros(X.shape[1]), method='L-BFGS-B').x


def bits(X, y, w):
    """Per-trial cross-entropy (bits) of the choices y under weights w."""
    return -log_likelihoods(X, y, np.atleast_2d(w))[:, 0] / np.log(2)


def engaged_weights(params):
    return params[[f'w_eng_{n}' for n in INPUT_NAMES]].to_numpy(float)


def choice_table(states=None, lda=None, params=None, n_trunc=N_TRUNC):
    """One row per session: cross-entropies (bits/trial) overall and per trial type, and easy-trial
    accuracy. Only mice with engaged weights in MOUSE_PARAMS."""
    states = pd.read_parquet(STATES_PATH) if states is None else states
    lda = td.load_lda() if lda is None else lda
    params = pd.read_csv(MOUSE_PARAMS).set_index('animal') if params is None else params
    keep = set(lda['session'])
    rows = []
    for animal, adf in states.groupby('animal', sort=False):
        if animal not in params.index:
            continue
        pooled = raw_signed_contrast(adf)
        stats = (pooled.mean(), pooled.std())
        sess = {e: s.reset_index(drop=True) for e, s in adf.groupby('eid', sort=False)}
        cov = {e: build_session_covariates(s, stim_stats=stats) for e, s in sess.items()}
        w_eng = engaged_weights(params.loc[animal])
        for eid in sess:
            if eid not in keep:
                continue
            others = [e for e in sess if e != eid]
            if not others:
                continue
            w_glm = fit_glm(np.vstack([cov[e][0] for e in others]),
                            np.concatenate([cov[e][1] for e in others]).astype(float))
            X, y = cov[eid][0][:n_trunc], cov[eid][1][:n_trunc].astype(float)
            s = sess[eid].iloc[:n_trunc]
            c = s['signed_contrast'].abs().to_numpy()
            ce = {'engaged': bits(X, y, w_eng), 'glm': bits(X, y, w_glm)}
            row = dict(session=eid, mouse_name=animal, n_trials=len(y),
                       p_easy=(s.loc[c >= 0.25, 'rewarded'] == 1).mean())
            for name, v in ce.items():
                row[f'ce_{name}'] = v.mean()
                for tt, sel in TRIAL_TYPES.items():
                    row[f'ce_{name}_{tt}'] = v[sel(c)].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ entropies --------

def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return -(p * np.log(p)).sum()


def entropy_table(trials=None, lda=None, n_sub=N_SUB, n_sub_syl=N_SUB_SYL, seed=SEED):
    """One row per session: H_modes (trial-mode entropy / log K) and H_syl (pooled syllable entropy /
    log 32), as fig_engagement_syllables panels G and I. Equal-trial subsamples (the shortest
    session's count of trials with a mode)."""
    import trial_modes as tmodes
    trials = pd.read_parquet(td.TRIAL_MODES_FILE) if trials is None else trials
    lda = td.load_lda() if lda is None else lda
    mt = trials.dropna(subset=['trial_cluster']).copy()
    mt = mt[mt['session'].isin(lda['session'])].reset_index(drop=True)
    for e in tmodes.EPOCHS:
        mt[e] = mt[e].map(np.asarray)
    _, seqs = tmodes.trial_matrix_from_trials(mt)
    labels = mt['trial_cluster'].astype(int).to_numpy()
    k_modes = len(np.unique(labels))
    n_codes = 4 * td.N_PAW
    n_trials = int(mt.groupby('session').size().min())
    rng = np.random.default_rng(seed)
    rows = []
    for s, idx in mt.groupby('session').indices.items():
        lab = labels[idx]
        h_modes = np.mean([_entropy(np.bincount(rng.choice(lab, n_trials, replace=False)))
                           for _ in range(n_sub)])
        h_syl = []
        for _ in range(n_sub_syl):
            codes = seqs[idx][rng.choice(len(idx), n_trials, replace=False)]
            h_syl.append(_entropy(np.bincount(codes[~np.isnan(codes)].astype(int), minlength=n_codes)))
        rows.append(dict(session=s, H_modes=h_modes / np.log(k_modes), H_syl=np.mean(h_syl) / np.log(n_codes)))
    return pd.DataFrame(rows)


# --------------------------------------------------------- psychometric --------

BLOCKS = (0.2, 0.5, 0.8)      # probabilityLeft; each block gets its own threshold


def psychometric(c, block_idx, mu, sigma, gamma, lam):
    """P(right) = gamma + (1 - gamma - lam) * Phi((c - mu[block]) / sigma). gamma: lapses to the right
    (on strong-left trials), lam: lapses to the left; total lapse = gamma + lam."""
    from scipy.stats import norm
    return gamma + (1 - gamma - lam) * norm.cdf((c - mu[block_idx]) / sigma)


def fit_psychometric(c, block_idx, y):
    """MLE of (mu per block, sigma, gamma, lambda); contrast c in [-1, 1], y = chose right."""
    n_b = len(BLOCKS)

    def nll(th):
        p = np.clip(psychometric(c, block_idx, th[:n_b], th[n_b], th[n_b + 1], th[n_b + 2]), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    best = None
    for s0 in (0.05, 0.15, 0.4):                       # a few starts: sigma and the lapses trade off
        r = minimize(nll, np.r_[np.zeros(n_b), s0, 0.05, 0.05], method='L-BFGS-B',
                     bounds=[(-1, 1)] * n_b + [(0.005, 2), (0, 0.5), (0, 0.5)])
        best = r if best is None or r.fun < best.fun else best
    th = best.x
    return dict(sigma=th[n_b], gamma=th[n_b + 1], lam=th[n_b + 2], lapse=th[n_b + 1] + th[n_b + 2],
                **{f'mu_{b}': m for b, m in zip(BLOCKS, th[:n_b])})


def psychometric_table(states=None, sessions=None, n_trunc=N_TRUNC):
    """One row per mouse: the psychometric fit to its pooled trials (first n_trunc of each session in
    `sessions`). Choice reconstructed from side and reward, as everywhere here."""
    states = pd.read_parquet(STATES_PATH) if states is None else states
    if sessions is not None:
        states = states[states['eid'].isin(sessions)]
    states = states[states.groupby('eid').cumcount() < n_trunc]
    rows = []
    for animal, d in states.groupby('animal'):
        d = d[d['probabilityLeft'].isin(BLOCKS)]
        y = ((d['contrastLeft'].isna() & (d['rewarded'] == 1))
             | (d['contrastRight'].isna() & (d['rewarded'] == -1))).astype(float).to_numpy()
        block_idx = d['probabilityLeft'].map({b: i for i, b in enumerate(BLOCKS)}).to_numpy()
        rows.append(dict(mouse_name=animal, n_trials=len(y),
                         **fit_psychometric(d['signed_contrast'].to_numpy(), block_idx, y)))
    out = pd.DataFrame(rows)
    out['slope'] = 1 / out['sigma']
    return out


def rt_table(sessions, trial_meta=None, n_trunc=N_TRUNC, rt_window=(0.02, 10)):
    """One row per session: mean and SD of log RT (first movement - go cue) over the first n_trunc
    trials, RT in rt_window (as rt_along_lda1)."""
    path = trial_meta or next(f for f in [HERE.parent / 'data' / 'session_trial_meta_19-08-2026',
                                          HERE.parent / 'data' / 'session_trial_meta_10-07-2026'] if f.exists())
    m = pd.read_parquet(path, columns=['session', 'trial_id', 'reaction'])
    m = m[m['session'].isin(sessions)].sort_values(['session', 'trial_id'])
    m = m[m.groupby('session').cumcount() < n_trunc]
    m = m[m['reaction'].between(*rt_window)].assign(log_rt=lambda x: np.log(x['reaction']))
    return m.groupby('session')['log_rt'].agg(mean_log_rt='mean', sd_log_rt='std').reset_index()


def main():
    OUT.mkdir(exist_ok=True)
    lda = td.load_lda()
    ce = choice_table(lda=lda)
    ce.to_parquet(OUT / 'choice_entropy.parquet')
    print(f'cross-entropies: {len(ce)} sessions, {ce["mouse_name"].nunique()} mice')
    ent = entropy_table(lda=lda)
    ent.to_parquet(OUT / 'behaviour_entropy.parquet')
    print(f'entropies: {len(ent)} sessions')
    psy = psychometric_table(sessions=set(lda['session']))
    psy.to_parquet(OUT / 'psychometric.parquet')
    print(f'psychometric fits: {len(psy)} mice')
    rt = rt_table(set(lda['session']))
    rt.to_parquet(OUT / 'reaction_times.parquet')
    print(f'reaction times: {len(rt)} sessions')


if __name__ == '__main__':
    main()
