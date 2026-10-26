"""
DDM-HMM STATES: the collaborator's drift-diffusion + hidden-Markov fits on the neuromodulator sessions
======================================================================================================
Reads the `hmm/ddm-k{K}` group of the exported session files (neuromodulators/pose_qc_hmm_k4, written by
github.com/davcrom/ibl_neuromodulators scripts/import_ddm_hmm*.py) through the archive's own `h5_loaders`,
and puts the two fit formats on one convention.

USE IT (from any folder in the repo):

    import sys, pathlib
    _p = pathlib.Path.cwd().resolve()
    while not (_p / 'Functions' / 'ddm_hmm.py').exists() and _p != _p.parent:
        _p = _p.parent
    sys.path.insert(0, str(_p))
    from Functions import ddm_hmm as dh

    trials, params = dh.load_all(k=4)    # one row per fitted trial / one row per (mouse, state)

The model (per mouse, all of its sessions pooled): each hidden state is a DDM with
    B      bound
    k      drift-rate gain (drift = k * signed contrast ** alpha)
    a0     starting point as a fraction of the bound (0.5 = unbiased)
    tau    non-decision time (s)
    alpha  contrast exponent, shared across states
and a Markov chain over states from trial to trial. `trans_to_j[i]` = P(state j on t+1 | state i on t).

The two formats, as stored:
- new (`kind` attr present): K DDM states + one 'omission' state (last; its DDM params are NaN);
  trials carry viterbi_state, p_state_1..K+1, omission, c (|contrast| fraction), s (stimulus side +-1),
  choice SIGN-FLIPPED vs IBL, rt = response_times - stimOnTrigger_times. EM ran to its 1000-iteration cap
  (converged=False) in every fit.
- legacy (first-round fits, no `kind`): K DDM states, no omission state; map_state instead of viterbi_state;
  choice has the IBL sign; rt = response_times - stimOn_times; no-go and long-RT trials were dropped by the fit;
  run-summary attrs are mostly NaN and tau = 0.

What load_all returns uses neither format's own task columns: task variables come from the session's
trials table (IBL conventions), and the fit contributes only `state`, `p_state_*`, and `omission`.
    choice      IBL: -1 = rightward (CCW), +1 = leftward, 0 = no-go
    choice_right  True when the mouse chose right (choice == -1); NaN on no-go
    rt          response_times - stimOnTrigger_times (what the new fits model), for every trial
    state       Viterbi (new) / MAP (legacy) state, 1..K; omission-state trials keep their state number K+1
    is_omission True on trials assigned to the omission state (new format only)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / 'neuromodulators' / 'pose_qc_hmm_k4'

PARAMS = ['B', 'k', 'a0', 'tau', 'alpha']


def _loaders(archive=ARCHIVE):
    if str(archive) not in sys.path:
        sys.path.insert(0, str(archive))
    import h5_loaders
    return h5_loaders


def fitted_eids(k=4, archive=ARCHIVE):
    """eids whose file holds hmm/ddm-k{k}."""
    out = []
    for f in sorted((archive / 'sessions').glob('*.h5')):
        with h5py.File(f, 'r') as h5:
            if f'hmm/ddm-k{k}' in h5:
                out.append(f.stem)
    return out


def available_k(archive=ARCHIVE):
    """Every K stored in any session file."""
    ks = set()
    for f in (archive / 'sessions').glob('*.h5'):
        with h5py.File(f, 'r') as h5:
            ks |= {int(name.removeprefix('ddm-k')) for name in h5.get('hmm', {})}
    return sorted(ks)


def _session(eid, k, L):
    fit = L.load_hmm(eid, k)
    a, ft = fit['attrs'], fit['trials']
    legacy = 'kind' not in a
    t = L.load_trials(eid)
    p_cols = [c for c in ft.columns if c.startswith('p_state_')]
    keep = ft[['trial', *p_cols]].copy()
    keep['state'] = (ft['map_state'] if legacy else ft['viterbi_state']).astype(int).to_numpy()
    keep['omission'] = False if legacy else ft['omission'].astype(bool).to_numpy()
    out = t.merge(keep, on='trial', how='inner')
    out['eid'] = eid
    out['subject'] = a['subject']
    out['format'] = 'legacy' if legacy else 'new'
    return out, a


def load_all(k=4, archive=ARCHIVE):
    """(trials, params) for every session holding a K-state fit.

    trials: the session trials table joined on `trial` to the fit's states (inner join, so legacy fits'
    dropped trials are absent), plus the convention columns described in the module docstring, `session_idx`
    (chronological session number within the archive's sessions of that mouse), and `trial_frac`
    (position within session, 0-1).
    params: one row per (subject, state), DDM params + self-transition + init + `kind`; the transition matrix
    of each mouse in `params.attrs['trans'][subject]` (rows = from, cols = to).
    """
    L = _loaders(archive)
    rec = pd.read_csv(archive / 'recordings.csv')
    frames, attrs = [], {}
    for eid in fitted_eids(k, archive):
        f, a = _session(eid, k, L)
        frames.append(f)
        attrs.setdefault(a['subject'], a)
    trials = pd.concat(frames, ignore_index=True)

    meta = rec.drop_duplicates('eid').set_index('eid')
    nm = rec.groupby('eid')['target_NM'].agg(lambda x: '+'.join(sorted(set(x))))
    trials['target_NM'] = trials['eid'].map(nm)
    trials['session_type'] = trials['eid'].map(meta['session_type'])
    trials['start_time'] = pd.to_datetime(trials['eid'].map(meta['start_time']))
    order = (trials.drop_duplicates('eid').sort_values('start_time')
             .assign(session_idx=lambda d: d.groupby('subject').cumcount()).set_index('eid')['session_idx'])
    trials['session_idx'] = trials['eid'].map(order)
    trials = trials.sort_values(['subject', 'session_idx', 'trial']).reset_index(drop=True)
    trials['trial_frac'] = trials.groupby('eid')['trial'].transform(lambda x: (x - x.min()) / max(x.max() - x.min(), 1))

    trials['choice_right'] = np.where(trials['choice'] == 0, np.nan, (trials['choice'] == -1).astype(float))
    trials['correct'] = (trials['feedbackType'] == 1).astype(float)
    trials['rt'] = trials['response_times'] - trials['stimOnTrigger_times']
    trials['rt_move'] = trials['firstMovement_times'] - trials['stimOnTrigger_times']
    trials['is_omission'] = trials['omission']
    trials = trials.drop(columns='omission')

    rows, trans = [], {}
    for subj, a in attrs.items():
        states = np.asarray(a['state']).astype(int)
        T = np.array([a[f'trans_to_{s}'] for s in states]).T   # T[i, j] = P(j | i)
        trans[subj] = pd.DataFrame(T, index=states, columns=states)
        kind = (np.asarray(a['kind']).astype(str) if 'kind' in a else np.array(['ddm'] * len(states)))
        for i, s in enumerate(states):
            rows.append({'subject': subj, 'state': int(s), 'kind': kind[i],
                         **{p: float(np.asarray(a[p])[i]) for p in PARAMS},
                         'self_trans': T[i, i], 'init': float(np.asarray(a['init'])[i]),
                         'format': 'legacy' if 'kind' not in a else 'new',
                         'converged': a.get('converged'), 'iterations': a.get('iterations'),
                         'n_sessions_fit': a.get('n_sessions'), 'n_trials_fit': a.get('n_trials')})
    params = pd.DataFrame(rows)
    params.attrs['trans'] = trans
    return trials, params


def check_fit_tables(trials, params):
    """Sanity checks: posterior rows sum to 1, viterbi state has non-negligible posterior, one fit per mouse."""
    p = trials.filter(like='p_state_')
    out = {'p_sum_range': (np.nanmin(p.sum(1)), np.nanmax(p.sum(1)))}
    post_of_state = p.to_numpy()[np.arange(len(trials)), trials['state'].to_numpy() - 1]
    out['median_posterior_of_assigned_state'] = float(np.nanmedian(post_of_state))
    out['mice'] = params['subject'].nunique()
    return out


# ---- naming states across mice (2_ddm_hmm_states.ipynb) -------------------------------------------------
# Provisional thresholds set by eye on the K=4 fits; applied in this order.
K_DISENGAGED, A0_BIAS, ACC_ENGAGED, RT_SLOW = 0.3, 0.2, 0.9, 0.8
LABEL_COLORS = {'engaged': '#0072B2', 'slow': '#56B4E9', 'impulsive': '#009E73', 'biased-L': '#E69F00',
                'biased-R': '#CC79A7', 'disengaged': '#999999'}   # Okabe-Ito


def name_state(r):
    """r: a row with k, a0, acc_easy (accuracy at contrast >= 25%), rt_median (s)."""
    if r['k'] < K_DISENGAGED:
        return 'disengaged'
    if r['a0'] - 0.5 > A0_BIAS:
        return 'biased-R'
    if 0.5 - r['a0'] > A0_BIAS:
        return 'biased-L'
    if r['rt_median'] >= RT_SLOW:
        return 'slow'
    return 'engaged' if r['acc_easy'] >= ACC_ENGAGED else 'impulsive'


def go_trials(trials):
    """Go trials in DDM states (omission state / no-go dropped)."""
    return trials[~trials['is_omission'] & (trials['choice'] != 0)]


def state_labels(trials, params):
    """One row per (subject, DDM state): the features name_state reads, the label, and a 'tag' ('s2 engaged')."""
    d = go_trials(trials)
    S = d.groupby(['subject', 'state']).agg(
        n=('trial', 'size'),
        acc_easy=('correct', lambda x: x[d.loc[x.index, 'contrast'] >= 25].mean()),
        rt_median=('rt', 'median')).reset_index()
    S = S.merge(params.loc[params['kind'] == 'ddm', ['subject', 'state', 'B', 'k', 'a0', 'tau']],
                on=['subject', 'state'])
    S['label'] = S.apply(name_state, axis=1)
    S['tag'] = 's' + S['state'].astype(str) + ' ' + S['label']
    return S
