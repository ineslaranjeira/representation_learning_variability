"""
Data for "do syllables predict GLM-HMM state transitions?"
===========================================================
Builds one row per trial with (a) the GLM-HMM state and transition targets, (b) task history
for a baseline model, (c) per-epoch syllable usage, (d) the session's LD1, and turns it into a
lagged design matrix.

SOURCES (all already on disk; nothing is refit here)
  states     GLM-HMM/merged_behavioral_and_states.pqt   K=2, the fit load_states.ipynb uses
             GLM-HMM/k2_k3_pilot/all_k_posteriors.parquet   K=2/3/4 refits (54 mice, >=3 sessions)
  syllables  clustering/data_files/8_k_10_bin_syllables_02-10-2026
             one row per (trial, epoch), 10 bins; code = paw + 8 * (whisk + 2 * lick)
  LD1        clustering/data_files/mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026

ALIGNMENT. A session's Nth row in the states parquet is trial_id N in the syllable file:
correct / |contrast| agree on 100% of trials of all 312 shared sessions, block on 97%.
`check_alignment` recomputes this; build_trial_table refuses to run if it drops.

PAW NUMBERING. The 02-10-2026 syllables use the 'uniform' paw states under the 19Aug2026
relabelling; paper_style.paw_codes undoes it, so paw k below is uniform state k
(0 stillest ... 7 most vigorous; 2/5 left, 3/6 right). See PAW_NAMES.

THE TARGET. Transitions are flips of the most probable state (argmax of the smoothed posterior)
between trial t-1 and t. A transition is modelled as a HAZARD: conditional on the state at t-1,
does the animal leave it at t (or within HORIZON trials)? Conditioning matters: without it a
model can "predict transitions" just by recognising the state (disengaged bouts are short, so
being disengaged predicts a switch), which says nothing about what precedes the switch.

WHAT MAY PREDICT TRIAL t. Everything from trials t-1 ... t-LAG, plus (optionally) the
Pre-quiescence and Quiescence epochs of trial t itself, which end before the stimulus. Trial t's
Choice and ITI epochs are never used: they contain the choice that defines the state at t.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GLM_HMM_DIR = HERE.parent
PREFIX = GLM_HMM_DIR.parent
CACHE_DIR = HERE / 'cache'

STATES_FILE = GLM_HMM_DIR / 'merged_behavioral_and_states.pqt'
ALLK_POSTERIORS = GLM_HMM_DIR / 'k2_k3_pilot' / 'all_k_posteriors.parquet'
ALLK_WEIGHTS = GLM_HMM_DIR / 'k2_k3_pilot' / 'all_k_weights.csv'
# clustering/data_files on the Mac, data/ on the linux machine
SYLLABLE_FILE = next((p for p in [PREFIX / 'clustering' / 'data_files' / '8_k_10_bin_syllables_02-10-2026',
                                  PREFIX / 'data' / '8_k_10_bin_syllables_02-10-2026'] if p.exists()),
                     PREFIX / 'clustering' / 'data_files' / '8_k_10_bin_syllables_02-10-2026')
LDA_FILE = PREFIX / 'clustering' / 'data_files' / 'mouse_LDA_5_bins_raw_shrink0.5_wmouse_360_02-10-2026'
TRIAL_MODES_FILE = PREFIX / '3_trial_modes' / 'trial_modes_8_k_10_bin_syllables_02-10-2026' / 'trials.pqt'

EPOCHS = ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']   # order within a trial
PRE_STIMULUS = ['Pre-quiescence', 'Quiescence']               # usable from the current trial
N_PAW = 8
PAW_NAMES = ['still', 'slow', 'modL', 'modR', 'mod', 'fastL', 'fastR', 'vfast']
TARGETS = ('disengage', 'reengage', 'any', 'disengaged')
STATE_TARGETS = ('disengaged',)   # the state itself, not a transition

if str(PREFIX) not in sys.path:
    sys.path.insert(0, str(PREFIX))


# ---------------------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------------------
class _Np2Unpickler(pickle.Unpickler):
    """The LDA pickles were written with numpy 2 (module numpy._core); iblenv has numpy 1."""

    def find_class(self, module, name):
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def load_lda(path=LDA_FILE, n_components=3):
    try:
        lda = pd.read_pickle(path)
    except ModuleNotFoundError:
        with open(path, 'rb') as fh:
            lda = _Np2Unpickler(fh).load()
    lda = lda.rename(columns={i: f'lda_{i + 1}' for i in range(n_components)})
    return lda[['session', 'mouse_name', 'lab'] + [f'lda_{i + 1}' for i in range(n_components)]]


def _engaged_state_allk(K):
    """Per mouse, the state with the most stimulus-following weight. In ssm's sign convention a
    NEGATIVE stim weight follows the stimulus (see session_glmhmm_em.py)."""
    w = pd.read_csv(ALLK_WEIGHTS)
    w = w[w['K'] == K]
    return w.loc[w.groupby('mouse_name')['stim'].idxmin(), ['mouse_name', 'state']] \
        .set_index('mouse_name')['state']


def load_states(source='k2_original', K=2):
    """One row per trial: eid, mouse_name, trial_id, state (int), engaged (bool), p_engaged,
    plus the task variables of that trial.

    source='k2_original': merged_behavioral_and_states.pqt (state1 = engaged, as in
        load_states.ipynb). 459 sessions, every mouse.
    source='all_k': the K-state refits in all_k_posteriors.parquet (K in 2, 3, 4); only mice
        with >= 3 sessions. The engaged state is picked per mouse from the weights.
    """
    beh = pd.read_parquet(STATES_FILE)
    beh['trial_id'] = beh.groupby('eid').cumcount()
    beh = beh.rename(columns={'animal': 'mouse_name'})
    right_correct = beh['contrastLeft'].isna() & (beh['rewarded'] == 1)
    right_incorrect = beh['contrastRight'].isna() & (beh['rewarded'] == -1)
    beh['choice_right'] = (right_correct | right_incorrect).astype(int)
    beh['abs_contrast'] = beh['signed_contrast'].abs()
    task_cols = ['eid', 'mouse_name', 'trial_id', 'rewarded', 'abs_contrast', 'signed_contrast',
                 'probabilityLeft', 'choice_right']

    if source == 'k2_original':
        out = beh[task_cols + ['p_state1']].copy()
        out['p_engaged'] = out.pop('p_state1')
        out['engaged'] = out['p_engaged'] >= 0.5
        out['state'] = (~out['engaged']).astype(int)      # 0 engaged, 1 disengaged
        return out

    if source != 'all_k':
        raise ValueError(f"source must be 'k2_original' or 'all_k', got {source!r}")
    post = pd.read_parquet(ALLK_POSTERIORS, filters=[('K', '==', K)])
    wide = post.pivot_table(index=['mouse_name', 'eid', 'trial_idx'], columns='state',
                            values='posterior').reset_index()
    states = list(range(K))
    wide['state'] = wide[states].to_numpy().argmax(axis=1)
    eng = _engaged_state_allk(K)
    wide['engaged_state'] = wide['mouse_name'].map(eng)
    wide['p_engaged'] = wide[states].to_numpy()[np.arange(len(wide)), wide['engaged_state'].to_numpy()]
    wide['engaged'] = wide['state'] == wide['engaged_state']
    wide = wide.rename(columns={'trial_idx': 'trial_id'})
    out = wide[['eid', 'trial_id', 'state', 'engaged', 'p_engaged']].merge(
        beh[task_cols], on=['eid', 'trial_id'], how='inner')
    return out


def load_syllable_features(feature_set='decomposed', path=SYLLABLE_FILE, cache=True):
    """One row per (eid, trial_id), one column per '<epoch>|<feature>' = fraction of that
    epoch's 10 bins spent in the feature. NaN bins are left out of the denominator.

    feature_set='decomposed': paw state one-hot (8) + whisk + lick, per epoch -> 40 columns.
        Paw fractions sum to 1, so one is redundant; the ridge penalty handles that.
    feature_set='joint': the 32 joint syllable codes, per epoch -> 128 columns.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f'syllable_features_{feature_set}_{Path(path).name}.parquet'
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    import paper_style as ps
    ps.use_paw_states('uniform')

    syl = pd.read_parquet(path)
    split = syl['sample'].str.split()
    syl['eid'] = split.str[0]
    syl['trial_id'] = split.str[1].astype(float).astype(int)
    codes = ps.paw_codes(np.stack(syl['binned_sequence'].to_numpy()).astype(float))
    valid = ~np.isnan(codes)
    n_valid = valid.sum(axis=1).astype(float)
    n_valid[n_valid == 0] = np.nan
    ci = np.where(valid, codes, 0).astype(int)

    cols = {}
    if feature_set == 'decomposed':
        paw, whisk, lick = ci % N_PAW, (ci // N_PAW) % 2, ci // (2 * N_PAW)
        for k in range(N_PAW):
            cols[f'paw_{PAW_NAMES[k]}'] = ((paw == k) & valid).sum(axis=1) / n_valid
        cols['whisk'] = ((whisk == 1) & valid).sum(axis=1) / n_valid
        cols['lick'] = ((lick == 1) & valid).sum(axis=1) / n_valid
    elif feature_set == 'joint':
        for c in range(4 * N_PAW):
            w, l = (c // N_PAW) % 2, c // (2 * N_PAW)
            cols[f'syl{c:02d}_{PAW_NAMES[c % N_PAW]}{"_w" if w else ""}{"_l" if l else ""}'] = \
                ((ci == c) & valid).sum(axis=1) / n_valid
    else:
        raise ValueError(f"feature_set must be 'decomposed' or 'joint', got {feature_set!r}")

    long = pd.concat([syl[['eid', 'trial_id', 'broader_label', 'trial_type']].reset_index(drop=True),
                      pd.DataFrame(cols)], axis=1)
    feat_names = list(cols)
    wide = long.pivot_table(index=['eid', 'trial_id'], columns='broader_label', values=feat_names)
    wide.columns = [f'{ep}|{f}' for f, ep in wide.columns]
    wide = wide[[f'{ep}|{f}' for ep in EPOCHS for f in feat_names]].reset_index()
    tt = long.drop_duplicates(['eid', 'trial_id'])[['eid', 'trial_id', 'trial_type']]
    wide = wide.merge(tt, on=['eid', 'trial_id'], how='left')
    if cache:
        wide.to_parquet(cache_path)
    return wide


def load_trial_modes(path=TRIAL_MODES_FILE):
    """(eid, trial_id, trial_mode) from 3_trial_modes/trial_modes_proficient.ipynb. Trials that
    were not embedded (a NaN syllable bin) or fell on the KDE background get mode -1."""
    m = pd.read_parquet(path, columns=['session', 'trial_id', 'trial_cluster'])
    m = m.rename(columns={'session': 'eid'})
    m['trial_id'] = m['trial_id'].astype(int)
    m['trial_mode'] = m['trial_cluster'].fillna(-1).astype(int)
    return m[['eid', 'trial_id', 'trial_mode']]


def syllable_columns(df):
    return [c for c in df.columns if '|' in c and c.split('|')[0] in EPOCHS]


def check_alignment(table):
    """Fraction of trials per session whose syllable-file trial_type agrees with the states
    parquet on correctness and |contrast|."""
    parts = table['trial_type'].str.split(expand=True)
    ok = ((parts[0] == 'correct') == (table['rewarded'] == 1)) & \
        np.isclose(parts[1].astype(float), table['abs_contrast'])
    return ok.groupby(table['eid']).mean()


# ---------------------------------------------------------------------------------------------
# trial table
# ---------------------------------------------------------------------------------------------
def build_trial_table(source='k2_original', K=2, feature_set='decomposed', horizon=1,
                      min_alignment=0.99, trial_modes=None):
    """Trial table with targets, task history inputs, syllables and LD1.

    Targets (NaN where undefined):
      leave_<h>    1 if the state at any of t..t+h-1 differs from the state at t-1
      disengage    leave_h, defined only where the animal was ENGAGED at t-1
      reengage     leave_h, defined only where it was NOT engaged at t-1
                   (for K>2 this means leaving the current disengaged state, which can be
                   into another disengaged state)
      any          leave_h on every trial (not a hazard; occupancy leaks in, see module doc)
      disengaged   1 if the most probable state at t is not the engaged one (every trial)
    """
    states = load_states(source, K)
    syl = load_syllable_features(feature_set)
    lda = load_lda()

    df = states.merge(syl, on=['eid', 'trial_id'], how='inner')
    align = check_alignment(df)
    bad = align[align < min_alignment]
    if len(bad):
        raise RuntimeError(f'{len(bad)} sessions fail the syllable/state alignment check: '
                           f'{bad.head().to_dict()}')
    df = df.merge(lda.rename(columns={'session': 'eid', 'mouse_name': 'lda_mouse'}),
                  on='eid', how='inner')
    if trial_modes is not None:
        df = df.merge(load_trial_modes(trial_modes), on=['eid', 'trial_id'], how='left')
        df['trial_mode'] = df['trial_mode'].fillna(-1).astype(int)
    df = df.sort_values(['eid', 'trial_id']).reset_index(drop=True)

    g = df.groupby('eid', sort=False)
    df['n_trials_session'] = g['trial_id'].transform('size')
    prev_state = g['state'].shift(1)
    df['prev_engaged'] = g['engaged'].shift(1)
    # did the state change between t-1 and any of t .. t+horizon-1 ?
    left = np.zeros(len(df), dtype=float)
    valid = prev_state.notna().to_numpy()
    for h in range(horizon):
        fut = g['state'].shift(-h)
        valid &= fut.notna().to_numpy()
        left = np.maximum(left, (fut != prev_state).to_numpy(dtype=float))
    df['leave'] = np.where(valid, left, np.nan)
    pe = df['prev_engaged']
    df['disengage'] = np.where(pe == True, df['leave'], np.nan)    # noqa: E712
    df['reengage'] = np.where(pe == False, df['leave'], np.nan)    # noqa: E712
    df['any'] = df['leave']
    df['disengaged'] = (~df['engaged'].astype(bool)).astype(float)

    # dwell so far: trials since the last flip, known at t-1
    flip = (df['state'] != prev_state) & prev_state.notna()
    run_id = flip.groupby(df['eid']).cumsum()
    df['dwell'] = df.groupby([df['eid'], run_id]).cumcount() + 1
    df['dwell_prev'] = df.groupby('eid', sort=False)['dwell'].shift(1)
    df['trial_frac'] = df['trial_id'] / df['n_trials_session']
    df.attrs.update(source=source, K=K, feature_set=feature_set, horizon=horizon)
    return df


# ---------------------------------------------------------------------------------------------
# lagged design
# ---------------------------------------------------------------------------------------------
def baseline_frame(df, lag, dwell=True, current_outcome=False):
    """Task-history regressors, all known before trial t's stimulus:
    reward, |contrast| and block-congruent choice at t-1..t-lag; position in session; and
    (dwell=True) log dwell time in the current state, from the state sequence up to t-1.

    dwell must be False when the target is the state itself: long dwells are engaged dwells, so
    it would hand the model the previous state. current_outcome=True adds trial t's own reward,
    |contrast| and block-congruent choice -- needed as the baseline when trial t's Choice/ITI
    syllables are inputs (mode='concurrent'), since those epochs carry the outcome."""
    g = df.groupby('eid', sort=False)
    congruent = ((df['choice_right'] == 1) == (df['probabilityLeft'] < 0.5)).astype(float)
    congruent[df['probabilityLeft'] == 0.5] = 0.5
    cols = {'trial_frac': df['trial_frac']}
    if dwell:
        cols['log_dwell'] = np.log1p(df['dwell_prev'])
    if current_outcome:
        cols['reward_t'] = df['rewarded']
        cols['abs_contrast_t'] = df['abs_contrast']
        cols['congruent_t'] = congruent
    for l in range(1, lag + 1):
        cols[f'reward_t-{l}'] = g['rewarded'].shift(l)
        cols[f'abs_contrast_t-{l}'] = g['abs_contrast'].shift(l)
        cols[f'congruent_t-{l}'] = congruent.groupby(df['eid']).shift(l)
    return pd.DataFrame(cols, index=df.index)


def syllable_frame(df, lag, epochs=EPOCHS, current_prestim=True, lag_mode='stack',
                   center='session', current_epochs=None):
    """Lagged syllable regressors. Column names are '<lag>|<epoch>|<feature>' with lag 't' for
    the current trial's pre-stimulus epochs and 't-l' for past trials (or 't-1:t-L' when
    lag_mode='mean', which averages lags 1..L into one block).

    center='session' subtracts each session's mean usage first, so the model sees only
    trial-to-trial fluctuations. That is the default because LD1 is itself built from
    session-level syllable usage: without centring, "sessions that whisk more also switch more"
    and "whisking more right before a switch" are mixed, and the first is partly LD1 again.
    """
    base_cols = [c for c in syllable_columns(df) if c.split('|')[0] in epochs]
    X0 = df[base_cols].astype(float)
    if center == 'session':
        X0 = X0 - X0.groupby(df['eid']).transform('mean')
    elif center is not None:
        raise ValueError("center must be 'session' or None")
    g = X0.groupby(df['eid'], sort=False)

    blocks = []
    if current_epochs is None:
        current_epochs = PRE_STIMULUS if current_prestim else []
    if current_epochs:
        cur = [c for c in base_cols if c.split('|')[0] in current_epochs]
        blocks.append(X0[cur].add_prefix('t|'))
    shifted = [g.shift(l) for l in range(1, lag + 1)]
    if lag_mode == 'stack':
        blocks += [s.add_prefix(f't-{l}|') for l, s in enumerate(shifted, start=1)]
    elif lag_mode == 'mean':
        blocks.append((sum(shifted) / lag).add_prefix(f't-1:t-{lag}|'))
    else:
        raise ValueError("lag_mode must be 'stack' or 'mean'")
    return pd.concat(blocks, axis=1)


def mode_frame(df, lag, include_current, lag_mode='stack', center='session'):
    """Lagged one-hot trial modes ('<lag>|mode|m<k>', background/unembedded = m-1). A trial mode
    describes the WHOLE trial, Choice and ITI included, so trial t's own mode is only allowed
    when include_current (mode='concurrent')."""
    if 'trial_mode' not in df:
        raise ValueError('build_trial_table(..., trial_modes=td.TRIAL_MODES_FILE) first')
    oh = pd.get_dummies(df['trial_mode'], prefix='', prefix_sep='').astype(float)
    oh.columns = [f'mode|m{int(c)}' for c in oh.columns]
    if center == 'session':
        oh = oh - oh.groupby(df['eid']).transform('mean')
    g = oh.groupby(df['eid'], sort=False)
    blocks = [oh.add_prefix('t|')] if include_current else []
    shifted = [g.shift(l) for l in range(1, lag + 1)]
    if lag_mode == 'stack':
        blocks += [s.add_prefix(f't-{l}|') for l, s in enumerate(shifted, start=1)]
    elif shifted:
        blocks.append((sum(shifted) / lag).add_prefix(f't-1:t-{lag}|'))
    if not blocks:
        raise ValueError('no mode features: lag 0 without the current trial')
    return pd.concat(blocks, axis=1)


def design(df, target='disengage', lag=3, epochs=EPOCHS, current_prestim=True, lag_mode='stack',
           center='session', mode='predictive', features='syllables'):
    """(X_base, X_syll, y, meta) for one target and lag. Rows: target defined and at least `lag`
    trials of history. Missing syllable epochs are mean-imputed (0 after centring; the session
    mean otherwise) -- they are ~1% of bins.

    mode='predictive': trial t contributes only its pre-stimulus epochs (or none, with
        current_prestim=False) -- "can the state be told before the stimulus?"
    mode='concurrent': trial t contributes ALL its epochs, and the baseline gets trial t's
        outcome -- "do engaged and disengaged trials look different, beyond their outcome?"
    For state targets the baseline drops dwell time (it would leak the previous state)."""
    if target not in TARGETS:
        raise ValueError(f'target must be one of {TARGETS}')
    if mode not in ('predictive', 'concurrent'):
        raise ValueError("mode must be 'predictive' or 'concurrent'")
    concurrent = mode == 'concurrent'
    Xb = baseline_frame(df, lag, dwell=target not in STATE_TARGETS, current_outcome=concurrent)
    if features not in ('syllables', 'modes', 'both'):
        raise ValueError("features must be 'syllables', 'modes' or 'both'")
    parts = []
    if features in ('syllables', 'both'):
        parts.append(syllable_frame(df, lag, epochs, current_prestim, lag_mode, center,
                                    current_epochs=list(epochs) if concurrent else None))
    if features in ('modes', 'both'):
        parts.append(mode_frame(df, lag, concurrent, lag_mode, center))
    Xs = pd.concat(parts, axis=1)
    keep = df[target].notna() & Xb.notna().all(axis=1) & (df.groupby('eid').cumcount() >= lag)
    Xb, Xs = Xb[keep], Xs[keep]
    Xs = Xs.fillna(0.0 if center == 'session' else Xs.mean())
    meta = df.loc[keep, ['eid', 'mouse_name', 'lab', 'trial_id', 'lda_1', 'lda_2', 'engaged']]
    return Xb, Xs, df.loc[keep, target].astype(int).to_numpy(), meta.reset_index(drop=True)


def feature_groups(columns, by=('lag', 'epoch', 'modality')):
    """Map each syllable column to a group label for grouped importance. `by` picks which of
    lag / epoch / modality (paw / whisk / lick, or the joint code) the label keeps."""
    groups = {}
    for c in columns:
        lag, epoch, feat = c.split('|')
        modality = ('mode' if epoch == 'mode' else 'paw' if feat.startswith('paw_')
                    else feat if feat in ('whisk', 'lick') else 'joint')
        parts = {'lag': lag, 'epoch': epoch, 'modality': modality, 'feature': feat}
        groups.setdefault(' / '.join(parts[b] for b in by), []).append(c)
    return groups


# ---------------------------------------------------------------------------------------------
# descriptive: syllable usage around transitions
# ---------------------------------------------------------------------------------------------
def event_triggered(df, target='disengage', window=(-10, 5), min_dwell=5):
    """Session-centred syllable usage on trials around each transition (relative trial 0 = the
    first trial in the new state), averaged within mouse, then mean +- SEM across mice.

    Only transitions preceded by >= `min_dwell` trials in the old state are used, so the
    pre-transition window is not itself a mix of states. Not a model -- a picture of what the
    regression is trying to pick up.
    """
    cols = syllable_columns(df)
    X = df[cols].astype(float)
    X = X - X.groupby(df['eid']).transform('mean')
    onset = (df[target] == 1) & (df['dwell_prev'] >= min_dwell)
    pos = np.arange(len(df))
    eid = df['eid'].to_numpy()
    rows = []
    for t in pos[onset.to_numpy()]:
        for r in range(window[0], window[1] + 1):
            u = t + r
            if 0 <= u < len(df) and eid[u] == eid[t]:
                rows.append((df['mouse_name'].iat[t], t, r, u))
    idx = pd.DataFrame(rows, columns=['mouse_name', 'event', 'rel', 'row'])
    vals = X.to_numpy()[idx['row'].to_numpy()]
    long = pd.concat([idx[['mouse_name', 'rel']], pd.DataFrame(vals, columns=cols)], axis=1)
    per_mouse = long.groupby(['mouse_name', 'rel']).mean()
    agg = per_mouse.groupby('rel').agg(['mean', 'sem'])
    out = agg.stack(level=0, future_stack=True).reset_index().rename(columns={'level_1': 'column'})
    out[['epoch', 'feature']] = out['column'].str.split('|', expand=True)
    out.attrs['n_events'] = int(onset.sum())
    out.attrs['n_mice'] = per_mouse.index.get_level_values(0).nunique()
    return out
