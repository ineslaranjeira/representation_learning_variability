"""
Fast, output-identical re-implementation of 5_syllable_generation.ipynb (used by
5_syllable_generation_fast.ipynb).

What is faster, and why the output does not change:
- align_bin_design_matrix / states_per_trial_phase rebuilt a boolean mask over the whole session
  for every trial and column (and called prepro() four times per trial). Here every
  "Bin > start & Bin <= end" mask is a contiguous slice of the sorted Bin column, found with
  np.searchsorted, and the slices are written in the SAME order (later trials overwrite earlier
  ones exactly as before). The re-alignment column new_bin was dropped right after, so it is
  not computed.
- The identifiable-state strings are built column-wise instead of row by row.
- Syllable/raw binning: one sort per session instead of groupby().apply(list), and a mode equal
  to scipy.stats.mode (np.unique: most frequent value, the smallest on ties, NaN counted as a
  value) without scipy's per-call overhead. Means use np.mean on the same slices.
- Sessions run in parallel (joblib), results are concatenated in the original session order.
"""
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import zscore

from segmentation_functions import broader_label, idxs_from_files, prepro, state_identifiability

DATA = '/home/ines/repositories/representation_learning_variability/paper-individuality/data/'

# ------------------------------------------------------------------------------------------------
# THE CHOICES. A run is (DATASET, MOVEMENT, K, WHEEL, K_WHEEL, STATE_SPACE, LICK_HMM); everything
# else follows.
# ------------------------------------------------------------------------------------------------
# where each dataset keeps its inputs; 1-camera datasets only have the left paw
DATASETS = {
    'proficient': dict(root=DATA, design_matrices=DATA + 'design_matrices/', wavelets=DATA + 'paw_wavelets/',
                       cameras=2, sub=''),
    'neuromodulators': dict(root=DATA + 'neuromodulators/', design_matrices=DATA + 'design_matrices/1_camera_setup/NM/',
                            wavelets=DATA + 'neuromodulators/left_paw_wavelets/', cameras=1, sub=''),
    **{stage: dict(root=DATA + 'training/', design_matrices=DATA + f'design_matrices/1_camera_setup/{stage}/',
                   wavelets=DATA + 'training/left_paw_wavelets/', cameras=1, sub=f'{stage}/')
       for stage in ('session_1', 'last_training', 'biased')},
}
# the movement states that give the FIRST digit of a syllable (-> 3.3's VAR). With WHEEL = True the
# wheel's own states (3.3 with VAR = 'wheel') add a LAST digit; not with left_paw_wheel, which has the
# wheel inside already.
MOVEMENTS = {'both_paws': 'paw', 'left_paw': 'left_paw', 'left_paw_wheel': 'left_paw_wheel'}
WHEEL_VAR = 'wheel'
# which clustering defined them (3.3_wavelet_clusters_uniform's output folders; every k it was run
# with sits in the same folder, as most_likely_states_<k>_<mouse><eid>.npy)
STATE_SPACES = {'own': '{var}_states_uniform_raw_sessionz/',                 # fitted on this dataset
                'stages': '{var}_states_uniform_raw_sessionz_fit-stages/'}   # FIT_NAME='stages'
# the whisker / lick HMM fits (second and third digit), in <dataset root>/hmm/most_likely_states/.
# The leading number is num_train_batches, which the old (3-element) pickles need.
WHISKER_FIT = '5_gaussian_kmeans_em_zsc_True_k0_s2'
LICK_HMMS = {'standard': '5_poisson_prior_em_zsc_False_k0_s2',
             '100ms': '5_poisson_prior_em_zsc_False_k0_s2_bin100ms'}
# 'auto': 100 ms for the left-camera datasets (neuromodulators, training stages), standard for
# proficient (which has no 100 ms fit)

# The paw states of the CURRENT data/states_files and 8_k_10_bin_syllables_02-10-2026 went through this
# relabelling; it is kept for (proficient, both_paws, own) so those files stay reproducible.
# paper_style's 'uniform' set undoes it at load (syllable_paw_map). Every other run keeps the
# vigor numbering of 3.3 (0 = stillest).
PAW_RELABEL_19AUG2026 = {0: 0, 1: 2, 2: 1, 3: 6, 4: 7, 5: 5, 6: 4, 7: 3}
# (dataset, movement, k, wheel, state space, lick HMM) that writes to the historical names
DEFAULT_RUN = ('proficient', 'both_paws', 8, False, 'own', 'standard')

RAW_COLUMNS = ['Bin', 'Lick count', 'avg_wheel_vel', 'whisker_me', 'avg_whisker_me',
               'l_paw_x', 'l_paw_y', 'r_paw_x', 'r_paw_y']
STATES_FILE_COLUMNS = ['Bin', 'Lick count', 'avg_wheel_vel', 'whisker_me', 'l_paw_x', 'l_paw_y', 'r_paw_x',
                       'r_paw_y', 'most_likely_states', 'correct', 'choice', 'contrast', 'block', 'reaction',
                       'response', 'elongation', 'wsls', 'trial_id', 'goCueTrigger_times',
                       'identifiable_states', 'label', 'mouse_name', 'session', 'broader_label']
TRIAL_LEVEL_VARS = ['correct', 'choice', 'contrast', 'block', 'reaction', 'response', 'elongation', 'wsls',
                    'trial_id']
TRIAL_TYPE_AGG = ['correct_str', 'contrast_str', 'block_str', 'choice']
GROUP_KEYS = ['sample', 'trial_type', 'broader_label', 'mouse_name']


def _states_dir(root, var, state_space):
    return root + STATE_SPACES[state_space].format(var=var)


def _ks(d):
    """The k values a states folder holds."""
    if not os.path.isdir(d):
        return []
    return sorted({int(f.split('_')[3]) for f in os.listdir(d) if f.startswith('most_likely_states_')})


def available():
    """Every (dataset, segmentation, state space) with states on disk, and the k values it has."""
    rows = [(d, v, sp, _ks(_states_dir(DATASETS[d]['root'], v, sp)))
            for d in DATASETS for v in [*MOVEMENTS.values(), WHEEL_VAR] for sp in STATE_SPACES]
    return pd.DataFrame([r for r in rows if r[3]], columns=['dataset', 'segmentation', 'state_space', 'k'])


def default_lick(dataset):
    return '100ms' if DATASETS[dataset]['cameras'] == 1 else 'standard'


def run_names(dataset, movement, state_space, lick_hmm='auto', k=8, wheel=False, k_wheel=8):
    """(states folder, file-name tag) of a run, relative to the dataset root -- no checks, so
    readers can find outputs without the inputs being on disk."""
    lick_hmm = default_lick(dataset) if lick_hmm == 'auto' else lick_hmm
    if (dataset, movement, int(k), bool(wheel), state_space, lick_hmm) == DEFAULT_RUN:
        return 'states_files/', ''
    name = (f'{movement}{"+wheel" if wheel else ""}_k{k}{f"-{k_wheel}" if wheel else ""}'
            f'_{state_space}_lick-{lick_hmm}')
    return f'states_files_{name}/' + DATASETS[dataset]['sub'], f'{name}_{dataset}_'


def syllables_glob(dataset, movement, state_space, lick_hmm='auto', k=8, wheel=False, k_wheel=8,
                   target_length=10):
    """Glob pattern of a run's syllable files (any date)."""
    _, tag = run_names(dataset, movement, state_space, lick_hmm, k, wheel, k_wheel)
    return f"{DATASETS[dataset]['root']}{tag}{k}_k_{target_length}_bin_syllables_*"


def resolve(dataset, movement, state_space, lick_hmm='auto', k=8, wheel=False, k_wheel=8,
            states_root=None, out_root=None):
    """The full configuration of one run, checked. `states_root` / `out_root` redirect where the
    clustering states are read / the outputs are written (tests); default: the dataset's root."""
    for name, val, opts in (('DATASET', dataset, DATASETS), ('MOVEMENT', movement, MOVEMENTS),
                            ('STATE_SPACE', state_space, STATE_SPACES), ('LICK_HMM', lick_hmm, [*LICK_HMMS, 'auto'])):
        if val not in opts:
            raise ValueError(f'{name} must be one of {list(opts)}, got {val!r}')
    ds = DATASETS[dataset]
    if movement == 'both_paws' and ds['cameras'] == 1:
        raise ValueError(f'{dataset} is a 1-camera dataset: only the left paw is tracked (left_paw / left_paw_wheel)')
    if wheel and movement == 'left_paw_wheel':
        raise ValueError('left_paw_wheel already contains the wheel: use WHEEL = False, or MOVEMENT = left_paw')
    segs = {'movement': (MOVEMENTS[movement], int(k))}
    if wheel:
        segs['wheel'] = (WHEEL_VAR, int(k_wheel))
    dirs = {}
    for seg, (var, kk) in segs.items():
        d = _states_dir(states_root or ds['root'], var, state_space)
        if kk not in _ks(d):
            have = _ks(d)
            raise FileNotFoundError(f'no {var} states with k = {kk} for ({dataset}, {state_space}) in {d}; '
                                    + (f'that folder has k = {have}' if have else 'that folder does not exist')
                                    + ' (run 3.3 with this VAR / K first)\navailable:\n' + available().to_string())
        dirs[seg] = d
    if lick_hmm == 'auto':
        lick_hmm = default_lick(dataset)
    run = (dataset, movement, int(k), bool(wheel), state_space, lick_hmm)
    out_sub, tag = run_names(dataset, movement, state_space, lick_hmm, k, wheel, k_wheel)
    root = out_root or ds['root']
    hmm = ds['root'] + 'hmm/most_likely_states/'
    cfg = dict(dataset=dataset, movement=movement, k=int(k), wheel=bool(wheel), k_wheel=int(k_wheel) if wheel else None,
               state_space=state_space, lick_hmm=lick_hmm,
               relabel=PAW_RELABEL_19AUG2026 if run[:5] == DEFAULT_RUN[:5] else None,
               design_matrices=ds['design_matrices'], wavelets=ds['wavelets'],
               states_dir=dirs['movement'], wheel_states_dir=dirs.get('wheel'),
               whisker_dir=hmm + WHISKER_FIT + '/', lick_dir=hmm + LICK_HMMS[lick_hmm] + '/',
               num_train_batches=int(WHISKER_FIT.split('_')[0]),
               out_states=root + out_sub, out_root=root, tag=tag)
    assert int(LICK_HMMS[lick_hmm].split('_')[0]) == cfg['num_train_batches']
    for key in ('design_matrices', 'wavelets', 'whisker_dir', 'lick_dir'):
        if not os.path.isdir(cfg[key]):
            raise FileNotFoundError(f'{key}: {cfg[key]} does not exist')
    # digits of an identifiable state, in the original's order: movement, whisker, lick[, wheel];
    # its integer code counts through them with the movement fastest (the original's mapping loops)
    cfg['digits'] = [('movement', cfg['k']), ('whisker_me', 2), ('Lick count', 2)] + ([('wheel', cfg['k_wheel'])] if wheel else [])
    return cfg


def describe(cfg):
    keys = ['dataset', 'movement', 'k', 'wheel', 'k_wheel', 'state_space', 'lick_hmm', 'relabel', 'design_matrices',
            'wavelets', 'states_dir', 'wheel_states_dir', 'whisker_dir', 'lick_dir', 'out_states', 'out_root', 'tag']
    w = max(map(len, keys))
    return '\n'.join(f'{k:<{w}}  {cfg[k]}' for k in keys)


# ------------------------------------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------------------------------------
def all_sessions(cfg):
    """(mouse, eid) of every design matrix, in the order the original notebook used (os.listdir)."""
    files = [f for f in os.listdir(cfg['design_matrices']) if 'design_matrix' in f and 'standardized' not in f]
    idxs, _ = idxs_from_files(files)
    return [(m[37:], m[:36]) for m in np.atleast_1d(idxs)]


def input_files(cfg, mouse, eid):
    fit_id = mouse + eid
    return dict(states=cfg['states_dir'] + f"most_likely_states_{cfg['k']}_{fit_id}.npy",
                wheel_states=cfg['wheel_states_dir'] and cfg['wheel_states_dir'] + f"most_likely_states_{cfg['k_wheel']}_{fit_id}.npy",
                whisker=cfg['whisker_dir'] + 'whisker_me_' + fit_id,
                lick=cfg['lick_dir'] + 'Lick count_' + fit_id,
                wavelets=cfg['wavelets'] + f'paw_vel_wavelets_{eid}_{mouse}',
                trials=cfg['design_matrices'] + f'session_trials_{eid}_{mouse}')


def sessions_with_inputs(cfg):
    """Sessions that have movement, whisker and lick (and wheel) states (the original's cell 4)."""
    need = ('states', 'whisker', 'lick') + (('wheel_states',) if cfg['wheel'] else ())
    return [(m, e) for m, e in all_sessions(cfg)
            if all(os.path.exists(input_files(cfg, m, e)[k]) for k in need)]


def states_file(cfg, mouse, eid):
    return cfg['out_states'] + f"{cfg['k']}_states_file_{eid}_{mouse}"


# ------------------------------------------------------------------------------------------------
# Step 1: per-session states files
# ------------------------------------------------------------------------------------------------
def _span(bins, lo, hi):
    """Rows with lo < Bin <= hi, as a slice of the sorted Bin array (empty if a bound is NaN)."""
    if not (lo == lo and hi == hi):
        return slice(0, 0)
    a = np.searchsorted(bins, lo, side='right')
    b = np.searchsorted(bins, hi, side='right')
    return slice(a, max(a, b))


def _as_column(arr):
    """Object columns in which nothing was written stay float, as pandas left them."""
    if arr.dtype == object and all(isinstance(x, float) and x != x for x in arr):
        return arr.astype(float)
    return arr


def _trial_columns(bins, trials, pp):
    """align_bin_design_matrix(init, end, ['goCueTrigger_times'], ...) without new_bin."""
    n = len(bins)
    out = {c: np.full(n, np.nan) for c in ('correct', 'contrast', 'block', 'reaction', 'response', 'elongation',
                                          'trial_id', 'goCueTrigger_times')}
    out['choice'] = np.full(n, np.nan, dtype=object)
    out['wsls'] = np.full(n, np.nan, dtype=object)
    feedback, choice = trials['feedbackType'], trials['choice']
    values = dict(contrast=np.abs(pp['signed_contrast']), block=trials['probabilityLeft'],
                  reaction=pp['reaction'], response=pp['response'], elongation=pp['elongation'],
                  wsls=pp['wsls'], trial_id=trials['index'], goCueTrigger_times=trials['goCueTrigger_times'])
    starts = trials['intervals_0']
    for t in range(len(trials) - 1):
        s = _span(bins, starts[t], starts[t + 1])
        if feedback[t] == 1:
            out['correct'][s] = 1
        elif feedback[t] == -1:
            out['correct'][s] = 0
        if choice[t] == 1:
            out['choice'][s] = 'right'
        elif choice[t] == -1:
            out['choice'][s] = 'left'
        elif choice[t] == 0:
            out['choice'][s] = 'no_go'
        else:
            raise ValueError(f'Unexpected choice value {choice[t]!r} at trial {t}')
        for c in ('reaction', 'response', 'elongation', 'contrast', 'block', 'wsls', 'trial_id',
                  'goCueTrigger_times'):
            out[c][s] = values[c][t]
    return {c: _as_column(a) for c, a in out.items()}


def _labels(bins, trials, pp):
    """states_per_trial_phase(...)['label']."""
    label = np.full(len(bins), np.nan, dtype=object)
    go, qp = trials['goCueTrigger_times'], trials['quiescencePeriod']
    i0, i1 = trials['intervals_0'], trials['intervals_1']
    fb, fm, ch = trials['feedback_times'], trials['firstMovement_times'], trials['choice']
    ftype, sc = trials['feedbackType'], pp['signed_contrast']
    for t in range(len(trials)):
        label[_span(bins, i0[t], go[t] - qp[t])] = 'Pre-quiescence'
        label[_span(bins, go[t] - qp[t], go[t])] = 'Quiescence'
        if ftype[t] == -1.:
            label[_span(bins, fb[t], i1[t] - 1)] = 'ITI'
        elif ftype[t] == 1.:
            label[_span(bins, fb[t], i1[t])] = 'ITI'
        if ch[t] == -1:
            label[_span(bins, fm[t], fb[t])] = 'Left choice'
        elif ch[t] == 1.:
            label[_span(bins, fm[t], fb[t])] = 'Right choice'
        elif ch[t] == 0:
            label[_span(bins, fm[t], fb[t])] = 'No go'
        if sc[t] < 0:
            label[_span(bins, go[t], fm[t])] = 'Stimulus left'
        elif sc[t] > 0:
            label[_span(bins, go[t], fm[t])] = 'Stimulus right'
        elif sc[t] == 0:
            label[_span(bins, go[t], fm[t])] = 'Stimulus zero'
    return _as_column(label)


def _hmm_states(cfg, path, var, dm):
    """Whisker / lick HMM states as a column of dm (NaN where the HMM has none)."""
    obj = pickle.load(open(path, 'rb'))
    col = dm['Bin'] * np.nan
    if len(obj) == 4:            # newer fits: (states, bins, ...)
        states, bins = obj[0], obj[1]
        dm[var + '_states'] = col
        dm.loc[dm['Bin'].isin(bins), var + '_states'] = states
    elif len(obj) == 3:          # older fits: states of the frames where `var` is defined, cut to batches
        states = obj[0]
        mask = dm[var].notna().to_numpy()
        n = (int(mask.sum()) // cfg['num_train_batches']) * cfg['num_train_batches']
        assert n == len(states), f'{var}: mask ({n}) != states ({len(states)}) for {path}'
        dm[var + '_states'] = col
        dm.loc[dm.index[mask][:n], var + '_states'] = states
    else:
        raise ValueError(f'unexpected HMM file layout ({len(obj)} elements): {path}')
    return dm[['Bin', var + '_states']]


def make_states_file(cfg, mouse, eid):
    """One session -> the states file of the original cell 13 (same rows, columns and values)."""
    f = input_files(cfg, mouse, eid)
    trials = pd.read_parquet(f['trials'], engine='pyarrow').reset_index()
    dm = pd.read_parquet(f['wavelets'])
    beh = 'movement_states'

    # movement states: the first digit
    states, bins = np.load(open(f['states'], 'rb'))
    if cfg['relabel'] is not None:
        states = np.vectorize(cfg['relabel'].get)(states)
    dm[beh] = dm['Bin'] * np.nan
    dm.loc[dm['Bin'].isin(bins), beh] = states
    keep = [c for c in RAW_COLUMNS if c in dm.columns]
    session_states = dm[keep + [beh]]
    for var, path in (('whisker_me', f['whisker']), ('Lick count', f['lick'])):
        session_states = session_states.merge(_hmm_states(cfg, path, var, dm.copy()), on='Bin', how='outer')
    if cfg['wheel']:                 # the wheel's own states, matched on Bin
        w_states, w_bins = np.load(open(f['wheel_states'], 'rb'))
        session_states = session_states.merge(pd.DataFrame({'Bin': w_bins, 'wheel_states': w_states}),
                                              on='Bin', how='left')

    # whisker / lick states: 1 = the state with the larger signal
    session_states = state_identifiability(session_states, ['whisker_me', 'Lick count'])
    state_cols = [beh, 'whisker_me_states', 'Lick count_states'] + (['wheel_states'] if cfg['wheel'] else [])
    valid = session_states[state_cols].notna().all(axis=1).to_numpy()
    ints = session_states.loc[valid, state_cols].to_numpy().astype(int)
    for j, (name, n) in enumerate(cfg['digits']):
        if ints.size and (ints[:, j].min() < 0 or ints[:, j].max() >= n):
            raise ValueError(f'{name} states outside 0..{n - 1} for {mouse} {eid}: is k right?')
    # the string: digits side by side (as the original), separated by '-' once a k exceeds 10
    sep = '' if max(n for _, n in cfg['digits']) <= 10 else '-'
    strs = session_states.loc[valid, state_cols].astype(int).astype(str)
    codes = strs[state_cols[0]]
    for c in state_cols[1:]:
        codes = codes + sep + strs[c]
    combined = np.full(len(session_states), np.nan, dtype=object)
    combined[valid] = codes.tolist()
    # the integer: mixed radix, the first digit fastest -- the original's identifiable_mapping
    integer = np.full(len(session_states), np.nan)
    code, base = np.zeros(len(ints)), 1
    for j, (_, n) in enumerate(cfg['digits']):
        code += ints[:, j] * base
        base *= n
    integer[valid] = code
    session_states['identifiable_states'] = combined
    session_states['most_likely_states'] = integer
    final = dm[['Bin']].merge(session_states, on='Bin', how='left')

    # trial structure
    b = final['Bin'].to_numpy()
    if not (np.all(np.isfinite(b)) and np.all(np.diff(b) >= 0)):
        raise ValueError(f'Bin is not sorted / finite for {mouse} {eid}: the slice logic needs it')
    pp = prepro(trials)
    for c, v in _trial_columns(b, trials, pp).items():
        final[c] = v
    final['label'] = _labels(b, trials, pp)
    final['mouse_name'] = mouse
    final['session'] = eid
    final = broader_label(final)
    return final[STATES_FILE_COLUMNS]


def _quiet(f):
    """Run f without the pandas chained-assignment / dtype warnings that prepro() and the
    original's column upcasts raise in every session (they do not change the result)."""
    def wrapped(*a, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
            # state_identifiability's nanmean of a state that never occurs (the original warns too)
            warnings.filterwarnings('ignore', 'Mean of empty slice')
            warnings.filterwarnings('ignore', 'invalid value encountered in scalar divide')
            return f(*a, **kw)
    return wrapped


@_quiet
def _states_job(cfg, mouse, eid, overwrite):
    out = states_file(cfg, mouse, eid)
    if os.path.exists(out) and not overwrite:
        return 'kept'
    make_states_file(cfg, mouse, eid).to_parquet(out, compression='gzip')
    return 'written'


def check_provenance(cfg):
    """The states folder records the configuration that filled it; refuse to mix configurations."""
    os.makedirs(cfg['out_states'], exist_ok=True)
    rec = {k: cfg[k] for k in ('dataset', 'movement', 'k', 'wheel', 'k_wheel', 'state_space', 'lick_hmm', 'states_dir',
                               'wheel_states_dir', 'whisker_dir', 'lick_dir', 'wavelets', 'design_matrices')}
    rec['relabel'] = None if cfg['relabel'] is None else {str(a): b for a, b in cfg['relabel'].items()}
    path = cfg['out_states'] + '_config.json'
    if os.path.exists(path):
        old = json.load(open(path))
        diff = {k: (old.get(k), v) for k, v in rec.items() if old.get(k) != v}
        if diff:
            raise RuntimeError(f"{cfg['out_states']} was filled with a different configuration: {diff}. "
                               'Use another folder, or delete it to start over.')
    elif any(f.endswith(tuple(str(i) for i in range(10))) or '_states_file_' in f
             for f in os.listdir(cfg['out_states'])):
        print(f"NOTE: {cfg['out_states']} already has states files with no recorded configuration "
              '(written by the original notebook). Existing files are kept unless OVERWRITE = True.')
    json.dump(rec, open(path, 'w'), indent=1)


def run_states(cfg, sessions, n_jobs=4, overwrite=False):
    check_provenance(cfg)
    res = Parallel(n_jobs=n_jobs)(delayed(_states_job)(cfg, m, e, overwrite) for m, e in sessions)
    return pd.Series(res).value_counts().to_dict()


# ------------------------------------------------------------------------------------------------
# Steps 2-4: syllables, trials, raw sequences
# ------------------------------------------------------------------------------------------------
def _mode(b):
    v, c = np.unique(b, return_counts=True)
    return v[np.argmax(c)]


def rescale_sequence(seq, target_length, estimator):
    """segmentation_functions.rescale_sequence, with the same result."""
    n = len(seq)
    if n == target_length:
        return np.array(seq)
    if target_length < n:
        q, r = divmod(n, target_length)
        edges = np.r_[0, np.cumsum([q + 1] * r + [q] * (target_length - r))]
        f = _mode if estimator == 'mode' else np.mean
        return np.array([f(seq[edges[i]:edges[i + 1]]) for i in range(target_length)])
    idx = np.floor(np.linspace(0, n - 1, target_length)).astype(int)
    return np.array(seq)[idx]


def trial_types(df):
    """The two columns of define_trial_types the binning uses: trial_type and sample."""
    c = df['correct']
    assert c.isna().all() or set(c.dropna().unique()) <= {0., 1.}, 'correct must be 0 / 1 / NaN'
    correct_str = pd.Series(np.where(c == 1., 'correct', np.where(c == 0., 'incorrect', None)), index=df.index)
    parts = [correct_str.fillna('unknown'), df['contrast'].astype(str).fillna('unknown'),
             df['block'].astype(str).fillna('unknown'), df['choice'].fillna('unknown')]
    df = df.copy()
    df['trial_type'] = parts[0].str.cat(parts[1:], sep=' ')
    df['sample'] = df['session'] + ' ' + df['trial_id'].astype(str)
    return df


def bin_groups(df, value_cols, estimators, target_length):
    """groupby(GROUP_KEYS)[col].apply(list) + rescale_sequence, for several columns at once."""
    g = df.groupby(GROUP_KEYS, sort=True)
    codes = g.ngroup().to_numpy(dtype=float)         # rows with a NaN key belong to no group
    out = g.size().index.to_frame(index=False)
    ok = np.isfinite(codes) & (codes >= 0)
    c = codes[ok].astype(int)
    order = np.flatnonzero(ok)[np.argsort(c, kind='stable')]
    counts = np.bincount(c, minlength=len(out))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    for col, est in zip(value_cols, estimators):
        # the original went through groupby().apply(list): Python scalars, so ints come back int64
        vals = np.array(df[col].to_numpy()[order].tolist())
        res = np.empty(len(out), dtype=object)
        for i, (s, n) in enumerate(zip(starts, counts)):
            res[i] = rescale_sequence(vals[s:s + n], target_length, est)
        out[col] = res
    return out


@_quiet
def _syllables_job(path, target_length):
    st = pd.read_parquet(path, columns=['mouse_name', 'session', 'correct', 'contrast', 'block', 'choice',
                                        'trial_id', 'broader_label', 'most_likely_states', 'goCueTrigger_times'])
    st = trial_types(st.dropna(subset=['goCueTrigger_times']))
    out = bin_groups(st, ['most_likely_states'], ['mode'], target_length)
    return out.rename(columns={'most_likely_states': 'binned_sequence'})[
        ['mouse_name', 'sample', 'trial_type', 'broader_label', 'binned_sequence']]


@_quiet
def run_syllables(cfg, sessions, target_length, n_jobs=4):
    paths = [states_file(cfg, m, e) for m, e in sessions]
    for p in paths:
        if not os.path.exists(p):
            print(p + ' not available')
    parts = Parallel(n_jobs=n_jobs)(delayed(_syllables_job)(p, target_length) for p in paths if os.path.exists(p))
    head = pd.DataFrame(columns=['mouse_name', 'trial_type', 'broader_label', 'binned_sequence'])
    return pd.concat([head] + parts, ignore_index=True)


@_quiet
def run_trials(cfg, sessions):
    """Trial-level table. Unlike the original, a session without a states file is SKIPPED (the
    original appended the previous session's trials again)."""
    head = pd.DataFrame(columns=['mouse_name', 'session'] + TRIAL_LEVEL_VARS)
    parts = []
    for m, e in sessions:
        p = states_file(cfg, m, e)
        if not os.path.exists(p):
            print(p + ' not available')
            continue
        t = pd.read_parquet(p, columns=['mouse_name'] + TRIAL_LEVEL_VARS)
        t['session'] = e
        parts.append(t.drop_duplicates().dropna(subset=['trial_id']))
    return pd.concat([head] + parts, ignore_index=True)


RAW_VARS = ['Lick count', 'whisker_me', 'l_paw_x_vel', 'l_paw_y_vel', 'r_paw_x_vel', 'r_paw_y_vel']


@_quiet
def _raw_job(path, target_length):
    st = trial_types(pd.read_parquet(path))
    st['whisker_me'] = zscore(np.array(st['whisker_me']), nan_policy='omit', axis=0)
    for name, src, half in (('l_paw_x_vel', 'l_paw_x', True), ('l_paw_y_vel', 'l_paw_y', True),
                            ('r_paw_x_vel', 'r_paw_x', False), ('r_paw_y_vel', 'r_paw_y', False)):
        v = np.full(len(st), np.nan)
        x = st[src] / 2 if half else st[src]
        v[1:] = zscore(np.diff(x), nan_policy='omit', axis=0)
        st[name] = v
    out = bin_groups(st, RAW_VARS, ['mode'] + ['mean'] * 5, target_length)
    return out.rename(columns={v: v + '_binned_sequence' for v in RAW_VARS})


@_quiet
def run_raw(cfg, sessions, target_length, n_jobs=4):
    paths = [states_file(cfg, m, e) for m, e in sessions]
    for p in paths:
        if not os.path.exists(p):
            print(p + ' not available')
    parts = Parallel(n_jobs=n_jobs)(delayed(_raw_job)(p, target_length) for p in paths if os.path.exists(p))
    return pd.concat(parts, ignore_index=True)


def output_name(cfg, kind, target_length=None):
    date = datetime.now().strftime('%d-%m-%Y')
    if kind == 'syllables':
        return f"{cfg['out_root']}{cfg['tag']}{cfg['k']}_k_{target_length}_bin_syllables_{date}"
    if kind == 'trials':
        return f"{cfg['out_root']}all_trials_{cfg['tag']}{date}"
    if kind == 'raw':
        return f"{cfg['out_root']}{cfg['tag']}{target_length}_bin_raw_{date}"
    raise ValueError(kind)
