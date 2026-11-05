"""
Wheel MOVEMENT clusters -- the code behind 3.4_wheel_movement_clusters.ipynb.

Not to be confused with 3.3's wheel WAVELET clusters (wavelet_clusters.py, VAR = 'wheel'), which
label every frame from its wavelet power. Here the unit is a whole wheel movement:

  1. movements  IBL's alf/_ibl_wheelMoves.intervals (the detector behind firstMovement_times),
                padded by PAD_S on each side, cached with the session's trials (`fetch`)
  2. shape      the movement's signed wheel velocity from the design matrix, in rad/s (NOT
                divided by its peak: vigor is part of the shape), linearly time-normalised to NT bins
  3. fit        a sample of N_PER_MOUSE movements from every mouse of the fit cohort, split equally
                across its sessions; PCA to VAR of the variance, + log10 duration, standardised.
                Movements are first split by their travel balance into -side / balanced / +side
                (|balance| > SPLIT): with absolute velocity a single mixture builds nested, centred
                clusters of increasing speed that hold both directions. One Gaussian mixture per
                group, KG clusters each.
  4. describe   per cluster: the stimulus side it reports (the wheel sign is MEASURED on correct
                response-epoch movements), vigor (median peak speed, VIGOR_CUTS) and jitter
                (direction reversals); clusters are numbered left -> both -> right, by vigor
  5. states     every frame of a session: 0 = not in a movement ('still'), 1..K = its cluster, or a
                GROUPING of the clusters (e.g. side x vigor). Same file layout as 3.3
                (most_likely_states_<k>_<mouse><eid>.npy = [states, bins]), so syllable_pipeline reads
                them like any other states, with k = number of states including still.

Run as a script to fill the movement cache of a dataset:
    python wheel_movement_clusters.py fetch proficient
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

DATA = '/home/ines/repositories/representation_learning_variability/paper-individuality/data/'
PI = os.path.dirname(DATA.rstrip('/')) + '/'

# each dataset: its 60 Hz design matrices (velocity column) and its output root, as syllable_pipeline's DATASETS
DATASETS = {
    'proficient': dict(root=DATA, design_matrices=DATA + 'design_matrices/', velocity='avg_wheel_vel'),
}
MOVES_DIR = 'wheel_moves/'                    # <root>/wheel_moves/{intervals,trials}_<eid>_<mouse>
STATES_DIR = 'wheel_movement_states/'         # <root>/wheel_movement_states/most_likely_states_<k>_<mouse><eid>.npy

PAD_S = 0.05            # s added before onset / after offset (IBL intervals often start late / end early)
NT = 200                # bins per time-normalised movement
VAR = 0.95              # PCA variance kept (as the paw states' CUTOFF)
N_PER_MOUSE = 2000      # movements per mouse in the fit, split equally across its sessions
SPLIT = 0.5             # |travel balance| above which a movement is one-sided
KG = (7, 6, 7)          # clusters for -side / balanced / +side groups
N_INIT, SEED = 5, 0
VIGOR_CUTS = (1.0, 2.75)    # rad/s, median peak speed of a cluster: low | mid | high
JITTER_CUTS = (1, 3)        # direction reversals: <= 1 smooth, <= 3 corrective, more jittery
SIDES = ('left', 'both', 'right')
VIGORS = ('low', 'mid', 'high')
# merged states (state 0 = still in all of them):
#   side_vigor   every cluster merged by its side x vigor label (9 groups -> k = 10)
#   sided_both2  one-sided clusters kept as they are, the balanced ('both': mostly corrective / jittery)
#                clusters merged into 'both low' and 'both high' (mid + high) -> k = 17. Merging only the
#                balanced clusters keeps 90 % of the 20-cluster information (NMI) and gives the highest
#                mouse ICC of occupancy (0.70 vs 0.64 for all 20; wheel_movements/7_wheel_state_definition, Part A)
#   small_side_vigor  every low-vigor cluster (tiny, mostly post-feedback twitches that co-occur with a still
#                paw, whatever their side) merged into one 'small' state; mid / high kept by side -> k = 8
GROUPINGS = ('side_vigor', 'sided_both2', 'small_side_vigor')
# colours: the paper's state-palette rule with the wheel's own hue pair (teal = reports left,
# orange = reports right; not the paw's violet / amber, no blue / red = LD1). Chroma scaled down: at
# full chroma the pale teal turns neon. See representation_learning/wheel_movements/6_segmentation_snippets.
WHEEL_HUES = (195, 45)
CHROMA_SCALE = 0.55
HATCH = {'smooth': '', 'corrective': '..', 'jittery': '////'}


# ------------------------------------------------------------------------------------------------
# Sessions and the movement cache
# ------------------------------------------------------------------------------------------------
def moves_dir(dataset):
    return DATASETS[dataset]['root'] + MOVES_DIR


def states_dir(dataset):
    return DATASETS[dataset]['root'] + STATES_DIR


def fit_cohort():
    """(mouse, eid) the fit samples from: the proficient LDA cohort -- the paper's syllable sessions
    minus the QC sheet's 'filtered out', mice with >= 3 sessions (4_mice.functions.filter_sequences)."""
    sys.path.insert(0, PI + '4_mice')
    from functions import qc_exclusions
    d = pd.read_parquet(DATA + '8_k_10_bin_syllables_02-10-2026', columns=['mouse_name', 'sample'])
    s = pd.DataFrame({'mouse': d.mouse_name, 'eid': d['sample'].str[:36]}).drop_duplicates()
    s = s[~s.eid.isin(qc_exclusions('Proficient', 'filtered_out', verbose=False))]
    s = s[s.groupby('mouse').eid.transform('size') >= 3]
    return s.sort_values(['mouse', 'eid']).reset_index(drop=True)


def _fetch_one(dataset, eid, mouse):
    d = moves_dir(dataset)
    if os.path.exists(d + f'intervals_{eid}_{mouse}.npy'):
        return 'exists'
    from one.api import ONE
    one = ONE(mode='remote')
    try:
        iv = one.load_object(eid, 'wheelMoves', collection='alf')['intervals']
        tr = one.load_object(eid, 'trials', collection='alf')
        n = len(tr['intervals'])
        tr = type(tr)({k: v for k, v in tr.items() if len(v) == n}).to_df()     # some sessions carry an empty attribute
        tr.to_parquet(d + f'trials_{eid}_{mouse}')
        np.save(d + f'intervals_{eid}_{mouse}.npy', iv)
        return 'ok'
    except Exception as e:
        return f'error: {type(e).__name__}: {e}'[:200]


def fetch(dataset, sessions, n_jobs=8):
    """Download and cache IBL's wheelMoves intervals and the trials of (mouse, eid) sessions."""
    os.makedirs(moves_dir(dataset), exist_ok=True)
    res = Parallel(n_jobs=n_jobs)(delayed(_fetch_one)(dataset, e, m) for m, e in sessions)
    return pd.Series(res, index=pd.MultiIndex.from_tuples(sessions, names=['mouse', 'eid']))


def has_movements(dataset, mouse, eid):
    return os.path.exists(moves_dir(dataset) + f'intervals_{eid}_{mouse}.npy')


def load_session(dataset, mouse, eid):
    """60 Hz times and wheel velocity of the design matrix, movement intervals, trials."""
    ds = DATASETS[dataset]
    dm = pd.read_parquet(ds['design_matrices'] + f'design_matrix_{eid}_{mouse}', columns=['Bin', ds['velocity']])
    iv = np.load(moves_dir(dataset) + f'intervals_{eid}_{mouse}.npy')
    tr = pd.read_parquet(moves_dir(dataset) + f'trials_{eid}_{mouse}')
    return dm.Bin.values, dm[ds['velocity']].values.astype(float), iv, tr


# ------------------------------------------------------------------------------------------------
# Movements -> shapes
# ------------------------------------------------------------------------------------------------
def shapes(t, v, intervals, pad=PAD_S, nt=NT):
    """Signed, time-normalised velocity (rad/s) of every movement inside the design matrix's time
    range (it only covers the cameras' overlap). -> (shapes (n, nt), table: movement, onset, offset,
    duration, peak)."""
    from scipy.interpolate import interp1d
    S, rows = [], []
    for k, (on, off) in enumerate(intervals):
        if on - pad < t[0] or off + pad > t[-1]:
            continue
        i0, i1 = np.searchsorted(t, [on - pad, off + pad])
        vv = v[i0:i1]
        if len(vv) < 4 or not np.isfinite(vv).all():
            continue
        x = np.linspace(0, 1, len(vv))
        S.append(interp1d(x, vv)(np.linspace(0, 1, nt)))
        rows.append(dict(movement=k, onset=on, offset=off, duration=off - on, peak=np.abs(vv).max()))
    return np.array(S, np.float32).reshape(-1, nt), pd.DataFrame(rows)


def stim_times(trials):
    """stimOn_times, filled with goCue_times where missing (some sessions have none at all)."""
    return trials.stimOn_times.fillna(trials.goCue_times)


def epoch_of(on, trials):
    """(epoch, trial index) of a time: pre-stim (trial start -> stimulus), response (stimulus ->
    feedback) or post-feedback (feedback -> next trial start)."""
    i = np.searchsorted(trials.intervals_0.values, on, side='right') - 1
    if i < 0:
        return 'post-feedback', -1
    r = trials.iloc[i]
    if on < stim_times(trials).iloc[i]:
        return 'pre-stim', i
    if on <= r.feedback_times:
        return 'response', i
    return 'post-feedback', i


def session_shapes(dataset, mouse, eid):
    """Shapes of a session + per-movement table with trial context (epoch, trial, stimulus side, correct)."""
    t, v, iv, tr = load_session(dataset, mouse, eid)
    S, info = shapes(t, v, iv)
    info['mouse'], info['eid'] = mouse, eid
    if len(info):
        ep = [epoch_of(o, tr) for o in info.onset]
        info['epoch'] = [a for a, _ in ep]
        info['trial'] = [b for _, b in ep]
        j = np.maximum(info.trial, 0)
        info['stim_side'] = np.where(tr.contrastRight.notna().values, 'right', 'left')[j]
        info['correct'] = (tr.feedbackType.values == 1)[j]
    return S, info


def balance(S):
    """Travel balance per movement: (+ travel - - travel) / total, in wheel coordinates."""
    pos = np.clip(S, 0, None).sum(1)
    neg = np.clip(-S, 0, None).sum(1)
    return (pos - neg) / (pos + neg)


def side_group(S, split=SPLIT):
    b = balance(S)
    return np.where(b > split, 2, np.where(b < -split, 0, 1))


def reversals(a, frac=0.1):
    """Direction reversals of one shape, ignoring wiggles below frac x its peak speed."""
    sg = np.sign(np.where(np.abs(a) > frac * np.abs(a).max(), a, 0))
    sg = sg[sg != 0]
    return int((np.diff(sg) != 0).sum())


# ------------------------------------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------------------------------------
def sample(cohort, all_s, all_i, n_per_mouse=N_PER_MOUSE, seed=SEED):
    """N_PER_MOUSE movements from every mouse, split equally across its sessions (uniform within)."""
    rng = np.random.default_rng(seed)
    Ss, Is = [], []
    for mouse, g in cohort.groupby('mouse'):
        per = int(np.ceil(n_per_mouse / len(g)))
        for e in g.eid:
            n = len(all_i[e])
            idx = rng.choice(n, min(per, n), replace=False)
            Ss.append(all_s[e][idx])
            Is.append(all_i[e].iloc[idx])
    return np.concatenate(Ss), pd.concat(Is, ignore_index=True)


def _features(model, S, duration):
    F = np.column_stack([model['pca'].transform(S)[:, :model['n_pc']], np.log10(duration)])
    return (F - model['MU']) / model['SD']


def _raw_labels(model, S, duration):
    F, g_ = _features(model, S, duration), side_group(S, model['SPLIT'])
    out = np.empty(len(S), int)
    for g in range(3):
        if (g_ == g).any():
            out[g_ == g] = model['GM'][g].predict(F[g_ == g]) + model['OFF'][g]
    return out


def fit(S, info):
    """PCA + side-split mixtures on a fit sample, then the cluster description and numbering."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    pca = PCA().fit(S)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), VAR) + 1)
    F0 = np.column_stack([pca.transform(S)[:, :n_pc], np.log10(info.duration)])
    model = dict(pca=pca, n_pc=n_pc, MU=F0.mean(0), SD=F0.std(0), SPLIT=SPLIT, KG=KG, K=int(sum(KG)),
                 OFF=np.r_[0, np.cumsum(KG)[:-1]], NT=NT, PAD_S=PAD_S, VAR=VAR)
    F = (F0 - model['MU']) / model['SD']
    grp = side_group(S)
    model['GM'] = {g: GaussianMixture(KG[g], covariance_type='full', n_init=N_INIT, random_state=SEED).fit(F[grp == g])
                   for g in range(3)}
    raw = _raw_labels(model, S, info.duration)

    # which wheel sign reports a right stimulus: correct response-epoch movements
    b = balance(S)
    c = info[(info.epoch == 'response') & info.correct & (info.trial >= 0)]
    p_right = (b[c.index][c.stim_side == 'right'] > 0).mean()
    p_left = (b[c.index][c.stim_side == 'left'] > 0).mean()
    model['SIGN_RIGHT'] = 1 if p_right > p_left else -1
    model['sign_check'] = dict(p_pos_given_right=float(p_right), p_pos_given_left=float(p_left))

    d = pd.DataFrame({'raw': raw, 'side_index': b * model['SIGN_RIGHT'], 'peak': info.peak.values,
                      'duration': info.duration.values, 'reversals': [reversals(a) for a in S],
                      'epoch': info.epoch.values})
    desc = d.groupby('raw').agg(fraction=('raw', lambda x: len(x) / len(d)), side_index=('side_index', 'median'),
                                peak_rad_s=('peak', 'median'), reversals=('reversals', 'median'),
                                duration_s=('duration', 'median'),
                                response=('epoch', lambda e: (e == 'response').mean()),
                                post_feedback=('epoch', lambda e: (e == 'post-feedback').mean()))
    desc['side'] = np.where(desc.side_index > .5, 'right', np.where(desc.side_index < -.5, 'left', 'both'))
    desc['vigor'] = pd.cut(desc.peak_rad_s, [0, *VIGOR_CUTS, np.inf], labels=list(VIGORS)).astype(str)
    desc['jitter'] = pd.cut(desc.reversals, [-1, *JITTER_CUTS, 99], labels=list(HATCH)).astype(str)
    desc = (desc.assign(_s=desc.side.map({s: i for i, s in enumerate(SIDES)}))
            .sort_values(['_s', 'peak_rad_s']).drop(columns='_s'))
    model['REL'] = {int(r): i for i, r in enumerate(desc.index)}       # mixture label -> cluster number
    desc = desc.reset_index().rename(columns={'raw': 'mixture_label'})
    desc.index.name = 'cluster'
    desc['label'] = desc.side + ' ' + desc.vigor + ' ' + desc.jitter
    model['desc'] = desc
    return model


def predict(model, S, duration):
    """Cluster number (0..K-1) of each movement."""
    return np.array([model['REL'][r] for r in _raw_labels(model, S, duration)], int)


# ------------------------------------------------------------------------------------------------
# States
# ------------------------------------------------------------------------------------------------
def group_of(model, grouping):
    """Group name of every cluster."""
    d = model['desc']
    names = [f'{c} {d.label[c]}' for c in range(model['K'])]
    if grouping is None:
        return names
    if grouping == 'side_vigor':
        return list(d.side + ' ' + d.vigor)
    if grouping == 'small_side_vigor':
        return ['small' if v == 'low' else f'{s} {v}' for s, v in zip(d.side, d.vigor)]
    if grouping == 'sided_both2':
        return [n if s != 'both' else 'both ' + ('low' if v == 'low' else 'high')
                for n, s, v in zip(names, d.side, d.vigor)]
    raise ValueError(f'grouping must be None or one of {GROUPINGS}, got {grouping!r}')


def state_names(model, grouping=None):
    """Names of the states, state 0 = still; groups in cluster order (left -> both -> right, by vigor)."""
    return ['still'] + list(dict.fromkeys(group_of(model, grouping)))


def state_map(model, grouping=None):
    """Cluster number -> state number (1..)."""
    if grouping is None:
        return {c: c + 1 for c in range(model['K'])}
    names = state_names(model, grouping)
    return {c: names.index(g) for c, g in enumerate(group_of(model, grouping))}


def frame_states(model, t, v, intervals, grouping=None):
    """State of every design-matrix frame: 0 = not in a movement, 1.. = the movement's cluster (or group)."""
    S, info = shapes(t, v, intervals)
    out = np.zeros(len(t), int)
    if len(info):
        m = state_map(model, grouping)
        for on, off, c in zip(info.onset, info.offset, predict(model, S, info.duration)):
            i0, i1 = np.searchsorted(t, [on, off])
            out[i0:i1] = m[c]
    return out


def colours(model, grouping=None):
    """State colours (still first, white): the paper's state-palette rule with WHEEL_HUES."""
    sys.path.insert(0, PI)
    import paper_style as ps
    d = model['desc']
    if grouping is None:
        vig, side = np.log(d.peak_rad_s.values), d.side_index.values
    else:
        g = np.array(group_of(model, grouping))
        names = state_names(model, grouping)[1:]
        w = d.fraction.values
        vig = np.array([np.log(np.average(d.peak_rad_s.values[g == n], weights=w[g == n])) for n in names])
        side = np.array([np.average(d.side_index.values[g == n], weights=w[g == n]) for n in names])
    return ['#FFFFFF'] + ps.state_palette(vig, gap=-CHROMA_SCALE * side, hues=WHEEL_HUES)


def k_of(model, grouping=None):
    """The k of the states files = number of states, still included."""
    return len(state_names(model, grouping))


def save_model(dataset, model):
    d = states_dir(dataset)
    os.makedirs(d, exist_ok=True)
    with open(d + f"model_k{k_of(model)}.pkl", 'wb') as f:
        pickle.dump(model, f)
    info = {'K_clusters': model['K'], 'sign_check': model['sign_check'],
            'states': {str(k_of(model, g)): dict(grouping=g, names=state_names(model, g)) for g in [None, *GROUPINGS]}}
    json.dump(info, open(d + 'states.json', 'w'), indent=1)


def load_model(dataset, k=21):
    with open(states_dir(dataset) + f'model_k{k}.pkl', 'rb') as f:
        return pickle.load(f)


def _write_one(dataset, model, mouse, eid, groupings, overwrite):
    d = states_dir(dataset)
    t, v, iv, _ = load_session(dataset, mouse, eid)
    done = []
    for g in groupings:
        f = d + f'most_likely_states_{k_of(model, g)}_{mouse}{eid}.npy'
        if os.path.exists(f) and not overwrite:
            continue
        np.save(f, np.array([frame_states(model, t, v, iv, g), t]))
        done.append(k_of(model, g))
    return done


def write_states(dataset, model, sessions, groupings=(None, *GROUPINGS), n_jobs=8, overwrite=False):
    """Frame states of every (mouse, eid), one file per grouping, in 3.3's layout."""
    os.makedirs(states_dir(dataset), exist_ok=True)
    res = Parallel(n_jobs=n_jobs)(delayed(_write_one)(dataset, model, m, e, groupings, overwrite) for m, e in sessions)
    return pd.Series([len(r) for r in res]).value_counts().to_dict()


if __name__ == '__main__':
    if sys.argv[1:2] == ['fetch']:
        dataset = sys.argv[2]
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import syllable_pipeline as sp
        # every session the default syllable run of this dataset uses, plus the fit cohort
        cfg = sp.resolve(dataset, 'both_paws', 'own')
        sess = sorted(set(sp.sessions_with_inputs(cfg)) | set(map(tuple, fit_cohort()[['mouse', 'eid']].values)))
        res = fetch(dataset, sess)
        print(len(sess), 'sessions:', res.str.split(':').str[0].value_counts().to_dict())
        print(res[res.str.startswith('error')])
