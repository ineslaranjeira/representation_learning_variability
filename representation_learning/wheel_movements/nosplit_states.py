"""
No split, no merge wheel states -> states files -> syllables, for the LDA test (4_mice/wheel_lda_test.py wheel<K+1>).

The model is the one of 7_wheel_state_definition section F: ONE GaussianMixture on all movements, on the paper model's
features (shared PCA of the signed absolute velocity + log duration, z-scored), fitted on the same fit sample
(wmc.sample, 2000 movements / mouse), K clusters numbered by median peak. It is packed in wheel_movement_clusters'
model format so that wmc.predict / frame_states / write_states work unchanged: SPLIT = 1.0 puts every movement in the
middle group (|balance| <= 1 always), which holds the single mixture.

Writes NEW files only (nothing existing is touched):
  data/wheel_movement_states/model_k<K+1>_nosplit.pkl
  data/wheel_movement_states/most_likely_states_<K+1>_<mouse><eid>.npy           (0 = still, 1..K = cluster)
  data/states_files_both_paws+wheelmov-sep_k8-<K+1>_own_lick-standard/          (syllable pipeline states files)
  data/both_paws+wheelmov-sep_k8-<K+1>_own_lick-standard_proficient_8_k_10_bin_syllables_<date>

    python nosplit_states.py 6 [model|states|syllables]...      (default: all three)

SIDED variant (--sided): the same K clusters, each split post hoc by the sign of the WHOLE movement's travel balance
(+ = CCW): state = 1 + 2 * cluster + (balance > 0), so K clusters -> 2K states + still (K_WHEEL = 2K + 1). Uses the
saved no-split model (run without --sided first); writes most_likely_states_<2K+1>_* and its syllables, plus
model_k<2K+1>_nosplit_sided.pkl (the no-split model with sided = True, for names / colours).

    python nosplit_states.py 6 --sided [states|syllables]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PI = pathlib.Path(__file__).resolve().parents[2] / 'paper-individuality'
sys.path.insert(0, str(PI / 'segmentation'))
import wheel_movement_clusters as wmc
import syllable_pipeline as sp

DATASET = 'proficient'


def build_model(k):
    paper = wmc.load_model(DATASET, 21)
    coh = wmc.fit_cohort()
    out = Parallel(n_jobs=12)(delayed(wmc.session_shapes)(DATASET, m, e) for m, e in zip(coh.mouse, coh.eid))
    S, info = wmc.sample(coh, {e: s for (s, _), e in zip(out, coh.eid)}, {e: i for (_, i), e in zip(out, coh.eid)})
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(k, covariance_type='full', n_init=wmc.N_INIT, random_state=wmc.SEED).fit(
        wmc._features(paper, S, info.duration))
    raw = gm.predict(wmc._features(paper, S, info.duration))
    b = wmc.balance(S)
    d = (pd.DataFrame({'raw': raw, 'side_index': b, 'peak': info.peak.values, 'duration': info.duration.values,
                       'reversals': [wmc.reversals(a) for a in S], 'epoch': info.epoch.values})
         .groupby('raw').agg(fraction=('raw', lambda x: len(x) / len(raw)), side_index=('side_index', 'median'),
                             peak_rad_s=('peak', 'median'), reversals=('reversals', 'median'),
                             duration_s=('duration', 'median'),
                             response=('epoch', lambda e: (e == 'response').mean()),
                             post_feedback=('epoch', lambda e: (e == 'post-feedback').mean()))
         .sort_values('peak_rad_s'))
    model = {key: paper[key] for key in ('pca', 'n_pc', 'MU', 'SD', 'NT', 'PAD_S', 'VAR', 'SIGN_RIGHT', 'sign_check')}
    model.update(SPLIT=1.0, KG=(0, k, 0), K=k, OFF=np.array([0, 0, 0]), GM={1: gm},
                 REL={int(r): i for i, r in enumerate(d.index)}, variant='nosplit')
    d = d.reset_index().rename(columns={'raw': 'mixture_label'})
    d.index.name = 'cluster'
    d['side'] = 'both'                        # direction is not a property of these clusters (each ~50 / 50)
    d['vigor'] = pd.cut(d.peak_rad_s, [0, *wmc.VIGOR_CUTS, np.inf], labels=list(wmc.VIGORS)).astype(str)
    d['jitter'] = pd.cut(d.reversals, [-1, *wmc.JITTER_CUTS, 99], labels=list(wmc.HATCH)).astype(str)
    d['label'] = [f'U{c + 1}' for c in d.index]
    model['desc'] = d
    # sanity: the packed model gives the mixture's labels, renumbered
    assert (wmc.predict(model, S, info.duration) == np.array([model['REL'][r] for r in raw])).all()
    path = wmc.states_dir(DATASET) + f'model_k{k + 1}_nosplit.pkl'
    assert not os.path.exists(path), f'{path} exists'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(d.round(2).to_string())
    print('wrote', path)
    return model


def load(k):
    with open(wmc.states_dir(DATASET) + f'model_k{k + 1}_nosplit.pkl', 'rb') as f:
        return pickle.load(f)


def write_states(k):
    model = load(k)
    taken = [f for f in os.listdir(wmc.states_dir(DATASET)) if f.startswith(f'most_likely_states_{k + 1}_')]
    assert not taken, f'K_WHEEL = {k + 1} states files already exist ({len(taken)})'
    cfg = sp.resolve(DATASET, 'both_paws', 'own')
    sessions = [s for s in sp.sessions_with_inputs(cfg) if wmc.has_movements(DATASET, *s)]
    print(wmc.write_states(DATASET, model, sessions, groupings=(None,), n_jobs=8), 'files per session: count')


def _sided_one(model, mouse, eid, kw):
    f = wmc.states_dir(DATASET) + f'most_likely_states_{kw}_{mouse}{eid}.npy'
    if os.path.exists(f):
        return 0
    t, v, iv, _ = wmc.load_session(DATASET, mouse, eid)
    S, info = wmc.shapes(t, v, iv)
    out = np.zeros(len(t), int)
    if len(info):
        c = wmc.predict(model, S, info.duration)
        st = 1 + 2 * c + (wmc.balance(S) > 0)
        for on, off, s_ in zip(info.onset, info.offset, st):
            i0, i1 = np.searchsorted(t, [on, off])
            out[i0:i1] = s_
    np.save(f, np.array([out, t]))
    return 1


def write_states_sided(k):
    model = load(k)
    kw = 2 * k + 1
    path = wmc.states_dir(DATASET) + f'model_k{kw}_nosplit_sided.pkl'
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            pickle.dump(dict(model, sided=True, K_WHEEL=kw), f)
    taken = [f for f in os.listdir(wmc.states_dir(DATASET)) if f.startswith(f'most_likely_states_{kw}_')]
    assert not taken, f'K_WHEEL = {kw} states files already exist ({len(taken)})'
    cfg = sp.resolve(DATASET, 'both_paws', 'own')
    sessions = [s for s in sp.sessions_with_inputs(cfg) if wmc.has_movements(DATASET, *s)]
    n = Parallel(n_jobs=8)(delayed(_sided_one)(model, m, e, kw) for m, e in sessions)
    print(f'{sum(n)} sided states files written (K_WHEEL = {kw})')


def syllables(k, k_wheel=None):
    cfg = sp.resolve(DATASET, 'both_paws', 'own', 'standard', k=8, wheel=True, k_wheel=k_wheel or k + 1,
                     wheel_segmentation='movement', wheel_binning='separate')
    sessions = sp.sessions_with_inputs(cfg)
    print(sp.describe(cfg))
    print(f'{len(sessions)} sessions with all their states')
    print(sp.run_states(cfg, sessions, n_jobs=4, overwrite=False))
    syl = sp.run_syllables(cfg, sessions, 10, n_jobs=4)
    path = sp.output_name(cfg, 'syllables', 10)
    syl.to_parquet(path, compression='gzip')
    print(f'{len(syl)} rows -> {path}')


if __name__ == '__main__':
    k = int(sys.argv[1])
    if '--sided' in sys.argv:
        steps = [a for a in sys.argv[2:] if a != '--sided'] or ['states', 'syllables']
        if 'states' in steps:
            write_states_sided(k)
        if 'syllables' in steps:
            syllables(k, k_wheel=2 * k + 1)
        sys.exit()
    steps = sys.argv[2:] or ['model', 'states', 'syllables']
    if 'model' in steps:
        build_model(k)
    if 'states' in steps:
        write_states(k)
    if 'syllables' in steps:
        syllables(k)
