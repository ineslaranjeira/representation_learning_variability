"""Which hyperparameters actually matter for the 2-state Gaussian HMM?

Every knob GaussianHMM exposes, perturbed one at a time, judged by what it does to the
SEGMENTATION (frame agreement + syllable duration) rather than to the likelihood -- the
lesson from the lag work being that held-out LL moves when the segmentation does not.

  num_states                       structural: 2 vs 3 vs 4
  transition_matrix_stickiness     kappa, as in the AR work
  method                           kmeans (available here) vs prior
  emission_prior_scale             NIW scale -- can floor the state covariance
  emission_prior_concentration     NIW concentration on the mean
  transition_matrix_concentration  Dirichlet on the transition rows

Run from this directory.
"""
import os, sys, time
sys.path.insert(0, '..')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
import jax.random as jr
from dynamax.hidden_markov_model import GaussianHMM
from hmm_dynamic_functions import (load_fit_variable, prepare_batches, orient_states,
                                   dwell_times, pick_fold)
from hmm_gaussian_functions import cross_validate_gaussmodel

DM = '../../data/design_matrices/'
AR = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
PRE = 'best_results_whisker_me_'
FPS, NB, ITERS = 60.0, 5, 100

WANT = ['034e726f',   # AR was fine here
        'a8a8af78',   # AR flickered (17 ms)
        '02fbb6da',   # AR under-segmented (1950 ms)
        '15763234']   # Gaussian went long (850 ms)

VARIANTS = [
    ('baseline',              {}, 2, 'kmeans'),
    ('kappa=1e3',             {'transition_matrix_stickiness': 1e3}, 2, 'kmeans'),
    ('kappa=1e5',             {'transition_matrix_stickiness': 1e5}, 2, 'kmeans'),
    ('init=prior',            {}, 2, 'prior'),
    ('emis_prior_scale=1e-2', {'emission_prior_scale': 1e-2}, 2, 'kmeans'),
    ('emis_prior_scale=1.0',  {'emission_prior_scale': 1.0}, 2, 'kmeans'),
    ('emis_prior_conc=1.0',   {'emission_prior_concentration': 1.0}, 2, 'kmeans'),
    ('trans_conc=10',         {'transition_matrix_concentration': 10.0}, 2, 'kmeans'),
    ('num_states=3',          {}, 3, 'kmeans'),
    ('num_states=4',          {}, 4, 'kmeans'),
]

disk = {n[-36:][:8]: (n[len(PRE):-36], n[-36:])
        for n in sorted(os.listdir(AR)) if n.startswith(PRE)}

rows = []
for e8 in WANT:
    if e8 not in disk:
        print(f'!! {e8} missing', flush=True); continue
    mouse, eid = disk[e8]
    dm = load_fit_variable(DM, eid, mouse, ['whisker_me'], True)
    sa, train_em, _ = prepare_batches(dm, NB)
    ed = sa.shape[1]
    fold_len = train_em.shape[1]
    ref = None
    for label, kw, K, meth in VARIANTS:
        t0 = time.time()
        try:
            model = GaussianHMM(K, ed, **kw)
            vll, fp, _, bll = cross_validate_gaussmodel(
                model, jr.PRNGKey(0), train_em, NB, 'em', method=meth, num_iters=ITERS)
            vll = np.asarray(vll) / fold_len
            fold = pick_fold(vll)
            m2 = GaussianHMM(K, ed, **kw)
            p, _ = m2.initialize(key=jr.PRNGKey(0), method='prior',
                                 initial_probs=fp[0].probs[fold],
                                 transition_matrix=np.asarray(fp[1].transition_matrix)[fold],
                                 emission_means=fp[2].means[fold],
                                 emission_covariances=fp[2].covs[fold])
            s = np.asarray(m2.most_likely_states(p, sa))
            mu = np.asarray(fp[2].means)[fold].ravel()
            if K == 2:
                s = orient_states(s, sa)
            else:
                s = np.argsort(np.argsort(mu))[s]      # relabel so the top state is highest-mean
            dw = dwell_times(s)
            if ref is None:
                ref = s.copy()
            # K>2: binarise to top-state-vs-rest so it is comparable to the 2-state answer
            sb = (s == (K - 1)).astype(int) if K > 2 else s
            rows.append(dict(mouse=mouse, eid=e8, variant=label, num_states=K,
                             med_dwell_ms=float(np.median(dw)) * 1000 / FPS,
                             n_seg=len(dw), occ_top=float(np.mean(s == (K - 1))),
                             raw_ll=float(np.nanmean(vll)),
                             agree_vs_baseline=float(np.mean(sb == ref)),
                             state_means=np.round(np.sort(mu), 3).tolist(),
                             secs=time.time() - t0, error=''))
            print(f'{mouse} {e8} {label:22s} dwell {rows[-1]["med_dwell_ms"]:7.1f} ms  '
                  f'nseg {len(dw):6d}  LL {rows[-1]["raw_ll"]:+.4f}  '
                  f'agree {rows[-1]["agree_vs_baseline"]:.4f}  ({time.time()-t0:.0f}s)',
                  flush=True)
        except Exception as ex:
            print(f'{mouse} {e8} {label:22s} FAILED {type(ex).__name__}: {ex}', flush=True)
            rows.append(dict(mouse=mouse, eid=e8, variant=label, num_states=K,
                             error=f'{type(ex).__name__}: {ex}'))

t = pd.DataFrame(rows)
t.to_csv('gaussian_hyperparams.csv', index=False)
ok = t[t.error == ''] if 'error' in t else t
print('\n=== effect of each knob on the SEGMENTATION (median over sessions) ===')
print(ok.groupby('variant').agg(
    n=('eid', 'count'), agree=('agree_vs_baseline', 'median'),
    dwell_ms=('med_dwell_ms', 'median'), n_seg=('n_seg', 'median'),
    raw_ll=('raw_ll', 'median')).to_string(float_format=lambda x: f'{x:.4f}'))
