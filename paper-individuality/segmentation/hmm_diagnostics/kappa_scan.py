"""MoSeq-style kappa scan for the whisker AR-HMM.

Fits typical sessions across a properly-scaled kappa grid at a fixed lag and records
median syllable duration + held-out bits. Produces the evidence for setting kappa=0:
the duration-vs-kappa curve sits above the model-free changepoint target (~450 ms)
already at kappa=0.

Also dumps the decoded state sequence for one example session at the lowest and
highest kappa, for the side-by-side snippet figure.

Env: OUTCSV, OUTNPZ.  Run from this directory.
"""
import sys, os, re, pickle, csv, time
SEG = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.insert(0, SEG)
os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np, pandas as pd
from scipy.stats import zscore
import jax.numpy as jnp, jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM
from segmentation_functions import compute_inputs, cross_validate_armodel

F, DM = '../data/hmm/grid_search/5_prior_em_zsc_True', '../data/design_matrices/'
NB = 5
LAG = 10                                   # the corrected rule's modal lag
KAPPAS = [0.0, 1e3, 1e4, 5e4, 1e5, 2e5]
SESSIONS = ['7cec9792', 'c51f34d8', 'edd22318', '19e66dc9', '93ad879a', 'dfd8e7df']
EXAMPLE = 'edd22318'                       # dumped for the snippet figure
OUTCSV, OUTNPZ = os.environ['OUTCSV'], os.environ['OUTNPZ']


def dwell_of(s):
    ch = np.where(np.diff(s) != 0)[0]
    return np.diff(np.concatenate(([0], ch + 1, [len(s)])))


files = {}
for f in os.listdir(F):
    if not f.startswith('best_results_whisker_me_'):
        continue
    m = re.search(r'([0-9a-f-]{36})$', f)
    if m and m.group(1)[:8] in SESSIONS:
        files[m.group(1)] = f[len('best_results_whisker_me_'):-36]

fh = open(OUTCSV, 'w', newline='')
w = csv.DictWriter(fh, fieldnames=['mouse', 'eid', 'lag', 'kappa', 'bits_LL',
                                   'med_dwell_frames', 'med_dwell_ms', 'mean_dwell_ms',
                                   'n_seg', 'diag', 'secs'])
w.writeheader()
dump = {}
t0 = time.time()
for eid, mouse in sorted(files.items(), key=lambda kv: kv[1]):
    arr = zscore(np.array(pd.read_parquet(DM + f'design_matrix_{eid}_{mouse}')[['whisker_me']].dropna()),
                 axis=0, nan_policy='omit')
    nt, ed = arr.shape
    short = np.array(arr[:(nt // NB) * NB])
    train_em = jnp.stack(jnp.split(short, NB))
    inp = compute_inputs(short, LAG, ed)
    train_in = jnp.stack(jnp.split(inp, NB))
    fold_len = len(short) / NB
    for kap in KAPPAS:
        t1 = time.time()
        try:
            m = LinearAutoregressiveHMM(2, ed, num_lags=LAG, transition_matrix_stickiness=kap)
            vll, fitp, _, bll = cross_validate_armodel(m, jr.PRNGKey(0), train_em, train_in,
                                                       'prior', NB, 'em')
            bits = (np.asarray(vll) - np.asarray(bll)) / fold_len * np.log(2)
            fold = int(np.nanargmax(bits))
            A = np.asarray(fitp[1].transition_matrix)[fold]
            mdl = LinearAutoregressiveHMM(2, ed, num_lags=LAG, transition_matrix_stickiness=kap)
            p, _ = mdl.initialize(key=jr.PRNGKey(0), method='prior',
                                  initial_probs=fitp[0].probs[fold], transition_matrix=A,
                                  emission_weights=fitp[2].weights[fold],
                                  emission_biases=fitp[2].biases[fold],
                                  emission_covariances=fitp[2].covs[fold], emissions=short)
            s = np.asarray(mdl.most_likely_states(p, short, inp))
            if short[s == 1, 0].mean() < short[s == 0, 0].mean():
                s = 1 - s
            d = dwell_of(s)
            w.writerow(dict(mouse=mouse, eid=eid, lag=LAG, kappa=kap,
                            bits_LL=round(float(np.nanmean(bits)), 5),
                            med_dwell_frames=float(np.median(d)),
                            med_dwell_ms=round(float(np.median(d)) / 60 * 1000, 1),
                            mean_dwell_ms=round(float(d.mean()) / 60 * 1000, 1),
                            n_seg=len(d), diag=round(float(np.mean(np.diag(A))), 6),
                            secs=round(time.time() - t1, 1)))
            if eid[:8] == EXAMPLE and kap in (KAPPAS[0], KAPPAS[-1]):
                dump[f'states_k{kap:g}'] = s.astype(np.int8)
                dump['signal'] = short[:, 0].astype(np.float32)
                dump[f'bits_k{kap:g}'] = float(np.nanmean(bits))
                dump['mouse'] = mouse
                dump['eid'] = eid
        except Exception as e:
            w.writerow(dict(mouse=mouse, eid=eid, lag=LAG, kappa=kap,
                            bits_LL=f'ERR {type(e).__name__}', secs=round(time.time() - t1, 1)))
        fh.flush()
        print(f'{mouse} {eid[:8]} k={kap:g}  {time.time()-t1:.0f}s', flush=True)
fh.close()
if dump:
    np.savez_compressed(OUTNPZ, **dump)
print(f'DONE in {time.time()-t0:.0f}s')
