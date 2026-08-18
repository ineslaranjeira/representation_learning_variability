"""Does kappa change the SEGMENTATION? (whisker AR-HMM and lick Poisson-HMM)

For each session, fit across a kappa grid and measure the state-sequence agreement
against the kappa=0 fit. This is the evidence for "kappa=0 is defensible because
varying it does not change the segmentation" -- or the evidence against it.

Kappa grids are scaled per modality to the number of state exits, so both span
"nothing" to "roughly double the dwell".

Env: OUTCSV.  Run from this directory.
"""
import sys, os, re, pickle, csv, time
SEG = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.insert(0, SEG); os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np, pandas as pd
import jax.numpy as jnp, jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM, PoissonHMM
from segmentation_functions import compute_inputs, cross_validate_armodel, cross_validate_poismodel

NB = 5
OUT = os.environ['OUTCSV']
WHISK_K = [0.0, 1e3, 1e4, 5e4, 1e5, 2e5]
LICK_K = [0.0, 1e2, 1e3, 5e3, 2.5e4, 1e5]
LAG = 10

fh = open(OUT, 'w', newline='')
w = csv.DictWriter(fh, fieldnames=['modality', 'group', 'mouse', 'eid', 'fps', 'kappa',
                                   'bits', 'med_dwell_f', 'med_dwell_ms', 'n_seg', 'occ',
                                   'agree_vs_k0', 'secs'])
w.writeheader()


def dwell(s):
    ch = np.where(np.diff(s) != 0)[0]
    return np.diff(np.concatenate(([0], ch + 1, [len(s)])))


def emit(mod, grp, mouse, eid, fps, kap, bits, s, s0, t0):
    d = dwell(s)
    ag = ''
    if s0 is not None:
        a = (s == s0).mean()
        ag = round(float(max(a, 1 - a)), 4)
    w.writerow(dict(modality=mod, group=grp, mouse=mouse, eid=eid, fps=fps, kappa=kap,
                    bits=round(float(bits), 5), med_dwell_f=float(np.median(d)),
                    med_dwell_ms=round(float(np.median(d)) / fps * 1000, 1),
                    n_seg=len(d), occ=round(float(s.mean()), 4), agree_vs_k0=ag,
                    secs=round(time.time() - t0, 1)))
    fh.flush()


# ---------------- whisker ----------------
FW, DMW = '../data/hmm/grid_search/5_prior_em_zsc_True', '../data/design_matrices/'
from scipy.stats import zscore
WS = ['7cec9792', 'c51f34d8', 'edd22318', '19e66dc9', '93ad879a', 'dfd8e7df']
for f in sorted(os.listdir(FW)):
    if not f.startswith('best_results_whisker_me_'):
        continue
    m = re.search(r'([0-9a-f-]{36})$', f)
    if not m or m.group(1)[:8] not in WS:
        continue
    eid = m.group(1); mouse = f[len('best_results_whisker_me_'):-36]
    arr = zscore(np.array(pd.read_parquet(DMW + f'design_matrix_{eid}_{mouse}')[['whisker_me']].dropna()),
                 axis=0, nan_policy='omit')
    nt, ed = arr.shape
    short = np.array(arr[:(nt // NB) * NB])
    tr = jnp.stack(jnp.split(short, NB))
    inp = compute_inputs(short, LAG, ed)
    tri = jnp.stack(jnp.split(inp, NB))
    fl = len(short) / NB
    s0 = None
    for kap in WHISK_K:
        t0 = time.time()
        try:
            mm = LinearAutoregressiveHMM(2, ed, num_lags=LAG, transition_matrix_stickiness=kap)
            vll, fp, _, bll = cross_validate_armodel(mm, jr.PRNGKey(0), tr, tri, 'prior', NB, 'em')
            bits = (np.asarray(vll) - np.asarray(bll)) / fl * np.log(2)
            fold = int(np.nanargmax(bits))
            md = LinearAutoregressiveHMM(2, ed, num_lags=LAG, transition_matrix_stickiness=kap)
            p, _ = md.initialize(key=jr.PRNGKey(0), method='prior',
                                 initial_probs=fp[0].probs[fold],
                                 transition_matrix=np.asarray(fp[1].transition_matrix)[fold],
                                 emission_weights=fp[2].weights[fold],
                                 emission_biases=fp[2].biases[fold],
                                 emission_covariances=fp[2].covs[fold], emissions=short)
            s = np.asarray(md.most_likely_states(p, short, inp))
            if short[s == 1, 0].mean() < short[s == 0, 0].mean():
                s = 1 - s
            emit('whisker', 'typical', mouse, eid[:8], 60.0, kap, np.nanmean(bits), s, s0, t0)
            if kap == 0.0:
                s0 = s
        except Exception as e:
            print('ERR whisker', eid[:8], kap, e, flush=True)
    print(f'whisker {mouse} {eid[:8]} done', flush=True)

# ---------------- lick ----------------
FL = '../data/training/hmm/grid_search/5_prior_em_zsc_False'
q = pd.read_csv('hmm_diagnostics/lick_training_quality.csv')
good = list(q[(q.med_bout_licks > 1) & (q.n_seg > 5)].nlargest(4, 'n_licks').eid)
bad = list(q[q.med_bout_licks <= 1].nlargest(4, 'n_licks').eid)
dmi = {}
for p in ['../data/design_matrices/1_camera_setup/session_1',
          '../data/design_matrices/1_camera_setup/extra_bwm']:
    for f in os.listdir(p):
        if f.startswith('design_matrix'):
            dmi[f.split('_')[2]] = os.path.join(p, f)
for grp, lst in (('good tracking', good), ('bad tracking', bad)):
    for e8 in lst:
        eid = [k for k in dmi if k.startswith(e8)][0]
        mouse = q[q.eid == e8].mouse.iloc[0]
        x = pd.read_parquet(dmi[eid])[['Lick count']].dropna().values
        nt, ed = x.shape
        short = np.array(x[:(nt // NB) * NB])
        tr = jnp.stack(jnp.split(short, NB))
        fl = len(short) / NB
        s0 = None
        for kap in LICK_K:
            t0 = time.time()
            try:
                mm = PoissonHMM(2, ed, transition_matrix_stickiness=kap)
                vll, fp, _, bll = cross_validate_poismodel(mm, jr.PRNGKey(0), tr, NB, 'em')
                bits = (np.asarray(vll) - np.asarray(bll)) / fl * np.log(2)
                fold = int(np.nanargmax(bits))
                md = PoissonHMM(2, ed, transition_matrix_stickiness=kap)
                p, _ = md.initialize(key=jr.PRNGKey(0), method='prior',
                                     initial_probs=fp[0].probs[fold],
                                     transition_matrix=np.asarray(fp[1].transition_matrix)[fold],
                                     emission_rates=fp[2].rates[fold])
                s = np.asarray(md.most_likely_states(p, short))
                if np.asarray(fp[2].rates)[fold].ravel()[1] < np.asarray(fp[2].rates)[fold].ravel()[0]:
                    s = 1 - s
                emit('lick', grp, mouse, e8, 30.0, kap, np.nanmean(bits), s, s0, t0)
                if kap == 0.0:
                    s0 = s
            except Exception as ex:
                print('ERR lick', e8, kap, ex, flush=True)
        print(f'lick {mouse} {e8} done', flush=True)
fh.close()
print('DONE')
