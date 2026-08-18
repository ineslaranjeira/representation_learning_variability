"""Calibrate the minimum-gain floor against SEGMENTATION CHANGE rather than picking a number.

For each session, decode the state sequence at every lag in the grid (kappa=0).
For every pair of lags, record:
    dbits       = held-out bits_LL(longer) - bits_LL(shorter)
    relabelled  = fraction of frames whose state assignment differs
Pooling those pairs gives an empirical mapping from "likelihood gained" to
"segmentation actually changed", so delta can be stated as:
    "the smallest held-out gain that relabels more than X% of frames"
which puts the threshold in units of the scientific output instead of bits.
"""
import sys, os, re, pickle, csv, time, itertools
SEG = '/home/ines/repositories/representation_learning_variability/paper-individuality/segmentation'
sys.path.insert(0, SEG)
os.chdir(SEG)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import numpy as np, pandas as pd
from scipy.stats import zscore
import jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM
from segmentation_functions import get_bits_LL, compute_inputs

F, DM = '../data/hmm/grid_search/5_prior_em_zsc_True', '../data/design_matrices/'
NB = 5
OUT = os.environ['OUTCSV']
NMAX = int(os.environ.get('NMAX', '80'))

files = []
for f in sorted(os.listdir(F)):
    if not f.startswith('best_results_whisker_me_'):
        continue
    m = re.search(r'([0-9a-f-]{36})$', f)
    if m:
        files.append((f, m.group(1), f[len('best_results_whisker_me_'):-36]))
# even spread across the cohort rather than the alphabetical head
step = max(1, len(files) // NMAX)
files = files[::step][:NMAX]

fh = open(OUT, 'w', newline='')
w = csv.DictWriter(fh, fieldnames=['mouse', 'eid', 'lag_a', 'lag_b', 'bits_a', 'bits_b',
                                   'dbits', 'relabelled', 'dwell_a', 'dwell_b', 'n_frames'])
w.writeheader()
t0 = time.time()
for i, (pkl, eid, mouse) in enumerate(files):
    dmf = DM + f'design_matrix_{eid}_{mouse}'
    if not os.path.exists(dmf):
        continue
    try:
        all_lls, all_base, _, allfit, _, params = pickle.load(open(os.path.join(F, pkl), 'rb'))
    except Exception:
        continue
    _, Lags, kappas = params
    if 0 not in kappas:
        continue
    ik0 = list(kappas).index(0)
    arr = zscore(np.array(pd.read_parquet(dmf)[['whisker_me']].dropna()),
                 axis=0, nan_policy='omit')
    nt, ed = arr.shape
    short = np.array(arr[:(nt // NB) * NB])
    bits, _, bestfold = get_bits_LL(all_lls, all_base, arr, NB, params, 2)

    seqs, mbits, dwells = {}, {}, {}
    for lag in sorted(Lags):
        il = list(Lags).index(lag)
        bf = bestfold[ik0, il]
        if np.isnan(bf):
            continue
        fold = int(bf)
        fitp = allfit[lag][0]
        inp = compute_inputs(short, lag, ed)
        m = LinearAutoregressiveHMM(2, ed, num_lags=lag, transition_matrix_stickiness=0)
        p, _ = m.initialize(key=jr.PRNGKey(0), method='prior',
                           initial_probs=fitp[0].probs[fold],
                           transition_matrix=np.asarray(fitp[1].transition_matrix)[fold],
                           emission_weights=fitp[2].weights[fold],
                           emission_biases=fitp[2].biases[fold],
                           emission_covariances=fitp[2].covs[fold], emissions=short)
        s = np.asarray(m.most_likely_states(p, short, inp))
        # orient consistently: state 1 == higher whisker ME, so labels are comparable across lags
        if short[s == 1, 0].mean() < short[s == 0, 0].mean():
            s = 1 - s
        seqs[lag] = s
        mbits[lag] = float(np.nanmean(bits[ik0, il]))
        ch = np.where(np.diff(s) != 0)[0]
        dwells[lag] = float(np.median(np.diff(np.concatenate(([0], ch + 1, [len(s)])))))

    for a, b in itertools.combinations(sorted(seqs), 2):
        w.writerow(dict(mouse=mouse, eid=eid, lag_a=a, lag_b=b,
                        bits_a=round(mbits[a], 5), bits_b=round(mbits[b], 5),
                        dbits=round(mbits[b] - mbits[a], 5),
                        relabelled=round(float((seqs[a] != seqs[b]).mean()), 5),
                        dwell_a=dwells[a], dwell_b=dwells[b], n_frames=len(short)))
    fh.flush()
    if i % 10 == 0:
        print(f'{i}/{len(files)}  {time.time()-t0:.0f}s', flush=True)
fh.close()
print(f'DONE in {time.time()-t0:.0f}s')
