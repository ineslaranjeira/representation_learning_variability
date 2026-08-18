"""Decode every whisker session at the current lag and the paired-rule lag.

Writes one CSV row per session as it goes, so partial results are usable.
Answers two things at once:
  - how much the segmentation changes when the lag changes (all 321, not a sample)
  - which sessions show the 49368f16 flicker signature (implausibly short dwell)
"""
import sys, os, re, pickle, csv, time
SEG = '/home/ines/repositories/representation_learning_variability/paper-individuality/segmentation'
sys.path.insert(0, SEG)
os.chdir(SEG)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
from scipy.stats import zscore
import jax.random as jr
from dynamax.hidden_markov_model import LinearAutoregressiveHMM
from segmentation_functions import get_bits_LL, find_best_param, compute_inputs
from paired_selection import find_best_param_paired

F = '../data/hmm/grid_search/5_prior_em_zsc_True'
DM = '../data/design_matrices/'
OUT = os.environ['OUTCSV']


def dwell(s):
    ch = np.where(np.diff(s) != 0)[0]
    return np.diff(np.concatenate(([0], ch + 1, [len(s)])))


def decode(arr, lag, kappa, fitp, fold):
    ed = arr.shape[1]
    short = arr[:(len(arr) // 5) * 5]
    inp = compute_inputs(short, lag, ed)
    m = LinearAutoregressiveHMM(2, ed, num_lags=lag, transition_matrix_stickiness=kappa)
    p, _ = m.initialize(key=jr.PRNGKey(0), method='prior',
                        initial_probs=fitp[0].probs[fold],
                        transition_matrix=fitp[1].transition_matrix[fold],
                        emission_weights=fitp[2].weights[fold],
                        emission_biases=fitp[2].biases[fold],
                        emission_covariances=fitp[2].covs[fold],
                        emissions=short)
    return np.asarray(m.most_likely_states(p, short, inp))


cols = ['mouse', 'eid', 'cur_lag', 'cur_kappa', 'new_lag', 'new_kappa', 'bits_best',
        'dwell_cur', 'dwell_new', 'p10_dwell_cur', 'p10_dwell_new',
        'nseg_cur', 'nseg_new', 'occ_cur', 'occ_new', 'agreement', 'note']
fh = open(OUT, 'w', newline='')
w = csv.DictWriter(fh, fieldnames=cols)
w.writeheader()

files = sorted(f for f in os.listdir(F) if f.startswith('best_results_whisker_me_'))
t0 = time.time()
for i, f in enumerate(files):
    m = re.search(r'([0-9a-f-]{36})$', f)
    if not m:
        continue
    eid = m.group(1)
    mouse = f[len('best_results_whisker_me_'):-36]
    r = dict.fromkeys(cols, '')
    r.update(mouse=mouse, eid=eid)
    try:
        all_lls, all_base, _, allfit, _, params = pickle.load(open(os.path.join(F, f), 'rb'))
        _, Lags, kappas = params
        dmf = DM + f'design_matrix_{eid}_{mouse}'
        if not os.path.exists(dmf):
            r['note'] = 'no design matrix'
            w.writerow(r); fh.flush(); continue
        arr = zscore(np.array(pd.read_parquet(dmf)[['whisker_me']].dropna()),
                     axis=0, nan_policy='omit')
        bits, _, bestfold = get_bits_LL(all_lls, all_base, arr, 5, params, 2)
        ck, cl, _ = find_best_param(bits, params, 2)
        pk, pl, _ = find_best_param_paired(bits, params, 2, rule='ttest')
        r.update(cur_lag=cl, cur_kappa=ck, new_lag=pl, new_kappa=pk,
                 bits_best=round(float(np.nanmax(np.nanmean(bits, axis=2))), 4))
        out = {}
        for tag, (lg, kp) in (('cur', (cl, ck)), ('new', (pl, pk))):
            bf = bestfold[list(kappas).index(kp), list(Lags).index(lg)]
            if np.isnan(bf):
                out[tag] = None
                continue
            s = decode(arr, lg, kp, allfit[lg][kp], int(bf))
            d = dwell(s)
            out[tag] = dict(s=s, med=float(np.median(d)), p10=float(np.percentile(d, 10)),
                            n=len(d), occ=float(s.mean()))
        if out.get('cur'):
            a = out['cur']
            r.update(dwell_cur=a['med'], p10_dwell_cur=a['p10'], nseg_cur=a['n'],
                     occ_cur=round(a['occ'], 3))
        if out.get('new'):
            b = out['new']
            r.update(dwell_new=b['med'], p10_dwell_new=b['p10'], nseg_new=b['n'],
                     occ_new=round(b['occ'], 3))
        if out.get('cur') and out.get('new'):
            L = min(len(out['cur']['s']), len(out['new']['s']))
            ag = (out['cur']['s'][:L] == out['new']['s'][:L]).mean()
            r['agreement'] = round(float(max(ag, 1 - ag)), 4)
        if not out.get('cur') or not out.get('new'):
            r['note'] = 'fold NaN for one of the two cells'
    except Exception as e:
        r['note'] = f'{type(e).__name__}: {e}'[:120]
    w.writerow(r)
    fh.flush()
    if i % 20 == 0:
        print(f'{i}/{len(files)}  {time.time()-t0:.0f}s', flush=True)
fh.close()
print(f'DONE {len(files)} sessions in {time.time()-t0:.0f}s')
