"""Fit the ORIGINAL lag grid [1, 10, 20, 30] on example sessions and compare rules.

The new pipeline only fitted powers of 2, so the original grid values 10/20/30 have to be
fitted to test the original grid honestly. Then:

  - original rule (unpaired, find_2_best_param) on the original grid
  - paired rule on the original grid
  - what the new pipeline chose on the doubling grid

and decode all of them so the segmentations can be compared by eye.
Run from this directory.
"""
import os, sys, pickle, time
sys.path.insert(0, '..')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from hmm_dynamic_functions import (load_fit_variable, prepare_batches, compute_inputs,
                                   fit_ar_lag, decode_ar_states, orient_states,
                                   dwell_times, pick_fold)

D   = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
DM  = '../../data/design_matrices/'
LEG = '../../data/hmm/most_likely_states/5_prior_em_zsc_True/'
PRE = 'best_results_whisker_me_'
# the old grid was not one fixed thing across runs: 40 appears too (CSH_ZAD_026's
# saved choice IS 40), so fit the widest version of it
ORIG_GRID = [1, 10, 20, 30, 40]
FPS, NB = 60.0, 5

CASES = [   # ONLY_LAST=1 reruns just the final case, which crashed on KeyError: 40
   # (mouse, eid8, why)
    ('ZFM-01577', '63f3dbc1', 'largest old-vs-new disagreement; new lag 64'),
    ('CSHL045',   '034e726f', 'typical: old lag 1, new lag 8'),
    ('NYU-12',    'a8a8af78', 'degenerate flicker; new rule kept lag 1'),
    ('DY_010',    '02fbb6da', 'old pipeline chose lag 10 here'),
    ('ZM_2240',   '510b1a50', 'old lag 10; flagged as under-segmented'),
    ('CSH_ZAD_026','15763234', 'old lag 40, new lag 32'),
]

def select_unpaired(prof, lags):
    P = np.asarray(prof, float); mean = np.nanmean(P, axis=1)
    i = int(np.nanargmax(mean))
    ci = np.nanstd(P, axis=1) / np.sqrt(P.shape[1]) * 1.96
    if lags[i] == min(lags): return lags[i]
    ok = np.where((mean + ci) >= (mean - ci)[i])[0]
    return lags[int(np.min(ok))]

def select_paired(prof, lags, alpha=0.05):
    cur = 0
    for j in range(1, len(lags)):
        dif = np.asarray(prof[j], float) - np.asarray(prof[cur], float)
        ok = np.isfinite(dif); n = int(ok.sum())
        if n < 2: continue
        mu = dif[ok].mean(); se = dif[ok].std(ddof=1) / np.sqrt(n)
        if se == 0:
            if mu > 0: cur = j
            continue
        if mu - tdist.ppf(1 - alpha / 2, n - 1) * se > 0: cur = j
    return lags[cur]

# resolve the 8-char prefixes against what is on disk
if os.environ.get('ONLY_LAST'):
    CASES = CASES[-1:]

disk = {}
for n in sorted(os.listdir(D)):
    if n.startswith(PRE):
        disk[n[-36:][:8]] = (n[len(PRE):-36], n[-36:])

out, seqs = [], {}
for mouse, e8, why in CASES:
    if e8 not in disk:
        print(f'!! {mouse} {e8} not on disk'); continue
    mouse_d, eid = disk[e8]
    d = pickle.load(open(D + PRE + mouse_d + eid, 'rb'))
    dm = load_fit_variable(DM, eid, mouse_d, ['whisker_me'], True)
    sa, train_em, _ = prepare_batches(dm, NB)
    ed = sa.shape[1]
    t0 = time.time()

    # per-fold held-out LL on the ORIGINAL grid (reuse lag 1, fit 10/20/30)
    prof_raw, prof_bits = [], []
    for lag in ORIG_GRID:
        if lag in d['all_lls']:
            raw, base, fp = d['all_lls'][lag], d['all_baseline_lls'][lag], d['all_fit_params'][lag]
        else:
            raw, base, fp, _ = fit_ar_lag(sa, train_em, lag, 2, ed, NB, 'prior', 'em',
                                          kappa=0.0, num_iters=100)
        prof_raw.append(np.asarray(raw, float))
        prof_bits.append((np.asarray(raw, float) - np.asarray(base, float)) * np.log(2))
        d['all_fit_params'][lag] = fp          # keep for decoding
        d['all_lls'][lag] = raw

    picks = dict(
        orig_rule_orig_grid   = select_unpaired(prof_bits, ORIG_GRID),   # exactly 4.1/4.2
        orig_rule_raw         = select_unpaired(prof_raw,  ORIG_GRID),
        paired_orig_grid      = select_paired(prof_bits,   ORIG_GRID),
        paired_orig_grid_raw  = select_paired(prof_raw,    ORIG_GRID),
        new_pipeline          = d['best_lag'],
    )
    legfn = LEG + 'whisker_me_' + mouse_d + eid
    picks['old_pipeline_file'] = (int(pickle.load(open(legfn,'rb'))[2][1])
                                 if os.path.exists(legfn) else None)

    # decode every distinct lag that any rule chose
    # only decode lags we actually have parameters for (a legacy lag outside ORIG_GRID
    # was never fitted here)
    for lag in sorted({v for v in picks.values() if v and v in d['all_fit_params']}):
        fold = pick_fold(d['all_lls'][lag])
        mi = compute_inputs(sa, lag, ed)
        s, _ = decode_ar_states(sa, mi, lag, 2, ed, d['all_fit_params'][lag], fold, 'prior')
        s = orient_states(s, sa)
        dw = dwell_times(s)
        seqs[(e8, lag)] = s
        out.append(dict(mouse=mouse_d, eid=e8, why=why, lag=lag,
                        med_dwell_ms=float(np.median(dw)) * 1000 / FPS, n_seg=len(dw),
                        occ=float(np.mean(s == 1)),
                        raw_ll=float(np.nanmean(d['all_lls'][lag])), **picks))
    print(f'{mouse_d} {e8}: {picks}  ({time.time()-t0:.0f}s)', flush=True)

suffix = '_last' if os.environ.get('ONLY_LAST') else ''
pd.DataFrame(out).to_csv(f'original_grid_test{suffix}.csv', index=False)
pickle.dump({k: v.astype(np.int8) for k, v in seqs.items()},
            open(f'original_grid_states{suffix}.pkl', 'wb'))
print(f'\nwrote original_grid_test{suffix}.csv + original_grid_states{suffix}.pkl')
