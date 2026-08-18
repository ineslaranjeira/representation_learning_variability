"""What does a LONG AR lag do to the segmentation?

The question this answers: if we kept raising the cap until no session sat at it, would
that be harmless? Decodes every lag in each session's grid, reusing the stored fit_params
(no refitting), and tracks what happens to the syllables.

Only sessions whose grid already reaches 128+ can be asked, since only they were fitted
that far. Run from this directory.
"""
import os, sys, pickle
sys.path.insert(0, '..')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
from hmm_dynamic_functions import (load_fit_variable, prepare_batches, compute_inputs,
                                   decode_ar_states, orient_states, dwell_times, pick_fold)

D = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
DM = '../../data/design_matrices/'
PRE = 'best_results_whisker_me_'
FPS = 60.0
MINCAP = int(os.environ.get('MINCAP', '128'))

files = []
for n in sorted(os.listdir(D)):
    if not n.startswith(PRE):
        continue
    d = pickle.load(open(D + n, 'rb'))
    if d['cap'] >= MINCAP:
        files.append((n[len(PRE):-36], n[-36:], d))
print(f'{len(files)} sessions with grid reaching {MINCAP}', flush=True)

rows = []
for k, (mouse, eid, d) in enumerate(files):
    dm = load_fit_variable(DM, eid, mouse, ['whisker_me'], True)
    shortened_array, _, _ = prepare_batches(dm, 5)
    ed = shortened_array.shape[1]
    ref = None
    for lag in sorted(d['all_lls']):
        fp = d['all_fit_params'][lag]
        fold = pick_fold(d['all_lls'][lag])
        if fold is None:
            continue
        mi = compute_inputs(shortened_array, lag, ed)
        s, _ = decode_ar_states(shortened_array, mi, lag, 2, ed, fp, fold, 'prior')
        s = orient_states(s, shortened_array)
        dw = dwell_times(s)
        if ref is None:
            ref = s
        rows.append(dict(mouse=mouse, eid=eid[:8], lag=lag, tau=d['tau'], cap=d['cap'],
                         adopted=d['best_lag'],
                         med_dwell_ms=float(np.median(dw)) * 1000 / FPS,
                         n_seg=len(dw),
                         occ=float(np.mean(s == 1)),
                         agree_vs_shortest=float(np.mean(s == ref)),
                         raw_ll=float(np.nanmean(d['all_lls'][lag]))))
        del mi
    print(f'  [{k+1}/{len(files)}] {mouse} {eid[:8]} done', flush=True)

r = pd.DataFrame(rows)
r.to_csv('lag_ceiling_effect.csv', index=False)
print('\nwrote lag_ceiling_effect.csv')
print('\n=== median across sessions, by lag ===')
print(r.groupby('lag').agg(n=('eid', 'count'), med_dwell_ms=('med_dwell_ms', 'median'),
                           n_seg=('n_seg', 'median'), occ=('occ', 'median'),
                           agree_vs_lag1=('agree_vs_shortest', 'median'),
                           raw_ll=('raw_ll', 'median')).to_string(float_format=lambda x: f'{x:.3f}'))
