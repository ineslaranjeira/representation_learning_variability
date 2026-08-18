"""Does going all the way to the longest fitted lag change the segmentation you keep?

Decodes each session at its ADOPTED lag and at the LARGEST lag in its grid, and compares.
This is the number that matters for "should I raise the cap": if the answer is ~identical
syllables, raising the cap buys nothing regardless of what the likelihood says.
"""
import os, sys, pickle
sys.path.insert(0, '..')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
from hmm_dynamic_functions import (load_fit_variable, prepare_batches, compute_inputs,
                                   decode_ar_states, orient_states, dwell_times, pick_fold)
D = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
DM = '../../data/design_matrices/'; PRE = 'best_results_whisker_me_'; FPS = 60.0

todo = []
for n in sorted(os.listdir(D)):
    if not n.startswith(PRE):
        continue
    d = pickle.load(open(D + n, 'rb'))
    if d['cap'] >= 128:
        todo.append((n[len(PRE):-36], n[-36:], d))
print(f'{len(todo)} sessions', flush=True)

rows = []
for k, (mouse, eid, d) in enumerate(todo):
    dm = load_fit_variable(DM, eid, mouse, ['whisker_me'], True)
    sa, _, _ = prepare_batches(dm, 5)
    ed = sa.shape[1]
    out = {}
    for lag in (d['best_lag'], max(d['all_lls'])):
        fold = pick_fold(d['all_lls'][lag])
        mi = compute_inputs(sa, lag, ed)
        s, _ = decode_ar_states(sa, mi, lag, 2, ed, d['all_fit_params'][lag], fold, 'prior')
        out[lag] = orient_states(s, sa)
        del mi
    a, b = d['best_lag'], max(d['all_lls'])
    dwa, dwb = dwell_times(out[a]), dwell_times(out[b])
    rows.append(dict(mouse=mouse, eid=eid[:8], adopted=a, maxlag=b,
                     agree=float(np.mean(out[a] == out[b])),
                     dwell_adopted=float(np.median(dwa)) * 1000 / FPS,
                     dwell_max=float(np.median(dwb)) * 1000 / FPS,
                     nseg_adopted=len(dwa), nseg_max=len(dwb)))
    print(f'  [{k+1}/{len(todo)}] {mouse} {eid[:8]}  lag {a}->{b}  '
          f'agree {rows[-1]["agree"]:.4f}', flush=True)

r = pd.DataFrame(rows)
r.to_csv('lag_adopted_vs_max.csv', index=False)
print('\n=== adopted lag vs the longest fitted lag ===')
print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f'\nmedian frame agreement: {r.agree.median():.4f}   (min {r.agree.min():.4f})')
print(f'median dwell {r.dwell_adopted.median():.0f} ms -> {r.dwell_max.median():.0f} ms')
print(f'median segments {r.nseg_adopted.median():.0f} -> {r.nseg_max.median():.0f}')
