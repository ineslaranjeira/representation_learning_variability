"""2-state GAUSSIAN HMM vs the AR-HMM on the same sessions, same data preparation.

Fits the Gaussian model (no lag at all), then compares its segmentation frame-for-frame
with the AR fit already on disk. Alignment is safe because both use
load_fit_variable + prepare_batches, so the arrays are the same rows.

Note on comparing likelihoods: raw held-out LL is comparable between the two (both are
log densities of the same observations), but the AR model CONDITIONS ON PAST OBSERVATIONS,
so it has strictly more information and should win. bits_LL is NOT comparable -- different
emission families have different prior-sampled baselines, the same trap as Poisson vs
Bernoulli. So raw LL is reported and bits is not compared.

Run from this directory.
"""
import os, sys, pickle, time
sys.path.insert(0, '..')
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np, pandas as pd
import hmm_gaussian_functions as G
from hmm_dynamic_functions import load_fit_variable, prepare_batches, dwell_times

AR  = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
GA  = '../../data/hmm/grid_search_gaussian/5_kmeans_em_zsc_True_gaussian/'
DM  = '../../data/design_matrices/'
PRE = 'best_results_whisker_me_'
FPS, NB = 60.0, 5
os.makedirs(GA, exist_ok=True)

WANT = ['63f3dbc1', '034e726f', 'a8a8af78', '02fbb6da', '510b1a50', '15763234']

disk = {n[-36:][:8]: (n[len(PRE):-36], n[-36:])
        for n in sorted(os.listdir(AR)) if n.startswith(PRE)}

rows = []
for e8 in WANT:
    if e8 not in disk:
        print(f'!! {e8} not in the AR run'); continue
    mouse, eid = disk[e8]
    t0 = time.time()
    r = G.run_session_gaussian((mouse, eid), ['whisker_me'], True, 2, NB, 'kmeans', 'em',
                               save_path=GA, data_path=DM, fps=FPS)
    if r['error']:
        print(f'{mouse} {e8}: ERROR {r["error"]}'); continue

    g = pickle.load(open(GA + PRE + mouse + eid, 'rb'))
    a = pickle.load(open(AR + PRE + mouse + eid, 'rb'))
    sg = np.asarray(g['most_likely_states']); sa = np.asarray(a['most_likely_states'])
    n = min(len(sg), len(sa))
    agree = float(np.mean(sg[:n] == sa[:n]))
    dg, da = dwell_times(sg), dwell_times(sa)
    rows.append(dict(
        mouse=mouse, eid=e8, ar_lag=a['best_lag'],
        gauss_dwell_ms=float(np.median(dg)) * 1000 / FPS,
        ar_dwell_ms=float(np.median(da)) * 1000 / FPS,
        gauss_nseg=len(dg), ar_nseg=len(da),
        gauss_occ=float(np.mean(sg == 1)), ar_occ=float(np.mean(sa == 1)),
        gauss_raw_ll=float(np.nanmean(g['all_lls'][0])),
        ar_raw_ll=float(np.nanmean(a['all_lls'][a['best_lag']])),
        mean_low=r['mean_low'], mean_high=r['mean_high'],
        agree=agree, secs=time.time() - t0))
    print(f'{mouse} {e8}: AR lag {a["best_lag"]:>3} | dwell {rows[-1]["ar_dwell_ms"]:.0f}'
          f' -> {rows[-1]["gauss_dwell_ms"]:.0f} ms | agree {agree:.4f}'
          f' | {rows[-1]["secs"]:.0f}s', flush=True)

t = pd.DataFrame(rows)
t.to_csv('gaussian_vs_ar.csv', index=False)
print('\n=== Gaussian HMM vs AR-HMM ===')
print(t[['mouse','eid','ar_lag','ar_dwell_ms','gauss_dwell_ms','ar_nseg','gauss_nseg',
         'ar_occ','gauss_occ','ar_raw_ll','gauss_raw_ll','agree']].to_string(
         index=False, float_format=lambda x: f'{x:.3f}'))
print(f'\nmedian frame agreement AR vs Gaussian: {t.agree.median():.4f}')
print(f'median dwell  AR {t.ar_dwell_ms.median():.0f} ms  vs  Gaussian {t.gauss_dwell_ms.median():.0f} ms'
      f'   (model-free changepoint anchor 450 ms, IQR 379-550)')
print(f'median raw held-out LL  AR {t.ar_raw_ll.median():+.4f}  vs  Gaussian {t.gauss_raw_ll.median():+.4f}'
      f'  -> AR better by {t.ar_raw_ll.median()-t.gauss_raw_ll.median():.4f} nats/frame')
print(f'fitted state means (z): low {t.mean_low.median():+.2f}, high {t.mean_high.median():+.2f}')
