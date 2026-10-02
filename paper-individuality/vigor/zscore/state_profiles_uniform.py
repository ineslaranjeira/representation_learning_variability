"""
PER-STATE WAVELET PROFILES OF THE UNIFORM PAW STATES
Mean session-z-scored wavelet power of every (paw, axis, frequency) in each state -- the
format of vigor/paw_bias/state_profiles_19Ago2026.csv, which describes the PRODUCTION states.
States as written by segmentation/3.3_wavelet_clusters_uniform.ipynb (vigor-numbered, 0 =
stillest), over every labelled frame of every session, each session z-scored by its own
full-session mean/SD (exactly what the clustering saw).
Output: vigor/paw_bias/state_profiles_uniform.csv   (read by paper_style's 'uniform' set)
"""
import os, pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

DATA = pathlib.Path(__file__).resolve().parents[2] / 'data'
OUT = pathlib.Path(__file__).resolve().parents[1] / 'paw_bias' / 'state_profiles_uniform.csv'
K = 8
BANDS = ['0.5', '1.0', '2.0', '4.0', '8.0']
PAW = [f'{p}_{a}{b}' for p in ('l_paw', 'r_paw') for a in 'xy' for b in BANDS]


def one(tag):
    W = pd.read_parquet(DATA / 'paw_wavelets' / f'paw_vel_wavelets_{tag}', columns=['Bin'] + PAW)
    A = W[PAW].to_numpy(float)
    ok = np.isfinite(A).all(1)
    Z = (A[ok] - A[ok].mean(0)) / A[ok].std(0)
    st = np.load(DATA / 'paw_states_uniform_raw_sessionz' / f'most_likely_states_{K}_{tag[37:]}{tag[:36]}.npy')
    assert len(st[0]) == len(Z), tag                    # the clustering labelled exactly these frames
    s = st[0].astype(int)
    return np.array([Z[s == k].sum(0) for k in range(K)]), np.bincount(s, minlength=K)


if __name__ == '__main__':
    tags = sorted(f[len('paw_vel_wavelets_'):] for f in os.listdir(DATA / 'paw_wavelets')
                  if f.startswith('paw_vel_wavelets_'))
    res = Parallel(n_jobs=18)(delayed(one)(t) for t in tags)
    S = sum(r[0] for r in res)
    n = sum(r[1] for r in res)
    prof = pd.DataFrame(S / n[:, None], columns=PAW)
    prof.index.name = 'state'
    prof.to_csv(OUT)
    print(f'{OUT}: {len(tags)} sessions')
    print(pd.DataFrame({'vigor': prof.mean(1),
                        'L-R gap': prof[PAW[:10]].mean(1) - prof[PAW[10:]].mean(1),
                        'share': n / n.sum()}).round(3))
