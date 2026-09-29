"""
CONTROLS FOR THE LD1-vs-VIGOR RESULT
=====================================
Five things could make the headline correlations uninteresting, and each gets a test:

 1  EMBEDDING-SPECIFIC. The shrinkage embedding is the current one, but the result
    should not depend on it. Re-run against every LDA file on disk.
 2  EPOCH METRICS HAVE NO CEILING YET. The whole-session metrics were checked for
    split-half reliability; the per-epoch ones were not, and an unreliable measure
    cannot produce a large correlation -- so a large one from an unreliable measure
    is the thing to be suspicious of.
 3  WHEEL vs PAW ARE NOT INDEPENDENT. A mouse that moves its paws a lot moves the
    wheel a lot. Head-to-head partials say which one LD1 is actually about.
 4  PRE-QUIESCENCE vs THE WHOLE SESSION. Is the epoch result a separate fact, or
    the session-wide one showing up in one epoch?
 5  n_sessions / duration. A mouse with more sessions has a better-estimated LD1 and
    a better-estimated vigor; if both scale with n, that alone makes a correlation.
"""
import os
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import vigor_vs_lda as V

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEADLINE = ['wheel_mean_speed_moving', 'wheel_frac_moving', 'wheel_speed_Pre-quiescence',
            'paw_mean_speed_moving', 'paw_speed_Quiescence']


def mouse_table(embedding):
    os.environ['EMBEDDING'] = embedding
    import importlib
    importlib.reload(V)
    S, lds = V.load()
    num = S.select_dtypes(include=[np.number]).columns
    M = S.groupby('mouse_name')[list(num)].mean()
    M['lab'] = S.groupby('mouse_name')['lab'].first()
    M['n_sessions'] = S.groupby('mouse_name').size()
    return S, M


def main():
    print('#' * 78)
    print('1. DOES IT DEPEND ON THE EMBEDDING?')
    print('#' * 78)
    files = sorted(f for f in os.listdir(ROOT / 'clustering' / 'data_files')
                   if f.startswith('mouse_LDA_5_bins') and 'lab_' not in f)
    print(f'{"embedding":46s} {"n":>3s} ' + ' '.join(f'{m[:18]:>19s}' for m in HEADLINE[:3]))
    for f in files:
        try:
            _, M = mouse_table(f)
        except Exception as e:                                    # noqa: BLE001
            print(f'{f:46s}  -- skipped ({type(e).__name__})')
            continue
        cells = []
        for m in HEADLINE[:3]:
            r, p = spearmanr(M[m], M['LD1'], nan_policy='omit')
            cells.append(f'{r:+.3f}{"*" if p < 0.05 else " "} (p={p:.3f})')
        print(f'{f:46s} {len(M):3d} ' + ' '.join(f'{c:>19s}' for c in cells))
    print('  SIGN IS ARBITRARY per file -- an eigenvector is defined up to sign, so')
    print('  compare |rho| across rows, and signs only within a row.')

    S, M = mouse_table('mouse_LDA_5_bins_raw_shrink0.5_360_28-09-2026')
    lab = M['lab'].to_numpy()

    print('\n' + '#' * 78)
    print('2. RELIABILITY OF THE PER-EPOCH METRICS')
    print('#' * 78)
    rel_ld1, _ = V.split_half_reliability(S, 'LD1')
    for m in V.EPOCH_METRICS:
        r, n = V.split_half_reliability(S, m)
        rr = spearmanr(M[m], M['LD1'], nan_policy='omit')[0]
        print(f'{m:28s} rel={r:6.3f}  ceiling={np.sqrt(max(r, 0) * rel_ld1):.3f}  '
              f'rho={rr:+.3f}  rho/ceil={rr / np.sqrt(max(r, 1e-9) * rel_ld1):+.3f}')

    print('\n' + '#' * 78)
    print('3. WHEEL vs PAW, HEAD TO HEAD')
    print('#' * 78)
    print(f'r(wheel_mean_speed_moving, paw_mean_speed_moving) = '
          f'{spearmanr(M["wheel_mean_speed_moving"], M["paw_mean_speed_moving"])[0]:+.3f}')
    for a, b in [('wheel_mean_speed_moving', 'paw_mean_speed_moving'),
                 ('paw_mean_speed_moving', 'wheel_mean_speed_moving')]:
        r0 = spearmanr(M[a], M['LD1'], nan_policy='omit')[0]
        r1, p1, n = V.partial_spearman(M[a].to_numpy(), M['LD1'].to_numpy(),
                                       M[b].to_numpy())
        print(f'{a:26s} vs LD1: raw {r0:+.3f} -> controlling {b}: {r1:+.3f} (p={p1:.4f})')

    print('\n' + '#' * 78)
    print('4. IS PRE-QUIESCENCE WHEEL SPEED A SEPARATE FACT?')
    print('#' * 78)
    tgt = 'wheel_speed_Pre-quiescence'
    for ctrl in ['wheel_mean_speed_moving', 'wheel_mean_speed', 'wheel_frac_moving',
                 'wheel_speed_ITI', 'wheel_speed_Choice', 'paw_speed_Pre-quiescence']:
        r1, p1, n = V.partial_spearman(M[tgt].to_numpy(), M['LD1'].to_numpy(),
                                       M[ctrl].to_numpy())
        print(f'  controlling {ctrl:26s}: {r1:+.3f}  p={p1:.4f}  '
              f'[r(tgt,ctrl)={spearmanr(M[tgt], M[ctrl])[0]:+.3f}]')
    rc = V.center_within(M[tgt].to_numpy(), lab)
    print(f'  lab-centred: {spearmanr(rc, V.center_within(M["LD1"].to_numpy(), lab))[0]:+.3f}')
    print('  per-lab rho (labs with >=5 mice), to check it is not one lab:')
    for L, g in M.groupby('lab'):
        if len(g) >= 5:
            r, p = spearmanr(g[tgt], g['LD1'])
            print(f'    {L:22s} n={len(g):2d}  rho={r:+.3f}  p={p:.3f}')

    print('\n' + '#' * 78)
    print('5. n_sessions AND DURATION')
    print('#' * 78)
    for nz in ['n_sessions', 'duration_min']:
        r, p = spearmanr(M[nz], M['LD1'], nan_policy='omit')
        print(f'  LD1 vs {nz:14s}: rho={r:+.3f} p={p:.3f}')
        for m in HEADLINE:
            rr = spearmanr(M[m], M[nz], nan_policy='omit')[0]
            r1, p1, _ = V.partial_spearman(M[m].to_numpy(), M['LD1'].to_numpy(),
                                           M[nz].to_numpy())
            print(f'    {m:28s} r(metric,{nz[:4]})={rr:+.3f}   '
                  f'rho|{nz[:4]}={r1:+.3f} (p={p1:.4f})')


if __name__ == '__main__':
    main()
