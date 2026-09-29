"""
FOLLOW-UP ON THE STRONGEST RESULT: WHEEL SPEED IN THE PRE-QUIESCENCE EPOCH
==========================================================================
Three things it could be other than "vigor at trial onset":

 A  EPOCH LENGTH. Pre-quiescence is not a fixed window -- it ends when the mouse
    holds still long enough. A mouse that keeps moving has a LONGER pre-quiescence
    AND a higher mean speed in it, by construction. If LD1 tracks the length rather
    than the speed, that is a task-timing fact, not a vigor one.
 B  REACTION TIME. `rt-along-lda1` already established that RT falls along LD1. If
    fast movers are fast responders, this is that result in a different coat.
 C  THE GATING CONTRAST. Pre-quiescence is positive and Quiescence is negative. That
    pattern -- move more before the hold, less during it -- is a different claim from
    "moves more overall", and it is worth testing as its own contrast.
"""
import os
import re
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import vigor_vs_lda as V

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATES_DIR = ROOT / 'data' / 'states_files'
CACHE = HERE / 'epoch_timing.csv'


def build_timing():
    """Per-session epoch durations and trial-level timing, from the same files."""
    pat = re.compile(r'^8_states_file_([0-9a-f\-]{36})_(.+)$')
    rows = []
    for f in sorted(os.listdir(STATES_DIR)):
        m = pat.match(f)
        if not m:
            continue
        eid, mouse = m.groups()
        d = pd.read_parquet(STATES_DIR / f,
                            columns=['Bin', 'broader_label', 'trial_id', 'reaction',
                                     'response', 'elongation'])
        dt = float(np.median(np.diff(d['Bin'].to_numpy())))
        n_tr = d['trial_id'].nunique()
        r = dict(session=eid, mouse_name=mouse, n_trials=n_tr)
        for ep in V.EPOCH_METRICS and ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']:
            r[f'dur_{ep}'] = float((d['broader_label'] == ep).sum() * dt / max(n_tr, 1))
        tr = d.drop_duplicates('trial_id')
        for c in ['reaction', 'response', 'elongation']:
            r[f'median_{c}'] = float(tr[c].median())
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(CACHE, index=False)
    return T


def main():
    T = pd.read_csv(CACHE) if CACHE.exists() else build_timing()
    os.environ['EMBEDDING'] = 'mouse_LDA_5_bins_raw_shrink0.5_360_28-09-2026'
    S, _ = V.load()
    S = S.merge(T.drop(columns=['mouse_name']), on='session', how='left')
    num = S.select_dtypes(include=[np.number]).columns
    M = S.groupby('mouse_name')[list(num)].mean()
    M['lab'] = S.groupby('mouse_name')['lab'].first()
    lab = M['lab'].to_numpy()
    ld1 = M['LD1'].to_numpy()
    tgt = M['wheel_speed_Pre-quiescence'].to_numpy()

    print('#' * 78)
    print('A. EPOCH LENGTH (mean seconds of each epoch per trial)')
    print('#' * 78)
    for ep in ['Pre-quiescence', 'Quiescence', 'Choice', 'ITI']:
        c = f'dur_{ep}'
        r, p = spearmanr(M[c], ld1, nan_policy='omit')
        print(f'  {c:22s} mean={M[c].mean():6.2f}s  vs LD1: rho={r:+.3f} p={p:.4f}')
    r1, p1, _ = V.partial_spearman(tgt, ld1, M['dur_Pre-quiescence'].to_numpy())
    print(f'  wheel_speed_Pre-quiescence vs LD1 controlling its epoch length: '
          f'{r1:+.3f} (p={p1:.4f})   [raw {spearmanr(tgt, ld1)[0]:+.3f}]')
    print(f'  r(speed, length) = '
          f'{spearmanr(tgt, M["dur_Pre-quiescence"])[0]:+.3f}')

    print('\n' + '#' * 78)
    print('B. REACTION TIME AND TRIAL TIMING')
    print('#' * 78)
    for c in ['median_reaction', 'median_response', 'median_elongation', 'n_trials']:
        r, p = spearmanr(M[c], ld1, nan_policy='omit')
        rt = spearmanr(M[c], tgt, nan_policy='omit')[0]
        r1, p1, _ = V.partial_spearman(tgt, ld1, M[c].to_numpy())
        print(f'  {c:20s} vs LD1 rho={r:+.3f} (p={p:.4f}) | r(with pre-q speed)={rt:+.3f}'
              f' | pre-q speed vs LD1 controlling it: {r1:+.3f} (p={p1:.4f})')

    print('\n' + '#' * 78)
    print('C. THE GATING CONTRAST: pre-quiescence minus quiescence')
    print('#' * 78)
    # log ratio, because the two epochs differ by more than an order of magnitude in
    # speed and a difference would be the pre-quiescence term almost exactly.
    gate = np.log(M['wheel_speed_Pre-quiescence'] / M['wheel_speed_Quiescence'])
    gate_paw = np.log(M['paw_speed_Pre-quiescence'] / M['paw_speed_Quiescence'])
    for name, g in [('wheel log(pre-q / quiescence)', gate.to_numpy()),
                    ('paw   log(pre-q / quiescence)', gate_paw.to_numpy())]:
        r, p = spearmanr(g, ld1, nan_policy='omit')
        rc, pc = spearmanr(V.center_within(g, lab), V.center_within(ld1, lab),
                           nan_policy='omit')
        print(f'  {name:30s} rho={r:+.3f} (p={p:.4f})   lab-centred {rc:+.3f} (p={pc:.4f})')
    Sg = np.log(S['wheel_speed_Pre-quiescence'] / S['wheel_speed_Quiescence'])
    S2 = S.assign(gate=Sg)
    rel, n = V.split_half_reliability(S2, 'gate')
    rel_ld1, _ = V.split_half_reliability(S, 'LD1')
    r = spearmanr(gate, ld1, nan_policy='omit')[0]
    print(f'  reliability {rel:.3f} (n={n}) -> ceiling {np.sqrt(rel * rel_ld1):.3f}, '
          f'rho/ceiling = {r / np.sqrt(rel * rel_ld1):+.3f}')
    print(f'  permutation p (10k): {V.perm_p(gate.to_numpy(), ld1):.4f}')


if __name__ == '__main__':
    main()
