"""Old rule vs new rule, and bits vs raw LL, on the EXISTING doubling-grid fits.

No refitting: every grid cell's per-fold held-out LL and baseline LL are in the pickles,
so all four rule x statistic combinations can be re-derived offline.

  unpaired  = faithful reimplementation of the lag half of `find_2_best_param`:
              argmax cell, then the SMALLEST lag whose upper 95% CI reaches the argmax's
              lower CI, with CI = nanstd(over folds)/sqrt(n)*1.96   <- ignores pairing
  paired    = sequential paired increments (the new rule)

Also answers: does it matter whether you use bits_LL or raw held-out LL?
Run from this directory.
"""
import os, sys, pickle
sys.path.insert(0, '..')
import numpy as np, pandas as pd
from scipy.stats import t as tdist

D = '../../data/hmm/grid_search_dynamic/5_prior_em_zsc_True_dynamic/'
LEG = '../../data/hmm/most_likely_states/5_prior_em_zsc_True/'
PRE = 'best_results_whisker_me_'


def select_unpaired(prof, lags):
    """The original rule (lag axis of find_2_best_param), single kappa."""
    P = np.asarray(prof, dtype=float)                 # (n_lags, n_folds)
    mean = np.nanmean(P, axis=1)
    i_best = int(np.nanargmax(mean))
    ci = np.nanstd(P, axis=1) / np.sqrt(P.shape[1]) * 1.96
    upper, lower = mean + ci, mean - ci
    if lags[i_best] == min(lags):
        return lags[i_best]
    ok = np.where(upper >= lower[i_best])[0]
    return lags[int(np.min(ok))]


def select_paired(prof, lags, alpha=0.05, one_sided=False):
    cur = 0
    for j in range(1, len(lags)):
        dif = np.asarray(prof[j], float) - np.asarray(prof[cur], float)
        ok = np.isfinite(dif); n = int(ok.sum())
        if n < 2:
            continue
        mu = dif[ok].mean(); se = dif[ok].std(ddof=1) / np.sqrt(n)
        if se == 0:
            if mu > 0: cur = j
            continue
        q = 1 - alpha if one_sided else 1 - alpha / 2
        if mu - tdist.ppf(q, n - 1) * se > 0:
            cur = j
    return lags[cur]


def legacy_lag(mouse, eid):
    fn = os.path.join(LEG, 'whisker_me_' + mouse + eid)
    if not os.path.exists(fn) or os.path.getsize(fn) == 0:
        return None
    return int(pickle.load(open(fn, 'rb'))[2][1])


rows = []
for n in sorted(os.listdir(D)):
    if not n.startswith(PRE):
        continue
    d = pickle.load(open(D + n, 'rb'))
    mouse, eid = n[len(PRE):-36], n[-36:]
    lags = sorted(d['all_lls'])
    raw = [np.asarray(d['all_lls'][l], float) for l in lags]
    # bits: baseline-subtracted, same per-frame scaling get_bits_LL uses. all_lls and
    # all_baseline_lls are already per frame here, so only the log(2) factor is left.
    bits = [(np.asarray(d['all_lls'][l], float)
             - np.asarray(d['all_baseline_lls'][l], float)) * np.log(2) for l in lags]
    rows.append(dict(
        mouse=mouse, eid=eid, cap=d['cap'], tau=d['tau'], shipped=d['best_lag'],
        old_lag=legacy_lag(mouse, eid),
        unpaired_raw=select_unpaired(raw, lags),   unpaired_bits=select_unpaired(bits, lags),
        paired_raw=select_paired(raw, lags),       paired_bits=select_paired(bits, lags),
        paired_raw_1s=select_paired(raw, lags, one_sided=True),
        paired_bits_1s=select_paired(bits, lags, one_sided=True),
    ))

r = pd.DataFrame(rows)
r.to_csv('rule_comparison.csv', index=False)
print(f'{len(r)} sessions\n')

print('=== sanity: does the shipped pipeline match paired+raw? ===')
print(f'  {int((r.shipped == r.paired_raw).sum())}/{len(r)} identical\n')

print('=== 1. BITS vs RAW LL: does the statistic change the answer? ===')
for rule in ['unpaired', 'paired', 'paired_1s' ]:
    a = r[f'{rule.replace("_1s","")}_raw' + ('_1s' if rule.endswith('1s') else '')]
    b = r[f'{rule.replace("_1s","")}_bits' + ('_1s' if rule.endswith('1s') else '')]
    print(f'  {rule:12s} same lag in {int((a == b).sum())}/{len(r)} sessions')

print('\n=== 2. UNPAIRED (your original rule) vs PAIRED, on the doubling grid ===')
print(pd.DataFrame({'unpaired (orig rule)': r.unpaired_raw.value_counts().sort_index(),
                    'paired 2-sided': r.paired_raw.value_counts().sort_index(),
                    'paired 1-sided': r.paired_raw_1s.value_counts().sort_index()}
                   ).fillna(0).astype(int).to_string())

print('\n=== 3. how close is the unpaired rule to what your OLD pipeline chose? ===')
o = r[r.old_lag.notna()]
print(f'  old pipeline picked lag 1 in {int((o.old_lag == 1).sum())}/{len(o)}')
print(f'  unpaired rule on the new grid picks lag 1 in {int((o.unpaired_raw == 1).sum())}/{len(o)}')

print('\n=== 4. why: fold effect vs lag effect (the pairing evidence) ===')
btw, wth = [], []
for n in sorted(os.listdir(D))[:80]:
    if not n.startswith(PRE):
        continue
    d = pickle.load(open(D + n, 'rb'))
    P = np.vstack([np.asarray(d['all_lls'][l], float) for l in sorted(d['all_lls'])])
    btw.append(np.nanstd(np.nanmean(P, axis=0)))      # spread BETWEEN folds
    wth.append(np.nanstd(np.nanmean(P, axis=1)))      # spread ACROSS lags
print(f'  median SD between folds (common effect) : {np.median(btw):.4f}')
print(f'  median SD across lags   (the signal)    : {np.median(wth):.4f}')
print(f'  ratio                                   : {np.median(btw)/np.median(wth):.1f}x')
