"""
SESSION MEASURES PER TRIAL EPOCH, AGAINST PROFICIENT
====================================================
The trial-resolved version of fig_time_horizon panel g. There, each measure is one number per session over the
whole video; here it is one number per session and TRIAL EPOCH (Pre-Q, Quiescence, Choice, ITI), from the
time-warped syllable profiles that syllable_correlations.ipynb builds (`recon`: sessions x 40 bins x features,
10 bins per epoch; per bin, the fraction of the session's trials in each syllable):

  whisking         mean P(whisk) over the epoch's 10 bins
  licking          mean P(lick)
  left-paw vigor   sum over paw+wheel states of P(state) x that state's left-paw power (state_palette.csv)
  wheel power      the same with the state's wheel power

Paw and wheel come from the same joint left paw + wheel states, whose wavelets are session-z-scored: like the
whole-session paw measure, they say how much of the epoch sits in high- vs low-power states, not raw speed.

Then, as in panel g: per mouse, the mean over its sessions; Spearman across mice between each earlier timepoint and
the same mice's Proficient mean; Prof. = odd vs even Proficient sessions (by date) -- the reliability ceiling.

The EPOCH AVERAGE: the mean of the four epochs' rho (Fisher z), so every epoch counts once whatever its duration
(the whole-session measure instead weights the session by real time, ITI and inter-trial gaps included). Its
p value comes from a permutation that keeps the trial structure: the earlier timepoint's mice are shuffled
against Proficient ONCE per permutation and the same shuffle applied to all four epochs, so the epochs'
dependence (same mice, correlated epochs) is in the null.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

EPOCHS = ['Pre-Q', 'Quiesc.', 'Choice', 'ITI']
MEASURES = ['whisk_frac', 'lick_frac', 'paw_vigor', 'wheel_power']


def session_epoch_measures(recon, power_paw, power_wheel, n_paw=8, i_whisk=8, i_lick=9):
    """(n_sessions, 4 epochs, 4 measures) from the time-warped profiles."""
    n_s, n_b, _ = recon.shape
    per = n_b // 4
    E = recon.reshape(n_s, 4, per, -1).mean(2)                       # sessions x epochs x features
    paw = E[..., :n_paw]
    return np.stack([E[..., i_whisk], E[..., i_lick], paw @ power_paw, paw @ power_wheel], axis=-1)


def _mouse_means(V, mice, sel):
    ms = np.unique(mice[sel])
    return pd.Index(ms), np.stack([V[sel & (mice == m)].mean(0) for m in ms])


def _rank_rho(a, b):
    """Spearman per column (epochs) of two (n_mice, n_epochs) arrays."""
    ra, rb = np.apply_along_axis(rankdata, 0, a), np.apply_along_axis(rankdata, 0, b)
    ra, rb = ra - ra.mean(0), rb - rb.mean(0)
    return (ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))


def _pairs(V, mice, timepoint, sessions, date, tps, ref='Proficient'):
    """For every earlier timepoint (and the Proficient halves): two aligned (n_mice, epochs, measures) arrays."""
    out = {}
    m_ref, P_ref = _mouse_means(V, mice, timepoint == ref)
    for tp in tps:
        if tp == ref:
            sel = timepoint == ref
            d = pd.DataFrame({'mouse': mice[sel], 'date': pd.Series(sessions[sel]).map(date).to_numpy(),
                              'i': np.flatnonzero(sel)}).dropna(subset=['date']).sort_values(['mouse', 'date'])
            d['half'] = d.groupby('mouse').cumcount() % 2
            keep = d.groupby('mouse')['half'].nunique() == 2
            d = d[d['mouse'].isin(keep[keep].index)]
            ms = np.unique(d['mouse'])
            a = np.stack([V[d.loc[(d['mouse'] == m) & (d['half'] == 0), 'i']].mean(0) for m in ms])
            b = np.stack([V[d.loc[(d['mouse'] == m) & (d['half'] == 1), 'i']].mean(0) for m in ms])
        else:
            m_tp, P_tp = _mouse_means(V, mice, timepoint == tp)
            ms = m_tp.intersection(m_ref)
            a, b = P_tp[m_tp.get_indexer(ms)], P_ref[m_ref.get_indexer(ms)]
        out[tp] = (a, b)
    return out


def epoch_correlations(V, mice, timepoint, sessions, date, tps, n_perm=10000, seed=0):
    """Long table: measure, tp, epoch (incl. 'Epoch mean'), rho, p, n. Per-epoch p: Spearman's; epoch-mean p:
    the shared-shuffle permutation (two-sided on the mean Fisher z)."""
    rng = np.random.default_rng(seed)
    rows = []
    for tp, (a, b) in _pairs(V, mice, timepoint, sessions, date, tps).items():
        n = len(a)
        for k, meas in enumerate(MEASURES):
            A, B = a[..., k], b[..., k]
            for e, ep in enumerate(EPOCHS):
                r = spearmanr(A[:, e], B[:, e])
                rows.append(dict(feature=meas, tp=tp, epoch=ep, rho=r.statistic, p=r.pvalue, n=n))
            z_obs = np.arctanh(np.clip(_rank_rho(A, B), -0.999999, 0.999999)).mean()
            null = np.empty(n_perm)
            for i in range(n_perm):
                perm = rng.permutation(n)                            # ONE shuffle of mice, all epochs
                null[i] = np.arctanh(np.clip(_rank_rho(A[perm], B), -0.999999, 0.999999)).mean()
            p = (1 + np.sum(np.abs(null) >= abs(z_obs))) / (1 + n_perm)
            rows.append(dict(feature=meas, tp=tp, epoch='Epoch mean', rho=np.tanh(z_obs), p=p, n=n))
    return pd.DataFrame(rows)
