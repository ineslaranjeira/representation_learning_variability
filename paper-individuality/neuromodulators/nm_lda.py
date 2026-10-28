"""
The mouse LDA on the NEUROMODULATOR dataset alone, and the session measures to read its axes.
=============================================================================================
Syllables: NM's own state space (left paw + wheel states fitted on NM, k = 8, whisk, 100 ms lick HMM,
10 bins per trial phase) -- not comparable code-for-code with the proficient 1-camera syllables, so axes
are compared with the BWM LD1 through what they CORRELATE with, never through their loadings.
Sessions: NM photometry sessions (recordings.csv of the export) of one protocol, by default the ephys
protocol; the design is lda_shrinkage_core.proficient_design (missing-data screen, >= 3 sessions per
mouse) with every other NM session passed as excluded.

Session measures, built with the proficient definitions so the two LD1s are read with the same rulers:
  vigor    vigor/lda/states_vigor.session_vigor on the NM states files (per-epoch wheel / paw speed,
           whole-session metrics); the right-paw columns are empty (1 camera)
  timing   per-epoch duration per trial, as vigor/lda/prequiescence_followup.build_timing
  task     log median RT (first movement), accuracy on contrast > 0.5, choice bias, block bias shift,
           as lab_controls.section_trial, from the export's trials tables
  engage   GLM-HMM K = 2 per mouse (trial_position_engagement): first 400 trials, engaged = P >= 0.5,
           as lab_controls.section_engage
"""
import os
import sys
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PI = HERE.parent
REPO = PI.parent
for p in [REPO, PI, PI / 'learning_individuality', PI / 'vigor' / 'lda']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import syllable_data as sd                      # noqa: E402
import lda_shrinkage_core as lc                 # noqa: E402
from Functions import nm_syllables as ns        # noqa: E402

NM_DATA = PI / 'data' / 'neuromodulators'
OUT = NM_DATA / 'nm_lda'
SYLLABLES = dict(sd.SYLLABLES, state_space='own')
N_TRUNC = 400                                   # as lab_controls.N_TRUNC
EPOCHS = lc.EPOCHS


def recordings():
    """One row per session: protocol, NM, target region, lab."""
    r = pd.read_csv(ns.ARCHIVE / 'recordings.csv')
    r['protocol'] = np.where(r['task_protocol'].str.contains('ephys'), 'ephys', 'biased')
    return r.drop_duplicates('eid').set_index('eid')[['subject', 'protocol', 'NM', 'target_NM', 'lab', 'day_n']]


def design(protocol='ephys', syllables=None, **kw):
    """(session x feature frame, mouse per session) for one protocol ('ephys', 'biased' or None = all).
    syllables: syllable_data settings, default NM's own state space (SYLLABLES); sd.SYLLABLES = the stages
    space the BWM 1-camera embedding uses."""
    syllables = SYLLABLES if syllables is None else syllables
    f = sd.syllable_file('Neuromodulators', **syllables)
    rec = recordings()
    keep = set(rec.index[rec['protocol'] == protocol] if protocol else rec.index)
    # sessions with syllables but no row in recordings.csv have no known protocol (nor trials h5): out too
    have = set(pd.read_parquet(f, columns=['sample'])['sample'].str[:36])
    other = sorted(have - keep)
    k, kw_ = sd.n_states(**syllables)
    return lc.proficient_design(f, list(other), k, kw_, **kw)


def mouse_meta():
    """Per mouse: NM type and fiber target(s) (a mouse may carry two fibers)."""
    r = pd.read_csv(ns.ARCHIVE / 'recordings.csv')
    return r.groupby('subject').agg(NM=('NM', 'first'),
                                    target=('target_NM', lambda s: '+'.join(sorted(s.unique()))))


def _cached_csv(name, make):
    f = OUT / f'{name}.csv'
    if f.exists():
        return pd.read_csv(f, index_col=0)
    OUT.mkdir(parents=True, exist_ok=True)
    t = make()
    t.to_csv(f)
    return t


def vigor():
    """states_vigor.session_vigor for every NM states file, indexed by session."""
    import states_vigor

    def make():
        rows = [states_vigor.session_vigor(str(f), f.name.split('_', 4)[4], eid)
                for eid, f in ns.states_files().items()]
        return pd.DataFrame(rows).set_index('session')
    return _cached_csv('vigor_sessions', make)


def timing():
    """Per-epoch duration per trial (s), as prequiescence_followup.build_timing."""
    def make():
        rows = []
        for eid, f in ns.states_files().items():
            d = pd.read_parquet(f, columns=['Bin', 'broader_label', 'trial_id'])
            dt = float(np.median(np.diff(d['Bin'].to_numpy())))
            n_tr = d['trial_id'].nunique()
            rows.append({'session': eid, **{f'dur_{ep}': float((d['broader_label'] == ep).sum() * dt / max(n_tr, 1))
                                            for ep in EPOCHS}})
        return pd.DataFrame(rows).set_index('session')
    return _cached_csv('epoch_timing', make)


def task(sessions):
    """lab_controls.section_trial's session measures from the export's trials tables."""
    def make():
        rows = []
        for eid in sessions:
            tr = ns.trial_table(eid)
            right = (tr['choice'] == 1).astype(float)          # IBL choice +1 = CCW = left stimulus reported
            zc = tr[tr['contrast'] == 0]
            blk = right[zc.index].groupby(zc['probabilityLeft']).mean()
            hc = tr[tr['contrast'] > 0]
            rows.append(dict(session=eid,
                             log_reaction=np.log(tr['rt'][tr['rt'] > 0].median()),
                             correct=tr.loc[tr['contrast'] > 0.5, 'correct'].mean(),
                             choice_bias=abs(right[hc.index].mean() - (hc['signed_contrast'] < 0).mean()),
                             bias=blk.get(0.8, np.nan) - blk.get(0.2, np.nan),
                             n_trials=len(tr)))
        return pd.DataFrame(rows).set_index('session')
    t = _cached_csv('task_sessions', make)
    missing = sorted(set(sessions) - set(t.index))
    if missing:
        (OUT / 'task_sessions.csv').unlink()
        t = task(sorted(set(sessions) | set(t.index)))
    return t.reindex(sessions)


def _run_lengths(x):
    d = np.diff(np.r_[0, x.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def engagement():
    """lab_controls.section_engage's measures from the per-mouse GLM-HMM K = 2 posteriors."""
    g = pd.read_parquet(NM_DATA / 'trial_position_engagement' / 'glmhmm_k2_per_mouse.pqt')
    g = g.sort_values(['eid', 'trial'])
    g = g[g.groupby('eid').cumcount() < N_TRUNC]

    def per_session(s):
        e = s['p_engaged'].to_numpy() >= 0.5
        p = s['p_engaged'].to_numpy().clip(1e-12, 1 - 1e-12)
        a, b = e[:-1], e[1:]
        return pd.Series({'frac_engaged': e.mean(),
                          'entropy': np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))),
                          'p_stay_eng': (a & b).sum() / a.sum() if a.sum() else np.nan,
                          'p_stay_dis': (~a & ~b).sum() / (~a).sum() if (~a).sum() else np.nan,
                          'switches': (a != b).sum(),
                          'short_bouts': (_run_lengths(~e) <= 2).sum()})
    return g.groupby('eid')[['p_engaged']].apply(per_session).rename_axis('session')


def session_measures(sessions):
    """Every measure, one row per session (columns from vigor are the per-epoch speeds and gates plus a few
    whole-session ones)."""
    import proficient_1cam as pc
    v = vigor()
    keep = ['wheel_mean_speed', 'wheel_mean_speed_moving', 'wheel_frac_moving', 'wheel_bout_rate_per_min',
            'l_paw_mean_speed', 'l_paw_mean_speed_moving', 'l_paw_frac_moving', 'l_paw_band_0.5_8']
    out = (v[keep].join(pc.epoch_speeds(v)).join(timing()).join(task(list(sessions)), how='right')
           .join(engagement()))
    return out.reindex(sessions)


def mouse_corr(sc, metrics, cols=('LD1', 'LD2', 'LD3'), n_perm=10000, seed=0):
    """Pearson and Spearman r across mouse means, p from permuting mice (one lab: no lab term)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    mm = sc.join(metrics, how='inner').groupby('mouse_name')[list(cols) + list(metrics.columns)].mean()
    rows = []
    for y in metrics.columns:
        for c in cols:
            m = mm[[c, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(m) < 6:
                continue
            x, v = m[c].to_numpy(), m[y].to_numpy()
            r = np.corrcoef(x, v)[0, 1]
            null = np.array([np.corrcoef(rng.permutation(x), v)[0, 1] for _ in range(n_perm)])
            rows.append(dict(metric=y, ld=c, n_mice=len(m), r=r, rho=spearmanr(x, v)[0],
                             p_perm=(1 + np.sum(np.abs(null) >= abs(r))) / (1 + n_perm)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# Comparing an NM axis with the BWM LD1 through its correlates
# ------------------------------------------------------------------------------------------------
# lab_controls' correlate labels -> the names of the session measures above
BATTERY_NAMES = {'Reaction time (log)': 'log_reaction', 'Accuracy (easy trials)': 'correct',
                 'Choice bias': 'choice_bias', 'Block bias shift': 'bias', 'State switches (K=2)': 'switches',
                 'P(stay engaged)': 'p_stay_eng', 'P(stay disengaged)': 'p_stay_dis',
                 'Short disengaged bouts (<=2 trials)': 'short_bouts', 'Posterior uncertainty': 'entropy',
                 'Fraction engaged (hard)': 'frac_engaged'}


def bwm_ld1_profile():
    """Pearson r of the BWM 1-camera LD1 with every measure, mouse means (saved by 1_proficient_1cam_ld1)."""
    import proficient_1cam as pc
    v = pd.read_csv(pc.OUT / 'ld1_vigor_correlates.csv')
    v = v[v['ld'] == 'LD1'].set_index('metric')['r']
    b = pd.read_csv(pc.OUT / 'ld1_battery.csv').set_index('correlate')['rho'].rename(index=BATTERY_NAMES)
    return pd.concat([v, b[~b.index.isin(v.index)]]).rename('BWM LD1')


def orient_by(sc, emb, M, metric='log_reaction', sign=-1, cols=('LD1', 'LD2', 'LD3')):
    """Flip each LD so that its mouse-mean correlation with `metric` has sign `sign` (default: faster RT =
    higher LD, the BWM LD1 convention)."""
    mm = sc[list(cols)].join(M[[metric]]).groupby(sc['mouse_name']).mean()
    for c in cols:
        s = sign * np.sign(np.corrcoef(mm[c], mm[metric])[0, 1])
        sc[c] *= s
        emb.W[:, int(c[2:]) - 1] *= s


def profile_test(sc, M, bwm, cols=('LD1', 'LD2', 'LD3'), orient='log_reaction', n_perm=5000, seed=0):
    """Similarity of each NM axis's correlate profile (r with every measure, mouse means) with the BWM LD1's:
    Pearson r across measures, `orient` itself left out (it fixed the sign). Null: mouse labels of the axis
    permuted, re-oriented by the same rule, profile recomputed -- so the measures' own intercorrelations are in
    the null. -> one row per axis: observed profile r, null 95th percentile, p (one-sided)."""
    rng = np.random.default_rng(seed)
    mm = sc[list(cols)].join(M).groupby(sc['mouse_name']).mean()
    names = [m for m in bwm.index if m in mm and m != orient]
    Y = mm[names].to_numpy(float)
    Yz = (Y - np.nanmean(Y, 0)) / np.nanstd(Y, 0)
    o = mm[orient].to_numpy(float)
    b = bwm[names].to_numpy(float)

    def profile(x):
        xz = (x - x.mean()) / x.std()
        r = np.nanmean(xz[:, None] * Yz, 0)
        r = r * -np.sign(np.corrcoef(x, o)[0, 1])
        return np.corrcoef(r, b)[0, 1]
    rows = []
    for c in cols:
        x = mm[c].to_numpy(float)
        obs = profile(x)
        null = np.array([profile(rng.permutation(x)) for _ in range(n_perm)])
        rows.append(dict(ld=c, profile_r=obs, null_95=np.percentile(null, 95),
                         p=(1 + np.sum(null >= obs)) / (1 + n_perm), n_measures=len(names), n_mice=len(mm)))
    return pd.DataFrame(rows)
