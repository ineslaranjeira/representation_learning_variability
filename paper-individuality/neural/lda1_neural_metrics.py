"""
LDA-1 NEURAL CORRELATES -- the essential compute, in one place
==============================================================
Three neural metrics are tested against LDA 1 in this paper, and until now each lived in
its own notebook with its own copy of the same machinery:

    firing rate        neural/firing_rate/fr_psth_ldabin.ipynb
    Fano factor        neural/fano_factor/ff_psth_ldabin.ipynb
    noise correlation  neural/noise_correlations/noise_corr_psth_lda.ipynb

All three read the SAME ~400 firing-rate pkl files, build the same
neurons x trials x time array from them, and test their metrics with the SAME
`age_effect_test` (which was copy-pasted verbatim into all three notebooks). This module
keeps only what is needed to reproduce the four tests per metric that the paper reports --

    baseline    the pre-stimulus window
    post        the post-stimulus window
    magnitude   the pre->post change (FF quench / r_SC quench / evoked rate)
    onset       WHEN the change starts (Fano factor only)

-- and makes one pass over the files instead of three.

WHAT IS DELIBERATELY NOT HERE. The descriptive PSTH-by-LDA-bin curves, mean-matching
(Churchland), RT matching, the engaged-trials filter, session balancing, neuron caps, the
split-half reliability and gate-null diagnostics, and firing rate's peak/trough latency
analysis. Every one of those either has no bearing on these ten tests or was switched off
in the notebooks that produced the published numbers (`RT_MATCH = False`,
`ENGAGED_ONLY = False`, `BALANCE_SESSIONS = False`, and firing rate's
`STRATIFY_BY_CONDITION` is a no-op while `N_TARGET is None`). They stay in their own
notebooks; this module is the reproduction path.

THE STATISTICS ARE UNCHANGED. `age_effect_test` below is the notebooks' function, moved
rather than rewritten: GLM slope of lda_1 as the test statistic, a permutation null that
shuffles lda_1 across SESSIONS with rows within a session kept yoked, and a
partial-correlation JZS Bayes factor via pingouin after Frisch-Waugh-Lovell residualising.
The unit of analysis stays what each metric is naturally computed at -- single neuron for
FF and FR, session x region for r_SC (r_SC is pairwise and already averaged over pairs) --
and so do the covariates.

TWO PASSES, ONE COMPUTE. `compute_tables()` applies no region or session cohort filter: it
records every region with at least MIN_NEURONS neurons and flags each session's goCue
validation. `apply_cohort()` then imposes a cohort on the tables. That split is what lets
the same compute serve both

    cohort='legacy'   each metric's own original filters -- reproduces the published
                      numbers, which is how a change here is checked
    cohort='common'   one region list and one session filter for all three metrics, so a
                      row of the summary figure is comparable to the row above it

USE IT:

    import lda1_neural_metrics as nm
    cfg = nm.default_config()
    tables = nm.compute_tables(cfg)          # slow: reads ~400 pkl files
    nm.save_tables(tables, cfg)              # parquet cache next to this file
    res = nm.run_tests(nm.apply_cohort(tables, cfg, 'common'), cfg)

or from the shell, to fill the cache:

    python lda1_neural_metrics.py
"""
import os
import pickle
import re
import warnings
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent          # .../paper-individuality/neural
PAPER = ROOT.parent                             # .../paper-individuality
CACHE = ROOT / 'lda1_tables'
DATE_FMT = '%d-%m-%Y'


# =============================================================================
# PARAMETERS -- every number the analysis depends on, in one dict
# =============================================================================
# Values are the ones the three notebooks were last run with, so the defaults reproduce
# their published numbers. Override in the notebook (`cfg['KEEP'] = ...`) rather than
# editing here, so the default stays the reproduction path.

# The 48-region list fano_factor/ff_psth_ldabin.ipynb selects on ("at least 10 sessions"),
# quoted verbatim -- including 'CUL4', which matches nothing in the current files (the
# acronym there is 'CUL4 5'), so 52 names select 48 regions. Left as-is on purpose: it is
# what produced the published FF numbers.
KEEP_48 = ['CA1', 'DG', 'MRN', 'CP', 'LP', 'CA3', 'ZI', 'PO', 'MOs', 'MOp',
           'APN', 'SCm', 'IRN', 'VPM', 'PAG', 'VISa', 'LSr', 'CUL4', 'MD', 'VISp', 'PIR',
           'LD', 'RSPv', 'SI', 'SUB', 'RT', 'PRM', 'RSPd', 'Eth', 'ANcr2', 'ACB', 'CENT3',
           'IC', 'IP', 'SSp-bfd', 'PB', 'LGd', 'VPL', 'VISam', 'NOT', 'NTS', 'ACAd', 'VM',
           'SPVI', 'RSPagl', 'CA2', 'LH', 'GPe', 'GRN', 'PARN', 'SCs', 'PoT']


def default_config():
    """A fresh copy of the parameters, so a notebook cannot mutate the module's defaults."""
    return dict(
        # --- inputs. First candidate that exists wins; the pick is printed. The LDA build
        # and the trial meta have both moved and been re-cut, and a stale hardcoded path
        # is how these notebooks break. The pinned entry is the one the published numbers
        # came from; later entries are only fallbacks.
        LDA_FILES=[PAPER / 'clustering/data_files/mouse_LDA_5_bins_25_18-09-2026',
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_cut25_16-09-2026',
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_lab_26_17-09-2026',
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_cut25_weighted_17-09-2026',
                   
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_cut30_09-09-2026',
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_cut30_16-09-2026',
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_cut19-08-2026',
                   PAPER / 'clustering/data_files/mouse_LDA_5_bins_cut06-07-2026'],
        TRIAL_META_FILES=[PAPER / 'data/session_trial_meta_19-08-2026',
                          PAPER / '4_mice/session_trial_meta_19-08-2026',
                          PAPER / '4_mice/session_trial_meta_06-07-2026'],
        FIRING_RATES_DIR=PAPER / 'data/firing_rates',

        # --- cohort
        REGION_LEVEL='beryl',        # 'beryl' | 'cosmos'
        # Regions thrown away before anything is computed. The three notebooks disagreed
        # here -- FF and FR dropped ['root', 'void'], noise correlation dropped only
        # ['void'] and so pooled every unassigned-tissue neuron of a probe into one 'root'
        # region that then passed MIN_NEURONS. Keeping 'root' in the raw pass and excluding
        # it in the cohort layer means the reproduction check can see the numbers the
        # noise-correlation notebook published while the figure never shows 'root'.
        DROP=['void'],
        DROP_COHORT=['root', 'void'],   # additionally dropped by apply_cohort
        # DROP_COHORT=['void'],   # additionally dropped by apply_cohort
        KEEP=list(KEEP_48),          # None = every region that passes MIN_NEURONS
        MIN_NEURONS=15,              # neurons per region in a session (ALL metrics)
        # r_SC's own neuron minimum, if it should be stricter than the others. It is a
        # PAIRWISE metric: 15 neurons is 105 pairs, and the estimate's noise falls with
        # the number of pairs, so the count that is generous for a per-neuron metric is
        # thin for this one. None = use MIN_NEURONS.
        MIN_NEURONS_RSC=None,
        # r_SC's real sample size is not how many neurons the region HAS, it is how many
        # have non-zero across-trial variance IN THE ANALYSIS WINDOW (`nvalid`). A neuron
        # can clear any rate floor -- especially a whole-recording one -- and still be flat
        # across every trial in a 204 ms window. Requiring nvalid to reach the same minimum
        # as the neuron count is what stops the estimator being read off its -1/(nvalid-1)
        # floor. True = enforce; False = the old behaviour (no nvalid requirement).
        REQUIRE_RSC_NVALID=True,
        MIN_TRIALS=40,               # trials per session-region for a window estimate
        # r_SC needs trials on which EVERY neuron in the region has data, so it cannot use
        # MIN_TRIALS without losing most session-regions. This was previously spelled
        # `>= MIN_NEURONS`, which was 15 by coincidence rather than by intent; naming it
        # changes no number but stops a change to MIN_NEURONS from moving a trial gate.
        MIN_TRIALS_RSC=15,           # complete trials per session-region for r_SC
        CORRECT_ONLY=False,          # all three notebooks published with this off
        VALIDATE_GOCUE=True,         # drop sessions whose trial_id join disagrees with ONE
        GOCUE_TOL=0.01,              # s

        # --- windows. ONE PRE AND ONE POST WINDOW FOR ALL THREE METRICS, so a column of
        # the summary figure means the same stretch of time in every row. The original
        # noise-correlation notebook used (0.0, 0.2) for its post window while FF and FR
        # used (0.1, 0.3); that is now unified onto the FF/FR window, and the old one is
        # computed alongside it as `r_sc_post_legacy` / `r_sc_quench_legacy` so the
        # reproduction path against the published r_SC numbers still exists.
        PRE_WINDOW=(-0.2, 0.0),
        POST_WINDOW=(0.1, 0.3),
        RSC_PRE_WINDOW=(-0.2, 0.0),        # = PRE_WINDOW
        RSC_POST_WINDOW=(0.1, 0.3),        # = POST_WINDOW (was (0.0, 0.2))
        RSC_POST_WINDOW_LEGACY=(0.0, 0.2),  # noise_corr's own CORR_WINDOW, kept for the check
        # WHICH BINS A TRIAL HAS TO BE COMPLETE IN, for r_SC only. r_SC is pairwise, so it
        # needs trials on which EVERY neuron in the region has data -- but only over the
        # bins it actually reads. This used to demand all 90 bins, -0.5 to 1.0 s, and that
        # threw away HALF of every session's trials: ~49% of trials are missing exactly one
        # bin, always the first one at t = -0.5 s, which is a bin-alignment rounding
        # artifact (it tracks nothing about the trial -- r = 0.02 with the gap to the next
        # trial) and which no r_SC window touches. Restricting the check to the range below
        # -- the tested windows plus the WIN_BINS smoothing reach, widened to cover the
        # plotted range of the PSTH panel -- takes trial retention from 50.7% to 99.4%.
        # Set to None for the old all-bins behaviour.
        RSC_COMPLETE_WINDOW=(-0.35, 0.65),

        # --- metric estimation
        WIN_BINS=6,                  # ~100 ms sliding window (FF time course, r_SC)
        SMOOTH_MODE='causal',        # 'causal' | 'centered'
        REMOVE_CONDITION=True,       # subtract side x contrast means -> noise variance
        # ONE RATE FLOOR FOR ALL THREE METRICS. The three used to disagree, in units that
        # hid the disagreement: firing rate gated at 1.0 Hz, Fano factor at MIN_WINDOW_COUNT
        # = 0.5 spikes per window -- which over a 12-bin x 17 ms = 0.204 s window is 2.45 Hz,
        # so FF was 2.45x stricter than FR -- and r_SC at nothing at all, which is how a
        # 25-neuron MOp population with a median rate of 0.22 Hz produced an r_SC of -0.279,
        # a value below the -1/(n-1) floor a real correlation matrix can reach.
        #
        # MIN_FR_HZ is now THE floor, in Hz, for every metric: a neuron must reach it in the
        # pre window AND the post window. FF's count gate is derived from it per window
        # (1.0 Hz x 0.204 s = 0.204 spikes) rather than set independently.
        MIN_FR_HZ=1.0,               # the floor, applied to FR, FF and r_SC alike
        # WHICH WINDOW(S) a neuron must clear the floor in. 'both' is the default and the
        # conservative choice -- a neuron that is silent in either window contributes a
        # degenerate value to that window's metric. 'pre' gates on BASELINE rate only,
        # which is what you want if the concern is selecting neurons on their RESPONSE:
        # gating on the post window conditions on a stimulus-driven quantity, and if
        # responsiveness tracks lda_1 then the cohort itself does too.
        FR_FLOOR_REQUIRE='both',     # 'both' | 'pre' | 'post' | 'either'
        # WHERE THE RATE COMES FROM.
        #   'window'  the neuron's mean rate in the pre / post analysis windows, gated
        #             per FR_FLOOR_REQUIRE. Measured from the same data the metrics are,
        #             but the post window is stimulus-driven, so gating on it conditions
        #             on responsiveness -- which itself tracks lda_1.
        #   'session' the spike sorter's own whole-recording firing rate, read from the
        #             unit QC table (clusters.metrics.firing_rate). This is what Zang et
        #             al. gate on (`clus_df['firing_rate'] > 1`). It is independent of the
        #             trial windows and of the stimulus entirely, so it cannot select on
        #             the response -- FR_FLOOR_REQUIRE does not apply to it. It needs the
        #             QC table (python fetch_unit_qc.py), and a neuron with no QC row is
        #             dropped.
        FR_FLOOR_SOURCE='window',    # 'window' | 'session'
        UNIFORM_FR_FLOOR=True,       # False = each metric's ORIGINAL gate, for the
                                     # reproduction path: FR 1 Hz, FF MIN_WINDOW_COUNT
                                     # (2.45 Hz), r_SC MIN_FR_HZ_RSC (None = none)
        MIN_WINDOW_COUNT=0.5,        # FF's absolute count gate; used only when
                                     # UNIFORM_FR_FLOOR is False
        MIN_MEAN_COUNT=0.01,         # FF time course: per-bin mean-count floor (unrelated:
                                     # this one keeps a NaN out of a single curve bin)
        # A rate floor on the neurons entering the r_SC pair matrix. None = no floor, which
        # is what the noise-correlation notebook did and what the cached tables hold. It is
        # worth turning on: a spike-count correlation is attenuated at low counts, so a
        # near-silent neuron does not merely add noise to the region average, it biases it
        # toward zero -- and a region's share of such neurons is not constant across
        # sessions. CHANGING IT REQUIRES RECOMPUTE=True: neuron selection happens while the
        # pkl files are read, not in the cohort layer.
        MIN_FR_HZ_RSC=None,          # r_SC's own floor; used only when UNIFORM_FR_FLOOR
                                     # is False (None there = the old no-floor behaviour)
        # SPIKE-SORTING PRESENCE RATIO -- the fraction of 10 s bins of the WHOLE recording
        # in which the unit fired at least one spike (ibllib's `presence_window` = 10). It
        # cannot be recomputed from the pkl files, which only ever saw -0.5..1.0 s around
        # each stimulus, so it is fetched once by fetch_unit_qc.py and cached next to the
        # metric tables. This is NOT a second rate floor: a unit at a steady 1.2 Hz passes
        # both, while a unit at 8 Hz for half the session and silent after passes the rate
        # floor and fails this one. That second unit is the one that matters here -- its
        # counts are drawn from two distributions, which inflates the across-trial variance
        # and so the Fano factor, by an amount that depends on when it was lost.
        # None = no gate (what every published number was computed with).
        MIN_PRESENCE_RATIO=None,     # e.g. 0.95 (Zang et al.) or 0.90 (Allen)
        # A rate floor imposed on the FIRING-RATE table by apply_cohort rather than at read
        # time. Unlike the two above this one is free -- fr_pre and fr_post are already in
        # the table -- so setting it to 2.45 puts the firing-rate row on exactly the Fano-
        # factor row's neurons without re-reading anything. None = keep MIN_FR_HZ's 1.0 Hz.
        COHORT_MIN_FR_HZ=None,

        # --- Fano-factor quench onset (ff_quench.ipynb's derivative-threshold latency)
        SEARCH_WINDOW=(-0.2, 0.2),
        PEAK_SEARCH_WINDOW=(-0.05, 0.08),
        ONSET_THRESHOLD_FRAC=0.25,
        SMOOTH_SIGMA=1.0,            # bins

        # --- firing-rate response onset (per neuron)
        # The FF quench onset works without a responsiveness gate because it measures a
        # drop every neuron is assumed to show. A RATE onset cannot: 64% of neurons are
        # excited and 36% suppressed, so an argmax finds the start of a rise in one group
        # and a noise peak in the other, and an unmodulated neuron yields a number drawn
        # from wherever the noise happened to peak. So two things are done here that the
        # FF onset does not do:
        #   1. RESPONSIVENESS GATE. Per neuron, the across-trial mean of (post - pre) over
        #      its standard error. Neurons below FR_RESPONSIVE_Z get no onset (NaN), which
        #      makes this column a SELECTED SUBSET -- unlike the three FR tests beside it.
        #      Whether the pass rate itself tracks lda_1 is reported by compute_tables and
        #      has to be checked before the onset effect is interpreted.
        #   2. SIGN FLIP. Suppressed neurons' curves are negated so that every retained
        #      neuron shows a rise, and one definition of onset applies to both.
        # The latency itself is then the same velocity-threshold rule the FF onset uses:
        # peak, then back to the first crossing of FR_ONSET_THRESHOLD_FRAC of the steepest
        # slope leading up to it.
        FR_ONSET=True,
        FR_SEARCH_WINDOW=(-0.2, 0.35),
        FR_PEAK_SEARCH_WINDOW=(0.0, 0.30),
        FR_ONSET_THRESHOLD_FRAC=0.25,
        FR_RESPONSIVE_Z=2.0,

        # --- test
        N_PERM=2000,
        SEED=0,
        FDR_METHOD='fdr_bh',
        # WHICH TESTS FORM ONE FAMILY. 'metric' corrects within a row -- the four Fano-factor
        # tests among themselves, the three firing-rate ones among themselves -- which is the
        # family a reader comparing windows WITHIN one metric is exposed to. 'all' corrects
        # across the whole grid of ten, the family a reader scanning the figure for any
        # effect is exposed to. Both are computed either way and both are printed; this
        # setting only picks which one `q_fdr` (and so the figure's annotation) reports.
        FDR_FAMILY='metric',        # 'metric' | 'all'
        ALPHA=0.05,
    )


def _first_existing(paths, what, verbose=True):
    """The pinned choice first, fallbacks after it. Never silent: the pick is printed and a
    fallback says so, because a wrong input file here surfaces as a plausible-looking
    number rather than an error."""
    paths = [Path(p) for p in paths]
    for i, p in enumerate(paths):
        if p.exists():
            if verbose:
                print(f'{what}: {p.name}' + ('' if i == 0 else
                      f'   <-- FALLBACK, not the pinned {paths[0].name}'))
            return p
    raise FileNotFoundError(f'no {what} found. Looked for:\n  '
                            + '\n  '.join(str(p) for p in paths))


# =============================================================================
# INPUTS
# =============================================================================

def load_lda(cfg, verbose=True):
    """The per-session LDA embedding. Returns (lda_1 by session, bin by session)."""
    f = _first_existing(cfg['LDA_FILES'], 'LDA file', verbose)
    lda = pd.read_pickle(f).rename(columns={0: 'lda_1'})
    lda['bin'] = lda['binned1'].cat.codes
    if verbose:
        print(f"  {lda['session'].nunique()} sessions, bins {sorted(lda['bin'].unique())}")
    return (dict(zip(lda['session'], lda['lda_1'])),
            dict(zip(lda['session'], lda['bin'])))


def load_lda_frame(cfg, components=(1, 2, 3), verbose=False):
    """Per-session LDA coordinates, one column `lda_<k>` per requested component.

    load_lda above returns only component 1, because that is the axis the paper is about.
    This returns as many as are asked for, so a panel or a test can be repeated on LD2/LD3
    -- the leading discriminants are near-tied in explained variance and rotate among
    themselves, so LD1 is one slice of a subspace rather than the only direction in it.

    The LDA file's columns are INTEGER-indexed from 0, so component k is column k-1. A
    component the file does not have is skipped rather than invented; the caller sees which
    columns came back.
    """
    f = _first_existing(cfg['LDA_FILES'], 'LDA file', verbose)
    lda = pd.read_pickle(f)
    cols = {}
    for k in components:
        if (k - 1) in lda.columns:
            cols[f'lda_{k}'] = lda[k - 1].astype(float)
        elif verbose:
            print(f'  LDA file has no component {k} (column {k - 1}) -- skipped')
    out = pd.DataFrame({'session': lda['session'], **cols})
    return out.dropna(subset=['session']).drop_duplicates('session').reset_index(drop=True)


def attach_lda(tables, cfg=None, components=(1, 2, 3), verbose=False):
    """Add the `lda_<k>` columns to every metric table, by session.

    Done at the COHORT layer, not in compute_tables, on purpose: the tables are cached
    parquet files that take a pass over ~400 pkl files to rebuild, and which LDA components
    one wants to look at is a question asked long after that pass. Attaching here means
    LD2/LD3 cost a dict lookup instead of RECOMPUTE=True. lda_1 is left alone if it is
    already there, so the cached column -- the one every published number was computed
    from -- is never silently replaced.
    """
    cfg = cfg or default_config()
    frame = load_lda_frame(cfg, components=components, verbose=verbose)
    have = [c for c in frame.columns if c != 'session']
    maps = {c: dict(zip(frame['session'], frame[c])) for c in have}
    out = dict(tables)
    added = set()
    for k, t in out.items():
        if not isinstance(t, pd.DataFrame) or 'session' not in t.columns:
            continue
        t = t.copy()
        for c, m in maps.items():
            if c not in t.columns:
                t[c] = t['session'].map(m)
                added.add(c)
        out[k] = t
    if verbose and added:
        print(f"  attached {', '.join(sorted(added))} to the metric tables")
    return out


def load_trial_meta(cfg, verbose=True):
    """Per-trial metadata. Returns (correct trial ids by session, goCue times by session,
    mouse name by session).

    The goCue times are used only to VALIDATE that a firing-rate file's trial_id means the
    same trial ONE means by it -- noise_corr_psth_lda.ipynb found sessions where it does
    not, and skipped them. Keeping that check is what makes one session set defensible
    across all three metrics.
    """
    f = _first_existing(cfg['TRIAL_META_FILES'], 'trial meta file', verbose)
    meta = pd.read_parquet(f)
    correct = (meta[meta['feedback'] == 'correct']
               .groupby('session')['trial_id'].apply(set).to_dict())
    gocue = {s: g.set_index('trial_id')['goCueTrigger_times']
             for s, g in meta.groupby('session')}
    mouse = (meta.groupby('session')['mouse_name'].first().to_dict()
             if 'mouse_name' in meta.columns else {})
    return correct, gocue, mouse


# =============================================================================
# METRIC PRIMITIVES -- lifted from the three notebooks, arithmetic untouched
# =============================================================================

def smooth_time(A, W, mode):
    """Sliding-window mean over the last axis.

    'causal'   value at t uses bins (t-W, t]   -- the notebooks' default
    'centered' symmetric window
    'leading'  value at t uses bins [t, t+W)   -- the window DIRECTION the legacy
               psths_fanofactor file was built with, needed to reproduce the quench-onset
               timing (a causal window shifts every latency by the window length)
    """
    if mode == 'centered':
        return uniform_filter1d(A, W, axis=-1, mode='nearest')
    if mode == 'leading':
        c = np.cumsum(A, axis=-1); Tn = A.shape[-1]; out = np.empty_like(A)
        out[..., :Tn - W] = (c[..., W:] - c[..., :Tn - W]) / W
        for k in range(W):
            idx = Tn - W + k
            prev = c[..., idx - 1] if idx > 0 else 0.0
            out[..., idx] = (c[..., -1] - prev) / (Tn - idx)
        return out
    c = np.cumsum(A, axis=-1); out = np.empty_like(A)
    out[..., :W] = c[..., :W] / np.arange(1, W + 1)
    out[..., W:] = (c[..., W:] - c[..., :-W]) / W
    return out


def _condition_residual(X, cond, axis=1):
    """Subtract the per-condition mean along the trial axis, in place on a copy. This is
    what makes the variance a NOISE variance: the stimulus (side x contrast) contribution
    to trial-to-trial spread is removed first, so what is left is variability at fixed
    stimulus."""
    R = X.astype(float).copy()
    for cc in pd.unique(cond):
        ci = np.where(cond == cc)[0]
        R[:, ci] -= np.nanmean(X[:, ci], axis=axis, keepdims=True)
    return R


def _rate_floor_mask(rate_pre, rate_post, floor, require='both'):
    """Which neurons clear a rate floor, given which window(s) have to clear it.

    Kept in one function because the gate is applied in two places -- the region-level
    neuron set and the firing-rate table -- and the two silently disagreeing is exactly
    how FF ended up 2.45x stricter than FR for so long.
    """
    ok = np.isfinite(rate_pre) & np.isfinite(rate_post)
    if require == 'pre':
        return ok & (rate_pre >= floor)
    if require == 'post':
        return ok & (rate_post >= floor)
    if require == 'either':
        return ok & ((rate_pre >= floor) | (rate_post >= floor))
    if require == 'both':
        return ok & (rate_pre >= floor) & (rate_post >= floor)
    raise ValueError("FR_FLOOR_REQUIRE must be 'both', 'pre', 'post' or 'either'")


def window_ff(counts, cond, cfg, min_count=None):
    """counts: neurons x trials, spike counts summed over ONE fixed window. Per-neuron
    condition-adjusted Fano factor -- the paper's single-neuron FF metric.

    `min_count` is the gate in SPIKES for this window. The caller derives it from the
    shared MIN_FR_HZ and the window's own duration, so the same floor in Hz applies to
    windows of different lengths; it falls back to cfg['MIN_WINDOW_COUNT'] (the original
    absolute gate) when not given.
    """
    if min_count is None:
        min_count = cfg['MIN_WINDOW_COUNT']
    mean = np.nanmean(counts, axis=1)
    if cfg['REMOVE_CONDITION']:
        var = np.nanvar(_condition_residual(counts, cond), axis=1, ddof=1)
    else:
        var = np.nanvar(counts, axis=1, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ff = var / (mean + 1e-6)
    ff[(mean <= min_count) | ~np.isfinite(ff)] = np.nan
    return ff


def ff_timecourse(A, cond, min_count, cfg):
    """A: neurons x trials x T windowed spike counts. The same Fano factor as window_ff,
    but collapsed on neither axis -- one FF curve per neuron, which is what the quench
    onset needs (each neuron's own drop, not the region average's)."""
    mean = np.nanmean(A, axis=1)
    if cfg['REMOVE_CONDITION']:
        var = np.nanvar(_condition_residual(A, cond), axis=1, ddof=1)
    else:
        var = np.nanvar(A, axis=1, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ff = var / (mean + 1e-6)
    ff[(mean <= min_count) | ~np.isfinite(ff)] = np.nan
    return ff


def ff_curve_region(A, cond, cfg):
    """A: neurons x trials x T windowed spike counts. Condition-adjusted Fano factor per
    time bin, averaged over the region's neurons -- the descriptive FF time course that the
    PSTH-by-LDA-bin panel shows (as distinct from `window_ff`, which is the tested
    single-neuron metric)."""
    mean = np.nanmean(A, axis=1)
    if cfg['REMOVE_CONDITION'] and cond is not None:
        var = np.nanvar(_condition_residual(A, cond), axis=1, ddof=1)
    else:
        var = np.nanvar(A, axis=1, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ff = var / (mean + 1e-6)
    ff[(mean <= cfg['MIN_MEAN_COUNT']) | ~np.isfinite(ff)] = np.nan
    return np.nanmean(ff, axis=0)


def rsc_curve(A, cond, cfg):
    """A: neurons x trials x T over trials COMPLETE for every neuron in the region. Mean
    off-diagonal pairwise correlation per time bin, after removing condition means -> the
    spike-count noise correlation r_SC.

    Computed from the sum of z-scores rather than a pairwise matrix: sum_ij z_i z_j is the
    square of the column sum minus the diagonal, so the mean off-diagonal correlation
    comes out in one pass over neurons instead of n^2 pair correlations.

    Returns (off, nvalid). `nvalid` is the number of neurons with non-zero across-trial
    variance IN THAT BIN, and it is returned rather than kept internal because it is the
    estimator's real sample size: a correlation matrix over n neurons cannot have a mean
    off-diagonal below -1/(n-1), and this estimator's floor is -1/(nvalid-1). When most
    of a region is silent in a bin, nvalid collapses to a handful and the estimate sits on
    that floor -- which is not a measurement of anything. The caller gates on it.
    """
    R = A.astype(float)
    if cfg['REMOVE_CONDITION'] and cond is not None:
        R = _condition_residual(A, cond)
    mean = np.nanmean(R, axis=1, keepdims=True)
    std = np.nanstd(R, axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        # a neuron silent across every trial in a bin has std 0 -> its z is not defined;
        # nvalid below counts it out rather than letting it poison the pair sum
        Z = (R - mean) / std
    Z[~np.isfinite(Z)] = np.nan
    nL = R.shape[1]
    s = np.nansum(Z, axis=0)
    sumsq = np.nansum(s ** 2, axis=0)
    nvalid = ((np.isfinite(std[:, 0, :])) & (std[:, 0, :] > 0)).sum(0).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        off = (sumsq / (nL - 1) - nvalid) / (nvalid * (nvalid - 1))
    off[nvalid < 2] = np.nan
    return off, nvalid


_FUNCTIONAL_CACHE = {}


def functional_group_map(area_acronyms, br):
    """The ad-hoc functional grouping ff_quench.ipynb uses as the quench-onset region
    covariate: Hippocampus / Thalamus / Visual / Motor / Somatosensory cortex called out by
    name or ancestry, everything else falling back to its Cosmos parent. Kept because the
    onset test's covariate structure has to match the analysis it reproduces -- it is not
    this module's own region convention."""
    unknown = [a for a in pd.unique(area_acronyms)
               if a not in _FUNCTIONAL_CACHE
               and not (isinstance(a, float) and np.isnan(a))]
    if unknown:
        cosmos = dict(zip(unknown, br.acronym2acronym(unknown, mapping='Cosmos')))
        for a in unknown:
            try:
                anc = br.ancestors(ids=br.acronym2id(a))['acronym']
                if 'TH' in anc or a in ['PO', 'LP']:
                    _FUNCTIONAL_CACHE[a] = 'Thalamus'
                elif 'HPF' in anc or a in ['CA1', 'DG']:
                    _FUNCTIONAL_CACHE[a] = 'Hippocampus'
                elif 'Isocortex' in anc and any(v in a for v in ['VIS', 'RSP']):
                    _FUNCTIONAL_CACHE[a] = 'Visual Cortex'
                elif 'Isocortex' in anc:
                    if any(m in a for m in ['MOp', 'MOs', 'MO']):
                        _FUNCTIONAL_CACHE[a] = 'Motor Cortex'
                    elif any(s in a for s in ['SSp', 'SSs', 'SS']):
                        _FUNCTIONAL_CACHE[a] = 'Somatosensory Cortex'
                    else:
                        _FUNCTIONAL_CACHE[a] = 'Other Cortex'
                else:
                    _FUNCTIONAL_CACHE[a] = cosmos.get(a, 'Other')
            except Exception:
                _FUNCTIONAL_CACHE[a] = cosmos.get(a, 'Unassigned')
    return pd.Series(area_acronyms).map(_FUNCTIONAL_CACHE).values


def response_onset(psth, t_search, peak_mask, binwidth, frac):
    """One neuron's firing-rate response onset, on a curve already oriented so that the
    response is a RISE (suppressed neurons negated by the caller).

    The mirror image of `quench_onset`: find the response peak, then walk BACK to the first
    moment the rise reached `frac` of its steepest slope. Falls back to the peak time when
    no crossing exists, exactly as the quench version does.
    """
    i_peak = np.argmax(psth[peak_mask])
    t_peak = t_search[peak_mask][i_peak]
    deriv = np.gradient(psth, binwidth)
    before = t_search <= t_peak
    if not before.any():
        return t_peak
    threshold = np.max(deriv[before]) * frac
    if not np.isfinite(threshold) or threshold <= 0:
        return t_peak
    hits = np.where((deriv > threshold) & before)[0]
    return t_search[hits[0]] if len(hits) else t_peak


def quench_onset(ff_smooth, t_search, peak_mask, binwidth, cfg):
    """One neuron's FF quench onset time.

    Not where the curve bottoms out -- where the drop STARTS. Find the FF bump peak inside
    a narrow window at stimulus onset, then walk forward to the first time the derivative
    crosses ONSET_THRESHOLD_FRAC of its steepest post-peak downward slope (a standard
    velocity-threshold latency). Falls back to the peak time when there is no crossing.
    """
    i_peak = np.argmax(ff_smooth[peak_mask])
    t_peak = t_search[peak_mask][i_peak]
    deriv = np.gradient(ff_smooth, binwidth)
    after = t_search >= t_peak
    threshold = np.min(deriv[after]) * cfg['ONSET_THRESHOLD_FRAC']
    hits = np.where((deriv < threshold) & after)[0]
    return t_search[hits[0]] if len(hits) else t_peak


# =============================================================================
# THE ONE PASS OVER THE FIRING-RATE FILES
# =============================================================================

def compute_tables(cfg=None, verbose=True, max_files=None):
    """Read every firing-rate pkl file once and return the three metric tables.

    NO cohort filter is applied: every region with at least MIN_NEURONS neurons is kept and
    each row carries its raw beryl acronym plus its session's goCue-validation flag, so
    `apply_cohort` can impose a region list or a session set afterwards without a re-read.
    Region selection is post-hoc-safe because all three metrics are computed WITHIN a
    region (FF and FR per neuron, r_SC over pairs inside one region), so dropping region X
    cannot change region Y's numbers.

    Returns {'fr': ..., 'ff': ..., 'rsc': ..., 'sessions': ..., 'time': ...}.
    """
    from iblatlas.regions import BrainRegions
    cfg = cfg or default_config()
    br = BrainRegions()
    lda1_map, bin_map = load_lda(cfg, verbose)
    # r_SC picks its neurons while the files are read, so its presence-ratio gate has to be
    # here rather than in apply_cohort
    pr_map, sess_fr_map = None, None
    _need_qc = (cfg.get('MIN_PRESENCE_RATIO') is not None
                or cfg.get('FR_FLOOR_SOURCE') == 'session')
    if _need_qc:
        _qc = load_unit_qc(verbose=verbose)
        if cfg.get('MIN_PRESENCE_RATIO') is not None:
            pr_map = dict(zip(_qc['nuid'], _qc['presence_ratio']))
        if cfg.get('FR_FLOOR_SOURCE') == 'session':
            sess_fr_map = dict(zip(_qc['nuid'], _qc['firing_rate']))
    correct_by_session, gocue_by_session, mouse_by_session = load_trial_meta(cfg, verbose)

    fr_dir = Path(cfg['FIRING_RATES_DIR'])
    pkl_files = sorted(f for f in os.listdir(fr_dir) if f.startswith('firing_rate_'))
    if max_files:
        pkl_files = pkl_files[:max_files]

    # --- the time axis, from the first file (identical in all of them)
    with open(fr_dir / pkl_files[0], 'rb') as f:
        s0 = pickle.load(f)
    tcols = sorted([c for c in s0.columns if c.startswith('t_')],
                   key=lambda x: float(x.split('_')[1]))
    tsec = np.array([float(c.split('_')[1]) for c in tcols])
    T = len(tcols)
    binwidth = float(np.median(np.diff(tsec)))

    pre_mask = (tsec >= cfg['PRE_WINDOW'][0]) & (tsec < cfg['PRE_WINDOW'][1])
    post_mask = (tsec >= cfg['POST_WINDOW'][0]) & (tsec < cfg['POST_WINDOW'][1])
    # half-open, matching pre_mask/post_mask above: "the same window" has to mean the same
    # set of time bins, and the noise-correlation notebook's closed interval took one bin
    # more at each edge than FF and FR did
    rsc_pre_mask = (tsec >= cfg['RSC_PRE_WINDOW'][0]) & (tsec < cfg['RSC_PRE_WINDOW'][1])
    rsc_post_mask = (tsec >= cfg['RSC_POST_WINDOW'][0]) & (tsec < cfg['RSC_POST_WINDOW'][1])
    rsc_post_legacy_mask = ((tsec >= cfg['RSC_POST_WINDOW_LEGACY'][0])
                            & (tsec <= cfg['RSC_POST_WINDOW_LEGACY'][1]))
    # the bins a trial must be complete in for r_SC -- see RSC_COMPLETE_WINDOW
    if cfg.get('RSC_COMPLETE_WINDOW') is None:
        rsc_need_mask = np.ones(T, dtype=bool)
    else:
        _lo_c, _hi_c = cfg['RSC_COMPLETE_WINDOW']
        rsc_need_mask = (tsec >= _lo_c - 1e-9) & (tsec <= _hi_c + 1e-9)

    # Gaussian smoothing needs real data on both sides of the search window or the
    # derivative picks up an edge artefact exactly where the onset is looked for. Pad by
    # the kernel's radius (scipy's truncate=4), smooth the padded slice, then cut back.
    search_mask = (tsec >= cfg['SEARCH_WINDOW'][0]) & (tsec <= cfg['SEARCH_WINDOW'][1])
    t_search = tsec[search_mask]
    peak_local_mask = ((t_search >= cfg['PEAK_SEARCH_WINDOW'][0])
                       & (t_search <= cfg['PEAK_SEARCH_WINDOW'][1]))
    _si = np.where(search_mask)[0]
    _pad = int(4 * cfg['SMOOTH_SIGMA']) + 1
    _lo, _hi = max(0, _si[0] - _pad), min(T, _si[-1] + 1 + _pad)
    pad_mask = np.zeros(T, dtype=bool); pad_mask[_lo:_hi] = True
    pad_offset = _si[0] - _lo

    # THE SHARED RATE FLOOR, converted into each window's own spike-count gate. A window
    # is pre_mask.sum() bins long, so MIN_FR_HZ Hz is MIN_FR_HZ * duration spikes in it.
    pre_dur, post_dur = pre_mask.sum() * binwidth, post_mask.sum() * binwidth
    if cfg.get('UNIFORM_FR_FLOOR', False):
        ff_pre_gate, ff_post_gate = cfg['MIN_FR_HZ'] * pre_dur, cfg['MIN_FR_HZ'] * post_dur
        floor_hz = cfg['MIN_FR_HZ']
    else:
        ff_pre_gate = ff_post_gate = cfg['MIN_WINDOW_COUNT']
        floor_hz = cfg.get('MIN_FR_HZ_RSC')

    if verbose:
        print(f'{T} time bins, {tsec.min():.2f}..{tsec.max():.2f} s, binwidth '
              f'{binwidth*1000:.1f} ms; pre {pre_mask.sum()} bins, post {post_mask.sum()} '
              f'bins | {len(pkl_files)} files')
        if cfg.get('UNIFORM_FR_FLOOR', False):
            _src = cfg.get('FR_FLOOR_SOURCE', 'window')
            _where = ('whole-recording rate (sorter QC table)' if _src == 'session'
                      else f"{cfg.get('FR_FLOOR_REQUIRE', 'both')} analysis window(s)")
            print(f"  one rate floor: {cfg['MIN_FR_HZ']} Hz for FR, FF and r_SC, "
                  f"from the {_where} "
                  f"(FF gate = {ff_pre_gate:.3f} / {ff_post_gate:.3f} spikes per window)")
            print(f"  min neurons: {cfg['MIN_NEURONS']} per region"
                  f"{', r_SC ' + str(cfg['MIN_NEURONS_RSC']) if cfg.get('MIN_NEURONS_RSC') else ''}")
        else:
            print(f"  per-metric gates: FR {cfg['MIN_FR_HZ']} Hz | FF "
                  f"{cfg['MIN_WINDOW_COUNT']} spikes ({cfg['MIN_WINDOW_COUNT']/pre_dur:.2f} Hz)"
                  f" | r_SC {floor_hz or 'none'}")

    # the FR onset's own search / peak / padding masks, mirroring the FF ones above
    fr_search_mask = ((tsec >= cfg['FR_SEARCH_WINDOW'][0])
                      & (tsec <= cfg['FR_SEARCH_WINDOW'][1]))
    fr_t_search = tsec[fr_search_mask]
    fr_peak_local = ((fr_t_search >= cfg['FR_PEAK_SEARCH_WINDOW'][0])
                     & (fr_t_search <= cfg['FR_PEAK_SEARCH_WINDOW'][1]))
    _fi = np.where(fr_search_mask)[0]
    _flo, _fhi = max(0, _fi[0] - _pad), min(T, _fi[-1] + 1 + _pad)
    fr_pad_mask = np.zeros(T, dtype=bool); fr_pad_mask[_flo:_fhi] = True
    fr_pad_offset = _fi[0] - _flo

    fr_rows, ff_rows, onset_rows, rsc_rows, sess_rows = [], [], [], [], []
    n_drop_pr = 0        # neurons dropped by the presence-ratio gate
    n_drop_nvalid = 0    # r_SC rows dropped for too few neurons with variance in-window
    n_fr_resp = n_fr_tested = 0   # firing-rate responsiveness gate pass rate
    curve_rows = []      # one descriptive time course per metric x session x region
    n_gocue_fail = 0
    for i, fn in enumerate(pkl_files):
        try:
            with open(fr_dir / fn, 'rb') as f:
                d = pickle.load(f)
            d = d[~d['area'].isin(cfg['DROP'])]
            if len(d) == 0:
                continue
            session = d['session'].iloc[0]
            if session not in lda1_map:
                continue

            d = d.copy()
            d['nuid'] = d['pid'].astype(str) + '__' + d['neuron_id'].astype(str)
            neurons = sorted(d['nuid'].unique()); nidx = {n: k for k, n in enumerate(neurons)}
            trials = sorted(d['trial_id'].unique()); tix = {t: k for k, t in enumerate(trials)}

            # Does this file's trial_id mean what ONE means by it? Recorded per session
            # rather than acted on here, so the cohort decision stays in one place.
            gocue = gocue_by_session.get(session)
            if gocue is None:
                gocue_ok = False
            else:
                own = (d.drop_duplicates('trial_id').set_index('trial_id')['event_time']
                       .reindex(trials))
                gocue_ok = not ((own - gocue.reindex(trials)).abs() > cfg['GOCUE_TOL']).any()
            if not gocue_ok:
                n_gocue_fail += 1

            # A: neurons x trials x T, in Hz (the t_* columns are already rates)
            A = np.full((len(neurons), len(trials), T), np.nan)
            A[d['nuid'].map(nidx).values, d['trial_id'].map(tix).values, :] = d[tcols].values
            counts = A * binwidth      # spike counts -- what a Fano factor needs

            area_raw = d.groupby('nuid')['area'].first().reindex(neurons)
            functional = functional_group_map(area_raw.values, br)
            area = area_raw
            if cfg['REGION_LEVEL'] == 'cosmos':
                uniq = area_raw.dropna().unique()
                area = area_raw.map(dict(zip(uniq, br.acronym2acronym(uniq, mapping='Cosmos'))))
            neu_area = area.values
            neu_beryl = area_raw.values

            cond_all = (d.drop_duplicates('trial_id').set_index('trial_id')['condition']
                        .reindex(trials).values)
            if cfg['CORRECT_ONLY']:
                cs = correct_by_session.get(session, set())
                trial_ok = np.array([t in cs for t in trials], dtype=bool)
            else:
                trial_ok = np.ones(len(trials), dtype=bool)
            valid_idx = np.where(trial_ok)[0]

            sess_rows.append(dict(session=session, pid_file=fn, gocue_ok=gocue_ok,
                                  mouse_name=mouse_by_session.get(session),
                                  lda_1=lda1_map[session], bin=bin_map[session],
                                  n_trials_total=len(trials), n_trials_valid=len(valid_idx)))

            _min_n_rsc = cfg.get('MIN_NEURONS_RSC') or cfg['MIN_NEURONS']
            for region in pd.unique(neu_area):
                if region is None or (isinstance(region, float) and np.isnan(region)):
                    continue
                ni = np.where(neu_area == region)[0]
                # THE PRESENCE-RATIO GATE, applied to the region's neuron set before
                # anything is computed from it, so firing rate, Fano factor, the quench
                # onset, r_SC AND the descriptive time courses all describe the same
                # neurons. A neuron with no QC row is dropped rather than kept: an
                # un-checked neuron passing a quality gate by default is how a gate
                # silently stops being one.
                if pr_map is not None:
                    _pr = np.array([pr_map.get(n, np.nan)
                                    for n in np.asarray(neurons, dtype=object)[ni]], float)
                    n_drop_pr += int((~(np.nan_to_num(_pr, nan=-1.0)
                                        >= cfg['MIN_PRESENCE_RATIO'])).sum())
                    ni = ni[np.nan_to_num(_pr, nan=-1.0) >= cfg['MIN_PRESENCE_RATIO']]

                # THE RATE FLOOR, also on the region's neuron set and also before the
                # MIN_NEURONS check, so that "at least 15 neurons" means at least 15
                # USABLE neurons for EVERY metric. It used to be applied per neuron inside
                # the firing-rate and Fano-factor blocks -- i.e. after the region had
                # already been admitted -- so a region could pass with 15 neurons and then
                # contribute three, while r_SC re-checked and so required 15 that cleared
                # the floor. Only r_SC was counting what it actually used.
                # Skipped under UNIFORM_FR_FLOOR=False, where each metric keeps its own
                # gate, which is what the reproduction path needs.
                if floor_hz is not None and len(ni):
                    if sess_fr_map is not None:
                        # the sorter's whole-recording rate: one number per neuron, with
                        # no window and no stimulus in it
                        _sr = np.array([sess_fr_map.get(n, np.nan) for n
                                        in np.asarray(neurons, dtype=object)[ni]], float)
                        ni = ni[np.nan_to_num(_sr, nan=-1.0) >= floor_hz]
                    else:
                        _rp = np.nanmean(np.nanmean(A[ni][:, valid_idx][:, :, pre_mask],
                                                    axis=2), axis=1)
                        _rq = np.nanmean(np.nanmean(A[ni][:, valid_idx][:, :, post_mask],
                                                    axis=2), axis=1)
                        ni = ni[_rate_floor_mask(_rp, _rq, floor_hz,
                                                 cfg.get('FR_FLOOR_REQUIRE', 'both'))]
                if len(ni) < cfg['MIN_NEURONS']:
                    continue
                nu_names = np.array(neurons)[ni]
                # the beryl acronym is carried alongside so a cosmos run can still be
                # filtered on a beryl region list
                beryl = pd.Series(neu_beryl[ni]).mode()
                beryl = beryl.iloc[0] if len(beryl) else region

                # ---------------- noise correlation (session x region) ----------------
                # r_SC is pairwise, so it needs trials on which EVERY neuron in the region
                # has data -- a NaN for one neuron would otherwise silently drop different
                # trials from different pairs.
                # an optional rate floor on the neurons that enter the pair matrix, the
                # same gate FF and FR already apply -- see MIN_FR_HZ_RSC in default_config
                # r_SC now takes the SAME rate floor as FR and FF. Without it, a region
                # of near-silent neurons leaves only a handful with non-zero variance in
                # any time bin, and the estimator collapses onto its own -1/(nvalid-1)
                # floor -- which is where the -0.279 outlier came from.
                ni_rsc = ni      # presence-ratio AND rate gated above, like every metric
                Asub = A[ni_rsc]
                complete = (~np.isnan(Asub[:, :, rsc_need_mask]).any(axis=(0, 2))
                            if len(ni_rsc) else np.zeros(len(trials), bool))
                comp_idx = np.where(complete & trial_ok)[0]
                if len(ni_rsc) >= _min_n_rsc and len(comp_idx) >= cfg['MIN_TRIALS_RSC']:
                    Aw = smooth_time(Asub[:, comp_idx, :], cfg['WIN_BINS'], cfg['SMOOTH_MODE'])
                    curve, nvalid = rsc_curve(Aw, cond_all[comp_idx], cfg)
                    # the estimator's own sample size, over the bins actually read
                    _tested = rsc_pre_mask | rsc_post_mask | rsc_post_legacy_mask
                    nvalid_min = int(np.nanmin(nvalid[_tested])) if _tested.any() else 0
                    if not (cfg.get('REQUIRE_RSC_NVALID', True)
                            and nvalid_min < _min_n_rsc):
                        rsc_rows.append(dict(
                            session=session, region=region, beryl=beryl,
                            n_neurons=len(ni_rsc), n_trials=len(comp_idx),
                            nvalid_min=nvalid_min,
                            r_sc_pre=np.nanmean(curve[rsc_pre_mask]),
                            r_sc_post=np.nanmean(curve[rsc_post_mask]),
                            r_sc_post_legacy=np.nanmean(curve[rsc_post_legacy_mask])))
                        curve_rows.append(dict(metric='rsc', session=session, region=region,
                                               beryl=beryl, n_neurons=len(ni_rsc),
                                               n_trials=len(comp_idx), curve=curve))
                    else:
                        n_drop_nvalid += 1

                if len(valid_idx) < cfg['MIN_TRIALS']:
                    continue
                cond_valid = cond_all[valid_idx]

                # ---------------- descriptive time courses (for the PSTH column) --------
                # Region-averaged, one curve per session x region: firing rate is the trial-
                # then-neuron mean rate, Fano factor the condition-adjusted var/mean of the
                # WIN_BINS-windowed counts. These are what the PSTH panel averages within
                # an LDA-1 bin; they are descriptive only -- every test above is run on the
                # per-neuron window metrics, not on these curves.
                curve_rows.append(dict(
                    metric='fr', session=session, region=region, beryl=beryl,
                    n_neurons=len(ni), n_trials=len(valid_idx),
                    curve=np.nanmean(np.nanmean(A[ni][:, valid_idx, :], axis=1), axis=0)))
                curve_rows.append(dict(
                    metric='ff', session=session, region=region, beryl=beryl,
                    n_neurons=len(ni), n_trials=len(valid_idx),
                    curve=ff_curve_region(
                        smooth_time(counts[ni][:, valid_idx, :], cfg['WIN_BINS'],
                                    cfg['SMOOTH_MODE']) * cfg['WIN_BINS'],
                        cond_valid, cfg)))

                # ---------------- firing rate (per neuron) ----------------
                fr_pre = np.nanmean(np.nanmean(A[ni][:, valid_idx][:, :, pre_mask], axis=2), axis=1)
                fr_post = np.nanmean(np.nanmean(A[ni][:, valid_idx][:, :, post_mask], axis=2), axis=1)
                if sess_fr_map is not None:
                    # ni was already gated on the whole-recording rate; re-gating on the
                    # windows here would quietly reintroduce the window criterion the
                    # session floor exists to avoid
                    keep = np.isfinite(fr_pre) & np.isfinite(fr_post)
                else:
                    keep = _rate_floor_mask(fr_pre, fr_post, cfg['MIN_FR_HZ'],
                                            cfg.get('FR_FLOOR_REQUIRE', 'both'))
                # ---- response onset + responsiveness, per neuron (see FR_ONSET) ----
                fr_onset = np.full(len(ni), np.nan)
                fr_z = np.full(len(ni), np.nan)
                if cfg.get('FR_ONSET', True) and fr_peak_local.any():
                    pre_tr = np.nanmean(A[ni][:, valid_idx][:, :, pre_mask], axis=2)
                    post_tr = np.nanmean(A[ni][:, valid_idx][:, :, post_mask], axis=2)
                    d = post_tr - pre_tr
                    n_ok = np.sum(np.isfinite(d), axis=1)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        sem = np.nanstd(d, axis=1, ddof=1) / np.sqrt(np.maximum(n_ok, 1))
                        fr_z = np.nanmean(d, axis=1) / sem
                    # leading window, like the FF onset: a causal one would shift every
                    # latency by the window length
                    psth = np.nanmean(smooth_time(A[ni][:, valid_idx, :],
                                                  cfg['WIN_BINS'], 'leading'), axis=1)
                    # orient every retained neuron as a rise
                    sign = np.where(np.nanmean(d, axis=1) < 0, -1.0, 1.0)
                    psth = psth * sign[:, None]
                    resp = np.isfinite(fr_z) & (np.abs(fr_z) >= cfg['FR_RESPONSIVE_Z'])
                    pad = psth[:, fr_pad_mask]
                    finite = np.all(np.isfinite(pad), axis=1) & resp
                    if finite.any():
                        sm = gaussian_filter1d(pad[finite], sigma=cfg['SMOOTH_SIGMA'], axis=1)
                        sm = sm[:, fr_pad_offset:fr_pad_offset + len(fr_t_search)]
                        for k, j in enumerate(np.where(finite)[0]):
                            fr_onset[j] = response_onset(
                                sm[k], fr_t_search, fr_peak_local, binwidth,
                                cfg['FR_ONSET_THRESHOLD_FRAC'])
                n_fr_resp += int(np.sum(np.isfinite(fr_onset)))
                n_fr_tested += int(keep.sum())

                for nu, a, b, o, z in zip(nu_names[keep], fr_pre[keep], fr_post[keep],
                                          fr_onset[keep], fr_z[keep]):
                    fr_rows.append(dict(session=session, region=region, beryl=beryl, nuid=nu,
                                        n_trials=len(valid_idx), fr_pre=a, fr_post=b,
                                        fr_evoked=b - a, fr_onset_time=o, fr_resp_z=z))

                # ---------------- Fano factor (per neuron) ----------------
                c_pre = np.nansum(counts[ni][:, valid_idx][:, :, pre_mask], axis=2)
                c_post = np.nansum(counts[ni][:, valid_idx][:, :, post_mask], axis=2)
                ff_pre = window_ff(c_pre, cond_valid, cfg, ff_pre_gate)
                ff_post = window_ff(c_post, cond_valid, cfg, ff_post_gate)
                keep = np.isfinite(ff_pre) & np.isfinite(ff_post)
                for nu, a, b in zip(nu_names[keep], ff_pre[keep], ff_post[keep]):
                    ff_rows.append(dict(session=session, region=region, beryl=beryl, nuid=nu,
                                        n_trials=len(valid_idx), ff_pre=a, ff_post=b,
                                        ff_quench=a - b))

                # ---------------- Fano-factor quench onset (per neuron) ----------------
                # The leading window is not a stylistic choice: the onset time is read off
                # this curve, and a causal window would shift every latency by WIN_BINS.
                Aw_neu = (smooth_time(counts[ni][:, valid_idx, :], cfg['WIN_BINS'], 'leading')
                          * cfg['WIN_BINS'])
                ff_pad = ff_timecourse(Aw_neu[:, :, pad_mask], cond_valid,
                                       cfg['MIN_MEAN_COUNT'], cfg)
                finite = np.all(np.isfinite(ff_pad), axis=1)
                if finite.any() and peak_local_mask.any():
                    sm = gaussian_filter1d(ff_pad[finite], sigma=cfg['SMOOTH_SIGMA'], axis=1)
                    sm = sm[:, pad_offset:pad_offset + len(t_search)]
                    for k, nu in enumerate(nu_names[finite]):
                        onset_rows.append(dict(
                            session=session, nuid=nu,
                            quench_onset_time=quench_onset(sm[k], t_search, peak_local_mask,
                                                           binwidth, cfg),
                            functional_region=functional[ni][finite][k]))
            if verbose and (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(pkl_files)} files...')
        except Exception as e:
            print(f'Error {fn}: {type(e).__name__}: {e}')

    sessions = pd.DataFrame(sess_rows).drop_duplicates('session').reset_index(drop=True)
    fr = pd.DataFrame(fr_rows)
    ff = pd.DataFrame(ff_rows)
    rsc = pd.DataFrame(rsc_rows)
    onset = pd.DataFrame(onset_rows)
    curves = pd.DataFrame(curve_rows)

    # derived columns. Strictly positive, right-skewed metrics are log-transformed; signed
    # differences stay on their natural scale -- the paper's own rule.
    for tbl, cols in ((fr, ('fr_pre', 'fr_post')), (ff, ('ff_pre', 'ff_post'))):
        if len(tbl):
            for c in cols:
                tbl['log_' + c] = np.log(tbl[c].clip(lower=1e-6))
    if len(rsc):
        rsc['r_sc_quench'] = rsc['r_sc_pre'] - rsc['r_sc_post']
        rsc['r_sc_quench_legacy'] = rsc['r_sc_pre'] - rsc['r_sc_post_legacy']
    if len(ff) and len(onset):
        ff = ff.merge(onset, on=['session', 'nuid'], how='left')

    if pr_map is not None:
        for tbl in (fr, ff):
            if len(tbl):
                tbl['presence_ratio'] = tbl['nuid'].map(pr_map)

    gocue_map = dict(zip(sessions['session'], sessions['gocue_ok']))
    for tbl in (fr, ff, rsc, curves):
        if len(tbl):
            tbl['lda_1'] = tbl['session'].map(lda1_map)
            tbl['bin'] = tbl['session'].map(bin_map)
            tbl['mouse_name'] = tbl['session'].map(mouse_by_session)
            tbl['gocue_ok'] = tbl['session'].map(gocue_map)

    if verbose:
        print(f"\nfiring rate : {len(fr):>6} neurons  | {fr['session'].nunique()} sessions")
        print(f"Fano factor : {len(ff):>6} neurons  | {ff['session'].nunique()} sessions"
              f" | onset on {ff['quench_onset_time'].notna().sum()}")
        print(f"noise corr  : {len(rsc):>6} sess-reg | {rsc['session'].nunique()} sessions")
        print(f"curves      : {len(curves):>6} rows     | "
              + ', '.join(f'{m} {n}' for m, n in curves['metric'].value_counts().items()))
        print(f"goCue validation failed for {n_gocue_fail} of {len(pkl_files)} files")
        if n_fr_tested:
            print(f"FR onset: {n_fr_resp} of {n_fr_tested} neurons responsive at "
                  f"|z| >= {cfg['FR_RESPONSIVE_Z']} ({100*n_fr_resp/n_fr_tested:.0f}%) -- "
                  'this column is a SELECTED SUBSET; check whether the pass rate tracks lda_1')
        if n_drop_nvalid:
            print(f'r_SC: dropped {n_drop_nvalid} session-regions with fewer than '
                  f'{cfg.get("MIN_NEURONS_RSC") or cfg["MIN_NEURONS"]} neurons carrying '
                  'variance in the tested windows')
        if pr_map is not None:
            print(f"presence ratio >= {cfg['MIN_PRESENCE_RATIO']}: dropped {n_drop_pr} "
                  f"neuron-region entries")

    return dict(fr=fr, ff=ff, rsc=rsc, curves=curves, sessions=sessions,
                time=pd.DataFrame(dict(tsec=tsec)))


# =============================================================================
# COHORT
# =============================================================================

def apply_cohort(tables, cfg=None, cohort='common', verbose=True):
    """Impose a cohort on the metric tables.

    'legacy'  each metric's own original filters: the 48-region KEEP list for Fano factor,
              every region for firing rate and r_SC, and the goCue check for r_SC only.
              Reproduces the three notebooks' published numbers -- the check that a change
              to this module changed nothing it should not.
    'common'  ONE region list (cfg['KEEP']) and the goCue check for all three metrics, so
              a row of the summary figure is comparable to the row above it. The tables are
              NOT intersected down to a common session set -- see the note below.
    'none'    no filter.
    """
    cfg = cfg or default_config()
    keep = cfg['KEEP']
    out = {k: v.copy() for k, v in tables.items()}
    # LD2/LD3 alongside the cached lda_1, so a test or a panel can be repeated on another
    # component without rebuilding the tables. Adds columns only; lda_1 is never touched.
    out = attach_lda(out, cfg, verbose=verbose)
    reg_col = 'beryl' if cfg['REGION_LEVEL'] == 'cosmos' else 'region'

    def _curves(metric, region_filter, gocue_filter):
        """The descriptive curves for one metric, filtered exactly like that metric's own
        table -- so the PSTH panel and the scatters beside it always describe the same
        sessions and regions."""
        c = out['curves']
        c = c[c['metric'] == metric]
        c = c[~c[reg_col].isin(drop_extra[metric])]
        if region_filter and keep:
            c = c[c[reg_col].isin(keep)]
        return c[c['gocue_ok']] if gocue_filter else c

    if cohort == 'none':
        rules = {'fr': (False, False), 'ff': (False, False), 'rsc': (False, False)}
    elif cohort == 'legacy':
        rules = {'fr': (False, False), 'ff': (True, False), 'rsc': (False, True)}
    elif cohort == 'common':
        rules = {'fr': (True, True), 'ff': (True, True), 'rsc': (True, True)}
    else:
        raise ValueError("cohort must be 'legacy', 'common' or 'none'")

    # 'root'/'void' are unassigned tissue, not a brain region: dropped everywhere except
    # the one place that has to reproduce a notebook which kept them (legacy r_SC).
    drop_extra = {k: cfg['DROP_COHORT'] for k in rules}
    if cohort == 'legacy':
        drop_extra['rsc'] = cfg['DROP']

    for k, (region_filter, gocue_filter) in rules.items():
        t = out[k]
        t = t[~t[reg_col].isin(drop_extra[k])]
        if region_filter and keep:
            t = t[t[reg_col].isin(keep)]
        out[k] = t[t['gocue_ok']] if gocue_filter else t

    # NOTE: the presence-ratio gate is NOT here. It is applied in compute_tables, to the
    # region's neuron set, so that the descriptive time courses are built from the same
    # neurons as the tests -- which a cohort-layer gate could not do (`curves` are region
    # averages and carry no neuron id). Changing it therefore needs RECOMPUTE=True.

    # the firing-rate row's rate floor, raised here rather than at read time so it costs no
    # re-read. Never applied to 'legacy', which has to reproduce the published FR numbers.
    if cohort != 'legacy' and cfg.get('COHORT_MIN_FR_HZ') is not None:
        f = out['fr']
        n0 = len(f)
        out['fr'] = f[(f['fr_pre'] >= cfg['COHORT_MIN_FR_HZ'])
                      & (f['fr_post'] >= cfg['COHORT_MIN_FR_HZ'])]
        if verbose:
            print(f"  FR rate floor {cfg['COHORT_MIN_FR_HZ']} Hz: {n0} -> {len(out['fr'])} neurons")

    # NO SESSION INTERSECTION. An earlier version also cut every table down to the
    # sessions present in all three, on the reasoning that a row of the figure is only
    # comparable to the row above it if both describe the same sessions. It is dropped:
    # each metric already has its own reason for dropping a session (r_SC needs trials on
    # which every neuron in the region has data, FF needs a neuron to clear the count gate
    # in both windows), and throwing away a session from the firing-rate row because the
    # Fano-factor row could not use it costs n for no gain -- the tests are correlations
    # with lda_1 within a metric, not paired comparisons across metrics. With the current
    # files it was a no-op anyway: all three tables end on the same 206 sessions.

    kept_curves = []
    for k, (region_filter, gocue_filter) in rules.items():
        kept_curves.append(_curves(k, region_filter, gocue_filter))
    out['curves'] = pd.concat(kept_curves, ignore_index=True) if kept_curves else out['curves']

    for k in ('fr', 'ff', 'rsc'):
        out[k] = out[k].reset_index(drop=True)
    if verbose:
        print(f"cohort '{cohort}': "
              + ' | '.join(f"{k} {len(out[k])} rows / {out[k]['session'].nunique()} sessions"
                           for k in ('fr', 'ff', 'rsc')))
    return out


def bin_curves(curves, metric, n_bins=5, binning='quantile', return_edges=False,
               predictor='lda_1'):
    """Descriptive curves averaged the way the PSTH-by-LDA-bin panels do it: first average
    a session's regions into one curve per session, then stack the sessions of an LDA-1
    bin. Returns {bin: sessions x T array}, so the caller can take the mean and the SEM
    ACROSS SESSIONS -- the level the LDA-1 comparison lives at. Averaging regions and
    sessions in one step would instead weight a session by how many regions it contributed.

    `binning` is a DISPLAY choice and nothing more -- no test in this module bins anything,
    they all use lda_1 as a continuous predictor.

        'file'      the `binned1` cut carried in the LDA file: five EQUAL-WIDTH intervals
                    of 3.79 units. LDA 1 is unimodal and right-skewed, so these are wildly
                    unequal in n -- in the current cohort 86 sessions land in bin 1 and 5
                    in bin 4. All five curves are then drawn with equal visual weight,
                    which invites reading a gradient resting almost entirely on the middle.
        'quantile'  equal-COUNT bins cut on the sessions actually plotted (~37 each at
                    n_bins=5), so every curve is estimated from a comparable sample and the
                    SEM bands mean the same thing across bins. The cost: the bins are no
                    longer equally SPACED in LDA 1 -- the extremes are wide -- so the
                    legend must print each bin's RANGE, not just its n.

    With `return_edges`, returns (agg, edges) where edges[b] is that bin's (lo, hi) in
    LDA-1 units, for exactly that legend.
    """
    c = curves[curves['metric'] == metric]
    if not len(c):
        return ({}, {}) if return_edges else {}
    if predictor not in c.columns:
        raise KeyError(f"no {predictor} column -- run the tables through attach_lda first")
    if binning == 'file' and predictor != 'lda_1':
        # `bin` comes from the LDA file's binned1, which cuts COMPONENT 1. Reusing it for
        # another component would colour the curves by a variable they are not grouped on.
        raise ValueError(f"binning='file' only exists for lda_1; use 'quantile' for "
                         f"{predictor}")
    edges = {}
    if binning == 'quantile':
        per = c.drop_duplicates('session')[['session', predictor]].dropna()
        codes, cut = pd.qcut(per[predictor], n_bins, labels=False, retbins=True,
                             duplicates='drop')
        per = per.assign(_qbin=np.asarray(codes, dtype=int))
        c = (c.drop(columns=['bin'], errors='ignore')
              .merge(per[['session', '_qbin']], on='session', how='inner')
              .rename(columns={'_qbin': 'bin'}))
        edges = {b: (float(cut[b]), float(cut[b + 1])) for b in range(len(cut) - 1)}
    elif binning == 'file':
        for b, g in c.drop_duplicates('session').groupby('bin'):
            edges[int(b)] = (float(g[predictor].min()), float(g[predictor].max()))
    else:
        raise ValueError("binning must be 'file' or 'quantile'")
    with warnings.catch_warnings():
        # a session-region curve can be all-NaN (every bin below the mean-count floor);
        # nanmean warns on the empty slice and returns NaN, which is the right answer
        warnings.simplefilter('ignore', RuntimeWarning)
        per_sess = (c.groupby(['session', 'bin'])['curve']
                    .apply(lambda cs: np.nanmean(
                        np.vstack([np.asarray(v, float) for v in cs.values]), axis=0))
                    .reset_index())
    agg = {int(b): np.vstack(g['curve'].values) for b, g in per_sess.groupby('bin')}
    return (agg, edges) if return_edges else agg


# =============================================================================
# THE TEST -- unchanged from the notebooks
# =============================================================================

def _design_matrix(df, region_col='region', extra_covars=('n_trials',)):
    """Intercept + region dummies (drop first) + extra numeric covariates, as a plain array."""
    dummies = pd.get_dummies(df[region_col], drop_first=True)
    Z = pd.concat([pd.Series(1.0, index=df.index, name='Intercept'), dummies,
                   df[list(extra_covars)].astype(float)], axis=1)
    return Z.values.astype(float)


def age_effect_test(df, y_col, predictor='lda_1', region_col='region',
                    extra_covars=('n_trials',), n_perm=2000, seed=0, label=None):
    """The paper's age-effect test with `predictor` (here lda_1) in place of age_years:
    GLM slope as test statistic, session-stratified permutation p-value, and a
    partial-correlation Bayes factor in place of the R BayesFactor package.

    Moved verbatim from ff_psth_ldabin / fr_psth_ldabin / noise_corr_psth_lda, which each
    carried an identical copy. Do not "improve" it: its output is the published statistic.
    """
    import pingouin as pg
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[y_col, predictor, region_col] + list(extra_covars)).reset_index(drop=True)
    y = d[y_col].values.astype(float)
    x = d[predictor].values.astype(float)
    Z = _design_matrix(d, region_col, extra_covars)
    Zpinv = np.linalg.pinv(Z)

    # Frisch-Waugh-Lovell: residualize y and the predictor on all other covariates once; the
    # slope of y_resid on x_resid equals the predictor's coefficient in the full multiple GLM.
    y_resid = y - Z @ (Zpinv @ y)
    x_resid_obs = x - Z @ (Zpinv @ x)
    slope_obs = np.sum(x_resid_obs * y_resid) / np.sum(x_resid_obs ** 2)

    r_partial, _ = pearsonr(x_resid_obs, y_resid)
    n_eff = max(len(d) - (Z.shape[1] - 1), 3)   # drop df used by region dummies + extra covariates
    bf10 = pg.bayesfactor_pearson(r_partial, n_eff)

    # Permutation: shuffle the predictor across SESSIONS, keeping rows within a session
    # yoked, as in the paper's session-based label shuffling for neural metrics.
    sess_to_x = d.groupby('session')[predictor].first()
    sess_index = d['session'].values
    rng = np.random.default_rng(seed)
    perm_slopes = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = pd.Series(rng.permutation(sess_to_x.values), index=sess_to_x.index)
        x_perm = shuffled.reindex(sess_index).values.astype(float)
        x_perm_resid = x_perm - Z @ (Zpinv @ x_perm)
        perm_slopes[i] = np.sum(x_perm_resid * y_resid) / np.sum(x_perm_resid ** 2)
    p_perm = np.mean(np.abs(perm_slopes) >= np.abs(slope_obs))

    gdf = d.copy(); gdf['_y'] = y
    # built conditionally: with no extra covariates the old unconditional ' + ' left a
    # dangling operator and patsy refused the formula, so a covariate-free model -- the
    # obvious thing to want when checking whether a covariate matters -- could not be run
    formula = f"_y ~ {predictor} + C({region_col})"
    if len(extra_covars):
        formula += ' + ' + ' + '.join(extra_covars)
    glm_fit = smf.glm(formula, data=gdf, family=sm.families.Gaussian()).fit()

    return dict(label=label or y_col, n=len(d), n_sessions=d['session'].nunique(),
                slope=slope_obs, r_partial=r_partial, bf10=bf10, p_perm=p_perm,
                glm_slope=glm_fit.params[predictor], glm_p=glm_fit.pvalues[predictor])


def bf_label(bf):
    if bf > 10: return 'strong H1'
    if bf > 3: return 'moderate H1'
    if bf < 1 / 10: return 'strong H0'
    if bf < 1 / 3: return 'moderate H0'
    return 'inconclusive'


# =============================================================================
# THE GRID OF TESTS
# =============================================================================
# Rows of the summary figure, in order, and what each column tests. `None` is a cell with
# no test -- there is no onset metric for firing rate or noise correlation (firing rate's
# peak latency is a different quantity on a different neuron subset, so it does not belong
# in this column).
METRIC_ROWS = ['Firing rate', 'Fano factor', 'Noise correlation']
TEST_COLS = ['Baseline', 'Post-stimulus', 'Response magnitude', 'Response onset']
# The figure carries one more column than there are tests: the leftmost panel is the
# descriptive time course per LDA-1 bin, which is what the four tests to its right are
# reading windows off. It is not a test and takes no p-value.
FIG_COLS = ['PSTH'] + TEST_COLS

# (curve metric key, y label, reference line the curve should be read against, and which
# cfg windows to shade -- r_SC's windows differ from FF's and FR's, see default_config)
PSTH_ROWS = {
    'Firing rate': ('fr', 'Firing rate (Hz)', None, 'PRE_WINDOW', 'POST_WINDOW'),
    'Fano factor': ('ff', 'Fano factor', 1.0, 'PRE_WINDOW', 'POST_WINDOW'),
    # RSC_*_WINDOW now equals PRE_/POST_WINDOW, so all three rows shade the same stretch
    'Noise correlation': ('rsc', r'$r_{SC}$', 0.0, 'RSC_PRE_WINDOW', 'RSC_POST_WINDOW'),
}

# (table, y column, display column for the scatter, axis label, region covariate, extra covariates)
GRID = {
    ('Firing rate', 'Baseline'):
        ('fr', 'log_fr_pre', 'fr_pre', 'FR baseline (Hz)', 'region', ('n_trials',)),
    ('Firing rate', 'Post-stimulus'):
        ('fr', 'log_fr_post', 'fr_post', 'FR post-stimulus (Hz)', 'region', ('n_trials',)),
    ('Firing rate', 'Response magnitude'):
        ('fr', 'fr_evoked', 'fr_evoked', 'FR evoked (post - pre, Hz)', 'region', ('n_trials',)),
    ('Firing rate', 'Response onset'):
        ('fr', 'fr_onset_time', 'fr_onset_time', 'FR response onset (s)', 'region',
         ('n_trials',)),

    ('Fano factor', 'Baseline'):
        ('ff', 'log_ff_pre', 'ff_pre', 'FF baseline', 'region', ('n_trials',)),
    ('Fano factor', 'Post-stimulus'):
        ('ff', 'log_ff_post', 'ff_post', 'FF post-stimulus', 'region', ('n_trials',)),
    ('Fano factor', 'Response magnitude'):
        ('ff', 'ff_quench', 'ff_quench', 'FF quench (pre - post)', 'region', ('n_trials',)),
    # C(region), like every other test in the grid. The source notebook used
    # C(functional_region) -- 12 hand-built levels (Thalamus, Hippocampus, Visual Cortex,
    # ... with a Cosmos fallback for whatever fell through, and RSP lumped into "Visual"
    # by a substring match) -- on the reasoning that latency is organised by processing
    # stage rather than by fine parcellation. It made this the one cell of the figure
    # adjusting for anatomy at 12 levels while the other nine used 46. Switched for
    # consistency; it costs nothing (r -0.045 -> -0.039, p .0030 -> .0025). The
    # functional_region column is still computed, so this is reversible.
    ('Fano factor', 'Response onset'):
        ('ff', 'quench_onset_time', 'quench_onset_time', 'FF quench onset (s)',
         'region', ('n_trials',)),

    ('Noise correlation', 'Baseline'):
        ('rsc', 'r_sc_pre', 'r_sc_pre', r'$r_{SC}$ baseline', 'region', ('n_trials', 'n_neurons')),
    ('Noise correlation', 'Post-stimulus'):
        ('rsc', 'r_sc_post', 'r_sc_post', r'$r_{SC}$ post-stimulus', 'region',
         ('n_trials', 'n_neurons')),
    ('Noise correlation', 'Response magnitude'):
        ('rsc', 'r_sc_quench', 'r_sc_quench', r'$r_{SC}$ quench (pre - post)', 'region',
         ('n_trials', 'n_neurons')),
    ('Noise correlation', 'Response onset'): None,
}


def run_tests(tables, cfg=None, verbose=True, family=None, rsc_legacy=False,
              predictor='lda_1'):
    """Every test in GRID, then the multiple-comparisons correction.

    TWO FAMILIES, BOTH REPORTED. `q_fdr_metric` corrects within a metric (the four
    Fano-factor tests among themselves, and so on) and `q_fdr_all` corrects across all ten
    tests in the grid; `q_fdr` is whichever of the two cfg['FDR_FAMILY'] (or the `family`
    argument) selects, and is what the figure annotates. Which one is right depends on the
    claim being made -- "FF variability tracks LDA 1, and here is which window" is a
    within-metric family, "some neural measure tracks LDA 1" is the family of ten -- so the
    choice belongs to the analysis and both numbers stay visible.

    `rsc_legacy=True` tests r_SC on its ORIGINAL post-stimulus window (0.0-0.2 s) instead of
    the window now shared with FF and FR (0.1-0.3 s) -- only the reproduction check wants
    that.

    The correction is applied to p_perm (the session-level permutation p), NOT to the Bayes
    factor: BF10 is computed at the row n -- thousands of neurons -- while the permutation
    resamples ~200 session labels, which is why a BF of 1e7 can sit next to p = .02. The
    permutation p is the one of the two that respects the level the predictor varies at.
    """
    from statsmodels.stats.multitest import multipletests
    cfg = cfg or default_config()
    rows = []
    for metric in METRIC_ROWS:
        for test in TEST_COLS:
            spec = GRID[(metric, test)]
            if spec is None:
                continue
            tbl, y_col, disp, ylabel, region_col, covars = spec
            if rsc_legacy and tbl == 'rsc' and y_col in ('r_sc_post', 'r_sc_quench'):
                y_col = disp = y_col + '_legacy'
                ylabel = ylabel + ' (0-0.2 s)'
            df = tables[tbl]
            if y_col not in df.columns or df[y_col].notna().sum() == 0:
                print(f'  skipping {metric} / {test}: no {y_col}')
                continue
            if predictor not in df.columns:
                raise KeyError(f"table '{tbl}' has no {predictor} column -- pass the tables "
                               "through apply_cohort (or attach_lda) first")
            if verbose:
                print(f'  {metric:<18} {test:<14} {y_col} ~ {predictor} + C({region_col}) + '
                      + ' + '.join(covars))
            r = age_effect_test(df, y_col, predictor=predictor, region_col=region_col,
                                extra_covars=covars,
                                n_perm=cfg['N_PERM'], seed=cfg['SEED'])
            # `predictor` travels with the row so a panel cannot draw one component's points
            # under another component's r and p -- the figure reads it rather than assuming
            r.update(metric=metric, test=test, table=tbl, y_col=y_col, display_col=disp,
                     ylabel=ylabel, region_col=region_col, predictor=predictor,
                     covars=' + '.join(covars), unit='session-region' if tbl == 'rsc' else 'neuron')
            rows.append(r)

    res = pd.DataFrame(rows)

    # family of ten: the whole grid
    _, res['q_fdr_all'], _, _ = multipletests(res['p_perm'].values, alpha=cfg['ALPHA'],
                                              method=cfg['FDR_METHOD'])
    _, res['q_holm_all'], _, _ = multipletests(res['p_perm'].values, alpha=cfg['ALPHA'],
                                               method='holm')
    # family per metric: each row of the figure corrected among its own tests
    res['q_fdr_metric'] = np.nan
    res['q_holm_metric'] = np.nan
    for metric, g in res.groupby('metric'):
        _, q_m, _, _ = multipletests(g['p_perm'].values, alpha=cfg['ALPHA'],
                                     method=cfg['FDR_METHOD'])
        _, h_m, _, _ = multipletests(g['p_perm'].values, alpha=cfg['ALPHA'], method='holm')
        res.loc[g.index, 'q_fdr_metric'] = q_m
        res.loc[g.index, 'q_holm_metric'] = h_m

    fam = family or cfg['FDR_FAMILY']
    if fam not in ('metric', 'all'):
        raise ValueError("FDR_FAMILY must be 'metric' or 'all'")
    res['family'] = fam
    res['n_family'] = (res.groupby('metric')['metric'].transform('size') if fam == 'metric'
                       else len(res))
    res['q_fdr'] = res[f'q_fdr_{fam}']
    res['q_holm'] = res[f'q_holm_{fam}']
    res['fdr_sig'] = res['q_fdr'] < cfg['ALPHA']
    # THE LINE-DRAWING RULE for the figure: the uncorrected session-level permutation p,
    # as agreed. q_fdr and q_holm sit beside it in the annotation so the reader can see
    # what survives correction and what does not.
    res['significant'] = res['p_perm'] < cfg['ALPHA']
    res['evidence'] = res['bf10'].map(bf_label)
    return res


def results_table(res, p_floor=None):
    """The results as text, one row per test. `p_floor` (default 1/N_PERM) is the
    permutation's resolution: a p of exactly 0 means "no permutation reached the observed
    slope", which is a bound, not a zero, so it prints as '<floor'."""
    floor = p_floor or 1.0 / max(res['n'].size and 2000, 1)
    def _p(v):
        return f'<{floor:.4f}' if v <= 0 else f'{v:.4f}'
    fam = res['family'].iloc[0] if 'family' in res.columns else 'all'
    lines = [f"{'metric':<18}{'test':<20}{'unit':>14}{'n':>8}{'n_sess':>8}{'slope':>10}"
             f"{'r_part':>9}{'BF10':>10}{'evidence':>14}{'p_perm':>10}"
             f"{'q_metric':>10}{'q_all':>9}{'q_Holm':>9}{'':>6}",
             f"{'':<18}{'':<20}{'':>14}{'':>8}{'':>8}{'':>10}{'':>9}{'':>10}{'':>14}{'':>10}"
             + f"{'<-- reported' if fam == 'metric' else '':>10}"
             + f"{'<-- reported' if fam == 'all' else '':>9}"]
    for _, r in res.iterrows():
        lines.append(
            f"{r['metric']:<18}{r['test']:<20}{r['unit']:>14}{r['n']:>8}{r['n_sessions']:>8}"
            f"{r['slope']:>10.4f}{r['r_partial']:>9.4f}{r['bf10']:>10.3g}"
            f"{r['evidence']:>14}{_p(r['p_perm']):>10}"
            f"{r['q_fdr_metric']:>10.4f}{r['q_fdr_all']:>9.4f}{r['q_holm']:>9.4f}"
            f"{'  *' if r['fdr_sig'] else '   ':>6}")
    return '\n'.join(lines)


# =============================================================================
# CACHE
# =============================================================================

def cache_tag(cfg=None):
    """The cache name for one QC configuration.

    The presence-ratio gate and the r_SC completeness window both change what
    compute_tables PRODUCES, so their variants must not share a cache stem: load_tables
    otherwise picks the newest file by mtime, and a run with the gate on would silently
    read back a run with it off. Encoding the settings in the name makes the two coexist,
    which is what lets the no-filter and filtered versions be compared side by side.

        MIN_PRESENCE_RATIO=None, RSC_COMPLETE_WINDOW=(-.35,.65)  -> lda1_neural_nopr
        MIN_PRESENCE_RATIO=0.90                                  -> lda1_neural_pr90
        MIN_PRESENCE_RATIO=0.95, RSC_COMPLETE_WINDOW=None        -> lda1_neural_pr95_allbins
    """
    cfg = cfg or default_config()
    pr = cfg.get('MIN_PRESENCE_RATIO')
    stem = 'lda1_neural' + (f'_pr{int(round(pr * 100)):02d}' if pr is not None else '_nopr')
    if cfg.get('RSC_COMPLETE_WINDOW') is None:
        stem += '_allbins'
    if cfg.get('UNIFORM_FR_FLOOR', False):
        stem += f"_fr{str(cfg['MIN_FR_HZ']).replace('.', '')}"
    if cfg.get('FR_FLOOR_SOURCE') == 'session':
        stem += '_sess'
    elif cfg.get('FR_FLOOR_REQUIRE', 'both') != 'both':
        stem += f"_{cfg['FR_FLOOR_REQUIRE']}"
    if cfg.get('MIN_NEURONS_RSC'):
        stem += f"_rscn{cfg['MIN_NEURONS_RSC']}"
    return stem


def save_tables(tables, cfg=None, tag=None, dated=True):
    """Write the metric tables to parquet so the figure can be redrawn without re-reading
    ~400 pkl files. The name encodes the QC configuration -- see cache_tag."""
    CACHE.mkdir(parents=True, exist_ok=True)
    stem = tag or cache_tag(cfg)
    if dated:
        stem = f"{stem}_{_date.today().strftime(DATE_FMT)}"
    paths = []
    for k, v in tables.items():
        p = CACHE / f'{stem}__{k}.pqt'
        v.to_parquet(p, index=False)
        paths.append(p)
    print('wrote ' + ', '.join(p.name for p in paths))
    return paths


def load_unit_qc(tag=None, verbose=True):
    """The per-neuron spike-sorting QC table written by fetch_unit_qc.py, keyed by `nuid`.

    Newest `unit_qc_*.pqt` in the cache unless `tag` names one. Raises rather than returning
    empty: a silently absent QC table would mean a presence-ratio gate that quietly filters
    nothing, which is worse than a crash.
    """
    files = sorted(CACHE.glob(f"unit_qc_{tag or '*'}.pqt"))
    if not files:
        raise FileNotFoundError(
            f'no unit_qc_*.pqt in {CACHE} -- run `python fetch_unit_qc.py` first '
            '(it re-reads the cluster tables from ONE, a few minutes over ~400 probes)')
    qc = pd.read_parquet(files[-1])

    # COVERAGE GUARD. A partial QC table is the dangerous failure mode here: neurons with
    # no QC row are dropped, so a table covering a handful of probes would not error, it
    # would quietly delete almost every neuron in the study. Refuse anything that does not
    # cover nearly all of the firing-rate files it will be used against.
    n_probes_expected = len([f for f in os.listdir(Path(default_config()['FIRING_RATES_DIR']))
                             if f.startswith('firing_rate_')])
    if qc['pid'].nunique() < 0.95 * n_probes_expected:
        raise ValueError(
            f"{files[-1].name} covers only {qc['pid'].nunique()} of {n_probes_expected} "
            'probes -- it is a partial or still-running fetch. Wait for fetch_unit_qc.py '
            'to finish, or delete the file and re-run it.')

    if verbose:
        print(f"unit QC '{files[-1].stem}': {len(qc)} neurons over {qc['pid'].nunique()} "
              f"probes, presence_ratio median {qc['presence_ratio'].median():.3f}")
    return qc


def presence_ratio_map(cfg=None, tag=None, verbose=True):
    """{nuid: presence_ratio}, for the compute pass (which has no table to join onto)."""
    qc = load_unit_qc(tag, verbose)
    return dict(zip(qc['nuid'], qc['presence_ratio']))


def load_tables(tag=None, cfg=None, verbose=True):
    """Read back cached tables.

    With a `cfg`, only caches written under THAT QC configuration are eligible (newest
    first) -- so switching the presence-ratio gate on and re-running with RECOMPUTE=False
    raises rather than handing back the unfiltered tables. Without a cfg it falls back to
    the newest cache of any configuration, which is what a quick look wants.
    """
    stem = tag
    if stem is None and cfg is not None:
        want = cache_tag(cfg)
        # EXACT tag match, not a prefix one. `glob(f'{want}_*')` looked right and was
        # wrong: tags nest, so 'lda1_neural_nopr_fr10_*' also matches
        # 'lda1_neural_nopr_fr10_sess_<date>' -- a DIFFERENT configuration. Sorted newest
        # first, a session-floor run then silently served a window-floor request. Only the
        # dd-mm-yyyy stamp may follow the tag.
        _date_re = re.compile(r'^\d{2}-\d{2}-\d{4}$')
        cands = [p for p in CACHE.glob(f'{want}_*__ff.pqt')
                 if _date_re.match(p.name.split('__')[0][len(want) + 1:])]
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(
                f"no cached tables for this QC configuration ('{want}') in {CACHE}. "
                'Set RECOMPUTE = True to build it, or pass cfg=None to load whatever '
                'cache is newest regardless of how it was gated.')
        stem = cands[0].name.split('__')[0]
    if stem is None:
        cands = sorted(CACHE.glob('*__ff.pqt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f'no cached tables in {CACHE}')
        stem = cands[0].name.split('__')[0]
    out = {}
    for p in sorted(CACHE.glob(f'{stem}__*.pqt')):
        out[p.name.split('__')[1].replace('.pqt', '')] = pd.read_parquet(p)
    if verbose:
        print(f"loaded '{stem}': " + ', '.join(f'{k} ({len(v)})' for k, v in out.items()))
    return out


if __name__ == '__main__':
    import sys
    cfg = default_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tables = compute_tables(cfg, max_files=n)
    save_tables(tables, cfg, tag='lda1_neural_smoke' if n else None)
