# HMM grid-search audit — findings

Investigation of `4.1_hmm_search.ipynb` / `4.2_hmm_best.ipynb` and the whisker
AR-HMM fits in `data/hmm/grid_search/5_prior_em_zsc_{True,False}`.
Started 2026-08-07; κ, lag and the replacement pipeline updated 2026-08-17.
All numbers below are measured, not estimated.

**Where this landed:** both hyperparameters are now resolved, in opposite ways. κ is
**fixed at 0** because the segmentation is provably invariant to it across the whole
defensible range (§7). The lag is **selected per session** by a paired sequential test on
a doubling grid capped at the signal's own decorrelation time (§2). Both live in
`4.0_hmm_dynamic.ipynb` + `hmm_dynamic_functions.py`, which replace 4.1 and 4.2.

---

## Bottom line, ranked by what to actually do

1. **`find_2_best_param` uses the wrong error bar.** It compares grid cells with each
   cell's own across-fold SD, but the 5 CV folds are the *same time blocks* for every
   (lag, κ) cell, so the comparisons are **paired**. Pairing removes **93%** of the
   noise. Consequence: the rule returns the grid *minimum* for ~98% of sessions
   (lag 1 in 312/319, κ=0 in 318/319) — it is not selecting, it is returning a constant.
2. **Drop κ from the grid — set it to 0.** Not because it is too small to matter as
   gridded, but because the segmentation is **invariant** to it over the entire range the
   data can support, and CV *rejects* everything above that range. See §7. Removing it is
   3× fewer fits.
3. **Two sessions produce degenerate segmentations** and one is unflagged:
   `7af49c00` (NYU-37) and `a8a8af78` (NYU-12) both flip state essentially every frame.
   Both are fixed by the lag correction. See §3.
4. **6 sessions need refitting** for a NaN-handling bug. See §5.
5. **But the lag fix barely changes results.** Median frame-wise agreement between the
   old-rule and corrected-rule segmentation across all 323 sessions is **0.980**; only
   8% fall below 0.95. This matters for *defensibility* of the model-selection claim,
   not because existing results are wrong. The dynamic pipeline reproduces this: median
   agreement old-vs-new over the first 6 refitted sessions is **0.979**.
6. **Never compare bits_LL across κ, or across model families.** The CV baseline is the
   *unfitted prior-sampled* model, and `initialize(method='prior')` samples from a prior
   that contains κ — so the baseline moves with κ and bits is not a common yardstick.
   This is what made κ=0 look like a winner when κ had no effect at all, and what made
   Poisson look like it beat Bernoulli 10/10 when it loses 9/10 (§7). Use **raw held-out
   LL** whenever the baseline is not shared.

---

## 1. The selection-rule bug

Paired vs unpaired test, "is lag L better than lag 1?" (n=292, marginalised over κ):

| lag vs 1 | current (unpaired) | paired t-test |
|---|---|---|
| 10 | 2/292 significant | **275/292** |
| 20 | 3/292 | **267/292** |
| 30 | 5/292 | **267/292** |

Across-fold SD of bits_LL is 0.125; SD of the *paired difference* (lag30−lag1) is 0.009.

Decisive single case — `7af49c00`: held-out **0.317 bits at lag 1 vs 0.477 at lag 10**.
The rule declared lag 1 "not significantly worse" when it is 50% worse. The problem is
the error bar, not a philosophical disagreement about parsimony.

Fix: `paired_selection.py` (in the parent folder), signature-compatible with
`find_best_param`. Use `rule='ttest'`.

```python
from paired_selection import find_best_param_paired
best_kappa, best_lag, mean_bits_LL = find_best_param_paired(bits_LL, params, param_num, rule='ttest')
```

**My own error, for the record:** my first version compared every cell against the
argmax cell. The argmax of 12 noisy cells is upward-biased by selection (winner's
curse), which made competitors look worse and dragged the choice toward the argmax's
lag. The shipped version tests *consecutive increments* instead.

## 2. The corrected rule does not converge on this grid

| | at lag ceiling, grid ≤30 | grid ≤60 |
|---|---|---|
| corrected rule, no floor | **55%** (162/295) | 29% (8/28) |
| + BIC-derived floor (N_eff=N) | 26% | 4% |

So the two rules fail in opposite directions: the current one pins to the floor (lag 1),
the corrected one to the ceiling. Neither is a good *per-session* selector.

At-ceiling sessions accept a median final step of **0.0026 bits** vs 0.0190 for the
rest — they're at the boundary because the paired test detects trivially small gains,
not because long lags help. Extending the fixed grid is not the fix either (a grid to 60
still leaves 29% at ceiling with no floor); the resolution is a **per-session cap**, below.

Whisker ME autocorrelation, for context: r = 0.71 at lag 10, 0.57 at 20, 0.45 at 30,
0.22 at 60. CV-LL keeps improving as the filter lengthens because the signal really is
autocorrelated out there — which is the *scientific* reason to cap the lag, not extend
it: **lag and κ are competing explanations for the same autocorrelation**, and every
frame of lag moves explanatory power out of the state and into the AR filter. ME dominant
frequency is 5 Hz (12-frame cycle), so lag 10–12 ≈ one cycle.

Two autocorrelation times are in play and they are not the same number: the **integral**
time (τ ≈ 43 frames cohort-wide) and the **1/e crossing** (median 32 per session), which
is what the cap uses. Beware the first implementation of this: `np.argmax(ac < cutoff)`
returns 0 when the ACF *never* crosses, which silently assigned τ = 1 to the **most**
autocorrelated sessions — exactly backwards. `decorrelation_time` now returns NaN in that
case and the caller falls back to a cap of 128.

Only 5 sessions have a final accepted gain > 0.02 bits and would genuinely benefit from
lags beyond 30: `a8a8af78` (+0.107), `0f25376f` (+0.037), `68775ca0` (+0.030),
`63f3dbc1` (+0.029), `49368f16` (+0.022). Two of those are the flicker sessions.

### RESOLVED (2026-08-17): cap the grid, do not floor the gain

The floor was dropped. What replaced it:

**The grid doubles and is capped at the signal's own decorrelation time.**
`1, 2, 4, 8, …` up to a per-session cap = the first ACF lag below **1/e**, snapped to the
nearest power of 2 in log2 space (`lag_grid_for_session`). Median cap over the cohort is
**32 frames** (533 ms). Sessions measured so far: τ = 28–74 frames → caps of 32 or 64.

**Why a cap rather than a floor.** At this data size *no likelihood criterion identifies
the AR order* — held-out CV, AIC and BIC all rise monotonically to whatever the largest
order tested is. With N ≈ 2×10⁵ frames and ~10² parameters, a `k·ln(N)` penalty is
~10⁻⁵ per frame while the LL differences are ~10⁻³, so the penalty is ~100× too small to
bind. Adding a floor on top of CV also double-counts complexity, which CV already
charges for. The ceiling therefore has to come from outside the likelihood, and the
principled outside argument is that **a filter longer than the signal's own decorrelation
time absorbs structure the latent state should be explaining** — lag and κ are competing
explanations for the same autocorrelation.

**The three floor routes below are kept as evidence, not as a recommendation.** They agree
on δ ≈ 0.004–0.018, but the calibration then showed everything in that range is 33 ms of
boundary jitter — so no threshold in it is scientifically meaningful, which is the reason
the floor was abandoned rather than tuned.

**Selection rule detail that matters.** Testing every cell against the argmax is wrong
(winner's curse) and testing consecutive increments is right — but the two genuinely
disagree here: on the doubling grid, argmax puts **95%** of sessions at the ceiling
against **44%** for the sequential rule, and they choose differently in **53%** of
sessions. The sequential rule compares each candidate against the *currently adopted*
lag, so skipping is allowed: `ZFM-01577` rejects 16, then adopts 32 (gain twice as large)
and then 64. `dynamic_fits_profiles.png` plots the tested quantity per session.

**Legacy floor routes (historical):**

| route | δ |
|---|---|
| empirical sweep (arbitrary — do not use as the justification) | 0.005–0.01 |
| parameter cost: `Δk·ln(N_eff)/(2·N_fold·ln2)`, Δk = n_states·ed·Δlag | 0.004–0.006 |
| calibrated against segmentation change (below) | 0.008 (1% of frames) / 0.018 (2%) |

**Calibration** (`delta_calibration.png`, 500 lag-pairs × 80 sessions): held-out bits
gained maps onto fraction of frames relabelled with **Spearman ρ = 0.92** — so bits_LL is
a faithful proxy for segmentation change. Median relabelling: 0.36% at Δbits 0.002,
1.06% at 0.010, 1.71% at 0.020, 4.4% at 0.100. At-ceiling rate: 14% (grid≤30) / 4%
(grid≤60) with δ=0.008; 7% / 4% with δ=0.018.

**BUT — 1% and 2% relabelling are both pure boundary jitter** (`relabelling_1pct_2pct.png`):

| | 1% case (PL033 `837b4e6a`, lag 10→30) | 2% case (UCLA033 `111c1762`, lag 1→30) |
|---|---|---|
| relabelled | 2,372 frames = **40 s** of 67 min | 5,421 frames = **90 s** of 75 min |
| disagreement runs | 790 | 1,886 |
| median run length | **2 frames = 33 ms** | **2 frames = 33 ms** |
| runs adjacent to a state boundary | **790/790 (100%)** | **1,886/1,886 (100%)** |
| high-whisking bouts | 3,060 → 3,117 | 2,916 → 3,174 |

Not one disagreement run falls in the interior of a state. In a 10 s window, 1%
relabelling is ~0.1 s of disagreement in ~2 spots of one or two frames each. **The same
bouts are found; their edges move by one or two frames.**

This undercuts the calibration as a *justification*: if every δ in the plausible range
only shifts bout edges by ~33 ms, the threshold isn't measuring anything scientific.
**So don't defend a number — report the sensitivity.** Show the lag distribution and the
downstream result for δ ∈ {0, parameter-cost floor, 2×}, and state that segmentation
differences are sub-50-ms boundary placement. That is a stronger claim than any
calibration, and it follows from the 0.980 cohort-wide agreement in §Bottom-line-5.

If you do want a scientifically meaningful calibration target, use bout *count*, dwell
distribution, or syllable identity rather than frame agreement — a 2%-relabelling change
corresponds to a ~9% change in bout count, which is a threshold worth arguing about.

Caveat: all of the above describes the ~98% of sessions that were never degenerate. For
`7af49c00` / `a8a8af78` the disagreement is 15–50% of frames and whole bouts do appear
and vanish (§3).

## 3. Three independent failure modes among "bad fits"

Screens are independent — low bits_LL does **not** predict a big segmentation change
(agreement ≈0.98 in every bits quartile). You need all three.

```python
bad_signal  = bits_best < 0.35      # reproduces 4 hand-picked + ee212778, 87ad026d
degenerate  = median_dwell <= 10    # a8a8af78, 7af49c00, 49368f16, 8c33abef, 7f6b86f9
fit_failed  = np.isnan(bits).any()  # 7f6b86f9: 2 of 5 folds failed in ALL 12 cells
at_ceiling  = adopted_lag == max(grid)   # log this every run
```

**Over-segmentation — fixed by the lag.** `a8a8af78` (dwell 1→29, bits 0.423→0.586),
`49368f16` (5→18, 0.677→0.903), and all six ZFM sessions (dwell 7–10 → 9–17, bits up).
Their fast switching was largely a lag artifact, not a cohort property.

**Under-segmentation — a `num_states` problem, not a lag problem.** `02fbb6da`
(dwell 123), `510b1a50` (79), `ee212778` (55), `0deb75fb` (50): long dwell, low bits,
and the lag changes nothing (123→122, 79→78). The `DY_010` figure shows the model
missing an obvious bout around 5–8 s. Two states can't describe these sessions.

**Genuine data problem.** `7f6b86f9` is the only session where a longer lag *hurts*
(bits 0.653→0.609). 2 of 5 folds fail in every cell. Keep excluded.

**Reconsider two exclusions:** `a8a8af78` and `8c33abef` are both in
`sessions_to_exclude`. `a8a8af78` is fully rescued by the lag fix. `8c33abef` has the
*highest* bits_LL of the whole flagged set (1.25) — fast-switching but well fit.

## 4. Sessions to reconsider / add

```python
# meet bits_best < 0.35 but not on the hand-curated list
'ee212778-3903-4f5b-ac4b-a72f22debf03',  # NYU-39, 0.336
'87ad026d-5b95-4022-8d59-c260870d830f',  # PL034,  0.345
# degenerate (dwell 1 frame, 74,908 segments) and NOT flagged anywhere
'7af49c00-63dd-4fed-b2e0-1b3bd945b20b',  # NYU-37 -- fixed by lag 30
```

## 5. NaN-handling rerun (independent of everything above)

`4.1` used to drop a timestep if **any** design-matrix column was NaN, then select the
fit variable — so whisker/lick fits were gated on right-paw tracking. Fixed in the
working tree (uncommitted as of this audit); `4.2` got the same fix in `2de6d18`.

Rerun these (2 of the original 8 are already in `sessions_to_exclude`):

```python
rerun = [
  "5569f363-0934-464e-9a5b-77c8e67791a1",  # NYU-30       52.4% of usable frames discarded
  "5ec72172-3901-4771-8777-6e9490ca51fc",  # NYU-30       44.3%
  "77e6dc6a-66ed-433c-b1a2-778c914f523c",  # NYU-30       29.1%
  "2e22c1fc-eec6-4856-85a0-7dba8668f646",  # DY_020       22.0%
  "81a78eac-9d36-4f90-a73a-7eb3ad7f770b",  # CSH_ZAD_026  20.6%
  "626126d5-eecf-4e9b-900e-ec29a17ece07",  # CSH_ZAD_026  16.7%
]   # already excluded: ebe2efe3 (30.6%), 8c33abef (21.5%)
```

**The design matrices themselves did not change.** 309/319 whisker and 317/341 lick
pickles are bit-identical to what the current design matrices produce under the old
rule; the April 2026 file mtimes were a Google-Drive sync (`b54e626`), not a
regeneration. 37 lick pickles have no design matrix on disk (orphans, all ZFM).

## 6. History of `4.1_hmm_search.ipynb`

`fig1_segmentation/3.2.1` (2025-09-09, empty) → `0_pre-processing/3.2.1` (2025-09-22,
first real content) → `segmentation/4.1` (2026-06-15, `181418e`). The **fitting math
never changed** — `cross_validate_armodel`, `cross_validate_poismodel`,
`compute_inputs`, `idxs_from_files` are byte-identical from 2025-09-22 to today. Every
committed change was configuration (var, paths, exclusion lists, `num_train_batches`
5↔10↔5). The only substantive change is the NaN fix in §5.

## 7. κ: set it to 0, and here is why that is a result rather than a default

### How κ acts

dynamax puts a Dirichlet prior on each transition-matrix row,
`A[k,·] ~ Dir(α·1 + κ·e_k)` with α=1 and κ = `transition_matrix_stickiness` added to the
**diagonal only**. The EM M-step is the MAP estimate `A[k,j] ∝ N[k,j] + α − 1`, so with
α=1 the −1 cancels and **κ is literally fake self-transitions added to the tally**:

```
A[k,k] = (N_kk + κ) / (N_kk + κ + N_kj)      E[dwell] = 1 + (N_kk + κ) / N_kj
```

### Why it scales with frame count

`N_kk` and `N_kj` are counts, so they grow linearly with N; κ is a fixed count, so its
relative weight shrinks as 1/N. Rearranged:

```
κ = N_exits · (D_target − D_current),    N_exits ∝ N
```

**κ is in units of frames-worth-of-evidence, not a dimensionless strength.** A κ tuned on
one dataset does not transfer to another of different size (keypoint-MoSeq's docs say the
same). Within this set, fold lengths run ~35k–80k frames, so one κ value is ~2× stronger
in the shortest sessions than the longest — "one κ for the dataset" only means something
if you scale κ ∝ N per session or pool the fit.

Measured here: a typical session has ~2,000–3,300 exits per state and 75k–140k
self-transitions on the training set, so **moving dwell by one frame costs κ ≈ 2,000–3,000**.
The old grid maxed at 50 — about 2% of one frame. That is why κ=0 won 318/319 times.

### Where κ actually enters (a point I first got wrong)

κ is **not** just an initialization detail. It enters in three places:

1. the **M-step, every EM iteration** — as the pseudo-counts above. This is the one that
   matters and it does not go away with any choice of `method`.
2. **initialization**, when `method='prior'`, because the sampled prior contains κ.
3. the **CV baseline**, for the same reason — which is the trap in §Bottom-line-6.

So `method='kmeans'` does *not* remove κ; it only removes route 2. Conversely, fixing
κ=0 does make routes 2 and 3 κ-independent, which is what leaves one fixed baseline
across the whole lag grid and makes bits comparable *along the lag axis*.

### The invariance argument — the main reason κ=0 (2026-08-17)

This is the argument to use, because it does not depend on a target duration.

**Over the range the data can support, κ changes nothing measurable.** At κ=1000 — the top
of the old grid — against κ=0 on 6 whisker sessions:

| | result |
|---|---|
| frame agreement with the κ=0 segmentation | **0.9996–0.9999** |
| median dwell | **identical** (250 / 433 / 267 / 367 / 400 / 383 ms, unchanged) |
| held-out raw LL, paired over 5 folds × 8 sessions | +0.00001 nats/frame, worse in 4/8 (n.s.) |

Worst agreement anywhere in the searched whisker grid is 0.9823, and 0.99935 median for
60 Hz lick. `kappa_insensitivity_60Hz.png` panel C shows *why*: κ only bites when
transitions are scarce, and these sessions have thousands.

**And above that range, CV actively rejects κ** (`kappa_high_test.csv`, 8 sessions × 5 folds,
paired on raw held-out LL):

| κ | mean Δ vs κ=0 (nats/frame) | worse in | sign test | paired t |
|---|---|---|---|---|
| 1,000 | +0.000010 | 4/8 | p = 1.00 | p = 0.68 |
| 10,000 | −0.000380 | 7/8 | p = 0.070 | p = 0.21 |
| 50,000 | −0.001200 | **8/8** | **p = 0.0078** | **p = 0.0015** |
| 200,000 | −0.003571 | **8/8** | **p = 0.0078** | p = 1.3×10⁻⁶ |
| 500,000 | −0.006445 | **8/8** | **p = 0.0078** | p = 1.1×10⁻⁷ |

So the answer to *"should I worry about κ ≥ 10⁴, and can I defend a narrow grid?"* is that
the grid does not need to be wide because the axis is **bounded on both sides by
measurement**: below ~10³ nothing changes, above ~5×10⁴ held-out likelihood gets
significantly worse in every session tested. The mechanism for the upper bound is the
pseudo-count scale — a session has ~2,000–3,300 exits per state, so κ ≈ 10³ is the most
the data will absorb before the prior starts overriding it.

Independent confirmation of the same bound from three directions:

- **units** — κ is frames-worth-of-evidence, and one frame of dwell costs κ ≈ 2,000–3,000
- **invariance** — nothing moves below that
- **CV** — everything above it is rejected

`lick_kappa_snippet.png` is the visual version: a lick session where a paired test calls
κ=100 *significantly* better (by +0.0000079 bits) is indistinguishable by eye, whereas
κ=5×10⁵ — 500× outside the grid — genuinely changes the labels.

### The MoSeq-style duration argument (supporting, not primary)

MoSeq does not select κ by likelihood (which cannot see κ). It runs a κ scan and picks
the κ whose median syllable duration matches the **model-free changepoint duration** of
the data (~400 ms target for rodents). Computing that anchor on whisker ME
(filtered-derivative changepoints, 60 sessions, σ=2 frames):

| | duration |
|---|---|
| model-free changepoints | **450 ms** (IQR 379–550) |
| current fits, median decoded dwell | **467 ms** |
| MoSeq rodent target | ~400 ms |

All three agree, so there is nothing for κ to buy. Solving `κ = N_exits·(D−D₀)` for a
450 ms target on typical sessions gives a **negative** number — which does not mean
"negative κ", it means the target is already exceeded and the constraint is one-sided, so
the answer is **κ = 0**. κ can only lengthen dwell; it has no way to shorten it.

The degenerate sessions would need κ ≈ 7×10⁵ (`7af49c00`, `a8a8af78`) or 2×10⁵
(`49368f16`) — a prior outweighing a large fraction of the data. **κ cannot rescue lag-1
flicker; the lag correction does** (dwell 1→29 frames on `a8a8af78`).

### Lick fits and κ (60 Hz and 30 Hz)

**60 Hz cohort — κ=0 is effectively unanimous.** Re-running the *paired* sequential rule
over all 356 lick pickles with the grid (0, 1, 5, 10, 50, 100, 500, 1000):

| chosen κ | sessions |
|---|---|
| **0** | **354** |
| 10 | 1 |
| 100 | 1 |

and the two exceptions pick values far below the ~10³ threshold where anything changes.

**30 Hz training cohort — the poor fits are not a κ problem.** They are (a) data quality:
20 of 75 sessions have isolated single-frame detections and lick rates spanning
0.2–4.7 Hz, and (b) EM instability, with several fits collapsing to a single state at
*some* κ (`kappa_insensitivity_all.png`, hollow markers). κ cannot repair either.

**Bernoulli vs Poisson at 30 Hz.** Since 2 licks can fall in one 30 Hz frame, Poisson is
not obviously wrong — but the counts are near-binary, so Bernoulli is better specified,
and on **raw** held-out LL it wins **9/10 sessions by +0.020 nats/frame (+0.029 bits)**.
The segmentations are nonetheless the same: median frame agreement **0.991**, min 0.930.
So this is a specification improvement with no practical consequence — not a reason to
refit. Note the trap: the `bits` column says Poisson wins 10/10, because the two families
have different baselines. That comparison is meaningless (§Bottom-line-6).

### Recommendation

- Set κ=0 and remove it from the grid. Justify it as *"the segmentation is invariant to κ
  over the entire range the data supports, and cross-validation rejects everything
  beyond it"* — not as "κ=0 won the CV", which is meaningless (§1), and not primarily on
  the duration match, which depends on a smoothing choice.
- Reuse the duration check as a **diagnostic**: a session whose median dwell falls far
  below ~450 ms is a fit failure to flag (§3), not something to patch with κ.
- Revisit stickiness only if you deliberately want syllables coarser than the data's
  natural timescale — and then use an HSMM with explicit durations, not a Dirichlet prior.

Caveats: the changepoint target is smoothing-dependent (450 / 1,250 / 2,104 ms at σ =
2 / 4 / 8 frames), so the filter has to be fixed and justified; whisker ME is 1-D whereas
MoSeq's changepoints come from multi-dimensional pose PCs; and match like with like —
MoSeq's ~400 ms is a *median*, whereas the transition-matrix mean dwell here is
560–1,600 ms because geometric dwell distributions are heavy-tailed.

---

## 8. The replacement pipeline (2026-08-17)

`4.1_hmm_search.ipynb` + `4.2_hmm_best.ipynb` are superseded by
**`4.0_hmm_dynamic.ipynb`** (a thin launcher) + **`hmm_dynamic_functions.py`** (the logic).
Everything that fits or cross-validates is still `cross_validate_armodel` /
`cross_validate_poismodel` from `segmentation_functions`, unchanged.

| | old | new |
|---|---|---|
| κ | searched, [0,5,50] or [0…1000] | **fixed at 0** (§7) |
| lag grid | fixed `[1,10,20,30]` | doubling, capped per session at the ACF 1/e crossing (§2) |
| selection | `find_2_best_param`, unpaired SD, on bits | sequential **paired** increments on **raw** held-out LL |
| decoding | separate notebook, reloads + re-initialises | same pass, reuses fitted params |
| saved | all grid cells + a 1.24 MB copy of the design matrix | all grid cells + a fingerprint (pickles 140–375 KB) |
| errors | bare `except:` printing a mouse name | recorded per session in the assessment CSV |

The pickle also carries the full per-fold LL table, the lag profile and every selection
step, so any alternative rule (a floor, a different α, argmax) can be re-derived without
refitting. 4.2's exact 3-tuple output format is written as well, so downstream notebooks
need no change.

**Measured behaviour**, first 6 sessions: **3.8 min at `n_jobs=6`** → ~3.5 h for the full
334; 6/6 `fit_ok`; selected lags 8, 8, 16, 32, 32, 64 with 3/6 at cap; median frame
agreement against the old fits **0.979**.

Two operational notes learned the hard way:

- **Memory, not CPU, is the constraint.** Peak RSS is ~3–4 GB *per worker* even on the
  smallest session (JAX compiles a separate executable per lag shape). On a 31 GB box
  `n_jobs=8` exhausts RAM and swap and thrashes; **`n_jobs=5–6`** is the working range.
- The `jax_plugins.xla_cuda12` / `CUDA_ERROR_NO_DEVICE` traceback at import is **harmless**
  — plugin discovery failing over to the CPU backend. `JAX_PLATFORM_NAME=cpu` skips it.

**`ibl_witten_26` is the instructive case** for why the rule is conservative: going 8→16
gives per-fold gains `[+0.0495, −0.0297, +0.0111, +0.0849, +0.1183]` — mean +0.047, but
one dissenting fold pushes the 95% CI to [−0.026, +0.120], so it stops at 8. With 5 folds
a single dissent is enough to block. That is the intended behaviour, but it means the
cohort's lag distribution is shifted low by fold noise, and n_folds is the knob that
would change it — not α.

## Files here

| file | what |
|---|---|
| `lag_*.png` (16) | whisker trace + state overlay, current vs corrected lag, per flagged session |
| `lag_and_kappa_NYU-37_7af49c00.png` | 3-panel: current lag / corrected lag / corrected + κ=20,000 |
| `dynamic_fits_snippets.png` | the new pipeline's fits: signal + states, one row per session |
| `dynamic_fits_profiles.png` | **per session: absolute LL profile (context) and the paired gain ± 95% CI the rule actually tests** |
| `kappa_insensitivity_60Hz.png` | κ vs duration / agreement / transition count — the invariance argument |
| `kappa_insensitivity_all.png` | as above, plus the 30 Hz training lick sessions (collapsed fits visible) |
| `kappa_snippet.png` / `kappa_scan.png` | whisker κ scan: what large κ does to a trace |
| `lick_kappa_snippet.png` | lick states at κ=0 vs κ=100 (invisible) and vs κ=5×10⁵ (real) |
| `kappa_high_test.csv` | 8 sessions × 5 folds × κ ∈ [0 … 5×10⁵], raw held-out LL — the rejection table |
| `kappa_insensitivity.csv` / `kappa_lick60.csv` | duration, agreement and occupancy vs κ |
| `bernoulli_vs_poisson.csv` | 30 Hz lick, both emission families, raw LL + agreement |
| `lick_training_quality.csv` | 75 training sessions: lick rate, ILI, bouts, dwell, collapse flag |
| `plot_dynamic_fits.py` | the two `dynamic_fits_*` figures (reads pickles, no refitting) |
| `plot_kappa_insensitivity.py` | the two `kappa_insensitivity_*` figures |
| `kappa_high_test.py` / `kappa_lick60.py` / `kappa_insensitivity.py` | the κ measurements |
| `bernoulli_vs_poisson.py` | the emission-family comparison |
| `plot_lick_kappa_snippet.py` | the lick κ snippet figure |
| `delta_calibration.png` | bits gained vs frames relabelled, 500 pairs — how to read a δ off the data |
| `relabelling_1pct_2pct.png` | **what 1% and 2% relabelling actually looks like** (33 ms of edge jitter) |
| `delta_calibration_pairs.csv` | the 500 lag-pairs: Δbits, relabelled fraction, dwell at both lags |
| `lag_selection_all_sessions.csv` | all 323 sessions, both rules, total bits, last accepted gain |
| `lag_selection_bic.csv` | + BIC-derived floors and τ per session |
| `lag_at_ceiling.csv` / `_delta01.csv` | sessions at the lag ceiling (170 with no floor / 39 with δ=0.01) |
| `plot_lag_panels.py` | the 16 per-session lag figures |
| `plot_states.py` | the 3-panel lag+κ figure |
| `calibrate_delta.py` | decodes every lag per session → `delta_calibration_pairs.csv` |
| `plot_relabel.py` | the 1%/2% relabelling figure |
| `decode_all.py` | all 323 sessions at both lags → `decode_all.csv` |
| `../paired_selection.py` | the corrected selection rule |

All scripts are self-contained: run from this directory with the env vars named at the
top of each (`OUTCSV`, `OUTDIR`, `OUTPNG`). They read the grid-search pickles in
`../../data/hmm/grid_search/5_prior_em_zsc_True` and the design matrices in
`../../data/design_matrices` — no refitting, so they take minutes not hours.

Left in the session scratchpad only (will be cleaned up):
`grid_search_5_prior_em_provenance.csv` (per-pickle provenance vs the current design
matrices, §5), `decode_all.csv`, `kappa_sweep.csv` (partial κ sweep, 86/96 fits,
abandoned). Regenerate `decode_all.csv` with `decode_all.py` if needed.

## Open / unfinished

**Closed since the first draft:**

- **δ (minimum-gain floor): dropped entirely**, replaced by the per-session grid cap (§2).
  Three routes agreed on 0.004–0.018, but everything in that range is 33 ms of boundary
  jitter, so no threshold in it is scientifically meaningful — and the one principled
  version (the BIC penalty) is ~100× too small to bind while also double-counting what CV
  already charges. If you ever do want a defended number, calibrate against bout *count*
  or dwell distribution, not frame agreement.
- **κ: closed at 0** by the invariance + CV-rejection argument (§7), not by the duration
  match. The κ=20,000 pilot noted below is consistent with this: it raises implied dwell
  49→124 frames at an LL cost, i.e. it *can* act, but only by overriding the data.
- **Lick selection audited.** Under the paired rule, **354/356** 60 Hz sessions choose
  κ=0 (§7). *The earlier claim here — "46% of lick sessions sit in the top two κ values
  with 99/356 at the κ ceiling of 1000" — was wrong*; it reflected the unpaired rule's
  output, not a real preference. The choice is essentially unanimously the grid floor.
- **Bernoulli vs Poisson for 30 Hz lick**: Bernoulli is better specified and wins 9/10 on
  raw LL, but the segmentations agree at 0.991 — no refit warranted (§7).

**Still open:**

- **`num_states` was never searched.** On this evidence it is now clearly the most
  interesting remaining axis — the likely explanation for the under-segmented low-bits
  sessions `02fbb6da` and `510b1a50` (§3), which neither lag nor κ fixes.
- **5-fold CV is conservative for lag selection.** One dissenting fold blocks an
  otherwise-large gain (`ibl_witten_26`, §8). More folds, not a looser α, is the fix if
  the cohort's lag distribution looks too low.
- **Whether the denser doubling grid changes the at-ceiling rate** is only measured on 6
  sessions so far (3/6 at cap, vs 44% for the old grid under the same rule).
- Optional: 5 sessions had final accepted gains > 0.02 bits under the old grid
  (`a8a8af78`, `0f25376f`, `68775ca0`, `63f3dbc1`, `49368f16`) and are the ones most
  likely to move; `63f3dbc1` is already the largest old-vs-new disagreement (agreement
  0.806, segments 11,461 → 7,529).
