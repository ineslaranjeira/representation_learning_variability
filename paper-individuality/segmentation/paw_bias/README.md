# Left/right forepaw movement bias across animals

Companion analysis to `3.3_wavelet_clusters.ipynb` (the notebook itself is **not modified** by
anything in this folder). 332 sessions, 101 mice, 11 labs, median 74 min/session; 76 mice have
≥2 sessions (307 sessions), 58 have ≥3.

```
qc_paw_identity.py         # which lightningPose label is which physical forepaw
extract_paw_bias.py        # per-session metrics  -> paw_bias_sessions.csv
analyse_paw_bias.py        # ICC / permutation / lab / frequency / task
confound_followup.py       # noise-corrected + gain-invariant re-analysis
test_downsampling.py       # RAW vs QUANT vs MATCH vs WHITEN  (section 4)
evaluate_downsampling.py   # scores those variants against C1-C5
within_camera_laterality.py# tau / near-far / camera-gain decomposition (section 5)
measure_camera_scale.py    # spatial scale ratio from rig landmarks (section 5)
gamma_spectrum.py          # per-frequency gamma -> shows it is temporal (section 5)
temporal_emulation_test.py # bounds how much of it the resampling stage can explain
eks_smoothing_test.py      # lightningPose smoother transfer function -> the dominant cause
plot_paw_bias.py           # figures/fig1..fig5
plot_downsampling.py       # figures/fig6
```

All laterality indices are `LI = (left - right) / (left + right)`, so **positive = left paw**.

---

## 0. Two things had to be settled first

**Paw identity — verified, not assumed.** Both side cameras label `paw_l` *and* `paw_r`, and the
design-matrix pipeline takes `paw_r` from each and calls them `l_paw` / `r_paw`. Across the 68
sessions with both cameras cached locally (10 labs), left-camera `paw_r` correlates with
right-camera **`paw_l`** (r = 0.64) far more than with right-camera `paw_r` (r = 0.34), in 64/68
sessions. The labels are therefore image-space with mirrored cameras, so `l_paw` and `r_paw` are
genuinely *different* paws. Comparing the scaled excursion of each physical paw between the two
views identifies which camera sees it near: `l_paw` is the near view in 62/68 sessions, likewise
`r_paw`. **Conclusion: `l_paw` = left forepaw, `r_paw` = right forepaw, both the well-resolved
near view. The pipeline's naming is correct.**

**A large right-paw amplitude artefact.** `get_speed` divides left-camera pixels by 2 (1280×1024
vs 640×512). That correctly equates spatial scale — but it also halves the left camera's *tracking
noise* while leaving the right camera's at full size:

| | left paw | right paw |
|---|---|---|
| high-frequency jitter residual | 1.67 px | 3.16 px (**1.95×**) |
| 32 Hz wavelet power (≈ pure noise; a forepaw cannot oscillate at 32 Hz) | — | LI = **−0.48** |
| per-paw SNR, P(0.5–8 Hz) / P(32 Hz) | 133 | 51 (**2.6× worse**) |

Raw `LI[0.5–8 Hz power]` correlates **r = +0.80** with `LI[32 Hz noise]`, and raw frame-difference
speed makes the right paw look ~2× faster in nearly every session. The spectral signature confirms
it: the L/R difference grows monotonically with frequency and flips sign at ~4 Hz
(`fig4`, right panel) — exactly what a flat additive noise floor on the right paw produces, since
noise dominates where real power is lowest.

> **Therefore no population-level left/right paw preference can be claimed from this dataset.**
> The group means in `fig2` are instrumental. Only *individual differences* around them are
> interpretable. Sections 4–5 pin down *which* artefact is responsible — it turns out to be a
> multiplicative camera-gain error, **not** this noise floor, which affects only speed-type
> measures. Read section 4 before acting on anything in section 0.

---

## 1. Paw bias is a strong and stable individual trait

ICC(1), sessions nested in mice (76 mice, 307 sessions). Permutation null shuffles session→mouse;
its 95th percentile is ≈ 0.08 for every metric, so all of these are far outside chance
(p_perm = 1e-4).

| metric | ICC | contamination (r with 32 Hz noise LI) |
|---|---|---|
| LI amplitude, 0.5–8 Hz power | **0.815** | +0.80 |
| LI SNR (gain-invariant — but see §4, over-corrects) | *0.845* | −0.14 |
| bilateral coupling (peak xcorr) | **0.790** | −0.19 |
| LI bout rate at matched duty cycle | 0.512 | −0.23 |
| lead–lag (which paw leads) | 0.453 | −0.21 |
| *LI jitter (the artefact itself)* | *0.824* | *+0.92* |

Supporting evidence:

- **Across-session reliability r = 0.867**, against a within-session split-half ceiling of
  **r = 0.889** — i.e. a mouse's paw bias is reproduced between sessions about as well as between
  two halves of the same session. It is essentially noise-free at the session level.
- **55% of the 58 mice with ≥3 sessions have *every* session on the same side of the population
  median** (chance 13%, p_perm = 1e-4).
- **Not a lab effect.** Between-lab differences are significant by session but vanish once each
  mouse counts once (F = 0.73, p = 0.69). Centring each metric within its lab leaves
  ICC = 0.789.
- The scale-free metrics (bout rate, lead–lag) are only weakly noise-contaminated and remain
  highly significant, so the individuality is not purely instrumental.

**Is this individuality or stable per-mouse tracking geometry?** `LI[jitter]` is itself
mouse-stable (ICC = 0.824), and each mouse is typically recorded on one rig with fixed camera
geometry, so the concern is real. Sections 4–5 settle most of it:

- equalising the noise floors leaves the ICC at 0.827 (from 0.829), so the noise asymmetry is not
  what makes the index mouse-stable;
- the pipeline's index tracks the camera-free laterality estimate at **r = 0.931**, and the camera
  gain term γ is uncorrelated with true laterality (r = 0.035, p = 0.77), so γ acts as an additive
  offset rather than a source of individual variation.

What remains is a residual: γ varies between labs and contributes at most ~13% of the variance in
the index (var γ = 0.092² against var index ≈ 0.25²). Fully separating rig geometry from behaviour
would still need the same mouse recorded on two rigs.

---

## 2. Paw bias predicts wheel-turn direction and choice side

Mouse-level correlations (n = 101 mice), permutation p, plus the partial correlation controlling
for `LI[32 Hz noise]`:

| | r | ρ | p_perm | partial on noise |
|---|---|---|---|---|
| LI amplitude vs wheel-direction bias | **+0.407** | +0.483 | **0.0002** | r = +0.379 |
| LI **SNR** (gain-invariant) vs wheel-direction bias | **+0.315** | +0.415 | **0.0016** | — |
| LI amplitude vs P(right choice), all trials | +0.226 | +0.306 | 0.024 | r = +0.254, p = 0.010 |
| LI SNR vs P(right choice), all trials | +0.235 | +0.261 | 0.018 | — |
| LI amplitude vs P(right choice), 0% contrast | +0.189 | +0.259 | 0.058 | r = +0.217, p = 0.029 |
| LI amplitude vs performance | +0.010 | +0.008 | 0.92 | r = +0.011 |

**Direction verified empirically**, not assumed: `choice == -1` is the right-side report (94.6% of
high-contrast right-stimulus trials, matching 94.6% accuracy), and positive wheel velocity is what
drives it (mean +1.1 vs −1.2 in the 300 ms after first movement). So the finding reads coherently
in one direction:

> **Mice that use their left paw relatively more also turn the wheel more in the report-right
> direction, and make more right-side choices — including on 0% contrast trials, where there is no
> correct answer.** Paw-usage bias and choice-side bias are two faces of one individual asymmetry.

Nothing predicts task performance. Note that `LI[noise]` is itself weakly related to wheel bias
(r = +0.23, p = 0.021), which is why the partial correlation matters — it survives. The strongest
control is in section 4: explicitly equalising the noise floors leaves the wheel-bias correlation
completely intact (r = +0.432 → +0.432, p = 0.0003).

⚠️ The `LI[SNR]` rows above should be **discounted**. Section 4 shows that dividing by each paw's
own 32 Hz power over-corrects — it injects the inverse of the noise asymmetry — and it destroys the
wheel-bias correlation (r = +0.148, p = 0.26). Use the plain band-power index with the gain
correction from section 5 instead.

---

## 3. Bilateral coordination is its own individual trait

The peak cross-correlation of the two paw speed traces is 0.49 ± 0.10 across mice, with
**ICC = 0.790** — how tightly a mouse couples its two forepaws is a strong, stable individual
property. It is unrelated to performance (r = −0.035). Mean lead–lag is essentially zero
(−4 ms, p = 0.23), but it is mouse-specific (ICC = 0.453; 55% of mice sign-consistent vs 24.5%
chance), i.e. *which* paw leads is idiosyncratic rather than shared.

---

## 4. Tested: does degrading the left camera fix it?  (`test_downsampling.py`)

Two surrogates for "downsample the left camera to 640×512" — real downsampling would mean
re-encoding the video and re-running lightningPose, which cannot be done post-hoc — plus the
whitening alternative, on 213 sessions from 58 mice with ≥3 sessions each:

- **QUANT** snap left-paw coordinates onto the right camera's pixel grid (the *floor* of the
  downsampling effect: sub-pixel precision only).
- **MATCH** add Gaussian positional noise to the left paw, calibrated per session so its 32 Hz
  power equals the right paw's (emulates the *full* effect; the calibration is verified, not assumed).
- **WHITEN** divide each paw's 0.5–8 Hz bands by its own 32 Hz band.

| Criterion | RAW | QUANT | MATCH | WHITEN |
|---|---|---|---|---|
| **C1** LI[32 Hz] → 0 | −0.403 | −0.125 | **+0.014** ✅ | 0 by construction |
| **C2** r(LI[band power], noise) → 0 | +0.771 | +0.771 | +0.767 ❌ | −0.125 |
| **C3** per-frequency slope | −0.030/oct | −0.027 | −0.025 | — |
| **C4** between-mouse ICC | 0.829 | 0.829 | **0.827** | 0.791 |
| **C5** LI × wheel bias | r=+0.432, p=0.0007 | +0.434 | **+0.432, p=0.0003** | +0.148, p=0.26 |

**MATCH does equalise the noise floors, and it changes the metric the clustering uses by nothing at
all:** `LI[0.5–8 Hz power]` correlates **r = 0.9999** between RAW and MATCH (mean −0.078 → −0.070).
By contrast the *speed* index does move, and substantially: r = 0.79, with its mean flipping sign
from −0.107 to +0.048.

There is a clean analytic reason. The wavelets run on *velocity*, i.e. differenced position, and
differencing shapes white tracking noise into a high-pass spectrum, |H(f)|² ∝ sin²(πf/fs):

| band | 0.5 Hz | 1 Hz | 2 Hz | 4 Hz | 8 Hz | 32 Hz |
|---|---|---|---|---|---|---|
| noise power, relative to 32 Hz | 0.07% | 0.28% | 1.1% | 4.4% | 16.7% | 100% |

The 32 Hz channel is ~90× more noise-sensitive than the 2 Hz channel. The noise floor dominates
exactly where it was *detected* and is nearly absent where the clustering actually lives.

> **This corrects an earlier conclusion in this analysis.** The r = +0.80 correlation between the
> raw amplitude index and the 32 Hz noise index was read as causal contamination of the 0.5–8 Hz
> band. It is not: equalising the noise leaves that correlation at +0.767 and the metric at
> r = 0.9999 of itself. The correlation reflects a shared cause (per-mouse/per-rig camera geometry
> driving both), not contamination. **Downsampling the left camera is the wrong fix, and whitening
> is actively harmful** — it swings the index from −0.14 to +0.26 and destroys the wheel-bias
> correlation (p = 0.26), because the divisor is itself asymmetric.

## 5. What the population-level bias actually is: a frame-rate artefact

Each side camera tracks *both* paws, which allows laterality to be measured **within a single
camera**, never comparing across the two. With paw identity from section 0:

```
LI_leftcam  = LI( leftcam paw_r  [LEFT paw, near],  leftcam paw_l  [RIGHT paw, far ] )
LI_rightcam = LI( rightcam paw_l [LEFT paw, far ],  rightcam paw_r [RIGHT paw, near] )
```

Writing the four log band-powers in terms of true laterality **τ**, near/far foreshortening **f**
and camera gain **γ** gives an exactly-determined system (4 log-amplitudes → 3 independent
contrasts), solved on the 68 two-camera sessions (41 mice, 10 labs):

| term | estimate | evidence |
|---|---|---|
| **τ** true laterality | **+0.027, p = 0.36** | 38/68 positive — *no population paw preference* |
| **f** near/far foreshortening | +0.139 | 66/68 positive, p = 2e−25 |
| **γ** camera gain | **+0.089** | 57/68 positive, p = 2e−11 |
| pipeline index (τ − γ) | −0.062 | p = 0.047 |

The direct evidence is a double dissociation: the *same two physical paws* measured by the
**swapped** cameras give LI = **+0.116** where the pipeline's pairing gives **−0.062**. Swap which
camera measures which paw and the sign flips — so the population-level bias is about the cameras,
not the paws.

γ = +0.089 corresponds to the left camera reading ~16% low overall
(exp(−2γ) = 0.837). **But γ is not a spatial scale error, and the fix is not a constant.**

### The spatial correction is already right  (`measure_camera_scale.py`)

The ÷2 assumes the left camera has exactly twice the linear resolution. That is measurable directly,
from landmarks both cameras see — including the **lick tube, which is rig hardware of fixed physical
size**, so it needs no assumption about the mouse:

| reference | measured left/right ratio | vs 2.0 | vs 1.675 |
|---|---|---|---|
| lick tube length (rigid rig hardware) | **1.980** | p = 0.25 ✓ | p ≈ 0 ✗ |
| pupil diameter (bilaterally symmetric) | **1.986** | p = 0.90 ✓ | p ≈ 0 ✗ |
| nose tip → tube top | 1.954 | p = 6e−4 | p ≈ 0 ✗ |

**The rig is spatially symmetric and the ÷2 is correct.** No constant rescaling will fix γ.

### γ is temporal: it comes from the frame-rate asymmetry  (`gamma_spectrum.py`)

A spatial gain error is flat across frequency. γ is not — it grows steeply:

| frequency | 0.5 Hz | 1 Hz | 2 Hz | 4 Hz | 8 Hz |
|---|---|---|---|---|---|
| γ | +0.036 | +0.037 | +0.045 | +0.076 | **+0.168** |
| left/right amplitude ratio | 0.93 | 0.93 | 0.92 | 0.86 | **0.72** |

(paired 8 Hz vs 0.5 Hz: t = +10.0, p = 5e−15)

The cause is the **frame rate**, not the side. The left camera usually runs at 60 fps and the right at
150 fps, and the slower camera under-measures fast movement for two reasons: a longer exposure
(up to ~16.7 ms vs ≤6.7 ms) blurs a moving paw toward the middle of its excursion, and coarser
temporal sampling plus linear interpolation onto the 60 Hz analysis grid — a triangular kernel one
full sample wide — attenuates further. The current pipeline compounds this by low-pass filtering
*only* the faster camera, which preserves its amplitude while the slower one goes unfiltered.

The decisive evidence is that **the bias reverses in the rigs where the left camera is the fast one**:

| rig configuration | n | γ @ 0.5 Hz | γ @ 8 Hz |
|---|---|---|---|
| left 60 Hz, right 150 Hz | 61 | +0.037 | **+0.192** |
| left 150 Hz, right 60 Hz | 5 | −0.006 | **−0.047** |

γ @ 8 Hz correlates with log₂(right_fr / left_fr) at r = +0.56 (p = 7e−7); γ @ 0.5 Hz does not
(r = 0.10, p = 0.43). And τ stays non-significant at *every* frequency (+0.009 to +0.033, all
p > 0.3) — there is no population paw preference at any timescale.

> **Correction.** An earlier version of this file recommended setting
> `RESOLUTION['left'] ≈ 2.39`. That was wrong twice over: the value was inverted (it would have
> doubled the bias rather than removing it), and the whole approach was aimed at the wrong dimension
> — the problem is temporal, so no constant spatial factor can fix it.

### Where the attenuation originates — and why the resampling code cannot fix it
(`temporal_emulation_test.py`)

Two mechanisms are quantifiable directly. The 30 Hz anti-alias filter is **not** one of them: at
150 Hz sampling a 4th-order 30 Hz Butterworth has gain 0.999996 at 8 Hz and 1.000000 below 4 Hz, so
it is transparent in the clustering band. It is also correct — resampling 150 → 60 Hz *requires*
anti-aliasing below the 30 Hz Nyquist, and the filter is applied on `if fr > 60`, i.e. to whichever
camera is fast, not to a fixed side.

Processing genuinely-150 Hz traces *as if* they had come from a 60 fps camera (box-average over
1/60 s to model the longer exposure, decimate, then the same route) bounds what exposure and
sampling can account for:

| frequency | emulated penalty | observed penalty | gap explained |
|---|---|---|---|
| 0.5 Hz | 1.002 | 0.929 | **0%** |
| 1 Hz | 1.000 | 0.926 | 0% |
| 2 Hz | 0.994 | 0.906 | 6% |
| 4 Hz | 0.975 | 0.841 | 16% |
| 8 Hz | 0.901 | 0.681 | **31%** |

So exposure and sampling explain about a third of the effect at 8 Hz and **none** of the ~7%
frequency-flat component. The remainder arises **before the design matrix**, in lightningPose
itself — plausibly its ensemble-Kalman post-processing, which would smooth over a longer real-time
window at a lower frame rate, though that has not been verified here.

> **Two further corrections.** Earlier versions of this file recommended (a) not low-pass filtering
> only one camera and (b) box-averaging the fast camera to match the slow one. Both are withdrawn:
> (a) the filter is transparent in the band and is correct anti-aliasing applied by frame rate, not
> by side; (b) it recovers only ~1/3 of the penalty at 8 Hz and none at low frequency.

### The mechanism, measured  (`eks_smoothing_test.py`)

The pose files carry two versions of every keypoint: `<kp>_ens_median`, the median across the
ensemble of networks with no temporal smoothing, and `<kp>_x`, the output of lightningPose's
ensemble Kalman smoother. **The ratio of their velocity spectra is the smoother's transfer
function**, measurable at each camera's native rate, entirely independent of the design matrix.

| frequency | 60 Hz camera | 150 Hz camera | ratio 60/150 | observed L/R | explained |
|---|---|---|---|---|---|
| 0.5 Hz | 0.993 | 0.984 | 1.009 | 0.929 | −13% |
| 2 Hz | 0.961 | 0.979 | 0.982 | 0.906 | 19% |
| 4 Hz | 0.889 | 0.973 | 0.914 | 0.841 | 54% |
| 8 Hz | **0.718** | **0.956** | **0.750** | 0.681 | **78%** |

The smoother removes 28% of real 8 Hz movement from a 60 fps recording and only 4% from a 150 fps
one — consistent with its time constant being set per *frame* rather than per second, so that at
60 fps it smooths over a 2.5× longer real-time window. In the 5 reversed rigs the roles swap, as
they must if this is causal: the left/right ratio becomes 1.025 instead of 0.750.

**Stacking the two independently measured stages accounts for the frequency-dependent part almost
exactly:**

| frequency | LP smoother | resampling + exposure | predicted | observed | residual |
|---|---|---|---|---|---|
| 0.5 Hz | 1.009 | 1.002 | 1.011 | 0.929 | 0.919 |
| 1 Hz | 1.003 | 1.000 | 1.003 | 0.926 | 0.923 |
| 2 Hz | 0.982 | 0.994 | 0.976 | 0.906 | 0.928 |
| 4 Hz | 0.914 | 0.975 | 0.891 | 0.841 | 0.944 |
| 8 Hz | 0.750 | 0.901 | **0.676** | **0.681** | 1.008 |

So γ has exactly **two** components:

1. **A frequency-dependent part — solved.** lightningPose's smoother (dominant, 78% at 8 Hz) plus
   exposure and resampling (secondary, 31%). Fully accounted for at 8 Hz.
2. **A frequency-flat ~7% deficit — still open.** The residual is 0.929 ± 0.009 across 0.5–4 Hz,
   remarkably constant. It is not the smoother, not resampling, not exposure, not the anti-alias
   filter, and not tracking noise — all of those are measured and excluded. Remaining candidates,
   in order of plausibility:
   - **depth perspective**: the scale references (tube, pupil, nose) all sit at the midline or head,
     while the near paw is much closer to the camera. Slightly different working distances would
     make px/mm differ at the paw's depth even where it matches at the head — frequency-flat by
     construction, and invisible to the landmark test.
   - **residual scale error**: measured ratios were 1.980 / 1.986 / 1.954, not exactly 2.000; a true
     value near 1.96 is worth γ ≈ +0.01, about a quarter of the 0.036.
   - **estimator leakage**: the decomposition assumes near/far foreshortening is identical in both
     cameras. If it is not, part of this residual is really asymmetric f — which also means τ carries
     a small bias, so "no population preference" should read as "none detectable above this
     residual", not an exact zero.

Note also that γ's session-to-session sd (0.092) *exceeds* its mean (0.089), and only 22% of its
variance is between-lab. Most of γ is session-level, which points at how the mouse sits on a given
day rather than at fixed hardware.

### What to do instead

1. **Build the design matrix from the `_ens_median` columns rather than the smoothed ones.** The
   smoother is the single largest contributor to the bias, and it removes nothing you need: the
   noise it suppresses is high-frequency, and the MATCH test already established that the 32 Hz
   noise floor does not affect the 0.5–8 Hz clustering band (r = 0.9999). This should remove ~78%
   of the 8 Hz bias at no real cost, and it is directly testable by re-running
   `gamma_spectrum.py` on the `_ens_median` traces.
2. **For any laterality claim — use the within-camera estimator.** Both paws from a single camera
   cancels γ *exactly, whatever the residual cause*, with no correction constant.
3. **Cheapest — keep features at ≤2 Hz**, where the total penalty is ~7% rather than 32%, at the
   cost of the fast bands the lateralised states rely on.
4. **Do not change the resampling or filtering code.** It contributes little and the filter is
   correct anti-aliasing.

### What this does *not* change

Everything in sections 1–3 survives untouched, because γ is uncorrelated with τ (r = 0.035,
p = 0.77) and so acts as an additive offset rather than a driver of individual variation:

- between-mouse ICC: 0.829 → 0.827 under MATCH; the pipeline index tracks true laterality at
  **r = 0.931**;
- the wheel-bias correlation: r = +0.432 → +0.432 (p = 0.0003).

So **the individuality and the choice-side link are real; only the group mean was an artefact.**

## Notes / limitations

- `LI[bout duration]` is algebraically `−LI[bout rate]` at matched duty cycle; only one is
  informative.
- Lead–lag and bout statistics are computed on NaN-dropped traces, so bins adjacent to camera
  dropouts contribute small edge artefacts (NaN fraction is typically 0–4%).
- Tracking NaNs are dropped, never interpolated.
