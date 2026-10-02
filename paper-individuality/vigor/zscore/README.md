# How should the paw wavelets be subsampled and normalised?

```
audit_density_subsample.py   the ORIGINAL 3.2 -> 3.3 pipeline, audited (K = 10, 25 sessions)
run_variants.py              runs segmentation/3.3_wavelet_clusters_uniform.ipynb per (LOG, NORM)
  executed/                    -> the executed notebook of each variant, with its plots
compare_pipelines.py         the four variants vs production: how different, how much rig
  results_compare_pipelines.txt
  compare_pipelines_features.npz   -> its cache (delete to rebuild)
```

The pipeline copies themselves live next to the originals, which are untouched:
`segmentation/3.2_wavelet_subsample_uniform.ipynb` and
`segmentation/3.3_wavelet_clusters_uniform.ipynb`. They write to new folders only:
`data/paw_subsampled_wavelets_uniform/` and `data/paw_states_uniform_{raw|log}_{session|global}z/`.

## What was wrong with the original

Found by `audit_density_subsample.py`.

- **The density-weighted subsample.** 3.2 kept 2,000 of 20,000 frames with p ∝ the KDE density
  of a per-session t-SNE. Frames already occur in proportion to density, so this counts it
  twice. The training set's cluster make-up then differs from the session's by a total
  variation of 0.09, against 0.02 for a uniform draw. The stillest cluster comes out at ×0.89
  and the rarest vigorous one at ×0.71. Berman et al. 2014, the inspiration, did something else:
  they drew from each watershed region **in proportion to that region's probability mass**,
  which preserves each fly's composition. A uniform draw does the same in expectation.
- **Train/label mismatch.** Training frames were z-scored by the biased subsample's own
  stats; labelled frames by full-session stats. The same frames change label ~10% of the time.
- **Redundant z-scores.** The pooled z-score in 3.3 and `global_mean/global_std` at labelling
  are exact no-ops once each session is mean 0 / SD 1.
- **No log.** The amplitudes are positive and skewed (skew 2–4). k-means spends its clusters
  splitting the low-power bulk: two clusters hold ~60% of frames, the top one holds 0.7%.
- **Smaller bugs.** The elbow was fitted on PCs 0–1 only. `sessions_to_exclude` compared eids
  with `(mouse, eid)` tuples, so nothing was ever excluded. The example t-SNE skipped the
  session z-score.

## The variants

All use a uniform 2,000 frames per session, one standardisation with the same statistics for
training and labelling, PCA to 95%, and KMeans(8, seed 2024). States are renumbered stillest → most vigorous.

| `NORM` | what it removes |
|---|---|
| `session` | each session's own mean/SD: the overall amplitude level (camera distance, pixel scale, **and real vigor**) |
| `global` | one pooled mean/SD: only puts the features on a common scale. **Between-session level, rig included, stays in** |

`LOG`: `log(x + 0.1)`. The floor keeps the tracking jitter of a still paw (amplitudes around
0.01) from becoming the widest part of the space.

## Results (2026-09-30, all rebuilt from current data)

`results_compare_pipelines.txt`. LDA cohort: 260 sessions, 56 mice, 10 labs. Paw-only
features: 4 epochs × 10 bins, one-hot, 280 per session. ARI is computed over all 332 sessions.

| variant | ARI vs prod | state share | mouse | mouse\|lab | lab η² | lab LOSO | lab LOMO | ICC |
|---|---|---|---|---|---|---|---|---|
| production (density, session z) | 1 | 0.009–0.397 | 0.759 | 0.691 | 0.131 | 0.273 | 0.250 | 0.440 |
| density, session z (refit) | 0.97 | | 0.756 | 0.692 | 0.138 | 0.312 | 0.258 | 0.499 |
| density, no z | 0.59 | | 0.744 | 0.704 | 0.121 | 0.338 | 0.169 | 0.568 |
| uniform, raw, session z | 0.77 | 0.008–0.377 | 0.744 | 0.681 | 0.135 | 0.323 | 0.277 | 0.438 |
| uniform, raw, global z | 0.61 | 0.013–0.453 | 0.723 | 0.685 | 0.117 | 0.296 | 0.165 | 0.658 |
| uniform, log, session z | 0.28 | 0.088–0.171 | 0.650 | 0.547 | 0.139 | 0.342 | 0.358 | 0.432 |
| uniform, log, global z | 0.29 | 0.041–0.167 | 0.742 | 0.729 | 0.152 | 0.362 | 0.312 | 0.785 |

- **The uniform subsample + consistent z-score changes little** (ARI 0.77; every metric within
  about 2 points of production).
- **The log redraws the states** (ARI about 0.28) and balances them (every state 4–17%). Combined with
  per-session z it is the worst variant here: mouse ID drops 11 points and lab LOMO rises to 0.36.
- **Dropping the per-session z does not add lab, in both subsamples.** Lab LOMO falls from
  0.26 to 0.17 (density) and from 0.28 to 0.17 (uniform), with mouse ID unchanged. The between-mouse
  share (ICC) rises, but a lab label cannot tell mouse from rig-within-lab, so it needs the
  geometry control before it can be read as individuality.
- Lab LOMO rests on 56 mice. Treat differences under about 0.1 as unconfirmed (no CIs yet).

`rare_state_check.py`: a near-empty state that most sessions share (60–68% of sessions under
1%) is a property of the method, not of the session. Its occupancy is still somewhat
repeatable within a mouse (ICC about 0.34). Only raw + global z gives a rare state with lab structure
beyond the null (p = 0.047).

## Raw vs syllables (`raw_vs_syllables.py` → `results_raw_vs_syllables.txt`)

Same cohort and protocol. Every block is standardised across sessions before the LDA. "Raw,
time-resolved" is the syllables' own input with the clustering taken out: the same
4 epochs × 10 bins, averaged over trials, using paw wavelet log-amplitudes, whisker ME and lick count.

| block | dims | mouse | mouse\|lab | lab η² | lab LOMO | ICC |
|---|---|---|---|---|---|---|
| syllables, 360 (paw+whisk+lick) | 360 | 0.867 | 0.767 | 0.177 | **0.350** | 0.752 |
| raw, paw+whisk+lick, global | 880 | 0.949 | 0.869 | 0.189 | **0.642** | 0.776 |
| raw, paw+whisk+lick, session z | 880 | 0.922 | 0.851 | 0.190 | 0.581 | 0.462 |
| syllables, paw only (production) | 280 | 0.760 | 0.705 | 0.131 | **0.242** | 0.440 |
| raw, paw only, global | 800 | 0.900 | 0.832 | 0.166 | 0.565 | 0.775 |
| raw, paw only, session z | 800 | 0.872 | 0.806 | 0.179 | 0.550 | 0.459 |
| syllables, whisk+lick only | 80 | 0.699 | 0.642 | 0.337 | 0.427 | 0.781 |
| raw, whisk+lick, global / session z | 80 | 0.751 / 0.778 | 0.610 / 0.664 | 0.464 / 0.304 | 0.612 / 0.542 | |
| raw paw, whole-session scale | 40 | 0.696 | 0.609 | 0.137 | 0.146 | 0.675 |
| raw paw, between-feature correlations | 190 | 0.803 | 0.727 | 0.137 | 0.346 | 0.426 |

- **The clustering, not the z-score, is what removes the lab signal that generalises across
  mice.** Lab from a held-out mouse drops from 0.64 (raw, global) to 0.58 with the session z,
  then to 0.35 once segmented. For paw alone it drops from 0.57 to 0.55, then to 0.24.
- **The cost is mouse ID.** Syllables are 8–10 points below raw, before and after lab-centring.
- **Whisk + lick carry most of the syllables' lab signal** (lab η² 0.34, LOMO 0.43). The paw
  states are the lab-light channel.
- **Caveat: dimensions differ** (360 vs 880), and more dimensions help any decoder. A
  dimension-matched control (raw reduced to the syllables' dimensionality) is still to do.

## Wheel states (`wheel_states.py` → `results_wheel_states.txt`)

- **No z-score needed for the wheel level.** Session mean log amplitude has lab η² 0.142
  against a null of 0.107 (p = 0.21) and ICC by mouse 0.43. The exception is **8 Hz**, which has
  lab η² 0.36 (p = 0.0005). That is likely encoder/sampling differences, so drop it.
- **Paw and wheel do NOT decorrelate much.** Frame-level r(log paw power, log wheel power)
  has a median of 0.93. Session-level levels have r = 0.84 (0.78 across mice). State NMI with the paw
  states is 0.32–0.39.
- **How many states:** wheel-alone mouse ID plateaus at K = 4–5 (global: 0.70, 0.73).
- **Paw 280 + wheel (global, K=4–5):** mouse |lab rises from 0.69 to 0.75–0.76, but lab LOMO rises
  from 0.25 to 0.40–0.44.
- **360 + wheel: no gain in lab-free individuality** (mouse |lab 0.80 → 0.80–0.81) and lab
  LOMO rises from 0.40 to 0.52–0.58. Per-session-z wheel adds less lab, and no individuality either.
- The wheel's lab signal is camera-free, so it is lab behaviour or rig mechanics, not optics.
- Note: the 360 row reads mouse |lab 0.800 here and 0.767 in raw_vs_syllables. The
  difference is the column standardisation (not applied here), so compare rows within one table.

## Dimension-matched control: raw paw VELOCITY (`raw_velocity_vs_syllables.py`)

The paw wavelets (20 per bin) are replaced by the velocity they are computed from, using
`get_speed`'s definition, as speed per axis (4 per bin) or per paw (2 per bin):
`log(|v| + 1 px/s)`, in the same 4 × 10 bins.

| block | dims | mouse | mouse\|lab | lab η² | lab LOMO |
|---|---|---|---|---|---|
| syllables, 360 | 360 | 0.867 | 0.767 | 0.177 | **0.350** |
| raw \|v\| per axis + whisk + lick, global / session z | 240 | 0.914 / 0.904 | 0.849 / 0.841 | 0.311 / 0.251 | 0.619 / 0.608 |
| raw speed per paw + whisk + lick, global / session z | 160 | 0.890 / 0.894 | 0.805 / 0.822 | 0.351 / 0.266 | 0.627 / 0.577 |
| syllables, paw only | 280 | 0.760 | 0.705 | 0.131 | **0.242** |
| raw speed per paw only, global / session z | 80 | 0.760 / 0.783 | 0.695 / 0.679 | 0.258 / 0.228 | 0.454 / 0.446 |

**The lab reduction from segmenting is not a dimensionality artefact.** With FEWER dimensions
than the syllables, raw data carries about twice the lab signal that transfers to a held-out
mouse. Paw speed alone (80 dims) identifies mice exactly as well as the paw syllables, with
nearly double the lab signal: same individuality, half the lab. The per-session z again removes
little (at most 0.63 → 0.58).

## The chosen pipeline, and its states (2026-09-30)

The chosen pipeline: **uniform subsample + per-session z (fixed), no log, no global z, no wheel**.
It is `segmentation/3.3_wavelet_clusters_uniform.ipynb` with its defaults (`LOG = False`,
`NORM = 'session'`, `N_INIT = 10`), writing `data/paw_states_uniform_raw_sessionz/`.

**k-means needs `n_init = 10`** (`why_states_changed*.py`). scikit-learn ≥ 1.4 defaults to a
single initialisation, which lands in worse local optima. With seed 2024, one init gave
inertia 5.1167e6 and a symmetric moderate pair. With 10 inits, every seed converges to
5.1128e6, and the moderate pair is lateralised. The production recipe with 10 inits reproduces
production's states, so **production is the stable best solution for its data**, and the
laterality of 3/4 does not hang on a seed. **Every uniform-variant number above was computed
with one init, so the uniform rows of the pipeline table need rerunning. The density rows
(`rerun_no_zscore`'s fit) used 10 inits and stand.**

States are **numbered by vigor** (0 = stillest, 7 = most vigorous). They match the production states
one to one (81% of frames on the matched pairs):

| uniform | 0 still | 1 slow | 2 mod. LEFT | 3 mod. RIGHT | 4 mod. sym | 5 fast LEFT | 6 fast RIGHT | 7 very fast |
|---|---|---|---|---|---|---|---|---|
| production | 0 | 1 | 3 | 4 | 2 | 5 | 6 | 7 |

The differences in character: 7 is symmetric (+0.06 vs −0.38), and the moderate RIGHT state is the
more lateralised of the moderate pair. Palette: `paper_style.use_paw_states('uniform')`.

**Which fix moves the states more** (`normalisation_effect.py`, K = 8, 10 inits, same frames):

| change | ARI | frames changing state |
|---|---|---|
| normalisation fix only (subsample stats → full-session stats) | 0.773 | 10.2% |
| sampling fix only (density → uniform) | 0.823 | 16.1% |
| both (production → new) | 0.790 | 18.0% |

Neither changes the state STRUCTURE: all three solutions have the same lateralised pairs.
They move boundaries. The normalisation mismatch mostly shifts the big still/slow boundary
(hence the lower ARI, which the big states dominate). The sampling moves more frames, in the
smaller states.

## Syllables vs matched raw (per-session z), with 95% CIs (`ci_segmentation_vs_raw.py`)

Figures: `segmentation_vs_raw_figures.ipynb`. The CIs are over mice. Accuracies use 2,000 paired
bootstrap resamples. Lab η² uses a leave-one-mouse-out jackknife, because the mouse bootstrap
inflates η²: duplicated mice put its percentile interval entirely above the estimate.
Differences are syllables − raw.

| | mouse ID | mouse ID, lab-centred | lab ID, held-out mouse | lab η² |
|---|---|---|---|---|
| paw (280 vs 80) | −0.02 [−0.08, 0.03] | +0.03 [−0.03, 0.08] | **−0.20 [−0.30, −0.12]** | **−0.10 [−0.17, −0.03]** |
| whisk + lick (80 vs 80) | **−0.08 [−0.12, −0.03]** | −0.02 [−0.08, 0.03] | **−0.12 [−0.20, −0.04]** | +0.03 [−0.04, 0.11] |
| all (360 vs 160) | −0.03 [−0.07, 0.01] | **−0.06 [−0.10, −0.01]** | **−0.23 [−0.31, −0.14]** | **−0.09 [−0.13, −0.04]** |

### Updated 2026-10-02: today's syllables (`8_k_10_bin_syllables_02-10-2026`, uniform paw states)

`SYLLABLE_FILE` in `raw_vs_syllables.py` sets the syllables for all three scripts. The 19-08-2026
results are in `archive_syllables_19-08-2026/`. Same cohort (260 / 56 / 10); the raw blocks
are unchanged. Syllables − matched raw (per-session z), 95% CI:

| | mouse ID | mouse ID, lab-centred | lab ID, held-out mouse | lab η² |
|---|---|---|---|---|
| paw (280 vs 80) | −0.06 [−0.12, 0.00] | −0.04 [−0.10, 0.03] | **−0.23 [−0.33, −0.14]** | **−0.10 [−0.17, −0.03]** |
| whisk + lick (80 vs 80) | **−0.08 [−0.12, −0.03]** | −0.02 [−0.08, 0.04] | **−0.13 [−0.21, −0.05]** | +0.03 [−0.04, 0.11] |
| all (360 vs 160) | **−0.04 [−0.08, −0.01]** | **−0.07 [−0.11, −0.03]** | **−0.22 [−0.31, −0.13]** | **−0.09 [−0.14, −0.04]** |
