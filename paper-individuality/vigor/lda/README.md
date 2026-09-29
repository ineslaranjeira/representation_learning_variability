# Does LD1 track wheel vigor, or paw vigor?

```
states_vigor.py             per-session wheel AND paw vigor from data/states_files
  vigor_sessions_states.csv   -> its cache (332 sessions, 101 mice)
make_embedding.py           regenerates the shrinkage LDA embedding locally
vigor_vs_lda.py             the tests            -> results_main.txt
controls.py                 embedding / circularity / nuisance -> results_controls.txt
prequiescence_followup.py   the strongest result -> results_prequiescence.txt
wavelet_matched.py          the paw null in the pipeline's OWN 20 features,
                            and what the per-session z-score does
                                                 -> results_wavelet_matched.txt
plot_vigor_lda.py           figures/lda1_vs_vigor_28-09-2026
```

Embedding: `clustering/data_files/mouse_LDA_5_bins_raw_shrink0.5_360_28-09-2026`, a local
re-run of `4_mice/lda_shrinkage.ipynb` — 260 sessions, 56 mice, 10 labs. **All 56 mice now
have vigor data**; the earlier check in `vigor/wheel/wheel_vigor_first_vs_proficient.ipynb`
had 53, because the mac's Google Drive design-matrix folder was missing three.

## Answer: wheel, not paw — and the task gate more than either

| | rho vs LD1 | lab-centred | ceiling | rho/ceiling |
|---|---|---|---|---|
| **wheel** speed while moving, whole session | **+0.387** (q = 0.042) | +0.312 (p = 0.019) | 0.93 | 0.41 |
| **wheel** speed, pre-quiescence epoch | **+0.424** (q = 0.009) | **+0.424** (p = 0.0011) | 0.95 | 0.45 |
| **wheel** log(pre-quiescence / quiescence) | **+0.441** (p = 0.0007) | +0.285 (p = 0.033) | 0.95 | 0.46 |
| **paw** log(pre-quiescence / quiescence) | **+0.484** (p = 0.0002) | **+0.373** (p = 0.005) | — | — |
| paw speed while moving, whole session | +0.267 (q = 0.23) | +0.272 | 0.95 | 0.28 |
| paw 0.5–8 Hz velocity power | −0.033 | +0.041 | 0.92 | −0.04 |

q is Benjamini–Hochberg within each family (13 wheel metrics, 12 paw, 8 epoch). Signs are
relative only — an eigenvector is defined up to sign.

**Head to head settles it.** The two are correlated (rho = +0.48 across mice), so each was
partialled on the other:

* wheel speed vs LD1, controlling paw speed: **+0.387 → +0.294, p = 0.028** — survives.
* paw speed vs LD1, controlling wheel speed: **+0.267 → +0.050, p = 0.71** — gone.

That direction matters because the two are not symmetric evidence. The LDA features are
per-epoch occupancies of syllables built from **paw** velocity wavelets, so "LD1 tracks paw
amplitude" would be close to restating what the states are. The wheel enters nothing in the
pipeline (`8_k_10_bin_syllables_19-08-2026` is paw + whisk + lick), so the wheel correlation
is a genuinely external correlate — and it is the one that survives.

**The paw result is more interesting as a null than as a finding.** Whole-session paw
amplitude is essentially unrelated to LD1 (0.5–8 Hz band power, rho = −0.03), even though
the states are literally clusters of that signal, and even though the measure is reliable
(0.86 split-half). LD1 is not a "how much does this mouse move its paws" axis. That axis is
**LD3**, which `vigor/laterality` already showed is paw *laterality*: here `paw_LI_band` vs
LD3 = −0.470 while vs LD1 = −0.040.

## The strongest single result: wheel speed before the quiescence hold

`wheel_speed_Pre-quiescence` is the largest correlate of LD1 anywhere in this folder
(+0.424, q = 0.009), and **it is exactly unchanged by lab-centring** (+0.424, p = 0.0011) —
the only headline metric here of which that is true. LD1 itself does have a lab effect
(Kruskal p = 0.009 across mice), so this is the column that is about mice.

Its sign flips between the two epochs of the trial's start: **+0.424 pre-quiescence, −0.279
during quiescence**. Mice high on LD1 arrive at the trial moving and then hold still; mice
low on LD1 do the opposite. Taken as one contrast, `log(pre-q / quiescence)`, it reaches
+0.441 for the wheel (p_perm = 0.0009 over 10⁴) and +0.484 for the paws — and the paw version
keeps more of itself under lab-centring (+0.373, p = 0.005) than the wheel version does.

> So the readable statement is not "LD1 mice move more" — whole-session `mean_speed`,
> `distance_per_min` and `p95_speed` are all flat (|rho| ≤ 0.23, none surviving FDR). It is
> **how sharply movement is gated by the task's hold requirement.**

### Four things it is not

* **Not epoch length.** Pre-quiescence ends when the mouse holds still, so a mouse that keeps
  moving gets a longer epoch *and* a higher mean speed in it. But epoch length itself is
  unrelated to LD1 (rho = +0.026, p = 0.85), and controlling it *raises* the correlation to
  +0.435.
* **Not whole-session vigor in disguise.** Controlling `wheel_mean_speed_moving` (with which
  it correlates +0.73) leaves +0.325, p = 0.015; controlling `wheel_mean_speed`, +0.456;
  controlling the ITI or Choice epochs, +0.43.
* **Not only reaction time.** `median_reaction` is itself an LD1 correlate here (+0.396,
  p = 0.003, consistent with the existing RT-along-LD1 result), but pre-quiescence speed
  survives controlling it: +0.340, p = 0.010. The two overlap and are not the same thing.
* **Not one lab.** 6 of the 7 labs with ≥5 mice give a positive within-lab rho (angelakilab
  −0.21 is the exception); mainenlab +0.86, hausserlab +0.77, churchlandlab_ucla +0.70.

## Robustness

**Across embeddings.** Re-run against every `mouse_LDA_5_bins*` file on disk that carries the
current schema (10 of them, spanning the PCA-15/25/55 pipelines, the paw-only variant and
the shrinkage one):

| | range of \|rho\| | significant |
|---|---|---|
| `wheel_mean_speed_moving` | 0.228 – 0.431 | 9 / 10 |
| `wheel_speed_Pre-quiescence` | 0.254 – 0.496 | 9 / 10 |
| `wheel_frac_moving` | 0.046 – 0.238 | 0 / 10 |

The one non-significant row in each is `mouse_LDA_5_bins_paw_55_18-09-2026` (p = 0.085 / 0.054). Nothing here depends on the shrinkage embedding.

**Nuisance.** LD1 is unrelated to a mouse's session count (rho = +0.20, p = 0.13) and to
session duration (+0.15, p = 0.27); every headline correlation is unchanged when either is
partialled out. The paw high-frequency noise index — the camera-gain axis
`vigor/paw_bias` identified — does not carry the effect either: partialling it leaves
wheel speed at +0.407 and pre-quiescence speed at +0.461.

**Measurement ceiling.** LD1's own split-half reliability across sessions is 0.975, and the
vigor metrics run 0.69–0.93, so ceilings are 0.82–0.95 and almost nothing is lost to
attenuation. Corrected, the effects sit at 0.41–0.46 of their maximum.

## Relation to what was already in the repo

`vigor/wheel/wheel_vigor_first_vs_proficient.ipynb` ran this as a *premise check* on 53 mice
with the mac's design matrices and reported `mean_speed_moving` vs LD1 at **rho = +0.356,
p = 0.009, falling to 0.242 (p = 0.081) after lab-centring**, concluding it "motivates the
hypothesis and confirms nothing".

That result is confirmed, not overturned, and it gets stronger. **The raw signals were never
the problem**: for a session in both folders, `design_matrices/design_matrix_<eid>_<mouse>`
and `states_files/8_states_file_<eid>_<mouse>` agree to max |diff| = 0 on `avg_wheel_vel`,
`l_paw_x`, `r_paw_x` and `whisker_me`, on the same 1/60 s grid. What changed is coverage
(56 mice rather than 53) and the epoch labels the states files carry, which the design
matrices do not. With the full cohort the same metric reads +0.387 and lab-centring now
leaves +0.312 at p = 0.019 rather than p = 0.081.

The two co-primaries still load on LD1 with **opposite** signs (`mean_speed_moving` +0.39,
`frac_moving` −0.18), which is why their mixture `mean_speed` sits at +0.07. That was
predicted in advance in the earlier notebook and it reproduces.

## The paw null, re-asked in the pipeline's own features

`states_vigor.py` takes paw speed as `hypot(dx, dy) * fs` — unsigned, broadband, 2D. The
segmentation pipeline does something different: `get_speed(..., split=True)` keeps the
**signed x and y components separately**, and the Morlet transform runs on each
(`3.1.1_paw_wavelets.ipynb` cell 8), giving the 20 features the clustering actually uses —
`{l,r}_paw_{x,y}` at `{0.5, 1, 2, 4, 8}` Hz (`3.3_wavelet_clusters.ipynb` cell 3). Wavelet
amplitude is itself unsigned, so the difference is not signed-vs-unsigned: it is
**band-limited per-axis** against **broadband 2D**.

**It makes no difference at the session level.** Read straight from `data/paw_wavelets/`
(all 260 embedding sessions present), the mean of the 20 features correlates with
`paw_mean_speed` at **rho = +0.975** (`paw_band_0.5_8`: +0.922; per paw, +0.973 / +0.915).

**And the null gets cleaner, not weaker.** In the pipeline's own units, pre-standardisation:

| feature | rho vs LD1 | lab-centred |
|---|---|---|
| mean of all 20 | +0.009 | +0.022 |
| left paw / right paw | −0.062 / +0.001 | −0.037 / +0.058 |
| by band, 0.5 / 1 / 2 / 4 / 8 Hz | −0.066 … −0.042 | all \|rho\| < 0.08 |
| high/low band ratio (8 / 0.5 Hz) | −0.021 | −0.021 |

**0 of the 20 individual features reach p < 0.05 uncorrected**; the largest is
`l_paw_x8.0` at |rho| = 0.124. So the +0.267 that `paw_mean_speed_moving` showed is *not*
amplitude — `paw_mean_speed` (unconditional) sits at +0.007. What carries it is the
conditioning on movement, i.e. a property of the **shape** of the speed distribution, not
its level.

### Why: the wavelet features are z-scored within session, twice

* `3.2_wavelet_subsample.ipynb`: the supersession is built as `zscore(resampled_data,
  axis=0)` **per session**, then stacked — so the clustering is discovered in
  within-session-standardised space.
* `3.3_wavelet_clusters.ipynb` cell 20, which assigns a state to **every bin of every
  session**:
  ```python
  mouse_data = stats.zscore(var_array[not_nan, :], axis=0, nan_policy='omit')  # within session
  mouse_data = (mouse_data - global_mean) / global_std                          # then globally
  session_pca = pca_model.transform(mouse_data)[:, :optimal_n_components]
  states = np.argmin(cdist(session_pca, centroids), axis=1)
  ```

Each session's 20 features are centred and scaled by **that session's own** mean and SD
before any state is assigned. A mouse's absolute movement amplitude is therefore removed
from the state labels by construction, which is exactly what the table above shows.

### But amplitude is not *gone* from the feature space — it is just not LD1

Ridge regression of session-level paw amplitude on the 360 syllable features, 5-fold
**grouped by mouse** so no fold shares an animal:

| target | CV R² | r |
|---|---|---|
| paw amplitude, mean of the 20 (log) | **+0.231** | +0.519 |
| paw amplitude, left paw (log) | +0.069 | +0.408 |
| wheel speed while moving | +0.032 | +0.417 |
| wheel speed, pre-quiescence | +0.248 | +0.531 |
| LD1 (positive control — it is a linear function of these features) | +1.000 | +1.000 |

Per-column z-scoring removes each feature's **level** but preserves the **shape** of its
distribution, and the supersession centroids live in globally-standardised space — so a
session whose standardised distribution is shaped differently lands on different states.
Roughly a quarter of the variance in paw amplitude survives into the occupancies.

> **So the right statement is not "amplitude is unmeasurable here". It is that amplitude
> is partly encoded and LD1 is simply not the direction it lies along** — the same shape
> the LD3 result has, where laterality *is* in there and sits on a different axis.

## Method notes

* **Wheel metrics reproduce `wheel_vigor.session_vigor` exactly** — same 30 Hz resampling,
  the same 0.1 move threshold, the same 99.5th-percentile winsorising — so the numbers are
  comparable with the early-vs-proficient analysis. All 332 sessions are natively 60 Hz
  (checked), so the resampling is a no-op for between-mouse ranking here; it is kept only
  for comparability with the training sessions, which are 30 Hz.
* **Paw speed is computed at the native 60 Hz**, from frame-to-frame position. `l_paw_*` is
  divided by 2 (left camera 1280×1024 vs right 640×512); `vigor/paw_bias` measured that
  factor from rig landmarks at 1.980, p = 0.25 against 2.0, so it is correct.
* **Tracking NaNs are dropped, never interpolated**, and a difference is taken only between
  samples one frame apart — a difference across a dropout is a gap, not a movement.
* **`*_bout_rate_p75` and `*_mean_bout_p75_s` use a threshold set from each session's own
  75th percentile**, so the duty cycle is 0.25 by construction and any per-session
  multiplicative gain (camera distance, lightningPose smoother attenuation) cancels exactly.
  Neither correlates with LD1 (|rho| ≤ 0.07 for the wheel, ≤ 0.26 for the paws) — for the
  wheel the LD1 signal is in amplitude and in epoch contrast, not in bout structure at
  matched duty cycle. For the paw it is in neither; see the section below.
* Paw amplitude is **not** gain-free across the two cameras (60 fps vs 150 fps, and the pose
  smoother removes ~28% of real 8 Hz movement at 60 fps against ~4% at 150 fps — see
  `vigor/paw_bias` §5). Both paws are therefore reported separately as well as averaged, and
  `paw_hf_noise` is carried as a covariate. It does not drive anything here.

## Caveats

1. **`frac_moving` is dead.** It was a pre-registered co-primary in the earlier wheel work
   and it correlates with LD1 at −0.18 (p = 0.18) here, in none of the 10 embeddings. Only
   `mean_speed_moving` — vigor *given* movement — carries the relationship.
2. **The gating contrast was found, not predicted.** The epoch split was run because the LDA
   features are per-epoch; the pre-quiescence/quiescence sign flip came out of that, it was
   not hypothesised. It survives FDR within the 8-metric epoch family and every control
   above, but it wants a confirmatory test on a held-out cohort before it is a claim.
3. **n = 56 mice.** A lab-centred effect of 0.3 is around the edge of what this resolves; the
   earlier notebook's "more mice for the LD1 check" is still the right ask.
