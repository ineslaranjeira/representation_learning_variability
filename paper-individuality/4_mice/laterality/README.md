# Is LD3 a left/right forepaw axis?

```
laterality_features.py            per-session position / wavelet / syllable features
paw_laterality_sessions.csv         -> its cache (332 sessions)
lda3_paw_laterality.ipynb         the analysis
```

Embedding: `clustering/data_files/mouse_LDA_5_bins_25_18-09-2026` — 269 sessions,
58 mice, 10 labs. All laterality indices are `LI = (left − right) / (left + right)`,
so **positive = left paw**, following `segmentation/paw_bias`.

## Answer: yes, and LD3 is the only leading axis that is

Mouse level, n = 58, FDR across the 12 measures × 8 LDs screened:

| measure | view | r with LD3 | q | partial on 32 Hz noise |
|---|---|---|---|---|
| LI-weighted state occupancy | syllables (the LDA's own input) | **−0.63** | 0.002 | −0.48 (p = 2e−4) |
| left-states − right-states | syllables | **−0.61** | 0.002 | −0.45 (p = 5e−4) |
| 0.5–8 Hz velocity power LI | wavelets | **−0.62** | 0.002 | −0.43 (p = 1e−3) |
| lead–lag, + = left leads | wavelets, *gain-invariant* | **+0.43** | 0.014 | +0.36 (p = 7e−3) |
| excursion / postural spread LI | raw position | +0.00 / +0.05 | n.s. | — |

LD3 is the strongest of LD1–LD8 for every amplitude measure. Sign: **high LD3 = more
right-paw movement**. At the session level (n = 269) with a permutation that shuffles
mice and keeps their sessions yoked, r = −0.61 / −0.59, p_perm = 1e−4.

## Why it is LD3

Four of the eight HMM paw states are lateralised (`state_profiles_19Ago2026.csv`:
state 3 LI = +0.63 and state 5 +0.31 are left; state 4 −0.81 and state 6 −0.41 are
right). A state's correlation with LD3 is a near-linear function of how lateralised
that state is, driven mostly by state 4 (r = +0.61) against state 3 (r = −0.45).
Whisk and lick contribute nothing (|r| < 0.01). In the reconstructed design matrix
the effect is flat across all four epochs and all 10 bins — a whole-session trait,
not something that happens at a moment of the trial.

## The controls

* **Camera gain / tracking noise.** The 32 Hz noise index correlates −0.49 with LD3
  and +0.78 with the amplitude index, so it had to be partialled out; the syllable
  and wavelet relations keep ~70% of their size and stay at p < 1e−3.
* **Lab / rig.** LD3 is the *only* leading axis a lab label does not explain
  (F = 0.75, p = 0.67, η² = 0.12, against LD1 η² = 0.52, LD2 0.35, LD4 0.54), and
  centring every variable within its lab leaves the correlations unchanged
  (−0.60 / −0.64). A rig-geometry explanation would have to survive both.
* **Gain-invariant measures.** Lead–lag works (+0.43, partial +0.36). Bout rate at
  matched duty cycle (+0.24) and 4 Hz spectral shape (−0.36) do not survive the noise
  partial, so the evidence rests on amplitude plus lead–lag, not on all of them.
* **Measurement ceiling.** Within-session split-half of the amplitude LI is r = 0.894,
  so the measure is close to noise-free and the correlations are not attenuated much.

## Three caveats that belong in any caption

1. **The population mean of every index here is instrumental, not biology.** The two
   forepaws are seen by two different cameras; `segmentation/paw_bias` shows the group
   left/right offset is camera gain. Only the *spread across mice* is interpretable.
2. **LD3 is the closest axis, not exactly the axis.** The LDA's leading eigenvalues are
   near-tied, so the axes are free to rotate among themselves. LD3 alone gives
   cross-validated R² = 0.29–0.35 for the syllable indices; LD1–LD8 jointly give
   0.73–0.83, with LD3 the largest single component of the best direction
   (|cos| = 0.67–0.73). Read it as "laterality lies close to LD3 within the leading
   subspace".
3. **Position is not amplitude.** Postural spread and excursion of each paw carry no
   LD3 signal at all. The states are fit on velocity wavelets, so what LD3 reads is how
   much each paw *moves*, not where it sits.

## Related work already in the repo

* `segmentation/paw_bias/` — the paw-bias metrics, their validation and the full camera
  artefact analysis. Read its README before acting on anything here.
* `lab/lda1_vs_raw_paw_laterality.py` — the same question asked of LD1 and LD2, against
  raw design-matrix positions. This folder is its LD3 counterpart and reuses its
  conventions and its `state_LI` definition.
