# Fine-grained behaviour, latent states and neuromodulators

**Question (from the abstract):** choice-only models (2-state GLM-HMM) lump together internal states that fine-grained behaviour can tell apart (disengaged vs impulsive lapses; confident vs hesitant engagement). Neuromodulator (NM) activity should follow those finer states.

**Design rule:** states are defined from behaviour only. The NM photometry is held-out validation and is never used to fit the states, so "NMs differ between states" is not circular.

## Three levels of behavioural description

| level | model | uses | coverage |
|---|---|---|---|
| 1 choice | GLM-HMM K=2 (paper model, per mouse) | choice, stimulus, history | all 36 mice (`trial_position_engagement.py glmhmm`) |
| 2 choice + RT | DDM-HMM K=4 (collaborator's, per mouse) | choice + reaction time | 8 mice: DR-5HT 2, VTA-DA 2, LC-NE 3, SNc-DA 1; no NBM-ACh |
| 3 fine behaviour | vigor, syllables, trial modes | paw, wheel, whisking, licking, at 30 Hz | all 36 mice |

The DDM-HMM already separates engaged, slow, impulsive, biased and disengaged states (notebook `2_ddm_hmm_states`). So the question is no longer "can behaviour split lapses?". It is **what each level adds over the one below**, and whether level 3 adds anything beyond the DDM-HMM.

Known so far: the DDM states map onto trial modes mostly through RT (~90% of the shared information). What survives RT matching is pre-trial paw stillness in disengaged states (`3_ddm_hmm_trial_modes`). Notebook `4_ddm_hmm_photometry` gives the NM trace per state as a deviation from the session mean, but only descriptively.

## Analyses

### A. How the state models relate (8 mice)
- Align the GLM-HMM and DDM-HMM per trial: how do the K=4 kinds split the GLM-HMM engaged and lapse states?
- Expected: GLM-HMM "disengaged" = DDM disengaged + impulsive + biased, and GLM-HMM "engaged" = DDM engaged + slow.
- This shows which distinctions the choice-only model misses.

**Result (`1_state_models.ipynb`):** the expectation above does not hold. The two models barely share information (H(DDM | GLM) / H(DDM) = .94). The DDM-HMM follows response timing (its "disengaged" = RT ~4–5 s, often still accurate). The GLM-HMM lapse state follows side bias and choice repetition (P(engaged) correlates +.4 with biased-L and −.4 with biased-R). Smoothing the DDM kinds over 21 trials doesn't change this.

**Fit check (same notebook):** in 6 of 8 mice the DDM-HMM switches every 3–9 trials and works more like a mixture over RT/accuracy regimes. It misses the slow structure: log RT is autocorrelated over ~100 trials and accuracy streaks over ~10, and the fitted chain reproduces neither. "Impulsive" mostly labels fast errors (weak evidence for a persistent state). Only ZFM-08776 and 09365 have sticky fits.

### 1. Pre-stimulus prediction, model-free (`2_prestim_prediction.ipynb`, `prestim.py`; replaces B–C as the next step)
Motor state before the stimulus predicts RT (35/37 mice), not accuracy. The NM baseline adds beyond motor for RT in every NM (small), and pre-stimulus VTA/SNc DA predicts *lower* accuracy. Pre-stimulus motor state also changes the NM stimulus response (movement carry-over not yet excluded).

### B. Which state axis do the NMs follow: choice/bias (GLM) or timing (DDM)? (8 mice, within fiber)
- Revised after A: the models are two different axes, not nested. Fit both state labels jointly to the NM signal (per sample, within fiber, RT and trial position held fixed). Ask which one explains NM variance, and whether each adds to the other.
- Within the GLM-HMM lapse state, compare DDM impulsive vs disengaged vs biased trials. Within the GLM-HMM engaged state, compare DDM engaged vs slow.
- Readouts: pre-stimulus baseline, stimulus response, feedback response. Per fiber, then across mice.
- **RT is the main confound** (the DDM states are RT regimes). Match on RT deciles, or include log RT, and also align to movement onset.
- Hypotheses:
  - DR-5HT lower before impulsive trials (5HT and waiting);
  - LC-NE pre-stimulus high in impulsive, low in disengaged (arousal);
  - DA expectation responses lower in disengaged.
- Formal version: held-out NM variance explained by GLM-HMM state vs DDM state at a fixed sample.

### C. Does fine behaviour add beyond the DDM-HMM? (8 mice)
- Within each DDM state, at matched RT: do vigor, whisking, licking and pre-trial stillness explain NM variance beyond the state label?
- Sub-split DDM states by fine behaviour, e.g. disengaged-still vs disengaged-moving, or impulsive with vs without pre-stimulus movement. Test NM differences within the sub-splits.
- This is the abstract's core claim, in its strongest form.

### D. Confidence within the engaged state (all mice for the behaviour; DA for the test)
- A behavioural confidence proxy: anticipatory licking before feedback, wheel decisiveness, settling after the response.
- Validate it with the confidence signature: it rises with |contrast| on correct trials and falls on errors.
- DA test: on correct trials at the same contrast, is the reward response larger when the proxy says the mouse was less confident (reward prediction error)?
- Use the DDM engaged state where it exists, and GLM-HMM engaged otherwise.

### E. Beyond 8 mice
- The DDM-HMM covers 8 of 36 mice and no NBM-ACh, so B and C have 1–3 mice per NM.
- Options:
  1. Ask the collaborator to fit the remaining mice. Their importer stores every K fitted, so other K values may exist too.
  2. Fit a fine-behaviour + choice + RT state model ourselves on all mice, and check it against the DDM-HMM on the 8 mice.
- Option 2 is also the formal "behaviour-informed HMM": its states should explain more held-out NM variance than the choice-only GLM-HMM at the same number of states.

## Throughout
- **Units.** Fiber, then mouse. Compare states within fiber: the bilateral NBM mice show coupling depends on the fiber (`9_coupling_individuality`).
- **Photometry window.** Only ±1 s snippets around stimulus and feedback, so the pre-stimulus window is the only tonic proxy. Switches over many trials and ITI dynamics need the continuous photometry.
- **Session drift.** Disengaged and slow states are late in the session, and so is the slow NM drift (notebook 7, section 5). Control trial position in every contrast.

## Order
A (quick, sets the vocabulary) → B → C → D. E runs in parallel: ask the collaborator now.

## Files
- `states.py`: loading and aligning GLM-HMM, DDM-HMM and per-trial behaviour/NM features (functions only).
- `1_state_models.ipynb` (A), `2_nm_by_state.ipynb` (B), `3_fine_behaviour_within_state.ipynb` (C), `4_confidence.ipynb` (D).
- Notebooks 0–6 in `neuromodulators/` are the groundwork: trial modes, DDM-HMM states, PSTHs.
