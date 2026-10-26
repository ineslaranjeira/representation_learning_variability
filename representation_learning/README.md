# Representation learning during IBL training

**Question.** When a mouse learns the IBL task slowly, is the bottleneck learning *what to do* (a
slow value/policy update over states it already represents), or learning *which states exist*
(discovering the task's structure: the stimulus, quiescence, the timing of the trial)?

## Where things stand (October 2026)

| notebook | what it does | main result |
|---|---|---|
| `0_toy_known_states` | actor-critic agent that is *given* the task states (M1) on a toy task (`toy_task.py`) | learns in ~100 trials, ~100× faster than mice (median ~9,600 training trials). A small learning rate could close the gap, so trial counts alone can't decide between the two accounts |
| `1_toy_wheel_threshold` | the same agent must respond with the wheel | bad movement execution costs time, not trials; the motor side doesn't explain slow learning in the toy |
| `2_epoch_structure_wheel` | do mice learn *where* in the trial to keep still? (wheel, training sessions) | no selective stillness: violations rise over training while free movement stays flat. The trial start is not signalled, so the only usable clock is time since feedback. Learning-speed groups barely differ |
| `3_choice_strategy` | per-session choice model: stimulus, bias, win-stay / lose-shift | fast and slow learners differ in **when** the stimulus starts to drive choices (onset median 4 vs 13 sessions); most variation in learning speed sits before that onset |
| `4_onset_predictors` | what, in sessions 1–2, predicts that onset | not the debiasing experience loop; **hidden stimulus effects** do (RT depends on contrast, errors are slower) — the stimulus registers before choices follow it |

The modelling sketch has three agents with the same learning rule and different state
representations: **M0** (no structure: all no-stimulus time is one state), **M1** (known structure:
stimulus, plus clocks for time since feedback and since the last movement), **M2** (state discovery: a
tree that splits states when the TD error says a cut matters).

## Why behaviour alone is not enough

All three models learn with the same rule; they differ in *which states the critic values*. Choices
show the policy, one bit per trial, and a slow learning rate and slow state discovery can produce the
same learning curve (notebook 0). Notebooks 2–4 also suggest that what the mouse *does* and what it
*knows* come apart: behaviour shows no selective stillness, and the stimulus affects RT before it
affects choice.

## Proposal: dopamine as a read-out of the critic

If midbrain DA carries the TD error, each DA response is a read-out of the critic's value V(s) at
that moment. That shows which states the mouse distinguishes, separately from what it does. With the
wheel, three tests become possible that behaviour can't run.

1. **The same movement, two contexts.** A wheel movement during quiescence restarts the timer and
   delays reward; the same movement in the ITI costs nothing. If the mouse represents the quiescence
   state (M1-like), a quiescence violation should evoke a negative RPE and a free ITI movement should
   not. Compare movements matched on kinematics, which controls the movement-related DA signal. This
   is the critic-side version of notebook 2: the critic may have learned the structure even though
   behaviour doesn't show it.
2. **The clock.** The only cue that the stimulus is coming is time since feedback. If the mouse uses
   that clock, DA should change systematically with elapsed time, towards the expected trial start.
   The ITI is ~1 s longer after errors, so post-correct vs post-error traces test *which* clock (time
   since feedback, or time to trial start), as the behavioural clock index in notebook 2 does.
3. **The pre-onset phase.** Slow learners spend many sessions with stimulus weight ≈ 0 while hidden
   stimulus effects already exist (notebook 4). Do DA stimulus and feedback responses become
   stimulus-specific *before* choices do? If they do, the representation exists before the policy
   uses it, and learning speed sits in the actor, not in state discovery.

**Formal version:** fit M0/M1/M2-style models to each mouse's DA responses (not its choices) and
compare them, session by session. This is a within-mouse comparison over thousands of trials, so a
handful of mice is enough.

**What it can't do:** with a few mice it shows the mechanism in each of them, but it can't explain
*why* fast and slow learners differ. That needs the ~100-mouse behavioural cohort (notebooks 3–4).

**Caveats:**
- DA-as-RPE is an assumption. VTA is the cleaner RPE signal; SNc carries more movement-related
  activity, which makes test 1 harder and the wheel regressors essential.
- GCaMP is slow (hundreds of ms). Early in training, RTs are long, which helps separate stimulus,
  movement and feedback; late in training they overlap.
- Tests 1 and 2 need the **continuous** trace through the ITI and quiescence. The export in
  `../neuromodulators/pose_qc_hmm_k4/` has only peri-event snippets (−1 to +1 s around stimulus and
  feedback) and no wheel, so it can't be used for this. The continuous signal is on Alyx (below).

## Data availability: DA mice with photometry during training

From `da_inventory.py` (Alyx query, 26 Oct 2026; outputs `da_inventory_mice.csv`,
`da_inventory_sessions.csv`). "With signal" = training sessions with the processed continuous
`alf/photometry/photometry.signal.pqt`; all but a few of these also have the wheel. "First" = the
first training session that has it.

**2025–26 cohort: recorded from the first training session, nearly every session.**

| mouse | target | training sessions | with signal (+ wheel) | first | reached biased |
|---|---|---|---|---|---|
| ZFM-08818 | VTA | 66 | 61 (61) | 1 | yes |
| ZFM-08828 | VTA | 69 | 63 (63) | 1 | yes |
| ZFM-09044 | VTA | 65 | 57 (57) | 1 | yes |
| ZFM-09394 | VTA-r | 16 | 14 (14) | 1 | yes |
| ZFM-08827 | VTA | 65 | 59 (57) | 1 | **no** |
| ZFM-08689 | VTA | 5 | 5 (5) | 1 | no (stopped) |
| ZFM-09139 | SNc | 31 | 30 (30) | 1 | yes |
| ZFM-09140 | SNc | 16 | 15 (15) | 1 | yes |
| ZFM-09343 | SNc-l | 14 | 13 (13) | 1 | yes |
| ZFM-09826 | SNc-r | 27 | 27 (27) | 1 | yes |
| ZFM-09346 | SNc-r | 54 | 51 (51) | 2 | **no** |
| ZFM-09350 | SNc-r | 54 | 51 (51) | 1 | **no** |
| ZFM-09980 | SNc-r | 19 | 14 (14) | 1 | no (recent) |
| ZFM-09983 | SNc-r | 19 | 19 (19) | 1 | no (recent) |

**2021–22 cohort: recorded from session 1–3, but only about half the training sessions.**
ZFM-03447 (VTA, 32/54), ZFM-04026 (VTA, 41/68), ZFM-04022 (SNc, 26/57), ZFM-03448 (SNc, 50/105),
ZFM-04019 (SNc, 15/41), ZFM-03450 (SNc, 19/62). ZFM-08488 and 08554 (VTA, 2025) start only at training
session 13 (7/19 each). ZFM-08551 and the 2018 Dat-Cre mice have no photometry.

**Reading.**
- **VTA, complete from session 1: 4 mice that reached criterion**, spanning fast (ZFM-09394,
  16 sessions) to slow (ZFM-08818/08828/09044, 65–69 sessions), plus **one VTA mouse that never
  reached criterion** (ZFM-08827, 59 recorded sessions). That's enough for the within-mouse
  comparison, and the slow learners have long pre-onset phases, which is what test 3 needs.
- **SNc, complete from session 1:** 4 that reached criterion and 2 long non-learners
  (ZFM-09346/09350), useful as a movement-heavy comparison.
- Not yet checked: signal quality in training sessions (the pose-QC table covers biased/ephys only),
  whether the signal spans the whole session, and the behavioural onset of stimulus use in these
  mice (from their trial tables, as `choice_strategy.py`).

## Next steps

1. Load one complete VTA mouse (e.g. ZFM-08828): continuous signal + wheel + trials for a few early
   and late training sessions, check signal quality and alignment.
2. Compute stimulus-use onset (`choice_strategy.py` / `onset_predictors.py`) for the DA mice, to place
   each session relative to onset.
3. Test 1 (same movement, two contexts) on VTA mice, before and after onset.
4. Simulate what M0/M1/M2 predict for the three DA read-outs with the toy agents, before fitting.
