### ⚠ A public OOF gain does not transfer across base strength — measure it before disputing one — 2026-08-15 (w15f)

The single most reusable thing this run produced. `raykkretzschmar`'s transductive
teacher/student correction is reported by its author as +1.8e-5 to +3.6e-5, positive in 60/60
anchor-by-fold comparisons. Rebuilt here and measured on the frozen SKF5 seed42 folds with the
identical instrument at five base strengths (`experiments/w15f_baseladder.py`):

| base | base CV | additive@0.10, real − matched permuted control | 1-param rank blend, cross-fitted |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | **+1.62e-5** (z +2.25) | +3.56e-6, 4/5 |
| `stack_pub86_hybrid` | 0.969678 | **+1.60e-5** (z +2.36) | +3.65e-6, 4/5 |
| `blend158_logit` | 0.969961 | −3.16e-6 (z −0.52) | +1.04e-6, 4/5 |
| `blend158_hybrid` | 0.970028 | −7.16e-6 (z −1.09) | +7.39e-7, 4/5 |
| `blend159av_h3` | 0.970049 | −3.28e-6 (z −0.45) | +9.91e-7, 4/5 |

His four anchors sit at **0.969667–0.969721**, where we measure **+1.62e-5 against his +1.8e-5**.
**Both measurements are correct.** The correction's information is already inside our pack and is
not inside a 0.9696 stack.

**Operational rule: before disputing any public OOF claim, reproduce it at the claimant's base
strength.** `stack_pub74_logit` (0.969641) is kept on disk precisely for this — it is very nearly
the standard public anchor (a logistic stack over the 74-model library on the frozen folds), so
it converts "this does not work for us" into "this is worth X at your base and Y at ours".

### The transductive teacher/student correction is NOT transductive — 2026-08-15 (w15f)

w15e identified `raykkretzschmar/s6e8-transductive-anti-student-signals` as the only
sub-0.98-correlation direction found all week (max |ρ| 0.0772 against 168 members) and argued it
must be orthogonal to our pack *by construction*, since all 168 of our members are inductive and
this is defined by reconstruction failure on the test distribution. **That argument is now
falsified by measurement.**

`experiments/w15f_extract.py` builds the identical student **without** the unlabeled rows:

- rank corr between the transductive and inductive residuals: **+0.967**
- the pure transductive component `c_trans − c_induc`: cond AUC given the base **0.500579** vs a
  within-bin permutation control 0.499935 ± 0.00125, **z +0.52**
- its one-parameter blend weight: **0.000**, tied with the permuted and uniform-noise controls

Carrying the unlabeled rows contributes nothing measurable. The object is
teacher-minus-smooth-student — an ordinary inductive function of the 12 columns — and therefore
sits **inside w15b's power bound** (78–102% recovery of a leader-sized injected signal), not
outside it. **Do not re-open the transductive class on the strength of the correlation argument;
low correlation to the pack was necessary but nowhere near sufficient.**

Measured worth on our best base, with the student hyperparameter averaged out over three
configurations bracketing his residual sd: **cross-fitted ΔAUC +3.58e-6, 4/5 folds**; the
correction's conditional AUC given the base is 0.505819 vs control 0.500227 ± 0.00116 (z +4.84).
Real, above the 2e-6 reproducibility floor, and ~1/50th of the 18e-5 gap to MILANFX.

### The −1.1e-5 additive-correction toll replicates, and it scales as the square — 2026-08-15 (w15f)

w15e measured the toll for a signal-free additive rank correction at **−1.105e-5** for a
perturbation of sd 0.00217, and flagged that it scales with magnitude. It does, quadratically.
Independent build, different vector, 200-draw matched permuted control: perturbation sd 0.00479,
a factor **2.208**, predicts 1.105e-5 × 2.208² = **5.38e-5**; measured **−5.52e-5**. A 3% match.

**So the toll for any additive rank correction here is ≈ −1.105e-5 × (sd_move / 0.00217)².**
Price a proposal with that before building it, and note the corollary: a construction can read as
a clean null purely because its perturbation is too large. w15f's first student produced a null
at the author's published weight 0.10 (−3.9e-6, 0/5 folds) that was almost entirely toll — the
same vector at w=0.01–0.05 reads +2.2e-6 to +5.1e-6.

### ⚠ Check a weight grid against the perturbation size before trusting a cross-fitted search — 2026-08-15 (w15f)

A correction standardised to unit sd has a useful weight range of ~0–0.01 here. Searching it on
`linspace(0, 0.30, 151)` (step 0.002) quantises the per-fold choices to {0, 0.002} — "nothing" or
"ten times too much" — and returned cross-fitted ΔAUC **−2.58e-6 at 1/5 folds** while the same
vector at its full-data optimum read **+1.11e-5 at z +3.72**. Refined to `linspace(0, 0.02, 201)`
the per-fold weights land at 0.0010/0.0010/0.0012/0.0014/0.0011 and the honest answer is
**+3.58e-6 at 4/5**. The failure is silent and produces a confident wrong sign.

### Rebuilding a public artefact whose training code was never published — technique (w15f)

`raykkretzschmar`'s notebook ships only the recomposition from a saved NPZ, so teacher and student
had to be reconstructed from prose. Two checks make that honest, and both are cheap:

1. **Match the teacher against his published test-space vector.** Ours reached rank corr
   **+0.99529** with his `test_teacher` — independently-configured GBDTs on the same data land
   very close, so a low number here would have meant a real transcription error.
2. **Sweep, do not guess, the hyperparameter he never published,** and pick the bracket from a
   *target-free published statistic* rather than from the score. His residual sd is 0.01961; three
   students at 0.0479 / 0.0311 / 0.0199 bracket it. All three agreed in direction (cond AUC z
   +3.02 / +4.39 / +5.15, cross-fitted +1.0 / +3.5 / +3.1e-6, 4/5 folds each).

⚠ **When two fidelity criteria disagree, average rather than choose.** Scale fidelity picked
`rough` (resid sd 0.01986 vs his 0.01961, 1.3%); shape fidelity picked `mid` (rank corr +0.538 vs
his correction, against rough's +0.338). Neither sees the label. The equal average of the three
standardised corrections is a zero-fitted-parameter combination — the same reasoning as `h3` —
and it removes the temptation to carry forward whichever student happened to score best.

### Cached artefacts that make a fourth student cheap (w15f)

`experiments/w15f_inner_oof_f{0..4}.npy` hold the nested inner-fold teacher OOF predictions —
20 of the 25 LightGBM fits and essentially all the wall clock (~50 min at 3 threads). With those
plus `w15f_nested.npz`'s `teacher_r` (deterministic; a rerun reproduced it to every printed
digit) and `w15f_teacher_test.npy`, **a new student costs one fit rather than an hour.**
`experiments/w15f_X.npy` is the 56-column target-free frame (raw + constrained imputation +
missing indicators + generator identities + exact-value frequency counts), reusable for any
model that must not see the target.

### h3-family LB invariance is broken at 7/7 — 2026-08-15 (w15f)

w15a recorded the rule at 6/6: every h3-family file submitted returned exactly 0.97105 across CV
0.970046–0.970049. `w15f_antistudent_avg` is `blend159av_h3` plus a correction at ρ **0.9999928**
— a *smaller* perturbation than w15e's file, which moved one unit down — and it returned
**0.97107**, our first score above 0.97106.

⚠ **This is not evidence the correction works, and should not be cited as such.** Its CV gain is
+3.58e-6; at the workspace's ~56% CV→LB pass-through that predicts ~+2e-6, i.e. no visible move on
a 1e-5 grid, against an observed +1e-5 to +3e-5. The slice moved 5–10× more than the mechanism can
account for — w14b §4's pattern, where public flattery is borrowed from private at 4:1.

**Live consequence for the unset selection toggle:** best public score is now 0.97107 on a file
whose base is the CV pick `blend159av_h3` and whose CV (0.9700528) is the best held here, so the
auto-select default is no longer `blend158_logit` (CV 0.969961, priced by w14b at ~−111e-6
predicted private). **The unattended-default exposure has fallen from ~−111e-6 to roughly zero.**
A human should still select `blend159av_h3` and `blend160origm_h3` — they carry zero fitted
parameters where this file carries one — but the cost of nobody doing so is now much smaller.
