# Journal — playground-series-s6e8

Append-only. Newest entries at the bottom.

---

## 2026-08-10 — slot 1/10 — angle: tune LightGBM properly against the fixed folds

**First run on this workspace. It was empty: no journal, no research, no submissions.**
`kaggle competitions submissions` returned "No submissions found", so there is no prior
work on this account to recover and no established score to protect.

### Orientation findings

- **Metric is ROC AUC** (confirmed from the competition page and every public notebook
  title). Submit continuous probabilities.
- **Daily submission cap is 10, not 5.** The CLI printed "9 submissions remaining today"
  after my first submit. The brief's guess of 5 was wrong. Recorded in `RESEARCH.md`.
- Data: 691,369 train / 296,302 test, 12 predictors, base rate 0.70942. Every column is
  4–20% missing in both splits.

### The two things that decided this run

**1. A public notebook that is a complete ablation record.**
`tomasa2/s6e8-what-moved-the-score-and-what-didn-t` documents 30+ measured experiments
with signed deltas. It is now distilled into `RESEARCH.md`. The headline results: full
resolution target-encoding of *every* column including the continuous ones is worth
+0.0023 (by far the biggest single lever); imputing *alongside* the NaNs rather than
replacing them is +0.0012; the decimal lattice is +0.0001. And critically, a list of
things that DO NOT work, which saves future runs from re-spending slots on them —
**concatenating the original dataset measures −0.0001**, so the single most reliable
Playground edge is dead in this competition. Pseudo-labeling is −0.0034.

**2. The public OOF library — `szymonkapiski/s6e8-oof-library-47-models`.**
Actually 74 models, with `manifest.csv`, `hyperparameters.json` and the full `src/`
training code. Every member is out-of-fold on
`StratifiedKFold(5, shuffle=True, random_state=42)` in original row order.

I adopted that exact fold scheme as our frozen CV (`agent/common.py:get_folds`) so our own
models stack against all 74 with no leakage. **I verified this rather than assuming it**:
`agent/stack.py --verify` re-scores all 74 arrays and every one reproduces its manifest
AUC to <5e-5. Row alignment confirmed.

### What I built

- `agent/common.py` — frozen folds, column lists, the OOF save/load contract.
- `agent/features.py` — constrained imputation (the generator identity
  `daily >= social + gaming + work_study` holds with zero violations, so the observed
  component sum is a *hard* lower bound on a missing `daily` and the slack is a hard
  upper bound on a missing component) + the full quantisation lattice + fold-safe
  target/frequency encoding with an inner 4-fold loop.
- `experiments/build_cache.py` — the enabling trick. The target encoding is
  fold-dependent but hyperparameter-independent, so it is computed once and cached
  (40 base + 144 TE/CT = 184 features, ~30s/fold). Every subsequent trial is pure
  LightGBM time. Without this a real sweep would not fit on 16 CPU cores.
- `experiments/tune_lgbm.py` — the sweep. `agent/run_lgbm.py` — 5-fold finalist runner.
- `agent/stack.py` — logit stacking with paired 50/50 honest evaluation.

### Why the angle is a real opening

Every LightGBM in the public library that uses the lattice features (`lat_*`, `latwide_*`,
`lattri_*`, `latmax_*`, best OOF **0.96768**) shares ONE hand-set parameter vector:
`lr 0.035, num_leaves 96, min_child_samples 40, subsample 0.9, colsample 0.6,
reg_lambda 5, max_depth 7`. The only tuned LightGBM in `hyperparameters.json`
(`lgbm_tuned`) was tuned against the far weaker raw+iterative-imputation features and
landed in a completely different regime (`max_depth 4, num_leaves 18`). **LightGBM has
never actually been tuned against these features.** That is the gap.

Sweep protocol: rank configs on fold 0 only — comparing configs on the *same* fold is a
paired comparison so fold-specific variance largely cancels — then re-run finalists on all
5 folds. Only the full-OOF number gets quoted or compared.

### Submitted

`slot1 stack-pub74-logit` — L2 logistic regression on `clip(log(p/(1-p)), ±30)` over all
74 library members.

| | |
|---|---|
| cross-fitted OOF (frozen folds) | **0.969641** |
| best single member (`naji03`/`naji05`) | 0.968815 |
| gain over best solo, paired 50/50 | **+0.000832 ± 0.000036, consistent across all 5 splits** |

Combiner comparison, paired on the same splits (this is why logit stacking, not averaging):

| combiner | vs best solo |
|---|---|
| logit stack (C = 0.01 … 10, all equivalent) | **+0.00083** |
| hill climb | +0.00065 |
| probability stack | −0.00067 |
| naive mean of all 74 | −0.00108 |

`lookup` (the Lookup-Transformer) takes the largest coefficient, 0.2415 — it is the most
decorrelated member in the library and the stacker knows it. Several members get negative
coefficients (`pub_ryota` −0.16, `cat` −0.11): the stacker is using weak-but-decorrelated
members as *corrections*, which is exactly what a hill climber cannot do.

**This is not filler.** It establishes my own CV→LB offset on a submission whose CV I
trust, which is the most useful single number I can collect right now. The public record
says LB ≈ CV + 0.0013; if that holds this lands ≈0.9709, around the top of the public LB.
Whatever the offset turns out to be, every later run gets to use it.

### Still running at time of writing

Stage-A sweep, 30 configs on fold 0 at lr 0.05. Partial ranking (fold-0 AUC):
`leaves63_depth7` 0.96668 > `leaves31_depth6` 0.96653 > `leaves192_depth9` 0.96644 >
`control` 0.96638 > `leaves128_depth8` 0.96634. **Smaller trees than the library's 96
leaves are winning** — consistent with 144 of the 184 features being target-derived and
therefore easy to overfit. The regularisation knobs the library never touched
(`path_smooth`, `feature_fraction_bynode`, `extra_trees`, `max_bin`, `min_child_samples`)
are still to come and are where I expect the real gain.

### Next run should do, in this order

1. **Read the LB result of `stack_pub74_logit` and write down the actual CV→LB offset.**
   That number calibrates every later decision. CV was 0.969641.
2. Finish/extend the sweep, then run the top 2–3 configs through `agent/run_lgbm.py`
   on all 5 folds **with `--frac`**. The decimal lattice (`frac_`, `d1_`) is a genuinely
   new channel that the library's `lat_*` models do not have — TE cannot see it, because
   "everything ending in .2 shares something" pools across integer parts and TE's levels
   do not. Measured +0.0001 elsewhere; never combined with these features.
3. Add the tuned member to the stack and measure the delta honestly. Expect it to be
   small — the library already has ~20 GBMs — so judge it on the paired 50/50 number,
   not on hope.
4. Do NOT re-try: original-dataset concatenation, pseudo-labeling, naive averaging,
   deep trees, pairwise/multi-resolution TE. All measured negative. See `RESEARCH.md`.
