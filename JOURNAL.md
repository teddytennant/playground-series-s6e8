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

### Results of the angle — LightGBM tuning was a null result, and that is the finding

Two-stage sweep on fold 0, all on the cached lattice matrices (184 features).

**Stage A** — 29 trials, one knob at a time off the library's hand-set control
(control fold-0 AUC 0.96638 at lr 0.05). Every winner pointed the same direction: less
capacity, more regularisation, which is what you would expect when 144 of the 184
features are target-derived and therefore cheap to split on and easy to overfit.

| knob | fold-0 AUC | vs control |
|---|---|---|
| `reg_lambda 80` | 0.96671 | +0.00033 |
| `max_bin 511` | 0.96669 | +0.00031 |
| `num_leaves 63, max_depth 7` | 0.96668 | +0.00030 |
| `feature_fraction_bynode 0.8` | 0.96663 | +0.00025 |
| `min_child_samples 250` | 0.96658 | +0.00020 |
| `subsample 1.0` (bagging off) | 0.96656 | +0.00018 |
| control (`num_leaves 96, l2 5`) | 0.96638 | — |
| `colsample_bytree 1.0` | 0.96616 | −0.00022 |
| `extra_trees` | 0.96601 | −0.00037 |

**Stage B** — 7 combinations of the winners, since capacity / leaf size / L2 all
regularise the same thing and are not additive. Best: `B_mcs250_l2_80`
(`num_leaves 63, max_depth 7, max_bin 511, min_child_samples 250, reg_lambda 80`)
at **0.96688**, i.e. +0.00050 over the control on fold 0. Pushing `reg_lambda` past the
stage-A grid edge did not keep paying (200 → 0.96677, 500 → 0.96663).

**Then it did not survive to full OOF.** Finalists on all 5 frozen folds at lr 0.025:

| model | full OOF AUC |
|---|---|
| library's best LightGBM (`lattri_lgbm` / `latmax_lgbm`, hand-set) | 0.96768 |
| `lgbm_tuned_lat` (tuned) | **0.96771** |
| `lgbm_tuned_lat_frac` (tuned + decimal lattice) | **0.96782** |

**Tuning bought +0.00003 — below the 0.00005 noise floor. Call it zero.** The +0.00050
fold-0 gain was mostly an artefact of comparing against a control handicapped by lr 0.05;
stage A itself showed lr 0.035 is worth +0.00027 over lr 0.05, so the honest like-for-like
tuning gain was never more than ~+0.0002, and on the full OOF it was nothing.

**The decimal lattice, by contrast, paid: +0.00011, and it won on all 5 folds
individually** (fold deltas +0.00017 / +0.00009 / +0.00012 / +0.00021 / +0.00007). At
0.96782 `lgbm_tuned_lat_frac` is the strongest single gradient-boosted model in any of the
libraries, edging `latr1_xgb` (0.96780).

That is the run's real lesson and it matches the ablation record exactly: **a new channel
beats tuning the existing one.** `frac_`/`d1_` are absent from every `lat_*` model in the
public library, and target encoding provably cannot see them — "everything ending in .2
shares something" pools across integer parts, and TE's levels do not.

### New members found: two more public OOF libraries on the same frozen folds

The community standardised on `StratifiedKFold(5, shuffle=True, random_state=42)`, so
several people publish stackable OOF. Pulled and wired into `data/ext_members/`:

- `raykkretzschmar/s6e8-fm-lattice-blend-members` — 5 factorization machines
  (0.96455–0.96739). A bilinear function class absent from the 74-member library.
- `dariushafshar/s6e8-golem-oof-library` — 7 models incl. a spline GAM (0.93438).

Measured with `experiments/member_value.py` (paired 50/50 splits, 6 reps — split noise is
~0.0002, an order of magnitude larger than these effects, so unpaired comparison would be
worthless):

| member set | paired delta vs 74-lib |
|---|---|
| + FM (5) | +0.000005 [consistent] |
| + golem_clean (5) | +0.000009 [consistent] |
| + golem **all** (incl. `golem_a`, `golem_f`) | +0.000007 **[SIGN FLIPS]** |
| + FM + golem_clean | +0.000014 [consistent] |
| + FM + golem_clean + my 2 | **+0.000016 [consistent]** |
| my 2 members alone, on the 74-lib | +0.000003 [consistent] |

`golem_a`/`golem_f` early-stop on the held-out validation fold — their author discloses
this as mildly optimistic. Including them makes the gain *worse* and flips its sign across
splits. **Measured out, not assumed out**; they are now in `DEFAULT_DROP` in `stack.py`
with the numbers in the comment.

### Submitted (2 of 10 today; cap is 10, 8 remaining)

| # | entry | CV (cross-fitted) | public LB | offset |
|---|---|---|---|---|
| 1 | `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| 2 | `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |

**A +0.000019 CV gain produced exactly zero LB movement.** Two independent calibration
points now agree the offset is **+0.00115 to +0.00117**, close to the +0.0013 in the
public record. Rank ~155/1331, top 11.6%; bronze cutoff is 0.97084, three ten-thousandths
above us.

`lgbm_tuned_lat_frac` earns the **5th-largest stacker coefficient** (0.1104) out of 88
members, behind only `lookup`, `pub_rmlp`, `latr1_xgb` and `tabm_deeper` — so the stacker
does value it, even though its marginal contribution to AUC is ~0.000002.

### The thing next run most needs to internalise: THE STACK IS SATURATED

This is the most important number of the day. Adding, on top of the 74-model library:

- a genuinely new **function class** (factorization machines) → +0.000005
- a genuinely new **channel** (the decimal lattice, inside a strong new GBDT) → +0.000002
- 12 further members from two independent authors → +0.000016 **total**

...while the gap from our 0.97081 to the #1 public 0.97120 is **+0.00039**, i.e. more
than twenty times everything the entire day of member-hunting produced.

**So the leaders are not doing what we are doing.** More members from the same public pool
will not close this. Do not spend another run adding GBDTs to the blend — the marginal
value is provably ~1e-6 and the FM author independently measured the same saturation
("once a function class is represented, strengthening it does nothing").

Two candidate explanations, and they call for opposite responses:
1. The leaders have private models or blend public *test-only* submission files (no OOF,
   so no honest weighting available to us).
2. **Part of that 0.0004 is public-LB overfitting.** ~1,331 teams selecting on a public
   slice, many with 50+ submissions (rank 2 has 51, rank 3 has 58). The private split
   will reshuffle. This account has been burned exactly this way before on
   `rogii-wellbore-geology-prediction`.

Given (2), chasing the public gap by tuning against LB feedback is precisely the failure
mode to avoid. Our CV 0.969660 is honest and our offset is measured twice.

### Next run, in this order

1. **Do not** re-run member-hunting or LightGBM tuning. Both are measured dead ends this
   week; the numbers are above and in `RESEARCH.md`.
2. Hunt a **new channel**, which is the only thing that has ever moved this dataset. The
   generator-forensics direction is the live one: the decimal lattice was found by asking
   what the generator *did*, not what the features mean. Look for further quantisation
   fingerprints — joint rounding, value-frequency structure, ordering artefacts in `id`.
   Check `pub_ryota` ("generator forensics") in the library for prior art.
3. Consider a **non-linear meta-model** (shallow GBM on the 88 logits, or regime-aware
   stacking on missingness count). Untested by us. The per-missingness breakdown printed
   by `run_lgbm.py` shows AUC falling 0.9754 → 0.9111 from 0 to ≥5 missing columns, so the
   members' relative strengths plausibly vary by regime. The library author measured
   regime-aware stacking at only +0.00004, so keep expectations low and judge it paired.
4. When choosing final submissions at the deadline: **select on CV**. Both entries so far
   sit at 0.97081 public; do not let public-LB ties or micro-differences drive the pick.
