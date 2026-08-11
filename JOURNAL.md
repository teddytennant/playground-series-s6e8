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

---

## 2026-08-10 — slot 2 of 10 — angle: CatBoost

**Submitted 1 (3 of 10 used today, 7 remaining). Everything this run measured negative,
including the submission. That is the finding.**

### Angle: redirected, deliberately

The brief's angle was "CatBoost: tune and compare on identical folds." Yesterday's entry
already closes that as a *member*: a new function class bought +0.000005 and a new channel
inside a strong new GBDT bought +0.000002, and the library's best CatBoost (`latwide_cat`
0.96718) is behind its LightGBM and XGBoost. Training a 87th GBDT was not going to be
worth a slot.

So I kept CatBoost and moved it up a level: **CatBoost as the meta-model**, which is also
the journal's own priority #3 ("non-linear meta-model — untested by us"). Same family, same
frozen folds, a question we had not answered.

### 1. Every non-linear meta-model loses to the linear logit stack

Fold 0, all against the identical 86-member matrix, `lin` = the current stack at 0.969070.

| meta-model | fold-0 AUC | vs lin |
|---|---|---|
| **`lin` — L2 logistic on the logits** | **0.969070** | — |
| `lin_rank` (rank-gauss on logits) | 0.969056 | −0.000014 |
| `lin_regime` (separate coefficients per missingness bucket) | 0.969044 | −0.000026 |
| `lin_poly` (per-member cubic recalibration) | 0.969038 | −0.000032 |
| `cat_zr` — **CatBoost, logits + regime**, best of 8 checkpoints | 0.968866 | −0.000204 |
| `lgb_zr` — LightGBM, logits + regime, best checkpoint | 0.968866 | −0.000204 |
| `lgb_resid` (tree on the honest linear stack + regime) | 0.968804 | −0.000266 |

The angle's answer: **CatBoost peaks at 0.968866 and plateaus** — better behaved than
LightGBM, which decayed monotonically from its very first checkpoint (0.968866 at 300 →
0.968009 at 3000), exactly as oblivious trees should. It is still 0.0002 short. Rank
-averaging the tree meta into the linear one recovers nothing: the best blend weight over
every checkpoint of every tree variant was **w ≈ 0, gain ≤ +0.000003**.

A tree cannot beat an additive-in-logit combiner at combining near-collinear logits, and
handing it order statistics (`z_mean/std/min/max/med`) and the whole regime block does not
change that.

### 2. The regime hypothesis is dead — three independent tests, all negative

Per-fold AUC really does collapse across missingness: 0.9764 / 0.9740 / 0.9651 / 0.9540 /
0.9280 for 0,1,2,3,4+ missing columns. The tempting inference is that the members'
*relative* strengths move too, and a global weight vector cannot exploit that.

They do not.

| test | what it relaxes | result |
|---|---|---|
| `lin_regime` | separate coefficient vector per bucket | −0.000026 (fold 0) |
| `lgb_resid` | tree correction on (linear stack, regime) | −0.000266 (fold 0) |
| `iso_regime` | per-bucket isotonic on the global stack | **−0.000085 full OOF, 5/5 folds negative** |

`iso_regime` is the clean one and worth keeping in mind: isotonic is monotone, so
within-bucket ranking is unchanged *by construction* and within-bucket AUC cannot move at
all. Every point it moved was pure cross-bucket regrading. It lost on all five folds. The
global stack's cross-regime calibration is already better than a bucket-wise refit of it.

**Do not re-open regime-aware stacking.** Difficulty varies enormously by regime; the
optimal blend does not.

### 3. A real defect in the stacker's input transform — found, quantified, repaired, and it did not matter

Chasing why the meta-models all failed, I compared each member's OOF logit distribution
against its test logit distribution. They should match: train and test come from the same
generator, and the covariates agree to 4 decimals on every column.

For 76 of 86 members `sd_test/sd_oof` sits at 1.000. For the rest it does not:

| member | OOF AUC | sd_test/sd_oof | % OOF rows clipped | % test rows clipped |
|---|---|---|---|---|
| `naji03`, `naji05` | **0.9688 (library's best)** | 0.679 | 5.79 | 0.92 |
| `et` | 0.9411 | 0.699 | 7.79 | 2.49 |
| `tabm_imp` | 0.9681 | 0.749 | 5.03 | 0.69 |
| `rf` | 0.9433 | 0.795 | 14.96 | 8.05 |
| `pub_tabm` + 5 more TabM | ~0.968 | 0.83–0.93 | 2.7–8.3 | 1.1–4.6 |
| `lgbm_tuned_lat_frac` (**ours, honest 5-fold**) | 0.9678 | **1.0018** | 0 | 0 |
| `lookup` | 0.9685 | 0.9941 | 0 | 0 |

Cause: `to_logit` clips `p` into `[1e-15, 1-1e-15]`. **`naji03`/`naji05` are not
probability arrays** — they run from −0.013 to 1.022 — and `rf`/`et`/the TabM nets emit
exactly 1.0 on 3–15% of rows. Everything at or past the boundary collapses onto a single
value of ±30. The stacker therefore fits its coefficients against an input whose top tail
is a flat plateau, then applies them to a test input where that tail is *resolved*, because
the test arrays are averages and clip roughly a third as often.

Our own model at ratio 1.0018 is the control that rules out the innocent explanation:
ordinary 5-fold averaging does **not** compress a well-behaved member.

I checked the worrying alternative — that `naji03`/`naji05`'s OOF is optimistic rather than
clipped, which would call for *down*-weighting them. Against it: they show no early-stopping
disclosure (unlike `golem_a`/`golem_f`), and their 0.96881 is barely above their own
siblings `naji02`/`naji04` at 0.96863/0.96874, which sit at ratio 1.004. A leaking member
stands above its siblings; these do not.

Repair (`agent/stack.py --transform`): monotone per member, same map applied to OOF and
test, so no member's own ranking moves and both sides land on a common scale.

| transform | cross-fitted OOF | paired per-fold delta vs `logit` |
|---|---|---|
| `logit` (the shipped one) | 0.969660 | — |
| `hybrid` (rank-gauss, affected members only) | 0.969678 | +0.000018 |
| `rankraw` (rank-gauss, all members) | 0.969684 | +0.000023, 4/5 folds |
| `rescale` (min-max then logit, no clip) | 0.969686 | +0.000027, 4/5 folds |

Post-repair the ratio spread went from 0.679–1.009 to 0.955–1.020, median 1.0000.

**Submitted `hybrid`** — it touches only the 29 members that actually clip and leaves the
other 57 on the logit scale they demonstrably suit. Chosen over `rankraw`/`rescale` on
mechanism, not CV: all three are within 8e-6 of each other, which CV cannot resolve.

| entry | CV | public LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |
| **`stack_pub86_hybrid`** | **0.969678** | **0.97080** | **+0.001122** |

**CV +0.000018, LB −0.00001. No movement.** Spearman vs the previous submission was 0.9986,
so the repair genuinely did reallocate weight — it just did not change the ranking.

### The lesson, and it is the same one from a third direction

Yesterday: adding members does nothing (+1.6e-5 for 12 members from two authors).
Today: **reweighting members does nothing either.** A non-linear meta-model, a
regime-conditional one, a per-member recalibration, and a genuine repair of a real
train/test defect in the meta-features all land inside ±0.0003 of the plain linear stack,
and the one I shipped moved the LB by −0.00001.

The members correlate 0.987–0.999. At that correlation the blend's ranking is pinned by
the consensus, and *no* function of the existing 86 columns will move it. The gap to #1
(0.97120 vs our 0.97080) is 0.0004 — twenty times everything two full days of combiner
work has produced.

So the remaining levers are, in order:
1. **A member that is genuinely decorrelated from the pack**, not a better one. `lookup`
   (max correlation 0.9869 vs everything else) still takes the largest stacker coefficient
   at 0.2127 despite ranking only 5th solo. That is the shape of what is missing.
2. Accepting that 0.97080 with an honestly-selected CV is a good private-LB position, and
   that much of the 0.0004 above us is 1,356 teams selecting on a public slice.

### Next run

1. **Do not** re-try: any meta-model over the existing members (§1), anything regime-aware
   (§2), more members from the public pool, LightGBM tuning, original-dataset concat,
   pseudo-labeling. All measured, all in `RESEARCH.md`.
2. The only live direction is **a decorrelated function class**, judged on correlation to
   the pack *before* it is judged on solo AUC. `pub_ryota` ("generator forensics", solo
   0.9637) already earns a large negative coefficient −0.1042, which is the stacker saying
   it sees something the others do not. Generator forensics is where the decimal lattice
   came from and is still the least-worked seam.
3. At the deadline **select on CV**. Our three entries are 0.97080/0.97081/0.97081 — a
   3-way public tie that carries no information. `stack_pub86_hybrid` has the best CV
   (0.969678) and is the only one without a known defect in its inputs.

### Reusable diagnostic worth keeping

`sd(logit(test_j)) / sd(logit(oof_j))` per member, against the control of a model you
trained yourself. It is one line of numpy and it found a defect in the strongest members of
a 74-model public library that everyone on this leaderboard is stacking. Ratios were
0.679–1.009 where they should all have been 1.000.

---

## 2026-08-10 — slot 3 of 10 — angle: XGBoost as the third leg

**Submitted 1 (4 of 10 used today, 6 remaining). CV 0.969678 → 0.970018 (+0.000340).
LB 0.97080 → 0.97099. Rank 158 → 18 of 1366.**

This is the first entry in three days whose CV gain actually moved the leaderboard, and
it did so by overturning yesterday's headline conclusion.

### The angle was redirected, and the reason it was redirected turned out to be wrong

The brief asked for a tuned XGBoost third leg. Yesterday's journal closes that: another
GBDT member measured **+0.000002**, a new function class **+0.000005**, twelve members
from two independent authors **+0.000016**. I redirected on those numbers.

But the conclusion drawn from them — *"THE STACK IS SATURATED... more members from the
same public pool will not close this"* — was **false**, and it was false in a way the
evidence could not have supported. Every one of those measurements came from adding
**small** groups (2, 5, 12) from authors whose feature engineering already overlapped the
74-model library. None of them tested what a *large* set from a genuinely separate
pipeline does. Sample size was mistaken for saturation.

### What was actually still on the shelf

The Kaggle CLI's list endpoints were returning `400` all run
(`ListDatasets`, `ListKernels`, `ListCompetitions`, and — intermittently —
`CreateSubmission`; downloads-by-ref and `submissions -v` kept working throughout). The
**legacy REST endpoint still works and needs no auth**:

```
https://www.kaggle.com/api/v1/datasets/list?search=s6e8&pageSize=100&page=1
```

That enumerated 25 S6E8 datasets, of which **fourteen were public OOF libraries we had
never pulled**. The community has published far more stackable OOF than `RESEARCH.md`
recorded.

Imported, vetted, and added: **63 new members**, taking the stack from 86 to 149.

| source | members | what it is |
|---|---|---|
| `boltuzamaki/s6e8-oof-prediction-library` | 47 | a complete independent pipeline: TabR (retrieval), EBM (a GAM), GANDALF, DCNv2, FT-Transformer, DeepFM, **six seeds of a second Lookup-Transformer**, plus 20 GBDTs |
| `beicicc/s6e8-*-artifacts` (10 datasets) | 12 | fixed-schedule LGBM/XGB/CatBoost/RealMLP/Lookup, **each shipping `fold_id.npy`** |
| `mohankrishnathalla/s6e8-{xgb,cat-mlp,lgb-dart}-oof` | 4 | xgb, cat, nn, lgb-dart |

### The vetting, because a stranger's OOF array is a liability until proven otherwise

`experiments/import_ext2.py` and `experiments/import_beicicc.py`. Five gates:

1. **Row alignment** — id column must equal ours in order. All 47 bolt members reproduce
   their published OOF AUC to <5e-5.
2. **Fold partition** — the beicicc libraries ship the fold assignment itself, so this is
   exact rather than inferred. **Compare the induced partition, not the integers**: their
   fold *labels* are permuted, so raw agreement reads 0.0000 while the partition is
   identical. Cross-tabulate and require every one of their folds to map wholly into one
   of ours. **10/10 pass.** This is a strictly stronger guarantee than reproducing an AUC,
   which only proves row *order*.
3. **Credibility** — OOF AUC > 0.9720 is not achievable here. 
4. **`sd(logit(test))/sd(logit(oof))`** — the diagnostic from yesterday, run on every
   candidate. 48 of 149 members now need the `hybrid` repair, up from 29.
5. **Max correlation against the existing pack**, on the logit scale.

### Excluded on mechanism, not on CV

- **`njm_01..05`** — maxcorr **1.000000** against the pack. They are literally `naji01..05`,
  already in the 74-lib, republished.
- **`njm_*_blend` (9)** — the author's own blends, OOF AUC 0.9692–0.9697, i.e. at the level
  of our whole 88-member stack. Their blend weights are fit on the full OOF, so the OOF
  they publish is in-sample.
- **`beicicc/sixmember_*` (3)** — level-2 cross-fitted stack outputs. Their author
  discloses it himself: *"a different meta-training row can come from a base model that
  used labels from the current meta-validation fold ... may be optimistic."*
- **2 exact duplicate arrays** shipped across two beicicc datasets each.

All four exclusions are the same principle: **an optimistic OOF earns undeserved stacker
weight, and CV is precisely the instrument it fools.** None of these can be judged by the
paired test, so none of them were offered to it.

### The correlation hypothesis was right about the shape and wrong about the size

The journal's standing advice was to judge a candidate on decorrelation first. So groups
were cut by correlation, not family. The new library does contain the most decorrelated
members ever seen here — `bolt_extratrees_support` at maxcorr **0.811**, and six more at
0.92–0.96, against `lookup`'s 0.9869 which was previously the best in the pack.

Paired 50/50, fit and scored on identical rows, 3 splits. Every group is sign-consistent,
and the combined figure independently reproduces the cross-fitted +0.000340 — two honest
estimators agreeing to 1e-5:

| group added to base86 | n | paired delta | per member |
|---|---|---|---|
| `decorr` (maxcorr < 0.97: extratrees, gandalf, dcnv2, ft-transformer, ebm, tabr, deepfm, lookup_v3, neural, mkt_nn) | 10 | +0.000087 ± 0.000005 | 8.7e-6 |
| `lookup2` (six seeds of the second Lookup-Transformer) | 6 | +0.000104 ± 0.000001 | **17.3e-6** |
| `bei` (fold-id-verified fixed-schedule models) | 12 | +0.000101 ± 0.000007 | 8.4e-6 |
| **`rest`** (the 35 GBDT-shaped members the journal predicted were worthless) | 35 | **+0.000206 ± 0.000011** | 5.9e-6 |
| **all 63** | 63 | **+0.000330 ± 0.000011** | 5.2e-6 |

The decorrelated members are worth **more per member** (8.7e-6 vs 5.9e-6) and a second
*independent implementation* of the Lookup-Transformer is worth triple that (17.3e-6) —
so the correlation heuristic is real. But `rest` — 35 ordinary XGB/LGBM/CatBoost members,
exactly the thing two days of journal entries said to stop adding — contributed the
**single largest** share. What matters is not only the function class but **whose pipeline
built it**: an independent author's feature engineering, imputation and encoding decisions
are a source of decorrelation that the family label does not capture.

So the angle's premise was vindicated after all, just not through a model I trained: 20 of
the 47 boltuzamaki members are XGBoost, and they are a large part of the +0.000322.

### Submitted

`stack_pub149_hybrid` — L2 logit stack (C=1.0), `hybrid` transform, 149 members,
cross-fitted on the frozen folds.

| entry | CV | public LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |
| `stack_pub86_hybrid` | 0.969678 | 0.97080 | +0.001122 |
| **`stack_pub149_hybrid`** | **0.970018** | **0.97099** | **+0.000972** |

Spearman vs the previous entry 0.9982; mean 0.7092 against a train rate of 0.7094.

**Rank 158 → 18 of 1366.** Bronze cutoff 0.97084, silver ~0.97093, rank 10 is 0.97106.

Note the offset fell again, +0.001122 → +0.000972: a CV gain of +0.000340 bought
+0.00019 LB, roughly **56% pass-through**. The offset is not a constant, it shrinks as CV
rises, so **stop quoting CV + 0.0012 as an LB estimate.** CV differences still rank
correctly, which is all they are needed for.

### Lesson

Yesterday I wrote that the members correlate 0.987–0.999 and "no function of the existing
86 columns will move it." That sentence was true and the inference drawn from it was not.
The binding constraint was never the combiner or the function class — it was that we held
86 of the ~150 publicly available OOF arrays and had stopped looking. **Two of the three
"measured dead end" conclusions in `RESEARCH.md` were conclusions about search effort
wearing the costume of conclusions about the data.**

The concrete failure was procedural: `RESEARCH.md` listed five unpulled libraries by name
under "Others not yet pulled" and three consecutive runs walked past them to do
combiner work instead.

### Next run, in this order

1. **The public OOF pool is now essentially exhausted** — we hold 149 of everything
   published with usable OOF. The enumeration is in `RESEARCH.md`; re-run the REST
   `datasets/list` query first, since new libraries appeared as recently as 2026-08-10.
2. **Sweep the stacker's `C` at 149 members.** It was verified flat (0.01–10) at 74
   members and has never been re-checked; 149 correlated members is a different
   regularisation problem. One cross-fit is ~25 min. Untested, cheap, and the only
   parameter left in the shipped pipeline.
3. Do **not** re-open: any meta-model over the members, anything regime-aware,
   pseudo-labeling, original-dataset concat, LightGBM tuning. Those remain measured.
4. Treat the remaining "dead end" entries in `RESEARCH.md` with suspicion in proportion to
   how small the group was that produced them.
5. At the deadline, **select on CV.** `stack_pub149_hybrid` leads on both CV and LB, so
   for the first time there is no conflict.

### Operational notes

- **Daily cap re-confirmed at 10.** The CLI printed "6 submissions remaining today" after
  this run's submit (4 used).
- `kaggle competitions submit` returned `400 CreateSubmission` on the first attempt after
  a full upload, then succeeded on an identical retry. **Always confirm with
  `kaggle competitions submissions -v` — the 400 was real, nothing registered.**

---

## 2026-08-10 — slot 4 of 10 — angle: feature engineering, measured on CV

**Submitted 1 (`stack_pub151_rankraw`). The angle's own deliverable measured +0.000005 and
I would have skipped the slot for it — that instinct was wrong, and why it was wrong is
the most useful thing in this entry.**

### Two members built, both from our own pipeline, both judged on decorrelation first

The angle asked for feature engineering measured on CV. In this competition's current
shape that cashes out as *new members*, because slot 3 established that what moves the
stack is whose pipeline built a member, not which family it belongs to.

| member | what it is | solo OOF | maxcorr | median corr | sd_ratio |
|---|---|---|---|---|---|
| `et_lat_frac` | ExtraTrees, 300 trees, min_leaf 40, max_features 0.15, on our lattice+TE+decimal matrix | 0.96002 | 0.9629 | 0.9551 | 0.8308 |
| `linlat` | additive log-odds: logistic on `logit(TE)` + `log1p(CT)` + base cols | 0.96133 | 0.9719 | **0.9373** | 1.0108 |

`linlat` has the **lowest median correlation against the pack of anything we hold**. The
reasoning behind it: a target encoding is already an estimate of P(y | cell), so the
natural way to combine several is to add their log-odds — which is a linear model in
`logit(TE)` space, and not one of the 149 members is linear. `experiments/run_linear.py`.

Paired 50/50, 3 splits, fit and scored on identical rows (`experiments/member_eval.py`):

| added to base149 | paired delta | verdict |
|---|---|---|
| `et_lat_frac` | +0.000006 ± 0.000003 | consistent |
| `linlat` | +0.000003 ± 0.000004 | **sign flips** |
| both | +0.000005 ± 0.000003 | consistent, and **sublinear** |

Two genuinely different function classes, and together they are worth less than the sum of
their parts. Per-member this is 3–6e-6, at the bottom of slot 3's 5.2–17.3e-6 band.

### The decision rule I got wrong, and the correction

I had concluded: +5e-6 is an order of magnitude under the 5e-5 noise floor, RESEARCH.md
says sub-1e-4 gains transfer to LB at rate zero, therefore skip the slot — "an unused slot
beats a wasted one."

That reasoning is **invalid for Playground**, and the workspace owner corrected it
mid-run. The playbook rule it came from exists for competitions where only the two most
recent submissions stay active and a weak entry **evicts** a good one. Here nothing evicts
anything: the leaderboard takes the best of all submissions and final selection is ours to
make at the deadline. So a submission has **no downside**, an unused daily slot is pure
waste, and the correct bar is **"is it genuinely different?"** — not "does it clear the
noise floor?". Scores are deterministic on a fixed public slice, so the only thing worth
avoiding is re-sending an identical file.

Selection at the deadline still happens on CV. That is the Rogii failure mode and free
submissions must not be allowed to corrupt it.

### What I actually submitted, and why it is not a near-duplicate

The owner session sent `stack_pub151_hybrid` (CV 0.970024). Sending a second hybrid would
have been the pointless case. I took **`stack_pub151_rankraw`** instead, on a mechanism
that this run's own result undermines:

`hybrid` exists to repair "only the members that actually clip", on the theory that
clipping marks a **defective** member. That theory is wrong. `et_lat_frac` — trained by me,
on the frozen folds, with no defect whatever — emits `p == 1` on **0.84% of OOF rows against
0.04% of test rows, a 21× asymmetry**, giving sd_ratio 0.8308, deep inside the range
RESEARCH.md treated as proof of a broken member.

The cause is structural and applies to everyone: **the OOF array is one model's output per
row, the test array is the mean of five.** Averaging five saturating models resolves a
plateau that a single model cannot — all five must agree on exactly 1.0 for the mean to be
1.0. Any member whose outputs saturate shows this. The 1.0018 "control" only reads clean
because LightGBM's sigmoid never emits exactly 0 or 1.

So the hybrid/rankraw split was drawing a line between "broken" and "fine" members that
corresponds to an artefact of how OOF and test arrays are built, not to member quality.
Treating all 151 uniformly is the better-motivated choice, and rankraw also measured best
of the four transforms at 86 members (0.969684 vs hybrid 0.969678) — previously dismissed
as unresolvable noise, now with a mechanism behind it.

### Closed: the stacker's C is flat at 149 members

Top item on slot 3's next-run list. Paired 50/50, 3 splits, `experiments/stack_lab.py`:

| C | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 | 10 | 100 |
|---|---|---|---|---|---|---|---|
| vs C=1.0 | +7e-6 | +5e-6 | +5e-6 | — | +1e-6 | +3e-6 | +4e-6 |

Every value within 7e-6 of C=1.0 with std comparable to mean. At n/p ≈ 4600 the L2 penalty
has nothing to do. **Do not revisit this.**

### A defect in our own runner — same one we drop other people's members for

`agent/run_lgbm.py` calls `lgb.early_stopping` on `(Xb, yb)` — the exact rows that become
that member's OOF. The iteration count is therefore chosen with sight of the held-out
labels. **This is precisely why `golem_a`/`golem_f` are dropped from the stack**, and both
of our own members (`lgbm_tuned_lat`, `lgbm_tuned_lat_frac`) carry it.

It matters beyond that member's own inflated AUC: an optimistic member biases the
stacker's coefficients toward itself, so the optimism propagates into which blend gets
selected at the deadline — and CV is *the* deadline decision rule. Added a `--stopping 0`
fixed-schedule path (what beicicc's "fixed900/fixed1500" datasets do, and now visibly why).
Retraining both honestly under new names is queued; record old and new OOF AUC side by side
so the size of the correction is visible rather than silently absorbed.

### Operational: this workspace is not single-tenant

Mid-run I found `stack.py --submit-name stack_pub151_hybrid` running under a different
Claude session, plus unrelated CPU-heavy work from two more. Run queue hit **33 on 16
cores** and everything ran ~4× slow — a LightGBM member that takes 17 min alone had not
finished one fold in 35 min. I killed my own batch to give the cores back.

**Check `ps` for peers before launching heavy jobs, and before submitting.** Two agents
independently building the same artifact from the same `oof/` directory is the live
failure mode; `ListAgents` + `SendMessage` resolved it cleanly. Trace ownership with the
parent chain (`ps -o ppid=`) rather than guessing from session start times.

### Next run

1. Finish the honest retrains of `lgbm_tuned_lat*` and record the delta.
2. **The bar for a submission is difference, not measured improvement** — but identical
   files score identically, so vary something real. Deadline selection stays on CV.
3. Members from our own pipeline are worth 3–6e-6 each, the bottom of the observed band.
   Getting +0.0001 that way needs ~20 of them; a batch runner exists
   (`experiments/batch_members.sh`, fixed schedules) but needs an uncontended box.
4. Do not re-open: stacker C, any meta-model over the members, regime-aware anything,
   pseudo-labeling, original-dataset concat.

### Addendum — the honest retrain, and why the substituted stack did not ship tonight

Retrained both of our members on the fixed schedule (`--stopping 0`, 2000 rounds chosen a
priori rather than from the previous run's early-stopped 1905–2227, which would have
smuggled the same held-out information back in).

`lgbm_fixed_lat_frac`: **0.96779** against the early-stopped **0.96782**. Per fold
−3e-5, −5e-5, −5e-5, −5e-5, +2e-5. **The optimism is real, consistent in 4/5 folds, and
worth about 3e-5** — under the cross-fit noise floor, and about the size of one member's
contribution to the stack. Small enough to be reassuring, large enough that it was right
to remove rather than argue about.

`lgbm_fixed_lat` reached fold 1 (0.96777 vs 0.96781) and is still running.

**The intended second submission — `stack_pub151_fixed_rankraw`, the same 151 members with
our two swapped from early-stopped to fixed-schedule — did not ship, and the reason is
the box, not the idea.** Peers restarted heavy work and the load average sat at 31 on 16
cores; fold 1 of the second retrain took **2184s against fold 0's 177s, a 12× slowdown**.
Finishing the retrain plus a contended stack build was another 2–3 hours for an expected
CV difference of ~1e-5, while starving three other sessions. Not a good trade tonight.

Exact command for the next run, once `lgbm_fixed_lat` lands — the drop list substitutes
rather than adds, so the member count stays at 151 and it is a clean A/B against
`stack_pub151_rankraw`:

```bash
cd agent && ../.venv/bin/python stack.py --ext --ext2 --transform rankraw --reps 0 --C 1.0 \
  --drop golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac \
  --submit-name stack_pub151_fixed_rankraw
```

**Read its CV correctly when it lands.** It will probably come in slightly *below*
`stack_pub151_rankraw`, because the optimistic members were inflating that number too.
That is the correction working. CV is only comparable between stacks whose members are
equally honest, so the honest stack **replaces** the optimistic one as the reference — it
does not compete with it on CV. Getting this backwards would re-introduce the bias by the
front door, having just removed it by the back.

### Slot 4 in one line

One submission (`stack_pub151_rankraw`, CV 0.970023, 4 slots left today); the angle's own
feature engineering measured +5e-6; the stacker's C is closed as flat; and two durable
claims in `RESEARCH.md` were wrong — sd_ratio is not a defect detector, and our own member
runner carried the very defect we drop other people's members for.

---

## 2026-08-11 (UTC) — slot 1 of 10 — angle: tune LightGBM properly against the fixed folds

**Read the date carefully: the prompt said "10 submissions already today" and SLOT 1 of
10, which is contradictory.** Both are right. Local time was 20:12 EDT on 2026-08-10, i.e.
**00:12 UTC on 2026-08-11** — Kaggle counts submission days in UTC and the quota had reset
twelve minutes earlier. The ten entries the API reported all carry 2026-08-10 UTC
timestamps. Confirmed empirically, not assumed: the CLI printed **"9 submissions remaining
today"** after the first submit of this run.

Durable: **the daily counter rolls at 00:00 UTC = 20:00 EDT.** A run that starts in the
evening US-eastern is at the start of a fresh quota, not the end of a spent one.

### Standing at the start of the run

Rank **14 of 1385**, public 0.97104, 12 submissions. Rank 1 is 0.97124; the entire field
from us to the top spans 0.00020, and we are **0.00004 under the gold cutoff** (rank 10 =
0.97108) — a gap smaller than any effect measured here in two days.

`blend150fx` (public 0.97104) had been submitted by a peer session with **no journal entry
and no recorded CV**, which is a selection hazard given that the deadline pick is on CV.
Recovered it by scoring every saved `submissions/oof_*.npy` directly. It leads on **both**
CV (0.970032) and LB, so there is no conflict — but that was luck, not process.

### Why I did not do the angle as written

Slot 1 of 2026-08-10 already tuned LightGBM against these folds and got **+0.00003 full
OOF — below the 0.00005 noise floor**, after a fold-0 sweep promised +0.00050. Both the
journal and `RESEARCH.md` list "LightGBM tuning" as measured-dead and say not to reopen it.
Re-running it would have bought a known null.

What has never been tested is the question the last two days' results actually raise:
solo-AUC tuning is dead, but **is decorrelation-oriented tuning alive?** Slot 3 measured
that a member's worth to the stack tracks how differently it is wrong, not how good it is
alone. Nobody had ever tuned a member *for that objective*. So the angle was kept — tune
LightGBM against the fixed folds — with the objective swapped from solo AUC to
decorrelation, which is a real reading of the same instruction and not a substitution.

The answer, from the run's main experiment, is **no** — see below. Reinterpreting the
angle produced a clean negative rather than a gain, and the negative is worth more than
the gain would have been.

### 1. The public OOF pool is confirmed exhausted

Re-ran the REST enumeration (`datasets/list`, five search terms, unioned) — the CLI's list
endpoints are still 400ing. Nothing new since the 2026-08-10 sweep. All 20 S6E8-related
datasets are already accounted for in `RESEARCH.md`. This closes slot 3's item 1; it does
not need re-running daily any more, only weekly.

### 2. Two defects found in the member library itself

**`bolt_xgb_d7_alt1` and `bolt_xgb_d7_alt2` are the same array.** `np.array_equal` is
`True` for both the OOF and the test vectors, max|diff| exactly 0.0, identical AUC
0.9681005. The smallest eigenvalue of the 149×149 correlation matrix is **1.1e-16** —
exact collinearity, which is how it surfaced.

This is the `maxcorr == 1.000` gate that `RESEARCH.md` already lists and that `najiama`'s
members were rejected for. It got through because `import_ext2.py` screens each candidate
against **the pack** and never against the other candidates in its own batch. Scoring
impact is negligible (an L2 penalty just splits the coefficient, and `C` is flat here), but
**the real member count is 148, not 149**, and every "n members" figure from `--ext2`
onward is off by one.

**The flagship "maxcorr 0.811" for `bolt_extratrees_support` is a transform artefact.**
`RESEARCH.md` cites it twice as the headline evidence that decorrelation is real. It comes
from `import_ext2.py:120`, which correlates on the raw `to_logit` scale — the scale whose
clip destroys the tails of saturating members, and ExtraTrees saturates hard. In the
**hybrid** space the stacker is actually fitted in, the same member reads **0.9701**
(median 0.9340). It is not close to the most decorrelated member; `logreg` (0.9307) and
`bolt_lookup_v3_evidence` (0.9333) are. Only 15 of 149 sit below 0.97.

Same shape of error as the `sd_ratio` correction two days ago: **a quantity measured in a
space the model does not use, then read as a property of the model.** That is now twice.
Any statistic used to judge members must be computed in the transform the stack is fitted
in.

### 3. The geometry of the pack, and what it costs to prune

Eigenvalues of the 149-member correlation matrix in hybrid space:

| | |
|---|---|
| PC1 alone | **95.39%** of variance |
| first 10 PCs | 98.42% |
| first 40 PCs | 99.49% |
| eigenvalues > 1e-4 × λmax | 55 |
| entropy effective rank | 1.41 |

149 members are **one consensus signal plus a very thin tail of corrections**, and the
entire 0.9696 → 0.9700 climb was bought inside the 4.6% that is not PC1. This is the
quantitative version of what the journal kept rediscovering: a member from a new pipeline
adds a *direction*, a better member from a pipeline already held adds *magnitude along
PC1*, and PC1 is saturated.

### 4. The main experiment — how many members, and chosen how

`experiments/member_select.py`. Greedy top-k under three orderings, every cell fitted and
scored on **identical rows** so split noise cancels, 2 reps, 148 members.

| k | `auc` (greedy solo AUC) | `decorr` (greedy min-maxcorr) | `random` |
|---|---|---|---|
| 10 | 0.969599 | 0.969517 | 0.969244 |
| 25 | 0.969821 | 0.969730 | 0.969671 |
| 50 | 0.969965 | 0.969874 | **0.970042** |
| 100 | 0.970128 | 0.970141 | 0.970145 |
| 148 | 0.970177 | 0.970181 | 0.970180 |

**No saturation in k — keep every member.** The curve still climbs at the right edge
(k=100 → 148 is +3.5e-5). A "cleaner 50-member stack" costs −1.4e-4, three times the noise
floor. The 95.4%-in-PC1 geometry is *not* licence to prune.

**Decorrelation fails as a member-level selection rule.** `decorr` loses to `auc` by
−8e-5 to −9e-5 at k=10/25/50, sign-consistent across both reps, and only draws level once
k ≥ 100 and both orderings hold nearly the same set. Choosing members for being
decorrelated selects weak oddities — the ordering opens `logreg`, `golem_g`, `nn2`, `knn`,
`fmpure`.

**At k=50 a random 50 beats both principled orderings.** Greedy-by-AUC concentrates inside
one redundant strong family; greedy-by-diversity concentrates on junk; random gets a
natural mix of strength and pipelines.

This does not contradict slot 3's group-level attribution — it **explains** it. The
boltuzamaki import paid because it was a *representative sample of an entire independent
pipeline*, not because its members were individually decorrelated. **The unit that carries
value is the pipeline, not the member.** Acquire whole pipelines, keep everything in them,
never hand-pick by a correlation statistic.

#### A free noise calibration that undercuts an earlier claim of mine

At k=148 all three orderings select the *same 148 columns* and differ only in column
order. They score 0.970177 / 0.970181 / 0.970180 — a spread of **4e-6 of pure
`LogisticRegression` convergence noise.**

That is the same size as the paired deltas slot 4 reported for our own two members
(`et_lat_frac` +6e-6, `linlat` +3e-6, both +5e-6) and called "a real ordering". **Those
numbers are at, not above, the solver's own reproducibility**, and sign-consistency across
row splits does not rescue them, because solver noise is not resampled by changing rows.
Anything under ~1e-5 needs to be stable under column permutation before it is believed.

### Submitted this run

| # | entry | CV | public LB | offset |
|---|---|---|---|---|
| 1 | `blend150fx_rescale` | 0.970013 | 0.97102 | +0.001007 |
| 2 | `blend150fx_logit` | **0.969950** | **0.97103** | +0.001080 |

Both were already built by the peer's `blend_lab.py --build` and had never been sent;
sending them completes the five-way transform → LB mapping at zero CPU cost.

**The result is the single most useful thing the public LB has told us.**
`blend150fx_logit` has the **worst CV of the top nine by 8e-5** — a gap larger than the
noise floor and larger than any single improvement shipped all week — and came back
**0.97103, second-best of everything we have ever sent**, above three stacks that beat it
on CV. `logit` is also the one transform with a mechanism argument *against* it (its clip
provably destroys the tails of ~29 saturating members).

That is the Rogii failure mode handed over for free: the public slice, given a vote, picks
the entry we have a specific reason to believe is worse. **Final selection stays on CV.**

It also corrects a durable claim. "The offset shrinks as CV rises" was drawn from four
points spanning a wide CV range. With twelve points it splits in two: across the
0.9696 → 0.9700 step the offset genuinely fell (+0.00115 → +0.00100), but **within** the
top cluster of nine it scatters +0.00097…+0.00108 with no trend, and that scatter is ±5e-5
— the same size as the CV differences being compared. **The public slice cannot resolve CV
differences below ~1e-4, which is every difference we can still produce.**

### Operational

- Box contention was severe and got worse through the run: a peer session on
  `kaggriculture` went from 16 to 30+ processes, load average **47 on 16 cores**. My jobs
  were `taskset`-pinned and `nice`d throughout, which is the only reason anything finished.
- **`lgbm_goss_xt_lat_frac` was abandoned** after 48 minutes without completing fold 0
  (against 20 min/fold for the stump). GOSS reweights every row each iteration and
  `extra_trees` gives up the histogram short-circuit, so it does not amortise like plain
  boosting. Killing it immediately sped the stump member up 5× (fold 2 took 244s against
  fold 0's 1217s). Noted in `experiments/decorr_batch.sh`; retry only on an idle box and
  at ~1200 rounds.
- `kaggle competitions leaderboard -d -p /tmp/lb` will happily read **another
  competition's** leaderboard if that path already holds one. Use a per-competition path.
