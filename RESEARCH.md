# Research — playground-series-s6e8

Durable facts. Anything learned once goes here so no later run pays for it twice.

## Competition basics

| | |
|---|---|
| Slug | `playground-series-s6e8` |
| Task | binary classification, predict `addicted_label` for each `id` |
| **Metric** | **ROC AUC** (confirmed: every public notebook titles scores ~0.96-0.97 AUC; ranking, not calibration) |
| Deadline | 2026-08-31 23:59 |
| Teams | ~1,326 |
| Reward | swag / medals |
| Train | 691,369 rows × 14 cols |
| Test | 296,302 rows |
| Base rate | 0.7094243450313797 (matches the constant in `sample_submission.csv`) |

Submit **continuous probabilities**, never thresholded — it is AUC.

### Submission command (verified working)

```bash
kaggle competitions submit -c playground-series-s6e8 -f <file.csv> -m "<message>"
kaggle competitions submissions -c playground-series-s6e8 -v | head -5
```

Submission format: `id,addicted_label` with a float in `addicted_label`.

**Daily cap: 10** — confirmed twice on 2026-08-10 ("9 submissions remaining today" after
the 1st, "6 submissions remaining today" after the 4th). This is NOT the usual Playground
5; the brief's guess of 5 was wrong. Re-confirm from the CLI line each run.

⚠ **`submit` can return `400 CreateSubmission` after a successful 100% upload, and
nothing registers.** Seen 2026-08-10 alongside the CLI's list endpoints all returning
400. An identical retry succeeded immediately. **Always confirm with
`kaggle competitions submissions -v | head -3` rather than trusting the exit status** —
and check before re-sending, so a retry does not burn a second slot on a submission that
did land.

## The data

12 predictors: 9 numeric + 3 categorical.

```
NUM = age, daily_screen_time_hours, social_media_hours, gaming_hours,
      work_study_hours, sleep_hours, notifications_per_day,
      app_opens_per_day, weekend_screen_time
CAT = gender (Male/Female/Other), stress_level (High/Low/Medium),
      academic_work_impact (Yes/No)
```

**Every column is 4–20% missing** in both train and test. Missingness looks MCAR
(NA-indicator features measured −0.00001).

### Two generator artefacts that matter more than anything else

1. **The accounting identity.** `daily_screen_time_hours >= social + gaming + work_study`
   holds with **zero violations** in competition data, and is violated in **60.7%** of
   the real source dataset. The generator manufactured it. Exploit it (we are scored on
   the synthetic distribution) but do not explain it in real-world terms.
   - Consequence: observed component sum is a hard LOWER bound on a missing `daily`;
     the slack is a hard UPPER bound on a missing component. Better than any statistical guess.
   - Also causes a Simpson's reversal: `work_study_hours` looks harmful marginally, is
     protective conditional on total screen time. It is pure arithmetic, not a real effect
     (in the source data `work_study_hours` has AUC 0.5007 — pure noise).

2. **The quantisation lattice.** Values were drawn then rounded, so each numeric column
   is a lattice of a few thousand repeated values (`daily_screen_time_hours`: 1,389
   distinct values over 691k rows ≈ 500 rows/level). The **first decimal digit** of
   `daily_screen_time_hours` swings the addiction rate by **8.5 points**. That is a
   fingerprint of how the numbers were written, not of behaviour.

## What works (measured deltas, from the public ablation record)

| idea | delta | verdict |
|---|---|---|
| **target + frequency encoding, ALL columns at full resolution** | **+0.0023 CV / +0.0017 LB** | ✅ the single biggest win |
| imputed columns *alongside* the NaNs (never replacing) | +0.0012 | ✅ |
| decimal lattice (`frac`, first digit) | +0.0001 | ✅ a genuinely new channel |
| logit stacking over the library | +0.0004 | ✅ |
| composition / ratio features | +0.0005 | ✅ (largely superseded by TE) |
| lower learning rate | +0.0002 | ✅ |
| 3-seed fold averaging | +0.0002 CV / +0.0001 LB | ✅ small |

### What does NOT work — do not re-spend runs on these

| idea | delta | note |
|---|---|---|
| **pseudo-labeling confident test rows** | **−0.0034** | worst of the lot |
| naive mean of a 12-model library | −0.0012 | cost scales with the worst member |
| tree depth 9–13 | to −0.0011 | |
| pairwise TE (2 parameterisations) | −0.0004 / −0.0001 | |
| multi-resolution TE | −0.0003 | |
| monotone constraints on screen columns | −0.0003 | |
| **concatenating the original dataset** | **−0.0001** | see below — the usual Playground edge is DEAD here |
| k-NN target encoding (0.936 standalone!) | +0.00001 | a new *view*, not a new *channel* |
| second decimal digit, integer-ness | +0.00001 | |
| TE smoothing sweep (10/50/200) | ≈0 | smoothing 10–20 is the sweet spot |
| NA-indicator features | −0.00001 | missingness is MCAR |

**The original dataset does not help.** It is
`jayjoshi37/smartphone-usage-and-addiction-prediction`. Concatenating it measured
**−0.00008**. The standard Playground "find the original dataset" edge is worthless here
because the generator invented the structure that the models actually use. Use the
original only as a *diagnostic* for what the generator did.

## The frozen CV scheme — USE THIS, do not change it

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

This is deliberately identical to the public OOF library's scheme (below), so our own
OOF predictions are directly stackable against 74 public models with no leakage.
`agent/common.py:get_folds` is the single source of truth. If it ever changes, every
saved `oof_*.npy` is invalid and the stack becomes dishonest.

### CV → LB relationship

CV **does not** estimate the LB score; CV **differences** do — and even those transfer at
well under 1:1.

The offset is **not a constant, and it shrinks as CV rises.** Four of our own points:

| entry | CV | LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |
| `stack_pub86_hybrid` | 0.969678 | 0.97080 | +0.001122 |
| `stack_pub149_hybrid` | 0.970018 | 0.97099 | **+0.000972** |

The one gain large enough to measure, +0.000340 CV, transferred as +0.00019 LB — about
**56% pass-through**. Everything smaller than ~0.0001 CV has transferred at a rate of
**zero**. So: **do not quote "CV + 0.0012" as an LB estimate**, and do not expect a CV
gain to arrive intact.

Trust CV for **ranking** decisions. Never as a leaderboard estimate.

### Noise floor

Per-fold spread within one 5-fold run massively overstates uncertainty. The real noise
floor is the std of repeated 5-fold **means** across partition seeds ≈ **0.00005**.
Any claimed gain smaller than that is nothing.

## The public OOF library — the most valuable asset in this competition

`szymonkapiski/s6e8-oof-library-47-models` (actually **74 models**, updated 2026-08-04).

```bash
kaggle datasets download -d szymonkapiski/s6e8-oof-library-47-models -p data/oof --unzip
```

- `oof/oof_<name>.npy` — 691,369 float64, original row order
- `oof/test_<name>.npy` — 296,302 float64, averaged over the 5 fold models
- `manifest.csv` — OOF AUC, family, feature set, notes per model
- `hyperparameters.json`, and **`src/` with the full training code**

All 74 are on the frozen scheme above, so they stack against ours directly.

Best members: `naji03`/`naji05` 0.96881, `tabm_seed3` 0.96867, `lookup` 0.96853,
`pub_rmlp` 0.96844.

**`lookup` is the diversity prize.** A Lookup-Transformer (per-value `nn.Embedding` +
4-layer transformer). Max correlation vs every other member 0.9869, where the rest of the
strong pack sits 0.987–0.999. Adding it alone was worth +0.000109 to the author's blend.
Architecture from `tamerlanomralinov/s6e8-lookup-transformer`.

### Other public OOF libraries on the SAME frozen folds

The community converged on one fold scheme, so several people publish stackable OOF.
Downloaded and symlinked into `data/ext_members/` with collision-proof names.

```bash
kaggle datasets download -d raykkretzschmar/s6e8-fm-lattice-blend-members -p data/ext/... --unzip
kaggle datasets download -d dariushafshar/s6e8-golem-oof-library         -p data/ext/... --unzip
```

- **`raykkretzschmar/s6e8-fm-lattice-blend-members`** — 5 factorization machines
  (`fmplr` 0.96739, `fmnum` 0.96713, `fmdeep` 0.96666, `fmwide` 0.96493, `fmpure` 0.96455).
  A bilinear function class absent from the 74-member library: one vector per
  (field, value), joint cells scoring as inner products, so all pairs are estimated
  jointly and thin lattice cells borrow strength. Author measures the whole family worth
  only **+0.000006** to a blend, and warns that `fmplr`/`fmnum` add **0.000000** on top of
  the other three — once a function class is represented, strengthening it does nothing.
  Also contains 2 band-local members that are NOT full length (not blend members).
- **`dariushafshar/s6e8-golem-oof-library`** — 7 models `a`–`g` (0.9344–0.9649), including
  a **spline GAM** (`c`, 0.934384) which is a genuinely different function class.
  ⚠ Members `a` and `f` early-stop on the held-out validation fold; the author discloses
  this as mildly optimistic and uncorrected. Keep them separable from the rest so their
  effect can be isolated — an optimistic OOF earns undeserved stacker weight.
- `dariushafshar/s6e8-measured-findings-pack` — folds, drift, noise-floor scenarios.

### The rest of the pool — pulled 2026-08-10, +63 members, worth +0.000340 CV

Three runs walked past the "not yet pulled" list above while doing combiner work. They
were worth **twenty times** everything the combiner work produced. `agent/stack.py --ext2`
loads them from `data/ext_members2/`; import + vetting is
`experiments/import_ext2.py` and `experiments/import_beicicc.py`.

- **`boltuzamaki/s6e8-oof-prediction-library`** — **47 members**, an entire independent
  pipeline, `oof_predictions.parquet` + `test_predictions.parquet` + `stream_index.csv`.
  Function classes present nowhere else: **TabR** (retrieval), **EBM** (a GAM), GANDALF,
  DCNv2, FT-Transformer, DeepFM, and **six seeds of a second Lookup-Transformer**
  (`lookup_v2_*`, 0.96830–0.96834). Plus ~20 XGB/LGBM/CatBoost. All 47 reproduce their
  published OOF AUC to <5e-5.
- **`beicicc/s6e8-*-artifacts`** — **10 separate small datasets**, 12 usable members
  (fixed-schedule LGBM / XGB / CatBoost / RealMLP / Lookup, 0.9625–0.9683). Uniquely,
  **every one ships `fold_id.npy`** — the fold assignment itself. Exclude their
  `sixmember_*` arrays: those are level-2 stack outputs and the author discloses the
  optimism himself.
- **`mohankrishnathalla/s6e8-{xgb,cat-mlp,lgb-dart}-oof`** — 4 members (`xgb`, `cat`,
  `nn`, `lgb`); `nn` is decorrelated at maxcorr 0.940.
- **`najiama/predicting-smartphone-addiction-oof-submission-csv`** — ⚠ **do not import.**
  `01..05` are byte-identical to `naji01..05` already in the 74-lib (maxcorr 1.000000);
  `07..17_blend` are the author's own blends whose weights are fit on the full OOF, so
  their published OOF is in-sample (AUC 0.9692–0.9697, i.e. at whole-stack level).
- Submission-only, no OOF, so not honestly weightable:
  `anhadmahajan06/ps-s6e8predicting-smartphone-addiction-submission`.

### Enumerating the pool — the CLI list endpoints are unreliable

As of 2026-08-10 the Kaggle CLI's RPC endpoints return `400` for **all** list operations
(`ListDatasets`, `ListKernels`, `ListCompetitions`) while downloads-by-ref,
`competitions submissions -v` and `competitions leaderboard -d` keep working. The
**legacy REST endpoint still works and needs no authentication**:

```bash
curl -s "https://www.kaggle.com/api/v1/datasets/list?search=s6e8&pageSize=100&page=1"
```

Run it with several search terms and union the `ref` fields — that is how the 14 missed
libraries were found. New ones appeared as recently as 2026-08-10, so re-run it every run.

### Always verify the split before stacking anyone's OOF

Most published S6E8 OOF arrays use a different fold count or average over seeds. Sanity
checks, weakest to strongest:

1. **Published-vs-computed OOF AUC.** Cheap, but proves only that the rows are in the
   right ORDER. It says nothing about which partition was used.
2. **Credibility.** An OOF AUC above ~0.9720 is not achievable here — the best honest
   single member across five independent libraries is 0.96881, and our 149-member stack
   only reaches 0.97002. Anything higher is early stopping on the validation fold, an
   in-sample blend, or a mixed partition.
3. **The fold-id gate — exact, when the library ships `fold_id.npy`.** Compare the induced
   **partition**, not the integers: fold *labels* are permuted between authors, so raw
   agreement reads 0.0000 while the partition is identical. Cross-tabulate their fold id
   against ours and require every one of their folds to map wholly into one of ours.
   All 10 beicicc libraries pass. See `experiments/import_beicicc.py:fold_gate`.

Do not use the CV→LB offset as a leak detector any more — it is not constant (see
"CV → LB relationship" above), so a low offset no longer implies a leaking member.

### Best single models by family (the bar to beat)

| family | best OOF | name |
|---|---|---|
| LightGBM | 0.96768 | `lattri_lgbm` / `latmax_lgbm` |
| XGBoost | 0.96780 | `latr1_xgb` |
| CatBoost | 0.96718 | `latwide_cat` |
| TabM | 0.96867 | `tabm_seed3` |

**Open opening:** the library's LightGBM hyperparameters on the lattice feature set were
**hand-set, never tuned** (`lr 0.035, num_leaves 96, min_child_samples 40, subsample 0.9,
colsample 0.6, reg_lambda 5, max_depth 7`). The only LGBM tuning in `hyperparameters.json`
is `lgbm_tuned`, which was tuned on the far weaker raw+iterative-imputation feature set
and landed in a completely different regime (`depth 4, num_leaves 18`). Tuning LGBM on the
lattice features is genuinely unexplored ground.

## Leaderboard shape

- Top public LB ≈ **0.97092**. Those entries are **stacks of other competitors'
  published prediction files**, not single models.
- An honest single-pipeline, train.csv-only entry lands ≈ **0.9677**.
- The gap between "honest solo" and "top of LB" is ≈ 0.0013 and is essentially all
  ensembling over public predictions.

## Stacking notes

- **Stack on logits**, not probabilities: `clip(log(p/(1-p)), ±30)`. Worth +0.00047 on a
  diverse 12-model library, only +0.00008 on an all-GBM one. This target saturates hard
  (the top screen-time decile has an addiction rate of 1.000, where probabilities have no
  resolution left and logits still do).
- **Hill climbing can only add; a linear stacker can subtract.** Weak-but-decorrelated
  members carry usable information as *corrections* with negative coefficients. If a hill
  climber zeroes a member out, try a linear stacker before discarding it.
- Fitting a stacker on the OOF matrix and scoring it on the same matrix reads high.
  Score it with a proper cross-fit, and compare alternatives with **paired** differences
  on the same row splits so split noise cancels.

## Environment (this machine)

- 16 CPU cores, 31 GB RAM, **no GPU**. `device="cuda"` paths in public code must be
  disabled.
- venv at `.venv` (uv, Python 3.12): pandas, numpy, scikit-learn, lightgbm, xgboost,
  catboost, pyarrow.
- Per-fold design matrices are cached to `cache/` by `experiments/build_cache.py` so
  hyperparameter trials cost only LightGBM time.

## Member arrays are NOT all probabilities — check before stacking

`agent/stack.py --transform` exists because `to_logit`'s clip into `[1e-15, 1-1e-15]`
silently destroys the tails of several members, and destroys them **asymmetrically between
OOF and test**.

- `naji03`, `naji05` (the library's two best, OOF 0.96881) emit values from **−0.013 to
  1.022**. They are not probabilities — most likely blends or regression-objective outputs.
- `rf` (14.96% of OOF rows), `et` (7.79%), `tabm_imp` (5.03%), `pub_tabm` (8.25%) and four
  more TabM nets emit exactly `1.0` on 3–15% of rows.
- Test arrays clip roughly **a third as often** as OOF arrays (naji03: 5.79% → 0.92%),
  because they are averages.

**The diagnostic**, one line and worth running against any new member library:

```python
sd_ratio = to_logit(test_j).std() / to_logit(oof_j).std()   # should be 1.000
```

Control: our own honest 5-fold `lgbm_tuned_lat_frac` sits at **1.0018**. Ordinary 5-fold
averaging does not compress a well-behaved member. Anything at 0.68–0.93 is a broken
transform, not a property of the model. Pre-repair spread across 86 members: 0.679–1.009.

Transforms available (all monotone per member, same map on OOF and test, so no member's
solo AUC changes):

| `--transform` | cross-fitted OOF | note |
|---|---|---|
| `logit` | 0.969660 | original; clips |
| `hybrid` | 0.969678 | rank-gauss on the 29 clipping members only |
| `rankraw` | 0.969684 | rank-gauss on all, ties averaged |
| `rescale` | 0.969686 | min-max over both splits then logit; only *partially* equalises |

Worth ~+0.00002 CV and **0.00000 LB**. Fix it because it is wrong, not because it scores.

## More things that do NOT work — measured here, do not re-spend runs

| idea | delta | note |
|---|---|---|
| CatBoost meta-model on logits + regime | −0.00020 | plateaus at 0.968866; better behaved than LGBM but still short |
| LightGBM meta-model on logits + regime | −0.00020 | decays monotonically from the first checkpoint |
| tree meta on (honest linear stack + regime) | −0.00027 | |
| rank-averaging any tree meta into the linear stack | ≤ +0.000003 | best blend weight is w ≈ 0 |
| **per-bucket coefficients by missingness** | −0.000026 | |
| **per-bucket isotonic on the global stack** | −0.000085, 5/5 folds | monotone, so this is a *pure* cross-regime test |
| per-member cubic recalibration (`lin_poly`) | −0.000032 | |
| rank-gauss on the clipped logits (`lin_rank`) | −0.000014 | rank the RAW values instead — ties matter |

**The linear logit stack is the right combiner. Stop looking for a better one.**

Per-fold AUC by missing-column count: 0.9764 / 0.9740 / 0.9651 / 0.9540 / 0.9280 for
0/1/2/3/4+. Difficulty varies enormously by regime; the optimal *blend* does not.

## What actually moves the stack — corrected 2026-08-10

An earlier version of this section was titled "Why nothing moves any more" and concluded
the stack was saturated. **That was wrong.** It generalised from three *small* member
additions to a claim about the data. Kept here in corrected form because the error is
more instructive than the fact.

| lever | gain | when |
|---|---|---|
| a new channel inside a strong new GBDT | +0.000002 | |
| a new function class (factorization machines), 5 members | +0.000005 | |
| 12 more members, two independent authors | +0.000016 | |
| repairing a real train/test defect in the meta-features | +0.000018 CV, −0.00001 LB | |
| **63 members from three further independent pipelines** | **+0.000340 CV, +0.00019 LB** | slot 3 |

Everything above the last row is a **group of 2–12 members from authors whose feature
engineering already overlapped the 74-model library.** None of them tested a large set
from a separate pipeline. Sample size was mistaken for saturation.

### Attribution of the gain — paired 50/50 on identical rows, 3 splits

`experiments/member_value2.py`. Every group is sign-consistent across all three splits,
and the combined figure independently reproduces the cross-fitted +0.000340.

| group added to the 86-member base | n | paired delta | per member |
|---|---|---|---|
| `decorr` — maxcorr < 0.97 (extratrees 0.811, gandalf, dcnv2, ft-transformer, ebm, tabr, deepfm, lookup_v3, neural, mkt_nn) | 10 | +0.000087 ± 0.000005 | 8.7e-6 |
| `lookup2` — six seeds of a second Lookup-Transformer | 6 | +0.000104 ± 0.000001 | **17.3e-6** |
| `bei` — fold-id-verified fixed-schedule models | 12 | +0.000101 ± 0.000007 | 8.4e-6 |
| `rest` — ordinary XGB/LGBM/CatBoost | 35 | +0.000206 ± 0.000011 | 5.9e-6 |
| **all 63 together** | 63 | **+0.000330 ± 0.000011** | 5.2e-6 |

Two things to carry forward, and they pull in different directions:

1. **Decorrelation is real.** `bolt_extratrees_support` sits at maxcorr **0.811** against
   all 148 others, where the previous diversity prize (`lookup`) was 0.9869. Decorrelated
   members are worth ~1.5× more each than GBDT-shaped ones, and a second *independent
   implementation* of the same idea (`lookup2`) is worth more per member than anything.
2. **But the largest single share came from `rest`** — 35 ordinary GBDTs, the exact thing
   the journal had twice concluded was worthless. **Whose pipeline built a member is a
   source of decorrelation that the family label does not capture.** An independent
   author's imputation, encoding and feature decisions differ even when the model class
   is identical.

Judge candidates on correlation to the pack first and solo AUC second — but do not
*reject* a group for being GBDT-shaped if it comes from a pipeline you do not already
hold. Measure it.
