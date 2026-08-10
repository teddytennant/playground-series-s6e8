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

**Daily cap: 10** — confirmed 2026-08-10. The CLI printed "9 submissions remaining today"
after the first submission of the day. This is NOT the usual Playground 5; the brief's
guess of 5 was wrong. Re-confirm from the CLI line each run.

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

CV **does not** estimate the LB score; CV **differences** do.
Measured offset: LB ≈ CV **+0.0013**, because every test prediction averages 5 fold
models while every OOF prediction comes from one. Two reference points from the public
record: CV 0.966012 → LB 0.96751, and CV 0.966217 → LB 0.96769.

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
