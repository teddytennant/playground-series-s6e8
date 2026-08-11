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
| `colsample_bytree 0.5` on a 12-column frame | −0.0006 | see below — it is a feature-set setting, not a model setting |
| pairwise TE (2 parameterisations) | −0.0004 / −0.0001 | |
| multi-resolution TE | −0.0003 | |
| monotone constraints on screen columns | −0.0003 | |
| **concatenating the original dataset** | **−0.0001** | see below — the usual Playground edge is DEAD here |
| k-NN target encoding (0.936 standalone!) | +0.00001 | a new *view*, not a new *channel* |
| second decimal digit, integer-ness | +0.00001 | |
| TE smoothing sweep (10/50/200) | ≈0 | smoothing 10–20 is the sweet spot |
| NA-indicator features | −0.00001 | missingness is MCAR |

**`colsample_bytree` belongs to the feature set, not to the model.** `0.5` was inherited
unexamined from the library's 184-column lattice matrices. On a 12-column frame it drops
half the predictors from every tree and costs ~0.0006 (inner-holdout 0.96082 plateau
against 0.96140 at `colsample 1.0`, and it starts at AUC 0.849 where the others start at
0.92+). Re-check it whenever the frame width changes.

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

All twelve of our own points, complete as of 2026-08-11:

| entry | CV | LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |
| `stack_pub86_hybrid` | 0.969678 | 0.97080 | +0.001122 |
| `blend150fx_logit` | 0.969950 | 0.97103 | +0.001080 |
| `blend150fx_rescale` | 0.970013 | 0.97102 | +0.001007 |
| `blend150fx_hybrid` | 0.970014 | 0.97099 | +0.000976 |
| `stack_pub149_hybrid` | 0.970018 | 0.97099 | +0.000972 |
| `stack_pub151_rankraw` | 0.970023 | 0.97102 | +0.000997 |
| `stack_pub151_hybrid` | 0.970024 | 0.97099 | +0.000966 |
| `blend150fx_rankraw` | 0.970024 | 0.97102 | +0.000996 |
| `stack_pub151_fixed_rankraw` | 0.970025 | 0.97103 | +0.001005 |
| `blend150fx` | 0.970032 | 0.97104 | +0.001008 |
| `blend150sx_rankraw` | 0.970022 | 0.97102 | +0.000998 |
| `blend150sx` | 0.970033 | 0.97104 | +0.001007 |
| `blend150sx_hybrid` | 0.970016 | 0.97101 | +0.000994 |
| `blend150sx_logit` | 0.969955 | 0.97104 | +0.001085 |
| `blend153` | 0.970034 | 0.97104 | +0.001006 |
| `blend156_rescale` | 0.970025 | 0.97105 | +0.001025 |
| `blend156_rankraw` | 0.970036 | 0.97104 | +0.001004 |
| **`blend156`** | **0.970042** | **0.97106** | +0.001018 |

#### ⚠ "A one-member change is invisible to the public slice" — withdrawn 2026-08-11

That was written after `blend150sx` matched `blend150fx` at 0.97104 and their `rankraw`
variants matched at 0.97102, i.e. from the two transforms where it happened to hold. The
other two were sent the next run and **both moved**: `blend150sx_hybrid` 0.97101 against
`blend150fx_hybrid` 0.97099, and `blend150sx_logit` 0.97104 against `blend150fx_logit`
0.97103. So a one-member change is worth 0 to 2e-5 of public score, sign unpredictable —
which is just the ±5e-5 slice scatter again, not a special fact about one member.

The general rule below is the one that survives: within the top cluster the public slice
cannot resolve CV differences below ~1e-4. Fresh confirmation the same run —
`blend156_rescale` is **4th of four** on CV (0.970025) and scored 0.97105, above
`blend156_rankraw`, which beats it by 1.1e-5 on CV.

The one gain large enough to measure, +0.000340 CV, transferred as +0.00019 LB — about
**56% pass-through**. So: **do not quote "CV + 0.0012" as an LB estimate**, and do not
expect a CV gain to arrive intact.

#### ⚠ "The offset shrinks as CV rises" was an artefact of four points — corrected 2026-08-11

That claim was drawn from four entries spanning a wide CV range. With twelve points it
resolves into two regimes, and only the first is real:

- **Across** the 0.9696 → 0.9700 step the offset genuinely fell, +0.00115 → +0.00100.
- **Within** the top cluster (CV 0.96995–0.97003, nine entries) the offset scatters over
  +0.00097…+0.00108 with **no trend in CV at all**. That scatter is ±5e-5 of LB — the
  same size as the CV differences being compared.

**The public slice cannot resolve CV differences below ~1e-4**, which is every difference
we are still capable of producing. The sharpest demonstration, and the reason this matters
more than a bookkeeping correction:

> `blend150fx_logit` has the **worst CV of the top nine, by 8e-5** — a gap bigger than the
> noise floor and bigger than any single improvement shipped all week — and it scored
> **0.97103**, second-best of everything we have sent and above three stacks that beat it
> on CV.

That is the Rogii failure mode offered for free. `logit` is also the one transform with a
mechanism argument *against* it (its clip provably destroys the tails of ~29 saturating
members). If the public slice is allowed a vote, it picks the entry we have a reason to
believe is worse.

Trust CV for **ranking** decisions. Never as a leaderboard estimate, and never let a
public-LB ordering break a CV tie.

### Noise floor

Per-fold spread within one 5-fold run massively overstates uncertainty. The real noise
floor is the std of repeated 5-fold **means** across partition seeds ≈ **0.00005**.
Any claimed gain smaller than that is nothing.

The **paired** 50/50 test resolves far below that floor, because fitting and scoring both
variants on identical rows cancels the split noise. It is the selection instrument; the
cross-fit is only the headline. A paired delta of +5e-6 that is sign-consistent across
splits is a real ordering, even though no cross-fit could ever see it.

### Submission economics — the daily slot has no alternative use

**A Playground submission cannot hurt you.** Nothing evicts anything: the public
leaderboard shows the best of *all* your submissions, and the two final entries are chosen
by you at the deadline. So the bar for spending a daily slot is **"is this file genuinely
different?"** — NOT "does the gain clear the noise floor?". An unused slot is pure waste.

This corrects a rule inherited from the standing playbook ("an unused slot beats a wasted
one"), which is written for competitions where only the two most recent submissions stay
active and a weak entry **evicts** a good one. That rule does not apply here. Slot 4 nearly
skipped a submission on it.

Two things it does NOT license:
- **Re-sending an identical file.** Scoring is deterministic on a fixed public slice, so a
  duplicate is the one genuinely pointless submission. Vary something real.
- **Selecting on the public LB.** Final selection stays on CV. That is the Rogii failure
  mode, and free submissions must not be allowed to corrupt it.

#### The agent's SLOT number is not a submission slot — keep a prepared queue

The daily counter rolls at **00:00 UTC = 20:00 EDT**, and the agent's runs are numbered
1..10 independently of it. On 2026-08-11 all ten submissions were spent between 00:16 and
04:04 UTC by agent-slots 1–3, so **agent-slots 4 through 10 that day had zero submissions
available** — seven consecutive research-only runs.

That is the normal case, not a mishap, and it has a consequence worth acting on: the
binding constraint is not ideas, it is *having files built when the counter rolls*. Leave
finished, CV-ranked `.csv` files in `submissions/` with their cross-fitted CV recorded in
the journal, so the first run after 20:00 EDT can send ten immediately instead of
spending its hour rebuilding. Check the clock before planning a run around a submission.

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

### Two new OOF savers appeared 2026-08-11 04:37 UTC and BOTH CRASHED — nothing to import

`mohankrishnathalla` published `s6e8-realmlp-oof-saver` and `s6e8-tabm-oof-saver` minutes
before the slot-5 run. Both use `StratifiedKFold(5, shuffle=True, random_state=42)`, i.e.
our exact frozen scheme, so they *would* be directly stackable. Neither produced a file:

- **RealMLP** — died at cell 6, `TypeError: RealMLPConstructorMixin.__init__() got an
  unexpected keyword argument 'verbose'`. `pytabkit`'s `RealMLP_TD_Classifier` does not
  take `verbose`. Ran 57s, produced nothing.
- **TabM** — `pip install tabm-torch` fails on Kaggle (`No matching distribution found`;
  the package does not exist under that name). The notebook falls back to an
  "alternative", ran 5,601s and emitted no OOF AUC.

Check with `kaggle kernels output <ref> -p <dir>` **before** planning around a published
saver: a kernel that lists `np.save('oof_*.npy')` in its source has not necessarily
produced one. When only a `.log` comes back, the run failed. Worth re-checking these two
in later runs — the author is iterating, and a working RealMLP would be a function class
the pool has only through `beicicc/s6e8-fixed4-realmlp-two-seed-artifacts`.

### Enumerating the pool — the CLI list endpoints are unreliable

As of 2026-08-10 the Kaggle CLI's RPC endpoints return `400` for **all** list operations
(`ListDatasets`, `ListKernels`, `ListCompetitions`) while downloads-by-ref,
`competitions submissions -v` and `competitions leaderboard -d` keep working. The
**legacy REST endpoint still works and needs no authentication**:

```bash
curl -s "https://www.kaggle.com/api/v1/datasets/list?search=s6e8&pageSize=100&page=1"
```

Run it with several search terms and union the `ref` fields — that is how the 14 missed
libraries were found.

**Re-run this weekly, not every run.** Re-enumerated 2026-08-11 04:45 UTC over three
search terms: **20 datasets, all already known, newest `lastUpdated` 2026-08-10.** The
pool has been static for two days and every member in it is already imported. The CLI's
`kernels list` was also working again on 2026-08-11 (the 400s of 2026-08-10 were
transient), so prefer it for *kernels* — that is where new material actually appears now.

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

#### ⚠ CLOSED 2026-08-11: you cannot tune a LightGBM into a decorrelated member

The "open opening" below was tuned for solo AUC on 2026-08-10 (**+0.00003, a null**) and
for *decorrelation* on 2026-08-11. The second attempt failed at its own objective:

`lgbm_stump_lat_frac` — depth **3** / **8** leaves against the pack's hand-set depth 7 /
96 leaves, fixed 4000 rounds. That is about the largest available move in tree shape and
it genuinely changed the model (solo OOF **0.96735**, i.e. 0.00044 *worse* than
`lgbm_fixed_lat_frac`). Its correlation with the pack: **maxcorr 0.9961**, median 0.9816 —
the *dense* part of the pack (pack median maxcorr 0.995; only 15 of 149 sit below 0.97).

Effect on the stack, cross-fitted on the frozen folds, all four transforms:
+5e-6 / +2e-6 / **−2e-6** / +4e-6, ensemble +1e-6. Sign-flipping and entirely inside the
±4e-6 solver floor. Public LB moved by **exactly zero**.

**Where a member lands is set by its function class and its pipeline, not its
hyperparameters.** This resolves the slot-3 puzzle: 35 ordinary XGB/LGBM/CatBoost members
from `boltuzamaki` were the largest single share of that day's +0.000340, while GBDT
hyperparameter variation inside our own pipeline is worth nothing. The difference is the
**imputation, encoding and feature decisions upstream of the model**, not the model.

So the only lever with demonstrated size still available is **a genuinely second
pipeline** built here — a different missing-data treatment or a no-TE representation — not
more models on `agent/features.py`'s single lineage.

#### Ordering, not target encoding, is what decorrelates a member — measured 2026-08-11

Five TE-free / alternative-upstream members were built on the frozen folds and profiled in
the **hybrid** space (`experiments/member_profile.py`). Two upstreams, two model families,
and the split is completely clean:

| member | representation | solo OOF | maxcorr | median corr |
|---|---|---|---|---|
| `xgb_cat_lattice` | all 12 cols as **unordered** lattice categoricals, no TE | 0.961074 | **0.9746** | 0.9350 |
| `cat_native` | same, but NaN lifted to its own level | 0.958941 | **0.9762** | 0.9532 |
| `cat_raw` | 12 raw cols, no TE, **ordering kept** | 0.963075 | 0.9933 | 0.9776 |
| `xgb_raw_nan` | 12 raw cols, no TE, **ordering kept** | 0.965152 | 0.9947 | 0.9801 |
| `cat_lat` | our own TE pipeline | 0.966353 | 0.9948 | 0.9677 |

Pack reference: median maxcorr 0.9949, min 0.9309 (`logreg`), 15 of 149 below 0.97.

1. **Dropping target encoding does nothing for decorrelation.** `xgb_raw_nan`'s nearest
   neighbour is `bei_xgb_identity_digit_raw12` at 0.9947 — beicicc already ships that
   pipeline. "No TE at all", named by three consecutive runs as the honest second lineage,
   is a **rediscovery**, not a new one. Same for `cat_raw` (nearest `bolt_cat_cpu5` 0.9933).
2. **Discarding the numeric ordering is what moves a member.** Replicated across XGBoost
   and CatBoost with *different* NaN handling, so it is a property of the representation.
   The two ordering-discarded members correlate only **0.9478** with each other; the two
   raw ones correlate **0.9902**.
3. **And it does not pay** — see the correction under "Attribution of the gain". Low
   correlation obtained this way is the model being worse, not a new direction.

**Untested and the obvious next move:** keep the full TE pipeline and *add* the unordered
lattice categoricals as extra columns instead of replacing everything with them. That
would separate the two effects — if the screening rule is right, such a member sits near
0.967 solo with maxcorr well under 0.99.

**Open opening (superseded — kept for the record):** the library's LightGBM hyperparameters on the lattice feature set were
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
  **Priced 2026-08-11:** 71 of 156 coefficients (46%) are negative at the shipped C=1. As
  C shrinks, the negative count falls 71 → 66 → 59 → 37 → 0 and AUC falls with it, down
  **−0.00104** by the time the last negative coefficient is gone. Non-negativity — NNLS,
  hill-climbing, any "average the models" blend — is not a safe default here, it is
  expensive. (Shrinking toward zero is not identical to constraining ≥ 0, so read this as
  a strong direction rather than a proof.)
- **Do not weight near-equal models by their OOF AUC.** `(auc − 0.5)^p` over the four
  transform stacks (0.96996–0.97004) returns weights uniform to rounding error at every
  p up to 32, and measured **exactly** +0.000000. The classic OOF-performance weighting
  needs spread between the things being weighted; there is none at this level.
- **Ensembling the four transforms: drop the worst, do not fit weights.** Paired 50/50,
  3/3 consistent: dropping the lowest-CV transform is +5e-6, a full 1,540-point simplex
  search is +6e-6. Three fitted parameters buy 1e-6 over one bit of information.
  `blend156_h3` (hybrid + rankraw + rescale, equal) cross-fits to **0.970046** vs
  `blend156`'s 0.970042. `logit` is the one to drop — its clip destroys the tails of ~49
  saturating members.
- Fitting a stacker on the OOF matrix and scoring it on the same matrix reads high.
  Score it with a proper cross-fit, and compare alternatives with **paired** differences
  on the same row splits so split noise cancels.

### `C` is not a regularisation strength until you divide by n

sklearn's `LogisticRegression` minimises `0.5*w'w + C * sum_i logloss_i`, so the penalty a
single row argues against is **`1/(C*n)`**, not `1/C`. At the 345,684-row paired fit half:

| C | 1 | 0.03 | 0.01 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|---|---|
| penalty per row | 2.9e-6 | 9.7e-5 | 2.9e-4 | 2.9e-3 | 0.029 | 0.29 | 2.9 | 29 |

Against a log-loss of order 0.5, **everything at C ≥ 0.01 is unregularised.** The
2026-08-10 sweep (`logs_csweep.txt`) covered 0.03–100, i.e. it fitted the same
unpenalised solution eight times and correctly found no difference — it did not measure
regularisation, and "C does not matter" is not what it established. To shrink 156
coefficients at this n, C has to reach ~1e-5 and below. `experiments/ridge_sweep.py`
extends the grid there and prints the penalty-per-row line so the arithmetic is visible in
the log. **Whenever n or the member count changes, recompute `1/(C*n)` before believing a
C sweep.**

**Measured 2026-08-11 over the full range, 156 members, paired 3 reps: still a null.**
C=0.001 is +7e-6 ± 15e-6 with the sign flipping; C=1e-4 and below are catastrophic
(−1.3e-4 to −1.0e-3). ‖w‖₂ drops by a third from C=1 to C=1e-3 with **no** change in AUC,
then the model degrades. The design is collinear (median pairwise 0.98, 95.4% PC1) but
n/p = **4,400**, and at that ratio the unregularised fit is already precise — the
collinearity is real and harmless. **Do not sweep stacker C again.** If anything set
C=0.01: +5e-6 ± 3e-6 consistent here, and +7e-6 at C=0.03 on the separate 149-member
sweep — six positive measurements, two sweeps, two member sets, all ~5e-6.

Two related defects, both fixed in `ridge_sweep.py` / `blend_lab.py` (2026-08-11):

- **C does not transfer across n.** The pipeline fits at 345,684 (paired), 553,095
  (cross-fit fold) and 691,369 (full fit, the one that predicts test). One shared C
  regularises the submission model 2.0× harder than the instrument that validated it.
  Both now take `--lam`, a penalty **per row**, and derive `C = 1/(lam*n)` per fit.
  Latent while C=1 leaves everything unpenalised.
- **L2 on unstandardised columns is not a neutral prior.** Hybrid column sds run
  **1.81–27.58**, so the widest member is shrunk ~230× less than the narrowest, purely
  from where the transform happens to put its logits. `--standardize` makes it isotropic.
  Built, not yet measured; low prior given shrinkage itself is a null.

## Environment (this machine)

- 16 CPU cores, 31 GB RAM, **no GPU**. `device="cuda"` paths in public code must be
  disabled.
- venv at `.venv` (uv, Python 3.12): pandas, numpy, scikit-learn, lightgbm, **xgboost
  3.4.0**, catboost, pyarrow.
- ⚠ **XGBoost rejects a pandas category INDEX of floating dtype** —
  `Category index from DataFrame has floating point dtype`. The numeric lattice levels are
  floats, so `both[c].astype("category")` fails. Factorize to integer level ids and
  rebuild: `pd.Categorical.from_codes(pd.factorize(s, sort=True)[0], np.arange(k, "int32"))`.
  Code `-1` (NaN) comes back as a missing value, which is what makes XGBoost route it by
  the node's learned default direction rather than as a level.
- Per-fold design matrices are cached to `cache/` by `experiments/build_cache.py` so
  hyperparameter trials cost only LightGBM time.
- The transformed member matrix is cached to `cache/meta_<transform>.npz` by
  `experiments/stack_lab.py:build`, so a combiner sweep costs only the logistic
  regressions instead of re-reading ~150 `.npy` pairs first.

### ⚠ This box is NOT single-tenant — check before launching anything heavy

Other Claude sessions run concurrently in this same workspace and elsewhere on the
machine. On 2026-08-10 the run queue hit **33 on 16 cores** and everything ran ~4× slow —
a LightGBM member that takes 17 minutes alone had not finished a single fold in 35.

```bash
ps -eo pcpu,pid,comm --sort=-pcpu | head        # who else is burning CPU
ps -o ppid= -p <pid>                            # trace ownership up to a `claude` process
```

- ⚠ **`pkill -f "<script> --name foo"` also kills the bash wrapper that launched it**,
  because the wrapper's own command line contains the pattern. It killed this session's
  shell mid-command (exit 144) on 2026-08-11. Use `pgrep -f ... | head -1` then
  `kill <pid>`.
- ⚠ **The same self-match silently breaks `pgrep` wait-loops**, and that failure is worse
  because it is quiet rather than fatal. A "wait until the job finishes" loop like
  `until ! pgrep -f "run_xgb.py --name probeLC"; do sleep 20; done` **never exits**: the
  bash process evaluating the loop has the pattern in its own command line, so `pgrep`
  matches the waiter forever and the job looks permanently unfinished. Break the
  self-match with a bracket class — the regex still matches the target, but the waiter's
  own literal command line no longer matches the regex:

  ```bash
  until ! pgrep -f "run_xgb.py --name pro[b]eLC" >/dev/null; do sleep 20; done
  ```
- Members left in `oof/` **outlive their author's session**. On 2026-08-11 the three
  `cat_*` members were owned by a session that had already exited (their processes were
  reparented to `systemd --user`, PPID 5788). Two live peers were messaged and neither
  owned them. Trace ownership with `ps -o ppid= -p <pid>`; a PPID of 1 or of the systemd
  user manager means **there is nobody to ask** — the members are yours to use, and no one
  else is going to submit them.
- **Before a long job:** check the load. Two agents at `n_jobs=-1` are slower than one.
- **Before submitting:** check whether a peer is building the same artifact. A peer session
  was independently running `stack.py --submit-name stack_pub151_hybrid` off the very
  `oof/` files this session had just written. `ListAgents` + `SendMessage` resolved it;
  trace ownership by parent chain rather than guessing from session start times.
- Anything written to `oof/` is picked up automatically by every later `stack.py` run,
  including a peer's. That directory is shared mutable state.

### Our own member runner had the golem_a/golem_f defect

`agent/run_lgbm.py`'s early-stopping path calls `lgb.early_stopping` on `(Xb, yb)` — the
exact rows that become that member's OOF, so the iteration count is chosen with sight of
the held-out labels. `lgbm_tuned_lat` and `lgbm_tuned_lat_frac` were both built that way.

It does not stay contained in that member's inflated AUC: an optimistic member biases the
stacker's coefficients toward itself, and CV is *the* deadline decision rule.

**Use `--stopping 0` for anything that will be stacked** (a fixed round count, no
`eval_set`). This is why the strongest independent public library names its datasets
`fixed900` / `fixed1500` / `fixed4000`. Keep the early-stopping path for *tuning* only,
where the comparison is what matters and the absolute level does not.

#### The honest way to tune here — `run_xgb.py --probe` / `run_catboost.py --inner`

"Tuning only" still leaks if the tuning holdout is a **validation fold**, because that is
the fold that becomes the OOF, and the round count then carries its labels into the
member. Both runners now carve the tuning holdout out of the fold's **training rows**:

```bash
run_xgb.py --name probeC --mode cat --probe 0.08 --rounds 4000 --probe-stopping 200 ...
# -> "PROBE best inner AUC 0.961600 @ round 401 of 601 -- nothing saved"
run_xgb.py --name xgb_cat_lattice --mode cat --rounds 450 ...   # frozen, no eval_set
```

`--probe` saves nothing at all, so a probe can never become a member by accident. Freeze
the round it reports (rounded up ~10%, since the real fold trains on ~8% more rows) into
`--rounds` and run with no eval set. Costs nothing over the old fold-0 sweep and removes
the last place optimism was entering.

**Measured size of the correction** (identical params, identical folds, 2000 fixed rounds):

| member | early-stopped OOF | fixed-schedule OOF | optimism |
|---|---|---|---|
| `lgbm_tuned_lat_frac` → `lgbm_fixed_lat_frac` | 0.96782 | **0.96779** | **+0.00003** |
| `lgbm_tuned_lat` → `lgbm_fixed_lat` | 0.96771 | in progress | — |

Per fold on `lat_frac`: −3e-5, −5e-5, −5e-5, −5e-5, +2e-5. So the optimism is real and
consistent (4/5 folds) but **small — about 3e-5**, which is under the cross-fit noise
floor and comparable to a single member's contribution to the stack.

Note the fixed count was chosen a priori (a round 2000) rather than from the previous
run's early-stopped iterations (1905–2227) — picking it from those would smuggle the same
held-out information back in through the back door.

⚠ **Do not compare an honest stack against an optimistic one on CV.** Substituting the
fixed members will likely *lower* the stack's CV slightly, because the inflated members
were inflating the stack's number too. That is the correction working, not a regression.
CV comparisons are only meaningful between stacks whose members are equally honest — so
the honest stack **replaces** the optimistic one as the reference, it does not compete
with it.

Note the fixed count was chosen a priori (a round 2000) rather than from the previous
run's early-stopped iterations (1905–2227) — picking it from those would smuggle the same
held-out information back in through the back door.

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

Pre-repair spread across 86 members: 0.679–1.009.

#### ⚠ sd_ratio is NOT a defect detector — corrected 2026-08-10 (slot 4)

This section used to say "anything at 0.68–0.93 is a broken transform, not a property of
the model", with our own `lgbm_tuned_lat_frac` at 1.0018 as the control proving it. **That
inference is wrong.**

Our own `et_lat_frac` — ExtraTrees, trained here, on the frozen folds, with no defect of
any kind — reads **sd_ratio 0.8308**, squarely inside the "broken" range. It emits `p == 1`
on **0.84% of OOF rows against 0.04% of test rows, a 21× asymmetry**, which is the same
shape as naji03's 5.79% → 0.92%.

The cause is structural and applies to every library: **the OOF array is one model's
output per row, while the test array is the mean of five fold models.** Averaging five
saturating models resolves a plateau a single model cannot — all five must agree on
exactly 1.0 for the mean to be 1.0. So *any* member whose outputs saturate produces
sd_ratio < 1, whether or not anything is wrong with it. The 1.0018 control only reads
clean because LightGBM's sigmoid never emits exactly 0 or 1; it was never evidence about
5-fold averaging in general.

Consequences:

- **Do not use sd_ratio to judge whether a published member is trustworthy.** Use the
  gates that survive: fold-id partition, maxcorr == 1.000 duplication, in-sample blends,
  level-2 stack outputs, credibility (OOF > 0.9720 is not achievable), and the paired test.
- **Blast radius checked: nothing was ever dropped on it.** `import_ext2.py` and
  `import_beicicc.py` only *report* it as a table column, and `stack.py`'s repair gate is a
  value check (`(O<=0)|(O>=1)`), not a ratio threshold. Every actual exclusion was on one
  of the gates above. No member was cut wrongly.
- Putting OOF and test on a common scale is still the right thing to do. The transform is
  fine; only the *interpretation* of the ratio was wrong.

Transforms available (all monotone per member, same map on OOF and test, so no member's
solo AUC changes):

| `--transform` | cross-fitted OOF @86 | @151 | note |
|---|---|---|---|
| `logit` | 0.969660 | — | original; clips |
| `hybrid` | 0.969678 | 0.970024 | rank-gauss on the clipping members only |
| `rankraw` | 0.969684 | **0.970023** | rank-gauss on all, ties averaged |
| `rescale` | 0.969686 | — | min-max over both splits then logit; only *partially* equalises |

Worth ~+0.00002 CV and **0.00000 LB**. Fix it because it is wrong, not because it scores.

`hybrid` and `rankraw` are indistinguishable on CV (1e-6 apart at 151 members, against a
5e-5 noise floor). **Prefer `rankraw` on mechanism**: `hybrid` only repairs the members it
judges "broken", and the finding above says that judgement separates members by an
artefact of OOF-vs-test construction rather than by quality. `rankraw` reports
sd_test/sd_oof of min 1.0000 / med 1.0000 / max 1.0253 — it equalises the scales fully,
where `hybrid` and `rescale` only do so partially.

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
| 2 new members we built ourselves (ExtraTrees + a linear model) | +0.000005 | slot 4 |
| stacker C, anywhere in 0.03–100 | +0.000007, i.e. flat | slot 4, closed |

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

#### ⚠ "Correlation first, solo AUC second" is BACKWARDS below ~0.966 solo — 2026-08-11

Two members were built specifically to be decorrelated and they are, by a wide margin —
and they were worth **nothing**. See "Ordering, not target encoding" below for how they
were built. Cross-fitted on the frozen folds, identical drops, four transforms:

| group added | n | maxcorr | solo OOF | logit / hybrid / rankraw / rescale | ensemble |
|---|---|---|---|---|---|
| the decorrelated pair | 2 | 0.9746, 0.9762 | 0.9611, 0.9589 | −1e-6 / +1e-6 / +5e-6 / +3e-6 | **+1e-6** |
| the redundant trio | 3 | 0.9933–0.9948 | 0.9631–0.9664 | +8e-6 / +6e-6 / +9e-6 / +5e-6 | **+8e-6** |

The pair sign-flips and sits inside the ±4e-6 solver floor — a null. The trio, whose
pipelines the pack **already holds**, is one sign across all four transforms and carries
effectively the whole 151 → 156 gain.

**Mechanism.** The pack is 95.4% PC1. A member gets to maxcorr 0.9746 here by being
*worse* — the pair are the two lowest solo AUCs of the five. Their residual is noise, not
signal in a new direction. The `decorr` group that did pay on slot 3 had solo AUCs near
the pack's.

**Decorrelation bought by discarding information is not decorrelation from an independent
pipeline.** The first destroys the signal that would have made the low correlation worth
having; only the second pays. This reconciles the whole record: 35 ordinary GBDTs from
boltuzamaki paid, while both deliberate decorrelation attempts built here — the depth-3
LightGBM stump (maxcorr 0.9961) and the categorical recode (0.9746) — did not.

**The operational rule:** screen on the *pair* (low maxcorr AND solo ≥ ~0.966), never on
correlation alone. A candidate below ~0.966 solo has to earn its place on a paired
measurement, whatever its correlation says.

### The geometry of the pack — measured 2026-08-11, and it explains everything above

Eigenvalues of the 149×149 correlation matrix, computed on the **hybrid** matrix (the
space the stacker actually sees), `cache/meta_hybrid.npz`:

| | |
|---|---|
| variance in PC1 alone | **95.39%** |
| first 3 PCs | 97.08% |
| first 10 PCs | 98.42% |
| first 40 PCs | 99.49% |
| eigenvalues > 1e-4 × λmax | **55** |
| entropy effective rank | **1.41** |
| smallest eigenvalue | **1.1e-16 → exact collinearity** (see the duplicate below) |

**149 members are one consensus signal plus a very thin tail of corrections.** The entire
job of the stacker lives in the 4.6% of variance that is not PC1, and the whole
0.9696 → 0.9700 climb was bought there. This is the quantitative version of what the
journal kept rediscovering:

- It explains why a member from a **new pipeline** beats a better member from a pipeline
  already held: the first adds a *direction*, the second adds *magnitude* along PC1, and
  PC1 is already saturated.
- It explains why the stacker's `C` is flat and why non-linear meta-models lose. There is
  no rich structure to regularise or bend — there are ~55 usable directions carrying 4.6%
  of the variance between them.
- It bounds the remaining upside honestly. Anything that lands inside the existing span
  is worth ~0 no matter how good its solo AUC.

#### ⚠ `bolt_xgb_d7_alt1` and `bolt_xgb_d7_alt2` are the SAME ARRAY — drop one

`np.array_equal` is `True` for both the OOF and the test vectors (max|diff| exactly 0.0,
identical AUC 0.9681005). This is the `maxcorr == 1.000` duplication condition that
`RESEARCH.md` already lists as a rejection gate and that `najiama`'s members were
excluded for — it slipped through the boltuzamaki import because that import screened
each candidate against the **pack**, never against the other candidates in its own batch.

Practical effect is small: with an L2 penalty an exactly duplicated column splits its
coefficient, which halves the effective penalty on that one direction — and `C` is flat
here anyway. So this is a correctness fix, not a scoring one. But **the real member count
is 148, not 149**, and every "n members" figure in this file and the journal is off by one
from `--ext2` onward. Screen new batches against themselves as well as against the pack.

#### ⚠ The flagship "maxcorr 0.811" for `bolt_extratrees_support` is a transform artefact

This file twice cites `bolt_extratrees_support` at **maxcorr 0.811** as the headline
evidence that decorrelation is real. That number comes from `import_ext2.py:120`, which
computes correlation on the raw **`to_logit`** scale — the very scale whose clip destroys
the tails of saturating members, and ExtraTrees saturates hard.

In the hybrid space the stacker actually uses, `bolt_extratrees_support` reads
**maxcorr 0.9701** (against `et`), median 0.9340, solo AUC 0.94322. It is not close to
being the most decorrelated member. The genuine extremes are:

| member | maxcorr (hybrid) |
|---|---|
| `logreg` | **0.9307** |
| `bolt_lookup_v3_evidence` | 0.9333 |
| `nn2` | 0.9491 |
| `golem_c` (spline GAM) | 0.9500 |
| `bolt_fttransformer` | 0.9512 |

Distribution over all 149: min 0.931 / p5 0.956 / median 0.995 / max 1.000. Only **15**
members sit below 0.97 and only **3** below 0.95.

This is the same shape of error as the `sd_ratio` correction: a quantity measured in a
space the model does not use, then read as a property of the model. **The group-level
conclusion still stands** — it was established by paired deltas, not by this number — but
the `maxcorr < 0.97` grouping in the slot-3 attribution table was cut in logit space, so
its membership is partly determined by which members saturate. Compute correlation in the
transform the stack is actually fitted in.

### How many members, and chosen how — measured 2026-08-11, `experiments/member_select.py`

Greedy top-k under three orderings, paired (every cell fitted and scored on identical
rows), 2 reps, 148 members after dropping the duplicate. `auc` = greedy by solo OOF AUC
measured on the fit half only; `decorr` = greedy minimum max-|corr| against those already
chosen, using no labels at all; `random` = the control.

| k | `auc` | `decorr` | `random` |
|---|---|---|---|
| 10 | 0.969599 | 0.969517 | 0.969244 |
| 25 | 0.969821 | 0.969730 | 0.969671 |
| 50 | 0.969965 | 0.969874 | **0.970042** |
| 100 | 0.970128 | 0.970141 | 0.970145 |
| 148 | 0.970177 | 0.970181 | 0.970180 |

**1. There is no saturation in k. Keep every member.** The curve is still climbing at the
right-hand edge: k=100 → 148 is worth +3.5e-5. Do not prune the stack, and do not read the
95.4%-in-PC1 geometry as licence to. A "cleaner, better-conditioned 50-member stack" costs
**−1.4e-4**, which is three times the noise floor.

**2. Decorrelation FAILS as a member-level selection rule.** `decorr` loses to `auc` by
−8e-5 to −9e-5 at k=10/25/50, sign-consistent across both reps, and only draws level once
k ≥ 100 (i.e. once both orderings have swept up nearly the same set). Selecting individual
members for being decorrelated picks up weak outliers — the ordering opens `logreg`,
`golem_g`, `nn2`, `knn`, `fmpure` — and a stack of oddities is worse than a stack of good
models.

**3. At k=50 a RANDOM 50 beats both principled orderings** (+7.7e-5 over `auc`, +1.7e-4
over `decorr`). Greedy-by-AUC concentrates inside one redundant strong family; greedy-by-
diversity concentrates on junk; random draws a natural mix of strength *and* pipelines.

This does not contradict slot 3's group-level attribution, it **explains** it. Adding
`boltuzamaki`'s 47 members paid because it was a *representative sample of a whole
independent pipeline* — not because its members were individually decorrelated. The unit
that carries value is the pipeline, not the member. So: **acquire whole pipelines, keep
everything they contain, and never hand-pick members by a correlation statistic.**

#### The solver's own noise floor is ±4e-6 — and it matters

At k=148 all three orderings select the *same 148 columns*, differing only in column
order. They score 0.970177 / 0.970181 / 0.970180. That spread — **4e-6** — is pure
`LogisticRegression` convergence noise, measured for free.

That is the same magnitude as the paired deltas slot 4 reported for our own two members
(`et_lat_frac` +6e-6, `linlat` +3e-6, both together +5e-6) and called "a real ordering".
**Those numbers are at, not above, the solver's own reproducibility.** Sign-consistency
across row splits does not rescue them, because the solver noise is not resampled by
changing rows. Treat any paired delta under ~1e-5 as unresolved unless it is also stable
under column permutation.
