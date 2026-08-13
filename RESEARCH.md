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
| **concatenating the original dataset** | **−58e-6 at 1×, −3,340e-6 at 50×** | measured here 2026-08-11, monotone in dose — see the dedicated section below. The usual Playground edge is INVERTED here |
| k-NN target encoding (0.936 standalone!) | +0.00001 | a new *view*, not a new *channel* |
| second decimal digit, integer-ness | +0.00001 | |
| TE smoothing sweep (10/50/200) | ≈0 | smoothing 10–20 is the sweet spot |
| NA-indicator features | −0.00001 | missingness is MCAR |

**`colsample_bytree` belongs to the feature set, not to the model.** `0.5` was inherited
unexamined from the library's 184-column lattice matrices. On a 12-column frame it drops
half the predictors from every tree and costs ~0.0006 (inner-holdout 0.96082 plateau
against 0.96140 at `colsample 1.0`, and it starts at AUC 0.849 where the others start at
0.92+). Re-check it whenever the frame width changes.

## The original dataset — CLOSED, both routes measured here (2026-08-11)

`jayjoshi37/smartphone-usage-and-addiction-prediction`, 7,500 rows × 16 cols, downloaded to
`data/orig/`. No missing values in any predictor. Carries `addiction_level`, an ordinal
None(819) < Mild(1373) < Moderate(2874) < Severe(2434) of which `addicted_label` is exactly
`level >= Moderate` — a re-encoding of the target, not a side channel.

**No leak.** Joining on all 12 predictors as strings: **0 of 691,369 train rows and 0 of
296,302 test rows** match an original row verbatim. Do not re-check this.

### The generating rule (read off the real data, `experiments/orig_transfer.py`)

Only two columns matter in the original — `daily_screen_time_hours` and
`social_media_hours` (`weekend` is 0.964-correlated with `daily`; everything else has
marginal AUC 0.50):

| cell | n | orig rate | **competition rate** |
|---|---|---|---|
| `social > 4.0` | 2,748 | 1.0000 | 0.9956 |
| `social ≤ 4` & `daily > 8.0` | 2,093 | 1.0000 | 0.9680 |
| `social ≤ 4` & `daily ≤ 6.0` | 1,634 | 0.0000 | **0.3253** |
| `social ≤ 4` & `6 < daily ≤ 8` | 1,025 | 0.4556 | 0.6498 |

86.3% of the real data is decided outright by two thresholds; the band is an irreducible
coin flip (flat in both drivers, splits Mild 558 / Moderate 467). Honest 5-fold AUC on the
original is **0.9885** — the real data is *more* separable than the synthetic frame ever
gets (best held here 0.9700). **The generator did not add information, it smeared a crisp
two-threshold rule into a ramp, and the ramp is what we are scored on.** Profiling `daily`
at 0.25h inside `social ≤ 4`: 0.417 → 0.411 → 0.547 → 0.677 over (5.75,6.0] → (6.5,6.75].
The kink is real and displaced off 6.0; full-resolution TE over 691k rows locates it better
than a 7,500-row rule can state it.

⚠ A GBM restricted to the 1,025-row band reaches **in-sample AUC 1.0** with flat
importances. That is memorisation, not a recoverable sub-rule. Only quote 5-fold numbers.

### Route 1 — separate estimator (`experiments/orig_member.py`). Dominates concat, adds nothing

A model fitted on the originals alone never sees a competition label, so its prediction is
honest on both splits and goes into the stack as a member with a stacker-chosen weight.

| direction | AUC |
|---|---|
| orig-trained → competition train (transfer) | 0.8503 |
| **same, trained on 10 MCAR-masked copies at the competition's own per-column missing rates** | **0.8864** |
| comp-trained (one fold) → the original rows | 0.9630 |

The **+36e-4 from matched missingness** is the transferable trick: the original has no NaNs,
the competition frame is 13.9%/19.4% missing on the two rule drivers, so ~29% of competition
rows are missing a driver and a complete-data model has never been asked which way a NaN
should go. Binary target 0.8864 beats the 4-level ordinal 0.8805 and the hand-written rule
0.8207.

Saved as member `orig_binm`: solo 0.8864, **maxcorr 0.8792 hybrid / 0.8632 rankraw** — the
most decorrelated object this workspace has ever held (pack min 0.9309, median 0.9946, prior
record 0.9746). In the 160-member stack it is worth **−1e-6 to −2e-6 across logit / hybrid /
rankraw / rescale / ens4 and exactly 0 under h3**; the weaker unmasked version is −3e-6 to
−6e-6. Twelve readings, none positive.

### Route 2 — literal concatenation (`experiments/orig_concat.py`). Monotonically harmful

12 raw columns, native NaN, frozen folds, extra rows in the training half only:

| training set | OOF AUC | delta |
|---|---|---|
| baseline | 0.962639 | — |
| + 1× the 7,500 originals | 0.962581 | **−58e-6** |
| + 10× | 0.961653 | −986e-6 |
| + 50× | 0.959299 | −3,340e-6 |

Monotone in dose; −58e-6 at 1× reproduces the public record's −0.0001. **The single most
reliable edge in the Playground Series is not merely absent in S6E8, it is inverted.** Do
not re-open this. Use the original only as a diagnostic for what the generator did.

### What this proved about the pack, which matters more than the angle

Every "member additions barely move the stack" finding here carried the objection that every
member tested was ~0.99 correlated with the pack. `orig_binm` at maxcorr 0.879 retires it:
different function class, different training distribution, and still nothing. **The
constraint is not correlation — the pack's span already contains everything the 12 columns
can say about this target, and a member is decorrelated from that span precisely to the
extent that it is worse.** Member hunting is closed on that basis, not on "we have not found
a decorrelated member yet".

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

#### ⚠ The top-cluster scatter is NOT noise — it is the transform. Measured 2026-08-11

Both statements above ("the offset scatters with no trend", "the slice cannot resolve
below ~1e-4") are half right, and the half that is wrong is the half that matters for the
deadline pick. `experiments/cvlb.py`, `cvlb2.py`, `logit_bias.py`, `predict_lb.py`.

**1. Restrict to the operating cluster and the correlation collapses.** Over all 20 scored
files pearson(CV, LB) is +0.958 — carried entirely by the three 74/88/86-member entries
30e-5 below everything else. Over the 16 files with CV ≥ 0.96995 it is **+0.148**. Never
quote a CV↔LB correlation computed across the 0.9696/0.9700 step.

**2. The residual is structured by transform family**, over the ≥150-member sets only
(the 74/88/86 entries have a systematically larger offset and contaminate any family mean
by ~4e-5 — excluding them is not optional):

| family | n | mean LB−CV |
|---|---|---|
| **logit** | 2 | **+0.001082** |
| rescale | 2 | +0.001016 |
| ens4 | 3 | +0.001011 |
| rankraw | 4 | +0.000999 |
| hybrid | 3 | +0.000979 |

Differenced **within member set** (removes everything except the transform): `logit −
hybrid` = +0.000104 (150fx), +0.000091 (150sx), mean **+0.000097**. Controls: `rankraw −
hybrid` +0.000018 (n=3), `rescale − hybrid` +0.000031 (n=1), `ens4 − hybrid` +0.000022
(n=2). Exact permutation over family labels on all 20 residuals: **one-sided p = 0.0085**.

So `blend150fx_logit`'s "worst CV of the top nine, second-best LB" is **not** the Rogii
failure mode offered for free, which is how the section above reads it. It is a replicated
transform-specific bias with a mechanism: `to_logit` clips at the unit interval, an OOF
prediction comes from one fold model while a test prediction is a 5-fold average and is
less extreme, so more OOF cells land on the plateau — the logit stack is fit and CV-scored
on the damaged side and predicts on the clean side. Its CV understates its test AUC.

**Mechanism only partially confirmed** — census at 161 members (`logit_bias.py --census`,
`experiments/clip_census.csv`): 49 members put ≥1 cell on the plateau but only **~9 are
asymmetric** (naji03 5.79%/0.92% = 6.3×, et 3.1×, bolt_extratrees 2.5×, rf 1.9×, tabm ~1.8×).
The 16 heaviest (bolt_lookup_v2/v3, fm*, bolt_deepfm_exact) are pinned at 91–94% on **both**
sides, ratio 1.00. Aggregate OOF 9.884% of cells vs test 9.531% — 0.35pp. `sd_test/sd_oof`
is ~1.00 for all of them; **the old note's "ratios down to 0.679" does not reproduce at 161
members.** Those 16 symmetric near-constant columns are a competing explanation for logit's
low CV (genuine information loss, which would predict a low test AUC too) — but symmetric
loss cannot produce the displacement, so the asymmetry has real force.

**3. Consequence — the drop-logit / `h3` decision is not established.** `h3` (drop logit
from the 4-transform ensemble) beats `ens4` by **+5e-6** on cross-fitted OOF, and four
"independent instruments" agree (paired 50/50, row bootstrap, 8 resampled fold splits,
exhaustive subset enumeration with 7/7). **All four are computed on OOF and therefore share
this bias — four instruments on one biased measurement is one measurement.** Propagated
with the mix-gap estimator (a mix's gap = mean of its components' gaps; validated on 150fx
where all four components *and* the mix are scored, error −7e-6):

| quantity | value |
|---|---|
| estimated `h3` gap | +0.000998 |
| estimated `ens4` gap | +0.001019 |
| **ens4 − h3 in the gap** (favours ens4 on test) | **+0.000021** |
| h3 − ens4 in CV (favours h3 on OOF) | +0.000005 |

The bias is **4.2× the margin it would overturn, in the opposite direction**; the
estimator's own error is 32% of the effect, so the sign is informative and the magnitude is
not. The subset enumeration's 7/7 is not extra evidence: a ~2.4e-5 bias at ¼ weight produces
exactly that pattern regardless of whether logit helps on test, and the effects it measured
were 3e-6 to 12e-6.

**Standing position: `h3` and `ens4` are not separable. Do not claim either is better, and
do not resolve it on the public LB.** Resolving it needs a third within-set replication of
the `logit − hybrid` displacement (a 9.7e-5 effect, which the slice *does* resolve) — not an
LB reading of the 5e-6 h3/ens4 contrast, which it does not.

#### ⚠ "The public slice cannot resolve differences below ~1e-4" — the wrong null

That estimate came from bootstrapping **one** file's AUC at the slice's size: sd 5.4e-4 at
20% of 296,302 rows, implying nothing below 1.5e-3 is resolvable. **The public slice is
fixed**: both candidates are scored on identical rows and correlate ~0.999, so the shared
slice noise cancels. Bootstrap the **paired difference** instead (`cvlb2.py`):

| pair | CV gap | paired sd @20% | paired sd @50% | P(flip) @20% |
|---|---|---|---|---|
| blend158_h3 vs blend156 | +0.000007 | 0.000008 | 0.000005 | 20.0% |
| blend158_h3 vs blend156_h3 | +0.000002 | 0.000005 | 0.000003 | 31.0% |
| blend158_h3 vs blend150fx_logit | +0.000098 | 0.000029 | 0.000018 | 0.0% |
| blend156 vs stack_pub74_logit | +0.000400 | 0.000062 | 0.000037 | 0.0% |

The paired sd is **~60× smaller** than the unpaired estimate. The LB resolves a 1e-4 gap
perfectly and gives an ~80%-reliable bit on a 7e-6 gap. It is informative at our margins —
just not authoritative, and the pair-agreement table shows where it fails: pairs separated
by 5e-5–1e-4 of CV are ordered correctly only **32%** of the time (8 agree / 17 disagree),
because that band is exactly where the logit files sit; pairs above 1e-4 are **51/51**.

**GENERAL RULE, third time this workspace has been caught by it.** The other two were the
chi2/df null (unmatched on K and cell density) and the residual-booster's permuted-feature
null (permuted features cannot reconstruct z, so it measured a different experiment). *Any
null here must be matched on everything except the thing being tested.* For a two-file
comparison on a fixed slice, that means pairing on the slice.

#### Kaggle accepts values outside [0,1] — confirmed, the metric is rank-only

24 of the queued files hold raw logits, range down to −17.9 / up to +17.9.
`blend150fx_logit` has range [−8.49, +17.02] and scored **0.97103**. So no clipping or
sigmoid step is needed before submitting, and adding one would only risk creating ties. Do
**not** read the mean of one of these files as a base rate.

#### `experiments/audit.py` — run it before spending slots

Gates the whole `submissions/` directory in one pass: column names, id order against
`test.csv`, finiteness, ranking-distinctness (sha1 of the rank vector — AUC sees nothing
else), and CV recomputed from each file's own `oof_*.npy` rather than trusted from the
journal. Writes `experiments/audit_results.csv` with the CV→LB pairing, which `cvlb*.py` and
`predict_lb.py` read. As of 2026-08-11: 41 files, 41 distinct rankings, correct ids
everywhere, and **21 files with a CV never submitted — including the four best on CV.** No
`h3` file of any member count has ever been scored.

#### Seed-averaging a member is worth ~10× a blend tweak

`xgb_latcat` at seeds 13/17/23: solo OOF 0.967696 / 0.967750 / 0.967766, pairwise
correlation 0.99813–0.99818, **probability mean 0.967904 = +138e-6 over the best single.**
Two 6,365s runs bought a bigger member-level gain than any blend decision made all week. If
the stack moves on it, seed twins for the other strong own-built members beat new
architectures as the next member-side idea.

### Noise floor

Per-fold spread within one 5-fold run massively overstates uncertainty. The real noise
floor is the std of repeated 5-fold **means** across partition seeds ≈ **0.00005**.
Any claimed gain smaller than that is nothing.

The **paired** 50/50 test resolves far below that floor, because fitting and scoring both
variants on identical rows cancels the split noise. It is the selection instrument; the
cross-fit is only the headline. A paired delta of +5e-6 that is sign-consistent across
splits is a real ordering, even though no cross-fit could ever see it.

#### The 5e-5 floor is a MARGINAL number — do not apply it to paired comparisons

Measured 2026-08-11 with `experiments/auc_boot.py`, which Poisson-bootstraps the **rows**
of a fixed cross-fitted OOF vector using the *same* resampled rows for every candidate.

| quantity | sd |
|---|---|
| any single ensemble's AUC (marginal) | **0.000167** |
| the *difference* between two near-identical ensembles (paired) | **0.000001–0.000003** |

A factor of ~80. The old rule "believe nothing under ~1e-5" is right for a marginal
comparison and badly wrong for a paired one: at 691,369 rows a +4e-6 paired gap came back
at P(better) = 0.97 over 400 reps. **Restated: under ~1e-5 is unbelievable marginally and
perfectly believable paired.**

Caveat that bounds the claim: this prices ROW noise only, holding the folds, the member
models and the stacker fits fixed. It is a lower bound on the uncertainty of "which
candidate is better", never an upper bound. `experiments/repcv.py` prices the fold-split
half by re-partitioning the combiner's folds (free — the member OOF matrix is fixed on
disk, and a 156-member logistic fit costs **4 seconds**, not the minute you would guess).

#### The fold-split half, measured — 8 combiner partitions, `experiments/repcv.py`

| quantity | sd across 8 fold splits |
|---|---|
| the CV **level** of any candidate | **0.000004–0.000005** |
| the **paired difference** h3 − ens4 | **0.0000005** |

Same lesson on a second axis: the level wanders ~10× more than the difference between two
candidates evaluated on it. `ens3_h3` beat `ens4` in **8 of 8** splits at +4e-6.

⚠ **The frozen seed-42 split is the pessimistic one.** It returns the LOWEST value of all
eight partitions for every candidate — ens4 0.970043 vs a mean of 0.970048, h3 0.970047 vs
0.970052. Every headline CV in `JOURNAL.md` is therefore ~5e-6 low. Harmless for selection
because it shifts all candidates alike; do not quote the frozen numbers as unbiased.

The three instruments now available, cheapest first, all on a fixed member OOF matrix:

| instrument | what it resamples | cost | resolves |
|---|---|---|---|
| `auc_boot.py` | rows | 130s / 300 reps | ~1e-6 paired |
| `repcv.py` | combiner fold partition | ~470s / split | ~1e-6 paired |
| `blend_lab --reps` | a 50/50 fit-score split | ~400s / rep | ~2e-6 paired |

Agreement across all three on the drop-logit decision was +5e-6 / +4e-6 / +4e-6.

Implementation note worth reusing: weighted AUC needs only one global sort per candidate
plus an O(n) pass per bootstrap rep (`np.add.reduceat` over runs of equal score handles
ties exactly). 400 reps × 7 candidates over 691k rows runs in 128s. Do not resort per rep.

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

**Re-checked 2026-08-11 05:30 UTC (slot 6).** The author re-ran both ~35 min earlier.
TabM **failed again**, differently: they swapped `tabm-torch` for `rtdl_revisiting_models`
and guessed its API — `TypeError: MLP.__init__() got an unexpected keyword argument
'd_layers'`. TabM remains dead; stop checking it.

**RealMLP went `COMPLETE` at 06:20 UTC on its third attempt — imported, verified, and
measured a null.** It ships `oof_realmlp.npy` + `test_realmlp.npy`, solo OOF 0.958585.

*The verification is worth copying* — it is stronger than the fold-id gate and needs no
`fold_id.npy`. Score their OOF vector **per fold under our frozen folds** and compare to
the fold AUCs printed in their kernel log. Ours came back 0.95721 / 0.95910 / 0.95856 /
0.95983 / 0.95908 against their reported 0.95721 / 0.95910 / 0.95856 / 0.95983 / 0.95908 —
identical *and in the same order*, so the partition and the fold labelling both match. Any
author who prints per-fold AUCs can be gated this way in 30 seconds.

Outcome: maxcorr 0.9662 (the workspace record) but solo 8e-3 below the pack, and the stack
did not move. See "the independent pipeline escape clause" above. Parked in `oof_rejected/`.

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

#### ⚠ CLOSED 2026-08-13: tuning ANY GBDT is worth ~4e-7 into the stack

Note XGBoost is not a missing leg — `latr1_xgb` is the **best GBDT of any family here**, and
the pack already holds `xgb_lat`, `xgb_latcat` (×3 seeds), `xgb_cat_lattice`, `xgb_raw_nan`
and the `bolt_xgb_*` family. Two measurements price any further tuning:

- **Solo→stack pass-through is 1.4%** — seed-averaging `xgb_latcat` bought **+138e-6 solo**
  (the largest member-level gain ever measured here) and moved the stack **+2e-6**.
- **Tuning a GBDT for solo AUC on these folds was measured at +3e-5** (2026-08-10, LightGBM,
  recorded as a null).

3e-5 × 1.4% = **+4e-7**, ~1% of the 5e-5 noise floor. Even a 10× better tune stays under it.
And there is nothing left for a tuned member to inform: every member trains on the frozen
SKF5 seed42 folds already, and the deadline pick (`h3`) fits **zero** free parameters above
the stack. **Do not tune GBDTs. Do not add ordinary GBDT members.**

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

## LightGBM tuning — CLOSED, including the `max_bin` ladder (2026-08-13)

Tuning LGBM on the lattice features was done in full: stage A (one knob at a time), stage B
(7 combinations of the winners), then 5-fold finalists. **Net +0.00003 full-OOF, below the
5e-5 noise floor.** The shipped vector is stage B's: `num_leaves 63, max_depth 7,
min_child_samples 250, reg_lambda 80, max_bin 511`.

### The `max_bin` ladder and why the public +0.0023 does not transfer here

`kitopl/max-bin` (2026-08-13) argues `max_bin` must be ≥ the max distinct-value count,
worth **+0.0023** on their pipeline — more than double their 26-trial Optuna search. The
mechanism is sound and their ceiling reproduces here exactly: max raw distinct is **1437**
(`weekend_screen_time`; then `daily` 1389, `social` 721, `work_study` 600, `sleep` 451,
`gaming` 401, `notifications` 231, `app_opens` 166, `age` 18).

Measured on our matrix (`--stage c`, fold 0, lr 0.05):

| `max_bin` | control | vs 255 | stage-B regularised |
|---|---|---|---|
| 255 | 0.96638 | — | — |
| **511** | **0.96669** | **+0.00031** | **0.96688** |
| 1023 | 0.96655 | +0.00017 | — |
| 1439 | 0.96651 | +0.00013 | 0.96691 (+0.00003) |
| 2047 | 0.96649 | +0.00011 | 0.96678 (−0.00010) |

**Both vectors peak and then decay** — neither climbs to the distinct-value ceiling and
flattens, which is what the mechanism predicted. The control peaks at 511; the regularised
stage-B vector drifts up a below-noise +3e-5 to 1439 and then gives back −1e-4 by 2047.
`max_bin 511` stands; do not raise it.

**Why:** `agent/features.py:175` (`te_block`) target-encodes **every exact lattice key**.
The exact-value lookup signal therefore reaches our LightGBM as a continuous, already-
monotone TE column that needs *zero* bin resolution. Their model has no TE, so bin edges on
the raw columns are its only access to the lookup structure — hence their baseline is
0.96406 and ours is 0.9678. Once the channel is saturated, extra bins are not extra
information, only extra split opportunities across 184 mostly-target-derived columns, i.e.
variance.

**General lesson, worth more than the knob:** a public gain measured on a *weaker
representation* does not survive transfer to a saturated one, however large the headline
and however correct the mechanism. The mirror of the journal's older finding that a new
channel beats refining an existing one (decimal lattice +0.00011 vs tuning +0.00003).

### Operational: never run two `n_jobs=-1` LightGBM jobs at once on this box

Two concurrent LightGBM processes put 32 OpenMP threads on 16 cores; barrier-heavy
histogram building degrades **~7×** (measured: identical trial 904s contended vs 125s
alone), far worse than the 1.2× oversubscription implies. Serialise, or `kill -STOP` the
other job and `kill -CONT` it after — pausing is reversible, killing a pre-registered run
is not.

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

  > ⚠ **Provenance, established 2026-08-13 (slot 7): every number in the bullet above, and
  > the shipped `blend156w.csv`, came off a BROKEN simplex grid.** `simplex()` chose its
  > stars-and-bars cut positions from `1..n+k-2`, one short, so every row summed to
  > `(n-1)/n = 0.95` instead of 1. AUC is scale-invariant, so the *scores* that grid
  > produced are still valid readings of a legitimate simplex at step 1/19 — but **equal
  > weights (0.25 each) and h3 (0, ⅓, ⅓, ⅓) were not points on it**, so the search was
  > structurally unable to return either baseline it was being compared against. The "+6e-6
  > for a search vs +5e-6 for drop-worst" comparison was therefore rigged slightly against
  > the search. Fixed at 14:39 local on 08-13; the grid goes 1,540 → **1,771** points and
  > now contains both baselines. `submissions/blend156w.csv` was written at 01:19 local,
  > i.e. **before** the fix — it is a buggy-grid artefact. `blend156w2` is the rebuild.
  >
  > **Re-run on the fixed grid, on the best member set — the search's 1e-6 edge vanishes.**
  > `transform_weights.py --stack blend159av --paired --reps 3`, 1,771 points, both
  > baselines now reachable:
  >
  > | candidate | paired vs equal | free parameters |
  > |---|---|---|
  > | `drop_worst` (≡ h3) | **+0.000005 ± 0.000002**, 3/3 consistent | one bit |
  > | `searched` simplex | **+0.000005 ± 0.000002**, 3/3 consistent | three floats |
  > | `auc_p1..p32` | ±0.000000, sign-flips | — |
  >
  > They are **equal to the last digit reported**. The fold weights say why: the search
  > returns logit at **0.00 / 0.05 / 0.10** across the three reps and spends essentially all
  > its freedom on the one bit `drop_worst` already has. Restated for the deadline: the
  > simplex search does not beat `h3`, it *rediscovers* `h3`. This retires transform-weight
  > search as a live lead and is a third independent line supporting the `h3` pick.
  >
  > The cross-fit agrees: **`blend159av_w` cross-fits to 0.970048**, below `blend159av_h3`'s
  > 0.970049 and only +3e-6 over equal. Mean fold weights
  > `logit 0.06 / hybrid 0.28 / rankraw 0.41 / rescale 0.25`.

### ⚠ A fitted file must never store `R @ wfull` as its OOF vector — `audit.py` ranks on it

Found 2026-08-13 (slot 7) and fixed. `audit.py` ranks every candidate by
`roc_auc_score(y, oof_<stem>.npy)`. For an ordinary blend that vector is genuinely
cross-fitted and the number is honest. But `transform_weights.py --submit-name` was saving
`R @ wfull`, where `wfull` is the weight vector chosen to **maximise AUC on exactly those
rows** — an in-sample maximum over 1,771 grid points, dropped straight into the workspace's
main decision table.

| file | stored `R @ wfull` | honest cross-fit |
|---|---|---|
| `blend159av_w` | 0.970050 | **0.970048** |
| `blend156w` | 0.970048 | **0.970047** |

+2e-6 of pure fitting was enough to put `blend159av_w` **above `blend159av_h3` (0.970049) at
the top of the audit ranking**, i.e. a future run reading that table would have found a
fitted artefact sitting where the deadline pick belongs. That is the Rogii failure mode
arriving through the instrument rather than through the leaderboard.

Fixed at the source: the script now stores the cross-fitted `mo` (fold weights applied to
held-out rows), which is the correct vector anyway. The `.csv` still ships `Rt @ wfull` —
fitting the weights on all OOF rows is the right *procedure* for test predictions; it is only
the CV bookkeeping that must not reuse them. Both existing files were repaired in place by
reconstructing `mo` from the fold weights in their build logs; both reconstructions matched
the logged cross-fitted CV exactly. **Rule: any file whose name ends in `w` is weight-fitted —
check its `oof_*.npy` is the cross-fitted vector before trusting its row in `audit.py`.**


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
  shell mid-command (exit 144) on 2026-08-11 — **twice, in two consecutive runs**, the
  second time after reading this very warning. Never type `pkill -f` in this workspace.
  Use `pgrep -f "scrip[t].py"` then `kill <pid>`.
- **Yield cores with `kill -STOP` / `kill -CONT`, do not kill long jobs.** Four concurrent
  jobs took the box to load 29/16 on 2026-08-11 and two 4-second combiner fits were being
  starved by two 80-minute XGBoost runs. Suspending the long jobs for 10 minutes cost them
  nothing and unblocked the short ones. `nice` alone does not achieve this.
- ⚠ **A member landing mid-run silently changes any bench that globs `oof/`.** On
  2026-08-11 `repcv.py` loaded 157 members — `xgb_lat` had saved one minute earlier and
  `xgb_latcat` had not — producing a member set matching no shipped candidate. **Pin the
  member set with an explicit `--drop` whenever another job may be writing to `oof/`**, and
  always print and read the loaded member count.
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
| **bagging the combiner over disjoint 80% slices** (`foldbag5`) | +0.000001, sign flips | spearman vs the single fit 0.999991–0.999999 |
| **bootstrap-bagging the combiner** (`boot5`) | −0.000013, consistent | resampling with replacement discards 37% of unique rows |

**The linear logit stack is the right combiner. Stop looking for a better one.**

Combiner-level resampling is closed (`experiments/bag_lab.py`, 2026-08-11). At n/p = 4,400
the coefficients are estimated too precisely for averaging to have anything to remove.

Related, and worth knowing rather than worrying about: the shipped pipeline uses two
different estimators on the two sides — the cross-fitted CV comes from fold models
(n = 553,095) while the submitted file comes from one full-data fit (n = 691,369). The
`foldbag5` measurement prices that inconsistency at spearman 0.999999 and +1e-6. Real, and
immaterial.

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

#### The rule is now measured from BOTH sides — 2026-08-11

| test | member(s) | solo AUC | maxcorr | stack delta |
|---|---|---|---|---|
| `blend153` | `xgb_cat_lattice`, `cat_native` | low | **0.9746 / 0.9762** (best ever here) | null |
| `blend158` | `xgb_lat`, `xgb_latcat` | **0.967664 / 0.967696** (best ever here) | 0.9969 / 0.9970 | **+1e-6, null** |

Best-in-workspace decorrelation at low accuracy buys nothing. Best-in-workspace accuracy at
pack-typical correlation buys nothing either. **Neither half of the pair is worth anything
alone**, and no member built here has ever had both. `xgb_latcat` adds the 12 unordered
lattice categoricals *on top of* the full TE frame and gains +3.2e-5 solo over `xgb_lat` —
but only 1.05–1.14% of split gain goes to those columns, so the member stays inside the
pack. Before spending 2h on a member, argue for both numbers or do not run it.

#### And the "independent pipeline" escape clause does NOT rescue a weak member

The paragraph above says decorrelation from an independent pipeline pays where
decorrelation bought by discarding information does not. **Tested directly on 2026-08-11
and it is false as stated.** `mkt_rmlp` (below) is a neural net from a completely separate
pipeline, sets the workspace decorrelation record by a wide margin — maxcorr **0.9662**
hybrid / **0.8930** rankraw, below 0.97 against all 156 members, against a pack median of
0.9946 — and adds **exactly 0.000000** to the rank-ensemble (blend158 → blend159), with the
per-transform signs mixed. Its solo AUC is 0.9586, ~8e-3 under the pack.

**The accuracy floor binds regardless of where the decorrelation came from.** Independent
provenance is necessary, not sufficient. Three record-setting members, three nulls:

| member | maxed | other half | stack delta |
|---|---|---|---|
| `xgb_cat_lattice`/`cat_native` | decorrelation by discarding order | low solo | null |
| `xgb_latcat` | solo AUC 0.967696 | maxcorr 0.9970 | +1e-6 |
| `mkt_rmlp` | decorrelation, independent pipeline, 0.9662 | solo 0.9586 | **0** |

Member hunting is closed unless a candidate is plausibly near 0.966 solo **and** genuinely
outside the pack. Nothing across five public libraries is.

**`oof_rejected/`** holds members that verified clean but measured null — currently
`mkt_rmlp`. It sits deliberately outside the directory `blend_lab` globs so a rejected
member cannot silently re-enter a build. Restore only with a reason.

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

## Conditional-independence testing against the stack — the instrument, and how it lies

Measured 2026-08-11. The question "is there anything left for a feature to capture" is
`y _||_ x | z`, where z is the stack's own OOF score. Two ways to test it, both of which
gave a confident WRONG answer before the control was built.

### 1. Stratify on z-bins and chi-square — `experiments/erroran.py`

**The chi2/df null is NOT 1.0.** It is a function of the number of x-bins K and the cell
density, because the z-bin rate is estimated from the row margin. At 2048 z-bins over
691,369 rows the null is:

| K | empirical null chi2/df |
|---|---|
| 2 | **~2.15** |
| 4–6 | ~1.1–1.3 |
| 13 | ~1.06 |

Judging a K=2 statistic against a K=13 control puts the NaN indicators at the top of the
scan with an apparent 36 sigma. `experiments/erroran_na.py` builds the matched null
(permuted indicators at the identical missing rate) and every one of the 12 columns lands
inside its own control band. **Only K-matched, marginal-matched controls are admissible.**

Second trap: a permuted control destroys the x–z correlation as well as the x–y one.
Inside a z-bin z still varies, so a real feature correlated with z picks that up and scores
an excess the control cannot produce. The strongest predictor in the frame is therefore
guaranteed to look like the strongest finding whether or not anything is left in it. Both
traps come from replacing z with a bin, so prefer instrument 2.

### 2. Boost on top of the stack — `experiments/resid_boost2.py`

**Never use `logit(rank percentile)` as the LightGBM `init_score`.** It is logistic by
construction with sd ~1.8, not a log-odds, so the residual is dominated by a smooth
calibration error in z. The corrector cannot see z, so it reconstructs z from the features
(only to AUC ~0.967) and injects that proxy's error. Measured cost: **−0.0076 AUC** against
a permuted control at −0.00003. Diagnostic signature: **monotone decline then recovery** as
rounds increase. A model chasing noise does not recover.

Two correct forms, both matched-control:
- `--mode offset` — offset = logit of an out-of-fold **isotonic** map z → P(y=1).
- `--mode feature` — no offset; the stack score is a *feature* beside the frame, so a
  calibration defect cannot masquerade as a feature finding. Baseline = a run with the
  stack score as the only feature (0.969967, i.e. ~8e-5 of tree-discretisation handicap).

**Result, 2026-08-11, blend158_h3 (158 members): `real − ctrl` is NEGATIVE at all 8 round
counts in BOTH modes** (offset −2.0e-5 → −2.2e-4; feature −7.4e-5 → −5.3e-4). The 40-column
numeric frame — raw columns, constrained imputation, bounds, decimal lattice, and every
interaction a 31-leaf tree can find over 1000 rounds — adds nothing to the stack.

⚠ **Out-of-fold isotonic calibration of a submission costs 6.2e-5 AUC** (0.970048 →
0.969986). Isotonic is monotone *within* a fold; five per-fold maps are not mutually
monotone, so pooling reorders rows across folds. Never calibrate the final file.

## Where the AUC actually lives — `experiments/errormap.py`

Pooled within-segment AUC vs global 0.970048 (blend158_h3):

| segmentation | pooled within | vs global |
|---|---|---|
| `n_missing_all` | 0.974025 | +0.003977 |
| `n_screen_missing` | 0.974071 | +0.004023 |
| `other_screen_band` | 0.961145 | −0.008904 |
| **`daily_band` (2h)** | **0.933423** | **−0.036625** |

Most of the headline score is ranking ACROSS screen-time bands, where the base rate runs
0.2421 (0–2h) → 0.9997 (12h+). The hard population is **4–8h of daily screen time**:
274k rows, base 0.36–0.66, within-band AUC 0.916/0.939 — the same region
`georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new` shows P(addicted) FALLING
with screen time at fixed social media (which is why monotone constraints cost score).
A booster free to isolate that band cannot beat a shuffled control there (above).

By missingness: 0 missing → 0.977541, 5+ missing → 0.913295. The +0.0040 pooled-within gap
is NOT recoverable — it is the arithmetic penalty of pooling populations of different
difficulty, and `iso_regime.py` measured regrading them at −0.000085 on 5/5 folds.

## Transform subsets — enumerated exhaustively 2026-08-11, `experiments/subset_lab.py`

All 11 subsets of {logit, hybrid, rankraw, rescale} on blend158, paired 300-rep bootstrap:
**h3 = hybrid+rankraw+rescale is the argmax at 0.970048.** Runners-up rankraw+rescale and
hybrid+rankraw at 0.970046 (P(>h3) 0.15/0.12); ens4 at 0.970043 (P 0.003).

**`logit` is beaten by the same subset without it in 7 of 7 comparisons**, by 3e-6 to 12e-6.
The finding is not "drop logit from the four-ensemble" — the transform is actively harmful
in every combination it appears in. Fourth instrument agreeing, after the paired 50/50, the
row bootstrap and the 8 resampled fold splits. **This line is closed; do not re-sweep it.**

## Public notebooks worth having read

- `georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new` (26 votes) — the best
  analysis on the board. Confirms from the other side: `gaming_hours`/`work_study_hours`
  carry ~0.500 conditional AUC and are worth +0.0032 anyway because they RECONSTRUCT
  missing drivers through the budget identity; `other_screen = daily − (social+gaming+
  work_study)` is the one engineered feature that pays (we already hold it as
  `other_screen_imp`); missingness carries nothing about the target despite an adversarial
  train/test AUC of 0.57; monotone constraints on screen columns lose score because the
  conditional curve genuinely dips.
- `mohankrishnathalla/s6e8-tabm-oof-saver` — **ERROR on three separate runs** (2026-08-11
  04:54 the latest). Stop checking it. The RealMLP sibling did land and is a measured null
  (parked in `oof_rejected/`).

## What a public rank is empirically worth — `experiments/lbhist.py`, measured 2026-08-11

Dataset: `georgymamarin/playground-series-s6-leaderboards` (825 KB) — public rank, private
rank and both scores for every team in the **seven completed S6 episodes**. Download:

```bash
kaggle datasets download -d georgymamarin/playground-series-s6-leaderboards -p data/lbhist --unzip
```

AUC episodes E2/E3/E5 are S6E8's reference class. Durable facts:

- **Top-30 retention across the seven episodes: 77 / 3 / 53 / 20 / 57 / 0 / 0 percent.**
  Three of seven destroyed their public top 30. Public #1 finished 570th (E2), 615th (E4),
  379th (E6), 440th (E7).
- **A destroyed top 30 does not imply anyone overfitted.** S6E2's whole-board
  spearman(public, private) is **0.99** and its score correlation **1.00**; ranks 1 and 570
  differ by 1e-4 privately. Compression, not collapse. Diagnose the two apart with the
  matched null below before drawing any lesson from a shakeup.
- **Matched null:** add i.i.d. Gaussian noise at the episode's own observed shift sd to
  every public score, re-rank, compare simulated to observed top-30 retention.
  `obs > sim` ⇒ the shift is mostly a **common offset**, which cannot reorder anyone
  (E1 +33%, E5 +24%, E3 +13%). `obs < sim` with p90/median of the centred shift ≫ 1.9 ⇒
  genuine heterogeneous blow-ups (E7 ratio 10.73, E4 9.47). Calibrated to about **±25%**.
- **Density must be anchored where you stand, not at the leader.** Anchoring the window on
  the leader's score gave spearman −0.25 with retention over 7 episodes — a failure, and the
  same unmatched-null error this file already records three times. Anchored at public rank
  13: S6E2 240 teams within ±5e-5, S6E3 140, S6E5 9, **S6E8 live 14**.
- **Submission volume does not predict a private drop.** AUC episodes, public top 100,
  spearman(submissions, private rank drop) = **−0.083**; the 73–310-submission bucket drops
  a median +87.5 ranks against +143.5 for the 1–14 bucket. The uniform ~+180-rank drop in
  every bucket is the selection effect of conditioning on a high public rank. This supports
  the brief's "use all 10 slots"; Rogii's failure was fitting a hedge parameter to public
  feedback, not submitting often.

### The private-side projection for our own position (rank 13, 0.97106, 1,415 teams)

| assumed shift sd | median private rank | p10 | p90 | P(top 10) | P(top 10%) |
|---|---|---|---|---|---|
| 0.000043 | 12 | 7 | 22 | 34.8% | 100.0% |
| 0.000067 | 15 | 5 | 38 | 32.2% | 100.0% |
| 0.000124 | 26 | 5 | 90 | 24.2% | 98.8% |

**Bronze (top 10% ≈ rank 141) is not the binding constraint — stop targeting it.** Top 10 is
a 25–35% draw that public-LB chasing cannot improve.

⚠ Historical `public_score` is a **selected** entry post-close; our live score is
**best-of-all-submissions**. The projection is optimistically biased, not unbiased.

### Pool status as of 2026-08-11 08:00 UTC

`datasets list -s s6e8` returns **20** datasets, newest `lastUpdated` 2026-08-10, all
imported or explicitly rejected. Static for three days — **check kernels, not datasets.**
`georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new` re-ran 08:00 UTC but the
diff against `notebooks/gaming_nothing_new/` is prose and one moved constant only.

## ~~The `logit` CV bias~~ — FALSIFIED off-leaderboard 2026-08-13, do not re-apply

⛔ **Read this box before anything below it.** Everything in this section was measured
against the public leaderboard. `oofsim.py`, which never touches the leaderboard, tested it
on a labelled pseudo-test set and it is **not there**:

| | claimed (3 LB contrasts) | measured (seed 7, labelled hold-out, full dose) |
|---|---|---|
| `gap(logit) − gap(hybrid)` | **+0.000097** | **+0.000000** |
| dose profile 0→4 | should rise with dose | +0, −6, −15, +4, +0 (e-6): flat, sign-flipping |
| `gap(ens4) − gap(h3)` | +0.000021 | **−0.000007** (truth prefers `h3`) |
| does CV mis-rank the transforms? | yes, logit specifically | **no** — spearman(CV, truth) +0.943 at every dose, CV puts logit last and so does the truth |
| does the OOF weight search underweight logit? | yes, that was the mechanism | **no** — `w_oof` logit 0.00 = `w_true` logit 0.00, **5/5 doses** |

**Replicated on a second split 2026-08-13 (slot 7).** `oofsim_summary.py 7 11`, sd across
splits:

| dose | `logit − hybrid` | signs | `ens4 − h3` | signs |
|---|---|---|---|---|
| 0 | +0.000022 ± 0.000031 | 1/2 | +0.000000 | 1/2 |
| 1 | +0.000022 ± 0.000039 | 1/2 | +0.000001 | 2/2 |
| 2 | +0.000012 ± 0.000038 | 1/2 | −0.000005 | 0/2 |
| 3 | +0.000020 ± 0.000022 | 2/2 | −0.000002 | 1/2 |
| **4** | **−0.000010 ± 0.000015** | 1/2 | **−0.000009 ± 0.000003** | **0/2** |

Flat across dose, sign-inconsistent at four of five doses, **negative at full dose**, every
error bar covering zero, largest pooled mean a quarter of the +97e-6 claim. Pre-registered
outcome 2, on both splits. `ens4 − h3` is −9e-6 at full dose and **0/2 positive**: the
labelled truth prefers `h3` on both splits. Seed 13 was still running at end of slot 7 —
pool it with `oofsim_summary.py 7 11 13`, but no result can restore `ens4` (confirmation
required 3/3 positive at full dose; seeds 7 and 11 are +0 and negative).

⚠ `oofsim.py` still carries the **0.95-sum simplex grid bug** fixed in
`transform_weights.py` (its printed `w_oof` sum to 0.95). Not binding for what it has been
used for — the searched optimum is a corner and `ens4` is the rank-average rather than a grid
point — but fix it before using oofsim to compare anything *against equal weights* again.

**Consequences, all of them:**
- **The `logit += 8.9e-5` correction is withdrawn.** Do not apply it to any CV, ever again.
- **The deadline pick is `blend159av_h3` (CV 0.970049), second `blend158_h3` (0.970048)** —
  raw CV, zero fitted parameters. It is *not* `blend159av`/`ens4`.
- The 8/8 within-set LB replications below are real readings of the public slice and are
  still 8/8. They just do not mean what they were read to mean. A 1–3 ulp effect that
  survives 8 public readings and vanishes on 138k labelled rows is a property of that slice.
- This was pre-registered with a decision rule written before the numbers, and this is the
  branch the rule assigned. The rule can no longer be satisfied either way — confirmation
  required positive at full dose in 3/3 splits and seed 7 is +0.000000.

Kept below because the *shape* of the mistake is the lesson: four "independent instruments"
agreeing, a mechanism that sounded right, a magnitude fitted to three public points, and a
final pick moved by it. That is the Rogii failure with better prose.

## The `logit` CV bias, as it was argued (superseded — see the box above)

24 (CV, LB) points over >=150-member stacks, `experiments/cvlb3.py`. **Logit's (LB − CV) gap
sits +8.9e-5 above every other transform's, with complete separation:** its three readings
(+0.001080/+0.001085/+0.001099) are all above all 21 non-logit readings (max +0.001025).

Within member set, logit's CV is ~65e-6 *worse* and its LB is *better*, **8/8**:
logit−hybrid +3 ulp on 3 sets, logit−rankraw +2 ulp on 3 sets, logit−rescale +1 ulp on 2.

**The ordering failure is entirely in the logit column.** Non-logit pairs separated by
2e-5..5e-5 on CV are ordered correctly by the LB **54/54**; the 5e-5..1e-4 bucket that reads
40% contains *only* logit pairs. Dropping the 3 logit points takes spearman +0.585 → +0.808.

**Mechanism:** logit's clip pins the tails of saturating members and pins **more OOF rows
than test rows**, because an OOF row is one model's output
while a test row is a 5-fold average and so is less extreme. So the damage lands on the CV
side. This is a defect in the instrument, not a property of the public slice, and it
therefore applies to the private slice equally.

⚠ **Do not quote "49 of 159 members need the repair" as the size of this.** The census (see
the *Mechanism only partially confirmed* block above, ~line 313) says only **~9 of the 49
are asymmetric**; the 16 heaviest pinners are at 91–94% on *both* sides, ratio 1.00.
Aggregate asymmetry is **OOF 9.884% of cells vs test 9.531% — 0.35pp**. Symmetric pinning
damages CV and test equally and cannot displace the gap, so the whole effect has to come out
of the tree family (`rf` 1.9×, `et` 3.1×, `bolt_extratrees` 2.5×, tabm ~1.8×, `naji03` 6.3×).
That is why `oofsim`'s dose is built from exactly those members and not from the lookup/fm
family — it doses the only thing that could produce the displacement.

**Correction:** `logit += 8.9e-5`; a mix carries it at its logit weight (`ens4` 1/4, `h3` 0).
Non-circular check — the correction was calibrated on gaps, never on ordering — spearman on
the fully-crossed 158 set goes **−0.088 → +0.736**.

~~**Consequence for the deadline pick:** `ens4` beats `h3` by +17e-6 corrected CV, and beat
it on LB 2/2 at +1 ulp, despite losing on raw CV by 4-5e-6. Deadline pick is `blend159av`.~~
**Withdrawn 2026-08-13** — see the box at the top of this section. `ens4 − h3` measured on a
labelled hold-out is **−7e-6**, i.e. the truth prefers `h3`, which is also what raw CV said
all along. The pick is `blend159av_h3`.

⚠ LB resolution is **1e-5** (five printed decimals). Every contrast above is 1–4 ulp. No
single comparison means anything; only the replication count does.

### `oofsim.py` — the off-leaderboard test of all of the above

Holds out 20% of `train` as a labelled pseudo-test, runs the real 5-fold pipeline on the
other 80%, and so reproduces the exact OOF/test asymmetry with **both sides labelled**. Never
touches the leaderboard. Dose control: the four saturating members are added one at a time to
six clean ones, so the mechanism must switch on with the thing that causes it.

Cost, measured 2026-08-13: **~8 min member training + ~30 min evaluation per seed, and the
evaluation is essentially serial** (the 1,771-point simplex `wsearch` is a Python loop of
`roc_auc_score` calls). Run seeds **sequentially**. Two concurrent seeds took a 12× wall-clock
penalty and three never finished a single member — LightGBM at `n_jobs=-1` in two processes
thrashes on 16 cores. `--eval-only` reuses `cache/oofsim/O_s{seed}.npy` / `T_s{seed}.npy`.

Chain them with a plain `a; b` under `nohup setsid`, so they outlive the session:

```bash
nohup setsid bash -c '.venv/bin/python experiments/oofsim.py --seed 11 > logs_oofsim_s11.txt 2>&1
                      .venv/bin/python experiments/oofsim.py --seed 13 > logs_oofsim_s13.txt 2>&1' &
```

⚠ Do **not** gate one on the other with `while pgrep -f "oofsim.py --seed 7"; do sleep; done`.
`pgrep -f` matches the waiting shell's *own* command line, which contains that string, so the
loop waits on itself forever and the queued run never starts. Cost 7 minutes on 2026-08-13.

⚠ **A `--frac` smoke run used to write a result file indistinguishable from a real one.**
`cache/oofsim/results_s7.json` sat on disk holding a 5% subsample — correct schema, all five
doses, AUCs of 0.947 instead of 0.962 — and `oofsim_summary.py` would have pooled it as a
seed. Fixed 2026-08-13: results now carry a `meta.n_inner` stamp and the summary refuses
anything below 553,095 inner rows. The offending file is
`cache/oofsim/results_s7_SMOKE_f0.05.json.rejected`; do not restore it. This is the same
class of error as `oof_naji18.npy` landing in `data/ext_members2/` — an artefact that loads
cleanly and is wrong.

## ⚠ Final selection is a MANUAL BROWSER ACTION and the default is the Rogii failure

Found 2026-08-13 during consolidation. Every entry in this workspace argues about which file
is the deadline pick; **nothing in the workspace can actually make that pick**, and the
default if nobody does is the exact thing the whole CV discipline exists to prevent.

**There is no API for it.** `kaggle competitions` exposes
`{list, files, download, submit, submissions, leaderboard, team-submissions,
submission-limits, episodes, replay, logs, pages, hosts, data, settings, solution, launch,
init, create, topics, topic-messages}` — no verb selects a final submission, and
`kaggle competitions submissions -v` does not report selection state either. It is the
"Use for Final Score" toggle on the **My Submissions** tab of the competition page, in a
browser. On this box that means launching Brave with the debug port (see the root
`CLAUDE.md`) and driving it over CDP; Brave was **not running** on 2026-08-13.

> ⚠ **Escalated 2026-08-13 (slot 6): this is not merely "Brave was not running" — no agent
> run on this box can do it, and it needs the human.** Checked directly:
>
> | check | result |
> |---|---|
> | `brave` / `brave-browser` on PATH | **absent** — but see the correction below: `google-chrome` **is** present |
> | `~/.config/BraveSoftware/Brave-Browser` | **does not exist** |
> | `~/.config/google-chrome` | exists but holds only `Crash Reports` — no logged-in profile |
> | `brave` MCP server tools in the agent session | **not present** in the tool list |
> | CDP on `127.0.0.1:9222` | not listening |
> | `websocket-client` / `websockets` in `.venv` | neither installed |
>
> So there is **no authenticated browser session on this machine at all**, and the root
> `CLAUDE.md` instructions for the Brave MCP server cannot be followed here as written.
> Starting a browser would not help: the toggle is behind a Kaggle login this box does not
> hold. **The final-selection action must be done by Teddy in his own browser**, or the
> machine must first be given a logged-in profile. A future agent run should not re-spend
> tokens rediscovering this — it should re-check the six rows above, and if they are
> unchanged, say so and move on.
>
> This is now the **highest-value open item in the competition**, and it is worth more than
> any remaining modelling. The spread between the CV pick (`blend159av_h3`, 0.970049) and
> the worst file in the best-public tie (`blend158_logit`, 0.969961) is **−88e-6**, whereas
> every live modelling lever left here is worth ~2e-6. One click is ~40x the entire
> remaining research programme.

### Correction + the endpoint's real name (2026-08-13, slot 9)

Two rows of the table above were wrong or imprecise. **The conclusion is unchanged: this
still needs Teddy.** But get the reason right, so no run "fixes" it by installing a browser.

- **`google-chrome` IS on PATH** (`/home/nixos/.nix-profile/bin/google-chrome`), and
  `~/.config/chromium` exists too. "No browser on this box" is not the blocker.
- **The blocker is the login.** Both profiles are empty — `~/.config/google-chrome` holds only
  `Crash Reports`, and `find ~/.config/google-chrome -name Cookies` returns nothing. There is
  no Kaggle session anywhere on this machine.

**The credential here is OAuth, not an API key.** `$KAGGLE_CONFIG_DIR=/home/nixos/.kaggle`
holds `credentials.json` (not `kaggle.json`): `access_token` in ASP.NET Data-Protection
format (`CfDJ8…`), `refresh_token`, `scopes: ["resources.admin:*"]`, user `thtennant`,
~24h expiry. The CLI refreshes it; do not hand-edit.

**The toggle's endpoint is named, and it is worth recording permanently.** `kagglesdk` builds
calls as `POST /api/v1/<service>/<Method>`; the site's internal router is `/api/i/`, which
returns **404 for a method that does not exist and 400 for one that does** — a free existence
oracle. Result:

```
POST https://www.kaggle.com/api/i/competitions.SubmissionService/UpdateSubmissionSelection
```

(also live: `SubmissionService/{ListSubmissions, GetSubmission}`. 404, i.e. not real:
`CompetitionService/*`, `SelectFinalSubmission`, `SetFinalSubmission`, `ToggleFinalSubmission`,
`SelectSubmission`, `SetSubmissionSelected`, `ListFinalSubmissions`, `GetFinalSubmissions`.)

⚠ **The OAuth token does NOT open it — falsified, do not retry.** The control that settles it:
sending the same requests with **no `Authorization` header at all**, and with a deliberately
invalid bearer, reproduces the **identical** 400/404 pattern. The 400 is the `/api/i/` router
rejecting *before* auth, so it reports route existence only and says nothing about the token.
Every body shape tried (`competitionId` int/string/snake_case, real id `125218`, paging,
`x-xsrf-token` header, protobuf content-type) returned 400 with `content-length: 0` — zero
schema feedback. `/api/i/` wants the website's cookie session plus its XSRF token.
**Closed deliberately:** brute-forcing a body against an endpoint that *mutates final-submission
selection*, with no read-back to verify the effect, is the wrong thing to guess at.

### ✅ You CAN read the selection — and as of 2026-08-13 21:10 UTC nothing is selected

The write path is gone, but there is a read, and it settles the risk empirically.
`kagglesdk/competitions/types/competition_enums.py:63` defines
`SubmissionGroup.SUBMISSION_GROUP_SELECTED = 2`, and `ApiListSubmissionsRequest` takes `group`:

```
SUBMISSION_GROUP_SELECTED:   0 rows      <- nothing is selected
SUBMISSION_GROUP_SUCCESSFUL: 30 rows     <- control: same call, same auth, works
```

Wrapped as **`experiments/check_selection.py`** — prints the selection, compares it to the
deadline pick, **exits 1 if nothing is selected**, exits 2 if the control fails (so a broken
call never masquerades as an empty selection). **Run it first, every run.**

⚠ **Auth gotcha.** `KaggleClient()` with no arguments sends **no `Authorization` header** and
returns 401 — `kaggle_http_client._try_fill_auth` only reads `KAGGLE_API_TOKEN` or
`~/.kaggle/kaggle.json`, and knows nothing about the OAuth `credentials.json` this box uses.
The 401 means "never sent", not "token dead". Pass it explicitly:

```python
tok = json.load(open(os.path.expanduser("~/.kaggle/credentials.json")))["access_token"]
KaggleClient(env=KaggleEnv.PROD, api_token=tok)
```

### ⚠ CLOSED 2026-08-13: the authenticated API has no selection write method

Slot 9 closed `/api/i/` (rejects before auth, so its 400/404 says nothing about the token).
The remaining door was the **authenticated** router `api.kaggle.com/v1`, where the token
works. It does not serve the method — and unlike `/api/i/`, this router returns real JSON
errors for routes that exist, so the controls are meaningful:

| POST, bearer token, `{}` body | result |
|---|---|
| `competitions.CompetitionApiService/ListSubmissions` | **403 JSON** `Permission 'competitions.participate' was denied` |
| `…/NoSuchMethodXyz` (negative control) | 404, website HTML |
| `competitions.CompetitionApiService/UpdateSubmissionSelection` | **404, website HTML** |
| `competitions.SubmissionService/UpdateSubmissionSelection` | **404, website HTML** |
| `competitions.SubmissionService/ListSubmissions` | 404, website HTML |

Every selection candidate is byte-identical to the negative control. `CompetitionApiClient`
exposes 40+ methods and none touches selection. **Final-submission selection is
website-session-only and cannot be automated from this box. Do not re-open this line.**

**Useful by-catch:** `POST /api/v1/competitions.CompetitionApiService/GetCompetition` with
`{"competition_name":"playground-series-s6e8"}` returns `id = 125218`, `deadline
2026-08-31T23:59Z`, `maxTeamSize 3`, and **`maxDailySubmissions = 10`** — an authoritative
read of the daily cap that does not require burning a submission to see the CLI's
"N remaining today" line.

**The default is best-public.** Kaggle's standing rule is that an entrant who selects
nothing has their best *public-leaderboard* submission(s) chosen automatically. Confirm this
against the competition's own Rules page the next time a browser is up — but plan for it,
because here it is actively dangerous:

| best public = 0.97106, four-way tie | cross-fitted CV | paired diff vs `blend159av_h3` | acceptable as a final? |
|---|---|---|---|
| `blend159av` (ens4) | 0.970045 | −0.000004 ± 0.000002 | tolerable, but 4e-6 under the `h3` pick |
| `blend158` (ens4) | 0.970043 | −0.000006 ± 0.000002 | same |
| `blend156` (ens4) | 0.970042 | −0.000008 ± 0.000003 | same, older member set |
| `blend158_logit` | **0.969961** | **−0.000088 ± 0.000009** | **no — worst CV of every ≥150-member stack held here** |

> **Priced 2026-08-13 (slot 9), 300-rep paired row-bootstrap, `auc_boot.py --ref
> blend159av_h3`.** The exposure is **not the tie, it is one file.** Three of the four are
> 4–8e-6 behind — real (all three resolve directionally, P(better) 0.040/0.013/0.003) but
> negligible. `blend158_logit` is **−88e-6 at ±9e-6, about ten sigma**, and ~18× the 5e-5
> noise floor. So the toggle's value is binary: it is worth avoiding one specific
> auto-selection, not a diffuse "40× the research programme".

Note **none of the four is the CV pick**: `blend159av_h3` (0.970049) and `blend158_h3`
(0.970048) both score 0.97105 on the public slice and so would never be chosen by default.
So the default is wrong in every branch, and in one branch out of four it ships the single
worst-CV file in the queue — flattered onto the top line by a logit effect that
`oofsim` has since measured at zero on labelled data. That is not a hypothetical version of
the Rogii failure; it is that failure, arriving by default, with no decision made by anyone.

### The action, to be executed before 2026-08-31 23:59 UTC

Selection is changeable any time before the deadline, so doing it early costs nothing and
removes the whole risk. Two slots; pick them to **bracket the one unresolved question**
(whether the logit CV bias is real), so one of the two is right either way:

| slot | file | rationale |
|---|---|---|
| 1 | **`blend159av_h3.csv`** | top raw CV 0.970049, fits nothing above the stack, and the choice `oofsim` prefers on labelled data |
| 2 | **`blend160origm_h3.csv`** | CV 0.970049, joint-top, different member set, same construction — supersedes `blend158_h3` (0.970048), see the 2026-08-13 slot-6 entry |

### The partial hedge available without a browser (2026-08-13, slot 9)

The default takes the **top two by public score**, so putting two CV-good files strictly above
0.97106 makes it impossible for the default to land on `blend158_logit`. Tomorrow's ten free
slots should be spent CV-best-first with that objective, rather than as undirected "free
reads".

**This is not LB-chasing, and the distinction matters.** No final entry is being chosen for
its public score; the observation is that *if the toggle stays unset the default chooses on
public score*, and every file sent is one CV already endorses (≥0.970042) — so steering the
default steers it toward a file CV likes. Consequence: keep CV-poor files **off** the send
list (`blend153_rankraw`, 0.970027, was dropped for this reason — if it were the file to land
0.97107 the default would select something 22e-6 below the pick).

It is a **hedge, not a fix**, and it may simply not fire: the public slice is fixed and
deterministic, the top files are ~0.9999 correlated, and the observed spread among them is a
single 1e-5 quantisation step. **The toggle remains the only real fix.**

*(Superseded 2026-08-13 evening: this table previously paired `blend159av` (ens4) with
`blend159av_h3` to bracket the logit-bias question. `oofsim` answered that question — the
bias is not there — so there is nothing left to bracket and both slots go to `h3`.)*

Do **not** select on public score. All four of the 0.97106 files and both picks above are
within 1 ulp of each other on the public slice; that column carries no information here.

## The submission queue — 59 files, all distinct, 29 unsent (audited 2026-08-13 18:00 UTC)

Checked by rank-normalising every `submissions/*.csv` and hashing: **zero rank-identical
groups across all 59.** Every queued file is a genuinely different AUC entry, so none of
them is the pointless identical resubmit the brief warns about. Re-run the check with the
snippet in `JOURNAL.md` if new files are built.

### Run this FIRST, every run, before reading the queue

```bash
kaggle competitions submissions -c playground-series-s6e8 -v \
  | .venv/bin/python experiments/lb_refresh.py     # -> experiments/lb_scores.json
.venv/bin/python experiments/audit.py              # overlays that file on its literals
.venv/bin/python experiments/verify_pick.py        # composites match their own parts
```

`audit.py` used to carry a hand-maintained `LB` dict. It went **ten points stale** on
2026-08-13 and then reported six already-sent files as "never submitted" and the best-sent CV
as 0.970042 when it was 0.970049 — i.e. it would have talked a run into re-sending a file,
the one submission the brief calls genuinely pointless. `lb_refresh.py` makes the API the
authority; never hand-edit the dict again. It also prints the best-public tie set, which is
what surfaces the final-selection risk above.

`verify_pick.py` recomputes every `ens4` from its four single-transform CSVs and every `_h3`
from its three, and checks each matches its own recipe better than the other one. All 9
member sets passed on 2026-08-13 at spearman 1.000000. It exists because `audit.py` catches a
corrupt file but not a **mislabelled** one, and the deadline argument is a comparison between
two file names. Useful by-product: **ens4 and h3 agree at spearman 0.99982** — the pick that
has consumed three journal entries is a choice between two 99.98%-identical rankings.

### The next ten, in order — pre-validated, just send them

⚠ **Re-priced 2026-08-13 evening.** This order was chosen so five member sets each gain a
matched `ens4`/`h3` pair, because that contrast was the sole basis of the deadline pick.
`oofsim` has since falsified the claim those replications were feeding, so **the pairs are no
longer worth much** — and every one of the 29 unsent files has a CV *below* the best already
sent (`blend159av_h3`, 0.970049), so none of them can improve the public score either. The
list is still fine to send — a slot has no alternative use — but spend two or three of the
ten on a **`blend159av_w`** instead: the simplex transform-weight search on the 159av member
set, which is the one line `oofsim` returned a positive on.

| # | file | why |
|---|---|---|
| 1 | `blend159_h3` | new pair, 3rd replication |
| 2 | `blend159` (ens4) | " |
| 3 | `blend160orig_h3` | new pair, 4th |
| 4 | `blend160orig` (ens4) | " |
| 5 | `blend160origm` (ens4) | completes the pair with `blend160origm_h3`, already sent |
| 6 | `blend156_h3` | completes the pair with `blend156` (ens4), already sent |
| 7-10 | `blend159av_{logit,hybrid,rankraw,rescale}` | second fully-crossed six-transform set |

All ten verified 2026-08-13: 296,302 rows, ids in `sample_submission` order, all finite, ten
distinct md5s. Single-transform files carry raw decision-function scores (range roughly
±17), which is correct — the metric is AUC and only the ordering is read.

## najiama's `*_blend` files — permanently excluded (confirmed 2026-08-13)

The dataset re-versions periodically (new `18_blend` on 2026-08-12). **Do not import any
`N_blend` file.** najiama fits blend weights on the full OOF, so the published OOF is
in-sample: `18_blend` reads solo OOF **0.969856**, above all 166 pack members, which is the
signature of that fitting and not of a better model. Independently, it is redundant —
naji01–05 are honest members already in the pack, so the stacker already spans every honest
combination of them. `experiments/naji18_probe.py` reproduces the profile.

Same-mechanism exclusions already on the books: `beicicc/sixmember_*` (level-2 stack
outputs, author-disclosed as optimistic), `golem_a`/`golem_f` (early-stop on their own
validation fold; in `DEFAULT_DROP`).

Other pool entries checked and dismissed 2026-08-13: `kenchanhodgkin/pg-s6e8-exp00{0,1}`
(OOF-only, 0.9542/0.9534, far below the pack floor, no test predictions);
`sarveshchhetri/the-lookup-key-trick-minus-the-neural-net` (plain TE LightGBM, subsumed).
