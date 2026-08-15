# w14a — durable facts for RESEARCH.md

## The stack is numerically singular: the pipeline's reproducibility floor is ~2e-6 AUC

A shipped file **cannot be regenerated** from `oof/` + the stacker. Measured end-to-end
(`experiments/w14a_repro.py`, rebuilds `blend159av_h3` from the frozen folds):

| transform | stored CV | rebuilt CV | delta |
|---|---|---|---|
| logit | 0.969965 | 0.969965 | +0.000000 |
| hybrid | 0.970029 | 0.970026 | −0.000003 |
| rankraw | 0.970034 | 0.970031 | −0.000003 |
| rescale | 0.970029 | 0.970027 | −0.000001 |
| **h3** | **0.970049** | **0.970047** | **−0.000002** |

99.91% of test rows change rank; spearman vs the stored pick 0.9999748.

**Cause, diagnosed with controls** (`experiments/w14a_solver_probe.py`):

- Condition number of the 159-member design ≈ **1.47e18** (4.22e18 on a rerun — the smallest
  singular value is rounding noise). That is past 1/eps for float64 (4.5e15): **singular to
  working precision.** Members correlate 0.987–0.999 and at `C=1` over 553,095 rows the L2
  penalty is ~0, so nothing selects a point in the flat valley.
- **In-process the code is bit-deterministic** — fold 0 fitted twice gives identical
  coefficients, at both thread counts. Ruled out: nondeterministic code.
- **Across BLAS thread counts (1 vs 4, separate processes)**: `n_iter` 85 vs 93,
  max|Δcoef| 6.25e-3, cosine 0.999654, max|Δ decision_function| 5.28e-2 — but
  **Δ fold-0 AUC only +1.9e-6** and Δ train logloss −2.3e-6. Far apart in coefficient space,
  the same point in loss and AUC.
- Ruled out by evidence, not assumption: no member file in the 159 set was written after the
  original build (mtimes ≤ 2026-08-11 03:49); zero `ConvergenceWarning`, `n_iter` 85–93
  against `max_iter=5000`.

**Operational consequences:**

1. **Submit the stored CSVs at the deadline. Never rebuild-and-resubmit** — a rebuild is a
   different file scoring the same to ~2e-6.
2. **Any CV difference below ~2e-6 is not a difference.** The joint-top cluster
   (`blend159av_h3` 0.970049 / `blend160origm_h3` 0.970049 / `blend158_h3` 0.970048 /
   `blend159_h3` 0.970047) spans exactly one reproducibility unit and is genuinely unordered.
   Results that survive: `blend158_logit` −87e-6 (44 floors), `ens4 − h3` −10e-6 (5 floors).
3. `blend_lab --lam` would give a unique solution. **Not a lead** — "stacker `C`" stays
   closed on performance grounds; this is a property of the instrument, recorded so no future
   run reads the wobble as a modelling result.

## The CV→LB regression is dead as a predictive instrument (R² 0.035)

`experiments/w14a_cvlb.py` — reads pairs from `audit_results.csv` rather than hard-coding
them the way `predict_lb.py` does, so it stays current. 27 files, ≥150-member family:

```
slope +0.148   R^2 0.035   pearson +0.187   spearman +0.618
residual sd 2.12e-05 = 2.12 LB quantisation steps
gap (LB - CV) mean +0.001011  sd 0.000031
```

At slope +0.148 you need **+6.7e-5 of CV to buy one 1e-5 LB step**, and the whole top-of-pack
CV spread is 7.6e-6. Do not use `predict_lb.py`-style predictions at current resolution.

**What predicts LB is the transform family, not CV.** Mean residual against the common fit:
hybrid −32.0e-6, rankraw −6.8e-6, rescale +5.8e-6, h3 +11.7e-6, ens4 +13.3e-6, logit
+18.9e-6. The transform spread (51e-6) is **7× the CV spread of the entire top pack**.

## The public slice cannot resolve anything inside a transform family — 5/5 at one value

Every `h3` file ever sent has returned **exactly 0.97105**: `blend158_h3`, `blend159av_h3`,
`blend160origm_h3` (08-13), `blend159_h3` (w14b), `blendtop3` (w14a) — across CV
0.970046–0.970049 and files that are genuinely distinct (spearman 0.999995 vs the pick).
Residual sd across the h3 group is 6.4e-8, i.e. zero. The top `ens4` files (CV ≥ 0.970042)
return 0.97106, 3/3.

`blendtop3`'s 0.97105 was **pre-registered before submission** from this rule and came back
exactly. The public LB reads transform identity and nothing else.

⚠ **Retire the "get two CV-good files strictly above 0.97106" hedge (planned 2026-08-13).**
It is measured impossible: h3 files land 0.97105, the best ens4 files land 0.97106 (tie, not
exceed). Nothing this pack can build goes above 0.97106, and per w14b building *for* the
public slice is paid back 4:1 on private.

## h3 vs ens4 on the public slice = the logit displacement, not independent evidence

`ens4` is `h3` plus `logit`. At fixed member set the LB prefers `ens4` by exactly one grid
step (blend158 and blend159av, 2/2) while CV prefers `h3` by 4–5e-6. The mean-of-component-
gaps estimator (validated on 150fx in `predict_lb.py`) attributes +5.0e-6 of that +1.0e-5 to
logit re-entering — same sign, half the size, within one grid step. So the LB's preference
for `ens4` is the logit displacement under another name, and by w14b's 4:1 borrowing result
it is evidence *against* `ens4`, not for it. `h3` stands as the pick.

## Re-pricing the unset-selection risk: the 4-way public tie bounds the damage at ~10e-6

The "−88e-6 / ~10σ / this is the whole competition" framing carried by three entries priced
the *file* and never the *draw*. Best-public is a **4-way tie at 0.97106**:

| file | CV | kind |
|---|---|---|
| `blend156` | 0.970042 | ens4 |
| `blend158` | 0.970043 | ens4 |
| **`blend158_logit`** | **0.969961** | **logit** |
| `blend159av` | 0.970045 | ens4 |

Kaggle's default selects the best public submissions up to the final-submission limit (2 for
Playground) and scores the **best** of the selected on private. Any 2-of-4 draw from this tie
therefore contains at least one CV-good ens4 file, and the private result is the max — so the
default's worst case is the `ens4 − h3` gap, **~−10e-6** (w14b's labelled-truth pooled
figure), not `blend158_logit`'s −87e-6 CV / −111e-6 predicted-private gap.

⚠ Assumptions, neither readable from the API (`ApiListCompetitionsRequest` returns
`max_daily_submissions=10`, `max_team_size=3`, no selection-limit field): (a) limit = 2,
(b) private = best of selected. Both Kaggle-standard. If the limit were 1 **and** the
tiebreak latest-first, `blend158_logit` alone is selected and the −111e-6 is live.

**The toggle item is re-sized, not closed.** Still worth ~10e-6 and still the only way to
make the outcome unconditional on an undocumented tiebreak — but a future run should not
spend itself on the "10σ, whole competition" framing.

## Board, 2026-08-14

`team_count` **1831** (the brief's ~1,326 is stale), our `user_rank` **20** (was 13). Leader
MILANFX 0.97124; five teams passed 0.97106 in the preceding day. `user_rank` is available
directly off `ApiListCompetitionsRequest(search=...)`, no leaderboard pull needed.
