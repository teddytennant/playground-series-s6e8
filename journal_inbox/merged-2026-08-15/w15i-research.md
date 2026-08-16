# w15i — durable facts for RESEARCH.md

## ⚠ THE DEADLINE RECOMMENDATION, PRICED — this is the section to act on

Competition closes **2026-08-31 23:59 UTC**. `check_selection.py` exits **1**: nothing is
selected. If that is still true at the deadline Kaggle auto-selects on best **public** score.

### What to click

Go to https://www.kaggle.com/competitions/playground-series-s6e8/submissions and select:

| slot | file | cross-fitted CV | public |
|---|---|---|---|
| **1st choice** | **`blend159av_h3.csv`** (ref 55488344) | 0.97004917 | 0.97105 |
| **2nd choice** | **`blend160origm_h3.csv`** (ref 55488441) | 0.97004865 | 0.97105 |

Verify with `.venv/bin/python experiments/check_selection.py` (exit 0 = done).

**This is unchanged from 2026-08-13, and w15i re-derived it rather than repeating it.** The
pick was previously argued from raw CV alone. It now survives the correct objective as well:
Kaggle scores a selection as the **MAX over the selected entries**, so the quantity to
maximise is `E[max]` over the *pair*, not the CV of either file. `experiments/w15i_pick.py`
simulates the private slice (296,302-row pseudo-test out of the 691,369 labelled rows, f=0.20
cut away, 600 reps) for all **11 files in the CV-top cluster** — every one of which scored
exactly 0.97105, so the public reading carries nothing that separates them — and ranks all 55
pairs by `E[max]`:

- The current recommendation ranks **3 of 55**.
- The nominal optimum (`blendtop3` + `blend159av_wh3`) beats it by **+0.19e-6**, which is
  **ten times below the stack's own 2e-6 reproducibility floor** (w14a §1).
- The entire 11-file cluster spans 3.3e-6 of mean private AUC and the option value of *any*
  pair over the best single file is 0.2–0.5e-6.

> **The identity of the second slot does not matter. What matters is that neither slot is
> `blend158_logit`.** Do not re-open the pick on a 0.19e-6 basis — that is exactly the
> noise-chasing this workspace forbids.

### What the click is worth — one number, with its uncertainty

`experiments/w15i_cvlb.py` §5. Private gap of each candidate against the pick via the
partition identity `private = (g − f·public)/(1 − f)` (w14b, validated at β −0.2517 vs the
exact −0.2500), with `g` the recomputed cross-fitted CV gap and the transfer noise
`σ(g_test − g_cv)` measured by subsampling (4.2–17.3e-6 depending on the file).

| | expected AUC cost of the default | 90% CI | in places |
|---|---|---|---|
| **final-submission limit = 2** (Kaggle-standard) | **+9.2e-6** | [2.5e-6, 16.0e-6] | **~3.4** |
| **limit = 1** | **+36.5e-6** | [26.3e-6, 47.1e-6] | ~13.7 |
| limit = 1, worst branch (`blend158_logit` alone) | +112e-6 | — | ~42 |

`P(cost > 0)` is **0.988** at limit 2 and **1.000** at limit 1. Places are at the observed
local board density of **3.7 teams per 1e-5** (15 teams within ±2e-5 of our 0.97106).
Sensitivity to `f`: at f=0.25 the limit-2 cost is 10.5e-6, at f=0.50 it is 20.8e-6 — the
number only gets larger if f is bigger than assumed.

⚠ **Two assumptions, and neither is readable from the API.** `get_submission_limits` (see
below) returns daily counters only; `ApiGetCompetitionRequest` has no selection-limit field;
`GetCompetitionSettings` returns **403**; the overview and rules pages are 5,555-byte JS
shells with no SSR. So (a) limit = 2 and (b) private = best of selected remain Kaggle-standard
assumptions. Both branches are priced above, which is why the recommendation does not depend
on resolving them.

### The dilution hedge — live, small, and not a substitute

The 2026-08-13 hedge was retired by w14a in its strong form ("get two files strictly above
0.97106" — measured impossible, the h3 family is pinned at 0.97105 and ens4 tops out at
0.97106). **The weak form still works and had never been priced:** sending more *CV-good*
`ens4` files that land at 0.97106 enlarges the tie the default draws from and dilutes
`blend158_logit` out of it.

| tie set | limit-1 E[cost] | limit-2 E[cost] | worst |
|---|---|---|---|
| current 4 (`blend156`,`blend158`,`blend158_logit`,`blend159av`) | 35.5e-6 | 9.24e-6 | 112e-6 |
| + `blend160origm` (landed 0.97106 today) | 30.1e-6 | 8.93e-6 | 112e-6 |
| + `blend160origm` + `blend159` | **26.7e-6** | 9.00e-6 | 112e-6 |

**It helps only the limit-1 branch, by ~9e-6, and does nothing about the worst case.** This is
*not* the practice w14b §4 warns against: nothing is constructed for the slice, every file
sent is one CV already endorses at −4 to −6e-6 from the pick. It is a cheap use of slots that
must be spent anyway. **It is not a substitute for the click.**

---

## ⚠ THE CV→LB REGRESSION IS NOT DEAD — w14a fitted it with an omitted variable

**Supersedes RESEARCH's "the regression that earlier runs used to predict LB is, at the
resolution we now work at, dead" and JOURNAL 2026-08-14 w14a §2.**

w14a pooled all readings across transform families and got slope +0.148, R² 0.035 (w15i
reproduces +0.223, R² 0.083 with today's seven extra readings). But the *same run* measured a
per-family residual offset spanning **51e-6** — seven times the CV spread of the whole top
pack. Pooling a categorical that large into a regression whose regressor spans 1e-5 is the
textbook setup where an omitted variable destroys the within-group slope.

`experiments/w15i_pick.py` §1 refits with **transform-family fixed effects** on all 34
≥150-member triples (CV recomputed from each file's stored `submissions/oof_*.npy`, LB read
live off the API):

```
                          n=34 (fitted)      n=36 (refitted after today's two)
pooled                    +0.223  R^2 0.083  +0.248  R^2 0.098   resid sd 1.93e-05
within-family (FE)        +1.772 +/- 0.242   +1.771 +/- 0.226    t = +7.33 -> +7.84
                          within R^2 0.674   0.687   resid sd 5.82e-06 -> 5.66e-06
                                                     = 0.57 LB grid steps
```

Assumption-free version, immune to the quantisation and to the slope's functional form: of
the **38 within-family pairs the 1e-5 grid can resolve at all, 36 are concordant (94.7%)**,
one-sided binomial **p = 2.7e-9**. Per family: ens4 and rankraw perfect, logit 3/0, rescale
2/0, hybrid the only mixed one at 5/2.

The `ens4` family is now the cleanest demonstration in the workspace — 8 files, perfectly
separated by CV, with no reading in between:

| CV | 0.970032 | 0.970033 | 0.970034 | 0.970042 | 0.970043 | 0.970043 | 0.970044 | 0.970045 |
|---|---|---|---|---|---|---|---|---|
| LB | 0.97104 | 0.97104 | 0.97104 | 0.97106 | 0.97106 | 0.97106 | 0.97106 | 0.97106 |

**Three consequences, all of which change how the LB should be used here:**

1. **The public LB is a SHARP instrument, not a noisy one.** Within a family the residual is
   0.58 grid steps — the reading is essentially determined by CV. The workspace has spent two
   days treating a 19.3e-6 pooled residual as the LB's precision; the real figure for a
   like-for-like comparison is **5.8e-6**.
2. **Its resolution, in CV units, is ~5.6e-6** (one 1e-5 grid step ÷ 1.77). Above that the LB
   orders correctly (30/32); below it, files tie. **This explains the h3 invariance
   quantitatively rather than as a brute fact:** the 11 h3/w files span 3.3e-6 of CV, i.e.
   5.9e-6 of predicted LB, which is under one grid step — so they *must* all print 0.97105.
   w14a's "the public slice cannot resolve anything inside the h3 family" is right, and now it
   has a threshold attached and a rule that generalises to any future file.
3. **The slope is >1 (t = 3.2 against the null slope = 1), and errors-in-variables in CV can
   only attenuate it toward 0.** The natural reading is that cross-fitted OOF differences
   *under-state* true differences by roughly 1.8×, plausibly the known single-fold-OOF vs
   5-fold-averaged-test asymmetry compressing CV gaps. ⚠ **Flagged as a hypothesis, not a
   result** — it rests on 34 points with a quantised response, and it is not used in the cost
   numbers above (which would roughly double if it were).

**Two out-of-sample confirmations, both pre-registered, both exact.**

| file | CV | predicted by the ens4-family fit | returned |
|---|---|---|---|
| `blend160origm` (sent 12:48 by another w15 run) | 0.9700442 | 0.971061 → **0.97106** | **0.97106** ✓ |
| `blend159` (w15i's slot, ref 55528043, prediction written into the submission message *before* sending) | 0.9700434 | 0.971059 → **0.97106** | **0.97106** ✓ |

Together with the h3-family rule (now 8/8 at 0.97105, w14a's construction), the workspace's
LB predictions are **10/10**. The family fit is the instrument to use for this; the pooled
regression is not.

**Stability over time — the map is not drifting, but epoch-wise fits are useless.** Per-epoch
slopes are +0.057 (08-10/11, n=17), −0.068 (08-13, n=10), +0.000 (08-14/15, n=7): they swing
sign because each day's batch has a *different family composition*, not because the
relationship moves. Out of sample, fitting on 08-10/11 and predicting the 17 later readings
gives RMSE 2.45e-5 for the pooled regression — **worse than the trivial constant-gap null**
(LB = CV + 0.001008), which gets RMSE 2.30e-5 at a bias of +2.3e-6 against the regression's
+23.4e-6. So: *the pooled regression should never be used to predict an LB score.* Use the
family fit, or use the constant gap.

Per-family residual against the common fit, all 34 readings:

| kind | n | mean resid | sd |
|---|---|---|---|
| hybrid | 5 | −33.9e-6 | 17.2e-6 |
| rankraw | 6 | −9.2e-6 | 8.4e-6 |
| rescale | 3 | +3.8e-6 | 15.6e-6 |
| **w** | 3 | **+8.1e-6** | **2.1e-7** |
| **h3** | 8 | **+7.9e-6** | **3.0e-7** |
| ens4 | 6 | +10.1e-6 | 9.7e-6 |
| logit | 3 | +21.8e-6 | 14.0e-6 |

**h3-family LB invariance is now 8/8** at exactly 0.97105 over a CV span of 3.34e-6 (sd(LB)
= 0.00e+00), and the `w` family is 3/3 at the same value — **11 files, one grid step, zero
scatter.**

---

## The logit displacement: the "three replications" arithmetic, checked independently

**w14b's correction is confirmed. The three readings are one reading.** `w15i_cvlb.py` §4
re-derives the displacements from recomputed CVs and live LB, then re-runs the shared-slice
simulation from the stored OOF (400 reps, pseudo-test 296,302 rows, f=0.20 slice, all three
member sets scored on the *identical* slice each rep):

```
150fx  dCV -6.44e-05  dLB +4.0e-05  displacement +1.04e-04
150sx  dCV -6.07e-05  dLB +3.0e-05  displacement +9.07e-05
158    dCV -6.69e-05  dLB +3.0e-05  displacement +9.69e-05      mean +9.74e-05

per-set slice-deviation sd     3.12-3.20e-05    (reproduces w14b's ~30e-6)
pairwise correlation           +0.9770 +0.9767 +0.9750   mean rho +0.9762
=> n_eff = 3/(1+2*rho)         1.016 of 3
sd of the MEAN of the three    3.13e-05   (independent would be 1.82e-05)
```

Arithmetic verified: `n_eff = n/(1+(n−1)ρ) = 3/(1+2×0.976) = 1.016`. w14b's ρ = +0.992 →
n_eff 1.01; mine ρ = +0.976 → 1.016. **Two independent implementations, same conclusion: the
naive `sd/√3 = 3.96e-6` is wrong by a factor of ~8.**

**But state the other half, because w14b's phrasing invites under-reading it.** One
observation at +3.11σ is still a p = **0.0025** one-sided reading (`P(shared draw ≥ observed)`
= 1/400 in simulation; +3.09σ after folding in the LB's 1e-5 quantisation). "One reading, not
three" is a correction to the *evidence count*, not a dismissal. What does weaken it is
**multiplicity**: the contrast was selected post hoc from among 6 transform families
(Šidák → p ≈ 0.015) or 15 transform pairs (p ≈ 0.037). So:

> **The logit displacement is a single, post-hoc-selected, ~3σ slice draw — real enough that
> it should not be called noise, weak enough that it cannot carry a deadline decision.**

And it does not have to: w14b's partition arithmetic makes the decision the same under every
value the evidence permits. With φ the fraction of the +97.8e-6 that is a genuine full-test
property of the transform, `blend158_logit`'s private gap against the pick is −112e-6 at
φ=0, −51e-6 at φ=0.5, and only reaches break-even (+10e-6) at φ=1 — a branch `oofsim`
excludes at ~37σ.

---

## New API capability: `get_submission_limits` — read the day's usage without submitting

`CompetitionApiClient.get_submission_limits` (kagglesdk, via the CLI's own venv at
`/home/nixos/.local/share/uv/tools/kaggle/bin/python`) returns the live counters:

```
limited_by_total = False
num_allowed_now  = 4        <- remaining today
num_today        = 6
num_total        = 39
```

This is authoritative and free — no need to burn a submission to read the CLI's
"N remaining today" line, and no need to count rows in `submissions -v` and guess at the UTC
boundary. **It does not carry a final-selection limit field**, and `get_competition_settings`
returns **403**, so the selection limit stays unreadable.

Also confirmed from `ApiGetCompetitionRequest`: `awards_points = False` — this Playground
episode awards **no ranking points and no medals**. The stake is placement and swag, which is
worth knowing before pricing how hard to push for the selection click.

## Housekeeping: `audit_results.csv` is stale again

Seven scored files are missing their `lb`: `blend159_h3`, `blendtop3`, `blend156_h3`,
`blend159av_wh3`, `blend156w2`, `blend159av_w`, `blend160orig_h3` (plus `blend159`,
`blend160origm` from today). `w15i_cvlb.py` reads the LB **live from the API** and recomputes
CV from the stored OOF, so it cannot go stale; prefer it to `audit_results.csv` for any CV→LB
work. Cross-check on the 36 shared names: **max |ΔCV| = 2.2e-16** — the recomputation
reproduces the recorded CV exactly, so only the LB column had drifted.
