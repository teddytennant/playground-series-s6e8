# w15b — the generator, the ceiling, and what a leaderboard gap costs

## ⚠ Correction to the generator story in RESEARCH.md

RESEARCH.md's "The generating rule" section says the generator smeared a crisp
two-threshold rule (`social > 4`, `daily > 8` / `<= 6`) into a ramp, and that "86.3% of
the real data is decided outright by two thresholds". **That is a true statement about
the 7,500-row original and a false one about the competition frame**, and the difference
is large enough to have misdirected two missions.

Measured on the frozen folds (`experiments/w15b_surface.py`), all AUCs on the *identical*
502,260-row subset where both rule drivers are observed:

| model | AUC (all 691k rows) | AUC (both drivers observed) |
|---|---|---|
| LightGBM on `social` + `daily` only | 0.914749 | 0.935111 |
| + `weekend_screen_time` | 0.936223 | 0.948971 |
| all 9 numeric columns | 0.963991 | 0.969833 |
| all 12 raw columns | see w15b_surface.json | " |
| **the pack `blend159av_h3`** | **0.970049** | " |

The two rule drivers, *given to a model at full lattice resolution with 553k training
rows per fold*, are worth 0.935 on the rows where they are fully observed. The pack is
35,000e-6 above that. **The label in the competition frame is not a two-variable object.**
Whatever the generator did, it distributed the label's dependence across essentially all
12 columns. Do not price a hypothesis off the two-threshold rule again.

Consequence for the mission that produced this file: "fit P(y | exact lattice cell) and
read off the ceiling" cannot work. A cell oracle on the rule drivers is a *floor* 35e-3
below where we already are, not a ceiling.

## The calibration identity — a free, exact instrument

If `y_i ~ Bernoulli(p_i)` independently, then for any score `s`

```
E[#pos * #neg * AUC] = sum_{i!=j} p_i (1-p_j) [1{s_i>s_j} + 0.5*1{s_i=s_j}]
E[#pos * #neg]       = (sum p)(sum 1-p) - sum p(1-p)
```

so the AUC achievable by any score against a Bernoulli(p) field is a closed form in
`p` and `s` alone — **no labels needed**. Setting `s = p` gives the Bayes AUC `A*(p)` of
the field, computable in O(n log n) by sorting. Implemented as
`experiments/w15b_price.py:bayes_auc`.

**Code control (must pass before any use): in-sample isotonic is calibrated by
construction, so `A* - observed` must be 0. Measured `-0.000000`.** Two in-sample
binned-PAVA arms give -13e-6 and +8e-6. The estimator is right.

### The identity's first consequence, and it is the one that kills the mission's framing

**For a calibrated score, `A*(s)` IS the observed AUC.** The Bayes AUC of our own
probability field equals the AUC we already measure. So *nothing about our own
predictions can reveal a ceiling above us* — the ceiling question is entirely a question
about what is MISSING, and no re-analysis of the pack can answer it. Any future run that
proposes to "estimate the achievable ceiling" from our own OOF is going to rediscover
this. It is a tautology, not a measurement.

### The identity's second consequence — a calibration diagnostic worth reusing

`A* - observed` is a **signed over-dispersion meter** for any probability field:
positive means the field is more spread than its discriminative power justifies.
`experiments/w15b_calib.py` sweeps calibrators on `blend159av_h3`:

| calibrator (all cross-fitted on frozen folds) | observed | A* | A* - obs | Var(p) |
|---|---|---|---|---|
| isotonic (sklearn, the obvious choice) | 0.969989 | 0.970106 | **+0.000117** | 0.143716 |
| binned-PAVA B=50 | 0.970035 | 0.969810 | -0.000225 | 0.143400 |
| binned-PAVA B=100 | 0.970024 | 0.969991 | **-0.000033** | 0.143595 |
| binned-PAVA B=200 | 0.970008 | 0.970046 | +0.000037 | 0.143655 |
| binned-PAVA B=2000 | 0.969992 | 0.970093 | +0.000101 | 0.143704 |
| binned-PAVA B=20000 | 0.969990 | 0.970103 | +0.000113 | 0.143713 |

**Plain cross-fitted isotonic is over-dispersed out of fold by +117e-6** — PAVA chases
noise into singleton blocks at the extremes and the `y_min/y_max` clip makes those blocks
maximally dispersed. The gap crosses zero at **B ≈ 130**. Use
`experiments/w15b_phat.npy` (B=100) as the calibrated field, not raw isotonic, whenever a
probability (not a ranking) is needed here.

Note `Var(p)` moves only 0.2% across *every* calibrator. **Dispersion is robustly
determined even where the AUC identity is not**, which is why the price ladder below is
stable while its baseline is not.

## The price of a leaderboard gap, in units of missing signal — `experiments/w15b_price2.py`

### ⚠ Two errors to avoid, both of which this workspace made first

`w15b_price.py` (superseded) priced a gap as the rise in the CEILING,
`A*(p_tau) - A*(p_hat)`. **That is the wrong difference and it understates tau by ~2.7x.**
Orthogonal signal does two things at once: it raises the ceiling *and* it degrades our own
ranking, because our score is now missing something. At tau = 0.36 the ceiling rises
+168e-6 while the pack's own score falls ~1069e-6 — the competitor's advantage is
**+1237e-6**, not +180e-6.

The second error hides inside the first: `p_tau = expit(logit(p_hat) + tau*z)` is
**inconsistent with our own observed AUC** for any tau > 0. If the truth carried that much
orthogonal signal we would be scoring 0.9689, not the 0.9700 we measure.

### The correct construction

Index the family of truths CONSISTENT WITH OUR MEASUREMENT by their orthogonal content:

```
p*(k, tau) = expit( k * logit(p_hat) + tau * z ),   z ~ N(0,1) independent
solve k(tau)  s.t.  auc_under(p*, p_hat) == our observed OOF AUC
gap(tau) = A*(p*) - observed AUC
```

Sharpness `k` rises just enough to keep our own score where it actually is while the
orthogonal component grows. Boundary check: at tau=0 the solver returns **k = 1.00000** and
gap = -33e-6, which is exactly the residual `A* - observed` of the chosen calibrator — the
right boundary condition and a check on the solver. AUC is rank-only and both `p_hat` and
`k*logit(p_hat)` are monotone in `logit(p_hat)`, so the Jensen mismatch between
`expit(k*lp)` and `E_z[p*]` cannot affect anything.

| tau | k solved | ceiling A* | **GAP over us** | solo AUC of z |
|---|---|---|---|---|
| 0.05 | 1.00109 | 0.970049 | +0.000025 | 0.5041 |
| 0.10 | 1.00228 | 0.970123 | +0.000099 | 0.5087 |
| 0.15 | 1.00427 | 0.970243 | +0.000219 | 0.5130 |
| 0.20 | 1.00716 | 0.970409 | +0.000385 | 0.5169 |
| 0.30 | 1.01518 | 0.970869 | +0.000845 | 0.5250 |
| 0.45 | 1.03340 | 0.971857 | +0.001833 | 0.5374 |
| 0.65 | 1.06796 | 0.973589 | +0.003565 | 0.5520 |
| 0.90 | 1.12575 | 0.976161 | +0.006137 | 0.5688 |

**Inverted onto the board — the reusable table:**

The chosen calibrator leaves a -33e-6 residual at tau=0 where a perfect one would give 0,
so both the raw and the offset-corrected inversions are given. **The true value is bracketed
by the two columns** — that spread is the honest precision of the whole exercise.

| target | tau (raw) | solo (raw) | tau (corr) | solo (corr) | **bracket** |
|---|---|---|---|---|---|
| +18e-6 rayk anti-student (author's low) | 0.044 | 0.5037 | 0.016 | 0.5016 | **0.502-0.504** |
| +36e-6 rayk anti-student (author's high) | 0.057 | 0.5048 | 0.031 | 0.5028 | **0.503-0.505** |
| +50e-6 one CV noise floor | 0.067 | 0.5057 | 0.043 | 0.5036 | 0.504-0.506 |
| +110e-6 gap to LB rank 2 (0.97117) | 0.105 | 0.5091 | 0.086 | 0.5074 | **0.507-0.509** |
| **+180e-6 gap to MILANFX (0.97124)** | 0.134 | 0.5116 | 0.120 | 0.5104 | **0.510-0.512** |
| +321e-6 same at 56% CV→LB pass-through | 0.181 | 0.5154 | 0.171 | 0.5147 | 0.515-0.515 |

**Independence is the CONSERVATIVE assumption** — a missing signal correlated with what we
hold buys *less* AUC per unit of dispersion, so these are LOWER bounds.

### Robustness — `experiments/w15b_price_robust.py`

The obvious objection is that this is a calibrator artefact, since the calibrator moves the
baseline `A*` by ~340e-6, about twice the effect being priced. It is not: the ladder inverts
a *dispersion* question and `Var(p)` moves only 0.2% across the sweep. Re-derived on four
fields spanning under-, correctly- and over-dispersed calibrations, the inversion is stable
to **±0.001 in solo AUC** at every target (MILANFX: 0.5310/0.5317/0.5310/0.5320 on the
uncorrected ladder; the correction shifts the level, not the stability).

### What this buys you — read this before proposing anything

The leader's edge is **not** "an entirely new column". It is a *modest* orthogonal nudge:
a predictor with standalone AUC ~0.511 that is genuinely independent of all 160 members.
For scale, `raykkretzschmar`'s transductive anti-student correction, at its author's own
claimed OOF value of +18e-6 to +36e-6, sits at solo AUC **0.504-0.505** — i.e. one such
signal covers roughly **10-20%** of the gap to MILANFX. Closing it needs ~3-7 independent
signals of that class, or one about 2.4x stronger in tau.

To price any future gap, **interpolate this table — do not re-derive it**, and do not use
the superseded `w15b_price.py` inversion.

## The pack has already absorbed the exact quantisation lattice

`experiments/w15b_lackfit.py`. For each exact lattice cell, `z_c = (O_c - E_c)/sqrt(V_c)`
with `O` the positive count, `E = sum of the calibrated pack score`, `V = sum s(1-s)`.
Under "the pack is right inside this cell" the `z_c` are N(0,1) whatever the cell rate is.
Null = **size-matched cell permutation** (same cell count, same sizes, shuffled
membership), which absorbs global miscalibration and the size distribution.

| lattice | cells | df | T/df (real) | T/df (permuted) | z |
|---|---|---|---|---|---|
| social+daily, n>=20 | 221,008 | 1,463 | **0.6706** | 1.000 | **-9.6** |
| daily only, n>=20 | 1,390 | 1,164 | **0.5400** | 1.000 | **-11.2** |
| social only, n>=20 | 722 | 642 | **0.4984** | 1.005 | **-9.8** |

**The permuted arm lands at T/df = 1.000 to three decimals in every row** — the cleanest
null this workspace has produced, and it validates the statistic and the calibration
simultaneously.

The real partition is **UNDER-dispersed by roughly 2x**: residuals inside real lattice
cells are *half* as variable as independent Bernoulli sampling permits. That is only
possible because the pack's full-resolution target encoding, computed on the 4 training
folds, makes each row's OOF score track the realised label counts of the *other* rows in
its own cell. **The pack is not missing the lattice structure; it is tracking it harder
than binomial noise allows.** Confirmed independently by blending: adding the out-of-fold
cell-rate oracle to the pack gives real-minus-permuted `+1e-6` at best and `-277e-6` at
weight 0.35.

⚠ Corollary worth carrying: the raw per-lattice-value rates look wildly non-smooth
(inside `social<=4`, `daily` 6.87 -> 0.683, 7.18 -> 0.473, 7.38 -> 0.839 at n~500 and
se~0.02, i.e. 10-15 sigma between neighbours — see `experiments/w15b_surface_cells.csv`).
**Do not read that as unexploited signal.** It is real, and it is already in the pack.

### The mechanism, measured rather than asserted — `experiments/w15b_tecause.py`

Same statistic per member, each with its own size-matched permuted control. The workspace's
naming convention splits the recipes for free (`*_lat*` carry lattice target encoding,
`*_raw*`/`*_native` do not):

| member | recipe | solo AUC | T/df (daily cells, n>=20) | own permuted null |
|---|---|---|---|---|
| **PACK `blend159av_h3`** | 159-member stack | 0.970049 | **0.5400** | 1.014 |
| `xgb_lat` | lattice TE | 0.967664 | 1.4185 | 0.983 |
| `xgb_latcat` | lattice TE | 0.967696 | 1.4321 | 0.993 |
| `lgbm_tuned_lat` | lattice TE | 0.967712 | 1.4906 | 0.987 |
| `cat_lat` | lattice TE | 0.966353 | 2.2799 | 1.001 |
| `cat_native` | raw | 0.958941 | **4.7928** | 0.989 |
| `xgb_raw_nan` | raw | 0.965152 | **4.8713** | 0.987 |
| `cat_raw` | raw | 0.963075 | **7.0904** | 0.995 |
| `orig_binm` | raw, orig-trained | 0.886379 | **34.0539** | 0.998 |

Every permuted null lands at 0.98-1.01, so the ladder is not an artefact of member
strength. Read top to bottom it is a clean dose-response in three stages:

1. **The lattice structure is real and large.** Raw-feature members are over-dispersed
   inside `daily` cells by **4.8-7.1x** — they cannot represent the per-lattice-value
   surface at all, and their residuals say so.
2. **Target encoding captures most of it**, dropping T/df to 1.4-2.3 — but *not all of it*.
   Individual TE members still carry genuine lattice lack-of-fit.
3. **Only the 159-member stack goes past 1.0, to 0.54.** Averaging 159 members that each
   track their cell's realised count with independent noise produces an `E_c` that tracks
   `O_c` more tightly than Bernoulli sampling permits.

Stage 2 is a durable and previously unrecorded fact: **a single TE member does not fully
absorb the lattice; the stack is what finishes the job.** That is a concrete mechanism for
part of the stack's advantage over its best member.

## Leaderboard shape (2026-08-15 ~12:00 UTC)

MILANFX 0.97124 (unmoved since 2026-08-10) is **7e-5 clear of rank 2**, while ranks 2-20
span only 11e-5 (0.97117 down to 0.97106) with adjacent ranks ~1e-5 apart. The leader is
an **outlier, not the top of a smooth field**: in five days none of 1,831 teams has come
within 7e-5. Price the gap to rank 2 (+11e-5, solo AUC ~0.524) separately from the gap to
MILANFX (+18e-5, solo AUC ~0.531) — they are different claims and only the first is
evidenced by more than one team.
