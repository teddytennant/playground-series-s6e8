# w15c — durable facts for RESEARCH.md

## Row identity is arithmetically impossible here — CLOSED, do not re-open

Every row in this competition is unique and always was going to be.

| | count | of |
|---|---|---|
| distinct 12-predictor keys in train | **691,369** | 691,369 |
| train rows in a duplicate group | **0** | 691,369 |
| test rows with an exact train twin | **2** | 296,302 |
| test-internal duplicate rows | **0** | 296,302 |

Price it before ever measuring it again. Per-column `P(two random rows agree)`, NaN counted as
its own level, multiplies to **2.22e-17** — an effective joint cardinality of **4.5e16** against
only 2.05e11 train×test pairs, so the **expected** number of exact twin pairs is **0.005**.
The intuition that "45MB of train and 296k test rows on a small lattice makes collisions an
arithmetic certainty" is a birthday-paradox error: the relevant number is the *product* of the
column cardinalities (1389 × 721 × 600 × 451 × 401 × 231 × 166 × 18 × 1437 × 3 × 3 × 2), not any
single column's 166–231.

Coverage only becomes non-trivial once the key is down to ~5 columns (28.9%), and at that point
a "lookup" is just a target encoding — which `agent/features.py:175` (`te_block`) already
computes fold-safely for every single, pair and triple. High-order keys are the only untried
region and they have no coverage.

**Measured null on in-fold lookups** (`experiments/w15c_lookup2.py`, map built strictly inside
each fold): conditional AUC of the lookup given the stack score, pooled over 200 quantile bins,
against a 24-seed within-bin permutation control. Six subsets from 2 to 12 columns:
**max |z| = 2.34**. Nothing.

**`id` is finished from both directions.** Conditional AUC of raw `id` given the stack score
**0.498269**; `corr(id, y − p) = −0.000215`; one-way ANOVA of the residual across `id % m`
null at m ∈ {2,3,4,5,7,10,16,64,100,1000} (max |z| 2.39); label base rate across 20 contiguous id
blocks sd **0.00201** against a binomial 0.00244, i.e. *sub*-binomial. Complements w14b, which
asked the slice-structure question rather than the conditional-on-features one.

⚠ Gotcha that cost one instrument: **do not test residual structure by comparing class means to
zero.** `blend159av_h3` is a rank-average, not a calibrated probability, so `mean(y − p) = +0.2094`
and every class sits ~400σ from zero while being identical to every other class. Compare classes
to *each other* (ANOVA against a label-permuted control).

## The +1.0e-3 CV→LB gap is 88% a train/test missingness-allocation shift

**Train and test do not share a missingness distribution.** All twelve columns differ, up to
|z| = 44:

| column | train NaN | test NaN | diff | z |
|---|---|---|---|---|
| `app_opens_per_day` | 0.1167 | 0.0868 | −0.0300 | −44.0 |
| `social_media_hours` | 0.1938 | 0.1600 | −0.0338 | −39.8 |
| `daily_screen_time_hours` | 0.1386 | 0.1107 | −0.0280 | −37.9 |
| `academic_work_impact` | 0.0640 | 0.0868 | +0.0228 | +40.6 |
| `age` | 0.0418 | 0.0578 | +0.0160 | +34.5 |
| `work_study_hours` | 0.0745 | 0.0937 | +0.0192 | +32.2 |
| `notifications_per_day` | 0.0978 | 0.1155 | +0.0177 | +26.6 |
| `stress_level` | 0.0798 | 0.0662 | −0.0135 | −23.3 |
| `sleep_hours` | 0.0643 | 0.0758 | +0.0114 | +20.7 |
| `gaming_hours` | 0.1834 | 0.2005 | +0.0171 | +19.9 |
| `gender` | 0.0420 | 0.0480 | +0.0060 | +13.3 |
| `weekend_screen_time` | 0.1621 | 0.1711 | +0.0090 | +11.1 |

But the missing-count **per row** matches (mean 1.2589 train vs 1.2729 test; whole distribution
agrees to 3 dp). So the masking **budget** is conserved and its **allocation** is not.

Because the two rule drivers are observed *more* often in test, and driver-observed rows are far
more rankable (w14d: cell A 0.984 vs both-drivers-missing 0.912), **test is an easier mix**.
Reweighting the OOF pool to the test mask distribution by the exact 4096-pattern density ratio
(ESS 642,050 / 691,369 = 92.9%), over 30 files that have both an OOF vector and an LB score:

| | mean | sd |
|---|---|---|
| CV → LB gap, unweighted | **+0.001025** | 0.000051 |
| CV → LB gap, reweighted | **+0.000120** | 0.000049 |
| reweighting shift | **+0.000905** | **0.000003** |

Bootstrap over the 296,302 test rows (200 reps): **+0.000903 ± 0.000027**, 95% CI
[+0.000854, +0.000961] — **34σ**. The effect is additive per column: reweighting one column's
rate at a time gives `daily` +0.000773, `social` +0.000696, `app_opens` +0.000376 and negatives
everywhere the rate rose, and the **sum of the twelve is +0.000912 against a joint +0.000903**.

### …and it is decision-neutral. Do not build on it.

- Shift **sd 3e-6**, full range 12e-6, across 30 files spanning 4e-4 of CV.
- **Correlated −0.97 with CV** — collinear, so it carries nothing CV does not.
- **Argmax unchanged**: `blend159av_h3` tops both criteria (0.970049 / 0.970952).
- LB-prediction residual sd **identical**: `lb ~ cv` 0.000024, `lb ~ cv_w` 0.000024. Spearman vs
  LB 0.7214 plain vs 0.7048 reweighted — reweighting ranks very slightly *worse*.
- **Does not rescue `blend158_logit`**: reweighted gap +0.000195 vs `blend159av_h3`'s +0.000098.
  The logit displacement survives the correction, confirming w14b from a new direction.

Whether the shift's *favourable direction* is designed: column-shuffle control (same twelve
deltas, wrong columns, 300 reps) gives z = +1.68, P = 0.050. Mildly unusual, no more — **the
sign is a coin flip that landed heads.** Build no theory on it.

⚠ The permutation control in `logs/w15c_shiftctrl.log` (random pattern reassignment, sd 0.003029,
z = 0.9) is the **wrong null** and should not be cited as weakening this. It asks "would an
arbitrary reweighting move the AUC?" when the weights are computed from measured rates, not
chosen; permuting patterns destroys the weight function's smoothness and triples its variance.

## The entire train/test difference is the mask — the observed values are identical

`experiments/w15c_adv.py`, adversarial validation decomposed against an instrument floor:

| arm | n | adv AUC |
|---|---|---|
| mask only (12 NaN indicators) | 987,671 | **0.56472** |
| full frame (values + mask) | 987,671 | 0.56280 |
| **values only, COMPLETE ROWS** (mask constant, cannot leak) | 382,289 | **0.49750** |
| CONTROL: train-vs-train random half | 269,185 | 0.50059 ← floor |

`values − control = −0.00309`, below the floor. All twelve per-column marginals on complete rows
are null: nine KS p = 0.106…0.962, three chi2 p = 0.233…0.727.

**This corrects the public record.** `georgymamarin/s6e8-why-gaming-hours-helps-but-adds-nothing-new`
reports "adversarial train/test AUC of 0.57" alongside "missingness carries nothing about the
target". Split those: the 0.57 is **100% the missingness mask**, the observed values are
identical between splits to a 382k-row instrument's precision, and while the mask carries nothing
about the *target* it carries **+0.0009 of AUC level** through the difficulty mix.

Structural consequence: **train and test were masked separately with different per-column rates
from a common value-generating process.** They are not a random split of one masked pool. This is
the one place the generator was demonstrably sloppy — but the masking is MCAR w.r.t. the target
and the value distributions are identical, so there is no exploit in it.

## Reusable code

- `experiments/w15c_shift.py::wauc(y, s, w)` — weighted ROC AUC, O(n log n), correct tie
  handling. **Verified against `sklearn.roc_auc_score(sample_weight=...)` to machine precision
  (max |diff| 3e-15) on unit weights, random gamma weights and heavily-tied scores.**
- `experiments/w15c_shift.py::maskcode(df)` — missingness pattern as int 0..4095. Vectorised;
  the obvious `.astype(str).agg(''.join, axis=1)` takes minutes on 691k rows, this takes ms.
- `experiments/w15c_lookup2.py::cond_auc_core` — pooled within-bin Mann-Whitney AUC, the right
  instrument for "does feature X add anything the model does not already have".

## Operational

- The Bash tool times out at 120 s. `timeout 5400 python ...` does **not** help; launch with
  `nohup ... &` and poll the log, or the run dies at 2 minutes with exit 143.
