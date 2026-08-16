# w15g — what the OOF criterion can and cannot be biased by

## ⚠ The pooled-vs-test-mode gap of a K-fold OOF is a MARGINAL-MISMATCH quantity

Write the pooled OOF AUC as an exact sum over the 5×5 grid of fold pairs. With `P_k`/`N_l`
the positives of fold k and the negatives of fold l,

```
AUC_pooled = sum_{k,l} U(P_k, N_l) / (Npos * Nneg)
A_within   = sum_k     U(P_k,N_k) / sum_k     |P_k||N_k|      <- both rows scored by the
A_cross    = sum_{k!=l} U(P_k,N_l) / sum_{k!=l} |P_k||N_l|       SAME model, which saw
                                                                 neither. This is exactly
                                                                 the train->test setup.
```

`experiments/w15g_coupling.py:FoldPairAUC` computes all 25 `U(P_k,N_l)` from ONE sort of
the score vector (reduceat over runs of equal scores, then `posgrp.T @ (cum + 0.5*neggrp)`),
so re-doing it for a different fold assignment costs O(n). Gated against
`roc_auc_score`: **difference 0.00e+00**.

**The theorem, and it is what makes the decomposition interpretable.** All four steps
below are empirical identities — none of them assumes independence:

```
A_kl   = int Hhat_l dGhat_k                                (definition of the U-statistic)
Hhat_l = (Fhat_l - pi*Ghat_l) / (1 - pi)                   (definition of Fhat_l)
int Ghat_l dGhat_k + int Ghat_k dGhat_l = 1                (tie-aware symmetry)
int Ghat_k dGhat_k = 1/2                                   (tie-aware self-pairing)
=>  if Fhat_l is the SAME for every fold and the folds are equal-sized and stratified,
    mean_{k!=l} A_kl  ==  mean_k A_kk   EXACTLY.
```

So **`A_cross - A_within` is a function of the five folds' score marginals and of nothing
else.** Label coupling between folds — row i's score depending on row j's label through a
shared target-encoded lattice cell — cannot move a pooled AUC, because the coupled pairs
are O(n) out of the O(n²) pairs the statistic averages.

Verified rather than asserted, `experiments/w15g_identity.py`, at the most extreme dose the
mechanism admits (cells of exactly 5 rows one per fold, score = the leave-fold-out mean of
the other four labels, i.e. 100% of each score is other rows' labels):

| arm | pooled AUC | cross − within | after fold-normalisation |
|---|---|---|---|
| pure coupling, 100% dose | **0.497469** | −15.0e-6 | **+0.07e-6** |
| the same + a 0.02 shift on fold 0's scores only | 0.496752 | **−910.5e-6** | **+0.07e-6** |
| real signal + 0.0 × coupling | 0.801196 | −4.3e-6 | |
| real signal + 0.5 × coupling | 0.799269 | −3.1e-6 | |
| real signal + 2.0 × coupling | **0.777706** | −7.3e-6 | |

Read the last three rows: **adding coupling makes the pooled AUC WORSE, monotonically.**
It never inflates. And a deliberate marginal shift moves the statistic 60× more than
maximal coupling does, while fold-normalisation kills both to 0.07e-6.

## Fold-safe target encoding does not inflate OOF AUC — measured, with a positive control

`experiments/w15g_teleak.py`. Real lattice cells, real frozen folds, **permuted labels**, so
the cells carry exactly zero true signal and the leave-fold-out cell mean is a pure
realised-count tracker. Score the permuted labels with it and read the pooled OOF AUC.

| arm | AUC | deviation from 0.5 |
|---|---|---|
| `social+daily`, smooth 20 (te_block's setting), **fold-safe**, n=400 | 0.500001 | **+0.9e-6 ± 52.0e-6 (z +0.02)** |
| same, encoder fitted on ALL folds (**positive control**) | 0.781131 | **+281,131e-6 (z +1151)** |

The instrument detects textbook target leakage at z > 1000 and reads **zero** for the
fold-safe recipe. Any future claim that our OOF is optimistic because of within-cell count
tracking has to get past this arm first.

## What the pooled-vs-within gap actually IS: five rulers, and it is a TE effect

`experiments/w15g_foldscale.py` splits `A_within - A_pooled` into the part a per-fold rank
map removes (`scale`) and the part it does not (`coupling`, which the theorem above forces
to zero). Control = a stratified re-partition of the same rows into five pseudo-folds.

| member | recipe | pooled | scale e-6 | ctrl scale | coupling e-6 |
|---|---|---|---|---|---|
| `cat_lat` | CatBoost + our full-resolution TE | 0.966353 | **+21.42** | −0.13 ± 0.92 | +0.00 |
| lib `lat_cat` | lib TE | 0.967007 | +12.08 | +0.01 ± 0.96 | +0.00 |
| lib `lookup` | exact-cell lookup | 0.968526 | +11.07 | −0.04 ± 0.66 | −0.07 |
| lib `lat_lgbm` | lib TE | 0.967400 | +9.35 | −0.46 ± 0.76 | +0.00 |
| `lgbm_stump_lat_frac` | TE, stumps | 0.967349 | +7.38 | −0.22 ± 0.95 | +0.00 |
| `cat_native` | CatBoost, native lattice cats | 0.958941 | +6.52 | −0.50 ± 1.07 | −0.00 |
| `lat_xgb`/`xgb_lat`/`lgbm_tuned_lat`/`xgb_latcat` | TE | ~0.9675 | +4.3 … +5.4 | ~1.0 | +0.00 |
| **PACK `blend159av_h3`** | 159-member stack | 0.970049 | **+3.52** | −0.06 ± 0.50 | +0.00 |
| `linlat` | TE, linear | 0.961335 | +2.89 | −0.19 ± 0.95 | −0.00 |
| lib `lgbm` | no TE | 0.964494 | +1.92 | −0.36 ± 1.16 | +0.00 |
| `et_lat_frac` | TE | 0.960020 | +1.64 | −0.19 ± 0.98 | +0.00 |
| lib `cat` / `cat_raw` / `xgb_raw_nan` / `logreg` | no TE | — | +1.1 … +1.5 | ~1.0 | +0.00 |
| `xgb_cat_lattice` | unordered cats, no TE | 0.961074 | **+0.69** | −0.23 ± 0.70 | −0.00 |
| `orig_binm` / `orig_bin` / `w15d_origrep_r` | **zero dose** (orig-trained) | — | +1.1 … +2.5 | ~1.8 | +0.00 |

Three things this pins down:

1. **A clean dose-response.** TE members sit at +4 to +21e-6 (z 4 to 23). Non-TE members
   sit at +0.7 to +1.9e-6 (z 1 to 2, i.e. null). The CatBoost family gives the cleanest
   ladder because it holds the algorithm fixed: `cat_raw` (no target statistics at all)
   **+1.31** → `cat_native` (ordered target statistics on all 12 lattice columns) **+6.52**
   → `cat_lat` (our TE pipeline underneath) **+21.42**.
2. **The zero-dose anchors calibrate the null.** `orig_binm` etc. are fitted on the
   7,500-row original, so the partition never entered them and their `scale` must be a pure
   draw. They read +1.1 to +2.5 against a control sd of ~1.8, i.e. z ≈ 1. The instrument's
   null is where it should be.
3. **Blending averages it away.** The 159-member pack reads **+3.52e-6** — below the 5e-5
   measurement noise floor and barely above w14a's 2e-6 stack reproducibility floor.

So the pooled OOF criterion is **pessimistic by 3.5e-6** for our stack. That is the entire
internal bias of the criterion, and it is not a number.

## The coupling-free criterion is pooled CV minus a constant — it decides nothing

`experiments/w15g_criterion.py`, all 69 blend OOFs held here:

| file | pooled CV | A_within | bias e-6 | rank pooled | rank within |
|---|---|---|---|---|---|
| `blendtop3` | 0.9700495 | 0.9700531 | −3.66 | 1 | **1** |
| `blend159av_wh3` | 0.9700493 | 0.9700527 | −3.41 | 2 | 3 |
| `blend159av_h3` | 0.9700492 | 0.9700527 | −3.53 | 3 | 2 |
| `blend160origm_h3` | 0.9700487 | 0.9700521 | −3.41 | 4 | 5 |
| `blend158_h3` | 0.9700483 | 0.9700522 | −3.92 | 5 | 4 |
| `blend159av_w` | 0.9700482 | 0.9700516 | −3.39 | 6 | 6 |
| `blend156_h3` | 0.9700461 | 0.9700502 | −4.10 | 12 | 10 |

**argmax is the same file under both criteria**; spearman(pooled, within) over all 69 blends
is **0.9976**; and the bias is −2.84 to −4.10e-6 across the whole top cluster, a spread of
1.3e-6 which is *below* the stack reproducibility floor. The rank moves are ±1–2 places
among files separated by less than 1e-6 of CV, i.e. inside w14a's floor.

**Do not build a "coupling-free" selection criterion. It is the existing one shifted by a
constant.** The deadline picks are unaffected.

## ⚠ The mechanism that IS the right size for the residual CV→LB gap: the OOF/test bagging asymmetry

⚠ **Not a new observation — a newly priced one.** `RESEARCH.md`'s `sd_ratio` section already
states "the OOF array is one model's output per row, while the test array is the mean of five
fold models", and uses it to explain why saturating members show `sd_test/sd_oof < 1`. Nobody
had converted it into AUC, or noticed it is the right size to be the residual CV→LB gap.

`agent/run_lgbm.py:116-118` — and this is the standard 5-fold convention, so it holds for
the public library members too:

```python
oof[iva] = pb                 # ONE model, trained on 80%
tp += pt / N_SPLITS           # the MEAN of FIVE such models
```

**Every member's test column is a five-model bag; its OOF column is a single model.** The
two are not the same estimator, and the difference is variance reduction, which is worth
real AUC. Measured on the frozen folds from the three `xgb_latcat` seed twins already on
disk (13/17/23):

| models averaged | OOF AUC | gain over one |
|---|---|---|
| 1 (mean of the three singles) | 0.967738 | — |
| 2 (mean over the 3 pairs) | 0.967863 | +125.0e-6 |
| 3 | 0.967904 | +166.9e-6 |

The variance-reduction law `gain(m) = G(1 − 1/m)` gives **G = +250.0e-6 from m=2** and
**+250.4e-6 from m=3** — a two-point extrapolation that agrees to 0.4e-6, so the law holds
exactly. At the m=5 our test side actually gets: **+200e-6.** Fold models differ in
*training data* as well as seed, so they are more diverse than seed twins and this is a
lower bound.

Direct end-to-end confirmation, `experiments/w15g_cvgap.py` — rebuild the whole geometry
inside the labelled rows (TRAIN 70% / HOLD 30%, stratified and random, so w15c's +905e-6
missingness confound is switched off by construction), then compare four ways of scoring the
same model on the same 207k HOLD rows. See `journal_inbox/w15g.md` for the table.

**Why this matters and the previous mechanism did not.** w15c measured 88% of the standing
+1.0e-3 CV→LB gap as the missingness-allocation shift, leaving **≈ +98e-6**. w15b's proposed
mechanism for that residual is measured here at **+0.9e-6 ± 52e-6** and is forbidden by the
theorem above. The bagging asymmetry is **+200 to +300e-6 for a single member**, i.e. the
first mechanism anyone has named that is at least as large as the thing it must explain.

⚠ **The honest limit:** I have not measured how much of it survives 160-member blending.
The argument that most of it does is that fold-f's training subsample is COMMON to every
member, so the fold-model idiosyncrasy does not average out across members the way seed
noise would — but that is an argument, not a measurement, and a run that wants the number
must build a multi-member version of `w15g_cvgap.py`.

## Decision consequences — there are none, and that is the point

- The bias is the same sign and nearly the same size for every file we hold, so it cancels
  out of every comparison. **The deadline picks are unchanged: `blend159av_h3` +
  `blend160origm_h3`.**
- It reprices nothing on the leaderboard either: every team's pipeline has it.
- What it does retire is the *mystery*. A CV→LB gap with no mechanism attached is the kind
  of thing that gets used to justify an exotic theory. It now has one, and it is boring.
