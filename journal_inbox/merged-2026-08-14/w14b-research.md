# w14b — durable facts for RESEARCH.md

## The public/private partition arithmetic — the thing that settles the logit question

The public and private slices **partition one 296,302-row test set**. So for any two files,
the paired AUC difference on the private slice is *forced* by the full-test difference and
the public reading:

```
private_gap  ~=  (g - f * public_gap) / (1 - f)          f = public slice fraction
```

`experiments/w14b_slicenoise2.py` §4 validates this empirically on labelled data: regressing
the private deviation on the public deviation gives **beta −0.245 (corr −0.998)** against the
exact-partition prediction of −f/(1−f) = −0.250, at f = 0.20. It holds to ~2%.

**Consequence, and it is the important one: a public-LB gain that is not backed by the true
full-test difference is not free — it is BORROWED from the private slice at 4:1.** A file
flattered by δ on the public slice is penalised by δ·f/(1−f) on the private one. This is a
sharper statement of the workspace's standing "select on CV" rule and it has a number in it.

⚠ **Do not build a file designed to top the public LB.** The 2026-08-13 plan's hedge ("get
two CV-good files above 0.97106 so the default cannot pick `blend158_logit`") is sound only
while every candidate is *already* CV-good. Deliberately leaning a blend toward whatever the
public slice likes would buy public rank and pay 4× for it on private.

## `blend158_logit`'s exposure is WORSE than −88e-6, not better

Applying the above to the live risk (nothing is selected; Kaggle auto-selects on best
public, and `blend158_logit` is tied for best public at 0.97106):

| assumed real transform effect | full-test gap `logit − h3` | ⇒ private gap |
|---|---|---|
| 0 | −86.9e-6 | **−111e-6** |
| +20e-6 (2σ ceiling from `oofsim`) | −66.9e-6 | −86e-6 |
| +40e-6 | −46.9e-6 | −61e-6 |
| +97e-6 (whole displacement real) | +10e-6 | +10e-6 |

Only the last row rescues it, and that row is excluded below. The journal's "−88e-6 / ~10σ"
is the *CV* gap; the **predicted private gap is ~−111e-6**. Robust to the LB's 1e-5
quantisation (public gap anywhere in 0..+20e-6 moves it only −109 to −114e-6).

## `oofsim` OVER-doses the clip mechanism by 5.7× — its null is stronger than it was read

The mechanism (logit's clip pins more single-fold OOF cells than 5-fold-averaged test cells)
was tested by `oofsim.py` and came back flat. That was read as "the test found nothing".
It is better than that — the test was **over-powered**, not merely null:

| pack | mean OOF pinned | mean test pinned | **aggregate asymmetry** |
|---|---|---|---|
| real, 161 members (`experiments/clip_census.csv`) | 9.884% | 9.531% | **+0.353pp** |
| `oofsim` dose-4, 10 members (`cache/oofsim/results_s7.json`) | 6.378% | 4.356% | **+2.022pp** |

**5.72×.** `oofsim` concentrates rf/et-grade asymmetry into 4 of 10 members; the real pack
dilutes comparable per-member asymmetry (worst real +8.8pp vs oofsim's +9.7pp) across 161.
So if the mechanism were real and linear in aggregate asymmetry, oofsim would have to show
**~5.7× the real-pack effect**. It shows −15e-6 ± 14e-6 at full dose (3 seeds pooled) —
which scales *down* to **−2.6e-6 ± 2.5e-6** for the real pack, against a +97e-6 claim.
**The clip mechanism is excluded by a factor of ~40, not merely unconfirmed.**

## The public slice behaves like a uniform random subsample — no id structure

`experiments/w14b_idstructure.py`. If Kaggle cut the public slice contiguously in id order,
or the generator drifted with id, the slice's draw would be wider than a random subsample's
and the displacement would be cheaper than it looks. It does not:

| pair | sd across 24 contiguous id blocks | sd across matched random subsamples | ratio |
|---|---|---|---|
| logit − hybrid | 43.9e-6 | 41.3e-6 | 1.06 |
| logit − h3 | 36.4e-6 | 39.2e-6 | 0.93 |
| rankraw − hybrid | 43.3e-6 | 32.1e-6 | 1.35 |
| h3 − ens4 | 8.3e-6 | 9.6e-6 | 0.86 |

Drift test (correlation of the per-block deviation with block index): **+0.073**. Train is
confirmed strictly ascending in `id`. So the random-subsample sd is the right null.

## Slice-draw sd on a paired difference, at public-slice size

`experiments/w14b_slicenoise2.py` §1, 250 reps, pseudo-test 296,302 rows drawn from the
691,369 labelled OOF pool. At **f = 0.20** the paired sd is ~30e-6 for `logit − h3` and
~30e-6 for `logit − hybrid`; it reproduces the 2026-08-11 `cvlb2.py` paired-bootstrap
number (29e-6) independently, via a different construction. Non-logit contrasts
(`rankraw − hybrid`, `h3 − rankraw`) land at |dev| ≈ 2e-6, i.e. **the null is well
calibrated everywhere logit is not involved** — the anomaly is specific, not a general
failure of the instrument.

## The three "replications" of the displacement are ONE reading

The 150fx / 150sx / 158 readings (+104e-6 / +91e-6 / +97e-6) are taken off **the same fixed
public slice** with files correlated ~0.9999. Simulated on a shared slice, their deviations
correlate **+0.992** pairwise → **effective independent reads 1.01 of 3**. So the evidence
is a *single* ~3.3σ draw, not three. Any future argument that counts LB replications of a
fixed-slice effect as independent evidence is wrong for this reason.

## `oofsim` seeds 7 + 11 + 13 pooled — the deadline pick is confirmed, item closed

`oofsim_summary.py 7 11 13`, full dose:

| contrast | pooled | signs |
|---|---|---|
| `logit − hybrid` | **−0.000015 ± 0.000014** | 1/3 |
| `ens4 − h3` | **−0.000010 ± 0.000003** | **0/3** |

Pre-registered confirmation required 3/3 positive at full dose. It is 1/3 and negative.
`ens4 − h3` is negative on all three splits: **the labelled truth prefers `h3`**, which is
what raw CV said. Seed 13 changes nothing; **this item is now closed.**

Also from the pooled run: the OOF-searched simplex weight recovers the true optimum exactly
at doses 3 and 4 on all three seeds (search leaves +0.000000 on the table), which is an
independent confirmation that transform-weight search is retired for the right reason.
