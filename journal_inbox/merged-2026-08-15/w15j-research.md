# w15j — durable facts for RESEARCH.md

⚠ **Merge order note.** The auto-selection section below was written before `w15i` landed.
**w15i's Monte Carlo pricing of the same risk supersedes my tiebreak enumeration** — quote
**+9.2e-6 (limit 2) / +36.5e-6 (limit 1)**, not my "−10e-6". My enumeration survives only as a
narrower statement: under every *natural* ordering rule the auto-pick is a CV-good `ens4` file,
so the +112e-6 worst branch requires an unnatural tiebreak. See `journal_inbox/w15i.md`.

⚠ **Before merging `w15b-research.md`: its "Leaderboard shape" section (lines 234–239) quotes
the SUPERSEDED solo-AUC figures 0.524 / 0.531. The corrected values from `w15b_price2.py` are
0.509 / 0.512.** Caught by w15g, verified independently here via `experiments/w15j_synth.py`.

## The public slice's resolving power is a function of rho, and the law is now confirmed on LIVE leaderboard data

w15a's central reframe — that there is no single "noise floor", only

    sd(paired slice gap) = sd_single * sqrt(2 * (1 - rho))

with `sd_single = 5.673e-4` — was established by resampling **labelled** rows. It had never
been checked against the leaderboard itself. This account holds 36 scored files spanning
rho 0.97 to 0.99999, which is a live test of it.

`experiments/w15j_lblaw.py` (pairs + ordering) and `w15j_lblaw2.py` (scale, quantiser-aware).
Method: take pairs of our own **sent** files whose cross-fitted CV differs by < 5e-6, so the
true full-test gap is ~0 and any observed LB difference is slice draw plus quantisation.
112 such pairs out of 630.

**Ordering (scale-free, does not depend on the unknown public-slice fraction):**

    spearman( 1 - rho , |dLB| ) = +0.7327   p = 4.2e-20   over 112 CV-matched pairs

**Scale.** Two estimator corrections that the first cut got wrong and that any future run
repeating this must make:

1. For a zero-mean gaussian difference `X`, `E|X| = sd(X)*sqrt(2/pi) = 0.798*sd(X)`. Comparing
   `sd(|X|)` against the law's `sd(X)` is the wrong statistic and is off by 1/0.603.
2. The LB is quantised to 1e-5. In the high-rho buckets the true gap is far under one grid
   step, so most pairs read `|dLB| = 0` exactly and any moment estimator is floored. Handled
   by forward-simulating the law **through** the quantiser rather than inverting it.

| bucket (1-rho) | n pairs | distinct files | law sd(X) | predicted E\|dLB\| | observed E\|dLB\| | ratio |
|---|---|---|---|---|---|---|
| <1e-5 | 10 | 13 | 2.13e-6 | 1.7e-6 | 1.0e-6 | 0.59 |
| 1e-5..1e-4 | 50 | 25 | 4.82e-6 | 3.83e-6 | 0.8e-6 | **0.21** |
| **1e-4..1e-3** | **36** | **27** | **1.42e-5** | **1.13e-5** | **1.28e-5** | **1.13** |
| **1e-3..1e-2** | **16** | **12** | **3.57e-5** | **2.85e-5** | **2.56e-5** | **0.90** |

**In the two buckets where the leaderboard can resolve anything at all, a prediction with
ZERO parameters fitted to the leaderboard lands at ratio 1.13 and 0.90.** `sd_single` came
from resampling labelled rows and was never tuned to any LB reading.

⚠ **Do not read the `1e-5..1e-4` bucket's ratio 0.21 as a failure of the law.** 45 of its 50
pairs come from the single 10-file top cluster, so it is ~1 independent read presented as 50 —
the same `n_eff` error w14b caught in the "3 replications" argument. Its z is not
interpretable. The two lower-rho buckets draw from 27 and 12 distinct files across different
transform families and are the informative ones.

**The high-rho endpoint, read directly off the board.** The 10 top-cluster files
(CV 0.970046–0.970049, pairwise rho >= 0.99992, median 0.99996) — `blend159av_wh3`,
`blend159av_h3`, `blend160origm_h3`, `blend158_h3`, `blend159av_w`, `blend159_h3`,
`blend156w`, `blend156w2`, `blend160orig_h3`, `blend156_h3` — all read **exactly 0.97105**,
LB sd **0**. The law predicts `sd(gap) = 4.8e-6` at that rho against a 1e-5 grid, so they are
*required* to read one value. Observed.

### Consequence: quote the right sigma

| team | public | gap to us | sigma at cross-team rho 0.9958 |
|---|---|---|---|
| MILANFX | 0.97124 | 180e-6 | **3.46** |
| Maher el Ouahabi | 0.97117 | 110e-6 | 2.12 |
| Don Mani | 0.97116 | 100e-6 | 1.92 |
| Optimistix | 0.97115 | 90e-6 | 1.73 |
| Utkarsh | 0.97113 | 70e-6 | 1.35 |

**Any future framing that quotes a single "5e-5 noise floor" for a cross-team comparison is
quoting the within-pack number and will overstate its own significance by ~10x.** The floor
for two files at rho 0.99999 is 2.5e-6; for two files at rho 0.9958 it is 53e-6. Same law.

## The auto-selection exposure is bounded at ~-10e-6 under EVERY enumerable tiebreak

`experiments/w15j_tiebreak.py`. RESEARCH re-sized the unset-toggle risk from -111e-6 to
~-10e-6 by noting best-public is a 4-way tie, but explicitly left one branch live:

> "If the limit were 1 **and** the tiebreak latest-first, `blend158_logit` alone is selected
> and the -111e-6 is live."

That branch is checkable and it is **false**. The tie, ordered by submission time:

| submitted | file | LB | CV | kind |
|---|---|---|---|---|
| 2026-08-11 03:35:44 | `blend156` | 0.97106 | 0.970042 | ens4 |
| 2026-08-13 17:16:16 | `blend158` | 0.97106 | 0.970043 | ens4 |
| 2026-08-13 17:16:26 | **`blend158_logit`** | 0.97106 | **0.969961** | **logit** |
| 2026-08-13 17:17:31 | `blend159av` | 0.97106 | 0.970045 | ens4 |

Six tiebreak rules x two selection limits, private gap vs the CV pick (`ens4 - h3 = -10e-6`,
`logit - h3 = -111e-6`, both w14b):

| rule | limit=1 pick | gap | limit=2 picks | gap (private = max of selected) |
|---|---|---|---|---|
| earliest submitted | `blend156` | -10e-6 | blend156 + blend158 | -10e-6 |
| latest submitted | `blend159av` | -10e-6 | blend159av + blend158_logit | -10e-6 |
| lowest ref id | `blend156` | -10e-6 | blend156 + blend158 | -10e-6 |
| highest ref id | `blend159av` | -10e-6 | blend159av + blend158_logit | -10e-6 |
| filename A-Z | `blend156` | -10e-6 | blend156 + blend158 | -10e-6 |
| filename Z-A | `blend159av` | -10e-6 | blend159av + blend158_logit | -10e-6 |

**Worst case across all 12 combinations: -10e-6. `blend158_logit` is selected uniquely under
NONE of them** — it is neither the earliest nor the latest, neither the lowest nor the highest
ref, and neither first nor last alphabetically, in a 4-way tie whose other three members are
all CV-good `ens4` files.

**Re-price the toggle accordingly.** It is still worth doing — an explicit selection makes the
outcome unconditional on an undocumented rule, and it buys the `h3` vs `ens4` 10e-6 — but
**five consecutive runs have opened with "this is the only live risk on the board" at a
-111e-6 price tag that is not reachable.** The honest figure is ~10e-6, which is two
reproducibility floors and comparable to everything else on the board. Stop leading with it.

⚠ This is contingent on the tie holding. It is stable unless a *future* submission scores
0.97107+ alone. Nothing this pack can build does (RESEARCH: 0.97106 is the measured ceiling),
and any file that did would be CV-good ens4 anyway.

### …and w15j's own submission tightened it further

`blend160origm` (CV 0.9700442, ens4) returned **0.97106** and joined the tie, making it
**5-way with four CV-good `ens4` files against one `logit`**. Re-running the enumeration:

| rule | limit=1 pick | limit=2 picks |
|---|---|---|
| earliest / lowest ref / A-Z | `blend156` | blend156 + blend158 |
| **latest / highest ref / Z-A** | **`blend160origm`** | **blend160origm + blend159av** |

**`blend158_logit` now appears in ZERO of the twelve enumerated outcomes.** Before this
submission the latest-first limit=2 draw was `blend159av + blend158_logit`; it is now
`blend160origm + blend159av`, both CV-good. That is a measured reduction in the standing
risk bought by a slot that would otherwise have gone to a near-copy, and it is the reason
to prefer the highest-CV never-sent **ens4** file over a lower-CV `hybrid` one when the
slot is otherwise free.

⚠ Note the distinction from w14b §4's warning, which stands: **nothing here was constructed
to chase the public slice.** `blend160origm` was selected as the highest-CV never-sent file
in the workspace; that it is `ens4` and therefore lands on the 0.97106 step is a property of
the file, not a lean. Building a file *for* the slice is still paid back 4:1 on private.

## The ens4 CV->LB ladder is a real step function — 4/4, and it was pre-registered

| CV | file | LB |
|---|---|---|
| 0.9700449 | `blend159av` | 0.97106 |
| **0.9700442** | **`blend160origm`** | **0.97106** (predicted before sending) |
| 0.9700432 | `blend158` | 0.97106 |
| 0.9700416 | `blend156` | 0.97106 |
| 0.9700343 | `blend153` | 0.97104 |
| 0.9700330 | `blend150sx` | 0.97104 |
| 0.9700319 | `blend150fx` | 0.97104 |

The step sits between CV 0.970034 and 0.970042. `blend160origm`'s 0.97106 was pre-registered
in its submission message from this ladder and came back exactly. This is the **only** family
where CV still buys an LB grid step; inside the top cluster (`h3`/`wh3`/`w`/`w2`, 10 files)
the LB is flat at 0.97105 and resolves nothing.

## The consolidated CV->LB map, current as of 39 submissions

`experiments/w15j_cvlb.py` -> `w15j_cvlb.csv`. Merges `audit_results.csv` CV against the LIVE
submission list, so the LB column stays current instead of going stale the way
`audit_results.csv`'s own `lb` column has.

| family | n sent | LB min | LB max | LB sd | CV span |
|---|---|---|---|---|---|
| **h3** | 6 | 0.97105 | 0.97105 | **0** | 3.1e-6 |
| **wh3** | 1 | 0.97105 | 0.97105 | — | — |
| **w** | 2 | 0.97105 | 0.97105 | **0** | 1.5e-6 |
| **w2** | 1 | 0.97105 | 0.97105 | — | — |
| ens4 | 6 | 0.97104 | 0.97106 | 1.1e-5 | 12.9e-6 |
| rescale | 3 | 0.97102 | 0.97105 | 1.7e-5 | 14.8e-6 |
| rankraw | 6 | 0.97102 | 0.97104 | 9.8e-6 | 14.6e-6 |
| hybrid | 6 | 0.9708 | 0.97103 | 8.4e-5 | 350e-6 |
| logit | 5 | 0.97081 | 0.97106 | 1.3e-4 | 320e-6 |

**New this run: the fitted-simplex families `w`/`w2`/`wh3` also read 0.97105.** The
"h3-family invariance" rule is better stated as: **every top-cluster file, whether it fits
zero, two, or three parameters above the stack, reads 0.97105.** 10/10. The invariance is a
property of the *cluster*, not of the h3 transform, and the rho-law above is why.

The `ens4` ladder is a genuine step function and is the one family where CV still buys an LB
grid step: CV >= 0.970042 -> 0.97106 (3/3), CV <= 0.970034 -> 0.97104 (3/3).

## Board, 2026-08-15 ~13:00 UTC

`kaggle competitions leaderboard -c playground-series-s6e8 -d` → zip with columns
`Rank, TeamId, TeamName, LastSubmissionDate, Score, SubmissionCount, TeamMemberUserNames`.

- **1,886 teams** (brief's ~1,326 stale; RESEARCH's 1,831 was 08-14).
- **Rank 19** — 18 strictly above, **3 tied at 0.97106**.
- Leader MILANFX **0.97124**, unchanged since 2026-08-10. Gap **180e-6**.
- **155 teams within 3e-4 of the top, 267 within 5e-4** — these reproduce w15a's plateau
  sizes exactly, measured independently on today's board. The plateau size is the free
  parameter that decides which arm of w15a's extreme-value model holds, and this confirms
  the *large*-plateau arm is the realistic one — the arm that requires a real skill spread
  (tau ≈ 92–104e-6) among the leaders rather than pure best-of-n selection noise. It is
  the strongest evidence against the "it's mostly slice draw" reading and should be quoted
  alongside it, not instead of it.

⚠ **`LastSubmissionDate` is NOT the date of the best-scoring submission.** It is the team's
most recent submission full stop. Do not read it as evidence about how Kaggle breaks a
best-score tie — it looks like exactly that and it is not. (Chased and falsified w15j.)

## Never-sent files, by CV, as of 39 submissions

After `blend160origm` went out this run, the strongest never-sent blends are `blend159`
(0.9700434, ens4), `blend160orig` (0.9700423, ens4), then the `rankraw` family from
0.9700343 down. Everything at CV > 0.970044 has now been scored.
