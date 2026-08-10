# Leaderboard notes — playground-series-s6e8

## Snapshot 2026-08-10 (1,331 teams)

| | score |
|---|---|
| #1 (MILANFX) | 0.97120 |
| #2 Maher el Ouahabi | 0.97113 |
| #3 Optimistix | 0.97112 |
| gold ≈ top 10 teams | 0.97106 |
| silver ≈ top 5% | 0.97092 |
| **bronze ≈ top 10%** | **0.97084** |
| **us (thtennant), rank ~155** | **0.97081** |

## The single most important structural fact

**The top of this leaderboard is compressed to almost nothing.** Rank 1 to rank 155 spans
**0.00039** total. Our 0.97081 sits **0.00003** below the bronze-medal cutoff.

Consequences for how to spend runs:

- A gain of **+0.0002** — small enough to be invisible in most competitions — would move
  us from rank ~155 to roughly the top 20. Rank leverage per unit of AUC is extreme here.
- Equally, this compression means public-LB *differences* between our own submissions are
  mostly noise on a 296k-row test set. Do not chase them. Select on CV.
- Medal boundaries are close enough that they will move as the field submits. Bronze is
  the realistic near-term target, silver is reachable, gold needs ~+0.00025 over current.

## Our CV → LB calibration (the number that matters)

| submission | CV (cross-fitted, frozen folds) | public LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | **0.97081** | **+0.00117** |

The public record predicted +0.0013; we measured **+0.00117** on our own pipeline. Use
+0.0012 as the working estimate for translating a CV number into an expected LB number,
and keep adding rows to this table — it is only trustworthy across repeated measurements.

Reminder of why the offset exists at all: every test prediction averages 5 fold models
while every OOF prediction comes from one. It is an offset, not a ranking change. **Trust
CV for ranking decisions, never as a leaderboard estimate.**

## Notable competitors / what the top is doing

- The leaders are **not** running better single models. Public notebooks make it clear the
  top entries are stacks over other competitors' published prediction files. An honest
  single pipeline trained on `train.csv` alone tops out around 0.9677 OOF.
- `najiama` (`s6e8-addiction-lb-0-97092`, 55 votes) publishes a blend submission but keeps
  the feature engineering and training private; the notebook is a prediction file plus a
  credits list. Their `naji0*` members are in the public OOF library and are the strongest
  individual members there (0.96881).
- `szymonkapiski` publishes the 74-model OOF library that our stack is built on — so a
  meaningful share of the field is stacking the same arrays we are. Beating them requires
  members they do not have, not a better combiner over the ones they do.
- `Don Mani` (0.97112, rank 4) and `tamerlanomralinov` (Lookup-Transformer) publish
  strong, distinctive notebooks worth re-reading when looking for a new channel.

## Where our headroom actually is

Since much of the field stacks the same public library, combiner improvements are close to
zero-sum. The leverage is in **adding a member nobody else has**:

- The decimal lattice (`frac_`, `d1_`) is absent from every `lat_*` model in the library.
- `lookup` takes the largest stacker coefficient (0.2415) purely because it is the least
  correlated member (max corr 0.9869 vs 0.987–0.999 for the rest). Decorrelation, not solo
  strength, is what buys blend weight.

## Submission log

| date | entry | CV (cross-fitted) | public LB | offset | rank |
|---|---|---|---|---|---|
| 2026-08-10 | `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 | ~155 |
| 2026-08-10 | `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 | ~155 |

**Working offset: +0.00115.** Two independent points now agree, and both sit a little
below the +0.0013 quoted in the public record.

**A +0.000019 CV improvement moved the public LB by 0.00000.** The public slice does not
resolve differences at that scale. This is the empirical justification for selecting on
CV: sub-0.0001 changes are simply not measurable on the LB, so any apparent LB movement at
that scale is noise being read as signal.
