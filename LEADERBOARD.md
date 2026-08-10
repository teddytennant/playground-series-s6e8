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

## 2026-08-10, end of slot 2

| | |
|---|---|
| teams | 1,356 (up from 1,331 this morning) |
| our best public | **0.97081** (`stack_pub74_logit`, `stack_pub88_mine_logit`) |
| today's slot-2 entry | 0.97080 (`stack_pub86_hybrid`) — no movement |
| our rank | ~158/1356, top 11.7% |
| bronze cutoff (top 10%) | **0.97084** at rank ~136 |
| gap to bronze | **+0.00004** |
| #1 (MILANFX) | 0.97120 |
| gap to #1 | +0.00040 |

Three submissions now sit at 0.97080/0.97081/0.97081 — a **public tie carrying no
information**. Do not let it drive the final pick; select on CV, where
`stack_pub86_hybrid` leads at 0.969678.

### CV → LB offset, three points

| entry | CV | LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |
| `stack_pub86_hybrid` | 0.969678 | 0.97080 | +0.001122 |

The offset is stable at **+0.00112 to +0.00117** and drifting *down* as CV goes up —
i.e. the last three CV gains (+1.9e-5, +1.8e-5) transferred to the LB at a rate of
**zero**. Anything smaller than ~0.0001 CV is not worth a slot.

Top of the board barely moves day to day (#1 unchanged since 2026-08-06), but ranks 2–10
churn constantly, which is what a crowd selecting on a public slice looks like. Rank 2 has
51 submissions, rank 3 has 58. Expect private-LB reshuffling in that band.

---

## 2026-08-10, after slot 3 — rank 158 → 18 of 1366

`stack_pub149_hybrid` scored **0.97099** (CV 0.970018), up from a three-way tie at
0.97080/0.97081/0.97081. First real LB movement in three days.

| band | score | note |
|---|---|---|
| #1 MILANFX | 0.97120 | unchanged since 2026-08-06, 11 submissions |
| #5 | 0.97110 | |
| #10 | 0.97106 | ~gold cutoff |
| #15 | 0.97101 | |
| **#18 us** | **0.97099** | 4 submissions |
| #25 | 0.97096 | |
| #40 | 0.97093 | |
| ~#68 | ~0.97093 | silver cutoff (top 5%) |
| #136 | 0.97084 | bronze cutoff (top 10%) |

We are **+0.00007 from rank 10**. The board is extremely dense: 0.97093 → 0.97106 spans
ranks 40 → 10, so a gain of one ten-thousandth is worth ~30 places up here.

### CV → LB offset, four points — it is NOT constant

| entry | CV | LB | offset |
|---|---|---|---|
| `stack_pub74_logit` | 0.969641 | 0.97081 | +0.001169 |
| `stack_pub88_mine_logit` | 0.969660 | 0.97081 | +0.001150 |
| `stack_pub86_hybrid` | 0.969678 | 0.97080 | +0.001122 |
| **`stack_pub149_hybrid`** | **0.970018** | **0.97099** | **+0.000972** |

The offset falls monotonically as CV rises. The +0.000340 CV gain arrived as +0.00019 LB
(**56% pass-through**); the three gains before it, all ≤2e-5, arrived as **zero**. Stop
treating "CV + 0.0012" as an LB estimate — but note the offset is also no longer usable
as a leak detector, which is what `RESEARCH.md` previously suggested it for.

### Note on the overfitting warning

Our four entries now rank identically on CV and on public LB, so for the first time there
is no selection conflict. `stack_pub149_hybrid` leads on both. Nothing in this run was
tuned against LB feedback: the member set was chosen by paired CV on frozen folds, and
the four excluded groups (`njm_*` dups, `njm_*_blend`, `sixmember_*`, exact duplicates)
were excluded on mechanism *before* any CV was consulted.

Ranks 2–10 still churn daily and rank 2/3 have 55/51 submissions, so expect private-LB
reshuffling in that band. Ours is a 4-submission position built on CV.

---

## 2026-08-10, slot 4 — two more entries, no LB movement expected

Board snapshot unchanged from the 17:49 pull: #1 MILANFX 0.97120, rank 10 = 0.97106,
rank 25 = 0.97096, rank 50 = 0.97092, rank 100 = 0.97086. We sit at 0.97099 (~#18).

Two entries went out at 151 members, differing only in the meta-feature transform:

| entry | members | transform | CV | LB |
|---|---|---|---|---|
| `stack_pub151_hybrid` | 151 | hybrid | 0.970024 | pending (sent by the owner session) |
| `stack_pub151_rankraw` | 151 | rankraw | 0.970023 | pending |

Both are +5e-6 CV over the shipped `stack_pub149_hybrid`, which is an order of magnitude
under the noise floor — **neither is expected to move the board**, and that is fine: on
Playground nothing evicts anything, so a slot spent on a genuinely different file is free.
Spearman of rankraw against the shipped 149 entry is 0.99916.

The reason for sending both: `hybrid` repairs only the members it judges broken, and slot
4 showed that judgement keys on an artefact of OOF-vs-test construction rather than on
member quality (see RESEARCH.md). `rankraw` treats all 151 uniformly. They are within 1e-6
on CV, so this is a mechanism preference, not a measured one.

### Watch at the deadline

Our entries are now 0.97080–0.97099 with CV 0.969641–0.970024, and CV and LB still rank
identically. If `rankraw` and `hybrid` come back split on the public slice, **that split
carries no information** — 1e-6 of CV cannot be resolved by a 296k-row slice either.
Select on CV, and if CV ties, prefer `rankraw` on mechanism.
