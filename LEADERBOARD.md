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

---

## 2026-08-11 (UTC) — rank 14 of 1385, and the transform split came back

**Standing: 0.97104 public, rank 14/1385, 12 submissions.** Rank 1 is 0.97124, so the
whole field from us to the top spans 0.00020. Cutoffs: rank 5 = 0.97110, rank 10 =
0.97108, rank 20 = 0.97099, rank 50 = 0.97092, rank 100 = 0.97086.

We are **0.00004 below the gold cutoff** — a difference smaller than every CV effect
measured in the last two days, i.e. not a difference we can steer by.

### The answer to the previous entry's open question

That entry asked what to conclude if `rankraw` and `hybrid` came back split on the public
slice, and answered in advance: *the split carries no information*. Both remaining
transforms were sent this run, so the full five-way mapping now exists — and the result is
sharper than "no information".

| entry | CV (cross-fitted) | public LB | offset |
|---|---|---|---|
| `blend150fx` (rank-avg of all 4) | **0.970032** | **0.97104** | +0.001008 |
| `stack_pub151_fixed_rankraw` | 0.970025 | 0.97103 | +0.001005 |
| `blend150fx_rankraw` | 0.970024 | 0.97102 | +0.000996 |
| `blend150fx_rescale` | 0.970013 | 0.97102 | +0.001007 |
| `blend150fx_hybrid` | 0.970014 | 0.97099 | +0.000976 |
| **`blend150fx_logit`** | **0.969950** ← worst | **0.97103** ← 2nd best | +0.001080 |

`blend150fx_logit` has the **lowest CV of the five by 8e-5** — a gap larger than the noise
floor and larger than any single improvement we shipped all week — and it scored **0.97103
public, one ten-thousandth off our best and above three stacks that beat it on CV.**

This is the Rogii failure mode in miniature, presented as a free gift. Selecting these
five on public LB would rank `logit` second; selecting on CV ranks it last. **We select on
CV.** The transform is the one thing here with a mechanism argument attached (`logit`'s
clip provably destroys the tails of ~29 saturating members), and it is the entry the
public slice likes.

### What this pins down about CV → LB

The old claim in `RESEARCH.md` — "the offset shrinks as CV rises" — was drawn from four
points spanning a large CV range. With twelve points it resolves into two regimes:

- **Across** the 0.9696 → 0.9700 step the offset genuinely fell, +0.00115 → +0.00100.
- **Within** the top cluster (CV 0.96995–0.97003) the offset scatters over
  +0.00098…+0.00108 with no trend. That scatter is ±5e-5 of LB — the same size as the CV
  differences being compared.

So the public slice cannot resolve CV differences below ~1e-4, which is every difference
we are still able to produce. Chase LB rank freely, since it costs nothing; read nothing
into it.

## 2026-08-11 03:40 UTC — rank ~13 of 1389 at 0.97106 (new best)

`blend156` returned **0.97106**, up from the 0.97104 that had held for two runs. First
public move in three runs, and the first time a new entry led on CV *and* on the LB.

Top of the board when checked at 02:25 UTC (before this submission):

| rank | score | subs | team |
|---|---|---|---|
| 1 | 0.97124 | 14 | MILANFX |
| 2 | 0.97115 | 55 | Don Mani |
| 3 | 0.97113 | 51 | Maher el Ouahabi |
| 4 | 0.97112 | 58 | Optimistix |
| 5 | 0.97110 | 16 | Mahog |
| … | | | |
| 13 | 0.97106 | 34 | magp |
| 15 | 0.97104 | 16 | **Teddy Tennant** |

Gap to #1 is **0.00019**. At the ~56% CV→LB pass-through measured on the only gain large
enough to trace, closing it needs roughly **+0.00034 CV** — which is 38× the +9e-6 this
run bought, and about the size of the entire slot-3 63-member import. Nothing incremental
gets there; it needs another whole independent pipeline group, and the public OOF pool is
exhausted.

Submission counts are worth noting against that. Ranks 2–4 have sent 51–58 entries to our
16, and rank 1 has sent 14. Volume is not what separates the top of this board.

## 2026-08-11 04:20 UTC — rank 13 of 1396, holding at 0.97106

No submission this run: all 10 daily slots were already spent by 04:04 UTC (the
`blend150sx`/`blend156` batch). Standing re-read from the downloaded LB snapshot rather
than moved.

| rank | score | team |
|---|---|---|
| 1 | 0.97124 | MILANFX |
| 2 | 0.97115 | Don Mani |
| 3 | 0.97113 | Maher el Ouahabi |
| 4 | 0.97112 | Optimistix |
| 5 | 0.97110 | Mahog |
| … | | |
| **13** | **0.97106** | **Teddy Tennant** |

The board is extremely compressed and it is worth writing down how compressed:

| rank | 1 | 5 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| score | 0.97124 | 0.97110 | 0.97108 | 0.97104 | 0.97099 | 0.97095 | 0.97092 |

**Ranks 10 to 50 span 1.6e-4 of AUC**, and ranks 5 to 15 span 6e-5 — which is *inside*
the ±5e-5 scatter the public slice already shows on files whose CV we know exactly. So
most of our visible rank is not a measurement of anything. Two entries of ours separated
by 1e-5 of CV have already landed 2e-5 apart on the LB in the wrong order.

Gap to #1 is 1.8e-4 of LB. At the ~56% CV→LB pass-through, closing it needs ≈ +3.2e-4 CV,
about the size of the entire slot-3 63-member import and ~35× this week's typical member
gain. That target has not moved and nothing incremental reaches it.

## Snapshot 2026-08-11 04:20 UTC — 1,396 teams, us rank 13

Refetched during slot 5 (research-only run, all 10 daily submissions already spent).

| rank | score | team |
|---|---|---|
| 1 | 0.97124 | MILANFX |
| 5 | 0.97110 | Mahog |
| 10 | 0.97108 | FunnyBishop |
| **13** | **0.97106** | **Teddy Tennant** (`blend156`) |
| 20 | 0.97099 | miki |
| 50 | 0.97092 | Charismatic Pizza Party |

Percentile cutoffs at 1,396 teams: top 5% = rank 69 (0.97092), top 10% = rank 139
(0.97084). Both are now comfortably behind us — we cleared them on slot 3 and the field
has not caught up.

**Fourteen teams sit at or above our 0.97106, and rank 10 is 0.97108.** The whole distance
from rank 13 to rank 10 is **2e-5 of public AUC**, which at the measured ~56% CV→LB
pass-through is about **+3.6e-5 of CV** — for once, a target of the same order as the
combiner work actually produces, rather than the 3.2e-4 needed to reach #1. It is still
inside the slice's own scatter, so treat it as a reason to keep shipping CV improvements,
never as a thing to tune towards.

## Snapshot 2026-08-11 06:00 UTC — 1,396 teams, us rank 13, unchanged

Refetched during slot 6 (research-only, the day's 10 submissions were spent by 04:04 UTC).
The board has barely moved in two hours: #1 MILANFX 0.97124 (unchanged since 21:01 UTC
yesterday), rank 10 = 0.97108, us 0.97106 with `blend156`.

Three teams did post overnight — `Orig_lab` 0.97109 (02:51 UTC), `AJboos` 0.97108
(02:20 UTC), `LeTuanM` 0.97107 (01:56 UTC) — all landing in the 0.97107–0.97109 band
immediately above us. That band is where the field piles up, and it is 1–3e-5 wide.

**What the slot-6 bootstrap adds to reading this board.** The paired row-bootstrap
(`experiments/auc_boot.py`) puts the *marginal* sd of a single AUC estimate at 1.67e-4 on
691k OOF rows. The public slice is a fraction of a 296k-row test set, so its own marginal
scatter is at least that large. **Ranks 1 through 50 span 3.2e-4 — under two marginal
standard deviations.** Essentially the entire visible leaderboard is inside the noise of
its own measurement, and the only reason our *paired* CV comparisons resolve at 2e-6 is
that they score every candidate on identical rows, which the leaderboard cannot do.

Concretely: do not read the 2e-5 gap to rank 10 as a target. It is not a distance, it is a
tie displayed as an ordering.

## 2026-08-11 07:00 UTC

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97124 |
| 2 | Don Mani | 0.97115 |
| 3 | Optimistix | 0.97113 |
| 4 | Maher el Ouahabi | 0.97113 |
| 5 | Mahog | 0.97110 |
| 6 | cstdy | 0.97110 |
| 7 | Orig_lab | 0.97109 |
| — | … | |
| **13** | **Teddy Tennant** | **0.97106** |

Held 13th overnight; the top of the board moved by 0.00000 since 2026-08-10 21:01 (MILANFX
has not been passed). Three teams entered the 0.97108–0.97115 band today, so the cluster
between us and the lead is thickening rather than the lead pulling away. The gap to first
is 0.00018 — roughly 4x the total CV spread of every candidate this workspace holds, which
is the honest reason to expect rank movement to come from someone else's slip, not ours.

## 2026-08-11 ~11:00 UTC — 1,410 teams, us rank 13

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97124 |
| 2 | Don Mani | 0.97115 |
| 3 | Optimistix | 0.97113 |
| 4 | Maher el Ouahabi | 0.97113 |
| 5 | Mahog | 0.97110 |
| 6 | cstdy | 0.97110 |
| 7 | Orig_lab | 0.97109 |
| 8 | midway2333 | 0.97109 |
| 9 | Utkarsh | 0.97108 |
| 10 | FunnyBishop | 0.97108 |
| 11 | AJboos | 0.97108 |
| 12 | LeTuanM | 0.97107 |
| **13** | **Teddy Tennant** | **0.97106** |
| 14 | magp | 0.97106 |

Team count 1,396 → 1,410. MILANFX still unpassed since 2026-08-10 21:01 UTC. Top-10 cutoff
is 0.97108, two ticks above us.

### ⚠ Correction: "the entire visible leaderboard is inside the noise" is wrong

The slot-6 note above argued that because the *marginal* sd of an AUC estimate is 1.67e-4
and ranks 1–50 span 3.2e-4, the whole board is a tie displayed as an ordering. **That is the
unmatched-null mistake, applied to the leaderboard.** Two teams are scored on the *same
fixed public rows*, so comparing them is a paired comparison and the marginal sd is not the
relevant scale — the shared slice noise cancels, exactly as it does for our own candidates.

Measured paired sds at a 20% slice (`experiments/cvlb2.py`), using pairs of our own files as
proxies for how correlated two strong submissions are:

| proxy pair | paired sd @20% |
|---|---|
| near-identical (corr ~0.999) | 0.000005–0.000008 |
| moderately different transforms | **0.000029** |

Taking the looser 2.9e-5 as a cross-team proxy:

- **The 0.00018 gap to MILANFX is ~6 paired sd. That is a real difference, not noise.**
  Someone is genuinely ahead of us and it is not a display artefact. The slot-6 reading let
  us off the hook for it.
- **The 2e-5 gap to the top-10 cutoff is ~0.7 paired sd — that one *is* a coin flip**, and
  the 0.97106–0.97110 pile-up (ranks 5–14, ten teams inside 4e-5) is genuinely unordered.

So the honest reading inverts the old one: the *local* ordering around us is noise, but the
*distance to the lead* is not. Closing 1.8e-4 needs something this workspace does not
currently have — the whole CV spread of all 41 candidates is 4.0e-4, and within the top
cluster it is 8.6e-5. Blend tweaks cannot get there; a genuinely better member can. The
seed-averaging result (+138e-6 solo for `xgb_latcat`, journal 2026-08-11 slot 8) is the
first thing in days pointed the right way.

## Snapshot 2026-08-11 08:08 UTC (1,415 teams) — and the first private-side estimate

| | score |
|---|---|
| #1 MILANFX | 0.97124 |
| top-10 cutoff | ~0.97108 |
| **us (thtennant), rank 13** | **0.97106** |
| bronze ≈ top 10% (rank ~141) | — |

MILANFX moved 0.97120 → 0.97124; our gap to the lead widened to 0.00018.

**`experiments/lbhist.py` prices the private draw for the first time**, using public *and*
private boards from the seven completed S6 episodes. Simulating a noisy private re-draw on
the live board at our score:

| assumed shift sd | median private rank | p90 | P(top 10) | P(top 10%) |
|---|---|---|---|---|
| 0.000043 (S6E2-like) | 12 | 22 | 34.8% | 100.0% |
| 0.000067 (S6E3-like) | 15 | 38 | 32.2% | 100.0% |
| 0.000124 (S6E5-like) | 26 | 90 | 24.2% | 98.8% |

**Bronze is no longer the target — it is nearly locked at rank 13 and top 10% ≈ rank 141.**
The live question is top 10, which is a 25–35% coin toss from here. Three of seven past
episodes destroyed their public top 30, but those boards carried 140–240 teams within ±5e-5
of rank 13 against S6E8's **14**, so the unavoidable component of our private risk is small.
That makes the avoidable component — selecting the deadline entries on CV — worth more, not
less. Full derivation and the failed leader-anchored density attempt are in the journal.

## Snapshot 2026-08-11 09:05 UTC (top of board) — three teams passed us today

| rank | team | score | submitted |
|---|---|---|---|
| 1 | MILANFX | 0.97124 | 08-10 21:01 |
| 2 | Don Mani | 0.97115 | **08-11 05:50** |
| 3 | Optimistix | 0.97114 | **08-11 08:07** |
| 4 | Maher el Ouahabi | 0.97113 | 08-10 09:34 |
| 5 | Mahog | 0.97110 | 08-09 14:23 |
| 6 | cstdy | 0.97110 | **08-11 07:48** |
| 7 | Orig_lab | 0.97109 | 08-11 02:51 |
| 8 | midway2333 | 0.97109 | 08-09 16:41 |
| 9 | Utkarsh | 0.97108 | 08-09 10:10 |
| 10 | FunnyBishop | 0.97108 | 08-09 11:05 |
| 11 | AJboos | 0.97108 | 08-11 06:36 |
| 12 | LeTuanM | 0.97107 | 08-11 01:56 |
| **13** | **Teddy Tennant** | **0.97106** | 08-11 04:04 |
| 14 | magp | 0.97106 | 08-11 02:33 |

MILANFX unchanged; the gap to the lead holds at 0.00018. **Rank 13 held but the ledge above
us is thickening** — Don Mani, Optimistix and cstdy all improved past us in the last nine
hours, and the top-10 cutoff has firmed at 0.97108, two ticks above us. The density figure
that drives the private projection is unchanged in shape: 14 teams within ±5e-5 of us.

⚠ A team is now called **`Orig_lab`** (0.97109). If the name means what it looks like, note
that this workspace measured both original-dataset routes on 2026-08-11 and found
concatenation monotonically harmful (−58e-6 at 1×) and the separate-estimator member worth
0 to −2e-6 in a 160-member stack. Their score is 3e-5 above ours and entirely consistent
with a good ordinary stack; nothing on the board suggests the original is paying anyone.

## 2026-08-13 (17:30 UTC)

**Rank 18 of ~1,400 at 0.97106** — down from rank 13 on 08-11 on an unchanged score. The
08-12 run was missed entirely, and the field closed the gap: six teams passed us in two days.

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97124 (unchanged since 08-10) |
| 2 | Don Mani | 0.97116 |
| 3 | Maher el Ouahabi | 0.97115 |
| 4 | Optimistix | 0.97114 |
| 5 | Romone Dunlop | 0.97112 |
| 6 | Orig_lab | 0.97111 |
| 7-9 | Mahog / Keanan / cstdy | 0.97110 |
| 10 | midway2333 | 0.97109 |
| 11-15 | Utkarsh, FunnyBishop, AJboos, LeTuanM, KeHao Liu | 0.97108 |
| 16-18 | MKhlystun, **Teddy Tennant**, magp | **0.97106** |

Ten submissions today moved the public score by **zero** — three of them tie 0.97106 and
none beat it. Combined with every internal instrument reading null, the honest read is that
the 159-member stack is saturated at ~0.97005 CV / 0.97106 LB and the remaining 1.8e-4 to
MILANFX is not reachable by anything in this workspace's current line.

### Same day, 18:06 UTC — two more passed us in 36 minutes

`magp` 0.97106 → **0.97109** (18:00) and `midway2333` → 0.97109, so we are now **rank 19**
(16 teams strictly above, three tied with us). The board's drift rate is the number worth
recording: **6 teams passed us in 36 minutes**, against ten of our own submissions that moved
the public score by zero. Extrapolated to the 18 days left, a static 0.97106 does not hold a
medal position.

That is a statement about the *public* slice only, and the private projection in
`lbhist.py` still says a ±5e-5 band this dense is mostly resorting noise. The conclusion is
unchanged and it is not "push harder on the public score": it is that the stack is saturated
and the remaining decision is **which** saturated file goes in as the final entry.

---

## 2026-08-16 03:5x UTC — w16a/w16b read, 1,943 teams

Full board pulled to `Teddy Tennant` row: **rank 41 of 1,943, score 0.97107**
(`w15f_antistudent_avg`, ref 55529992). 41 teams sit at or above us.

| rank | team | score | last sub |
|---|---|---|---|
| 1 | MILANFX | **0.97132** | 08-16 01:51 |
| 2 | Utkarsh | 0.97124 | 08-15 22:40 |
| 3 | Optimistix | 0.97123 | 08-16 01:08 |
| 4 | Maher el Ouahabi | 0.97122 | 08-16 00:26 |
| 5 | cstdy | 0.97121 | 08-15 23:30 |
| 6 | Laura Liepa | 0.97117 | 08-16 01:49 |
| 7-8 | Malhar Ujawane / Don Mani | 0.97116 | |
| 9-12 | choqui62, Felipe Tamaki, Miłosz, william950615 | 0.97115 | |
| **41** | **Teddy Tennant** | **0.97107** | 08-16 03:35 |

**The head of the board moved and it is not noise-shaped.** MILANFX was 0.97124 and static
since 08-10; it is now **0.97132**, +8e-5 in one step, and it moved on 08-16 01:51. Four
other teams also set new personal bests between 08-15 22:40 and 08-16 01:51. The gap from us
to first was 18e-5 on 08-15 and is now **25e-5**.

**Drift rate, the number that matters for the 15 days left.** Rank 41 today against rank 19
on 08-13 at a score 1e-5 higher. The field is passing a static file at roughly **7 teams/day**
and accelerating. Nothing this workspace has measured moves CV by more than ~1e-5, and the
CV→LB slope is +1.77, so the entire remaining internal toolkit is worth **<2e-5 LB** against
a 25e-5 gap. Do not plan a run that assumes stack refinement closes this.

### The top public notebook is now above us — and it is worthless

`najiama/ensemble-of-ensembles-lb-0-97111` (11 votes, run 08-16 03:28) claims **0.97111**,
which retires w15a's "no public notebook exceeds 0.97101". **It contains no model.** Read in
full (`notebooks/najiama_eoe_97111/`): it is a self-declared LB-probing demo built on
raykkretzschmar's public 0.97100 file, whose headline trick is `-df.lgbm_rank` inside
`np.lexsort` — sorting *against* his own LightGBM because the public slice rewarded it, which
the author labels "THE LB OVERFITTING HACK" and predicts will collapse on private. The only
live cell is a 0.1/0.9 average of Rayk's file with an undisclosed `Blend_submission.csv`.
There is no OOF anywhere in it by the author's own statement. **Do not pull, fork, or blend
it.** It also independently confirms the public slice is ~20% of the test set, which is the
`f = 0.20` this workspace has assumed since w14b.

---

## 2026-08-16 (w16c/w16f, slot 2) — board unchanged in the hour, our public best is now a 3-way tie

Board re-read at 04:04 UTC: **1,943 teams, leader MILANFX 0.97132, us rank 41 at 0.97107.**
Identical to slot 1's read 16 minutes earlier — the overnight burst has stopped for now.
Local density at our score: **2.5 teams per 1e-5** (10 teams within ±2e-5). That is the
conversion factor for every AUC number below: a 1e-5 gain is worth about 2.5 places here, so
the ~6e-6 the deadline pick moved this run is worth **~1.5 places**, and the 25e-5 gap to
first is worth ~62 places.

**Our best-public tie is now three files wide and all three are CV-endorsed.**

| public | files |
|---|---|
| **0.97107** | **3** — `w15f_antistudent_avg` (CV 0.9700527), `w16b_cellweight` (0.9700556), `w16f_armavg` (0.9700554) |
| 0.97106 | 7 — the ens4 shelf, including `blend158_logit` |
| 0.97105 | 13, the h3/w cluster |

All three members of the top tie are the same corrected object at different arm resolutions,
and all three are above every zero-parameter file on CV. Kaggle's default best-public auto-pick
therefore draws its **first** slot from a set in which every member is CV-endorsed, which is
the third consecutive run that has been true and is now robust rather than lucky. The exposure
remains the **second** slot, drawn from the 7-way tie at 0.97106 that contains `blend158_logit`.

**The corrected family occupies its own LB shelf.** Public score minus recomputed CV, by
transform family, over all 33 scored pack files (overall mean +0.001013, sd 2.58e-5):

| family | n | LB − CV | within-family sd |
|---|---|---|---|
| logit | 3 | +0.001088 | 9.6e-6 |
| rescale | 3 | +0.001018 | 9.5e-6 |
| ens4 | 9 | +0.001013 | 5.0e-6 |
| w | 3 | +0.001003 | 0.9e-6 |
| h3 | 8 | +0.001002 | 1.4e-6 |
| rankraw | 4 | +0.001000 | 4.0e-6 |
| hybrid | 3 | +0.000990 | 13.3e-6 |
| **corrected** | **3** | **+0.001015** | **1.5e-6** |

The corrected files sit ~13e-6 of gap above the h3 cluster they are built from, which is why
+6e-6 of CV bought a full 2e-5 of LB (0.97105 → 0.97107) rather than the +1.2e-5 the
within-family slope alone predicts. That is a **between-family** move and must not be read as
a slope; within the corrected family itself the three points span 2.9e-6 of CV and all print
0.97107, exactly as the ladder says they must.
