# ⏳ ONE MANUAL ACTION IS OUTSTANDING — final-submission selection

> ⛔ **2026-08-31 — TODAY IS THE LAST DAY, AND THE WINDOW SHUTS AT 23:59 UTC.** Thirty-three
> runs have now asked for this and nothing is selected. It is two ticks in a browser and it
> cannot be done from this machine: no Kaggle write API, no browser tool attached to the agent,
> no logged-in profile on the box. Every submission slot is spent (10/10 by 12:37Z) and the
> counter does not roll over before the close, so **this click is the only thing left that can
> change the private score.**

**Deadline 2026-08-31 23:59. Needs a browser. It cannot be done from the Kaggle API** (the
write path was probed and falsified 2026-08-13; only the *read* is exposed, which is what
`experiments/check_selection.py` uses).

Right now `check_selection.py` exits **1: nothing is selected**, so Kaggle will auto-pick the
two final entries by **public** score. That is selecting on the public slice — the exact
mistake that cost this account ~2.5 points of RMSE on `rogii-wellbore-geology-prediction`.

## What to click

<https://www.kaggle.com/competitions/playground-series-s6e8/submissions> → **Use for Final Score**
on exactly these two, and on nothing else:

| # | submission ref | file | CV | public |
|---|---|---|---|---|
| 1 | **55656399** | `w36_ad199stdcorr.csv` | 0.9701400060 | 0.97118 |
| 2 | **55588167** | `w23_ad187stdcorr.csv` | 0.9701150809 | 0.97116 |

Both are already submitted; this only flags them.

✅ **BOTH REFS RE-VERIFIED LIVE AGAINST THE KAGGLE API, 2026-08-31 ~15:05Z (w137)** — the first run
to check the ref → filename mapping rather than re-quoting this table. `55656399` resolves to
`w36_ad199stdcorr.csv` and `55588167` to `w23_ad187stdcorr.csv`, both `NAME_MATCH=YES` in the live
`SUBMISSION_GROUP_SUCCESSFUL` listing, and both CVs above match
`experiments/w48a_cv_recomputed.csv` to **all ten digits**. The table is current, not stale.

✅ **AND THE PICK IS THE CV ARGMAX OVER EVERY SUBMITTED FILE, NOT JUST THE REGISTERED ONES
(w138, 08-31 ~15:40Z, `experiments/w138a_coverage.py`).** `w48a_cv_recomputed.csv` is built from a
registry — four hand-maintained files list a stem, and only then is its OOF looked up — so it
covers **159 of the 201 submitted stems** and carries 10 that were never sent. Scoring every
submitted stem with an OOF on disk, registry or not, on the same `y` and the same metric:
**166 scorable, 0 beat pick 1.** The recompute reproduces the table on all 159 shared stems to
`max|diff| = 0.0000e-6`, so it is the same instrument and not a second opinion.

The 35 with no OOF are bounded rather than ignored: every one has public ≤ **0.97107**, and among
the 73 *tabled* stems in that same public band not one has `true_cv` above pick 1 — the best is
**84.31e-6 below** it, with `corr(public, true_cv) = 0.873` over 159 stems.

⚠ **NOTHING IN THAT AUDIT IS A NEW NUMBER.** The w69 family's CVs were already in
`w75a_erarefresh.csv` and `w79c_p9.json`; the auto-pair's were already in `w114a_misclick.json`.
What was missing was the **comparison**, in one place, on one instrument.

⚠ **Pick 2 sits at CV rank 30 of 169 in the registry table (rank 35 of the 166 scorable
submitted stems) on purpose.** It is the zero-parameter hedge against the
whole fitted-correction family failing (w16c). CV rank 2 (`w38_ad202stdcorr`) is nearly the same
file as pick 1 and buys almost no E[max]. **This is not an error in the table — do not "fix" it.**

## What happens if nobody clicks

Kaggle takes the top two by public score, which today is **determined** — exactly two files
sit at 0.97119 and everything else is at 0.97118 or below:

    w36_ad199stdcorr_ens4.csv   (ref 55714895)   CV 0.9701365875
    w38_ad202stdcorr_ens4.csv   (ref 55714897)   CV 0.9701349001

Neither is the CV-preferred pair. Priced on the live board:

| | cost of NOT clicking |
|---|---|
| tau = 0 | **+4.52e-6** |
| tau = 2.10e-6 (95% upper, refit 2026-08-24) | **+3.07e-6** |
| tau = 5.0e-6 (far above anything measured) | −0.50e-6 |

`experiments/w74a_clickprice.py`, two verbatim controls at 0.000e+00. Honest reading: **the
click is worth about 3–4.5e-6 of expected private AUC, and as an AUC number it is small.**

🔻 **THE UNIT w133 CALLED "THE UNIT THE COMPETITION PAYS" IS NOT PAID HERE. CORRECTED w134,
2026-08-31** (`experiments/w134a_awardunit.py`, 14 checks, FAILURES 0, all three readings live).
**This competition awards NO Kaggle medal.** The competition object reads `awards_points=False`
and `reward='Swag'`; the field is informative, not always-False, reading True on 7 of the 21
competitions in the same pull and False on every Getting Started entry; and the competition's
own **Prizes page** names *"Choice of Kaggle merchandise"* for **1st / 2nd / 3rd Place** and
never uses the word "medal". So there is no bronze band, no bronze cut and no medal here, and
the real award threshold is **top 3** — we are rank 309.

⚠ **THE NUMBERS BELOW SURVIVE; ONLY THEIR NAME CHANGES.** Read every "P(bronze)" as
**P(finishing in the top 10%)**, which is a true and checkable statement about *rank*. w133's
arithmetic was not wrong, its currency was. Carried through the public→private shake as a
paired delta — one noise draw, all arms read off it — the click moves **P(top 10%) by
+3.20 / +2.42 / +1.30 percentage points** at the S6E2 / S6E3 / S6E5 shift sds
(w133, `experiments/w133a_clickmedal.py`, 7 controls, FAILURES 0, live board 2026-08-31
13:14Z: rank **307 of 3,459**, top-10% line **345**, margin **+38**). Not four places, and not
a medal: **one to three points of finishing-position probability.**

⚠ **QUOTE THE δ THOSE THREE FIGURES USED (w135).** dP is **linear in δ**, and two different
numbers have shared the name "the click's price": **+4.5228e-6** is expected private AUC under
w74a's fitted GLS transfer, **+3.4185e-6** is the arithmetic CV difference of the two pair
maxima on the OOF arrays. The figures above used the first. At the second they read
**+2.42 / +1.83 / +0.98 pp**. Neither δ moves the direction of the click or the pick below.

🔴 **AND THE PRIVATE DRAW IS NOISIER THAN THE CLICK (w135, `w135a_clickpower.py`, 7 controls,
FAILURES 0).** Bootstrapped paired at private scale, `max(WANTED) − max(AUTO)` has mean
**+3.1372e-6** and **sd 3.7997e-6**, so **P(delta > 0) = 79.6%**: about **one private draw in
five** hands the better score to the pair Kaggle auto-picks. That is not a reason to skip the
click — the expectation is positive at every scale and a wrong tick is an order of magnitude
worse — it is a reason not to read tonight's single board as a verdict on the pick.

⛔ **DO NOT QUOTE "roughly four places" ANY MORE, IN EITHER DIRECTION.** It came from a
density fit, and +4.5228e-6 is *below the leaderboard's own printed resolution* — the Score
column is 5 d.p., so on the live board the delta crosses **zero** teams and the deterministic
answer is **0 places**, not 4. Both figures are artefacts of pricing a sub-resolution quantity
in a resolution-limited unit. The probability under a shake is the honest reading and the
places figure is not.

🎯 **The click is worth MOST when the board holds STILL.** dP ≈ φ(z)·δ/sd_shift, so it is
largest at the *smallest* shift sd (+3.20pp at S6E2's 43e-6) and smallest at the largest
(+1.30pp at S6E5's 124e-6) — the opposite of the intuition that a big shake makes small edges
matter more.

⚠ **ALL OF THAT PRICES ONLY THE *MISSING* CLICK.** Clicking the WRONG pair costs up to
**+81.92e-6, ~18x more** (see below) — closing THAT is what the one minute actually buys.

⚠ Do not "improve" on this list by picking the higher public scores. That is the failure.
Selection here is on CV, and the CV ordering is stable — `w36_ad199stdcorr` is argmax on
**both** defensible CV bases (w73), so this pick does not depend on an unsettled convention.

## 🔴 THIS HAS ALREADY HAPPENED TO THIS ACCOUNT, ONE COMPETITION AGO (w136, 2026-08-31)

Everything above prices the click against one counterfactual: *nobody clicks, and Kaggle
auto-selects the best two by **public** score.* Until today that sentence had never been read
off Kaggle. It has now been checked two ways (`experiments/w136a_autorule.py`, 23 checks,
FAILURES 0, all readings live).

⚠ **The rules do NOT state the criterion.** They confirm the limit — *"You may select up to two
(2) Final Submissions for judging"* — and they confirm automatic selection happens — *"the
Submission selected by the user, **or automatically selected by Kaggle in the event not selected
by the user**"* — and then they stop. The word *"public"* never appears in that clause, and
*"most recent"* appears nowhere in the rules at all.

✅ **So the rule was recovered from an outcome Kaggle actually produced**, on this account:
`rogii-wellbore-geology-prediction`, closed 2026-08-05, 8 scored submissions, nothing manually
selected, metric MSE so "best" points the opposite way from AUC.

| | public | private |
|---|---|---|
| best private available (submitted a day earlier) | 7.073 | **9.529** |
| what auto-selection took (the two best public) | 7.003 / 7.004 | **9.606** |
| **realised final placement, read off the private board** | — | **9.606, rank 2,781 of 6,125** |

Of the 28 possible pairs, **exactly one** reproduces 9.606, and it is the best-two-by-public
pair. "Most recent two" predicts 9.573 and is refuted; "best two by private" predicts 9.529 and
is refuted. Chance of matching by accident: **1 in 28**.

🎯 **The account held a strictly better file, by 0.077 MSE, and the default selection threw it
away.** That is the mechanism this click exists to defeat, and it is not hypothetical here.

⛔ **Do not over-read it either.** This does not prove clicking would have saved Rogii — that
file ranked 5th of 8 on public, and there are no Rogii CV artefacts to say whether any honest
criterion would have picked it. What it proves is that **the default is real, it is public-score
based, and on this account it has already chosen the worst available outcome once.**

## ⚠ AND DO NOT CLICK A FILE NAMED ANYWHERE ELSE (w114, 2026-08-29)

The two rows in the table above are the whole instruction. Clicking a *different* pair is a
far larger error than not clicking at all — priced on w74a's own estimator
(`experiments/w114a_misclick.py`, GATE R reproduces w74a's headline to 0.000e+00):

| what you click | cost vs the table above |
|---|---|
| nothing at all (Kaggle auto-selects on public) | **+4.52e-6** |
| `w21_ad187corr` + `w20_ad187_h3` | **+35.17e-6** — 7.8x worse |
| `w16i_schemeavg` + `blend159av_h3` | **+81.92e-6** — 18.1x worse |

🔴 **THE SAME ORDERING SURVIVES THE CHANGE OF UNIT, AND THE MIS-CLICK IS STILL THE HAZARD**
(w133, same file, control C7). Against the status quo of not clicking at all, in P(top 10%):

| what you click | dP(top 10%) at sd 43e-6 / 67e-6 / 124e-6 | vs the click |
|---|---|---|
| the two rows in the table above | **+3.20 / +2.42 / +1.30 pp** | — |
| `w21_ad187corr` + `w20_ad187_h3` | **−26.86 / −17.47 / −9.26 pp** | 8.4x / 7.2x / 7.1x |
| `w16i_schemeavg` + `blend159av_h3` | **−61.92 / −44.06 / −24.69 pp** | 19.4x / 18.2x / 19.0x |

Those ratios reproduce the AUC ratios (7.8x, 18.1x) without being told to. With no shake at
all the first wrong pair alone lands us at **rank 347 against a 345 top-10% line** — it drops
us out of the top decile outright, with no shake needed. **A wrong tick is far worse than no
tick**, and that is the one sentence in this file that never depended on the unit.

Those two wrong pairs are not hypothetical: until this run `check_selection.py` printed the
first inside a green-tick "✅ RESOLVED. WANTED HAS MOVED" box and annotated the second as
"current WANTED". Both were true in August's first week and have been false since 08-19/08-21.
That narration now lives behind `check_selection.py --history`, and standing check #48
(`w114b_selectguard`) fails if any printed line names a selectable file that is not WANTED.

## How to verify it took

    .venv/bin/python experiments/check_selection.py > /tmp/cs.txt 2>&1; echo $?   # 0 = selected

⛔ Read the status through a redirect or `$(...)`, never through a pipe — `$?` after a pipeline
is `tail`'s status and has already produced one fabricated finding in this workspace.

---

## Re-verified 2026-08-24 (w75) — the pick is unchanged and now stands on a wider check

- `w74b_clickstaleguard` **rc=0**: tier membership did not move on the 08-24 sends, so the
  **+4.52e-6** price above is live and exactly two files still sit at 0.97119.
- The slot-1 argmax was re-run over the **whole sent population** — all 122 of the 131 stems
  that have a stored OOF vector, with **CV recomputed from that vector**, not parsed from a
  submission description. `w36_ad199stdcorr` is the argmax. The 9 stems without an OOF all
  score ≤ 0.97107 public and cannot contend for a pick decided on CV.
- The CV→LB **era term** was refreshed from n=5 to n=27 (**−29.82 → −23.77e-6**). The pick does
  not move, and its binding margin is **era-INVARIANT**: the nearest era-deflated challenger,
  `w38_ad202stdcorr`, is in the same era, so the deflation cancels. Slot 1 would only lose the
  argmax at era = −40.35e-6, which is 7.4 se from the refreshed estimate.
- Slot 2 remains `w23_ad187stdcorr` on w64's settled reasoning: the whole slot-2 decision is
  worth **0.14e-6**, and if it is ever re-opened the candidate is `w38_ad202stdcorr`, not
  `w40_ad211stdcorr`.
- Across the 122 sent files, **Spearman(CV, public LB) = +0.820**. Selecting on CV here is
  supported by the data, not just by the Rogii precedent.
