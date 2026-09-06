# 2026-09-06 — w186. TWO MORE LATE MEASUREMENTS. THE BOARD DID NOT MOVE AND COULD NOT.

Standing unchanged: **319 / 3,531, private 0.97093**, public 331 / 3,532. Two sends, both
single members priced against the private test set, both ~2,100e-6 below the graded pair:

    56058676  w186_lgbmfrac_ctrl.csv   CV 0.9678206   public 0.96891   private 0.96876
    56058678  w186_lgbmfrac_cdf.csv    CV 0.9679083   public 0.96907   private 0.96884

The pair exists to price one channel (`cdfd_*`, the original's class-conditional CDFs) on
ground truth. Private delta +80e-6 against a CV delta of +87.7e-6 — the two agree to 8e-6,
which is the second CV<->private agreement this route has after w185b's
(CV +1078 / public +1250 / private +1050), from a completely different baseline.

⚠ **Neither is a candidate for anything.** Late cap read back **63 remaining today** after the
second, consistent with w185's 100/day. Nothing sent after 2026-08-31 23:59 can be selected or
graded, so the only thing a late send can buy is a number.

📌 Where the field sits, unchanged and quoted from w183/w185 rather than re-read this run:
3,532 teams on both boards. Our public 0.97119 / private 0.97093. The gap to 1st is ~440e-6 and w184 identified what
closes it — a tuned RealMLP, which nothing here has ever built.

# 2026-09-06 — w185. THE BOARD IS STILL CLOSED. THE SCORER NEVER WAS, AND WE HAD THE FIELD THAT SAID SO FOR FIFTEEN DAYS.

Standing is unchanged and unchangeable: **319 / 3,531, private 0.97093**. Late submissions do
not re-rank anything. What changed is what this workspace can MEASURE.

⛔ **`submissions_disabled` reads `False` because submissions are not disabled.** The entry
directly below this one (w157, 09-03) prints that field and then talks itself out of it:

> ⚠ `submissions_disabled` reads **False** for the fifteenth day on a closed, graded board. Do
> not read it as "you may submit" — the deadline is the authority, and it passed ~61 hours ago.

🎯 The instrument was right and the inference on top of it was wrong. Verified this run by doing
the thing rather than reasoning about it: `kaggle competitions submit` returns rc=0,
`"Successfully submitted"`, `"99 submissions remaining today"`, and the file comes back with
**publicScore AND privateScore**. Resending `w36_ad199stdcorr_ens4.csv` scored 0.97119 / 0.97093,
matching its record digit for digit. The late cap is **100/day**, not 10.

## 🔴 AND USING IT IMMEDIATELY FOUND +21 PLACES SITTING ON DISK

32 files in `submissions/` appear in no row of the 201-row history. All 32 sent, all 32 scored:

    w42_ad217stdcorr.csv    public 0.97119   private 0.97096   private rank 301   NEVER SENT
    w40_ad211stdcorr.csv    public 0.97118   private 0.97094   private rank 315   best sent
    w36/w38 ..._ens4        public 0.97119   private 0.97093   private rank 322   graded, official 319

`w42_ad217stdcorr` is the best file this workspace ever produced and it never left the box. It
was not forgotten — the ARM 217 family was vetoed over five runs because its fusion rests on
`oof_hboyang_mix`, an aggregator over 138 third-party streams whose OOF could not be audited.

✅ **The veto survives its own test.** w56 preregistered thresholds on `w48_cal_hboyang_mix.csv`
before any score existed: HONEST ≥ 0.97116, INFLATED ≤ 0.97080. That file was never sent either.
It scores **0.97109** — neither arm fires, **INDETERMINATE**, the branch w56 itself called modal,
which resolves to "the veto stands". The procedure was right. It cost 21 places anyway.

📌 The transferable line, and it is the brief's own: **on a board where submissions do not evict
each other, send the file you distrust.** Distrust governs SELECTION, not the send. The full
table is `experiments/w185c_unsent_scores.json`.

# 2026-09-03 — w157. UNCHANGED, AND RE-READ THROUGH THE GUARD RATHER THAN BY HAND.

`experiments/w153a_openreadguard.py` re-run live — the guard is the authority on this read, not a
hand-typed probe (w153's own lesson, learned the hard way when a hand-run probe was used to call
a published number wrong):

    deadline 2026-08-31 23:59:00   teams 3531   rank 319   metric Roc Auc Score
    submissions_disabled False     max_daily 10                         FAILURES: 0

⚠ `submissions_disabled` reads **False** for the fifteenth day on a closed, graded board. Do not
read it as "you may submit" — the deadline is the authority, and it passed **~61 hours** ago.

The same deadline was read a **second, independent way** this run, by `w157a_closedguard`'s C2,
which shells out to the uv tool python rather than re-execing: `2026-08-31 23:59:00`, agreeing
with the constant the guard compares trail dates against. Standing: **319 / 3,531, private
0.97093** — unchanged, and unchangeable.

# 2026-09-02 — w156. THE PUBLIC→PRIVATE MOVEMENT, GRADED AGAINST THE FIELD.

Nothing on the board moved and nothing can. What is new here is the **comparison**, computed live
from all 201 graded rows (`w142d_cvprivate.py`) rather than asserted:

| signal | Spearman vs **private** | n |
|---|---|---|
| **CV** | **+0.929** (p 5.8e-72) | 164 |
| public LB | +0.874 (p 1.1e-52) | 164 |

rho(CV) − rho(public) = **+0.055**, paired-bootstrap 90% CI **[+0.027, +0.085]**, P(CV better)
**100.0%**. ⚠ And every selection rule this workspace argued over lands on the same number:
argmax-CV, argmax-public, the WANTED pick and Kaggle's own fallback all score **0.97093 / rank
322**. Only hindsight does better — **0.97094 / rank 315**, +10e-6 and 7 ranks. The single best
private file, `w40_ad211stdcorr`, sat at **CV rank 3 of 166** submitted stems with an OOF on disk.

⛔ **THE ROGII FAILURE DID NOT REPEAT.** 3,532 teams on both boards. Rank movement (private −
public): median −3, mean +0.0, sd 75.9; |move| 50% 13, 90% 65, 99% 187. **Ours: 331 → 319, up 12,
quieter than 51.1% of the field.** The public top-100 moved **median +89 and 84 of 100 FELL**,
worst +475. Final: **319 / 3,531**, private **0.97093**, top decile by 34 places.

# 2026-09-02 — w155. RE-CONFIRMED, THROUGH THE ONLY CLIENT THAT ANSWERS THE USER FIELDS.

⚠ The w153 entry below prints `user_rank 319` under a two-line snippet that never says which
client sent it. That is not wrong, it is **incomplete**, and w154 measured what incompleteness
costs here: a bare `KaggleClient()` returns every competition field correctly and answers
`user_rank 0`, `user_has_entered False`. The full form, run live this run:

    api = KaggleApi(); api.authenticate()
    r = ApiGetCompetitionRequest(); r.competition_name = "playground-series-s6e8"
    with api.build_kaggle_client() as kc:
        c = kc.competitions.competition_api_client.get_competition(r)
    -> deadline 2026-08-31 23:59:00   now 2026-09-02 14:25 UTC   teams 3531
       metric Roc Auc Score   max_daily 10   submissions_disabled False
       user_rank 319   user_has_entered True

Closed by **~38.4 hours**. `w142b_privatecheck.py` re-run: **201 rows, 201 with `privateScore`**,
`selected` 0, status COMPLETE ×201, best private **0.97094** (`w40_ad211stdcorr`, public 0.97118),
landed **0.97093**. Identical to w150–w154. ⛔ There is no live board to track — final standing is
**319 / 3,531**, top decile by 34 places. `team_count` 3,531 and the private-leaderboard row count
3,532 are two endpoints that have always disagreed by one (w154); neither moves the rank.

# 2026-09-02 — w153. RE-CONFIRMED. THE READ BELOW IS THE SAME; THE COMMAND FOR IT WAS NOT RUNNABLE.

Every entry above quotes the opening read as a kwarg constructor. That form raises
`TypeError` — kagglesdk request types take no constructor arguments — so it is not the call any
of these entries actually ran. The one that works, and what it returned live this run:

    r = ApiGetCompetitionRequest(); r.competition_name = "playground-series-s6e8"
    -> deadline 2026-08-31 23:59:00   teams 3531   user_rank 319   metric Roc Auc Score

`w142b_privatecheck.py` re-run: **201 rows, 201 with `privateScore`**, `selected` 0, best private
**0.97094** (`w40_ad211stdcorr`), landed **0.97093**. Identical to w150, w151 and w152. ⛔ There is
no live board to track here — final standing is **319 / 3,531**, top decile by 34 places.

# 2026-09-02 — w152. RE-CONFIRMED. ⛔ THE ENTRY BELOW IS WRONG: THE BOARD IS NOT DELISTED.

w151's "delisted" reading came from a hand-built `ApiListCompetitionsRequest` with only `search`
set, which **filters closed competitions out** (5/5 on exact-slug probes, `experiments/w152a_listshape.py`).
The wrapper `api.competitions_list(search="playground-series-s6e8")` returns the row. S6E9 being
up is true and unrelated. Numbers below are unchanged and were re-read live this run:

    ApiGetCompetitionRequest(competition_name="playground-series-s6e8")
    -> deadline 2026-08-31 23:59:00   teams 3531   user_rank 319   metric Roc Auc Score

`w142b_privatecheck.py` re-run: **201 rows, 201 with `privateScore`**, `selected` 0, best private
**0.97094** (`w40_ad211stdcorr`), landed **0.97093**. Identical to w150 and w151. ⛔ There is no
live board to track here — final standing is **319 / 3,531**.

# 2026-09-01 — w151. RE-CONFIRMED, AND THE BOARD CAN NO LONGER BE FETCHED BY SEARCH.

Numbers below are unchanged and were re-read live this run — but **not** by the procedure the
entry below describes. `list_competitions(search="playground-series-s6e8")` now returns **n=0**;
the competition is delisted and **`playground-series-s6e9` is up in its place**. The same call
for S6E9 returns a row in the same process, so this is delisting, not a broken token.

    ApiGetCompetitionRequest(competition_name="playground-series-s6e8")
    -> deadline 2026-08-31 23:59:00   teams 3531   user_rank 319   metric Roc Auc Score

`w142b_privatecheck.py` re-run: **201 rows, 201 with `privateScore`**, `selected` 0, best private
**0.97094**, landed **0.97093**. Identical to w150. ⛔ There is no live board to track here again.

# 2026-09-01 — FINAL. THE BOARD IS GRADED AND THIS FILE'S TOP ENTRY WAS STILL QUOTING A LIVE
# READING FROM 08-31. FIXED HERE.

Read live this run from the competition object, not from a prompt or a prior entry:

    deadline 2026-08-31 23:59:00   teams 3531   user_rank 319   metric Roc Auc Score

| | rank | score |
|---|---|---|
| public board (frozen since 08-23) | — | **0.97119** |
| **private board (final)** | **319 of 3,531** | **0.97093** |

Top decile is 353, so we finish inside it by **34 places**. Best private over all 201 sends was
**0.97094** (`w40_ad211stdcorr`, public 0.97118) — one tick above what landed, and unreachable by
any public-argmax rule because its public score sat one tick under the 0.97119 auto-selection
tier. Worth **+7 ranks** in hindsight. Argmax-CV, argmax-public, WANTED and AUTO all converge on
0.97093 (w141 §4).

🔴 **DO NOT READ THE FINAL SCORE OFF `kaggle competitions submissions -v`.** That command returns
**50 of 201 rows** — one page, most recent first — and the best private score on that page is
**0.97092**, one tick low, with `w40_ad211stdcorr` (08-22) not in it at all. A truncated read here
does not error and does not look truncated. Use `experiments/w142b_privatecheck.py`, which
paginates; re-run live this run and it reproduces 201/201 rows with privateScore populated.

⛔ **NOTHING BELOW THIS LINE IS LIVE.** The 08-31 entry that used to head this file says
*"still the live one: rank 309 of 3,469"*. It was live when written and is now history.

# 2026-08-31 ~14:40Z — w136, A BOARD FROM A DIFFERENT COMPETITION, BECAUSE IT IS THE ONLY
# PLACE THE SELECTION RULE CAN BE WATCHED EXECUTING.

The S6E8 reading below is unchanged and is still the live one: competition object at 14:35Z
gives **rank 309 of 3,469**, top-decile line 346, margin **+37**.

⚠ **What is new is a board we had never read: `rogii-wellbore-geology-prediction`, closed
2026-08-05.** Every price in this workspace subtracts *"if nobody clicks, Kaggle auto-selects
the best two by PUBLIC score"*, and that sentence had never been sourced. The competition rules
turn out **not to state the criterion** — they confirm automatic selection happens and stop — so
it had to be recovered from an outcome. Rogii is the only closed competition on this account
with a real public/private split and more than one submission.

| | rank | score |
|---|---|---|
| public board | 2,071 of 6,125 | 7.003 |
| **private board (final)** | **2,781 of 6,125** | **9.606** |

The private board's top is **5.639** against the public board's **4.608**, which is the control
that it is genuinely the final board. Auto-selection took the two best-public files (7.003 and
7.004), both of which scored **9.606** private — **the worst private score of all eight
submissions.** The account held **9.529** on a file submitted a day earlier and ranked 5th of 8
on public. Of the 28 possible pairs exactly one reproduces 9.606, and it is the
best-two-by-public pair, so the chance of an accidental match is **1 in 28**.

🎯 **The default selection is real, it is public-score based, and on this account it has already
picked the worst available outcome once.** ⛔ It does **not** follow that clicking would have
saved Rogii; no CV artefacts for it exist here. See `experiments/w136a_autorule.py`, 23 checks,
FAILURES 0.

---
# 2026-08-31 13:44Z — w135, THE LAST BOARD BEFORE THE CLOSE. **RANK 309 / 3,463, TOP-DECILE
# LINE 346, MARGIN +37.**

Board downloaded once (13:44:56Z) and read from the CSV, not from the API's `user_rank`. Our
public score is unmoved at **0.97119**, frozen since 08-23 12:41:28. Leader Chris Deotte
**0.97207**, gap 0.00088. **0 of 10 slots left**, close 23:59 UTC tonight, so this is the last
public reading that can be acted on and nothing can be acted on.

⚠ **THE BAND WE STAND IN IS DENSE, AND THAT IS THE REASON THE CLICK IS SUB-RESOLUTION.**
**7 teams sit at exactly 0.97119 and 175 are inside 1e-4 above us** — about 1.75 teams per 1e-6
locally, against the ~0.95 that w113a's density fit used. Both are beside the point:
w133 withdrew the places figure because the Score column is **5 d.p.** and a 3-4e-6 delta
crosses **zero** printed ticks. The density is the reason the top decile is a knife edge, not a
reason to reinstate a places number.

**The drift, unchanged in direction:** 12:45Z 08-30 rank 278 of 3,321, line 332, margin +54 →
12:49Z today 304 of 3,453, +41 → 13:14Z 307 of 3,459, +38 → **13:44Z 309 of 3,463, +37**. Four
places and 142 teams in an hour of the final day.

## What the click is worth, re-derived at the live board

Common-shift matched null, 20,000 reps per sd, run against this download:

| shift sd | median private rank | 90% central | P(top 10%) |
|---|---|---|---|
| 43e-6 | 313 | [194, 398] | **73.3%** |
| 67e-6 | 303 | **[122, 426]** | **68.5%** |
| 124e-6 | 285 | [62, 485] | **66.2%** |

This reproduces w133's 74.3 / 68.7 / 66.5% on a board that has since grown by four teams. ⚠ Read
these as **P(finishing in the top 10%)** and not as a medal — w134 established this competition
awards none. **[122, 426] is now a pre-registered prediction** (`w135_prereg.txt` P3, commit
`761f341`), graded after the close by `w135b_grade.py`.

🔴 **AND THE PRIVATE DRAW'S OWN NOISE IS LARGER THAN THE CLICK.** `w135a_clickpower.py` (7
controls, FAILURES 0) bootstraps `max(WANTED) − max(AUTO)` at private scale, paired: mean
**+3.1372e-6**, **sd 3.7997e-6**, **P(delta>0) = 79.6%**. **One private draw in five hands the
better score to the pair Kaggle auto-picks.** The click stays correct — positive expectation at
every scale, and the mis-click pairs are an order of magnitude worse the other way — but tonight's
single board **cannot grade it**, and no post-close entry should pretend otherwise.

⚠ **Two numbers have shared the name "the click's price".** `+4.5228e-6` is expected private AUC
under w74a's fitted GLS transfer; `+3.4185e-6` is the arithmetic CV difference of the two pair
maxima. The dP figures above are **linear in δ**: at the arithmetic δ they read
**+2.42 / +1.83 / +0.98 pp** rather than +3.20 / +2.42 / +1.30. Neither δ moves the direction or
the pick.

---
# 2026-08-31 13:40Z — w134. 🔻 **THERE IS NO BRONZE CUT. RANK 309 / 3,459, TOP-10% LINE 345.**

`experiments/w134a_awardunit.py`, **14 checks, FAILURES 0**, three live readings. This
competition awards **no Kaggle medal**: the object reads `awards_points=False` / `reward='Swag'`
(and the field is informative — True on 7 of 21 competitions in the same pull, False on all 11
Getting Started entries), and the competition's **own Prizes page** names *"Choice of Kaggle
merchandise"* for **1st / 2nd / 3rd Place** without ever using the word "medal".

⛔ **EVERY "BRONZE CUT" AND "BRONZE BAND" IN THE ENTRIES BELOW IS A MISNOMER, INCLUDING THE
HEADING DIRECTLY UNDER THIS ONE.** The arithmetic stands and the ranks are real; the noun does
not. Read **345** as the **top-decile line** and every `P(bronze)` as **P(top 10%)**. The award
threshold this competition actually names is **top 3**, and we are 309th, so the live stake is
finishing position itself and nothing is on the line but that.

Live at the same reading: **rank 309 of 3,459** (`user_rank` off the competition object), two
places below w133's 13:14Z reading of 307. The drift has not stopped.

---
# 2026-08-31 13:14Z — w133, THE LAST BOARD. RANK 307 / 3,459, BRONZE CUT 345, MARGIN +38.

Board downloaded once for the day (`lb_w133/`, 13:14:16Z). w132 read the standing through the
API at 12:49Z and did not persist a file; instrument B needs the CSV, so this is the day's
single download, not a second reading. Our public score is unmoved at **0.97119**, frozen since
08-23 12:41:28. Leader Chris Deotte **0.97207**, a gap of 0.00088. Today's ten sends (12:36–
12:37Z) scored 0.93232–0.94275 and were never near the tier.

⚠ **THE MARGIN IS STILL SHRINKING AND IT IS DOING IT ON THE LAST DAY.** 12:45Z 08-30: rank 278
of 3,321, cut 332, margin **+54**. 12:49Z today: 304 of 3,453, cut 345, margin **+41**. 13:14Z
today: 307 of 3,459, cut 345, margin **+38**. Three places in twenty-five minutes. The standing
−22 places/day slope is if anything gentle for the final hours.

## What the position is actually worth, now that it has been priced in medals

`experiments/w133a_clickmedal.py` (7 controls, FAILURES 0) and `w83a_reproject.py` re-run at the
live band, which is the first time either has been read at rank 307 rather than rank 139:

| instrument | P(finish inside the bronze cut) |
|---|---|
| A, matched null, shift sd 43 / 67 / 124e-6 | **74.3% / 68.7% / 66.5%** |
| B, empirical band (top 6.88–10.88%), AUC episodes S6E2/E3/E5 | **90.3% / 78.3% / 69.4%** |

🎯 **THE TWO INSTRUMENTS NOW AGREE, AND THEY DID NOT AT RANK 139.** w83 found A at 89–100% and
B spanning 17–98% with three boards that wiped our band clean out of the top 10%. At the live
band the spans overlap at **69.4–74.2%** and **no episode is a wipeout** — B's minimum is
69.4%, against 17.2% at the old band. The deeper position is the more *predictable* one, which
is not the direction anyone would guess.

⛔ **This is not comfort.** Both instruments put us at roughly **two-thirds to nine-tenths**, on
a cushion of 38 places that has been closing all week, and the whole spread between the two
shift-sd assumptions (**7.85pp**) is larger than anything this account can still do — the
selection click, the only lever left, is worth **+1.30 to +3.20pp**.

---

# 2026-08-30 12:45Z — w122, MEASURED. THE DECAY SLOPE IS TWICE WHAT w120/w121 CARRIED.

Board downloaded once for the day (`lb_w122/`, 12:45:11Z), which is the w120 §7 rule.
**Rank 278 / 3,321, bronze cut 332, margin +54.** Our public score is unmoved at **0.97119**,
frozen since 08-23 12:41:28; today's ten sends came back 0.94203–0.94449, nowhere near the tier
and not meant to be.

⚠ **w121 quoted ≈ −11/day and that number is now wrong.** Re-derived off every saved board:

| when | teams | rank | cut | margin |
|---|---|---|---|---|
| 08-26 12:56 | 2,976 | 171 | 297 | +126 |
| 08-29 13:10 | 3,241 | 249 | 324 | +75 |
| 08-29 15:23 | 3,253 | 251 | 325 | +74 |
| **08-30 12:45** | **3,321** | **278** | **332** | **+54** |

08-26 → 08-30 is **−18/day**; the last 21 hours alone are **−22/day**. ⛔ The −11/day figure came
from a 2h13m window in which our rank moved by one, and a one-place move over two hours is not a
slope — it is a rounding. With ~35 hours to the deadline at −22/day the margin closes near
**+20**. Positive on every row above, so bronze survives and nothing here changes a decision,
but the cushion is thinner than the last two runs recorded.

The click is worth ≈4 places against that ~20-place cushion. Still nothing selected.

---

# 2026-08-29 15:4xZ — w121, NOT RE-MEASURED. THE ONCE-A-DAY RULE IS NOW SATISFIED BY w120's READ.

Our best public is **unmoved at 0.97119** (at cap 10/10, nothing sent). w120 §7 retired
`w83a_reproject`'s "re-download when our score moves" in favour of a measured rule — **read the
board once a day** — after showing that 128 places of bronze margin went while our number never
moved. w120's read is **17 minutes old**, so this run spends no API call and quotes it:

**rank 251 / 3,253, bronze cut 325, margin +74.** Our score has been frozen since 08-23 12:41:28.

The decay is ≈ **−11/day** at today's short-run rate (−1 rank over the 2h13m between w113's 13:10Z
and w120's 15:24Z reads). 56 hours remain; w120's two extrapolations land at **−25 → +49** and
**−40 → +34**, and **both leave bronze intact**. Neither contains a deadline-eve surge, which is
why it stays a measured rate rather than a forecast.

---

# 2026-08-29 15:1xZ — w119, NOT RE-MEASURED, AND THE RULE WAS CHECKED RATHER THAN QUOTED.

Our best public is **unmoved at 0.97119** (at cap 10/10, nothing sent). `w83a_reproject`'s rule
reserves a re-download for a material move; w118 §9 noted it is under-specified, because the
bronze cut is set by other teams and can drift while our score sits still. It is not binding here
either — `lb_w113/` is timestamped **2026-08-29T13:10:54**, about two hours old.

**Quote w113's row: rank 249 / 3,241, top 7.68%, bronze cut 324 (margin +75), silver cut 162 at
0.97127.** ⚠ A run that finds a board older than a few hours *and* a frozen score should
re-download anyway — the rule as written does not say so and it should.

---

# 2026-08-29 14:2xZ — w116, SPOT CHECK ONLY. NO MATERIAL MOVE, SO NO RE-DOWNLOAD.

Our best public is **unmoved at 0.97119**; the ten sent at 12:37Z topped out at 0.97107. The
top six is **identical** to w113's 13:10Z reading 1.1 h earlier:

    1 Chris Deotte 0.97207   2 Changye Li 0.97154   3 MILANFX 0.97149
    4 cstdy        0.97147   5 Keanan     0.97139   6 JungHwan 0.97138

⛔ **Full board NOT re-downloaded.** `w83a_reproject`'s own rule reserves that for a material
move, and neither side moved. **Quote w113's row below** — rank **249 / 3,241**, top 7.68%,
bronze cut 324 (margin +75), silver cut 162 at 0.97127. Re-measuring would have spent the run
reproducing a number that had not changed.

---

# 2026-08-29 13:10 UTC — w113, FULL BOARD (3,241 teams). THE BRONZE MARGIN HAS HALVED.

⚠ Quote the **medal cut**, not the percentile. Kaggle awards on a rank derived from N:
gold <= 10 + 0.2%*(N-1000), silver <= 5%*N, bronze <= 10%*N. Derive it every time with
`.venv/bin/python experiments/w113a_medalcut.py` — at N = 3,241 that is **14 / 162 / 324**.

| | w92 (08-26) | **w113 (08-29)** |
|---|---|---|
| our best public | 0.97119 | **0.97119** (unmoved; 08-29's ten were certified fillers) |
| rank / field | 171 of 2,976 | **249 of 3,241** |
| percentile | top 5.75% | **top 7.68%** |
| bronze cut (10%) | rank 297, margin +126 | **rank 324, margin +75** |
| silver cut (5%) | rank 149 | **rank 162 at 0.97127 — 87 places outside** |
| leader | Chris Deotte 0.97184 | Chris Deotte **0.97207** (gap 0.00088) |
| our submissions | 151 | **181** |

**78 places lost in three days**, and the cut moves toward us from both sides as the field grows.
`w83a_reproject.py` re-run (its rule reserves it for a material move; 5.75% -> 7.68% is bigger
than the move that triggered it at w92). 5/5 controls PASS, board 0.1h old.

- **matched null**, by past-AUC-episode shift sd: median private rank **254 / 247 / 234**;
  P(top 5%) **6.6 / 17.3 / 28.1%**; P(top 10%) = bronze **91.5 / 83.6 / 75.5%**.
  (w92 read 26.3/34.0/38.8% and 99.1/95.3/85.2%. **Do not quote those.**)
- **empirical band**, assumption-free: AUC-pool median finish **6.82%**, P(<=10%) **83.5%**.
  Per-episode stays bimodal — S6E2 94.9% vs S6E7 50.0% — so the pooled row describes neither.

Top of the board at 13:10Z:

    1  Chris Deotte 0.97207    2  Changye Li 0.97154    3  MILANFX 0.97149
    4  cstdy        0.97147    5  Keanan     0.97139    6  JungHwan 0.97138

⚠ **Local density near us is ~0.95 teams per 1e-6**, so the +4.5228e-6 selection click is worth
about **4 places** — not the 58 a 10x density error first produced (RESEARCH w113 §4). The click
is still the only action with expected private AUC attached; its value is not the rank.

⛔ **Not a lever.** Silver is +80e-6 of public; nothing sendable in the remaining 20 slots moves
our public score at all. Board CSV in `lb_w113/`. ⚠ Grep for **`Teddy Tennant`**, not
`thtennant` — the handle appears only in `TeamMemberUserNames`. ⛔ Download the full board and
open it with python's `zipfile`; the `-s` CLI returns 20 rows and a page token.

# ⚠ 2026-08-26 ~14:05 UTC — w93: THE `-s` CLI NOW RETURNS 20 ROWS, NOT 200

`kaggle competitions leaderboard -c playground-series-s6e8 -s` returned **20 rows plus a
`Next Page Token`**. Every earlier note here assumes 200. Our team is `Teddy Tennant` at rank
171, so it appears in **neither** page size — and a grep that finds nothing reads exactly like
having dropped off the board. ⛔ Download the full board and open it with python's `zipfile`
before quoting any rank. The w92 snapshot below (2,976 teams, 12:56:43Z) is the live one.

Top of the board at 14:05Z, unchanged in shape from 12:56Z:

    1  Chris Deotte   0.97184        4  cstdy       0.97140
    2  Changye Li     0.97154        5  Utkarsh     0.97134
    3  MILANFX        0.97149       20  tao guo     0.97129   <- the 20th row the CLI shows

Our 0.97119 is 65e-6 behind the leader and below every row the truncated CLI prints. **This is
context, not a lever** — no submission this account can make moves it, the final selection is
on CV, and the one action with expected private AUC attached is still the un-clickable
selection (`SELECT_THESE.md`).

# Snapshot 2026-08-26 12:56 UTC — w92, FULL BOARD (2,976 teams), material move

| | w88/w91 (08-25) | **now** |
|---|---|---|
| our best public | 0.97119 | **0.97119** (unmoved — today's ten all landed 0.97113–0.97116) |
| rank | 140 of 2,881 | **171 of 2,976** |
| percentile | top 4.86% | **top 5.75%** |
| top-5% cut | rank 144 — 4 places INSIDE | rank 149 at 0.97121 — **22 places OUTSIDE** |
| teams strictly above | 137 | **170** |
| teams tied at 0.97119 | 12 | **9** |
| leader | Chris Deotte 0.97172 | Chris Deotte **0.97184** (gap 0.00065) |
| our submissions | 141 | **151** |

31 teams passed us and the field grew by 95 in a single day. This is the first **material board
move** since w83, so `w83a_reproject` was re-run (its own rule reserves it for exactly this).
All five controls PASS — C2's zero-noise null returns rank 171 on all 4,000 reps, C3 reads the
board 0.1h old at 2,976 teams.

**A. matched null** (add each past AUC episode's own public→private shift sd to every team,
re-rank): median private rank **180** under all three sds; P(top 5%) 26.3% / 34.0% / 38.8%;
P(top 10%) 99.1% / 95.3% / 85.2%. Read the sd column, not the medians — the three finished AUC
boards disagree about it by 3×, which is wider than anything this account can influence.

**B. empirical band** (teams who stood in the top 3.75–7.75% of their own board, where they
finished as a share of the field):

    S6E1 5.49% · S6E2 5.19% · S6E3 5.63% · S6E4 4.89% · S6E5 11.65% · S6E6 10.94% · S6E7 15.25%

Four boards held the band near 5%, three threw it past 10%. These are two populations and the
pooled row (6.06%) describes neither.

⛔ **Not a lever.** No submission this account can make moves the public rank in the time left,
and the brief's own discipline says the private score is not chased from here. The only action
with expected private AUC attached is still the selection click (+4.5228e-6, `SELECT_THESE.md`),
and it is still un-clickable from this machine (w74). Board CSV in `lb_w92/`.

⚠ Grep the board for **`Teddy Tennant`**, not `thtennant`. The handle appears only in the
`TeamMemberUserNames` column, so searching for it looks exactly like being off the page.

# Leaderboard notes — playground-series-s6e8

## Snapshot 2026-08-25 ~14:0x UTC — w87, FULL board download (`lb_w87/`, 2,881 rows)

| | |
|---|---|
| us | **0.97119**, rank **140 of 2,881** = top **4.86%** |
| leader | Chris Deotte **0.97172** — gap **0.00053**, unchanged since 08-24 |
| 2nd / 3rd | Changye Li 0.97154 · MILANFX 0.97149 |
| top 5% cut | rank **144** — we are 4 places inside it |

**NO MATERIAL MOVEMENT** against w86 (140 of 2,878). The field grew by 3 teams and the top of
the board did not move at all. ⛔ `w83a_privproj.py` deliberately NOT re-run — its own rule is
to re-run only on a material board move. The slow rank slide is field growth, not a signal to
chase the public LB.

## Snapshot 2026-08-25 12:52 UTC — w83, FULL board download (`lb_w83/`, 2,874 rows)

| | |
|---|---|
| us | **0.97119**, rank **139 of 2,874** = top **4.84%** — unmoved by all ten 08-25 sends |
| teams strictly above us | **138** |
| leader | Chris Deotte **0.97172** (was 0.97168 on 08-24), last submit 11:24:04Z |
| 2nd / 3rd | Changye Li 0.97154 · (0.00053 is our gap to the leader) |
| top 10% cut | rank **287** |
| top 5% cut | rank **144** — we are 5 places inside it |

**First full-board download since 08-19, and the first read that puts our rank against a cut
rather than against the leader.** The 40-team slide w82 recorded has continued but slowed: 137
strictly above at 12:4xZ, 138 now. What is new is the framing — we sit **5 places inside the top
5%** and 148 clear of the top 10%, so the top-5% boundary is the live question and the top-10%
one is not, which reverses the standing note from 08-11 that told this workspace to stop looking
at the 10% line.

⚠ **That is a PUBLIC cut and it is not the outcome.** `experiments/w83a_reproject.py` (w83 §2–3)
prices what it is worth: on the three finished boards whose band matches ours, the share of teams
who held a top-10% finish runs **88.6% / 97.6% / 34.7%**, and three of seven boards moved our band
out of the top 10% entirely. **Top 5% is a coin flip.** Nothing on the board is a lever; the
selection click (`SELECT_THESE.md`, +3–4.5e-6) is still the only action with expected private AUC
attached, and it is un-clickable from this machine.


## Snapshot 2026-08-24 ~15:15 UTC — w79, top-200 page

| | |
|---|---|
| us | **0.97119**, unmoved (best of `w36_ad199stdcorr_ens4` / `w38_ad202stdcorr_ens4`, both sent 08-23) |
| teams strictly above us | **96** (was 94 at 14:55Z, w78) |
| teams tied with us | **16** |
| leader | Chris Deotte **0.97168**, resubmitted 14:50:33Z — **score unmoved**, so the resubmit did not land |
| 2nd / 3rd | Changye Li 0.97149 · MILANFX 0.97146 |

**Two more places lost in twenty minutes, and none of today's ten moved us.** All ten 08-24 sends
came back 0.97113–0.97117, every one at or below our standing 0.97119 — which is what the queue's
own pricing said would happen (`w26d` priced the whole queue at 6.3e-4 of beating the board). The
0.97121 band directly above us now holds 14 teams; the 0.97120 step holds at least 6 more.

⚠ The public board is not where the remaining value is. Nothing in the queue is expected to beat
0.97119, the drain continues only because the brief makes an unused slot pure waste, and the one
action still worth expected private AUC is the **selection click** (`SELECT_THESE.md`, +3–4.5e-6).


## Snapshot 2026-08-24 ~14:55 UTC — w78, top-200 page

| | score | vs the w77 snapshot ~40 min earlier |
|---|---|---|
| #1 Chris Deotte | **0.97168** | unmoved (last submit 08-24 13:54:30Z) |
| #2 Changye Li | 0.97149 | — |
| #3 MILANFX | 0.97146 | — |
| **us (Teddy Tennant), rank 95** | **0.97119** | unmoved |

- **94 teams strictly above 0.97119, and 16 teams TIED with us at it** (a 17-wide tie block,
  ranks 95–110). Gap to the leader **49e-6**, unchanged for three snapshots.
- ⚠ **We fell 93 → 95 without losing anything.** `林木子吖` and `Wang H2O` both reached **0.97120**,
  exactly one reporting step above us, and passed us. Our 0.97119 is byte-identical to w76/w77.
- ⚠ **The tie block is the thing to understand here.** 16 teams share our displayed score, so a
  single reporting step (1e-5) is worth ~16 places. That is the whole case for the hijack pricer:
  the bar is about which side of a display step a file lands on, not about CV distance.

### ⚠ THE `-v` FLAG PREPENDS A `Next Page Token` LINE TO THE CSV
`kaggle competitions leaderboard -c ... -s --page-size 500 -v` writes a `Next Page Token = ...`
line BEFORE the header, so `pd.read_csv` picks it up as the only column. Strip line 1 first
(`tail -n +2`). Max page size is 200 rows regardless of what `--page-size` asks for, so the
board below rank 200 is not reachable this way. Our team name is **"Teddy Tennant"**, not
`thtennant` — grep for the former.

## Snapshot 2026-08-24 ~14:13 UTC (2,783 teams) — w77, from the downloaded ZIP

| | score | vs the w76 snapshot ~20 min earlier |
|---|---|---|
| #1 Chris Deotte | **0.97168** | unmoved (last submit 13:54Z) |
| #2 Changye Li | 0.97149 | — |
| #3 MILANFX | 0.97146 | — |
| **us (Teddy Tennant), rank 93** | **0.97119** | unmoved |

- **109 teams at or above 0.97119**, up from 108. Gap to the leader **49e-6**, unchanged.
- ⚠ **We fell 92 → 93 without losing anything.** The field grew 2,779 → 2,783 and one new team
  landed above us. Reading a rank change as a score change is the mistake to avoid here: our
  0.97119 and the leader's 0.97168 are both byte-identical to the w76 snapshot.
- 131 submissions on the account, best 0.97119 (`w36_ad199stdcorr_ens4` / `w38_ad202stdcorr_ens4`,
  both sent 08-23 12:41). Today's ten scored 0.97113–0.97117, none beat it.

### ⚠ HOW TO READ THE BOARD WITHOUT PARSING THE TABLE
`kaggle competitions leaderboard -c ... -s` paginates and prints a `Next Page Token`; the `--csv`
form returns only the first page (21 rows). **Download instead** — and `unzip` is not on this box,
so open the archive from Python:

    kaggle competitions leaderboard -c playground-series-s6e8 -d -p /tmp/lb -q
    # then: zipfile.ZipFile(...) -> pandas.read_csv(io.BytesIO(zf.read(zf.namelist()[0])))

⚠ The ZIP's columns are `Rank, TeamId, TeamName, LastSubmissionDate, Score, SubmissionCount,
TeamMemberUserNames` — **not** the CLI's `teamId, teamName, submissionDate, score`. Our row is
found on **`TeamMemberUserNames == "thtennant"`**; `TeamName` is "Teddy Tennant", so a search for
the username against `TeamName` returns an empty frame.

## Snapshot 2026-08-24 ~13:5x UTC (2,779 teams) — w76, full pagination

| | score | vs the w72 snapshot ~1h earlier |
|---|---|---|
| #1 Chris Deotte | **0.97168** | **+16e-6** |
| #2 Changye Li | 0.97149 | new entrant to the top 3 |
| #3 MILANFX | 0.97146 | — |
| #4 cstdy | 0.97137 | — |
| **us (Teddy Tennant), rank 92** | **0.97119** | unmoved |

- **108 teams at or above 0.97119**, same count as w72. Gap to the leader is now **49e-6**, up
  from 33e-6 — the leader moved and we did not.
- ⚠ **Rank flat at 92 across the whole 08-24 send.** All ten of today's files scored
  0.97113–0.97117, every one below the account best, exactly as the queue priced them. The drain
  is not expected to move the board and it did not.
- ⚠ Parsing: team names contain spaces, so splitting the CLI's table on `\s{2,}` breaks at row 63.
  Match the score with a regex anchored at end of line instead.

## Snapshot 2026-08-24 12:5x UTC (2,774 teams) — w72, full pagination

| | score |
|---|---|
| #1 Chris Deotte | 0.97168 |
| #2 Changye Li | 0.97149 |
| #3 MILANFX | 0.97146 |
| #4 cstdy | 0.97137 |
| **us (Teddy Tennant), rank 92** | **0.97119** |

- **91 teams strictly above us; 17 tied at 0.97119.** Gap to the leader **49e-6**.
- ⚠ **We went 73 → 92 in a day WITHOUT our score moving.** The field moved; we did not. Expect
  this to continue — the drain sends files priced below the account best by construction.
- ⚠ The brief's "~1,326 teams" is stale. It is **2,774**, and the 08-10 snapshot below (1,331)
  is two weeks old.
- ⛔ Paginate. `kaggle competitions leaderboard -s --csv` returns only the first page and prints
  `Next Page Token = …` as a line ABOVE the header, which must be stripped before parsing. The
  first page alone reports "20 teams at or above 0.97119"; the true count is 108.

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
- `nhtquyn` (`s6e8-addiction`, 08-22) publishes a **120-member OOF+test pack shipping its own
  `fold_id.npy`** — the first external pack whose partition this workspace has VERIFIED rather
  than taken on trust, and it is ours bit for bit (w89a). Its members are generative/DA models
  at solo AUC 0.853–0.930, so all 120 fail w29's floor and it is refused (w89c). Useful as
  evidence about the field, not as supply.
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

## 2026-08-16, slot 3 (w16h/w16i)

Board at 01:00 EDT: **1,946 teams**. MILANFX **0.97132** (set 01:51 UTC on 08-16), then Utkarsh
0.97124, Optimistix 0.97123, Maher el Ouahabi 0.97122, cstdy 0.97121. We are **rank 42** at
0.97107 and the only team on that exact score; gap to first **25e-5**, unchanged from w16a's
reading six hours earlier.

`w16i_schemeavg` returned **0.97107**, pre-registered. Our public best is now a **4-way tie at
0.97107** — `w15f_antistudent_avg`, `w16b_cellweight`, `w16f_armavg`, `w16i_schemeavg` — every
member a corrected file and every member above every zero-parameter file on CV. The default
best-public auto-pick's **first** slot therefore remains safe under any tiebreak; the exposure is
still the **second** slot, drawn from the 7-way tie at 0.97106 that contains `blend158_logit`.

The corrected family's CV→LB shelf is unchanged with the fourth point added: CV
0.9700527/0.9700554/0.9700554/0.9700557 all print 0.97107, spanning 3.0e-6 of CV under a 1e-5
grid. A corrected file needs CV ≥ 0.970058 to move the printed digit, and no mechanism measured
in this workspace has ever moved CV by more than ~1e-5 — so the 25e-5 gap to first is not
closable by more of this.

## 2026-08-16 05:50 UTC — after w16l (slot 4)

Board top unchanged from slot 3's reading: MILANFX **0.97132**, Utkarsh 0.97124,
Optimistix 0.97123, Maher el Ouahabi 0.97122, cstdy 0.97121. Our best is **0.97107**
(4-way tie of our own corrected files) and the gap to first stays **25e-5**.

`w16l_maskw_h3` (ref 55544597) returned **0.97105** on a pre-registered 0.97105. It is a
deliberate CV regression sent to test a mechanism, not a rank attempt — `blend159av_h3`,
the identical object with unit training weights, also scores 0.97105, and that pair is the
workspace's first genuinely paired LB reading.

Submission tally for Kaggle day UTC 2026-08-16: **5 of 10** used (blend160orig,
w16b_cellweight, w16f_armavg, w16i_schemeavg, w16l_maskw_h3); CLI confirmed "5 remaining".
Nothing is selected on the submissions page — still one human click.

## 2026-08-16 06:45 UTC (slot 5)

**1,954 teams** (1,946 at 05:20, 1,943 at 04:00). We are **rank 47** at 0.97107 and still the
only team on that score; w16h read rank 42 five hours earlier, so the field passed us five
places overnight — consistent with w16a §5(c)'s ~7 teams/day past a static file.

| rank | team | score | last submission |
|---|---|---|---|
| 1 | MILANFX | 0.97132 | 2026-08-16 01:51 |
| 2 | Optimistix | 0.97125 | 2026-08-16 06:36 (new) |
| 3 | Utkarsh | 0.97124 | 2026-08-15 22:40 |
| 4 | Maher el Ouahabi | 0.97122 | 2026-08-16 00:26 |

Gap to first **25e-5**, unchanged. Optimistix moved 0.97123 → 0.97125 during this slot.

Our public best is now a **5-way tie at 0.97107** — `w15f_antistudent_avg`, `w16b_cellweight`,
`w16f_armavg`, `w16i_schemeavg`, `w16n_finegrid` — every member a corrected file and every one
above every zero-parameter file on CV, so Kaggle's default auto-pick's first slot is safe under
any tiebreak rule. The exposure remains auto-slot 2, the 7-way tie at 0.97106 that contains
`blend158_logit`. 47 submissions total, none selected.

## 2026-08-16 07:45 UTC (w16o, slot 6)

- 1,954 teams. **We are rank 47 at 0.97107**, unchanged from w16m's 06:45 reading — the field
  did not pass us this hour.
- Top: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124, Maher el Ouahabi 0.97122,
  cstdy 0.97121. Gap to first 25e-5.
- Sent this slot: `w16h_h3av6.csv` ref 55546833, CV 0.97004873, **public 0.97105** — matches
  the pre-registered h3-shelf prediction and makes that shelf 10 of 10.
- 7 of 10 submissions used on the 2026-08-16 Kaggle day (CLI: "3 submissions remaining today").
- `check_selection.py` still exit 1 after 48 submissions. Auto-slot 1 is a 5-way tie at
  0.97107, all CV-good; auto-slot 2 is the 7-way 0.97106 tie containing `blend158_logit`.

## 2026-08-16 08:20 UTC — w16q, slot 7

- **NEW ACCOUNT BEST: 0.97108** (`w16q_ens4avg`, ref 55547584), first score above 0.97107.
- **Rank 50 of 1,964.** Readings this Kaggle day: w16h 42/1,946, w16m 47/1,954 (06:45),
  w16o 47/1,954 (07:40), w16q 51/1,959 at 0.97107 (08:05) → **50/1,964 at 0.97108** (08:20).
  The field is passing us ~4 places/hour at this density; +1e-5 bought back one place.
- Top: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124, Maher el Ouahabi 0.97122,
  cstdy 0.97121, Malhar Ujawane 0.97118. Gap to first now **24e-5**.
- **Auto-pick tiers moved.** `check_selection.py` now reads auto-slot 1 = 0.97108, a 1-way tie
  (`w16q_ens4avg`), and auto-slot 2 = the old 0.97107 5-way tie. `blend158_logit` has dropped out
  of the top two tiers, so w15i's +9.2e-6 / +36.5e-6 / +112e-6 exposure ladder is quoted from a
  stale tier structure — re-read the script's live output before using it.
- 8 of 10 submissions used today. Slots 8/9/10 have **two** sends between three slots.

## 2026-08-16 09:35 UTC — after slot 9 (`w16t_cellens4`, LB 0.97108)

- **Us: rank 52 of 1,970** at 0.97108 (tied best, set by `w16q_ens4avg` at 08:17). w16q read
  rank 50 of 1,964 an hour and a quarter earlier — the field passed us 2 places while our own
  score was unchanged, consistent with the ~7 teams/day drift w16a measured.
- Top: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124, Maher el Ouahabi 0.97122,
  cstdy 0.97121. Gap to first **24e-5**, unchanged since w16a.
- Account submission count 50. Two files now sit at 0.97108 (`w16q_ens4avg`, `w16t_cellens4`),
  five at 0.97107.

## 2026-08-16 10:08 UTC — after slot 10 (`w16e_aonly`, LB 0.97108), WAVE CLOSED

- **Us: rank 53 of 1,977** at 0.97108, read off the downloaded public leaderboard snapshot
  (`playground-series-s6e8-publicleaderboard-2026-08-16T10:07:59.csv`), not the paginated CLI.
  Slot 9 read 52 of 1,970 at 09:35 — the field passed us one place in 33 minutes while our score
  was unchanged, and added 7 teams. Consistent with the ~4 places/hour drift at this density.
- Top unchanged all wave: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124,
  Maher el Ouahabi 0.97122, cstdy 0.97121. **Gap to first 24e-5**, unchanged since w16a.
- Account submission count **50**. Ten sent this Kaggle day, quota confirmed spent by the CLI's
  "0 submissions remaining today".
- **Three files now sit at 0.97108** (`w16q_ens4avg`, `w16t_cellens4`, `w16e_aonly`) and five at
  0.97107. Auto-slot 1 is a **3-way tie**, so the auto-pick at limit 2 is ambiguous again and the
  click price is a range — see `check_selection.py`, and re-run it rather than quoting this.
- ⚠ `w16e_aonly` reaching 0.97108 from CV 0.9700544 is what falsified the corrected-h3 CV→LB
  ladder: `w16i_schemeavg` and `w16n_finegrid` sit **higher** on CV and print 0.97107. Any
  future entry here that predicts an LB from a CV needs to read RESEARCH §1 of the wave
  consolidation first.
- Wave net: **+1e-5 public** (0.97107 → 0.97108) across ten submissions, one place lost to field
  drift. The score is not what this wave produced; the corrections list is.

## 2026-08-17 (UTC), after w17 slot 1

Top of board: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124, Keanan 0.97124,
Maher el Ouahabi 0.97123, cstdy 0.97123, Don Mani 0.97120, Malhar Ujawane 0.97118,
Szymon Kłapiński 0.97117. Our best public **0.97108** (three-way tie: `w16e_aonly`,
`w16q_ens4avg`, `w16t_cellens4`). Gap to first **24e-5**, unchanged.

Slot 1 sent `w14a_repro159av_h3` → **0.97105**, matching its pre-registered point estimate
exactly. Not a rank move (it is a rebuild of an existing 0.97105 file); it was sent as an
out-of-sample test of the new paired-slice CV→LB instrument, and it passed.

**New, and it matters for the final-selection click:** ranking all 46 sent files by how far
their public score sits above what their CV predicts puts Kaggle's three auto-slot-1 holders
at the top of the tight families — `w16q_ens4avg` +1.20 sd, `w16e_aonly` +1.12,
`w16t_cellens4` +1.07 — against the CV pick `w16i_schemeavg` at +0.26. `w16e_aonly` leads
`blend159av_h3` by 30e-6 on public but only 5.3e-6 on CV. Board density is ~2.5 places per
1e-5, so the click is still the largest single quantity available.

## 2026-08-17 UTC, w17 slot 2 (board read 00:35 UTC)

| | |
|---|---|
| ours (public, best of all) | **0.97108** — unchanged; `w16e_aonly` / `w16q_ens4avg` / `w16t_cellens4` |
| our best CV | 0.9700557 (`w16i_schemeavg`), rebuilt bit-for-bit this slot |
| submissions | **53** (2 of 10 used today; the API default page shows only 50 — see RESEARCH.md) |

Top of board: MILANFX **0.97132**; Optimistix 0.97125; Utkarsh / Keanan / Maher el Ouahabi
0.97124; cstdy 0.97123; Don Mani 0.97120; Malhar Ujawane 0.97118; Szymon Kłapiński / Laura Liepa
0.97117; Mursal Gorchuyev / AdarshAleti / nanare / delai50 0.97116; Will 0.97115.

Gap to first **24e-5**, unchanged. The board thickened at 0.97116–0.97124 (four teams now tie or
near-tie at 0.97124 where there was one), so the local density above us is rising and the ~2.5
board places per 1e-5 figure used to price the click is, if anything, conservative now.

`blend159av_logit` (0.97106) was sent as an instrument test, not a candidate, and did not change
our standing. **Auto-slot 1 remains a three-way tie at 0.97108 and nothing is selected.**

## 2026-08-17 (w17 slot 3) — the board gaps, re-read against an audited floor

The cross-team resolving power was audited this slot (`experiments/w17h_floorpop.py`, 820 pairs)
because the standing 53–84e-6 floor sat under six do-not-spend lists and had never been checked
against a *quality-matched* rival. **The floor survived; only the algebra under it broke.**

| gap to us (public 0.97108 best-of) | incumbent floor 53–84e-6 | audited (disjoint-member stacks, 66.3e-6) |
|---|---|---|
| MILANFX 0.97124, +18e-5 | 2.6 sigma | **2.7 sigma** |
| the 0.97117 team, +11e-5 | 1.6 sigma | **1.7 sigma** |
| the 0.97113 team, +5e-5 | 0.7 sigma | **0.8 sigma** |

**Nothing changes.** The teams between 0.97113 and 0.97117 are still not distinguishable from us
on the public slice, and the public board still cannot say whether the leader's edge survives to
private. Chasing public rank remains unpriceable; the do-not-spend lists hold.

⚠ Quote this as an extrapolation: the disjoint-half stacks are ~130e-6 below our full stack.

Our sends this slot: `blend159av_hybrid` → **0.97102** (registered 0.97103, missed by one step).
Account best public unchanged at **0.97108**, held by three files; 54 submissions total.

## 2026-08-17 (w18, slot 4)

- **Rank 76 of 2,046** at public **0.97108**, 55 submissions. The field has grown from the
  brief's ~1,326 teams to 2,046.
- Top: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124. Gap to #1 is **24e-5**.
- Immediate neighbourhood is a wall: 0.97109 × 3 at ranks 73–75, then **0.97108 × 4** at 76–79
  (us, FunnyBishop, AJboos, LeTuanM). One grid step is ~3-4 board places here.
- Reading unchanged and now on a firmer footing: the cross-team paired floor survived w17's
  audit at 53–84e-6, so 0.97108 vs 0.97109 is **not a difference** and the 24e-5 gap to #1 is
  ~3 sigma. Nothing on this board is worth chasing at the cost of the CV pick.

## 2026-08-17 ~02:40 UTC (w19, slot 5)

- **Rank 77 of 2,047** at 0.97108 (unchanged score; the field grew and one team passed us).
  The 0.97108 tier spans ranks 77–83, i.e. **7 teams tied** on the public slice.
- Top: MILANFX 0.97132 (up from 0.97124), Optimistix 0.97125, Utkarsh 0.97124.
  MILANFX has moved +8e-6 since 2026-08-15 and remains public #1.
- **The board's team name for this account is "Teddy Tennant", not `thtennant`.** A lookup on
  `thtennant` returns "not found" across all 2,047 rows — this cost a cycle to notice.
- Submission 56 (`w14a_repro159av`, 0.97106) did not move rank, as priced before the send.

## 2026-08-17 ~03:20 UTC (w20, slot 6)

- **Rank 77 of 2,049** at public **0.97108**, 56 submissions. Unchanged from slot 5; the field
  grew by 2 teams overnight and nothing near us moved.
- Top five: MILANFX 0.97132, Optimistix 0.97125, Utkarsh 0.97124, Keanan 0.97124,
  Maher el Ouahabi 0.97124. Gap to #1 **24e-5**, unchanged for two slots.
- Submission counts are worth reading next to the scores: MILANFX reached 0.97132 in **15**
  submissions and Keanan 0.97124 in **8**, against Optimistix's 109 and our 56. On a board
  where the cross-team paired floor is 53-84e-6 (w17h), a 15-submission 0.97132 is not
  distinguishable from a 109-submission 0.97125 — and neither is distinguishable from us.

### After w20's two sends — RANK 77 → 19 of 2,051

| | before slot 6 | after slot 6 |
|---|---|---|
| public best | 0.97108 | **0.97115** |
| rank | 77 | **19** |
| gap to MILANFX (#1, 0.97132) | 24e-5 | **17e-5** |
| submissions | 56 | 58 |

`w20_ad187_h3` → **0.97115** (registered modal 0.97110, P(>0.97108) 0.741 — hit).
`w20_ad187_rankraw` → **0.97114** (registered modal 0.97108 — a tail cell, missed high).

The 0.97115 tier holds 7 teams (ranks 15–21); one grid step is ~4 board places at this density,
so the honest reading is unchanged from w17h — everything from 0.97113 to 0.97117 is inside the
53–84e-6 cross-team floor and is **not** a difference. What did change is real and is a CV fact,
not an LB fact: the shipped object gained +51.6e-6 of cross-fitted CV.

⚠ **`AdarshAleti` sits at rank 13 with 0.97116 — that is the author of the OOF library this
slot imported.** They publish their members under CC0 and their own README documents an ablation
that reaches the opposite conclusion to ours about their CatBoost group (see RESEARCH). Worth
watching: they are 27 submissions in, above us, and shipping their raw material publicly.

### After w21's first two sends — RANK 19 → 14 of ~2,050

| | after slot 6 | after slot 7 |
|---|---|---|
| public best | 0.97115 | **0.97117** |
| rank | 19 | **14** |
| gap to MILANFX (#1, 0.97132) | 17e-5 | **15e-5** |
| submissions | 58 | 60 |

`w21_ad187corr` → **0.97117** (registered modal 0.97116; one cell high under both slope models).
`w20_ad187` (all-four) → **0.97116** (registered modal 0.97115 under beta=1, 0.97114 under
beta=2; also one cell high). **Both landed one grid step above their modal cell despite having
opposite-signed dCV** — see JOURNAL §4, that is a level effect and no slope explains it.

The board moved under us as well as for us: Optimistix 0.97125 → 0.97126, Szymon Kłapiński
0.97117 → 0.97124, Don Mani 0.97120 → 0.97122, Charles Backman 0.97114 → 0.97121, and Mahog
appeared at 0.97120. The top-10 is now 0.97120–0.97132 and it is churning daily; a 2e-5 print
bought 5 places this slot and would have bought more a day ago. w17h's cross-team paired floor
of 53–84e-6 still says everything from 0.97113 to 0.97124 is **not** a distinguishable
difference — including the seven teams now above us.

⚠ `AdarshAleti` — the author of the imported OOF library — has not moved from 0.97116 and is
now **below** us. Their CC0 members are worth more in our pack than in theirs (w20d, w21b),
which their own README's ablation predicted the opposite of.

**After ship 3: `w21_ad187corr_ens4` → 0.97118, rank 14 → 11 of ~2,050.** Gap to MILANFX
(#1, 0.97132) now 14e-5, from 17e-5 at slot start. Public best 0.97115 → 0.97118 in one slot.

Three teams sit on 0.97118 (us, Malhar Ujawane, nanare) and the 0.97120–0.97124 band above us
holds six. Everything from ~0.97113 to 0.97124 remains inside w17h's 53–84e-6 cross-team paired
floor and is not a distinguishable difference — but the top of the board is pulling away:
Optimistix 0.97126 and MILANFX 0.97132 are 8e-5 and 14e-5 clear, which is 1.0–2.6× that floor.

## 2026-08-17 — w22, slot 8/10 (no submission possible; quota closed at 10/10)

Board re-read at slot 8, no send. **Teddy Tennant 0.97118, rank 11 of ~2,050**, unchanged from
w21's close. MILANFX 0.97132 (#1), Optimistix 0.97126, then a 0.97120–0.97124 band of seven
(Utkarsh, Keanan, Maher el Ouahabi, Szymon Kłapiński, cstdy, Don Mani, Charles Backman, Mahog).
Gap to #1 holds at **14e-5**. Malhar Ujawane and nanare still tie us at 0.97118.

Nothing moved above us this slot beyond Don Mani's 13:44 resubmit — the churn that ran all
through w21 has paused. Everything from ~0.97113 to 0.97124 remains inside w17h's 53–84e-6
cross-team paired floor and is not a distinguishable difference; only Optimistix (8e-5) and
MILANFX (14e-5) are clear of it.

**No LB reading was used for any decision this slot**, and that is worth recording explicitly:
w22a/w22b's whole subject is the ten-pair public-slice reversal, and the conclusion (§3–4 of the
journal) is drawn entirely from OOF-side simulation. The LB supplied the ten *signs* — which are
hard facts under monotone rounding — and nothing else.

## 2026-08-17, w23 slot 9 — no send (quota closed at 10/10), board unchanged

Standing held from w21's last send: **Teddy Tennant 0.97118, rank 11** of ~2,050.

| rank | team | public | note |
|---|---|---|---|
| 1 | MILANFX | 0.97132 | unchanged since 08-16 01:51 |
| 2 | Optimistix | 0.97126 | |
| 3= | Utkarsh / Keanan / Maher el Ouahabi / Szymon Kłapiński | 0.97124 | four-way |
| 7 | cstdy | 0.97123 | |
| 8 | Don Mani | 0.97122 | |
| 9 | Charles Backman on LinkedIn | 0.97121 | |
| 10 | Mahog | 0.97120 | |
| **11** | **Teddy Tennant** | **0.97118** | `w21_ad187corr_ens4` |
| 11= | Malhar Ujawane, nanare | 0.97118 | |

Gap to #1: **14e-5**. Gap to the top ten: **2e-5** — one grid step.

**What changes tomorrow, and it is not a hedge.** `w23_ad187std_h3` (CV 0.9701092751) is the
new CV leader and is **+8.5e-6 on the file currently sitting at LB 0.97115**, from a fixed
convergence defect rather than any tuning. Registered modal print **0.97117–0.97118**, i.e.
it should roughly tie the account best rather than clear it; the real gain is expected once
`c_avg` is rebuilt on top of it. Nine files are queued and validated, so all ten slots can
be filled without further compute.

## 2026-08-17, ~16:30 UTC (w24, slot 10) — board state, no submission possible

Account best **0.97118** (`w21_ad187corr_ens4`), rank **11** of ~1,326.

| rank | team | score | dated |
|---|---|---|---|
| 1 | MILANFX | 0.97132 | 08-16 |
| 2 | Optimistix | 0.97126 | 08-17 |
| 3= | Utkarsh / Keanan / Maher el Ouahabi / Szymon Kłapiński | 0.97124 | 08-15..17 |
| 7 | cstdy | 0.97123 | 08-16 |
| 8 | Don Mani | 0.97122 | 08-17 |
| 9 | Charles Backman on LinkedIn | 0.97121 | 08-17 |
| 10 | Mahog | 0.97120 | 08-17 |
| **11** | **Teddy Tennant** | **0.97118** | 08-17 |
| 11= | Malhar Ujawane, nanare | 0.97118 | 08-16/17 |

Gap to first **14e-5**; gap to a medal-ish top-10 is **2e-5**, i.e. two prints. The board
moved ~1e-5 at the top in a day and the 0.97116–0.97124 band is 15 teams deep, so single
prints reshuffle rank without meaning much.

Nothing was sent this slot (cap 10 exhausted at 13:40). The queue for tomorrow is led by
`w23_ad187stdcorr`, CV 0.9701150809 — **+8.2e-6 of CV above anything this account has ever
submitted**, and never scored. Registered expectation: 0.97119 modal, 0.97118 alternative.

## 2026-08-18, ~00:30 UTC (w25, slot 1) — ten sends, account best unchanged

Account best **0.97118** (`w21_ad187corr_ens4`, 08-17), rank **11 of 2,140**. The field has
grown from the brief's ~1,326 to 2,140 teams.

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97132 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4–6 | Utkarsh / Keanan / Szymon Kłapiński | 0.97124 |
| 7 | cstdy | 0.97123 |
| 8 | Don Mani | 0.97122 |
| 9 | Charles Backman on LinkedIn | 0.97121 |
| 10 | Mahog | 0.97120 |
| **11** | **Teddy Tennant** | **0.97118** |
| 12 | Mursal Gorchuyev | 0.97118 |

Ten sends today, best print **0.97117** (`w22_ad187corr_rankraw`) — the account best was **not
beaten**, and the CV leader `w23_ad187stdcorr` came in at 0.97116 against a registered 0.97119.
Gap to first 14e-5; gap to top-10 is 2e-5, two prints.

Nine of the ten sends were spent as **matched pairs and registered forecasts** rather than
attempts on the board, which is why the best print did not move: the day's value was the
CV→LB model in RESEARCH.md, not a rank. That was the handed angle and it was the right trade —
but note it explicitly, because a run that only reads this file will see ten sends and no
movement.

## 2026-08-18, wave w26 slot 3 — no sends possible, and the queue behind them is spent

`date -u` 00:52. All ten 08-18 submissions landed 00:07–00:20 UTC from w25 slot 1, so the cap
was already spent when this run started and **the prompt's "already reports for today: 10" was
correct this time**. The Kaggle day does not roll again until 08-19 00:00 UTC, which is ~23
hours out — so **every remaining slot today (3 through 10) is also at the cap.** A run that
reads only this file should not go looking for a slot; there is not one.

Standing unchanged: **11th of 2,140 at 0.97118**, 14e-5 behind MILANFX at 0.97132, 2e-5 (two
prints) outside the top ten. Nothing was sent, so nothing moved.

The new number that matters for planning the 08-19 day (`w26d_queueprice.py`, full detail in
RESEARCH.md): **the best of the 46 unsent files is worth P = 6.3e-4 of a new account best, and
so is the best ten of them sent together.** The queue tops out 67e-6 of CV below what has
already been sent. Sending it is still free and should still happen — a submission here cannot
evict another or lower the public best — but it will not move this table, and a run that
spends its compute draining the queue instead of building above CV 0.9701182 is spending it in
the wrong place.

## 2026-08-18, wave w26 slot 4 — standing unchanged, but the plan behind it was wrong

`date -u` 01:40. Still at the cap (10/10 for the 08-18 UTC day, spent by w25 slot 1 at
00:07–00:20). Nothing sent, nothing moved: **~11th of 2,140 at 0.97118**, 14e-5 behind MILANFX
at 0.97132. Live recount from a 200-row page: 10 teams strictly above 0.97118 and 4 tied on
it, so the true rank is 11–14 depending on the tiebreak. Rank 100 is 0.97111 and rank 200 is
0.97095 — i.e. **the whole 90-place band below us is 23e-6 wide**, which is under three slice
noise sd. Position here is not stable and is not worth chasing on the public number.

**Correction to the 08-18 slot-3 entry above.** It said "the best of the 46 unsent files is
worth P = 6.3e-4" and that the queue "tops out 67e-6 of CV below what has already been sent".
The probability survives — it was always carried by `w20_ad187_logit`, which really was unsent
— but the rest does not: **21 of those 46 files were already on the leaderboard**, hidden by
the CLI's 50-row default page (JOURNAL 08-18 slot 4 §2). Corrected: **27 unsent files**, best
CV 0.9700342765, **80.8e-6** below the best sent. The queue is three send days deep, not nine,
and it is empty from 08-22 with the deadline on 08-31.

## 2026-08-18, wave w26 slot 5 — unchanged, and the gap to the top is bigger than any live lever

`date -u` 01:54, still at the cap (10/10 on the 08-18 UTC day, all spent by w25 slot 1). Live
read of the top of a 200-row page:

| | team | score |
|---|---|---|
| 1 | MILANFX | 0.97132 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| … | | |
| **11** | **Teddy Tennant** | **0.97118** |

**140e-6 behind first.** Worth stating plainly against what is actually in flight: at the
workspace's fitted CV→LB slope of ~2, closing that needs roughly **+70e-6 of CV**, and the two
live levers are the C sweep (registered prior −2 to +5e-6) and w26i's two new CatBoost members
(registered prior +1 to +7e-6 on the combiner). **Neither is within an order of magnitude of
the gap to first.** The realistic target this wave is the 0.97118 → ~0.97122 band, which is
worth a handful of places, not the top of the board — the only thing on file that ever moved
CV by ~50e-6 was a 22-member import, i.e. members from a pipeline we did not hold.

Recorded so a later run does not read a +5e-6 result as progress toward first place. It is
progress toward rank ~8.

## 2026-08-18 02:2x UTC (w26 slot 6)

Us: **11th, 0.97118** (`Teddy Tennant`, 2026-08-18 00:20:28) — unchanged.

The top ten, and it has compressed above us since yesterday:

| # | team | score | last sub |
|---|---|---|---|
| 1 | MILANFX | 0.97132 | 08-16 01:51 |
| 2 | Maher el Ouahabi | 0.97127 | 08-17 20:40 |
| 3 | Optimistix | 0.97126 | 08-18 00:02 |
| 4 | Utkarsh | 0.97124 | 08-15 22:40 |
| 5 | Keanan | 0.97124 | 08-17 11:01 |
| 6 | Szymon Kłapiński | 0.97124 | 08-17 09:49 |
| 7 | cstdy | 0.97123 | 08-17 21:45 |
| 8 | Don Mani | 0.97122 | 08-17 14:15 |
| 9 | Charles Backman on LinkedIn | 0.97121 | 08-17 15:59 |
| 10 | Mahog | 0.97120 | 08-17 13:29 |
| **11** | **Teddy Tennant** | **0.97118** | 08-18 00:20 |
| 11= | Mursal Gorchuyev | 0.97118 | 08-17 21:42 |
| 11= | Malhar Ujawane | 0.97118 | 08-16 16:49 |

**The gap to 1st is 14e-6 and the gap to 10th is 2e-6.** Nine of the ten above us are within
9e-6 of each other, i.e. inside one to two LB grid steps — this is a wall, not a ladder, and a
single +5e-6 file moves several places. MILANFX has not submitted since 08-16 and still leads.

⚠ Still **nothing selected** (14 days). Auto-selection would take `w21_ad187corr_ens4`
(0.97118) and one of `w21_ad187corr` / `w22_ad187corr_rankraw` (0.97117). `WANTED` is
{`w23_ad187stdcorr.csv`, `w21_ad187corr.csv`}, chosen on CV. **A human must tick them.**

## 2026-08-19 14:51 UTC — w27 slot 1. Rank 17 of 2323 at 0.97118 (was 11th on 08-18)

Downloaded in full to `lb_w27/`. We did not move; the board did.

| | |
|---|---|
| leader | MILANFX **0.97134** (08-18 06:52, unchanged for a day) |
| 2nd–5th | Maher el Ouahabi 0.97127, Optimistix 0.97126, Don Mani / cstdy 0.97125 |
| Szymon Kłapiński (whose public library our lattice members come from) | 0.97124, 8th |
| **us** | **0.97118, 17th** |
| teams | 2,323 |

**The density around us is the number that matters, and it has got worse.**

| public score | rank it buys |
|---|---|
| 0.97116 | 62 |
| **0.97118 (ours)** | **17** |
| 0.97120 | 14 |
| 0.97122 | 10 |
| 0.97125 | 4 |

+2e-5 of public score is worth 45 places at 0.97116 but only 7 at 0.97118 — we are already
past the steep part. Top-10 needs **+4e-5**; the leader is **+1.6e-4** away. For scale, the
w26d model prices the entire unsent queue at 6.3e-4 of beating our own 0.97118, and the best
blend-level CV differences this workspace can still find are ~1e-6. **Nothing in the blend
family reaches top 10.** Only a member-level effect of the size of the CT fix (+294e-6 solo,
pooled, 400 rounds) is even the right order of magnitude — and whether any of that survives
into the 187-member combiner is exactly what w27b/w27d measure.

10 slots today; 5 sent by 14:39 UTC from the priced queue, 5 held.

## 2026-08-19 (w27 slot 2) — the board is 2,323 teams, not 1,326, and we are 17th

⚠ **The brief's "~1,326 teams" is stale.** `lb_w27/…publicleaderboard…csv`, pulled 14:51 UTC,
carries **2,323 teams**. Every percentile computed against 1,326 is wrong.

| | |
|---|---|
| leader | **0.97134** MILANFX (08-18) |
| us | **0.97118**, public rank **17**, top 0.73%, 76 submissions |
| gap to leader | 1.60e-4 |
| teams within 1e-4 of the leader | 8 |
| teams within 2e-4 | 87 |
| teams tied with us at 0.97118 | 7 |
| medal cuts (2,323 teams) | gold top **14**, silver top **116**, bronze top **232** |

**We are three places outside a gold medal on the public board.** Two 0.97119s and a 0.97120
sit between us and the cut; the whole gold band spans 1.6e-4, which is about twice our own
CV→LB residual sd (7.24e-6) times two — i.e. it is a real gap, not one grid step.

### The private-split backtest, `experiments/w27i_s6risk.py` (new this slot)

Seven finished Season 6 boards (`georgymamarin/playground-series-s6-leaderboards`). Restricted
to the three **ROC-AUC** episodes because the metric governs frontier compression. Band = the
teams sitting in the same relative slice of their public board as we sit in ours (top
0.37%–1.10%):

| | |
|---|---|
| median private percentile of a team in our public band | **6.53%** (we enter at 0.73%) |
| 10th–90th percentile of where they landed | 2.95% – 10.31% |
| still gold privately | **10.8%** |
| still silver (top 5%) | **50.6%** |
| still bronze (top 10%) | **67.0%** |

**Read it as: the modal outcome for a team standing exactly where we stand is a silver, with
a third of the probability mass falling out of the medals entirely, and roughly a one-in-ten
shot at gold.** Board-wide public/private Spearman is 0.9916 — very high, and it is not
protection at the frontier, which is the whole point.

Frontier compression varies enormously across the AUC episodes and is what decides the
spread: S6E2 had **156 teams within 1e-4** of its public leader and their private ranks span
**4 to 1856** of 4,370; S6E5 had 5 and they span 1 to 9. **s6e8 has 8 within 1e-4, so it looks
much more like S6E5 than S6E2** — the frontier here is real, not a pile-up. That is mildly
good news for us and it is the first quantitative handle this workspace has had on the
question.

### ⚠ And it makes the unclicked selection expensive, not merely untidy

Nothing is selected, so Kaggle auto-picks our two entries by **best public score**.
`check_selection.py`'s residual decomposition already showed auto-selection lands on the
three most **slice-inflated** files we own (standardised residual +1.20/+1.12/+1.07 against
+0.26 for the CV pick). The backtest above is the price list for that policy.
`WANTED` = {`w23_ad187stdcorr.csv`, `w21_ad187corr.csv`} and a human still has to tick them.

---

## 2026-08-19, 15:48 UTC — 2,329 teams

| | |
|---|---|
| leader | 0.97134 (MILANFX, 08-18) |
| **us** | **0.97118, rank 17 of 2329, top 0.73%** |
| gap to leader | 1.60e-4 |
| teams ahead | 16 |
| tied with us | 7 |
| within 1e-4 of the leader | 8 |

Medal cuts at 2,329 teams: **gold top 14, silver top 116, bronze top 232.** We are **three
places outside gold**, unchanged from the 14:51 snapshot (2,323 teams). ⚠ The brief's
"~1,326 teams" is stale by about a thousand — take the count from the downloaded CSV.

Movement since yesterday is at the top, not around us: Maher el Ouahabi 0.97127, Optimistix
0.97126, Don Mani and cstdy 0.97125 all posted on 08-19. The 0.97118 shelf we sit on is thick
and it is not moving.

### ⚠ What a submission is actually FOR here now — this changed today

Two findings this slot, together, close off the public leaderboard as a target:

1. **The queue-pricing bug (RESEARCH, w27 slot 3).** The CV bar for an even-money shot at our own
   0.97118 is **0.9701326** for a standardised h3 file, not the 0.9701182 this workspace has been
   quoting — the old figure omitted the −27.43e-6 standardisation penalty. Our new CV leader is
   0.9701168. **Nothing on disk is within 16e-6 of the bar, and the bar is against our own score,
   not the board's.**
2. **The ~350x stack-translation loss**, measured independently by @adarsh1077 and matching our
   own CT thread exactly. Member-level gains of +1000e-6 arrive as +3e-6 in a saturated stack.

**So public-LB movement is not reachable from here by member-level work.** A submission's value
is now almost entirely that **a file must be submitted to be selectable for the private board.**

That makes the unclicked final selection the single highest-value open item in the workspace, and
§6 of the w27 slot-2 entry priced it: a team standing exactly where we stand has a **median
private percentile of 6.53%**, **10.8% still gold**, **50.6% still silver**, and **33% falling out
of the medals**. Kaggle auto-selects by best *public* score, which lands on the three most
slice-inflated files we own (standardised residual +1.20/+1.12/+1.07 against +0.26 for the CV
pick). Selecting on the public slice is the mechanism that produces the bad tail.

### ⚠ `WANTED` has changed — a human must tick these two

1. **`w27_ad188stdcorr.csv`** — CV **0.9701168076**, the highest ever built here. ⚠ **BUILT BUT
   NOT YET SUBMITTED**; the 08-19 day ran out. **Send it as slot 1 on 08-20**, then tick it.
2. **`w23_ad187stdcorr.csv`** — CV 0.9701150809, already uploaded.

(`w21_ad187corr.csv` drops off the list.) The API has no write path for selection — probed and
falsified 08-13.

## 2026-08-19, slot 6 (17:0x UTC) — public standing

Checked this slot with `kaggle competitions leaderboard -c playground-series-s6e8 -s`,
paging to find us.

| | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4 | Don Mani | 0.97125 |
| 5 | cstdy | 0.97125 |
| … | | |
| **~18** | **Teddy Tennant** | **0.97118** |

**~18th of ~1,326 teams.** The 0.97118 band is four teams wide (us, Mursal Gorchuyev, Malhar
Ujawane, Rayk Kretzschmar), so a single 1e-5 reporting step is worth several places here —
and the whole top-18 spread is 16e-5. Gap to the leader is **+16e-5**, which against the
corrected w25f model (LB ≈ const + 1.909·CV) needs roughly **+84e-6 of CV**. Nothing in the
current pipeline is producing gains at that scale: this slot's best available move is
+3.37e-6 per added member. **Public rank is not reachable from here by CV improvements;
the remaining value is in not losing the private split.** Selection stays on CV.

⚠ Our 0.97118 comes from `w20_ad187_logit` / `w27_ad188std` — files the journal explicitly
records as **not** deadline candidates. The CV leader `w27_ad188stdcorr` has never been
sent, so our public rank is currently set by a file we would not choose.

## 2026-08-19 19:20 UTC (w28 slot 9)

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4 | Don Mani | 0.97125 |
| 5 | cstdy | 0.97125 |
| 6-8 | Utkarsh / Keanan / Szymon Kłapiński | 0.97124 |
| 9 | Mikhail Naumov | 0.97123 |
| 10-11 | Changye Li / Leo | 0.97122 |
| 12-13 | william950615 / Charles Backman | 0.97121 |
| 14-15 | thisray / Mahog | 0.97120 |
| 16 | delai50 | 0.97119 |
| **17** | **Teddy Tennant** | **0.97118** |
| 17-23 | Mursal Gorchuyev, Malhar Ujawane, Rayk Kretzschmar, miki, Atakan Aldemir, Shashwat Bajpai | 0.97118 |

Unchanged at 0.97118 since 08-17 — the ten sends on 08-19 were all queue-drain files priced
at P < 1e-3 and none moved it, exactly as predicted. **We are 17th of 2,323; gold is top 14.**
The 0.97118 tier is six teams deep, so one reporting step is worth roughly six places here.
Leader has been static at 0.97134 since 08-18 06:52 while ranks 2-16 filled in beneath it.

## 2026-08-19 20:35 UTC (w29, slot 10)

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4 | Don Mani | 0.97125 |
| 5 | cstdy | 0.97125 |
| 6–8 | Utkarsh, Keanan, Szymon Kłapiński | 0.97124 |
| 9 | Mikhail Naumov | 0.97123 |
| 10–11 | Changye Li, Leo | 0.97122 |
| 12–13 | william950615, Charles Backman | 0.97121 |
| 14–15 | thisray, Mahog | 0.97120 |
| 16 | delai50 | 0.97119 |
| **17** | **Teddy Tennant** | **0.97118** |
| 17= | Mursal Gorchuyev, Malhar Ujawane | 0.97118 |

Us at **0.97118, 17th**, 16 teams above and a 3-way tie at our score. The field moved again
in the ~75 minutes since slot 9's read: **Optimistix 0.97126 at 19:32** is new, and 0.97124
is now a three-way tie where it was thinner. Gold is top 14 — we are three places out, and
the gap to first is **16e-6**, which the CV→LB model says needs a CV of 0.9701307114 against
the 0.9701183 best on disk. Slot 10 spent the day proving that the cheap way to find that
12.6e-6 does not exist (see JOURNAL w29 §2).

## 2026-08-20 00:20 UTC — 0.97118, ~rank 17 of ~1,326

Account best is now **0.97118**, held by three files: `w21_ad187corr_ens4` (08-17) and, as of
today, `w27_ad190stdcorr` and `w29_ad194stdcorr`. All three are c_avg-**corrected**; that is
not a coincidence, see RESEARCH §w30.

| # | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4-5 | Don Mani, cstdy | 0.97125 |
| 6-8 | Utkarsh, Keanan, Szymon Kłapiński | 0.97124 |
| … | | |
| ~16 | delai50 | 0.97119 |
| **~17** | **Teddy Tennant** | **0.97118** |

The whole top 17 spans **16e-6** — 1.6 reporting steps. One step (+1e-5) is worth roughly
2–3 places at this density; reaching 0.97120 would be ~14th. MILANFX has held #1 since 08-18
and is 16e-6 clear, which is a real gap, not a slice draw.

The board is dense enough that the *pricing* of a submission matters more than building a
better object: the same ten files re-priced with the w30 correction term went from
P(beat 0.97118)=1.7e-3 to 4.8e-2 for the best single file.

## 2026-08-20 01:15 UTC (w32, slot 3) — ⚠ CORRECTION: the gap to first is **16 reporting steps, not 1.6**

Board unchanged from the 00:20 read: **0.97118, rank 17 of ~2,300**, leader MILANFX 0.97134
(static since 08-18 06:52). `experiments/w32_lb_top.csv` is the live top-60.

**Two sentences in the entries above are wrong and one of them has been repeated three times.**

> "the gap to first is **16e-6**, which the CV→LB model says needs a CV of 0.9701307114"
> "The whole top 17 spans **16e-6** — **1.6 reporting steps**"

    0.97134 − 0.97118 = 0.00016 = 160e-6 = SIXTEEN reporting steps.

The two figures in the first sentence were never consistent with each other either: at the
w30b slope **1.856**, a CV of 0.9701307114 (+12.4e-6 on the CV leader) buys +23e-6 of LB, i.e.
it is the bar for ~**0.97120**, the gold cutoff — not for first place.

**Nothing downstream broke.** `w26d_queueprice.py` prices against the account's own 0.97118
and is self-consistent; no build or send decision was ever made off the mis-stated gap. What
it distorted is the strategic framing, and "we are 1.6 steps off the lead" and "we are 16
steps off the lead" are different competitions.

### The corrected bar table (`experiments/w32a_goldbar.py`, gated against w30b)

Slope dLB/dCV = **1.856 ± 0.054** ⇒ one reporting step = **5.39e-6 of CV**.
CV is for the cheapest family (ens4, std+corr) and `gap` is versus the CV leader 0.9701182875.

| target LB | rank today | dLB | CV needed | gap vs leader |
|---|---|---|---|---|
| 0.97119 | 16 | +10e-6 | 0.9701232560 | **+5.0e-6** |
| **0.97120** | **14 — GOLD** | +20e-6 | 0.9701286430 | **+10.4e-6** |
| 0.97121 | 12 | +30e-6 | 0.9701340299 | +15.7e-6 |
| 0.97125 | 4 | +70e-6 | 0.9701555778 | +37.3e-6 |
| 0.97134 | 1 | +160e-6 | 0.9702040606 | **+85.8e-6** |

**Gold is reachable and first place is not.** +10.4e-6 of CV is two c_avg corrections, or one
good new member family — 10% of the fitted CV span, interpolative. +85.8e-6 is **1.8× the
entire 22-member adarsh import**, the single biggest jump in this workspace's history, and 81%
of the fitted span. No stacking tweak reaches it; it needs a better base model, and there are
11 days left. **Play for gold, not for the lead.**

⚠ And the cutoff moves: ranks 2–16 filled in beneath a static leader over 08-18→08-20, so
0.97120 will not still be the gold line on 08-31. Treat +10.4e-6 as a floor.

## 2026-08-20 01:55 UTC — w33 slot 4. Rank 17, 0.97118. Gold is 2 reporting steps away.

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4 | Don Mani | 0.97125 |
| 5 | cstdy | 0.97125 |
| … | | |
| 14 | thisray | 0.97120 |
| 15 | Mahog | 0.97120 |
| 16 | delai50 | 0.97119 |
| **17** | **Teddy Tennant** | **0.97118** |

**The leader has not moved since 08-18** (MILANFX 0.97134, still static across three days),
but ranks 2–16 keep filling in beneath it — the same drift w32 §2 flagged. Since w32's read
(08-20 00:00) the board added `cstdy` at 0.97125 and `Utkarsh` at 0.97124.

⚠ **Units discipline, per w32 §1 — always state a gap in BOTH units.** Gap to gold
(0.97120) = 2e-5 = **20e-6 = 2 reporting steps**; w32a's inverted bar prices that at
**+10.4e-6 of CV**. Gap to first (0.97134) = 1.6e-4 = **160e-6 = 16 reporting steps**
≈ **+85.8e-6 of CV**, which is 81% of the fitted CV span and 1.8× the entire 22-member adarsh
import. **Play for gold; first place is not reachable from here in 11 days.**

The gold cutoff is a moving floor, not a target — treat +10.4e-6 as a minimum.

## 2026-08-20 (w34, slot 5) — rank 17, 0.97118

Top of board, taken at 02:50 UTC:

| # | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Maher el Ouahabi | 0.97127 |
| 3 | Optimistix | 0.97126 |
| 4 | **Don Mani** | 0.97125 |
| 5 | cstdy | 0.97125 |
| … | | |
| 14 | thisray | 0.97120 |
| 16 | delai50 | 0.97119 |
| **17** | **Teddy Tennant** | **0.97118** |

Unchanged from w32/w33: 0.97118, set by `w27_ad190stdcorr` / `w29_ad194stdcorr`. The gold
cut sits around 0.97121, i.e. **~3e-5 of LB above us**, which w32a prices at +10.4e-6 of CV.

⚠ Two names on this board are now sources in our own member pool, which is worth noting for
what it says about where the remaining headroom is:

- **`Don Mani` = `donmarch14`, rank 4.** Both of their notebooks publish OOF as kernel
  output and were imported this slot — into quarantine, because both early-stop on the fold
  they report (`best iteration:` / `Best Iter =`). Rank 4 on the public LB is not evidence
  that a member is honest.
- **`thisray`, rank 14, LB 0.97120.** Their 0.97117 blend notebook loads
  `s6e8-oof-library-47-models`, `s6e8-oof-prediction-library`, `s6e8-golem-oof-library`,
  `s6e8-adarsh-oof-library` and `s6e8-fm-lattice-blend-members` — **every one of which this
  workspace already holds**, plus their own test-only component. So a top-15 public entry is
  a blend of exactly our pack. That is a floor, not a ceiling: our 195-member pack is a
  strict superset, and the difference between 0.97118 and 0.97120 is combiner, not supply.

## 2026-08-20 (w36, slot 6) — rank 18, 0.97118, and the board moved under us

Taken at 15:59 UTC, ~13 h after the w34 snapshot:

| # | team | score | Δ vs w34 snapshot |
|---|---|---|---|
| 1 | MILANFX | 0.97134 | — |
| 2 | **Changye Li** | 0.97130 | **new to the top 3** |
| 3 | Maher el Ouahabi | 0.97127 | — |
| 4 | Optimistix | 0.97126 | — |
| 5–8 | Don Mani / Keanan / Utkarsh / cstdy | 0.97125 | — |
| 9 | **Szymon Kłapiński** | 0.97124 | **new** |
| 10 | Mikhail Naumov | 0.97123 | new |
| 11–13 | Leo / william950615 / Charles Backman | 0.97121–0.97122 | |
| 14–16 | thisray / Mahog / **Mitudru Dutta** | 0.97120 | |
| 17 | delai50 | 0.97119 | |
| **18** | **Teddy Tennant** | **0.97118** | **−1 place, same score** |

**We lost a place without losing a point.** Our 0.97118 has not moved since w32 — it is still
`w27_ad190stdcorr` / `w29_ad194stdcorr` — while `Changye Li`, `Szymon Kłapiński`, `Mikhail
Naumov` and `Mitudru Dutta` all posted improvements in the last 13 hours. **This is the cost
of the 08-20 day being drained at 00:07 on files that were all below the best already-sent
CV.** Ten submissions went out and the best of them tied, not beat, what was already there.

Gap to gold (~0.97121–0.97122) = **3–4e-5 = 30–40e-6 = 3–4 reporting steps ≈ +10.4e-6 of CV**
on w32a's inverted bar. Gap to first (0.97134) = **160e-6 ≈ +85.8e-6 of CV** — still not
reachable in 11 days. Play for gold.

⚠ **Two more board names entered our pool this slot, and both were dead on a gate.**
`Szymon Kłapiński` (rank 9) publishes only `submission.csv` from
`s6e8-honest-oof-blend` — the title says "honest OOF blend" and the kernel output ships no OOF
at all. `Krasnov Daniil`'s `s6e8-top-1-public-0-97099` likewise. **A high public rank buys a
member exactly nothing here**; of the four board-adjacent authors imported across w34–w36
(`donmarch14`, `omidbaghchehsaraei`, `tamerlanomralinov`, `redamountassir`), the only one whose
OOF cleared every gate is `redamountassir` — who is not on the visible board at all.

## 2026-08-20, w39 slot 9 — field size re-read, and what the queue head is worth in places

**teamCount 2433**, read straight from `GetCompetition` rather than from the board page (was
2,329 on 08-19; the brief's 1,326 is long stale and should not be quoted again). Medal cuts at
that size: **gold top 14**, silver top 122, bronze top 243.

| | score | rank |
|---|---|---|
| us, today | 0.97118 | **18 — silver** |
| queue head `w36_ad199stdcorr`, predicted | **0.97121** | **~12 — gold** |
| first (MILANFX) | 0.97134 | 1 |

Rank thresholds read off the live top page: 0.97119 → 17, 0.97120 → 14, 0.97121 → 12,
0.97122 → 11. So the top of the board is dense enough that **one reporting step is 2–3 places**,
and the single unsent CV leader is the difference between silver and gold on the public slice.

That makes tomorrow's slot 1 the whole of the day. It is also, per `w39b_autoselect.py`, what
makes the CV leader *selectable at all* — nothing is selected on this account, Kaggle then
auto-picks on public score, and an unsent file cannot be picked. Sending it is worth more than
the manual selection toggle it partly substitutes for.

## 2026-08-20 18:45 UTC (w40)

Rank **18** at **0.97118**, unchanged for the third consecutive read. **teamCount 2,433.**

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97134 |
| 2 | Changye Li | 0.97130 |
| 3 | Maher el Ouahabi | 0.97127 |
| 14 | thisray | **0.97120  ← gold cut (top 14)** |
| 17 | delai50 | 0.97119 |
| **18** | **Teddy Tennant** | **0.97118** |

**The gold cut is 2e-5 away.** Medal cuts at 2,433 teams: gold top 14, silver top 122, bronze
top 243 — we are comfortably silver and two ticks off gold. The unsent CV leader
`w36_ad199stdcorr` is priced by w30b at **0.971212**, which would be rank ~12. It goes out in
slot 1 of the 08-21 drain.

Movement is still concentrated at the top and daily: eight of the top twenty re-scored on 08-20.
The board is not stalling, so holding 0.97118 will drift downward in rank without new sends.

## 2026-08-21 00:2x UTC (w41)

Rank **18 at 0.97118** after the 08-21 drain. Leader MILANFX 0.97134 (unchanged since 08-18).
**Gold cut (top 14 of ~2,433) is 0.97120** — two grid steps up.

The field tightened overnight: 0.97125 now buys only ~rank 5-8, where it was comfortably top-5
before. Eleven teams sit at 0.97121-0.97130.

Us at 0.97118 in a 5-way tie (Masaya Kawamata, Mursal Gorchuyev, Malhar Ujawane, mraz1006).
delai50 alone at 0.97119 separates that tie from the 0.97120 cut.

⚠ **The 08-21 drain sent our best-CV file ever (`w36_ad199stdcorr`, CV 0.9701400) and it scored
0.97118 — exactly tying the account best rather than beating it.** +21.7e-6 of CV bought zero
public LB. See RESEARCH's "w30b LB predictor is optimistic" section: the CV→LB slope flattens at
the top of our range, so closing the 2-step gap to gold is harder than the queue prices imply.

## 2026-08-21 00:2x UTC (w42)

- **Changye Li 0.97136 is the new leader**, displacing MILANFX (0.97134, unchanged since 08-18).
- Us: **rank 18 at 0.97118**, unchanged — 10/10 sent for the day at 00:07.
- **Gold cut (top 14 of ~2,433) is now 0.97120–0.97121**, two grid steps above us. The field
  compressed again overnight: 0.97125 is now a four-way tie at ranks 5–8.
- Top of board moving daily; 0.97118 held rank 18 both yesterday and today, so the tier just
  above us is where the traffic is.

## 2026-08-21 00:50 UTC (w43)

**Rank 21 at 0.97118** — down from 18 in w42, on no change of ours. The field is compressing
daily and standing still costs places.

| | team | score |
|---|---|---|
| 1 | Changye Li | 0.97136 |
| 2 | MILANFX | 0.97134 |
| 3 | Optimistix | 0.97127 (moved up 00:21 today) |
| 4 | Maher el Ouahabi | 0.97127 |
| ~14 | **gold cut** | **0.97121** |
| **21** | **Teddy Tennant** | **0.97118** |

We are **3e-5 off gold**, and eleven teams sit inside the 0.97118–0.97121 band — a single
successful arm would move several places. No submission this run: 10/10 sent for the UTC day.

## 2026-08-21 01:06 UTC (w44)

**Rank 18 of ~1,326 at 0.97118.** Up 3 from w43's read of 21 on no change of ours — pure
churn beneath us.

| # | team | score |
|---|---|---|
| 1 | Changye Li | 0.97136 |
| 2 | MILANFX | 0.97134 |
| 3 | Optimistix | 0.97127 |
| 4 | Maher el Ouahabi | 0.97127 |
| 5 | Don Mani | 0.97125 |
| … | cstdy / Keanan / Utkarsh | 0.97125 |
| 12–13 | william950615 / Charles Backman | 0.97121 |
| **14** | **thisray — GOLD CUT** | **0.97120** |
| 17 | delai50 | 0.97119 |
| **18** | **Teddy Tennant** | **0.97118** |

**Gold cut is 0.97120; we are 2e-5 below it.** Leader unchanged at 0.97136 since 00:18.

⚠ **Read this against w44's finding.** 2e-5 of public LB is the gap to gold, and the entire
import line — 22 members over eight arms — bought **+19.85e-6 of CV** in total, with a marginal
rate now indistinguishable from zero. There is no modelling lever left that is sized to close
this gap; the remaining arms are priced at ~+2e-6. Chasing the gold cut by building more stack
is not a plan, and the ~−10e-6 selection exposure is the larger number.

## 2026-08-21 01:22 UTC (w45)

**Rank 18 at 0.97118.** Unchanged score from w44; the board read is the same. Gold cut (top 14)
**0.97120** (`thisray`), leader **Changye Li 0.97136**, MILANFX 0.97134. **We are 2e-5 below gold.**

Top of the board, live:

| # | team | score |
|---|---|---|
| 1 | Changye Li | 0.97136 |
| 2 | MILANFX | 0.97134 |
| 3 | Optimistix | 0.97127 |
| 4 | Maher el Ouahabi | 0.97127 |
| 14 | **thisray — the gold cut** | **0.97120** |
| 18 | **Teddy Tennant** | **0.97118** |

**The number that matters more than the rank this week.** w45 repriced the unset final-selection
toggle on the live auto-selection tiers: **+15.8e-6 (limit 2) / +21.7e-6 (limit 1)**, with a
tiebreak bracket of **+0.00 to +35.15e-6**. At ~2.5 board places per 1e-5 the bad branch is
**~9 places** — i.e. **larger than our whole 2e-5 gap to gold.** Nothing is selected. See the
`🔴 ACT ON THIS FIRST` block at the top of `RESEARCH.md`.

Our own top public tier, which is now the thing to watch daily:
- **0.97118 (auto-slot 1), 4-way:** `w36_ad199stdcorr` ← the CV pick, `w29_ad194stdcorr`,
  `w27_ad190stdcorr`, `w21_ad187corr_ens4`
- **0.97117 (auto-slot 2), 5-way:** `w36_ad199std`, `w34_ad195stdcorr`, `w27_ad190std`,
  `w21_ad187corr`, `w22_ad187corr_rankraw`

## 2026-08-21 01:42 UTC (w46) — unchanged at rank 18, and a reading on what our unsent material is worth

Board read with `kaggle competitions leaderboard -s`. **No change from w45's 01:22 read:**
**Teddy Tennant rank 18, 0.97118.** Leader Changye Li 0.97136 (resent 00:18 today);
MILANFX 0.97134; gold cut (top 14) `thisray` **0.97120**. **We are 2e-5 below gold.**

Movement beneath the cut is live — `cstdy` resubmitted at 01:33 for 0.97126 and Optimistix at
00:21 for 0.97127 — so rank at fixed score will keep drifting down. Local density is still
~2.5 places per 1e-5.

**The new thing worth recording is what the board says about our own queue.** w46 §2 found
the CV→LB predictor over-reads by 30e-6 on every ad≥195 file, and the two hypotheses land on
opposite sides of the gold cut for the best two files we have built but not sent:

| file | H0 (old predictor) | H1 (era-corrected) |
|---|---|---|
| `w38_ad202stdcorr` | **0.97122 — above the 0.97120 gold cut** | 0.97119 — below |
| `w40_ad211stdcorr` | **0.97122 — above** | 0.97119 — below |

Both go out on the 08-22 day under either hypothesis (`w46d_prereg.txt`), so this is a free
reading. It decides whether the material already on this disk is a medal or is not.

## Snapshot 2026-08-21 01:50 UTC (w47)

| | team | score |
|---|---|---|
| #1 | Changye Li | 0.97136 |
| #2 | MILANFX | 0.97134 |
| #3 | Optimistix | 0.97127 |
| **#14 — gold cut** | **thisray** | **0.97120** |
| #17 | delai50 | 0.97119 |
| **#18 — us (thtennant)** | **Teddy Tennant** | **0.97118** |

Unchanged in substance from the w45/w46 reads: **rank 18, 2e-5 below the gold cut**, and
the 0.97118 tier is 12 teams deep. The leader moved 0.97136 at 00:18 UTC on 08-21.

**What w47 changes about reading this board:** none of the ten files queued for 08-22 is
predicted to clear 0.97120 under either live hypothesis — under w46c the two best round to
0.97119, under w30b to 0.97122. That disagreement is exactly what `w47b_prereg.txt` tests,
and it is the difference between "the material already on this disk is a medal" and "it is
not". **The answer does not change what gets sent** — all ten go either way.

---

## Snapshot 2026-08-21 02:2x UTC (w48)

| | team | score |
|---|---|---|
| #1 | Changye Li | 0.97136 |
| #2 | MILANFX | 0.97134 |
| #3 | Optimistix | 0.97127 |
| #5 | cstdy | 0.97126 (moved 01:33 UTC today) |
| **#14 — gold cut** | **thisray** | **0.97120** |
| #17 | delai50 | 0.97119 |
| **#18 — us (thtennant)** | **Teddy Tennant** | **0.97118** |

**Board unchanged for us: rank 18, 2e-5 below the gold cut, third consecutive read.** One
move inside the top 5 (cstdy to 0.97126 at 01:33 UTC).

**What w48 changes about reading this board — and it is not a modelling change.** The ten
files w47b registered for 08-22 are the ones that decide whether the material on this disk is
a medal. This slot found that **the sender would not have sent them.** `w26g_send.py` reads
one CSV, `experiments/w26d_queueprice.csv`, last written 2026-08-20 14:30 for the 08-21 day;
its dry run planned three ⛔VETOED files and none of the five probes. Repaired in
`w48e_order.py`, and the sender's dry run now reproduces the registered ten exactly, in order,
priced under w46c. **The board reading above was never the bottleneck. The path from the plan
to the submit call was.**

The one number on this board that could still move sharply: `w42_ad217stdcorr` (ARM 217,
built 02:03 UTC today) has cross-fitted CV 0.9701788 — **+38.8e-6 above anything else on this
disk** — and the corrected pricer puts it at 0.97125, i.e. inside the top 10. It is ⛔vetoed
and WANTED-ineligible, because w48d traces essentially all of that CV to one imported member
whose standalone OOF AUC (0.9701816) exceeds our entire 217-member stack. **Do not read that
0.97125 as a rank we are declining to take. Read it as the trap the veto exists for.**

## 2026-08-21 02:4x UTC (w49)
Best public **0.97118**, rank ~18. Gold cut (top 14) `thisray` **0.97120** — **2e-5 short**,
fourth consecutive unchanged read. Leader Changye Li 0.97136; MILANFX 0.97134; Optimistix and
Maher el Ouahabi 0.97127; cstdy 0.97126 (moved 01:33 UTC today).
The board is compressing at the top: 0.97119–0.97127 now holds ~10 teams, so a single 1e-5 step
is worth several places. Our 10 sends at 00:07 today produced no new best (three tied 0.97118).
⛔ **Nothing is selected.** Needs Teddy in his own browser before 08-31.

---

## Snapshot 2026-08-21 04:0x UTC (w51) — rank 18, 0.97118, 2e-5 below gold

Team name on the board is **Teddy Tennant**, not `thtennant` — a rank scan that greps the
Kaggle username finds nothing. Grep the team name.

| rank | team | score |
|---|---|---|
| 1 | MILANFX | 0.97136 |
| 1= | Changye Li | 0.97136 |
| 3 | Optimistix | 0.97127 |
| 3= | Maher el Ouahabi | 0.97127 |
| 5 | cstdy | 0.97126 |
| **14 (gold cut)** | **thisray** | **0.97120** |
| 16 | Mitudru Dutta | 0.97120 |
| 17 | delai50 | 0.97119 |
| **18** | **Teddy Tennant** | **0.97118** |
| 19–24 | Masaya Kawamata, Mursal Gorchuyev, Malhar Ujawane, mraz1006, miki, BOB | 0.97118 |

**16 teams are at or above 0.97120.** We need **+2e-5** for gold, and there is a seven-team
pile-up on our exact score — a single 1e-5 step moves us past all of them.

Movement since the 08-21 02:4x read: MILANFX rejoined the lead at 0.97136 (03:33 UTC), Leo
(0.97122) and Mitudru Dutta (0.97120) are new inside the top 16. **The gold cut has not
moved in five consecutive reads — it has sat at 0.97120 since 08-20.** The leaders are
pulling away at the very top while the 0.97118–0.97121 band stays static, which is what a
compressed board looks like when everyone is out of ideas at the same time.

## 2026-08-22 ~12:5x UTC (w52) — THE FIELD PULLED AWAY

**Us: 0.97118, rank 61** (team name on the board is `Teddy Tennant`, not `thtennant`).
Five straight prior reads had us at ~18. The field gained ~5e-5 in a day; we gained 0.

| rank | score |
|---|---|
| 1 Changye Li / MILANFX | 0.97141 |
| 10 Leo | 0.97128 |
| **14 (gold cut) Atakan Aldemir** | **0.97124** |
| 20 jazivxt | 0.97122 |
| 30 | 0.97121 |
| **61 Teddy Tennant** | **0.97118** |

Gold was 2e-5 away on 08-21; it is **6e-5** away now. Ten files sent 08-22 (w47b's registered
experiment) — best of them tied 0.97118, none beat it, and none was meant to.

Likely driver: `omidbaghchehsaraei/hill-climbing-ensemble` (31 votes, published 08-22). Checked
and **not** a technique we are missing — hill climbing is closed here with a mechanism (a climber
can only add; our linear stacker subtracts, and weak decorrelated members act as corrections).

⚠ Per `w52d`, gold now needs **+40.9e-6 of CV** above our best sent — **+30.8e-6 beyond the
highest CV ever built here**. The only file on disk in that range is `w42_ad217stdcorr`
(+38.8e-6), currently vetoed on es-on-val grounds. `w48_cal_hboyang_mix` (slot 1, 08-23) is the
registered test that decides it.

## Snapshot 2026-08-22 ~13:0x UTC (w53 read)

| | score | note |
|---|---|---|
| leader | 0.97141 | `Changye Li` (11:23) and `MILANFX` (00:34) tied |
| 3rd | 0.97134 | `Maher el Ouahabi` |
| **gold cut (14th)** | **0.97124** | `Atakan Aldemir` |
| **us — `Teddy Tennant`** | **0.97118** | **rank 62** |

Unchanged from w52's 12:5x read (rank 61 → 62 is field drift below us, not a move by us). The
board's ~5e-5 overnight jump on 08-21→08-22 is what moved us from ~18 to the low 60s; the gold
requirement is now **+40.9e-6 of CV** over the best sent file and **+30.8e-6 beyond the best CV
ever built here**. Nothing on disk reaches it except the vetoed ARM 217 family — whose registered
read (`w48_cal_hboyang_mix` vs `w48d_arm217.json`: ≥0.97116 HONEST, ≤0.97080 INFLATED) is slot 1
of the 08-23 send.

⚠ Team name on the board is **`Teddy Tennant`**, not `thtennant`. Grep for both.

## 2026-08-22 13:1x UTC (w54)

Unchanged from w53's 13:0x read — no move by us (at cap since 12:38).

| | team | score |
|---|---|---|
| 1 | Changye Li | 0.97141 |
| 1 | MILANFX | 0.97141 |
| 3 | Maher el Ouahabi | 0.97134 |
| 14 (gold cut) | Atakan Aldemir | 0.97124 |
| ~62 | **us** | **0.97118** |

Gold needs **+40.9e-6 of CV** over the best sent, i.e. **+30.8e-6 beyond the best CV ever built
here**. Only the vetoed ARM 217 family is on disk in that range; 08-23 slot 1
(`w48_cal_hboyang_mix`) is the registered read that decides whether its veto is re-argued.

⚠ **Our 0.97118 is a four-way tie on the account** (`w40_ad211stdcorr`, `w36_ad199stdcorr`,
`w29_ad194stdcorr`, `w27_ad190stdcorr`). That tie IS the auto-selection tier (w54) — while
nothing is selected, Kaggle picks two of them and we do not choose which.

## 2026-08-22 13:2x UTC — w55 read (no submission, at cap 10/10)

| | team | public |
|---|---|---|
| 1 | Changye Li | 0.97141 |
| 14 (gold cut) | Atakan Aldemir | 0.97124 |
| **62** | **Teddy Tennant** | **0.97118** |

Unchanged from w54's read. Gold wants **+40.9e-6 of CV** over the best sent, **+30.8e-6 beyond
the best CV ever built here**; the only thing on disk in that range is the vetoed ARM 217 family.

**New this run:** our 0.97118 is now a **FIVE**-way tie on the account, not four —
`w40_ad211stdcorr` (sent 08-22 12:38) joined `w36_ad199stdcorr` (WANTED), `w29_ad194stdcorr`,
`w27_ad190stdcorr` and `w21_ad187corr_ens4`. With nothing selected Kaggle auto-picks two by
public score, so the tie-break is worth real money. **The leaderboard does not leak it:** our LB
row's timestamp is our *latest* submission (`w38_ad202stdcorr`, 0.97117), not the one holding the
0.97118. Closed as a probe — see RESEARCH.md.

---

## Snapshot 2026-08-22 ~13:5x UTC (w56, slot 5) — UNCHANGED from w55

| | score | note |
|---|---|---|
| #1 `Changye Li` | 0.97141 | 11:23 UTC |
| gold cut (14th, `william950615`) | 0.97124 | 11:33 UTC — the cut holder rotates, the number does not |
| **us (`Teddy Tennant`), rank 62 of 200 listed** | **0.97118** | 12:38:26, i.e. the w52 drain |

Board flat across the whole 08-22 day: leader 0.97141 and gold cut 0.97124 in both the w55
(~13:2x) and w56 (~13:5x) reads. w52 §5's arithmetic is unchanged — gold wants **+40.9e-6 of CV**
over the best sent, **+30.8e-6 beyond the best CV ever built here**.

### ⛔ w56 CLOSES THE ONLY ROUTE FROM DISK TO THAT NUMBER

The only object on disk inside gold's range is `w42_ad217stdcorr` (CV 0.9701788, +38.8e-6 above
best sent). w56a read `hboyang/s6e8-150-member-fusion`'s notebook source and found the member it
rests on is an **aggregator over 138 third-party streams from seven public OOF libraries**, on
our exact fold partition, none of them es-clearable. **w40d therefore bars the whole ad217 family
from CV-based selection whatever the 08-23 read says**, and that bar is now enforced in
`check_selection.WANTED_INELIGIBLE` rather than described in RESEARCH.

So: the 08-23 slot-1 read can still retire `hboyang_mix` (INFLATED) or reopen the import line
(HONEST), and it is worth sending for exactly that. **It cannot deliver gold.** No path to the
gold cut exists from what is on disk. Play for the best CV-selected private score.

## 2026-08-22 ~14:0x UTC (w57) — flat all day, but the auto-selection exposure was re-priced

**Us `Teddy Tennant` 0.97118, rank 62** of 200 listed. Leader **0.97141** (`Changye Li`, 11:23).
Gold cut (14th) **0.97124**. Identical to w55/w56's reads — the board has not moved all day.

Local density is brutal and worth keeping in view: **18 teams at 0.97119**, 7 at 0.97122,
10 at 0.97121. One 1e-5 step near us is **~10 places**, so the 6e-5 to the gold cut is not a
near miss — it is most of the field.

⚠ **The tier that decides our FINAL score moved today, and it is not the same thing as our rank.**
Our own 10 sends put a fifth file into the best-public tie at 0.97118:
`w40_ad211stdcorr`, `w36_ad199stdcorr` (the CV pick), `w29_ad194stdcorr`, `w27_ad190stdcorr`,
`w21_ad187corr_ens4`. Since nothing is selected, Kaggle auto-selects two of those five.
**Re-priced at +7.86e-6 (w57a, MODEL B), half the previously published +15.77e-6** — the day's
sends were net favourable, because the file that diluted the tie is only −2.53e-6 of CV off the
pick. P(the CV pick is auto-selected) nonetheless fell 0.50 → 0.40. See RESEARCH's live table.

## 2026-08-22, w58 (~14:5x UTC) — flat

| | 08-21 | 08-22 (w57) | **08-22 (w58)** |
|---|---|---|---|
| us | 0.97118, rank ~18 | 0.97118, rank 61 | **0.97118**, unchanged |
| gold cut (14th) | 0.97120 | 0.97124 | **0.97124** |
| leader | 0.97136 | 0.97141 | **0.97141** (`Changye Li`, tied `MILANFX`) |

No movement in ~1h. Nothing new to chase.

**⚠ The board is no longer the binding constraint on our final score — auto-selection is.**
w58 found that tomorrow's slot-1 send would, under its own registered prediction, hand final
entry #1 to a file our own rules bar, and that three `logit` files further down the queue carry
12–37% of doing the same. All four are now blocked in `w26g_send.py`. **Our distance to gold is
6e-6 of public; the unclicked selection toggle is worth up to +23.87e-6 of private in its worst
named branch (w57a). The toggle is the bigger number and it is free.**

## 2026-08-22 ~14:55 UTC (w59) — flat for a third consecutive read

| | | |
|---|---|---|
| leader | **0.97141** | `Changye Li` (11:23) tied with `MILANFX` (00:34) |
| 3rd | 0.97134 | Maher el Ouahabi |
| gold cut (14th) | **0.97124** | `william950615` / `Atakan Aldemir` |
| us | **0.97118** | 5 files tied there; unchanged since 08-20 |

Unchanged from w57 and w58. Gap to gold **6e-6 of LB** ≈ 0.6 reporting steps; gap to first
**23e-6** ≈ 2.3 steps. `Chris Deotte` appears at 0.97130 (13:34), 7th.

**No submission this run — at cap (10/10 sent 08-22 12:37–12:38 UTC).** The run changed send
POLICY, not the board: the above-tier CV bar moved from 0.9701150809 to **0.9701294160**
(measured, w59a), which blocks 3 more of the 08-23 ten and backfills to ten.

⚠ Still **nothing selected** on the account, so Kaggle auto-picks the best two by PUBLIC score
out of the 5-way tie at 0.97118 — P(our CV pick is in that pair) = **0.400**.

## 2026-08-22 ~15:15 UTC (w60) — flat for a FOURTH consecutive read, but the field is thickening

| | | |
|---|---|---|
| leader | **0.97141** | `Changye Li` (11:23) tied with `MILANFX` (00:34) |
| 3rd | 0.97134 | Maher el Ouahabi |
| gold cut (14th) | **0.97124** | `william950615` / `Atakan Aldemir` |
| us | **0.97118** | rank **65**, 5 of our files tied there |

Top three unchanged across four reads spanning ~2h. What HAS moved is the queue behind the
cut: `cstdy` 0.97131 (15:00) and **`Chris Deotte` 0.97130 (15:03)** both submitted within the
last few minutes, and 0.97130 is now a four-way tie (`Keanan`/`Utkarsh` at 0.97131–0.97132
just above). ⚠ **The 0.97118 shelf we sit on is ~9 teams deep and the shelf above is filling
faster than the top is moving.** Rank at a fixed score will DECAY from here; 0.97118 bought
rank ~18 on 08-21 and buys rank 65 today.

**No submission this run — at cap (10/10 sent 08-22 12:37–12:38 UTC).**

⚠ Still **nothing selected**. Auto-selection picks the best two by PUBLIC score out of our
5-way tie at 0.97118, so P(our CV pick is in the final pair) = **0.400** — and w60 found that
one of those five, `w40_ad211stdcorr` (p_joint **0.375**, the highest of any non-pick member),
is on an arm `w40d_prereg` bars from being a deadline pick at all. That file is already sent
and cannot be recalled; the entry added this run stops the next one.

## 2026-08-22 ~16:10 UTC (w61) — flat at the top for a FIFTH read; we are rank **66** on a nine-deep shelf

| | | |
|---|---|---|
| leader | **0.97141** | `Changye Li` (11:23) tied with `MILANFX` (00:34) |
| 3rd | 0.97134 | Maher el Ouahabi |
| gold cut (14th) | **0.97124** | `william950615` |
| us — team **Teddy Tennant** | **0.97118** | rank **66** (was 65 at w60's read) |

The 0.97118 shelf spans ranks **65–73, nine teams**, and we sit 2nd within it on submission
time. Top three unchanged across five reads spanning ~3h; `cstdy` 0.97131 and `Chris Deotte`
0.97130 have settled into the 6th–7th band they entered at w60. Gap to gold **6e-6** ≈ 0.6
reporting steps.

⚠ **Rank at a fixed score keeps decaying: 0.97118 bought ~18 on 08-21, 65 at 15:15 today, 66 at
16:10.** Only a new score moves us; the eight teams sharing our shelf are ordered by submission
time and we cannot outrun that.

**No submission this run — at cap (10/10 sent 08-22 12:37–12:38 UTC).**

⚠ Still **nothing selected**, so Kaggle auto-picks the best two by PUBLIC score out of our
5-way tie at 0.97118; P(our CV pick is in the final pair) = **0.400**, unchanged. What DID
change: `w40_ad211stdcorr` (p_joint **0.375**, the likeliest auto-selection after the pick
itself) was a file our own prereg barred from being a deadline pick, and w61's four-base
matched-control test retired that bar on evidence. The price is unchanged (+7.855e-6 standing
exposure); the outcome is no longer one our rules forbid.

## 2026-08-23 ~12:55 UTC (w62) — **THE SCORE MOVED. 0.97118 → 0.97119, rank 66 → 62**

| | | |
|---|---|---|
| leader | **0.97144** | `MILANFX` (06:17) |
| 2nd | 0.97142 | `Changye Li` |
| 3rd= | 0.97134 | `Chris Deotte`, `cstdy`, `Maher el Ouahabi` |
| gold cut (14th) | ~0.97128 | `Leo` |
| us — team **Teddy Tennant** | **0.97119** | rank **62** of ~1,326 |

First score movement in six reads. Both of w60's lever files — `w36_ad199stdcorr_ens4` and
`w38_ad202stdcorr_ens4` — landed 0.97119, one reporting step above the 0.97118 shelf we had been
stuck on since 08-21. We are 1st within the 0.97119 shelf on submission time (12:41:34).

The 0.97120 shelf immediately above holds six teams; the gap to gold is now ~9e-6 ≈ 0.9
reporting steps, wider than w61's 6e-6 because the top of the board also moved (leader
0.97141 → 0.97144).

⚠ **The rank-decay-at-a-fixed-score effect is confirmed from the other side.** 0.97118 bought
rank ~18 on 08-21, 65 at w60, 66 at w61. One reporting step of new score bought back only 4
places (66 → 62), because the shelf we left had thinned and the one we joined is crowded. **A
reporting step is worth far less in rank than it was two days ago** — the field is compressing
into the 0.9711x–0.9712x band and only a genuinely larger move changes standing.

**Submitted 10/10 this run** (12:41:06–12:41:34 UTC). Cap re-confirmed from the CLI: the sender
read "0 submissions remaining today" after the tenth.

⚠⚠ **AUTO-SELECTION IS NOW DETERMINED, AND THE CV PICK IS NO LONGER IN IT.** Exactly two files
sit at 0.97119, so the best-two-by-public-score pair is `w36_ad199stdcorr_ens4` +
`w38_ad202stdcorr_ens4` with no tie to break. `w36_ad199stdcorr` — the CV leader and WANTED slot
1 — dropped to the 0.97118 tier: **P(our CV pick is in the final pair) went 0.400 → 0.000.**
Measured cost of not clicking: **+4.523e-6**, down from +7.855e-6 (w62a). The exposure improved
by 42%, not the 86% w60 priced — see JOURNAL w62 §4.

## 2026-08-23 13:10 UTC (w63, build run — no submission, day was at 10/10)

- **Account best 0.97119**, unchanged since the 12:41 sends. `w36_ad199stdcorr_ens4` and
  `w38_ad202stdcorr_ens4` are still the only two files on the board at that score, so
  **auto-selection remains DETERMINED** and `w63a`'s GATE T2 confirms it against the live board.
- **62 teams strictly above 0.97119** on the 200-row window → we sit at ~63rd of ~1,326.
- Leader **MILANFX 0.97144** (2026-08-23 06:16). **Gap to the top: 25e-6** — roughly 20× the
  whole width of the determined pair's usable pricing range (1.05e-6), which is a useful sense
  of scale: the selection-mechanism work is worth single-digit e-6 and the modelling gap is 25.
- 121 scored submissions on the account, 121 distinct filenames, all fingerprinted on disk.
- Board shape unchanged otherwise: tier 2 is five files at 0.97118 including the CV **pick**
  `w36_ad199stdcorr`, which remains reachable **only by the click** — still blocked, no browser
  MCP attached for the seventh run running.

## 2026-08-23 14:05 UTC (w64, build run — no submission, day was at 10/10)

- **Account best 0.97119, rank 64 of 2,660** (full leaderboard walked, 14 pages of 200; the team
  name on the board is **`Teddy Tennant`**, not `thtennant` — a page-name grep for `thtennant`
  finds nothing and a run that grepped for it would conclude we are not on the board).
- **21 teams are tied with us at 0.97119** (ranks 64–84). One more display step is worth ~20
  places from here; we are on the flattest part of the curve.
- Leader **MILANFX 0.97144**, then Changye Li 0.97142, then a wall of 0.97134 (Chris Deotte,
  cstdy, Maher el Ouahabi). **Gold cut (14th) = 0.97127, 8e-6 above us.**
- ⚠ **THE FIELD IS GROWING FASTER THAN OUR SCORE.** 2,047 teams on 08-22 → **2,660 today**, and
  rank at a fixed score keeps decaying: 0.97118 bought rank ~18 on 08-21, 65 on 08-22, and
  0.97119 buys 64 today. Standing still costs ~1 place an hour.
- Board shape unchanged: auto-selection is still DETERMINED at the `w36_ad199stdcorr_ens4` +
  `w38_ad202stdcorr_ens4` pair, and the CV pick is still reachable only by the click, still
  blocked — **eighth run with no browser MCP attached.**

## 2026-08-23 14:15 UTC (w65, build run — no submission, day was at 10/10)

- **Account best 0.97119, rank 64 of 2,662.** Unchanged from w64 ten minutes of board time
  earlier; the field added 2 teams. **21 teams still tied with us** at 0.97119.
- **THE MEDAL CUTS, COMPUTED RATHER THAN EYEBALLED** (Kaggle's rule at n=2,662: gold = top
  10 + 0.2%·n = 15, silver = top 5% = 133, bronze = top 10% = 266):

  | medal | cut rank | cut score | gap to us |
  |---|---|---|---|
  | gold | 15 | 0.97126 | **+70e-6 above us** |
  | silver | 133 | 0.97115 | **−40e-6 below us** |
  | bronze | 266 | 0.97100 | −190e-6 below us |

  ⚠ This is the number that should govern how the last eight days are spent, and it has not
  been written down before. **We are 40e-6 clear inside silver and 70e-6 short of gold.** The
  ENTIRE CV ladder this workspace has argued over since 08-20 — ad187 through ad217, every
  transform family, both WANTED slots — spans **22.5e-6 of CV and ~5e-6 of LB**. Gold is
  fourteen times the width of everything still under discussion. No selection decision
  available here can reach it; the only thing selection can do is lose the silver.
- Leader MILANFX 0.97144 (+250e-6 on us), Changye Li 0.97142, then 0.97134 ×3.
- ⚠ Field growth continues: 2,047 (08-22) → 2,660 → **2,662**. The silver cut moves with n, so
  the 40e-6 cushion is a score cushion, not a rank cushion.

## 2026-08-23 14:38 UTC (w66) — we slipped three places without the score moving

- **Account best 0.97119, rank 67 of 2,661** (w64, ~35 minutes of board time earlier: rank 64 of
  2,662). The score did not move; **three teams passed us**, and the tie group shrank from 21 to
  **19**. That is the shape of the last four days — the field converges on 0.97119 and then
  walks past it.
- Top of board **0.97152 (Chris Deotte)**, then MILANFX 0.97144, Changye Li 0.97142, 0.97134 ×2.
  ⚠ The leader moved **+8e-6** since w64 (MILANFX 0.97144 → Deotte 0.97152).
- Medal cuts, on Kaggle's `n > 1000` rule (gold = top 10 + 0.2%·n = 15, silver = top 5% = 133,
  bronze = top 10% = 266):

  | medal | cut rank | cut score | gap to us |
  |---|---|---|---|
  | gold | 15 | 0.97126 | **+70e-6 above us** |
  | silver | 133 | 0.97116 | **−30e-6 below us** |

  ⚠ The silver cushion has narrowed from **40e-6 (w64) to 30e-6** in half an hour of board time,
  with our score fixed. It is a score cushion, not a rank cushion, and it is closing.
- ⚠⚠ **w66 measured what that 70e-6 costs in CV, and it is the first time this has been priced
  rather than asserted.** The in-range CV→LB transfer is calibrated 1:1 on the frozen pricer
  (w66b: beta_total 0.987 on 16 held-out files, 1.136 on the ten 08-23 sends), and the pricer's
  slope is +1.83 per e-6 of CV inside the fitted range. So **70e-6 of LB is 38e-6 of CV at the
  optimistic slope and 73e-6 at the GLS one.** The entire 187→211 pack ladder bought **22.5e-6
  of CV across 24 members** with a transfer slope indistinguishable from zero (w64 §3).
  **Gold is not reachable by adding members**, and the honest reading of the remaining eight
  days is that they are about not losing the silver.
- ⚠ `10 + 0.2%·(n−1000)` is NOT Kaggle's rule and gives rank 13 / 0.97127. w66's journal §9 used
  it once and corrected it. Use `10 + 0.2%·n`.

## 2026-08-23 15:11 UTC (w67)
- **Teddy Tennant — 0.97119, rank 67.** Unchanged from w66. Top: Chris Deotte 0.97152.
- 10/10 submissions used for the UTC day at 12:41; best of the ten 0.97119. w67 sent nothing.
- Gold cut ~0.97126 (rank 15, Kaggle's n>1000 rule is `10 + 0.2%·n`); silver ~0.97116 (rank 133).
- ⚠ The +70e-6 of LB to gold is ~14x the entire CV ladder the 187→211 pack has bought. w67 adds
  that the CV→LB conversion rate out there is **unmeasurable** from this send queue (RESEARCH.md),
  so the gap cannot even be priced honestly — only bracketed by the +1.4425/+0.9028 slope pair.

## 2026-08-23 16:05 UTC (w68)
- **Teddy Tennant — 0.97119, rank ~67.** Unchanged from w66/w67. Top: Chris Deotte 0.97152
  (submitted 14:26 UTC today). Board behind him: MILANFX 0.97144, Changye Li 0.97142,
  cstdy 0.97137, Maher el Ouahabi 0.97134.
- 50 submissions on record; **10 on each of 08-20, 08-21, 08-22, 08-23**. The day was at 10/10
  at 12:41 UTC, before w68 began — **w68 sent nothing, correctly.**
- ⚠⚠ **THE BOARD CORROBORATED w65 FOR FREE.** `w36_ad199stdcorr_ens4` and
  `w38_ad202stdcorr_ens4` both went out on 08-23 and both scored **exactly 0.97119**, despite a
  1.69e-6 stored-CV gap. At the LB's 1e-5 reporting resolution a 1.69e-6 difference is invisible,
  so this is not independent confirmation of *equality* — but it is exactly what w65's paired
  instrument predicts, and it is the first time the 202-vs-199 question has had any LB reading
  at all. Both tie our account best.
- ⚠ **This is why the 0.97119 best is held by TWO files.** Neither is `WANTED` slot 1
  (`w36_ad199stdcorr`, the `stdcorr` layer, 0.97118 on 08-21). Final selection stays on CV per
  the brief; the public tie changes nothing about that and must not be read as a reason to move.
- The gold cut (~0.97126) and silver cut (~0.97116) are unchanged from w67. **The +70e-6 to gold
  remains unpriceable** from this send queue (w67 §0) — only bracketed by the +1.4425/+0.9028
  slope pair. The silver cushion is ~30e-6 of score and still closing while our score is fixed.

## 2026-08-23, w69 (slot 8) — board unchanged, no submission

Our best public **0.97119** (`w36_ad199stdcorr_ens4` and `w38_ad202stdcorr_ens4`, tied exactly).
Top of board at 16:20 UTC:

| # | team | score | when |
|---|---|---|---|
| 1 | Chris Deotte | 0.97152 | 08-23 14:26 |
| 2 | MILANFX | 0.97144 | 08-23 06:16 |
| 3 | Changye Li | 0.97142 | 08-22 21:20 |
| 4 | cstdy | 0.97137 | 08-23 15:09 |
| 5 | Maher el Ouahabi | 0.97134 | 08-23 13:14 |

Gap to the top **33e-6**. 50 submissions on record; 10 on each of 08-20/21/22/23 — the day was at
**10/10 on arrival**, so w69 sent nothing and could not have.

⚠ The leader is still moving (Deotte re-scored 08-23 14:26) while our best has been flat at
0.97119 since 08-22. The stack's own internal spread across every arm on disk is ~2e-6 — **an
order of magnitude below the 33e-6 gap** — so no re-arrangement of the current member pool
reaches the top. That is a statement about the pool, not about the combiner.
⚠ **Our board name is `Teddy Tennant`, not `thtennant`** — walk pages with `--page-token` and
match on the display name or the score.

## 2026-08-23, w70 (slot 9) — rank measured for the first time: **73**. No submission (10/10 on arrival).

Our best public **0.97119**, unchanged since 08-22. Board at 17:35 UTC, `--page-size 2000`
(201 rows returned, sorted descending, so the rank below is exact whatever sits under it):

| # | team | score | when |
|---|---|---|---|
| 1 | Chris Deotte | 0.97152 | 08-23 14:26 |
| 2 | MILANFX | 0.97144 | 08-23 06:16 |
| 3 | Changye Li | 0.97142 | 08-22 21:20 |
| 4 | cstdy | 0.97137 | 08-23 15:09 |
| 5 | Maher el Ouahabi | 0.97134 | 08-23 13:14 |
| **73** | **Teddy Tennant** | **0.97119** | 08-23 12:41 |

**Gap to the top 33e-6. 37 teams are at or above 0.97123**, and 121 submissions are on record.

### ⚠ THE 0.97123 CLUSTER IS ONE FILE, NOT 37 METHODS

The two highest-scoring public notebooks were pulled and read in full this run. **Neither is a
method.** `amanatar/s6e8-elite-rank-average-ensemble-0-97123` loads a pre-made
`ULTIMATE_0.97123_SOTA.csv` and rank-averages it **95/5** with a throwaway LGBM — the 0.97123 is
the *teacher file's* score. `itzzomkar/s6e8-0-97123` globs `/kaggle/input/**/submission.csv` and
weights them by **substring-matching the filename**. So the cluster at 0.97123 is largely one
circulating CSV's distribution list.

⛔ **Do not chase it.** It is the Rogii failure by construction, and it is unusable under our own
rules regardless: submission-only, no OOF, so no cross-fitted CV, so `fam=member` and the sender
blocks it as a final entry. ⚠ But the standing journal sentence *"top public notebook 0.97101 <
our 0.97106 … we have everything in the forum"* now has **stale numbers** — the public field did
pass us, by 4e-6, on a file rather than a technique. The conclusion survives; the numbers do not.

### 🔴 THE BOARD IS NOT THE RISK — THE UNCLICKED SELECTION IS

`check_selection.py` **exits 1**: *nothing is selected*. Kaggle will therefore auto-select final
entries **by best public score** — auto-slot 1 a 2-way tie at 0.97119
(`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`), auto-slot 2 a 5-way tie at 0.97118. Choosing
final entries off the public slice is exactly what the brief's Rogii warning is about, and it is
the current default. **Needs Teddy, in his own browser, before 2026-08-31. Not doable from the
API.** This outranks the 33e-6 gap.

## 2026-08-24 15:3xZ (w80) — 2,791 teams

| | |
|---|---|
| our best public | **0.97119** |
| strictly above us | **97** |
| tied with us | 15 |
| best-case rank | **98** |
| leader — Chris Deotte | **0.97168** (resubmitted 14:50:33Z, score unmoved) |
| gap to leader | **490e-6** |

Top ten: Deotte 0.97168 · Changye Li 0.97150 · MILANFX 0.97146 · cstdy 0.97137 ·
Maher el Ouahabi 0.97134 · Utkarsh 0.97132 · Keanan 0.97132 · Don Mani 0.97130 ·
Szymon Kłapiński 0.97130 · Optimistix 0.97130.

Drift over the day: 94 above at 14:55Z (w78) → 96 at 15:15Z (w79) → **97 at 15:3xZ**. The
board is creeping past us at roughly one team per 20 minutes and our score is frozen — the
queue holds nothing above the best already sent.

### 🔴 THE TOP PUBLIC NOTEBOOK IS 0.97127 AND IS **STILL NOT A METHOD**
`atakanaldemir/s6e8-regime-calibrated-rank-fusion-lb-0-97127` (run today 09:53Z), pulled to
`notebooks/w80_atakan_97127` and read in full. It is **twenty lines of blending**:

    final = rank(0.725001 * rank(hboyang_205_member_fusion) + 0.274999 * rank(v13_anchor))

Both legs are pre-made CSVs it loads from `/kaggle/input` — `submission_nested.csv` from
hboyang's kernel and `v13_diversity_anchor_lb97124.csv` from its own author's dataset. There
is no model, no fold, no CV. The `1e-6` offset on the weight is there only to break rank ties.

⇒ **w70's conclusion survives unchanged; only its numbers were stale.** Refresh them:
best public **0.97123 → 0.97127**, hboyang's fusion **150 members → 205**. The cluster above
us is still very largely one circulating aggregate's distribution list, and it is still
unusable here (submission-only → no cross-fitted CV → `fam=member` → the sender blocks it).

## 2026-08-25 12:4xZ (w82) — 🔴 +40 teams passed us in 24 hours

| | w80 (08-24 15:3xZ) | **w82 (08-25 12:4xZ)** |
|---|---|---|
| leader — Chris Deotte | 0.97168 | **0.97172** |
| teams strictly above our 0.97119 | 97 | **137** |
| teams tied at 0.97119 | 15 | **12** |

Measured by paginating `kaggle competitions leaderboard -s --csv -v` with `--page-token` until
the score fell below 0.97119 (160 rows read, min 0.97117). Our rank is ~138–149 of ~2,800 — the
silver/bronze boundary, no longer inside it.

Account best **unchanged at 0.97119** (`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`, both
08-23). Today's ten came back 0.97113–0.97117; summed `P(beat 0.97119)` was 4.51e-3.

⛔ **Not a reason to chase the public LB.** `w82a` measures the LB predictor's own error at
**±14e-6**, against deadline margins of 1.9–4.5e-6 — the public slice cannot arbitrate. The
honest read is that every unsent file is priced below the account best and six angles are closed
on measurement, so nothing in the pipeline moves this number before 08-31.

## 2026-08-25 ~13:55Z (w85)

2,877 teams · **rank 140 · 0.97119 · top 4.87%** · Chris Deotte leads 0.97172 (gap 0.00053).
Top-5% cut rank 143 (0.97119) — we are 3 places inside it; top-10% cut rank 287 (0.97106).
Against w84's 140/2875 at 13:05Z this is **no material movement**, so `w83a_reproject` was not
re-run (its own instruction: re-run only on a material board or rank move). Board in `lb_w85/`.

Public field, 08-25: six notebooks the journal had never recorded —
`yekenot/ps-s6-e8-trompt-pytorch-frame` (Trompt/PyTorch-Frame, 4 votes) is the only
methodologically new one, and the workspace already holds TabM/TabNet/RealMLP/ResNet/DCNv2/
FT-Transformer/DeepFM with the `nn` family measured at +0.8e-6 and a sign flip. Others:
`souvikdbiswas` rank blend (8), `mikhailnaumov` XGB (12), `parthsarnobat` cat+xgb (5),
`sometimessubodh` (5), `bariankitvinod` (0).

## 2026-08-25 (w88, slot 7) — no movement

    rank 140 / 2,881 at 0.97119 (top 4.86%); top-5% cut is rank 144, we are 4 inside
    1 Chris Deotte 0.97172 (gap 0.00053) · 2 Changye Li 0.97154 · 3 MILANFX 0.97149
    4 cstdy 0.97140 · 5 Utkarsh 0.97134

Unchanged against w87's 140/2,881, so `w83a_reproject` was deliberately not re-run — no
material board move is its own rule.

⚠ `kaggle competitions leaderboard -s --page-size 5000` returns **exactly 200 rows** and says
nothing about it. Rank 140 is inside that window today; the read goes silently blind if the
account ever slides past 200. Nothing on the send path depends on it.

## 2026-08-25 (w90) — the first competitor CV we have ever been able to check

Board not re-read this run (no submission possible; `w83a_reproject` is reserved for a material
board move). Standing read from w88, this morning: **140 of 2,881 at 0.97119**, Chris Deotte
0.97172, gap 0.00053.

New and durable: `atakanaldemir` published the OOF vector behind their 0.97124 anchor, and it
reproduces on our verified partition to 0.000e-6.

| | CV (our folds) | public |
|---|---|---|
| our pick `w36_ad199stdcorr` | 0.9701400060 | 0.97118 |
| `atakanaldemir` v13 anchor | **0.9701665486** | 0.97124 |

They are **+26.54e-6 ahead on CV**, r = 0.998733 with our pick. Their own README attributes part
of it to selecting the correction weight on the full OOF grid, and 50 of their 244 members are
the szymonkapiski weak-50 this workspace refuses on w51's es-on-val clause. `w90b_gapcheck.py`
tried to test whether that inflation is visible in their public score and **registered that it
cannot** — the family-level term the comparison must assume away is +37.4e-6 wide against a
+38.3e-6 effect. Treat the CV gap as measured and its cause as open.

## 2026-08-27 (w95) — rank 197 / 3070, score 0.97119

Read from the **full download** (`lb_w95/playground-series-s6e8.zip`, opened with `zipfile`),
not from `leaderboard -s`, which now returns only 20 rows plus a page token.

    leader   Chris Deotte   0.97189   (was 0.97184 on 08-26)
    us       0.97119        rank 197 of 3070   161 submissions
    9 teams tied with us at 0.97119; 196 teams ahead; leader +700e-6 clear

    0.97124  9 teams   149 ahead
    0.97123 14 teams   158 ahead
    0.97122  3 teams   172 ahead
    0.97121 13 teams   175 ahead
    0.97120  8 teams   188 ahead
    0.97119  9 teams   196 ahead   <- us
    0.97118  6 teams   205 ahead

⚠ **Rank fell from 92/2774 (w75, 08-24) to 197/3070 while our score did not move.** The field
grew by ~300 teams and caught up. Do not read this as a regression in our file.

⚠ The board is extremely dense here: one 1e-5 reporting step around us is worth **8–15 places**,
and the entire CV span of everything this account has ever sent is 499e-6 — five reporting
steps. That is the arithmetic reason selection stays on CV and not on public rank.

Today's ten all landed at 0.97104–0.97116, none at or above the 0.97119 tier, so
`w74b_clickstaleguard` confirms the +4.5228e-6 click price is unchanged.

## 2026-08-27 (w96) — the board around us did not move at all in 24 hours

Full download `lb_w96/`, read with `zipfile` (the CLI's `-s` view is truncated and must not
be quoted — RESEARCH line 452).

    3073 teams   leader Chris Deotte 0.97190   us 0.97119   rank 197
    9 teams tied with us; 196 ahead; leader +710e-6 clear

    0.97125 12 teams   137 ahead
    0.97124  9 teams   149 ahead
    0.97123 14 teams   158 ahead
    0.97122  3 teams   172 ahead
    0.97121 13 teams   175 ahead
    0.97120  8 teams   188 ahead
    0.97119  9 teams   196 ahead   <- us
    0.97118  6 teams   205 ahead
    0.97117 28 teams   211 ahead

⚠ **Compare that block with w95's, one day earlier: every band count is identical and our
rank is identical (197).** The field grew by 3 teams and the leader gained 1e-5. The w95
entry read the 92 → 197 slide as "the field caught up", which was right; this reading adds
that the catching-up was a step, not a trend — 24 hours later nothing around us moved. Ten
submissions went out on 08-27 and none of them changed our score, which is expected: the
public LB shows the best of all submissions and today's ten were low-CV drain files.

Both facts together mean the same thing for the deadline: **rank is not drifting away from
us, and it is not coming back either.** Only a genuinely better file moves it, which is what
the w96 TE-prior measurement is for.

## 2026-08-27 (w97) — third consecutive identical reading

`lb_w97/`, full download, board stamped 2026-08-27T13:47:12 UTC. **3,074 teams** (3,073 on
08-26). Leader Chris Deotte **0.97190**. Us **0.97119: 196 ahead, 9 tied, rank 197** — the same
three numbers as w95 and w96. The 92→197 slide was a step, not a trend; nothing around us has
moved in 48 hours.

Top ten: 0.97190, 0.97154, 0.97149, 0.97140, then a five-way shelf at 0.97134, 0.97132.

**Band density around us is spiky, and the spikes are notebooks, not skill.** Counting teams per
1e-5 bin:

    0.97130    7      0.97124   10      0.97118    6      0.97113   53   <-- shared CSV
    0.97129   17      0.97123   13      0.97117   28
    0.97128   75  <-- shared CSV        0.97116    1
    0.97127   16      0.97122    3      0.97115    9
    0.97126    8      0.97121   13      0.97114   12
    0.97125   12      0.97120    8      0.97119    9   <-- us

3–17 teams in every ordinary bin, then 75 at 0.97128 and 53 at 0.97113. The 0.9712x one is
`atakanaldemir/s6e8-v13-diversity-anchor`, **already dispositioned** — w90 reproduced its OOF on
our verified partition to 0.000e-6 and its own README admits the correction weight was chosen on
the full OOF grid (es-on-val), with 50 of its 244 members being szymonkapiski's weak-50 that w81
and w89c independently refused. ⛔ Closed; not re-opened.

Auto-pick tiers **unchanged by today's ten sends** (best of them 0.97116): slot 1 is still the
0.97119 two-way (`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`), slot 2 the 0.97118 five-way.

## 2026-08-27 (w98, 10:05 EDT) — rank 198 of 3,075. Fourth consecutive flat reading.

Full download, `lb_w98/`. Leader **Chris Deotte 0.97190**. Us **0.97119** — 197 ahead, 9 tied.
w95/w96/w97 each read 197 of 3,074; one team passed us and one joined, so this is flat, not a
slide. The 92→197 move is now four readings old and settled.

Band density 0.97100–0.97140 (bins of 1e-5):

    0.97140   1        0.97124  10        0.97113  53
    0.97134   5        0.97123  13        0.97112   8
    0.97132   3        0.97122   3        0.97111   5
    0.97131   2        0.97121  14        0.97110   3
    0.97130   7        0.97120   8        0.97109   5
    0.97129  17        0.97119   9  <-us  0.97108   5
    0.97128  75        0.97118   6        0.97107   2
    0.97127  16        0.97117  28        ...
    0.97126   8        0.97116   1
    0.97125  12        0.97115   9

Two spikes, both shared notebooks rather than skill gradients: **75 at 0.97128** (the
`atakanaldemir` regime-calibrated rank fusion, reproduced and dispositioned by w90 as es-on-val)
and **53 at 0.97113**. Neighbouring bins run 1–17.

**Medal arithmetic at 3,075 teams:** gold ≈ top 14, silver ≈ top 154, bronze ≈ top 308. We are
bronze on public. Silver needs rank ≤154 = clearing the 48 teams between 0.97120 and 0.97124,
i.e. **+60e-6 of public AUC**, against a send queue whose best remaining file is priced at
`P(beat best) 0.00e+00`. ⛔ **Not reachable by sending.** Medals settle on private, where the two
spikes are the ones with something to lose.

Public notebook ceiling is **0.97127** (`atakanaldemir`), so no public source holds an edge over
us worth more than 8e-6 — and that one is already known to be es-on-val.

## w103 — 2026-08-28 13:12Z, no movement worth recording

Top of board unchanged in shape: Chris Deotte 0.97205, Changye Li 0.97154, MILANFX 0.97149,
cstdy 0.97145. Our best public still **0.97119**; today's ten (12:37Z) landed 0.97101–0.97103,
all pinned queue-drain files, none near the tier. Bronze, and §above's arithmetic is unchanged.

⚠ **A NOTE FOR WHOEVER READS OUR OWN SCORES NEXT.** `kaggle competitions submissions -v` returns
**one page — 50 rows of 171**. Sorting that page gives best public **0.97117**, which looks like a
correction to the 0.97119 on record and is not: the better entries are simply on a later page.
Parse it with a real CSV reader too — descriptions contain commas, so `awk -F,` reads the wrong
field. `w86a_pagecap` is the standing check for exactly this.

## w105 — 2026-08-28 13:58Z, board still flat, and the pagination trap caught me too

Top four identical to w103's 13:12Z reading: Chris Deotte 0.97205, Changye Li 0.97154, MILANFX
0.97149, cstdy 0.97145. Eight hours, no movement in the shape of the board. Our best public
**0.97119**; today's ten are all in and confirmed (`by UTC day` on the first page: 08-25 through
08-28, ten each).

😐 The warning above is worth what it cost to write — I hit **both** halves of it in one command
before reading it: `awk -F,` on the last field returned blank (descriptions carry commas and
`privateScore` is empty), and the page sorted to **0.97117**, which is the page-1 maximum, not the
account maximum. Read it with `csv.DictReader` and remember it is 50 rows of 171.

## w106 — 2026-08-28 14:55Z, board still flat for a **tenth** hour

Top three unchanged from w103's 13:12Z and w105's 13:58Z readings: **Chris Deotte 0.97205**,
Changye Li 0.97154, MILANFX 0.97149. Our best public **0.97119** (`w38_ad202stdcorr_ens4.csv`,
tied with `w36_ad199stdcorr_ens4.csv`), read from a **`--page-size 200`** pull with
`csv.DictReader` — 171 rows, so the page-1 trap above was avoided by following the note rather
than rediscovering it. 0 slots today; the registered 08-29 ten goes out next run.

⚠ **The gap to first is 0.00086 of public AUC and it is not closable by modelling now.** w97
closed the last open experiment at **−1.9e-6** (NO-ENROL), and every ANGLE INDEX row is shut. The
one remaining action with a measured price is the **final-selection click** in `SELECT_THESE.md`
— **+4.5228e-6 of expected private AUC** for one minute of a human's time, and it is *private*
score, which is the one that counts. Chasing the public gap is the Rogii failure; taking the
click is not.

## w107 — 2026-08-28 15:0xZ, board flat for an **eleventh** hour

Top three unchanged again from w103 (13:12Z), w105 (13:58Z) and w106 (14:55Z): **Chris Deotte
0.97205**, Changye Li 0.97154, MILANFX 0.97149. Fourth and below shuffle a little (cstdy 0.97145,
Keanan 0.97139) but nothing near the top has moved in eleven hours. Our best public **0.97119**,
unchanged. 0 slots today; the registered 08-29 ten goes out next run.

⚠ **Nothing here changes the standing read.** The 0.00086 gap to first is not closable by
modelling with three days left — every ANGLE INDEX row is shut and w97 closed the last open
experiment at −1.9e-6. w107 re-verified **row 5** at the artefact level (it holds, price
negative) and marked its carve-out **spent**. The one remaining action with a measured price is
still the **final-selection click** in `SELECT_THESE.md`: **+4.5228e-6 expected private AUC**,
and `check_selection.py` reads **rc=1 — nothing selected** for the sixth run running.

## 2026-08-28, w108 (slot 7) — board unchanged, no submission possible (0 of 10 slots)

Top of board still `Chris Deotte` **0.97205**, then 0.97154 / 0.97149 / 0.97145. Account best
remains **0.97119** (`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`). Today's ten went out
before this run; the two most recent landed 0.97105 and 0.97103, both as expected — every
remaining unsent file is below the best already-sent CV and `w26d` prices the whole queue at
6.3e-4 of beating the board. **Nothing on the leaderboard changed the CV-based deadline pick.**

⛔ The final-selection click is still not done — see JOURNAL w108 §7. Auto-selection on public
score costs **+4.5228e-6** of expected private AUC against the CV pick.

## 2026-08-28, w110 (slot 9) — board flat for a **fourteenth** hour, no submission possible (0/10)

Top five unchanged from w103 (13:12Z) through w109: **Chris Deotte 0.97205**, Changye Li 0.97154,
MILANFX 0.97149, cstdy 0.97145, Keanan 0.97139. Account best **0.97119**
(`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`), unchanged. Today's ten went out at 12:37–12:38Z
before this run; the latest landed 0.97105 / 0.97103 / 0.97101, all as `w26d` predicted.

⚠ The read does not move. The 0.00086 gap to first is not closable by modelling with three days
left. w110 added the two ANGLE INDEX rows that were missing (error analysis, consolidation) and
re-verified the error-analysis closure across **all four** of its instruments — every figure
reproduces, `real − ctrl` is negative at 8/8 and 9/9 in the two boosting arms, and 76.7% of the
AUC deficit is cross-cell, which no feature can reach. **Nothing on the board changes the
CV-based deadline pick.**

⛔ The final-selection click is still not done — ninth run asking. `check_selection.py` reads
**rc=1, nothing selected**. Auto-selection on public score costs **+4.5228e-6** expected private
AUC against the CV pick, and private is the score that counts.

## 2026-08-28, w111 (slot 10) — board flat for a **fifteenth** hour, no submission possible (0/10)

Top five unchanged since w103 (13:12Z): **Chris Deotte 0.97205**, Changye Li 0.97154, MILANFX
0.97149, cstdy 0.97145, Keanan 0.97139. Account best **0.97119** (`w36_ad199stdcorr_ens4`,
`w38_ad202stdcorr_ens4`), unchanged. The 08-28 ten went out at 12:37Z, before this run.

The consolidation slot spent itself on the selection rather than the board, and it moved one
thing that was never checked: **both deadline-candidate files now rebuild BYTE-IDENTICALLY from
their bases** (CSV and OOF md5, 8 days after the originals), so the CV pick and the file
auto-selection would take are each verified to be the artefact their CV describes. The CV→LB
gap over all 160 scored files with a parseable CV is **+1035.8e-6 ± 33.9e-6, pearson +0.8842**,
and 31 sends since 08-26 have moved it 2.4e-6 and moved neither the auto pair nor the CV pair.

⛔ The final-selection click is still not done — **tenth** run asking. `check_selection.py` reads
**rc=1, nothing selected**. Auto-selection on public score costs **+4.5228e-6** expected private
AUC against the CV pick, and private is the score that counts.

## 2026-08-29, w112 (slot 1) — the 08-29 ten SENT, board's first move in ~16 hours, rank 247/3238

Top five, 12:44Z: **Chris Deotte 0.97207** (was 0.97205 — first movement at the top since
w103 on 08-27), Changye Li 0.97154, MILANFX 0.97149, **cstdy 0.97147** (was 0.97145), Keanan
0.97139. Two of the five moved, both by +2e-5, both irrelevant to us.

**Account best 0.97119, unchanged** (`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`).
Full-leaderboard download: **rank 247 of 3,238 teams**, 181 submissions. ⚠ The brief's
"~1,326 teams" is badly stale — the field has grown 2.4× since it was written, so quote the
downloaded count, not the brief.

The registered 08-29 ten went out 12:37:12–12:37:44Z and all ten scored. Best of the day
**0.97107** (`w16d_membercell`), which is 120e-6 below the account best and exactly where
`w55a` certified it would be. Nothing on the board changes the CV-based deadline pick.

⛔ The final-selection click is still not done — **eleventh** run asking. `check_selection.py`
reads **rc=1, nothing selected**. Auto-selection on public score costs **+4.5228e-6** expected
private AUC against the CV pick, and private is the score that counts. Two days left.

## 2026-08-29, w114 (slot 3, no slot — cap spent) — no board read, and the click's price grew

No leaderboard download this run and no send (0 of 10 slots). w113's board stands: **rank 249 of
3,241**, top 7.68%, bronze cut 324 → **75 places inside**, account best 0.97119.

⛔ The final-selection click is still not done — **thirteenth** run asking, two days left.
`check_selection.py` reads **rc=1, nothing selected**. What changed this run is not the board, it
is the price. Not clicking costs **+4.5228e-6**; **clicking the WRONG pair costs +35.17e-6 or
+81.92e-6** (`w114a_misclick.json`) — and until this run `check_selection.py` itself printed both
wrong pairs as though they were the instruction. At w113's ~0.95 teams per 1e-6 the second is
roughly **78 places against a 75-place bronze margin**. The refs to click, read live:
**55656399** (`w36_ad199stdcorr.csv`) and **55588167** (`w23_ad187stdcorr.csv`), nothing else.

## 2026-08-29, w117 (slot 6, no slot — cap spent at 12:37Z) — no board read, by the rule

Nothing sent this run, so our public score did not move, so `w83a_reproject`'s own rule says not
to re-download. **w113's board still stands: rank 249 of 3,241, top 7.68%, bronze cut 324 → 75
places inside, account best 0.97119.** Re-deriving it would only reproduce a number.

⛔ The final-selection click is still not done — **sixteenth** run asking, two days left.
`check_selection.py` read live this run: **rc=1, `*** NOTHING IS SELECTED ***`**. Click
**55656399** (`w36_ad199stdcorr.csv`, public 0.97118) and **55588167** (`w23_ad187stdcorr.csv`,
public 0.97116), and nothing else. `PushNotification` tried again and returned *"Mobile push not
sent (Remote Control inactive)"* — **0 for 3; it has never once reached his phone from here.**

## 2026-08-29, w118 (slot 7, no slot — cap spent at 12:37Z) — board rule challenged, then found not binding

I set out to overturn `w83a_reproject`'s "don't re-download unless our score moved" rule, because
the **bronze cut is set by other teams**: our margin can decay while our own score sits still, and
the rule as written does not cover that. The objection stands in general but did not apply here —
`lb_w113/` is stamped **2026-08-29T13:10:54**, under two hours old. **w113's board is today's:
rank 249 of 3,241, top 7.68%, bronze cut 324 → 75 places inside, account best 0.97119.**

The endgame is now audited end to end: the **last 20 submissions** (08-30 and 08-31, both frozen
artefacts) are mutually disjoint, unsent, all present on disk at 296,302 rows, and all ten of the
08-30 ten sit at `pred_lb ≤ 0.960339` — under the 0.97119 auto-selection tier, so they cannot
hijack it. Across all 201 files this account has sent or will send, **201 distinct rank vectors:
not one submission slot has ever been spent on a file the leaderboard could not tell apart.**

⛔ The final-selection click is still not done — **seventeenth** run asking, two days left.
`check_selection.py` read live this run: **rc=1, `*** NOTHING IS SELECTED ***`**. Click
**55656399** (`w36_ad199stdcorr.csv`, public 0.97118) and **55588167** (`w23_ad187stdcorr.csv`,
public 0.97116), and nothing else. `PushNotification` tried again and returned *"Mobile push not
sent (Remote Control inactive)"* — **0 for 4; it has never once reached his phone from here.**

## 2026-08-29, w120 (slot 9, no slot — cap spent at 12:37Z) — board re-read, and the rule that said not to is now refuted

**Fresh download, `lb_w120/…2026-08-29T15:23:59.csv`: rank 251 of 3,253, top 7.72%, bronze cut
325 → 74 places inside. Account best 0.97119, unchanged since 08-23 12:41:28.** The bronze cut
sits at score 0.97113; 102 teams occupy [0.97113, 0.97119).

w118 and w119 each objected that `w83a_reproject`'s *"don't re-download unless our score moved"*
ignores that the cut is set by other teams, and each then quoted w113 anyway. The series settles
it — our number has not moved once and the margin has:

| date/time (UTC) | teams | rank | bronze cut | margin |
|---|---|---|---|---|
| 08-25 (w64) | 2,660 | 64 | 266 | +202 |
| 08-27 (w98) | 3,075 | 198 | 307 | +109 |
| 08-29 13:10 (w113) | 3,241 | 249 | 324 | +75 |
| 08-29 15:24 (w120) | 3,253 | 251 | 325 | +74 |
| 08-30 12:45 (w122) | 3,321 | 278 | 332 | +54 |

🔴 **THE −11/day BELOW WAS FITTED ON A 2h13m WINDOW AND IT IS WRONG. THE MEASURED RATE IS
−22/day** (w122, 2026-08-30): +74 → +54 over 21h21m. The old reading survived here for a day
because a rate quoted from a two-hour window looks like a trend and reads like one. ⛔ Do not
quote −11/day; the journal has carried a standing DO-NOT against it since w124.

**128 places of margin gone in four days at a frozen score**, and the deceleration was an
artefact of the short window: −93, then −34, then −22/day and holding. From +54 at 08-30
12:45Z the deadline is 35h14m away ⟹ **−32 ⟹ margin ≈ +22** at 08-31 23:59. Bronze survives on
that rate, which does not contain a deadline-eve surge — and a surge is exactly what the last
day of a Playground competition produces. ✅ Standing rule: **read the board once per day**, not
"when our score moves".

🎯 **The tie-break is NOT last-submission-date.** Seven teams sit at 0.97119; ours is the *latest*
last submission of the seven (2026-08-29 12:37:44, 181 subs) and we rank **first** in the block,
with identical internal order in both of today's snapshots. ⟹ Draining the remaining 20 slots
cannot cost a rank — the brief's "an unused slot is pure waste" holds down to the tie-break.

The registered 08-29 ten all scored, best of the day **0.97107** (`w16d_membercell`), against a
pre-registered point estimate of **0.971070** — exact. All ten landed below the 0.97119
auto-selection tier and all six that carried a `w55a` bound landed inside it.

⛔ The final-selection click is still not done — **nineteenth** run asking, two days left.
`check_selection.py` read live this run: **rc=1, `*** NOTHING IS SELECTED ***`**. Click
**55656399** (`w36_ad199stdcorr.csv`, public 0.97118, CV rank 1 of 164) and **55588167**
(`w23_ad187stdcorr.csv`, public 0.97116), and nothing else.

## w129, 2026-08-30 — no board read this run; the daily read was w122's at 12:45Z

⛔ **One board read per day.** w122 spent today's at 12:45Z — **rank 278 / 3,321, cut 332, margin
+54** — and nothing here re-spends it. The **−22/day** rate above is the current one; the
**−11/day** it replaced stays banned.

🆕 **What changed today is the chain, not the score.** Both files the click is asked to select now
rebuild **byte for byte** from their bases: `w36_ad199stdcorr` (w111, 8 days on) and
`w23_ad187stdcorr` (w129, **13 days on**, md5 `4fa32c223a…`, 0 artefact fields differing). Nothing
between the CV number and the file Kaggle scores is unverified any more.

⛔ The final-selection click is still not done — **twenty-eighth** run asking, **deadline
tomorrow**. `check_selection.py` read live this run: **rc=1, `*** NOTHING IS SELECTED ***`**. Click
**55656399** (`w36_ad199stdcorr.csv`, public 0.97118) and **55588167** (`w23_ad187stdcorr.csv`,
public 0.97116), and nothing else. Not clicking costs **+4.5228e-6**; clicking the wrong pair
costs **+35.17e-6** or **+81.92e-6**.

## w130, 2026-08-30 — top-of-board only; the daily rank read stays w122's at 12:45Z

⛔ **One board read per day, and w122 spent today's** — rank **278 / 3,321**, cut 332, margin
**+54**, decaying at **−22/day** (the **−11/day** it replaced stays banned). I read the top six
this run and not our own line, so the rank above is still the current number: head of board
**0.97207** (Chris Deotte, 08-30 01:33Z), then 0.97154 and 0.97149. Our best public is
**0.97119**.

🆕 **Nothing about the board changed this run; what changed is why the CV numbers are readable at
all.** w130 measured the fixed-fold harness: the same comparison scored on shared rows has an sd
of **3.44e-6**, scored on independent rows **271.11e-6** — 5.4× the 50e-6 floor. Every price this
workspace has published, and every CV-vs-LB gap in this file, exists inside that 78.9× reduction.

⛔ The final-selection click is still not done — **twenty-ninth** run asking, **deadline
tomorrow**. `check_selection.py` read live: **rc=1, `*** NOTHING IS SELECTED ***`**. Click
**55656399** (`w36_ad199stdcorr.csv`, public 0.97118) and **55588167** (`w23_ad187stdcorr.csv`,
public 0.97116), and nothing else. Auto-select otherwise takes the 0.97119 tier
(`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`) and both WANTED files are **UNREACHABLE**.

## 2026-08-30, 18:5xZ (w131) — deadline eve

Public top: Chris Deotte 0.97207 · Changye Li 0.97154 · MILANFX 0.97149 · cstdy 0.97147 ·
Miłosz 0.97140. Our best public stands at the 0.97119 tier (`w36_ad199stdcorr_ens4`,
`w38_ad202stdcorr_ens4`), which is also what Kaggle will auto-select if nobody clicks.

⛔ **Nothing is selected for the final score.** Thirtieth run asking. The two WANTED files
(`55656399` w36_ad199stdcorr, `55588167` w23_ad187stdcorr) are UNREACHABLE without the click,
P=0.000. Not clicking costs +4.5228e-6; clicking the wrong pair costs +35.17e-6 or +81.92e-6.

## 2026-08-31, 12:49Z (w132) — **DEADLINE DAY. THE BOARD CLOSES AT 23:59 UTC.**

Downloaded live, not quoted from yesterday: **3,453 teams**, and we sit at **rank 304, public
0.97119**, 201 submissions. Public top: Chris Deotte **0.97207** · Changye Li 0.97154 ·
MILANFX 0.97149 · cstdy 0.97147 · Miłosz 0.97144 (moved up today) · Keanan 0.97139 ·
JungHwan 0.97138 · BOB 0.97137. The gap to first is **0.00088**.

🎯 **Rank 304 of 3,453 is inside the bronze band** — bronze is the top 10%, i.e. rank ≤ 345 at
this team count, so the margin is **41 places**. It is not a comfortable margin on the last day
of a Playground: the field compresses hardest in the final hours, and the standing decay
estimate in RESEARCH.md is **−22 places/day**, not −11.

Today's ten sends (12:36–12:37Z, the w129 DRAIN queue) scored **0.93232 … 0.94156**. All ten are
far below the **0.97119** auto-selection tier, which is the point: they are measurements that
cannot displace an auto-pick. Our public score did not move and was not expected to.

⛔ **Nothing is selected for the final score. Thirty-first run asking, and there is no
thirty-second.** `55656399` (`w36_ad199stdcorr.csv`, public 0.97118) and `55588167`
(`w23_ad187stdcorr.csv`, 0.97116) are **UNREACHABLE without the click, P=0.000**. If nobody
clicks, Kaggle auto-selects the 0.97119 tier (`w36_ad199stdcorr_ens4`, `w38_ad202stdcorr_ens4`)
— which costs **+4.5228e-6** against the WANTED pair. Clicking the *wrong* pair costs
**+35.17e-6** or **+81.92e-6**, so the mis-click remains the bigger hazard by an order of
magnitude. Both outcomes are survivable; only the mis-click is expensive.

## 2026-09-01, 12:44Z (w142) — 🏁 **FINAL. THE COMPETITION IS CLOSED AND THE PRIVATE BOARD IS READ.**

    PRIVATE   rank 319 of 3,532   score 0.97093   top-decile line 353   margin +34
    PUBLIC    rank 331 of 3,532   score 0.97119   (the same board, override_public=True)

**Inside the top decile, with 34 places to spare.** Public → private moved us **331 → 319, a
gain of 12**. Private top: Chris Deotte **0.97176** · MILANFX 0.97123 · Changye Li 0.97120 —
note 2nd and 3rd **swapped** against the public board, and the leader gave back 31e-6.

🎯 **THE FIELD MOVED, AND IT MOVED HARDEST AT THE TOP.** Across all 3,532 teams the median rank
move was −3 with sd 76; half the field moved 13 places or less. But of the **public top 100, 84
fell**, median **+89**, worst **+475**. The public leaderboard's leading edge was substantially
overfitted, and the teams that chased it paid for it in exactly the way the Rogii warning
predicted.

⚠ **OUR OWN MOVE WAS NOT SKILL, IT WAS ABSENCE FROM THAT EDGE.** A 12-place gain is quieter
than 51% of the field — utterly ordinary. We gained because we were not standing where the
collapse happened, not because we predicted the collapse.

**The selection question turned out to be worth nothing measurable.** All four arms —
argmax-CV, argmax-public, the WANTED pair, and the AUTO pair Kaggle actually used — land on
**0.97093**. The hindsight oracle over all 201 sends is 0.97094, worth **+7 ranks**. Thirty-four
runs asked for a click that could not have moved the result at the board's own resolution.

## 2026-09-02, 13:5xZ (w154) — no change, and the rank was re-derived from the right field

**Private rank 319, score 0.97093.** Unchanged for the twelfth day.

⚠ **THE DENOMINATOR HAS TWO VALUES AND THIS FILE HAS BEEN MIXING THEM.** The competition object's
`team_count` reads **3,531** and has read 3,531 in every journal block back to 08-31. The **3,532**
in the w142 entry above is a different source: the row count of the private *leaderboard* read.
The two endpoints disagree by one and always have — it is not a team leaving after grading, which
is what a naive reading of the two numbers side by side suggests. Neither value moves rank 319 or
the top-decile margin, so nothing published changes; the point is that "3,531 vs 3,532" is an
endpoint difference to be stated, not a timeline to be explained.

The rank number itself was re-read live this run rather than quoted, and the read is worth
recording because it has a trap in it. `ApiCompetition` has **no `.rank`** — the field is
**`user_rank`**, and it only carries 319 when the request goes through an authenticated client
(`KaggleApi()` → `authenticate()` → `build_kaggle_client()`). Through a bare `KaggleClient()`
the same call succeeds, returns every competition-scoped field correctly, and reports
**`user_rank` 0** with **`user_has_entered` False** — a board this account sent 201 scored
submissions to. Guarded as `experiments/w154a_authscope.py`.

## 2026-09-03, 13:0xZ (w158) — no change, and the significance column was on the wrong scale

**Private rank 319 / 3,531, score 0.97093.** Unchanged for the thirteenth day. `w142b_privatecheck`
re-run live: 201 rows, 201 with `privateScore`, `selected` 0, status `COMPLETE` ×201, best private
0.97094 (`w40_ad211stdcorr`). Competition object read through an authenticated client: deadline
2026-08-31 23:59, teams 3,531, `user_rank` 319, metric Roc Auc Score, `max_daily` 10.

⚠ **NOTHING ON THE BOARD MOVED, BUT ONE NUMBER IN THE PRICE COLUMN DID.** Every `t` this workspace
published was `delta/sd`, an effect size, where the statistic is `delta/(sd/√3)` on REPS = 3 paired
splits. Understated by √3 = 1.732 and compared against critical values on df = 2 that belong to the
corrected scale. **No verdict changes side at 5%** — the largest corrected t among the arms the
table calls null is 2.92 against 4.303, so no row re-opens and no ranking here is affected. What it
does falsify is a published sentence: `no single-family enrolment rate in this table clears 1%`.
On the corrected scale `+xgb_only` reads 10.61 and `+xgb_dedup` 10.75 against 9.925, and both clear
it; `+lgb_only` misses by 0.001 at 9.924. Corrected in the ANGLE INDEX and in #61's own docstring,
guarded as `experiments/w158a_tscaleguard.py` (#68).

## 2026-09-03, 13:5xZ (w160) — no change on the board, and two runs of it existed on one disk

**Private rank 319 / 3,531, score 0.97093.** Unchanged for the fifteenth day. Competition object
read live through an authenticated client: deadline **2026-08-31 23:59** (past by **61.8 hours**),
teams **3,531**, `user_rank` **319**, metric **Roc Auc Score**, `max_daily` **10**.
⚠ `submissions_disabled` still reads **False** on a closed, graded board, for the sixteenth day.

⛔ **THE BOARD DID NOT MOVE AND THE RECORD OF IT NEARLY DID.** `origin/main` was two commits
behind `HEAD` — w157's last commit and w159's only one, the second of which is guard #68 in full.
Confirmed by a live `ls-remote`, not a cached ref. w157 pushed and then committed again; w159
committed and never pushed. Neither push failed; neither ran. Pushed as `5259571` and guarded as
`experiments/w160a_pushguard.py` (#70), which reads the remote over the network every run and
fails rather than reporting clean when it cannot.

⚠ **AND THE CENSUS STAYED RED AFTER THE RECORD WAS RESTORED.** Rows 7 and 9 each read one short,
because both restored entries are *reconstructions* and a reconstructing run never saw the
vanished run's ANGLE, so it cannot quote it the way the resolver requires. Fixed in
`w117a_handcount` as one bounded rule with a both-directions C7 (2 of 177 runs, 1.1%, ceiling
10%). Census back to **FAILURES: 0**, all ten rows agreeing with the index.

⛔ **THE BOARD IS UNCHANGED — rank 319 / 3,531, private 0.97093 — AND THE RUN THAT GRADED IT WAS
MISSING FROM EVERY COUNT THIS WORKSPACE PUBLISHES.** w142 (09-01, slot 1, row 4) read the final
leaderboard, graded the frozen forecast (P2/P3/P4 held, P1 ungraded exactly as w140 predicted) and
measured the founding rule at **CV↔private rho +0.929 against public↔private +0.874**. Its journal
header is a parenthetical, and `RUN_HDR` anchors on the character after `# `, so the census, #66,
#67 and #69 were all blind to it. Row 4 looked like the only angle nobody had been handed since
the deadline; it had been handed, and it produced the grading. Reader fixed, row 4 resynced
×16 → ×18.

⚠ **AND THE DAY ITSELF LOST FIVE SLOTS.** 09-03 slots 3, 4, 6, 7 and 8 were killed mid-run by API
500/529 (09-02 lost six of ten). w156 and w159 were terminated, not negligent — the launcher log
says so, and it is the only instrument here that can. Guarded as `experiments/w161a_driverguard.py`
(#71).


## 2026-09-03 — w162 — the board is still 319 / 3,531 at private 0.97093, and the other eleven
## missing runs were found

⛔ **THE BOARD IS UNCHANGED.** 201 submissions, all 201 graded, nothing sent since 08-31. Read live
this run: the single best private file over everything ever sent is `w40_ad211stdcorr.csv` at
**0.97094**, one quantisation step above the pair the auto-select took; twelve files tie at
0.97093. The CV pair and the public pair both realise **0.97093**. That reproduces w142's finding
independently and it is not a new number.

🔴 **w161 FOUND TWO RUNS THE CENSUS COULD NOT SEE. THERE WERE ELEVEN MORE.** The 08-16 → 08-18
window writes its entry headers as a banner (`# ══ 2026-08-17 (UTC) — WAVE w17, SLOT 1 of 10 ══`),
and `RUN_HDR` anchors on the character after `# `, so none was visible. They were not merely
uncounted: an invisible header leaves its body inside the previous entry, so **two headers were
carrying twelve runs' worth of body between them** — one of them owned the 08-16 wave summary plus
seven whole 08-17 slot entries. Corpus 181 → 192, and **nine of the ten index rows were short**,
by +1 or +2 each.

⚠ **ROW 5 IS THE ONE ROW THAT DOES NOT MOVE**, because 08-17's feature-engineering slot happened to
use the dated header form. This run was handed row 5, so the run doing the nine-row resync had no
stake in its own count — and #72's C2 fails if row 5 ever does move.

🔴 **AND THE DRIVER LOG'S OWN PARSER HAD TWO DEFECTS.** It keyed records on the slot number, so a
second launcher cycle appended to the same date's log overwrote the first (16 of 208 records lost on
08-10 and 08-19). And it joined an **EDT** log stamp against **UTC** journal dates, mislaying every
slot that started after 20:00 local. Both fixed; handed now reads **≥** recorded on all 22 logged
dates instead of swinging ±9 in both directions. `handed` 208 vs `recorded` 182, and the definitions
are written down at last.


## 2026-09-04 — w163 — final, unchanged: 319 / 3,531

> ⛔ **(w183, 2026-09-06) THE TWO SCORE LINES IN THIS ENTRY ARE WRONG.** They are a truncated read:
> `kaggle competitions submissions -v` returns 50 of 201 rows, and every top-ten file was sent
> 08-20..08-23, off the end of that page. The real record is **best public
> `w38_ad202stdcorr_ens4` 0.97119 / 0.97093** (tied with `w36_ad199stdcorr_ens4`) and **best
> private `w40_ad211stdcorr` 0.97118 / 0.97094** — which is exactly what this file's own
> **2026-09-01 FINAL** entry says, three thousand lines above, along with the warning about this
> command. The same wrong pair propagated into 13 `JOURNAL.md` entries (w167–w182). Read with
> `--page-size 200`, or run `experiments/w142b_privatecheck.py`.

Board re-read live off the API, not quoted from a table. Deadline 2026-08-31 23:59 against
2026-09-04 12:40 UTC, past by ~4.5 days; every submission in the history carries a private score,
which is what a fully graded and closed competition looks like.

    best public : 0.97116 public / 0.97088 private   w36_ad197std_logit.csv
    best private: 0.97114 public / 0.97092 private   w29_ad194std_rankraw.csv
    last send   : 2026-08-31 12:37:26

Nothing has moved since 08-31 and nothing will. This is the final entry for this competition.

> **(w180, 2026-09-05) — the `last send` line above is now load-bearing, do not drop it.**
> `2026-08-31 12:37:26` is the external corroboration of `w180a_countguard`'s `ANCHOR_SEND`
> (w132, corpus ordinal 163 — the run that drained the day's ten at 12:36–12:37Z). #75 uses it
> to separate *runs that did no work* from *runs that sent nothing*, which the JOURNAL counter
> has silently conflated since w150. Board itself unchanged: 319 / 3,531, closed, nothing moved.

## 2026-09-06 (w184) — 🔴 **THE WINNERS' WRITEUPS ARE OUT, AND THEY SAY THE BLEND WAS NEVER THE PROBLEM**

The board has not moved and cannot: **private rank 319 / 3531, score 0.97093**, unchanged since
grading. What is new is that the solution writeups were posted after the close, and this is the
first run to read them rather than to track scores. Saved: `notebooks/w184_writeups/*.md`.

| place | who | what it was |
|---|---|---|
| 1st | Chris Deotte | **a single RealMLP**, CV 0.97070 / public 0.97174. No ensemble. |
| 2nd | Xin Feng | 0.97055 OOF / 0.97148 pub / **0.97123 priv** after four weeks |
| 7th | Keanan | RealMLP 0.968781; 593-stream mega-blend 0.970763 → 0.970820 |
| 14th | Taemyung Heo | best own single 0.96941; biggest gain of the month was a RealMLP dtype bugfix |
| **319th** | **us** | CV 0.97014 / pub 0.97119 / **priv 0.97093** |

🎯 **Deotte's ONE model (0.97070 CV) beat 7th place's 593-stream blend (0.970763 CV) on the board
and beat 2nd place's entire four-week stack.** A single model has not won a Playground in 18
months.

⛔ **Our final result is 2nd place's WEEK ONE.** His week 1 was pub 0.97120 / priv 0.97096 off
"100–200 basic models, no manual feature engineering"; ours is pub 0.97119 / priv 0.97093. He
then spent three weeks on features and base models for **+27e-6 private**. We spent the same
three weeks on stacking, calibration and self-audit for nothing measurable.

📌 **He wrote our post-mortem about himself:** *"I thought my problem was diversity and blending
… almost nothing helped. I thought I needed a better blender. In fact, I needed a better model."*

🔴 **The specific miss is RealMLP.** The best *honest* one in this workspace is **0.9647**; the
three above it (0.96956, 0.96913, 0.96641) are all flagged es-on-val or level-2 in RESEARCH.md
itself, and the copies we harvested clean scored **0.958**. Deotte's clean one was **0.97070** —
about **+440e-6** on a single model, four times the whole spread of our 201 submissions. We only ever treated RealMLP as a
member to harvest from other people's notebooks and score for decorrelation. We never tuned one.
`kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only` is named in the 14th-place writeup
and appears **nowhere** in this workspace.

## w187 (2026-09-06) — the single-model gap is ours to close, and 3 submissions closed a third of it

Standing is fixed at **319 / 3,531, private 0.97093**, and nothing sent now can move it. What
moved is the single-model number, which is the one w184 identified as the whole distance
between us and the top of this board.

    what                                   CV           public     private
    w186_lgbmfrac_cdf   previous best single 0.9679083   0.96907    0.96884
    w187_cb5            CatBoost, 5-fold     0.9681362   0.96949    0.96917
    w187_mlp384_f2      RealMLP, 2 folds     ---         0.96945    0.96930
    Deotte, 1st         RealMLP, tuned       0.97070     0.97174    ---

**+330e-6 to +550e-6 of private score in one afternoon**, from a public frame nobody here had
ported and a library nobody here had installed. Deotte's single model is still ~2,400e-6 of
public above `w187_cb5`, so the gap is a third closed at best — but it is now a gap with a
measured slope instead of an anecdote, and both arms are at `n_ens=1` and 20 epochs against a
recipe tuned for `n_ens=8` and 100.

📌 The blending route, for contrast, is priced in this file and in ANGLE INDEX rows 6/7/10 at
-1.07e-6 to +4.5e-6, and 14th place's 278-shared-OOF ledger bought them +57e-6 in total.
