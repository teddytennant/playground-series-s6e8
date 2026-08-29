"""Read which submissions are currently flagged as final selections.

Kaggle's public API has no *write* path for final-submission selection (probed and
falsified 2026-08-13, see RESEARCH.md), but it does expose a read: ListSubmissions
accepts group=SUBMISSION_GROUP_SELECTED. That gives a verification channel for the
one action this competition still needs a human for.

Usage:  .venv/bin/python experiments/check_selection.py
        (needs the kaggle CLI's OAuth credentials at $KAGGLE_CONFIG_DIR)

Exit status 1 if nothing is selected, so it can be used as a deadline alarm.
"""

import json
import os
import subprocess
import sys

COMP = "playground-series-s6e8"
HERE = os.path.dirname(os.path.abspath(__file__))
# The deadline pick, per JOURNAL.md.
#
# CHANGED 2026-08-16 by w16c's audit, the first move since 2026-08-13. First pick is now
# `w16b_cellweight` (CV 0.9700554, the workspace's best) instead of `blend159av_h3`
# (CV 0.9700492): on 500 paired private-slice draws it wins by +6.49e-6 +/- 0.20 with
# P(better) 0.912, worth ~1.5 places at the live board density. The second pick stays a
# ZERO-parameter file on purpose — it is the hedge against the whole fitted-correction
# family failing, and it costs only 0.10e-6 of E[max] against the unhedged optimum
# (w16b_cellweight + w15f_antistudent_avg), an order of magnitude under the 2e-6 stack
# reproducibility floor. See experiments/w16c_audit.json.
#
# CHANGED AGAIN 2026-08-16 by w16i/w16k (slot 3). First pick moves from `w16b_cellweight` to
# `w16i_schemeavg`. Same protocol, same seed 1616, same f 0.20, the same simulated private
# slice scored for every file each rep (experiments/w16k_pickcheck2.py):
#
#   w16b_cellweight  mean 0.97004390  +6.49e-6 +/-0.20  P(better than blend159av_h3) 0.912
#   w16f_armavg      mean 0.97004361  +6.21e-6 +/-0.16  P 0.954
#   w16i_schemeavg   mean 0.97004389  +6.48e-6 +/-0.15  P 0.974   <-- new first pick
#
# w16i matches w16b's mean to +0.01e-6 (P 0.498 head to head — the same file for decision
# purposes) with a tighter paired sd and the highest P(better) of any candidate. The reason to
# prefer it is the CV, not the slice: w16b's 0.9700556 carries TWO stacked selections — w16c
# measured +1.78e-6 of arm-selection optimism, and w16i measures a further +1.55e-6 of
# scheme-selection optimism — so w16b's honest CV is 0.9700536, below w16i's 0.9700557, which
# needs no correction because nothing inside it is chosen. The second pick stays a
# ZERO-parameter file for the same hedging reason as before.
#
# NOT CHANGED 2026-08-16 by w16q (slot 7), but read this before the click.
#
# (a) Both picks are on the h3 side of the h3/ens4 transform contrast, and w16q measured that
#     contrast paired within member set on all SIX sets where both mixes are scored. h3 is
#     ABOVE ens4 on CV in 6/6 (mean +4.32e-6) and BELOW it on the public LB in 6/6 (every one
#     rounds one 1e-5 reporting step down, so the true LB difference is in (-20e-6, 0)). The
#     workspace's standing position — "h3 and ens4 are not separable, do not claim either is
#     better" — came from a mix-gap estimator validated on ONE member set (150fx, error -7e-6)
#     and it does not survive six direct paired readings. See experiments/w16q_ens4base.py.
#     WANTED was NOT moved on it: that is an LB measurement, final selection here is on CV,
#     and h3 wins on CV. The price of switching the zero-parameter hedge to `blend159av` is
#     -4.30e-6 of CV. Recorded so the human making the click knows it is a choice.
#
# (b) The "2e-6 stack reproducibility floor" quoted in the w16c and w16i blocks above does NOT
#     apply to either of them. w14a measured it on one rebuild of the 159-member LOGISTIC
#     stack (singular lbfgs, condition number ~1e18). The hedge costs those blocks dismiss —
#     0.33e-6 and 0.77e-6 — are E[max] differences computed from FIXED stored OOF vectors with
#     no logistic refit anywhere in them. w16q verified the applicable floor for that layer
#     directly: `w16i_schemeavg` and `w16m_widegrid` are two independent runs of the same
#     computation and their OOF vectors differ by max abs 0.0, their CVs by 0.000e+00, and
#     0 of 296,302 test rows differ in rank. The floor there is EXACTLY ZERO. Those two hedge
#     costs are real, not noise; they are small, and both decisions stand on their own error
#     bars, but do not dismiss a future sub-2e-6 quantity "because it is under the floor".
# NOT CHANGED 2026-08-16 by w16s (slot 8) either, but this is where h3/ens4 gets SETTLED.
# w16q left it open. w16s added `w16q_ens4avg` and `blend159av` to w16k's 500-rep simulated
# private slice (seed 1616, f 0.20, same paired protocol; the seven old candidates reproduced
# w16k's means at exactly 0.0000e-6, 7/7). experiments/w16s_pickcheck3.py:
#
#   blend159av_h3  - blend159av      CV d +4.30e-6   P(h3 better on ONE private draw) 0.912
#   w16i_schemeavg - w16q_ens4avg    CV d +4.15e-6   P(h3 better on ONE private draw) 0.908
#
# So the CV margin is not a coin flip, and BOTH WANTED files are on the winning side.
#
# The public-LB counter-reading is fully explained and needs no train/test story. Scoring the
# SAME draws on a public-SIZED slice (59,260 rows vs 237,042) the draw sd rises from 2.98e-6 to
# 7.08e-6 and swamps the ~5e-6 signal: P(a public-sized slice REVERSES h3) = 0.244. w16q's 6/6
# is ONE observation, not six — nested near-identical files scored against the same fixed slice
# — so it is a p~0.24 event. And w16q bounded the true LB difference at (-20e-6, 0): this model
# puts 0.244 of its mass in exactly that window and 0 of 500 draws below -20e-6. Do not reopen.
#
# The CROSS-AXIS HEDGE was evaluated and declined. Both WANTED files being h3-side means they
# fail together if the transform axis flips; {w16i_schemeavg, blend159av} would hedge the
# correction family AND the transform at the same p23+p0 cost. Its E[max] is 0.97004390 against
# WANTED's 0.97004391 — a price of -0.009e-6, tiny but real (w16q: the floor here is ZERO), so
# the pre-registered rule ">= current" failed and the pick stands. Recorded because E[max] on
# train draws can only see the world where the CV ordering is RIGHT, which is not the world the
# hedge is for: it shows the hedge is nearly FREE, not that it is worthless.
# CHANGED 2026-08-17 by w21 (slot 7) — the first move of WANTED in fourteen slots, and the
# rule that moved it was written down before the number that triggered it existed
# (experiments/w21_prereg.txt §1, authored before w21a_ad187corr.py printed its CV).
#
# The pair {w16i_schemeavg, blend159av_h3} never encoded two particular files. It encoded:
#       slot 1 = the highest-CV object built on the pack
#       slot 2 = the ZERO-FITTED-PARAMETER hedge on the SAME pack
# w20 imported 22 vetted members and the pack changed. Both slots move together, to the
# 187-pack analogues of exactly those two roles:
#
#   w21_ad187corr   CV 0.9701068814   <- slot 1, the 5-arm scheme average on the 187 h3 base
#   w20_ad187_h3    CV 0.9701008150   <- slot 2, the zero-parameter hedge, same pack
#   w16i_schemeavg  CV 0.9700556663   <- SUPERSEDED, -51.2e-6
#   blend159av_h3   CV 0.9700491721   <- SUPERSEDED, -57.7e-6
#
# +51e-6 is an order of magnitude clear of every noise scale that governs a CV comparison on
# the frozen folds (w16q: the rebuild floor for objects built on a fixed stored base is
# EXACTLY ZERO; the 5e-5 figure quoted elsewhere is a public-SLICE gap and does not apply
# here). Decided on CV alone. The LB prints that followed — 0.97117 and 0.97116, a new
# account best — are recorded in the journal and were NOT an input to this.
#
# What is GIVEN UP: cross-base insurance. Both files are now one pack, so a defect in that
# pack hits both. Registered in advance and accepted on the grounds that the packs are
# NESTED — the 187 pack CONTAINS the 159 pack — and that all 22 new members cleared w20a's
# fold-signature gate and reproduce their published OOF AUC to <5e-9. The prior pair carried
# the identical exposure (both 159-pack) and it was never called a hedge then.
#
# w20's own written precondition named {w20_ad187_h3, w16i_schemeavg}, a MIXED-pack pair.
# That is not what shipped, and the reason is recorded rather than left silent: w21a did not
# exist when w20 wrote it, so w20's slot-1 candidate was the uncorrected file. With the
# correction rebuilt, the same rule that produced w20's sentence produces this pair instead.
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# 2026-08-17 (w24, slot 10). WANTED MOVES AGAIN, ON A PRE-REGISTERED RULE (w24_prereg §3 R1,
# written before any w24 number existed), and slot 2 is REPLACED, not just re-labelled:
#
#   w23_ad187stdcorr  CV 0.9701150809  <- slot 1, NEW. NOT YET SUBMITTED, see below.
#   w21_ad187corr     CV 0.9701068814  <- slot 2, demoted from slot 1, LB 0.97117
#   w20_ad187_h3      CV 0.9701008150  <- dropped from the pair, LB 0.97115
#
# The new file is the SAME CONSTRUCTION as the old slot 1 — the 5-arm scheme-average c_avg
# correction on an h3 mix of the 187-member pack, `w21a_ad187corr.py` run verbatim with
# W21A_BASE swapped — differing ONLY in that its base stack standardises the member columns
# before the meta-logistic. w23 isolated why that matters (lbfgs's tol=1e-4 stopping rule
# carries the columns' scale; the members span sd 1.82..27.59; the unstandardised fit
# terminates early on that slack) and w24 confirmed the correction still pays on the fixed
# base: +5.716e-6 cross-fitted, against +6.066e-6 on the same base unstandardised.
#
# Margin over the old slot 1: +8.20e-6, which is 4x the ~2e-6 floor that applies to an object
# containing 25 fresh `ascend` fits. R1 required >= +5.0e-6 and got it. Decided on CV; the new
# file has NO leaderboard print at all, which is the cleanest possible version of that rule.
#
# ⚠ SLOT 1 IS NOT SELECTABLE UNTIL IT IS SENT. Today's cap (10) was exhausted before this file
# existed, so `w23_ad187stdcorr.csv` has never been uploaded and Kaggle cannot select what it
# has not received. This script will therefore report slot 1 UNSATISFIED until the next day's
# first submission goes out — that is the intended state, not a fault. It is #1 in
# `experiments/w23b_sendqueue.csv`. SEND IT FIRST, THEN CLICK.
#
# Why slot 2 is `w21_ad187corr` and not the zero-parameter hedge `w20_ad187_h3`: the exposure
# worth insuring against changed. Both WANTED files were already one pack, so pack risk was
# never hedged by that pair; what IS new and uninsured is the standardisation itself, which is
# one wave old. `w21_ad187corr` is the identical object WITHOUT it, so the pair now hedges the
# only untested variable, at a cost of 6.0e-6 of CV against the alternative pairing.
# ─────────────────────────────────────────────────────────────────────────────────────────
# 2026-08-19 (w28, slot 9, consolidation). WANTED MOVES — and the reason it had to be moved
# HERE rather than in the journal is itself the finding: the journal moved this pair on
# 08-19 slot 6 and again on slot 7, three separate entries say `WANTED` = {w27_ad188stdcorr,
# w23_ad187stdcorr}, and THIS CONSTANT WAS NEVER EDITED. The script whose whole job is to
# verify the deadline pick has been verifying the wrong pair for three slots. Journal prose
# is not a variable. If the pick moves, it moves in this line, in the same commit.
#
#   w27_ad190stdcorr.csv   CV 0.9701181344   <- slot 1, the 190-member pack
#   w23_ad187stdcorr.csv   CV 0.9701150809   <- slot 2, the 187-member pack, ALREADY SENT
#
# ⚠ SLOT 1 IS CHOSEN ON A PAIRED CONTRAST, NOT ON THE STORED CV LEVELS, and that distinction
# is new. w27 slot 8 measured a ~4e-6 REPRODUCIBILITY FLOOR on absolute cross-fitted CV: the
# same code, same matrix, same folds, varying only BLAS thread count, spans 3.68e-6 and the
# full range against the shipped build is 5.22e-6 (`experiments/w27z2_threads.py`). Every CV
# in the top of this workspace's table was built in a separate process:
#
#   w27_ad190stdcorr 0.9701181344 / w27_ad188stdcorr 0.9701168076   difference +1.33e-6
#   w27_ad190stdcorr / w23_ad187stdcorr 0.9701150809                difference +3.05e-6
#
# BOTH are inside the floor. Ranking these three files by their stored levels is not a
# measurement. What IS a measurement is w27 slot 8's M11(b): the 190 pack against the 188
# pack, refitted IN ONE PROCESS on identical partitions, mean +2.13e-6, se 0.64e-6, positive
# on 4 of 4 partitions. Paired contrasts cancel the thread effect EXACTLY (w27z's A-B is
# 0.000e+00 to the last digit), so that is the only sound reason to prefer the 190 file, and
# it is the reason. Do NOT re-justify this pick with "the highest CV ever built here".
#
# Slot 2 is `w23_ad187stdcorr` on the standing w24 R1 logic: the hedge insures the NEWEST
# uninsured variable, which is now the member additions themselves (187 -> 188 -> 190, the
# lattice CT arms and `ext_members6`). `w23_ad187stdcorr` is the IDENTICAL construction --
# standardised h3 base, 5-arm c_avg scheme average -- on the 187 pack without them, and it
# has a leaderboard print (0.97116) while both w27 files have none. Cost against pairing the
# two w27 twins: -3.05e-6 of CV, i.e. inside the floor above, for a genuine pack hedge.
#
# ⚠⚠ SLOT 1 IS NOT SELECTABLE UNTIL IT IS SENT, and it has now been unsent for FOUR slots
# while three journal entries said "send it first". The cause was mechanical, not editorial:
# `w26g_send.py` sends the top of `w26d_queueprice.csv`, and until w28 that queue was stale
# (24 rows, built before any of these files existed) so the sender could not see them. Fixed
# in w28 by re-running `w23b_sendqueue.py` and by adding a `priority` column that pins
# whatever is in WANTED to the head of the queue. Slot 2 IS selectable now.
#
# ⚠⚠ MOVED 2026-08-20 by w34 (slot 5) — the first change to the CV-leader slot since w23,
# and the first CV improvement here in a week. `w27_ad190stdcorr` (0.9701181344) is replaced
# by `w34_ad195stdcorr` (0.9701247949), the same construction — standardised h3 base, 5-arm
# c_avg scheme average — on the 194-member pack PLUS `om_ftt`, the FT-Transformer imported
# from `omidbaghchehsaraei` in w33.
#
# WHY THIS CLEARS THE BAR THAT w29 DID NOT. w29 built `w29_ad194stdcorr` at +0.15e-6 on the
# leader and correctly refused to move WANTED: 0.15e-6 is 4% of the ~4e-6 rebuild
# reproducibility floor, and taking the argmax because it is the argmax is the failure this
# file exists to prevent. The 190-over-188 promotion was made on M11(b)'s 4/4 at +2.13e-6.
# This is +6.51e-6 — 1.6x the reproducibility floor and 3x the margin that promoted the 190
# — and it is corroborated by three INDEPENDENT instruments that were run in this order:
#
#   1. w26i paired 50/50, run BEFORE any build   om_ftt  +7.5e-6, sign stable 2/2 reps
#   2. the h3 base, 195 vs the matched 194        +6.57e-6  (0.9701205753 vs 0.9701140064)
#   3. the corrected 5-arm shipping object        +6.51e-6  (0.9701247949 vs 0.9701182875)
#
# and by all FOUR transform stacks moving the same way against their matched w29 controls
# (logit +4e-6, hybrid +5e-6, rankraw +5e-6, rescale +8e-6 — 4/4, no cancellation). w29's
# own rankraw-negative pattern, which made its h3 null a cancellation rather than an effect,
# does not appear here.
#
# ⚠ PRE-REGISTERED, written while ARM 2 was still fitting: `w34_ad196stdcorr` (the same pack
# plus `om_cat`) is a SIGN CHECK on om_cat, NOT a candidate for this slot. The paired
# instrument priced om_cat at -2.5e-6 with a stable sign in 2/2 reps before either build
# started, and at maxcorr 0.9935 against `bolt_cat_nested_te` it is the ninth CatBoost of
# its kind in the pack. If ARM 2 lands marginally ABOVE ARM 1 anyway, that is a ~2e-6
# difference between near-twins against a prior measurement that says negative, and taking
# it would be exactly the argmax-chasing this file forbids. WANTED does not move to it.
#
# The second pick is UNCHANGED. `w23_ad187stdcorr` remains the pack hedge — identical
# construction on the 187 pack without any of the member additions, and it has a real
# leaderboard print (0.97116) where none of the w27/w29/w34 files do. Pairing the two
# newest twins was declined again for the reason w29 gave: their test rho is ~0.99998, so
# there is no diversification in it.
#
# ⚠⚠ `w34_ad195stdcorr.csv` IS NOT SELECTABLE UNTIL IT IS SENT. It was built at cap on the
# 08-20 UTC day. It is pinned to the head of the send queue by the `priority` column added
# in w28, so the 08-21 day's FIRST slot sends it. Do not let it sit unsent — that mistake
# has already cost this file four slots once.
# w36 slot 6. Moved from w34_ad195stdcorr on the rule pre-registered in
# experiments/w36b_prereg.txt BEFORE the number existed: WANTED moves iff the new corrected
# object clears 0.9701288 = the old leader 0.9701247949 plus the ~4e-6 rebuild floor.
#   w36_ad199stdcorr  CV 0.9701400060   -> clears by +11.2e-6, beats the old leader by +15.21e-6
# That is 3.8x the rebuild floor and 2.3x the om_ftt promotion that moved this slot last time.
# The h3 base moved +14.85e-6 against its MATCHED 195-member control on disk, 4/4 across the
# transform stacks with no cancellation, and scheme-selection optimism came in at +0.000e-6
# with 5/5 nested stability. The second pick is UNCHANGED: w23_ad187stdcorr is the pack hedge,
# identical construction on the 187 pack, and the only one of the two with a real LB print.
WANTED = {"w36_ad199stdcorr.csv", "w23_ad187stdcorr.csv"}

# ---------------------------------------------------------------------------------------
# ⛔ SLOT 2 IS SETTLED AT A MEASURED SIZE — DO NOT RE-OPEN IT (w64, 2026-08-23)
# ---------------------------------------------------------------------------------------
# Slot 2 was deferred as "a PACK HEDGE whose value is not its CV" for THREE consecutive runs
# (w61 §8.7, w62 §7.7, w63 §10.8), each of which said the same sentence and made no argument.
# `experiments/w64a_hedgeprice.py` made it, with two instruments and a decision rule fixed in
# `w64_prereg.txt` before either was run. The answer is not "we still cannot tell". It is that
# THE WHOLE DECISION IS WORTH 0.14e-6, and here is each half of why.
#
# (1) E[max] ON THE FITTED PRIVATE POSTERIOR (w63a's estimator, imported, GATE A reproduces its
#     recorded price to 0.000e+00). The entire `*stdcorr` ladder — the same construction at packs
#     187/188/190/194/195/197/202/211 — spans **0.1447e-6 of E[max] while spanning 22.51e-6 of
#     CV**. Slot 1's mean dominates and every candidate correlates with it above 0.9995, so the
#     second slot contributes almost nothing whichever file fills it. The contrast three runs
#     deferred (`w23_ad187stdcorr` -> `w38_ad202stdcorr`) is **+0.1444e-6** — under the ~2e-6
#     rebuild floor and under the +0.45e-6/parameter search-optimism floor.
#
#     ⚠ AND THE HEDGE'S OWN MECHANISM IS MEASURED AND IS NEARLY EMPTY. Decorrelation is the ONLY
#     channel through which a hedge can pay in this instrument. corr(slot1, incumbent) = 0.999544
#     against corr(slot1, challenger) = 0.999973. Forcing the incumbent's correlation up to the
#     challenger's, holding its own mean and sd, moves its E[max] by **0.0003e-6 of the 0.1447e-6
#     the mean deficit costs — 0.2%**. The incumbent would need corr 0.998410 to break even and
#     it reads 0.999544. The hedge does not pay for itself; it is simply too small to matter.
#
# (2) THE STRUCTURAL HALF, which (1) is blind to by construction — a Gaussian posterior with one
#     common gap cannot represent "the member-addition family transfers worse than its CV".
#     Measured directly: GLS of the public-slice residual (LB - CV) on PACK SIZE over all 53
#     scored pack-tagged files, packs 187..211, family fixed effects, covariance = the
#     workspace's own `Sxx = St + Sp` plus the LB reporting-rounding variance (sd 2.887e-6),
#     intervals scaled by sqrt(max(chi2/dof, 1)) = 1.090.
#
#         delta = -0.0586 e-6 per member,  95% CI [-0.7673, +0.6501]
#         break-even (22.507e-6 of CV over 15 members) = -1.5005 e-6 per member
#
#     The interval excludes the break-even by 4.0 sigma. ⚠ AND THE NULL IS INFORMATIVE, because
#     the power control says so: injecting a slope AT the break-even into the same 53 residuals
#     is recovered to 5e-14 and detected at 2 sigma. A regression that could not see the effect
#     would report the same tight zero.
#
# (3) THE DECISION. w64_prereg §4 fixed an AND rule before the numbers: move iff E[max] prefers
#     the challenger AND the pack-transfer slope clears the break-even. Read (i) NO / (ii) YES,
#     so **WANTED DOES NOT MOVE**. ⚠ §4(i) was itself AMBIGUOUS — the prereg registered P2 as a
#     NUMBER (> +1.0e-6) and §4(i) as a DESCRIPTION ("reads positive"), and at +0.1444e-6 those
#     are different answers. That is w63 §3's own defect committed one run after recording it.
#     Both readings are in `w64a_hedgeprice.json` (`p2_strict` false, `p2_loose` true) and the
#     CONSERVATIVE one was taken, because picking the reading after seeing where it points is
#     exactly what the rule exists to stop.
#
# ⚠⚠ AND THE THREE DEFERRALS NAMED THE WRONG FILE. All three say "`w40_ad211stdcorr` is +22.4e-6
# above slot 2 on CV and eligible". `w38_ad202stdcorr` is +22.5e-6 — HIGHER by 0.116e-6 — and
# ARM 202 is ARM 211 minus exactly the nine `yadoy666` union94 streams (w61a GATE M), so it
# carries no `WANTED_RETIRED` record at all while ad211 carries one whose own text says "THIS IS
# NOT es-CLEARANCE". The candidate argued about for three runs is dominated on CV and on
# provenance simultaneously by its own matched control. If slot 2 is ever revisited, revisit it
# with `w38_ad202stdcorr`, not `w40_ad211stdcorr`.
#
# ⛔ WHAT WOULD RE-OPEN THIS, AND NOTHING ELSE WOULD. Both halves are conditional on the board:
#   (a) a slot-1 candidate appears whose mean is NOT dominant — the E[max] flatness above is a
#       consequence of slot 1 sitting above every candidate, and a genuine two-peak board would
#       make slot 2 worth several e-6 again; or
#   (b) the pack ladder extends far enough that 24 members of extrapolation is no longer the
#       whole design — delta's interval is +-0.65 e-6/member and 40 more members would make a
#       transfer penalty this small worth 26e-6.
# Neither is true today. See experiments/w64a_hedgeprice.py, w64a_hedgeprice.json, w64b_hedgeguard.py.

# ---------------------------------------------------------------------------
# WANTED-INELIGIBILITY, ENFORCED (w56, 2026-08-22)
# ---------------------------------------------------------------------------
# `w40d_prereg` adopted the rule and nothing ever checked it:
#
#     "an arm containing un-es-cleared streams may be built, priced, queued and SENT,
#      but is NOT ELIGIBLE for check_selection.WANTED on CV alone. Its LB reading is
#      the discriminator; its CV is not."
#
# That rule lived only in RESEARCH.md, which is exactly the failure mode w54 and w55
# each paid a run for on the send path: a rule that lives in a paragraph is not a rule.
# It matters here because the two highest CVs ever built in this workspace are BOTH
# ineligible under it, and a future run reading the CV ledger will find them at the top:
#
#   w50_ad216stdcorr  CV 0.9701500880  — the highest on disk, +10.1e-6 above WANTED
#   w42_ad217stdcorr  CV 0.9701788     — +38.8e-6 above the best sent, the only object
#                                        on disk inside the gold cut's range
#
# Both fail on evidence that is already in, not on suspicion:
#   ad216 — w51 read the source logs: four of the five `ext_members16` members it rests
#           on are es-on-val, and the fifth is a level-2 stack over the same public
#           columns. Our fold partition is BYTE-IDENTICAL to theirs, so the inflation
#           lands on exactly the rows this workspace scores.
#   ad217 — w56a read `hboyang/s6e8-150-member-fusion`'s notebook source: `hboyang_mix`
#           is an AGGREGATOR over 138 third-party streams from seven public OOF
#           libraries (16 of them lookup-transformer-family, the family w51 convicted
#           by source read), on the same fold partition. Its streams cannot be
#           es-cleared here, so w40d binds.
#
# ⚠ THE 08-23 READ DOES NOT LIFT THIS. `w48d_arm217.json` sends `hboyang_mix`'s raw test
# vector and reads its public score. A HONEST result (lb >= 0.97116) rules out GROSS
# contamination and nothing narrower — w56a measured the rejection region at 430e-6 of
# member AUC while the whole disputed effect needs far less — so HONEST licenses SENDING
# and pricing ad217 on the LB, never SELECTING it on CV. Selecting on a public read is
# the Rogii failure this account has already made once.
#
# ⚠⚠ ARM 211 WAS MISSING FROM THIS DICT FOR TWO DAYS — AND IT IS THE ARM THE RULE IS ABOUT
# (added w60, 2026-08-22). w56 wrote this dict to enforce w40d_prereg.txt, quotes w40d by name
# in the comment above, and then entered ad216 and ad217 — the two arms w56 happened to be
# looking at. ARM 211 is the arm w40d_prereg was WRITTEN about; its clause is the source of the
# sentence this dict exists to enforce, verbatim and pre-registered before ARM 211 existed:
#
#     ⛔ ARM 211 IS NOT ELIGIBLE FOR check_selection.WANTED, WHATEVER ITS CV.
#
# So the rule was lifted out of a paragraph and into code, and lost its originating case in
# transit. ⚠ THE LESSON IS NARROWER AND NASTIER THAN w54/w55/w56's "a rule in a paragraph is
# not a rule": when you MOVE a rule into an enforcing structure, enumerate its cases FROM THE
# RULE, not from the instances in front of you. A partially-populated enforcer reads exactly
# like a complete one and every guard it passes is evidence of nothing.
#
# What the omission was costing, all of it live at the time it was found:
#   * `w26g_send.above_tier_reason` imports this dict. ad211 files clearing the w59 CV bar were
#     waved through ABOVE the auto tier — installed as final entries w40d forbids.
#   * `w40_ad211stdcorr` (CV 0.9701374733) is ALREADY in auto-selection TIER 1 at p_joint 0.375,
#     the second-highest CV of any tier-1 member (w57a_tierprice2.json). Nothing can unsend it;
#     the entry stops the NEXT one.
#   * It outranks `w23_ad187stdcorr` (0.9701150808) on CV, so a future run moving WANTED's weaker
#     slot down the CV ledger lands on a barred arm and `assert_wanted_eligible` stays silent.
#
# ⛔ THE RETIREMENT TEST IS REGISTERED IN w60_prereg.txt AND WAS DELIBERATELY NOT RUN BY THE RUN
# THAT ADDED THIS KEY. ARM 202 is ARM 211's matched control and the corrected-h3 delta is
# -0.116e-6 — the nine `yadoy666` streams bought nothing, far inside the ~4e-6 rebuild floor and
# nowhere near w40d's +40e-6 "disbelieve on sight" line. That is the right SHAPE of evidence, and
# w60 had a direct interest in it (ARM 211 carried w59 §7's lever), so w60 enforced the rule and
# left the retirement to a run with no stake: |delta| <= 4e-6 on ALL FOUR transform bases held on
# disk (h3, ens4, rescale, rankraw), quoted here in the same commit. One base is one reading.
#
# To retire an entry: retire it HERE, in the same commit as the evidence, with the
# reading that retires it quoted in the value. Do not delete the key to make a run pass.
WANTED_INELIGIBLE = {
    "w50_ad216": "w51 source read — 4/5 ext_members16 members es-on-val on our own folds, "
                 "5th is a level-2 over the same columns. Not WANTED on CV (w40d).",
    "w42_ad217": "w56a source read — hboyang_mix is an aggregator over 138 un-es-clearable "
                 "third-party streams. Not WANTED on CV (w40d). The 08-23 LB read does "
                 "not lift this; see w56a_arm217power.json.",
    # w69 (2026-08-23). ARM 208 = ARM 211 minus ext_members14. Keyed BEFORE the arm was built
    # and BEFORE its CV existed (w69_prereg.txt §2.1, committed 39843c6). ⚠ THE POINT, and the
    # reason this key is not redundant with the retirement below: `w40_ad211` was retired on
    # ARM 211's OWN matched control, and the final clause of that retirement says in terms
    # that it does NOT carry to "a different pack, a different combiner, a re-weighting".
    # ARM 208 is a different pack, and deleting three columns re-weights every remaining one
    # INCLUDING the nine `yadoy666` streams the bar is about. So ARM 208 does not inherit ARM
    # 211's clearance — it inherits ARM 211's BAR, and needs its own control to lose it.
    "w69_ad208": "w69_prereg §2.1, keyed before the arm was built and before its CV existed "
                 "— ARM 208 is ARM 211 minus the three ext_members14 members, a DIFFERENT "
                 "PACK, so w40_ad211's retirement expressly does not carry over. Not WANTED "
                 "on CV (w40d), whatever its CV. Lifted only by its own (208 − 199) matched "
                 "control within ±4e-6 on all four criterion bases, quoted here in the same "
                 "commit, and not by the run that built the arm.\n\nRESOLVED 2026-08-23 by w70, the DISINTERESTED run, on the COMPLETE four-seed w69a_factorial.json (provisional=false, FAILURES 0). The (208 - 199) matched control E_B_lo reads h3 +1.056, ens4 +1.096, rescale -0.497, rankraw +4.840 e-6. rankraw 4.840 EXCEEDS the +-4e-6 bar, so the criterion as registered is NOT met and THIS KEY STAYS. Applied as written: the rule was registered before the arm was built and is not reinterpreted now that its number is known.\n\nWHAT THE READING ALSO SHOWS, recorded because it cuts the OTHER way and a later run must see it: the bar is a CLOSENESS bar and ARM 208 fails it on MAGNITUDE, not direction. E_B_lo is POSITIVE 4/4 seeds on h3 (t +3.67) and ens4 (t +5.85) -- ARM 208 measures ~1.1e-6 BETTER than ARM 199 in-process. That is the OPPOSITE SIGN to the between-file stdcorr ladder (w70 sec 3: 208 reads 0.87e-6 BELOW 199 on shipped CV), and the in-process paired contrast is the measurement while the ladder is not -- w68's 5.095e-6 floor is larger than the whole gap. Only rankraw is out of band, on 4 seeds at se 1.085.\n\nIF A LATER RUN WANTS THIS LIFTED the honest route is MORE SEEDS on rankraw, pre-registered before they are drawn -- not a re-reading of these four.",
}

# ---------------------------------------------------------------------------------------
# RETIRED ENTRIES — a bar that was lifted ON EVIDENCE, kept here so it cannot be re-derived
# from scratch and cannot be re-imposed (or ignored) without meeting the record. w61, 08-22.
# ---------------------------------------------------------------------------------------
# ⚠ A RETIRED KEY IS NOT A DELETED KEY. w60_prereg's retirement clause says the retiring
# reading must be "quoted in the dict value in the same commit" — if retirement meant removing
# the key there would be no value to quote it in, and `w60b_ineligguard`'s REVERSE half (every
# key backed by a registration) would read a silent deletion and a considered retirement
# identically. So the key MOVES here, carrying the original clause AND the measurement.
WANTED_RETIRED = {
    "w40_ad211": (
        "RETIRED 2026-08-22 by w61a_armctl.py (verdict RETIRE, FAILURES 0) on the four-base "
        "matched-control test registered in w60_prereg.txt and left unrun by the run that "
        "registered it. THE ORIGINAL BAR (w40d_prereg.txt, committed BEFORE ARM 211 existed): "
        "the nine `yadoy666` union94 streams come from an AGGREGATOR and their es-on-val "
        "status is UNKNOWN — neither log-read nor absent-by-mechanism; ⛔ not WANTED on CV "
        "whatever its CV. THE READING THAT RETIRES IT, ad211 minus ad202, cross-fitted CV "
        "recomputed from the raw OOF vectors on the frozen SKF5 seed-42 folds: h3 +0.163, "
        "ens4 +0.381, rescale +0.946, rankraw -2.353 e-6 — worst |delta| 2.353e-6 against the "
        "±4e-6 rebuild floor, all four bases inside it, and the sign is not uniform. Reported "
        "and NOT decided on (w61_prereg P5): hybrid -3.767, logit +0.257, corrected-h3 -0.116 "
        "e-6, also all inside. ARM 211 is ARM 202 plus exactly those nine streams and nothing "
        "else (w40f_run.sh vs w38d_run.sh; w61a GATE M asserts it), so the nine bought NOTHING "
        "as a group. An es-on-val member's OOF is INFLATED, the combiner UP-weights it and the "
        "stack CV RISES — inflation is a gain here, not a loss — so a null group delta bounds "
        "the inflation the bar exists to refuse. ⚠ THIS IS NOT es-CLEARANCE. The nine streams' "
        "es status is still unknown and w40d's reasoning is still correct about them; what is "
        "measured is that ARM 211's CV does not depend on them. If a future arm ADDS weight to "
        "these streams — a different pack, a different combiner, a re-weighting — this "
        "retirement does not carry over and the bar must be re-derived on that arm's own "
        "matched control. See experiments/w61a_armctl.json.\n\nTHE BAR ABOVE WAS UNRESOLVABLE BY ITS OWN INSTRUMENT, AND THE RETIREMENT SURVIVES ANYWAY. w68 measured the noise on a difference BETWEEN TWO SHIPPED FILES at 5.095e-6 -- LARGER than the +-4e-6 bar the four readings above are quoted against -- so w61a's 'all four inside' was partly a statement about a floor. Neither run could have seen it: w61 decided 08-22, w68 measured the floor 08-23.\n\nRE-DERIVED 2026-08-23 by w70, the DISINTERESTED run, from the COMPLETE four-seed w69a_factorial.json -- the SAME estimand as an IN-PROCESS PAIRED contrast (E_B_hi = A211 - A202, where the shared arm and most of the partition term cancel): h3 -0.594, ens4 +0.233, rescale -1.070, rankraw -1.800 e-6. Worst |delta| 1.800e-6, ALL FOUR BASES INSIDE +-4e-6 -- the same verdict, now on an instrument that resolves the bar. The design's own sd on h3 is 1.379e-6 (w69 P6, CONFIRMED), 3.7x sharper than the between-file floor, and w69's P9 power control injected +4.0e-6 and recovered it at 13.91 se, so this null is evidence of ABSENCE rather than of blindness.\n\nTHE RETIREMENT WAS RIGHT AND ITS ORIGINAL REASONING WAS NOT. The between-file readings are kept above rather than deleted, as the documented weaker basis. Everything else in this entry is UNCHANGED -- this is still not es-clearance, and the carry-over clause above still binds; it is exactly what keyed w69_ad208."
    ),
}


def assert_wanted_eligible(wanted=WANTED):
    """Fail loudly if WANTED names an arm that w40d bars from CV-based selection.

    A RETIRED bar does not refuse — that is what retiring it on evidence bought — but it does
    NOT go quiet either. w61 lifted the ARM 211 bar on a matched-control measurement whose own
    value says it is not es-clearance, and the file it would let into WANTED
    (`w40_ad211stdcorr`, CV 0.9701374733) outranks WANTED's current slot 2 on CV. So any run
    that puts a formerly-barred arm in WANTED has to see the record it is standing on.
    """
    bad = [(w, why) for w in sorted(wanted)
           for pat, why in WANTED_INELIGIBLE.items() if w.startswith(pat)]
    if bad:
        print("\n⛔⛔ WANTED NAMES A w40d-INELIGIBLE ARM — REFUSING TO PROCEED")
        for w, why in bad:
            print(f"   {w}\n     {why}")
        print("   Retire the WANTED_INELIGIBLE entry on evidence, in the same commit,")
        print("   or pick a different file. Do not edit this check to make a run pass.")
        raise SystemExit(3)
    lifted = [(w, why) for w in sorted(wanted)
              for pat, why in WANTED_RETIRED.items() if w.startswith(pat)]
    for w, why in lifted:
        print(f"\n⚠ NOTICE — WANTED names {w}, an arm whose ineligibility bar was RETIRED.")
        print("   Read the record before treating its CV as ordinary:")
        print("   " + why[:300] + "...")


assert_wanted_eligible()

# kagglesdk lives in the CLI's own uv tool venv, not in .venv.
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"

# ----------------------------------------------------------------------------------------
# HISTORY, QUARANTINED BEHIND `--history` (w114, 2026-08-29).
#
# Every line below is the 08-16/08-17 click narration, lifted VERBATIM out of the
# NOTHING-IS-SELECTED branch it used to print unconditionally. It was moved because it is
# not merely stale, it is ACTIONABLY WRONG: it contains the sentence "WANTED is now
# {w21_ad187corr.csv, w20_ad187_h3.csv}" inside a green-tick RESOLVED box, three lines below
# the correct `Wanted:` line, and it annotates `w16i_schemeavg` / `blend159av_h3` as "current
# WANTED". WANTED moved off all four on 08-19 (w28) and 08-21 (w36b). All four are SENT, so
# Kaggle will offer them and the wrong click is REACHABLE.
#
# `w114a_misclick.py` prices those two reachable wrong pairs on w74a's own estimator
# (GATE R reproduces w74a's headline to 0.000e+00):
#
#     click {w21_ad187corr, w20_ad187_h3}      +35.17e-6   =  7.8x
#     click {w16i_schemeavg, blend159av_h3}    +81.92e-6   = 18.1x
#     ...against NOT CLICKING AT ALL, which is +4.5228e-6.
#
# 🎯 THE INSTRUMENT THAT EXISTS TO PREVENT A 4.5e-6 ERROR WAS ITSELF THE LARGEST REACHABLE
#   SOURCE OF A 35-82e-6 ONE. The prices below are kept for the record and are NOT the live
#   number; the live number is printed by the actionable block, out of the JSON artefacts.
# ----------------------------------------------------------------------------------------
CLICK_HISTORY = """\
REPRICED FOUR TIMES on 2026-08-16, most recently by w16w (slot 10). The history
matters only as a warning that this number does not keep: w15i's
+9.2/+36.5/+112e-6 ladder went stale when slot 7 pushed blend158_logit out of
both tiers; w16s's -0.24 to +2.14e-6 range went stale within one slot; w16u's
'DETERMINED +3.326e-6' went stale within two. Slot 10's w16e_aonly also scored
0.97108, so auto-slot 1 is a THREE-way tie and at limit 2 the auto-pick is
ambiguous again — a range over the undocumented tiebreak. But w16e_aonly is
H3-side while the other two are ens4-side twins on the same c_avg, so two of
the three branches now mix transforms instead of pairing correlated files.
Repriced on the same 500 paired draws, gated at 0.000e-12 (w16w_reprice.json):
  limit 2: cost of NOT clicking +0.831e-6 (aonly+cellens4, h3+ens4)
                               +0.936e-6 (aonly+ens4avg,   h3+ens4)
                               +3.326e-6 (ens4avg+cellens4, ens4+ens4)
  limit 1: +1.266e-6 (w16e_aonly) / +4.072e-6 (w16q_ens4avg) / +4.162e-6 (w16t)
~2.5 places per 1e-5 at the local density.
  Still worth clicking. Note WHY:
  auto-slot 1 is held by the best PUBLIC score, which is one of the WEAKEST
  corrected files on CV — auto-selection is the Rogii failure run by Kaggle.
  The E[max] figures above UNDERSTATE the click: WANTED spends its second slot
  on a zero-parameter file as insurance against the whole corrected family
  failing, and a simulation drawn from train rows cannot price that.
  ⚠ REPRICED A FIFTH TIME (w17d/w17g, 08-17 slot 2) — the figures above are the
  UNCONDITIONED ones and they are the low end. w16w reads each file's private
  AUC over pseudo-test draws, so E[private] = CV by construction and WANTED wins
  automatically; it never conditions on the public scores we ACTUALLY SAW, which
  are the entire reason the auto-pick is what it is. Public and private partition
  ONE test set, so a file inflated on public gives some of it back on private.
  Measured on 6,000 draws, gated at 0.000e-12 against w16w: the coupling slope is
  -0.2477 (exact partition -0.2500; AUC is not additive over a partition so this
  had to be measured) and the variance split is w 0.1238 against 0.125 predicted
  from row counts. w < f = 0.20 means conditioning makes the click MORE expensive.
  Conditioned, limit 1: +2.306e-6 (w16e_aonly) / +5.417e-6 (w16q_ens4avg)
                        / +5.562e-6 (w16t_cellens4).   [w17g_parametric.json]
  THE MAGNITUDE IS NOT THE POINT — a couple of e-6 is ~0.6-1.4 board places. The
  CERTAINTY is: P(the auto-pick beats the CV pick on the private slice) is
  0.041 / 0.033 / 0.061, NOT ~0.5. This workspace has been carrying the click as
  a hedge against a coin flip and it is not one. (w17d's exact non-parametric
  GLOBAL branch agrees on direction but has ESS 14.3 of 6,000 draws — do not
  quote its +1.827..+6.727e-6; PERFAM, ESS 100, gives +1.704..+4.185e-6.)
  ⚠ REPRICED A SIXTH TIME AND THIS ONE HAS NO SIMULATION IN IT AT ALL
  (w18a, 08-17 slot 4). The conditioning is linear-Gaussian and LAW-IF gives
  its whole covariance in closed form, so it is a GLS solve, not a resampling
  problem. Gate: LAW-IF reproduces w17d's 6,000-draw sigma_t/sigma_p over 15
  pairs at median error 0.51%, max 2.58%. The pseudo-test draw and the public
  slice draw are the SAME operation at two sizes, so S_t and S_p are EXACTLY
  proportional (0.142855576 on all 36 entries, spread 1e-12) and M is a scalar
  times I to 1.1e-10 — conditioning on all six files jointly is PROVABLY the
  same as conditioning each contrast on itself. w17g's 'approximation' was the
  exact answer; the GLOBAL/PERFAM bracket collapses; the ESS-14 number was
  estimating something closed-form.  limit 1: +2.328e-6 (w16e_aonly) /
  +5.509e-6 (w16q_ens4avg) / +5.666e-6 (w16t_cellens4);  limit 2: +1.977 ..
  +4.772e-6.  P(the auto file beats BOTH wanted files privately) 0.043 /
  0.031 / 0.058.   [w18a_lawif_joint.json]
  ⚠ (w18b claimed those were over-sharp, on blend159av_rankraw printing z -3.58
  against a family sd estimated on n=4. RETRACTED by w19a/w19b — that sd was the
  artefact, not the print. Fitting the per-file transfer sd tau by composite ML
  over all 194 within-family pairs of the 51 OOF-carrying sent files, with
  LAW-IF paired sds and EXACT triangular grid rounding, gives tau_hat 0.000e-6,
  95% profile upper 1.72e-6; and a 500-draw parametric bootstrap of the whole
  51-file system from the perfectly-calibrated null reproduces every statistic
  that looked like structure — NONE of 11 escapes its own null's central 95%.
  The paired instrument is CALIBRATED.)
  ⚠ BUT THE CLICK IS NOT ROBUST, AND THAT IS THE REAL CORRECTION (w19c/w19d,
  08-17 slot 5). Everything above is conditional on tau = 0 EXACTLY. tau is a
  free parameter the board cannot pin: at TEST-level tau = 1.72e-6, inside
  w19a's own 95% interval, the w16e_aonly branch goes +2.328 -> -1.493e-6 and
  P(auto beats both wanted) 0.043 -> 0.733. A SIGN FLIP. The mechanism is what
  the click has always been: the auto-pick holds public slot 1 on a 30e-6
  public lead against a 5.3e-6 CV lead, and tau = 0 is the only assumption
  under which that excess is 100% slice noise. Worse, the public board can
  NEVER identify the test/public split of tau — both add tau^2 to the same
  observed variance — so no future submission resolves this.
  MARGINALISED over the tau posterior and a flat prior on that split:
      E[cost of not clicking]  +1.890 / +5.300 / +5.562e-6
      P(the click LOSES)        0.037 /  0.000 /  0.000
      E[P(auto beats both)]     0.128 /  0.044 /  0.067
  STILL CLICK — it is the right side on every branch in expectation. But the
  0.043/0.031/0.058 above is the tau=0 CORNER, not the answer, and the phrase
  'not a coin flip' is no longer defensible for the w16e_aonly branch.
   [w19a_transfer.json, w19b_calib.json, w19c_clicksens.json, w19d_taupost.json]
  ############################################################
  ## ✅ RESOLVED 2026-08-17 (w21 slot 7). WANTED HAS MOVED.  ##
  ############################################################
  WANTED is now {w21_ad187corr.csv, w20_ad187_h3.csv}. The block below
  is the slot-6 supersession notice, KEPT because its preconditions are
  what authorised the move — but its closing recommendation is now
  SUPERSEDED IN TURN and must not be acted on. It named a MIXED-pack pair
  {w20_ad187_h3, w16i_schemeavg} because w21a did not exist when it was
  written; it also said 'or the corrected rebuild of the former, if it
  lands higher', and that rebuild landed higher:
      w21_ad187corr   CV 0.9701068814  LB 0.97117  <- slot 1, NEW BEST
      w20_ad187_h3    CV 0.9701008150  LB 0.97115  <- slot 2, 0-param hedge
      w20_ad187       CV 0.9700978895  LB 0.97116  (sent, not a pick)
      w16i_schemeavg  CV 0.9700556663  LB 0.97107  <- superseded, -51.2e-6
      blend159av_h3   CV 0.9700491721  LB 0.97105  <- superseded, -57.7e-6
  The move is on CV, by w21_prereg.txt §1's rule, written before the CV
  existed. The LB prints are recorded and were NOT an input.
  ############################################################
  ## (slot-6 notice follows, for the record)                 ##
  ############################################################
  w20 (08-17 slot 6) imported 22 members from adarsh1077's OOF library,
  vetted by a new fold-signature gate (experiments/w20a_foldgate.py).
  The resulting 187-member stack is the highest-CV object this workspace
  has ever produced, by a margin an order of magnitude clear of every
  scale in play (2e-6 rebuild floor, ~6e-6 within-family CV steps):
      w20_ad187_h3      CV 0.9701008150   LB 0.97115  <- new account best
      w20_ad187         CV 0.9700979      (not sent)
      w20_ad187_rankraw CV 0.9700915300   LB 0.97114
      w16i_schemeavg    CV 0.9700556663   LB 0.97107  <- current WANTED
      blend159av_h3     CV 0.9700491721   LB 0.97105  <- current WANTED
  i.e. the new h3 file is +45.2e-6 on the pick and +51.6e-6 on the
  insurance file, and it dominates blend159av_h3 on the SAME criterion
  that file was chosen for (zero fitted parameters).
  WANTED WAS NOT MOVED IN SLOT 6, AND THAT IS A PRE-REGISTERED HOLD, NOT
  AN OPINION (experiments/w20g_shipprereg.txt, written before the upload).
  The two stated preconditions are now BOTH MET:
    (1) w20d paired attribution finished: +47e-6 +/- 9 over 3 splits,
        sign-consistent, so the CV gain is the adarsh members and not
        base drift from 159 -> 165 members;
    (2) the c_avg/scheme-average correction (+6.4e-6, w16i) has NOT been
        rebuilt on the new base — which can only RAISE the new file.
  => THE NEXT SLOT SHOULD SET WANTED = {w20_ad187_h3.csv, w16i_schemeavg.csv}
     (or the corrected rebuild of the former, if it lands higher), keeping
     the second slot on the old corrected file as cross-base insurance.
  ⚠ AND THE CLICK HAS LARGELY COLLAPSED AS A PROBLEM. The entire w16-w19
  click analysis priced the risk that Kaggle's auto-pick takes a
  PUBLIC-INFLATED file over the CV pick. After slot 6 the best public file
  IS the best CV file (0.97115, held alone), and auto-slot 2 is the CV #3
  file at 0.97114. Re-price before quoting w19d's +1.890/+5.300/+5.562e-6:
  those numbers describe a board state that no longer exists.
  ⚠ note (w17a/w17b, 08-17 slot 1) — CORRECTS the w16w note below it.
  The ladder is NOT dead. Against a correct PAIRED null (500 draws of the
  leaderboard's own geometry, 46 sent files with stored OOF vectors), dLB
  tracks dCV within transform family at Spearman +0.850, sign agreement
  53/57 = 0.930, z +6.49. Among the tight families (h3, h3+corr, ens4,
  ens4+corr, w, rankraw, rescale) the observed rms(dLB - dCV) is 7.4e-6
  against 9.4e-6 predicted by slice draw plus grid rounding ALONE — ratio
  0.79, i.e. no residual mechanism. What slot 10 read as a broken law is ONE
  file 2.2 sd high: w16e_aonly vs w16i_schemeavg is dCV -1.24e-6 against
  dLB +10.0e-6 on a predicted sd of 5.2e-6. What DOES survive from w16w:
  never quote 'LB = CV + 0.0010143' to 1e-6 — the pair sd is 5-9e-6, i.e.
  +/- one grid step, so a CV difference under ~5e-6 is a coin flip on the
  public slice and one over ~15e-6 is resolved.
  ⚠ AND IT SHARPENS THIS CLICK. Ranking all 46 files by mean standardised
  residual (positive = public score HIGH for its CV) puts Kaggle's three
  auto-slot-1 holders at the top of the tight families: w16q_ens4avg +1.20,
  w16e_aonly +1.12, w16t_cellens4 +1.07, against w16i_schemeavg +0.26 and
  blend159av_h3 -0.78. Selecting on public score makes that partly circular
  and the decomposition is the non-circular part: w16e_aonly leads
  blend159av_h3 by 30e-6 on public but only 5.3e-6 on CV, so ~83% of the
  lead is slice-specific. Auto-selection picks the three most inflated files.
  (superseded) note (w16w, slot 10): claimed the ladder FALSIFIED and not
  monotone; read above. WANTED is unaffected either way — it is chosen on
  CV, and w16e_aonly's CV is below w16i_schemeavg's on both conventions.
  note (w16s): both wanted files are h3-side and that is now SETTLED, not
  deferred. h3 beats ens4 on CV with P 0.912 on a single private-sized draw;
  the public slice disagrees, but a public-SIZED slice reverses a true h3
  advantage 24% of the time and w16q's 6/6 is one draw. WANTED follows CV."""


SNIPPET = r"""
import json, os
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.competitions.types.competition_enums import SubmissionGroup

cfg = os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")
tok = json.load(open(os.path.expanduser(cfg + "/credentials.json")))["access_token"]

out = {}
for name, g in (("selected", SubmissionGroup.SUBMISSION_GROUP_SELECTED),
                ("successful", SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL)):
    with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
        r = ApiListSubmissionsRequest()
        r.competition_name = %r
        r.group = g
        r.page_size = 200   # ⚠ was 50; see the pagination note below (w17, 08-17 slot 2)
        resp = c.competitions.competition_api_client.list_submissions(r)
        out[name] = [(s.ref, s.file_name, s.public_score) for s in resp.submissions]
print(json.dumps(out))
""" % COMP


def _price_lines():
    """The live click price, out of the JSON artefacts — never a literal in this file.

    w74a_clickprice.json is the eighth pricing of "what does it cost not to click"; the
    seven before it all went stale on a board move and w45a's hard-coded tier is why the
    number is read rather than typed. w114a_misclick.json is the first pricing of the
    OTHER failure, clicking the wrong pair.
    """
    out = []
    try:
        w74 = json.load(open(os.path.join(HERE, "w74a_clickprice.json")))
        out.append(f"  cost of NOT clicking at all      {float(w74['cost_auto_pair']):+.4f}e-6"
                   f"   (tau=0; {float(w74['cost_at_tau_upper']):+.4f}e-6 at the 95% upper tau)")
    except (OSError, KeyError, ValueError):
        out.append("  ⚠ w74a_clickprice.json unreadable — the click price is UNKNOWN, not zero.")
    try:
        w114 = json.load(open(os.path.join(HERE, "w114a_misclick.json")))
        for pair, rec in sorted(w114["misclick"].items(), key=lambda kv: kv[1]["cost"]):
            out.append(f"  cost of clicking {pair:<38} {float(rec['cost']):+8.2f}e-6"
                       f"   = {float(rec['ratio']):.1f}x that")
        out.append("  ⚠ THE MIS-CLICK IS THE BIGGER HAZARD BY AN ORDER OF MAGNITUDE. Select on")
        out.append("    the two lines above. Do NOT select on the highest public score, and do")
        out.append("    NOT follow a file name out of `--history`.")
    except (OSError, KeyError, ValueError):
        out.append("  ⚠ w114a_misclick.json unreadable — run experiments/w114a_misclick.py.")
    return out


def main() -> int:
    proc = subprocess.run([KAGGLE_PY, "-c", SNIPPET], capture_output=True, text=True)
    if proc.returncode != 0:
        print("API call failed:\n" + proc.stderr.strip())
        return 2
    data = json.loads(proc.stdout)

    # The successful group is the control: if it is empty too, the call is broken
    # rather than the selection being empty, and the distinction matters.
    n_ok = len(data["successful"])
    selected = data["selected"]
    print(f"control: {n_ok} successful submissions visible")
    print("  ⚠ PAGINATION (w17 slot 2, 2026-08-17). This request now asks for page_size 200.")
    print("  It asked for 50 until this slot, and the account passed 50 submissions on 08-17,")
    print("  so the count above was silently a PAGE LENGTH and the tier computation below was")
    print("  reading a truncated list. The CLI path (`kaggle competitions submissions -v`) has")
    print("  the same default and hid stack_pub74_logit and stack_pub88_mine_logit from every")
    print("  API-reading script here. Both score 0.97081, so no tier ever moved — but COUNT the")
    print("  number above against `kaggle competitions submissions -v --page-size 200 | wc -l`")
    print("  before quoting it, and do not trust any list you did not ask a page size for.")
    if n_ok == 0:
        print("CONTROL FAILED — treat the selection reading below as unknown.")
        return 2

    # SELECTABILITY. Kaggle's dialog lists SUBMITTED entries only, so a WANTED file with no
    # submission behind it cannot be ticked no matter how good its CV is. This block exists
    # because `w27_ad188stdcorr` spent four slots as "send this first" in the journal while
    # nothing on this box ever checked whether it had actually gone out (w28).
    sent_names = {fn for _, fn, _ in data["successful"]}
    unsent = sorted(WANTED - sent_names)
    print("\nWANTED, and whether Kaggle can even offer it:")
    for w in sorted(WANTED):
        print(f"  {'SENT     ' if w in sent_names else 'NOT SENT '} {w}")
    if unsent:
        print(f"  ⚠ {len(unsent)} WANTED file(s) have never been submitted and are therefore")
        print("    NOT SELECTABLE. They are pinned to the head of experiments/w26d_queueprice.csv")
        print("    and `w26g_send.py` will send them first; nothing else can fix this.")

    if not selected:
        print(f"\n*** NOTHING IS SELECTED for {COMP}. ***")

        # ------------------------------------------------------------------ WHAT TO CLICK
        # Refs and scores are read out of the LIVE list above. Nothing here is a literal:
        # a hard-coded ref in this block is exactly the defect w114 removed from the notes.
        def _f(sc):
            try:
                return float(sc)
            except (TypeError, ValueError):
                return float("-inf")
        by_name = {fn: (ref, sc) for ref, fn, sc in data["successful"]}
        print("\n" + "=" * 78)
        print("  CLICK EXACTLY THESE TWO, AND NOTHING ELSE")
        print("=" * 78)
        print(f"  https://www.kaggle.com/competitions/{COMP}/submissions")
        print('  -> "Use for Final Score" on:')
        for w in sorted(WANTED, key=lambda k: -_f(by_name.get(k, (None, None))[1])):
            ref, sc = by_name.get(w, ("NOT SENT", None))
            print(f"       {str(ref):<12} {w:<26} public {sc}")
        print("  Both refs and both public scores are read live from the list above; this")
        print("  file hard-codes only the WANTED filenames themselves.")
        print("=" * 78)
        for line in _price_lines():
            print(line)

        # ------------------------------------------------------------------ THE AUTO-PICK
        print("\nIf nobody clicks, Kaggle auto-selects by best PUBLIC score. The two tiers it")
        print("would draw from, computed live rather than quoted from a stale journal entry:")
        scored = [(sc, fn) for _, fn, sc in data["successful"] if sc is not None]
        tiers = sorted({sc for sc, _ in scored}, reverse=True)[:2]
        for rank, sc in enumerate(tiers, start=1):
            members = sorted(fn for s, fn in scored if s == sc)
            print(f"  auto-slot {rank}: public {sc}, {len(members)}-way tie — "
                  + ", ".join(m.replace(".csv", "") for m in members))
        # The one file that must never reach a slot. w15j enumerated six tiebreak rules
        # and it is uniquely selected under none of them, but the exposure is real.
        risk = [fn for sc, fn in scored if fn == "blend158_logit.csv" and sc in tiers]
        if risk:
            tier = 1 + tiers.index(next(sc for sc, fn in scored if fn == risk[0]))
            print(f"  ** blend158_logit (CV 0.969961, ~88e-6 below the CV pick) is in "
                  f"auto-slot {tier}'s tie. **")

        if "--history" in sys.argv:
            print("\n" + CLICK_HISTORY)
        else:
            print("\nThe 08-16/08-17 click narration is behind `--history`. ⚠ It names four")
            print("  files as WANTED that have not been WANTED since 08-19/08-21, and acting")
            print("  on those names costs +35.2e-6 or +81.9e-6 (w114a_misclick.json) against")
            print("  the +4.5e-6 this whole click exists to save. READ IT AS HISTORY ONLY.")
        return 1

    print("\nselected:")
    for ref, fname, score in selected:
        mark = "OK " if fname in WANTED else "!! "
        print(f"  {mark}{ref}  {fname}  public={score}")
    got = {f for _, f, _ in selected}
    if got != WANTED:
        print(f"\nMISMATCH — wanted {sorted(WANTED)}, got {sorted(got)}")
        return 1
    print("\nSelection matches the deadline pick.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
