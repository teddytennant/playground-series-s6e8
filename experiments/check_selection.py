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
WANTED = {"w34_ad195stdcorr.csv", "w23_ad187stdcorr.csv"}

# kagglesdk lives in the CLI's own uv tool venv, not in .venv.
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"

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
        print("Kaggle will then auto-select by best PUBLIC score. The two tiers it would")
        print("draw from, computed live rather than quoted from a stale journal entry:")
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
        print("REPRICED FOUR TIMES on 2026-08-16, most recently by w16w (slot 10). The history")
        print("matters only as a warning that this number does not keep: w15i's")
        print("+9.2/+36.5/+112e-6 ladder went stale when slot 7 pushed blend158_logit out of")
        print("both tiers; w16s's -0.24 to +2.14e-6 range went stale within one slot; w16u's")
        print("'DETERMINED +3.326e-6' went stale within two. Slot 10's w16e_aonly also scored")
        print("0.97108, so auto-slot 1 is a THREE-way tie and at limit 2 the auto-pick is")
        print("ambiguous again — a range over the undocumented tiebreak. But w16e_aonly is")
        print("H3-side while the other two are ens4-side twins on the same c_avg, so two of")
        print("the three branches now mix transforms instead of pairing correlated files.")
        print("Repriced on the same 500 paired draws, gated at 0.000e-12 (w16w_reprice.json):")
        print("  limit 2: cost of NOT clicking +0.831e-6 (aonly+cellens4, h3+ens4)")
        print("                               +0.936e-6 (aonly+ens4avg,   h3+ens4)")
        print("                               +3.326e-6 (ens4avg+cellens4, ens4+ens4)")
        print("  limit 1: +1.266e-6 (w16e_aonly) / +4.072e-6 (w16q_ens4avg) / +4.162e-6 (w16t)")
        print("~2.5 places per 1e-5 at the local density.")
        print("  Still worth clicking. Note WHY:")
        print("  auto-slot 1 is held by the best PUBLIC score, which is one of the WEAKEST")
        print("  corrected files on CV — auto-selection is the Rogii failure run by Kaggle.")
        print("  The E[max] figures above UNDERSTATE the click: WANTED spends its second slot")
        print("  on a zero-parameter file as insurance against the whole corrected family")
        print("  failing, and a simulation drawn from train rows cannot price that.")
        print("  ⚠ REPRICED A FIFTH TIME (w17d/w17g, 08-17 slot 2) — the figures above are the")
        print("  UNCONDITIONED ones and they are the low end. w16w reads each file's private")
        print("  AUC over pseudo-test draws, so E[private] = CV by construction and WANTED wins")
        print("  automatically; it never conditions on the public scores we ACTUALLY SAW, which")
        print("  are the entire reason the auto-pick is what it is. Public and private partition")
        print("  ONE test set, so a file inflated on public gives some of it back on private.")
        print("  Measured on 6,000 draws, gated at 0.000e-12 against w16w: the coupling slope is")
        print("  -0.2477 (exact partition -0.2500; AUC is not additive over a partition so this")
        print("  had to be measured) and the variance split is w 0.1238 against 0.125 predicted")
        print("  from row counts. w < f = 0.20 means conditioning makes the click MORE expensive.")
        print("  Conditioned, limit 1: +2.306e-6 (w16e_aonly) / +5.417e-6 (w16q_ens4avg)")
        print("                        / +5.562e-6 (w16t_cellens4).   [w17g_parametric.json]")
        print("  THE MAGNITUDE IS NOT THE POINT — a couple of e-6 is ~0.6-1.4 board places. The")
        print("  CERTAINTY is: P(the auto-pick beats the CV pick on the private slice) is")
        print("  0.041 / 0.033 / 0.061, NOT ~0.5. This workspace has been carrying the click as")
        print("  a hedge against a coin flip and it is not one. (w17d's exact non-parametric")
        print("  GLOBAL branch agrees on direction but has ESS 14.3 of 6,000 draws — do not")
        print("  quote its +1.827..+6.727e-6; PERFAM, ESS 100, gives +1.704..+4.185e-6.)")
        print("  ⚠ REPRICED A SIXTH TIME AND THIS ONE HAS NO SIMULATION IN IT AT ALL")
        print("  (w18a, 08-17 slot 4). The conditioning is linear-Gaussian and LAW-IF gives")
        print("  its whole covariance in closed form, so it is a GLS solve, not a resampling")
        print("  problem. Gate: LAW-IF reproduces w17d's 6,000-draw sigma_t/sigma_p over 15")
        print("  pairs at median error 0.51%, max 2.58%. The pseudo-test draw and the public")
        print("  slice draw are the SAME operation at two sizes, so S_t and S_p are EXACTLY")
        print("  proportional (0.142855576 on all 36 entries, spread 1e-12) and M is a scalar")
        print("  times I to 1.1e-10 — conditioning on all six files jointly is PROVABLY the")
        print("  same as conditioning each contrast on itself. w17g's 'approximation' was the")
        print("  exact answer; the GLOBAL/PERFAM bracket collapses; the ESS-14 number was")
        print("  estimating something closed-form.  limit 1: +2.328e-6 (w16e_aonly) /")
        print("  +5.509e-6 (w16q_ens4avg) / +5.666e-6 (w16t_cellens4);  limit 2: +1.977 ..")
        print("  +4.772e-6.  P(the auto file beats BOTH wanted files privately) 0.043 /")
        print("  0.031 / 0.058.   [w18a_lawif_joint.json]")
        print("  ⚠ (w18b claimed those were over-sharp, on blend159av_rankraw printing z -3.58")
        print("  against a family sd estimated on n=4. RETRACTED by w19a/w19b — that sd was the")
        print("  artefact, not the print. Fitting the per-file transfer sd tau by composite ML")
        print("  over all 194 within-family pairs of the 51 OOF-carrying sent files, with")
        print("  LAW-IF paired sds and EXACT triangular grid rounding, gives tau_hat 0.000e-6,")
        print("  95% profile upper 1.72e-6; and a 500-draw parametric bootstrap of the whole")
        print("  51-file system from the perfectly-calibrated null reproduces every statistic")
        print("  that looked like structure — NONE of 11 escapes its own null's central 95%.")
        print("  The paired instrument is CALIBRATED.)")
        print("  ⚠ BUT THE CLICK IS NOT ROBUST, AND THAT IS THE REAL CORRECTION (w19c/w19d,")
        print("  08-17 slot 5). Everything above is conditional on tau = 0 EXACTLY. tau is a")
        print("  free parameter the board cannot pin: at TEST-level tau = 1.72e-6, inside")
        print("  w19a's own 95% interval, the w16e_aonly branch goes +2.328 -> -1.493e-6 and")
        print("  P(auto beats both wanted) 0.043 -> 0.733. A SIGN FLIP. The mechanism is what")
        print("  the click has always been: the auto-pick holds public slot 1 on a 30e-6")
        print("  public lead against a 5.3e-6 CV lead, and tau = 0 is the only assumption")
        print("  under which that excess is 100% slice noise. Worse, the public board can")
        print("  NEVER identify the test/public split of tau — both add tau^2 to the same")
        print("  observed variance — so no future submission resolves this.")
        print("  MARGINALISED over the tau posterior and a flat prior on that split:")
        print("      E[cost of not clicking]  +1.890 / +5.300 / +5.562e-6")
        print("      P(the click LOSES)        0.037 /  0.000 /  0.000")
        print("      E[P(auto beats both)]     0.128 /  0.044 /  0.067")
        print("  STILL CLICK — it is the right side on every branch in expectation. But the")
        print("  0.043/0.031/0.058 above is the tau=0 CORNER, not the answer, and the phrase")
        print("  'not a coin flip' is no longer defensible for the w16e_aonly branch.")
        print("   [w19a_transfer.json, w19b_calib.json, w19c_clicksens.json, w19d_taupost.json]")
        print(f"Wanted: {', '.join(sorted(WANTED))}")
        print("  ############################################################")
        print("  ## ✅ RESOLVED 2026-08-17 (w21 slot 7). WANTED HAS MOVED.  ##")
        print("  ############################################################")
        print("  WANTED is now {w21_ad187corr.csv, w20_ad187_h3.csv}. The block below")
        print("  is the slot-6 supersession notice, KEPT because its preconditions are")
        print("  what authorised the move — but its closing recommendation is now")
        print("  SUPERSEDED IN TURN and must not be acted on. It named a MIXED-pack pair")
        print("  {w20_ad187_h3, w16i_schemeavg} because w21a did not exist when it was")
        print("  written; it also said 'or the corrected rebuild of the former, if it")
        print("  lands higher', and that rebuild landed higher:")
        print("      w21_ad187corr   CV 0.9701068814  LB 0.97117  <- slot 1, NEW BEST")
        print("      w20_ad187_h3    CV 0.9701008150  LB 0.97115  <- slot 2, 0-param hedge")
        print("      w20_ad187       CV 0.9700978895  LB 0.97116  (sent, not a pick)")
        print("      w16i_schemeavg  CV 0.9700556663  LB 0.97107  <- superseded, -51.2e-6")
        print("      blend159av_h3   CV 0.9700491721  LB 0.97105  <- superseded, -57.7e-6")
        print("  The move is on CV, by w21_prereg.txt §1's rule, written before the CV")
        print("  existed. The LB prints are recorded and were NOT an input.")
        print("  ############################################################")
        print("  ## (slot-6 notice follows, for the record)                 ##")
        print("  ############################################################")
        print("  w20 (08-17 slot 6) imported 22 members from adarsh1077's OOF library,")
        print("  vetted by a new fold-signature gate (experiments/w20a_foldgate.py).")
        print("  The resulting 187-member stack is the highest-CV object this workspace")
        print("  has ever produced, by a margin an order of magnitude clear of every")
        print("  scale in play (2e-6 rebuild floor, ~6e-6 within-family CV steps):")
        print("      w20_ad187_h3      CV 0.9701008150   LB 0.97115  <- new account best")
        print("      w20_ad187         CV 0.9700979      (not sent)")
        print("      w20_ad187_rankraw CV 0.9700915300   LB 0.97114")
        print("      w16i_schemeavg    CV 0.9700556663   LB 0.97107  <- current WANTED")
        print("      blend159av_h3     CV 0.9700491721   LB 0.97105  <- current WANTED")
        print("  i.e. the new h3 file is +45.2e-6 on the pick and +51.6e-6 on the")
        print("  insurance file, and it dominates blend159av_h3 on the SAME criterion")
        print("  that file was chosen for (zero fitted parameters).")
        print("  WANTED WAS NOT MOVED IN SLOT 6, AND THAT IS A PRE-REGISTERED HOLD, NOT")
        print("  AN OPINION (experiments/w20g_shipprereg.txt, written before the upload).")
        print("  The two stated preconditions are now BOTH MET:")
        print("    (1) w20d paired attribution finished: +47e-6 +/- 9 over 3 splits,")
        print("        sign-consistent, so the CV gain is the adarsh members and not")
        print("        base drift from 159 -> 165 members;")
        print("    (2) the c_avg/scheme-average correction (+6.4e-6, w16i) has NOT been")
        print("        rebuilt on the new base — which can only RAISE the new file.")
        print("  => THE NEXT SLOT SHOULD SET WANTED = {w20_ad187_h3.csv, w16i_schemeavg.csv}")
        print("     (or the corrected rebuild of the former, if it lands higher), keeping")
        print("     the second slot on the old corrected file as cross-base insurance.")
        print("  ⚠ AND THE CLICK HAS LARGELY COLLAPSED AS A PROBLEM. The entire w16-w19")
        print("  click analysis priced the risk that Kaggle's auto-pick takes a")
        print("  PUBLIC-INFLATED file over the CV pick. After slot 6 the best public file")
        print("  IS the best CV file (0.97115, held alone), and auto-slot 2 is the CV #3")
        print("  file at 0.97114. Re-price before quoting w19d's +1.890/+5.300/+5.562e-6:")
        print("  those numbers describe a board state that no longer exists.")
        print("  ⚠ note (w17a/w17b, 08-17 slot 1) — CORRECTS the w16w note below it.")
        print("  The ladder is NOT dead. Against a correct PAIRED null (500 draws of the")
        print("  leaderboard's own geometry, 46 sent files with stored OOF vectors), dLB")
        print("  tracks dCV within transform family at Spearman +0.850, sign agreement")
        print("  53/57 = 0.930, z +6.49. Among the tight families (h3, h3+corr, ens4,")
        print("  ens4+corr, w, rankraw, rescale) the observed rms(dLB - dCV) is 7.4e-6")
        print("  against 9.4e-6 predicted by slice draw plus grid rounding ALONE — ratio")
        print("  0.79, i.e. no residual mechanism. What slot 10 read as a broken law is ONE")
        print("  file 2.2 sd high: w16e_aonly vs w16i_schemeavg is dCV -1.24e-6 against")
        print("  dLB +10.0e-6 on a predicted sd of 5.2e-6. What DOES survive from w16w:")
        print("  never quote 'LB = CV + 0.0010143' to 1e-6 — the pair sd is 5-9e-6, i.e.")
        print("  +/- one grid step, so a CV difference under ~5e-6 is a coin flip on the")
        print("  public slice and one over ~15e-6 is resolved.")
        print("  ⚠ AND IT SHARPENS THIS CLICK. Ranking all 46 files by mean standardised")
        print("  residual (positive = public score HIGH for its CV) puts Kaggle's three")
        print("  auto-slot-1 holders at the top of the tight families: w16q_ens4avg +1.20,")
        print("  w16e_aonly +1.12, w16t_cellens4 +1.07, against w16i_schemeavg +0.26 and")
        print("  blend159av_h3 -0.78. Selecting on public score makes that partly circular")
        print("  and the decomposition is the non-circular part: w16e_aonly leads")
        print("  blend159av_h3 by 30e-6 on public but only 5.3e-6 on CV, so ~83% of the")
        print("  lead is slice-specific. Auto-selection picks the three most inflated files.")
        print("  (superseded) note (w16w, slot 10): claimed the ladder FALSIFIED and not")
        print("  monotone; read above. WANTED is unaffected either way — it is chosen on")
        print("  CV, and w16e_aonly's CV is below w16i_schemeavg's on both conventions.")
        print("  note (w16s): both wanted files are h3-side and that is now SETTLED, not")
        print("  deferred. h3 beats ens4 on CV with P 0.912 on a single private-sized draw;")
        print("  the public slice disagrees, but a public-SIZED slice reverses a true h3")
        print("  advantage 24% of the time and w16q's 6/6 is one draw. WANTED follows CV.")
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
