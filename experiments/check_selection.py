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
WANTED = {"w16i_schemeavg.csv", "blend159av_h3.csv"}

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
        r.page_size = 50
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
    if n_ok == 0:
        print("CONTROL FAILED — treat the selection reading below as unknown.")
        return 2

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
        print(f"Wanted: {', '.join(sorted(WANTED))}")
        print("  ⚠ note (w16w, slot 10): the corrected-h3 CV->LB ladder that every prior")
        print("  price and prediction leaned on is FALSIFIED. w16e_aonly (CV 0.9700544)")
        print("  printed 0.97108 while w16i_schemeavg and w16n_finegrid (CV 0.9700557) print")
        print("  0.97107 — a LOWER-CV file one grid step HIGHER. The family's LB is not a")
        print("  monotone function of its CV at this resolution, so do NOT re-derive any")
        print("  exposure figure from 'LB = CV + 0.0010143'. WANTED is unaffected: it is")
        print("  chosen on CV, and w16e_aonly's CV is below w16i_schemeavg's either way.")
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
