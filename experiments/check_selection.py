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
        print("Priced by w15i: +9.2e-6 if the final-submission limit is 2, +36.5e-6 if it")
        print("is 1, +112e-6 in the worst branch. ~2.5 places per 1e-5 at the local density.")
        print(f"Wanted: {', '.join(sorted(WANTED))}")
        print("  note (w16q): both wanted files are h3-side. h3 beats ens4 on CV 6/6 and")
        print("  loses to it on the public LB 6/6. WANTED follows CV, as the brief requires;")
        print("  switching the zero-parameter hedge to blend159av costs -4.30e-6 of CV.")
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
