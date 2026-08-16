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
