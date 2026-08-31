"""w74b — the click price is VOID iff the auto-selection TIERS move, and nothing else.

WHY THIS GUARD REPLACES A RE-PRICE.
The click price has gone stale twice and been caught late twice:

    w45a  hard-coded its tier membership -> void on the 08-22 sends, caught by w57 a day on.
    w57a  derived tiers live but assumed a >=3-file tier 1 -> void on the 08-23 sends, caught
          by w62 a day on. Both times RESEARCH.md was still quoting the dead number.

w74 re-priced w62a's +4.5228e-6 on a board that had grown 121 -> 131 scored submissions and
got a drift of EXACTLY 0.000e-6. That zero is not luck and it is the point of this file. The
estimator is evaluated over NAMES = TIER1 | TIER2 | WANTED and over nothing else, so twenty
new files that all land below tier 2 cannot touch it -- and the common CV->LB gap they would
help pin down is a nuisance parameter that cancels in a CONTRAST between two pairs anyway.

    => The price depends on the live board ONLY through tier membership.
    => Staleness is therefore BINARY and cheap to test. Re-running the pricer to find out is
       ~7 minutes of OOF loading to re-derive a number that was already known to be unchanged.

So: check the membership, not the number. This runs in the time of one API call and belongs
in the standing sweep, which the pricer does not (it hits the API and takes minutes).

⚠ THE THIRD INPUT, and it is NOT covered here. `beta` comes from w17d_coupling.json, a stored
coupling artefact, and WANTED comes from check_selection. Both are checked for identity below
because a change in either also voids the price -- but neither is derived from the board, so
neither is a send-day hazard.

BOTH CONTROLS, per w72 §5.3: a guard verified only to FAIL when broken can still be a guard
that always fails. This one is verified to PASS on the recorded-good state and to FAIL on a
planted tier change, and it reports which.

    .venv/bin/python experiments/w74b_clickstaleguard.py     # 0 = price live, 1 = VOID
"""
from __future__ import annotations

import io, json, os, subprocess, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_list   # noqa: E402  paginated submission reads; see its docstring
sys.path.insert(0, HERE)
COMP = "playground-series-s6e8"
ART = os.path.join(HERE, "w74a_clickprice.json")

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def live_tiers():
    """The two top public-score tiers on the live board, derived exactly as w62a/w74a do."""
    # w140: paginated -- see kaggle_list.py. The old capped read could not detect its
    # own truncation at any page size above 200.
    sub = pd.DataFrame(kaggle_list.submissions())
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    top = sub.groupby("stem")["publicScore"].max().sort_values(ascending=False)
    vals = sorted(set(top.values), reverse=True)
    return (sorted(top[top == vals[0]].index), float(vals[0]),
            sorted(top[top == vals[1]].index), float(vals[1]), int(len(sub)))


def compare(t1, t1v, t2, t2v, rec, label):
    """Return a list of complaints. Empty list == the recorded price still applies."""
    bad = []
    if sorted(rec["tier1"]) != sorted(t1):
        bad.append(f"{label}: tier 1 membership moved {sorted(rec['tier1'])} -> {sorted(t1)}")
    if sorted(rec["tier2"]) != sorted(t2):
        bad.append(f"{label}: tier 2 membership moved {sorted(rec['tier2'])} -> {sorted(t2)}")
    if abs(float(rec["tier1_public"]) - t1v) > 1e-9:
        bad.append(f"{label}: tier 1 public {rec['tier1_public']} -> {t1v}")
    if abs(float(rec["tier2_public"]) - t2v) > 1e-9:
        bad.append(f"{label}: tier 2 public {rec['tier2_public']} -> {t2v}")
    if len(t1) != 2:
        bad.append(f"{label}: tier 1 holds {len(t1)} files -- the pair is no longer DETERMINED "
                   f"and w74a does not price that case at all")
    return bad


def main() -> None:
    rec = json.load(open(ART))
    print("=" * 92)
    print("w74b — is the published click price still live?")
    print("=" * 92)
    print(f"  artefact          w74a_clickprice.json")
    print(f"  recorded price    {rec['cost_auto_pair']:+.4f}e-6 at tau=0, "
          f"{rec['cost_at_tau_upper']:+.4f}e-6 at the 95% upper tau={rec['tau_upper']}")
    print(f"  recorded board    {rec['n_scored']} scored; tier1 {rec['tier1_public']:.5f} "
          f"{rec['tier1']}; tier2 {rec['tier2_public']:.5f} ({len(rec['tier2'])} files)")

    # ---------------------------------------------------------------- CONTROL 1 (positive)
    # The comparison must PASS on the state the artefact itself records. A guard that only
    # ever fails is indistinguishable from a broken guard (w72 §5.3).
    good = compare(rec["tier1"], float(rec["tier1_public"]),
                   rec["tier2"], float(rec["tier2_public"]), rec, "control+")
    if good:
        fail("CONTROL+ : the guard does not pass on the artefact's OWN recorded state: " + "; ".join(good))
    else:
        print("\n  ✅ CONTROL+  passes on the recorded-good state (guard is not stuck failing)")

    # ---------------------------------------------------------------- CONTROL 2 (negative)
    planted = compare(sorted(rec["tier1"] + ["w99_planted_hijacker"]), float(rec["tier1_public"]),
                      rec["tier2"], float(rec["tier2_public"]), rec, "control-")
    if not planted:
        fail("CONTROL- : a planted third tier-1 file did NOT trip the comparison")
    else:
        print(f"  ✅ CONTROL-  a planted tier-1 hijacker trips it ({len(planted)} complaints, "
              f"incl. the DETERMINED test)")

    # ---------------------------------------------------------------- WANTED / beta identity
    from check_selection import WANTED as CS_WANTED           # noqa: E402
    if sorted(x.replace(".csv", "") for x in CS_WANTED) != sorted(rec["wanted"]):
        fail(f"WANTED moved: priced against {sorted(rec['wanted'])}, check_selection now says "
             f"{sorted(x.replace('.csv','') for x in CS_WANTED)} -- the price is for the OLD pair")
    else:
        print(f"  ✅ WANTED    unchanged {sorted(rec['wanted'])}")
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    if abs(beta - float(rec["beta"])) > 1e-12:
        fail(f"beta moved {rec['beta']} -> {beta} -- w17d_coupling.json was rewritten")
    else:
        print(f"  ✅ beta      unchanged {beta:.12f}")

    # ---------------------------------------------------------------- THE LIVE TEST
    t1, t1v, t2, t2v, n = live_tiers()
    print(f"\n  live board        {n} scored; tier1 {t1v:.5f} {t1}; "
          f"tier2 {t2v:.5f} ({len(t2)} files)")
    drift = compare(t1, t1v, t2, t2v, rec, "LIVE")
    if drift:
        for d in drift:
            fail(d)
        print("\n  ⛔ THE PUBLISHED CLICK PRICE IS VOID. Do not quote it. Re-run "
              "w74a_clickprice.py,\n     which will re-derive the tiers and the mechanism "
              "from scratch.")
    else:
        print(f"\n  ✅ TIERS UNCHANGED since the price was taken ({rec['n_scored']} -> {n} "
              f"scored files, all\n     new ones below tier 2). The published "
              f"{rec['cost_auto_pair']:+.4f}e-6 STILL APPLIES — no re-price needed.")

    print(f"\nFAILURES: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
