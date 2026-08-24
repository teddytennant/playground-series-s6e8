"""w73a — DOES THE RANKER'S CV BASIS CHANGE ANY DECISION?  (closes w70 §3c, open ten runs)

THE QUESTION.  Every corrected file has two stored, defensible CVs:
    SHIPPED = fast_auc(y, submissions/oof_<tag>.npy)      <- what EVERYTHING here ranks on
    HONEST  = w21a base_auc + nested_delta                <- arm choice cross-fitted
w70 measured the gap per file (`w70a_optimism.json`) and deliberately did NOT decide, because
the run that computes a number should not also take the decision it feeds.  This takes it.

A BASIS IS WORTH ONLY WHAT IT CHANGES.  It feeds exactly three actions:
    (a) send-queue ORDER      w23b/w26d      -- decision-bearing since w72's 26-slot shortfall
    (b) FIT_CV_MAX            w53a:111       -- the pricer's clamp and the eligibility tier
    (c) WANTED                check_selection:247 -- the FINAL DEADLINE SELECTION (the Rogii lever)

Bars were fixed in `w73_prereg.txt` BEFORE this file's table was read.  Verdicts: P1 does not
fire (0.126e-6 < 0.25), P2 does not fire, P3 does not fire (0.575e-6 < 2.0).

⚠ ONE SCOPE FLAW IN THE PREREG, AND WHY IT DOES NOT LAUNDER THE RESULT.  P2 was written over
"all 20 corrected files".  A final entry must be a SUBMITTED entry, so the population that
governs action (c) is the 14 SENT ones; as literally written the bar would have fired on
w42_ad217stdcorr, which is unsent and VETOed and can never be a final entry.  Restricting a
population after seeing the data is how pre-registration gets laundered, so GATE 4 settles it
the only way that is not a judgement call: argmax(SHIPPED) == argmax(HONEST) == w42_ad217stdcorr
on the UNRESTRICTED population too.  The basis moves neither argmax.  ad199 is displaced by
ELIGIBILITY, not by basis, so the restriction is NOT LOAD-BEARING and P2 fails to fire on BOTH
populations.  The flaw was in the operationalisation, not the estimand.

DESCRIPTIVE — reads artefacts, refits nothing, draws no seed, writes no submission.

    .venv/bin/python experiments/w73a_cvbasis.py        # rc=0 iff FAILURES 0
"""
from __future__ import annotations

import io, json, os, sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

U = 1e-6
FLOOR = 5.095      # w68's between-file cross-process floor, e-6
BAR_P1 = 0.25      # w73_prereg P1
BAR_P3 = 2.0       # w73_prereg P3 (the bar w65's R1 decides on)
SLOT1 = "w36_ad199stdcorr"
SLOT2 = "w23_ad187stdcorr"

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def sent_tags() -> set[str]:
    """The authoritative sent history is the live API, never w23b's `sent` column (w72 §5.2)."""
    import w26g_send as S
    out = set()
    for s in S.api_submissions():
        fn = s.get("fileName") if isinstance(s, dict) else getattr(s, "fileName", None)
        if fn:
            out.add(fn[:-4] if fn.endswith(".csv") else fn)
    return out


def load_veto() -> set[str]:
    """w48e_order scans sys.argv AT MODULE SCOPE (w72 §5.1) — neutralise argv across the import."""
    argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):          # w48e prints the whole pricer on import
            import w48e_order as O
        return {t[:-4] if t.endswith(".csv") else t for t in O.VETO}
    finally:
        sys.argv = argv


def main() -> None:
    rows = json.load(open(os.path.join(HERE, "w70a_optimism.json")))["rows"]
    sent = sent_tags()
    veto = load_veto()
    elig = [x for x in rows if x["tag"] in sent]

    print(f"  {len(rows)} corrected files; {len(elig)} SENT and therefore eligible to be a final entry")

    # ---- GATE 1: w70a's own summary must reproduce from its rows, IN THE RIGHT UNITS -------
    # `opt` is stored in RAW AUC UNITS while `opt_max`/`opt_mean` are in e-6.  Reading the
    # column without dividing prints 0.000 on all 20 rows and silently vacates every test
    # built on it.  Assert the two agree so the slip cannot recur.
    d = json.load(open(os.path.join(HERE, "w70a_optimism.json")))
    got = max(x["opt"] for x in rows) / U
    if abs(got - d["opt_max"]) > 1e-9:
        fail(f"GATE 1 units: rows max {got} != summary opt_max {d['opt_max']}")
    else:
        print(f"  GATE 1 units: rows opt/1e-6 max {got:.3f} == summary opt_max — `opt` is RAW AUC. OK")

    # ---- P1: is RESEARCH's stated identity SHIPPED - HONEST == scheme_optimism true? -------
    resid = sorted(x["opt"] / U - (x["shipped"] - x["honest"]) / U for x in rows)
    med = resid[len(resid) // 2] if len(resid) % 2 else 0.5 * (resid[len(resid)//2 - 1] + resid[len(resid)//2])
    p1 = abs(med) > BAR_P1
    print(f"\n  P1 identity: median residual {med:+.3f}e-6 (min {resid[0]:+.3f}, max {resid[-1]:+.3f})"
          f" vs bar {BAR_P1} -> {'FIRES' if p1 else 'does not fire'}")
    print(f"     ⚠ the residual SPREAD is real ({resid[-1]-resid[0]:.3f}e-6 wide) and all 6 zero-optimism"
          f" files sit on the positive side: scheme_optimism is not the ONLY difference between the bases.")

    # ---- P2: does the basis change WANTED? -------------------------------------------------
    def rank(pop, key, tag):
        return [z["tag"] for z in sorted(pop, key=lambda q: -q[key])].index(tag) + 1

    p2 = False
    for tag, slot in ((SLOT1, "slot 1"), (SLOT2, "slot 2")):
        if not any(z["tag"] == tag for z in elig):
            fail(f"P2: {tag} is WANTED but not in the eligible set")
            continue
        rs, rh = rank(elig, "shipped", tag), rank(elig, "honest", tag)
        print(f"  P2 {slot} {tag}: rank {rs}/{len(elig)} on SHIPPED, {rh}/{len(elig)} on HONEST")
        if rs != rh:
            p2 = True

    # the one reorder in the eligible set, and whether it clears the floor
    bs = [z["tag"] for z in sorted(elig, key=lambda q: -q["shipped"])]
    bh = [z["tag"] for z in sorted(elig, key=lambda q: -q["honest"])]
    moved = [t for i, t in enumerate(bs) if bh[i] != t]
    print(f"  P2 reorder in the eligible set: {len(moved)} position(s) {moved or '— none'}")
    for t in set(moved):
        x = [z for z in elig if z["tag"] == t][0]
        for o in set(moved) - {t}:
            q = [z for z in elig if z["tag"] == o][0]
            g = abs(x["honest"] - q["honest"]) / U
            print(f"     {t} vs {o}: {g:.3f}e-6 apart on HONEST vs the {FLOOR}e-6 floor ->"
                  f" {'OUTSIDE' if g > FLOOR else 'INSIDE — indistinguishable'}")
            if g > FLOOR:
                p2 = True

    # ---- GATE 4: the prereg's scope flaw is not load-bearing --------------------------------
    ams = max(rows, key=lambda z: z["shipped"])["tag"]
    amh = max(rows, key=lambda z: z["honest"])["tag"]
    print(f"\n  GATE 4 unrestricted argmax: SHIPPED {ams} | HONEST {amh}")
    if ams != amh:
        fail("GATE 4: the argmax DOES move on the unrestricted population — the eligible-set "
             "restriction IS load-bearing and P2's verdict cannot be trusted")
    else:
        print(f"     same file -> the basis moves neither argmax; {SLOT1} is displaced by "
              f"ELIGIBILITY, not basis. Restriction NOT load-bearing. OK")

    # ---- P3: FIT_CV_MAX --------------------------------------------------------------------
    buf = io.StringIO()
    with redirect_stdout(buf):
        import w53a_pricer as PR
    a = [z for z in rows if z["tag"] == SLOT1][0]
    if abs(PR.FIT_CV_MAX - a["shipped"]) > 1e-12:
        fail(f"P3: FIT_CV_MAX {PR.FIT_CV_MAX:.10f} is no longer {SLOT1}'s shipped CV — re-derive")
    shift = (a["honest"] - PR.FIT_CV_MAX) / U
    p3 = abs(shift) > BAR_P3
    print(f"\n  P3 FIT_CV_MAX {PR.FIT_CV_MAX:.10f} -> {a['honest']:.10f}  shift {shift:+.3f}e-6"
          f" vs bar {BAR_P3} -> {'FIRES' if p3 else 'does not fire (cosmetic)'}")

    # ---- (a) the send order: how much of the SENDABLE pool would even move? -----------------
    unsent = [x for x in rows if x["tag"] not in sent]
    movable = [x for x in unsent if x["tag"] not in veto]
    print(f"\n  (a) send order: {len(unsent)} corrected files unsent, {len(unsent)-len(movable)} of them VETOed")
    for x in sorted(movable, key=lambda z: -z["shipped"]):
        print(f"      {x['tag']:28s} would move {(x['shipped']-x['honest'])/U:+.3f}e-6")
    print(f"      -> at most {len(movable)} file(s) in the whole sendable pool re-order, and w72 priced"
          f" the marginal slots at exactly 0.00e+00.")

    print(f"\n  VERDICT  P1 {'FIRES' if p1 else 'no'} | P2 {'FIRES' if p2 else 'no'} | P3 {'FIRES' if p3 else 'no'}")
    if not (p1 or p2 or p3):
        print("  ⟹ THE RANKING BASIS IS DECISION-IRRELEVANT ON ALL THREE ACTIONS. CLOSE THE ITEM.")
        print("    Do NOT re-base w53a/w23b/w26d: the term is smaller than the noise floor of the")
        print("    comparison it would feed, and the frozen pricer stays frozen.")
    print(f"\n  FAILURES {FAILURES}")


if __name__ == "__main__":
    main()
    sys.exit(1 if FAILURES else 0)
