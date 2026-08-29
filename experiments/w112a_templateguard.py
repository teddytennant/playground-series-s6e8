"""w112a — EVERY DESCRIPTION TEMPLATE THE SEND PATH EMITS IS READ BY THE CLASSIFIERS THAT
MATTER. (w112, 2026-08-29.)

WHY THIS EXISTS. This account's submission descriptions ARE its memory across runs — the brief
says so and `w48e_order.py` repeats it — and three separate guards read those descriptions by
grepping a LITERAL out of them: `w84a_pickargmax.CV_RE` pulls the CV, `w93b_cvlbaudit`
recognises a declared measurement, `w72b_dayguard` recognises a certification. Each literal was
written the day its template shipped, and nothing has ever checked that the SET of literals
covers the SET of templates.

On 2026-08-29 it did not. `w26g_send.py` emits two automated templates; `w93b`'s probe mark knew
only the older one. The day's ten were the first send to put four `w55 tail-fill` files far below
the blend band, `w93b` called them ATTEMPTS ON THE BOARD, `w92a_smokerun` went red for it, and
the same rebuild dropped the six sent members out of `w55a_unpriced.json` and turned `w72b` red
too. Two guards, one root cause, one missing string each.

🎯 A CLASSIFIER THAT READS A DECLARATION IS ONLY AS GOOD AS ITS LIST OF WORDINGS, AND A LIST OF
WORDINGS ROTS SILENTLY — it fails by classifying a file as the DEFAULT, never by erroring.

WHAT IT CHECKS. The literals are IMPORTED from the guards that own them, never retyped, so this
file and the things it guards cannot drift (w60b's rule).
  T1  COVERAGE. Every automated-era template head on the live send record is one of the heads
      registered below. A new template fails T1 until it is classified here — the hole is in
      the QUERY, not in the checker (w89b, w92a G3).
  T2  Every `queue-drain` description yields a CV under `w84a.CV_RE`. These are attempts; the
      CV->LB audit is computed from exactly this parse, so a template change that broke it
      would silently shrink the audit's sample instead of failing.
  T3  Every `tail-fill` description is recognised as a declared measurement by
      `w93b.PROBE_MARKS` AND carries `w72b.CERT_MARK`. This is the 08-29 failure, asserted.
  T4  NO DEAD LITERAL. Every mark in `w93b.PROBE_MARKS` matches at least one live description.
      A mark that matches nothing is either a rotted literal or a template that stopped being
      emitted, and both deserve a look.
  T5  CONTROL-. Each of T1/T2/T3/T4 is re-run against a mutated copy of the record or the mark
      list and must FAIL. A green suite of assertions that cannot fail is not evidence.

    .venv/bin/python experiments/w112a_templateguard.py     # rc=0 sound, rc=1 with the reason

⛔ Read `$?` from a command substitution or a redirect, never after a pipe (RESEARCH §9.1).
"""
from __future__ import annotations

import io, json, os, re, sys, contextlib

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w84a_pickargmax import CV_RE, SUBS_ARGV, fetch_raw, parse                      # noqa: E402
from w93b_cvlbaudit import PROBE_MARKS                                              # noqa: E402
from w91a_subdate import parse_sub_dates                                            # noqa: E402

OUT = os.path.join(HERE, "w112a_templateguard.json")

# The automated-era heads. Everything before them was hand-written per slot and is exempt by
# DATE, not by name: the first templated send is w26g's, and nothing hand-written has gone out
# since. ⚠ Widen this only together with the rules in T2/T3 — an unclassified head is the
# failure this guard is for, so adding a head without a rule defeats it.
HEADS = {
    "w26g queue-drain": "attempt: carries a stack CV and a predicted LB",
    "w55 tail-fill":    "declared measurement: no stack CV, certified below the tier",
    "w37 es-bias":      "declared measurement: the w37-era single-member probe",
}
AUTOMATED_FROM = "2026-08-22"          # the first day the registrar wrote the descriptions

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def cert_mark() -> str:
    """Import w72b's literal rather than retype it. w72b executes work at import, so its stdout
    is swallowed and its `sys.argv` is neutralised — the same manoeuvre w72b makes for w48e."""
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        with contextlib.redirect_stdout(io.StringIO()):
            import w72b_dayguard as D
    finally:
        sys.argv = saved
    return D.CERT_MARK


def head_of(desc: str) -> str:
    return " ".join(str(desc).split()[:2])


def classify(df: pd.DataFrame) -> pd.DataFrame:
    # ⛔ PARSE THE DATE, DO NOT COMPARE THE STRING. Lexicographic `>=` on the raw column happens
    # to be right for today's format and would go silently wrong the day the format moves --
    # which is exactly what w91 spent a run repairing across seven modules. Going through
    # `parse_sub_dates` also enrols this file in `w92a_smokerun`'s surface, so it is EXECUTED
    # by the suite rather than only type-checked.
    when = parse_sub_dates(df["date"])
    auto = df[when >= pd.Timestamp(AUTOMATED_FROM)].copy()
    # ⚠ NOT `head` -- `DataFrame.head` is a METHOD, so `auto.head` is the bound method
    # and every comparison against it is a silent `bool`, not a mask.
    auto["tmpl"] = auto["description"].map(head_of)
    return auto


def run(auto: pd.DataFrame, marks: tuple[str, ...], cmark: str, cv_re,
        every: pd.Series | None = None) -> list[str]:
    """The four assertions, as data, so T5 can re-run them against a mutated input."""
    out = []
    unknown = sorted(set(auto["tmpl"]) - set(HEADS))
    if unknown:
        out.append(f"T1 {len(unknown)} automated-era template head(s) no rule classifies: "
                   f"{unknown}")

    drain = auto[auto.tmpl == "w26g queue-drain"]
    nocv = drain[drain["description"].astype(str).str.extract(cv_re)[0].isna()]
    if len(nocv):
        out.append(f"T2 {len(nocv)} queue-drain description(s) carry no parseable CV: "
                   f"{sorted(nocv.fileName)[:5]}")

    tail = auto[auto.tmpl == "w55 tail-fill"]
    d = tail["description"].astype(str)
    declared = pd.Series(False, index=tail.index)
    for m in marks:
        declared |= d.str.contains(m, regex=False)
    if (~declared).any():
        out.append(f"T3 {int((~declared).sum())} tail-fill description(s) match NO probe mark, "
                   f"so w93b would score them as attempts: {sorted(tail.fileName[~declared])[:5]}")
    uncert = tail[~d.str.contains(cmark, regex=False)]
    if len(uncert):
        out.append(f"T3 {len(uncert)} tail-fill description(s) carry no certification, so w72b "
                   f"cannot rescue them once they leave the queue: {sorted(uncert.fileName)[:5]}")

    # ⚠ DEADNESS IS A CLAIM ABOUT THE WHOLE RECORD, NOT ABOUT THE WINDOW. T1 restricts to
    # the automated era because everything before it was hand-written per slot; T4 must not,
    # or a mark whose template simply stopped being emitted reads as rotted. Caught by this
    # guard on its own first run: `w37 es-bias` last went out before AUTOMATED_FROM, so inside
    # the window its mark matched nothing and T4 fired on a correct literal.
    alld = (auto["description"] if every is None else every).astype(str)
    dead = [m for m in marks if not alld.str.contains(m, regex=False).any()]
    if dead:
        out.append(f"T4 {len(dead)} probe mark(s) match no live description -- rotted literal "
                   f"or a retired template: {dead}")
    return out


def main() -> int:
    cmark = cert_mark()
    df = parse(fetch_raw(SUBS_ARGV))
    auto = classify(df)

    print("=" * 88)
    print("  w112a  DESCRIPTION-TEMPLATE COVERAGE GUARD")
    print("=" * 88)
    print(f"\n  {len(df)} submission(s) on record; {len(auto)} in the automated era "
          f"(date >= {AUTOMATED_FROM})")
    for h, why in sorted(HEADS.items()):
        print(f"    {int((auto.tmpl == h).sum()):>4}  {h:<20} {why}")
    print(f"  marks: {len(PROBE_MARKS)} probe, 1 certification "
          f"({cmark[:34]}...)\n")

    every = df["description"]
    for msg in run(auto, PROBE_MARKS, cmark, CV_RE, every):
        fail(msg)
    if not FAILURES:
        print("  ✅ T1-T4 every automated template is read by every classifier that needs it.")

    # ---- T5: each assertion must be able to fail -------------------------------------------
    print("\n  T5 CONTROL- (each mutation must produce the named failure)")
    controls = [
        ("T1", "an unregistered head appears",
         lambda a, m, c, r: (a.assign(tmpl=a.tmpl.mask(a.index == a.index[0], "w999 brand-new")),
                             m, c, r)),
        ("T2", "the CV regex stops matching",
         lambda a, m, c, r: (a, m, c, re.compile(r"MATCHES NOTHING (\d+)"))),
        ("T3", "the tail-fill probe mark is dropped",
         lambda a, m, c, r: (a, (m[0],), c, r)),
        ("T3", "the certification sentence is dropped",
         lambda a, m, c, r: (a, m, "NO DESCRIPTION SAYS THIS", r)),
        ("T4", "a mark nothing matches is added",
         lambda a, m, c, r: (a, tuple(m) + ("A LITERAL THAT ROTTED",), c, r)),
    ]
    for want, label, mutate in controls:
        got = run(*mutate(auto.copy(), PROBE_MARKS, cmark, CV_RE), every=every)
        hit = [g for g in got if g.startswith(want)]
        ok = bool(hit)
        print(f"    {want}  {label:<40} {'FIRES ✅' if ok else 'SILENT ❌'}")
        if not ok:
            fail(f"T5 control for {want} ({label}) did not fire -- the assertion is inert")

    json.dump({"n_submissions": int(len(df)), "n_automated": int(len(auto)),
               "heads": {h: int((auto.tmpl == h).sum()) for h in HEADS},
               "probe_marks": list(PROBE_MARKS), "cert_mark": cmark,
               "failures": FAILURES},
              open(OUT, "w"), indent=2, sort_keys=True)
    print(f"\n  FAILURES {FAILURES}   -> {os.path.relpath(OUT, os.path.dirname(HERE))}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
