"""w182b -- perturbs #66 and #67 against the live tree, both directions.

A guard that only ever agrees with the state it was written for is not tested. Each probe
edits RESEARCH.md in a temp copy, points the two guards at it, and asserts the DIRECTION of
the answer -- green where the edit is legitimate, red where it is not. The expected-red probes
matter more than the expected-green ones: w157/w156 both shipped green on a cell they were
misreading, so agreement proves nothing on its own.
"""
from __future__ import annotations

import os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
RESEARCH = os.path.join(ROOT, "RESEARCH.md")
JOURNAL = os.path.join(ROOT, "JOURNAL.md")
sys.path.insert(0, HERE)
from w156a_recordguard import handing_span, TRAIL  # noqa: E402

ENTRY_FULL = re.compile(r"→ w\d+ \d\d-\d\d(?: \(closed\))?")

ORIG = open(RESEARCH, encoding="utf-8").read()


def run(stem: str, text: str, args=()) -> tuple[int, str]:
    """Run one guard with RESEARCH.md temporarily replaced by `text`."""
    bak = ORIG
    try:
        open(RESEARCH, "w", encoding="utf-8").write(text)
        p = subprocess.run([PY, os.path.join(HERE, stem + ".py"), *args],
                           capture_output=True, text=True, timeout=600)
        return p.returncode, p.stdout + p.stderr
    finally:
        open(RESEARCH, "w", encoding="utf-8").write(bak)


def row_line(text: str, n: int) -> str:
    for line in text.split("\n"):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if m and int(m.group(1)) == n and "w117a_handcount" in line and len(line) > 200:
            return line
    raise SystemExit(f"row {n} not found")


FAILS = []


def probe(name: str, stem: str, text: str, want_rc: int, args=(), needle: str = "",
          baseline: bool = False):
    """⚠ (w182) A PERTURBATION THAT DOES NOT PERTURB IS SILENTLY GREEN, and this battery was
    bitten by it inside one run. Every probe below built its edit with `.replace()` against a
    hard-coded `→ w163 09-04 (closed)**` — the last entry in row 7 at the time. The moment this
    run appended `→ w182 09-05 (closed)` to that trail, four `.replace()` calls matched nothing,
    four probes ran the guards over UNMODIFIED text, and all four reported the shipped answer.
    Two of them wanted red and got green, which reads as a broken guard; the other two would have
    read as a working one. So: refuse to score a probe whose edit changed nothing."""
    if not baseline and text == ORIG:
        print(f"  FAIL {name:<62} NO-OP -- the edit matched nothing")
        FAILS.append(name + " (no-op)")
        return
    rc, out = run(stem, text, args)
    ok = (rc == want_rc) and (needle in out if needle else True)
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<62} rc={rc} want={want_rc}")
    if not ok:
        FAILS.append(name)
        for ln in out.split("\n"):
            if "FAIL" in ln or "fails on row" in ln:
                print("        " + ln.strip())


def main() -> int:
    r5, r7 = row_line(ORIG, 5), row_line(ORIG, 7)
    # The anchor is the LAST entry of row 7's handing span, found rather than typed, so the
    # battery keeps working after the trail is appended to. See probe()'s docstring.
    last = [(m.group(0)) for m in ENTRY_FULL.finditer(handing_span(r7))][-1]
    print(f"anchor: row 7's last handing entry is `{last}`\n")
    print("w182b -- perturbing #66 w156a_recordguard and #67 w157a_closedguard\n")

    print("the shipped tree")
    probe("shipped, #67", "w157a_closedguard", ORIG, 0, baseline=True)
    probe("shipped, #66", "w156a_recordguard", ORIG, 0, baseline=True)
    probe("shipped, #67 --control still fires", "w157a_closedguard", ORIG, 0, ("--control",),
          baseline=True)
    probe("shipped, #66 --control still fires", "w156a_recordguard", ORIG, 0, ("--control",),
          baseline=True)

    print("\nthe defect this run fixed, reintroduced")
    # Row 5's artefact chain moved INSIDE the bold span: now it really is claimed as a handing,
    # out of date order, and both the span rule and C3c must object.
    moved = r5.replace(" (closed)** (count from", " (closed) → w107 08-28** (count from")
    probe("row 5's w107 moved INSIDE the bold span -> C3c red", "w157a_closedguard",
          ORIG.replace(r5, moved), 1, (), "run backwards in date")

    print("\nlegitimate edits that must stay green")
    # The correct append this run makes to row 7. ⚠ TWO-STATE, and the state depends on
    # JOURNAL.md: a trail entry must be backed by a run header, so this is RED until this
    # run's entry is written and GREEN after. Run the battery again once the entry lands --
    # that is the whole point of writing the entry before claiming the guard works.
    app = r7.replace(last + "**", last + " → w997 09-05 (closed)**")
    # ⚠ THE FALSE RED THE OLD READER WOULD HAVE FIRED. A provenance pointer dated after the
    # deadline owes no (closed) marker, because it is not a handing.
    prov = r7.replace("· **artefacts verified**", "· w127a → w182 09-05 · **artefacts verified**")
    probe("row 7 gains a post-deadline PROVENANCE pointer, unmarked", "w157a_closedguard",
          ORIG.replace(r7, prov), 0)
    probe("  same, under #66", "w156a_recordguard", ORIG.replace(r7, prov), 0)

    print("\nedits that must go red")
    probe("row 7 appends a handing whose run has no journal entry -> #67 red",
          "w157a_closedguard", ORIG.replace(r7, app), 1)

    bad = r7.replace(last + "**", last + " → w181 09-05**")
    probe("row 7 appends a post-deadline HANDING with no marker", "w157a_closedguard",
          ORIG.replace(r7, bad), 1)
    back = r7.replace(last + "**", last + " → w152 09-02 (closed)**")
    probe("row 7 appends a handing dated BACKWARDS -> C3c red", "w157a_closedguard",
          ORIG.replace(r7, back), 1, (), "run backwards in date")
    ghost = r7.replace(last + "**", last + " → w996 09-05 (closed)**")
    probe("row 7 names a handing with no journal entry -> #66 red", "w156a_recordguard",
          ORIG.replace(r7, ghost), 1)
    # ⚠ DERIVED, NOT TYPED: `**×19,` was hard-coded here and stopped matching the moment
    # this run bumped the count to ×20, so the "strip the span" edit applied only half of
    # itself and left a still-valid span behind. Same lesson as probe()'s docstring.
    nospan = re.sub(r"\*\*(×\d+,)", r"\1", r7).replace("(closed)** ·", "(closed) ·")
    assert handing_span(nospan) == "", "the strip probe left a readable span behind"
    probe("row 7 loses its bold span -> the low-count arm fires", "w157a_closedguard",
          ORIG.replace(r7, nospan), 1)

    print("\nthe span rule itself, measured not asserted")
    inside = len(TRAIL.findall(handing_span(r5)))
    raw = len(TRAIL.findall(r5))
    print(f"  row 5: {raw} arrow links raw, {inside} inside the bold span, {raw - inside} excluded")
    if raw - inside != 1:
        FAILS.append("row 5 exclusion count")
    for n in (1, 2, 3, 4, 6, 7, 8, 9, 10):
        line = row_line(ORIG, n)
        if len(TRAIL.findall(handing_span(line))) != len(TRAIL.findall(line)):
            FAILS.append(f"row {n} unexpectedly excludes a link")
    print(f"  rows 1-4, 6-10: span and raw agree, so the rule excludes ONE site and no others")

    print(f"\nFAILURES: {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
