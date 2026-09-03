"""w157a -- appending to an ANGLE INDEX trail can MOVE the predecessor's `(closed)` marker onto
the new entry instead of adding a second one, and the row still reads as annotated (2026-09-03).

The ANGLE INDEX handing-count cells carry a trail of the runs that were handed each row:

    x18, from 08-10 -> w116 08-29 -> w125 08-30 -> w143 09-01 -> w152 09-02 (closed) - the most...

`(closed)` records that the run happened after the competition deadline (2026-08-31 23:59) and
so could not submit. Nine of the ten rows carry it on every post-deadline entry. Row 5 does not:
its `w143 09-01` is bare, and w143's own JOURNAL.md header reads
`BLOCKED: COMPETITION CLOSED. NO WORK DONE, BY DESIGN.` -- it was the FIRST run to be blocked by
the close, so it is the least ambiguous case in the whole table.

THE MARKER WAS NOT MISSING. IT WAS TAKEN. Commit fff8d08 (09-01) wrote row 5 as

    -> w143 09-01 (closed) - the most-handed row

and d44c082 (09-02), which extended the trail for w152, rewrote it as

    -> w143 09-01 -> w152 09-02 (closed) - the most-handed row

One `(closed)` went in and one came out, so a diff reader sees a marker on the newest entry and
a trail that grew by one, which is exactly what a correct append looks like.

WHY ROW 5 AND ONLY ROW 5. Every other row's trail is followed immediately by the cell's closing
prose (`. artefacts verified**`). Row 5 is the only one whose trail is followed by a dash clause
belonging to the trail itself (`- the most-handed row`), so the natural place to type the new
entry is BEFORE that clause -- which puts the cursor between `w143 09-01` and `(closed)`, and
the marker ends up after the wrong run. The defect is a property of the row's punctuation, not
of the run that made it.

w128 predicted the next defect in this family would be "in the closed column or in a pool
disclosure". w155 found the pool one (#65). This is the other half, and it is the first check
that reads the closed column at all: #53 units, #55 scope, #56 denominator and #57 opt-out all
read the price column, #65 reads pools, and #66 reads the trail's run ids while ignoring
everything printed after them.

CHECKED, each against an artefact rather than a quotation:

  C1  the trail entries in the ten handing-count cells, extracted as (run, date, marker), with
      C1b proving the matcher tracks the text rather than agreeing by luck.
  C2  the deadline, read LIVE off the competition object through an authenticated client
      (w153 for the request shape, w154 for the client), and cross-checked against the date
      this guard compares with. `kaggle` is not importable from this venv, so the read is
      shelled out to the uv tool python rather than re-execing a doc-parsing guard into it.
  C3  THE ASSERTION, two instruments that fail independently:
        C3a  THE DEADLINE RULE. A trail entry dated after the deadline must carry the marker;
             one dated on or before it must not. Cell text only -- no journal, no network.
             ⚠ MONOTONICITY WAS THE FIRST INSTRUMENT AND IT DOES NOT WORK. "Once marked, all
             later entries marked" holds here: the marker moved FORWARD onto the newer entry,
             so the trail is still monotone in date order and the bare w143 09-01 sits
             legitimately before the first marked one. Only the deadline separates them.
        C3b  AGREEMENT. Every trail entry's date must equal the date in that run's JOURNAL.md
             header. RUN_HDR is COPIED from w117a_handcount rather than imported, so drift
             there turns C4 red first.
  C4  the anchors this guard assumes: w117a_handcount's RUN_HDR, byte-compared against the
      copy in this file, and the row-identification rule that #65 and #66 also use, which
      lives in w156a_recordguard rather than in w117a_handcount.
  C5  --control over the frozen PRE-FIX row 5 cell, with the journal map and the deadline
      frozen too. w156's control had to be rebuilt because it borrowed live state and went
      silent the moment the fix landed; this one freezes all three inputs from the start.
  C6  scope and blindness, measured rather than assumed.

    .venv/bin/python experiments/w157a_closedguard.py            # rc 0 = clean
    .venv/bin/python experiments/w157a_closedguard.py --control  # exits 0, having fired

Exits 0 when every assertion holds, 1 otherwise, and prints FAILURES: n as its last line.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT / "RESEARCH.md"
JOURNAL = ROOT / "JOURNAL.md"
HANDCOUNT = HERE / "w117a_handcount.py"

COMP = "playground-series-s6e8"
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"

# The deadline this guard compares trail dates against. C2 re-reads it live and fails if the
# board disagrees, so this constant cannot drift away from the competition unnoticed.
DEADLINE_DATE = "08-31"
DEADLINE_FULL = "2026-08-31 23:59:00"

# COPIED from w117a_handcount.py:137, not imported, so that C4 fires if that file's idea of a
# run header moves. w156a_recordguard copies the same line for the same reason.
RUN_HDR = re.compile(r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,)")

ROW = re.compile(r"^\| (\d+) \| \*")

# A trail entry: the run, its date, and whether `(closed)` follows it. The marker group is
# optional, which is the whole point -- #66's TRAIL stops at the date and cannot see it.
ENTRY = re.compile(r"(?:→|->) w(\d+) (\d\d-\d\d)( \(closed\))?")

# Row 5 exactly as d44c082 left it, frozen. The defect is `w143 09-01` with no marker while
# `w152 09-02` carries one.
CONTROL_CELL = (
    "| 5 | *feature engineering: interactions, in-fold target and count encodings* | "
    "**×18, from 08-10 → w116 08-29 → w125 08-30 → w143 09-01 → w152 09-02 (closed) — the "
    "most-handed row** (count from `w117a_handcount`, not by hand) · w15b/w15d → w62 → "
    "w107 08-28 · **artefacts verified** | prices |"
)

# The journal dates the control is judged against, frozen alongside the cell so that C5 keeps
# working after the live corpus changes. w156: a control that borrows live state stops working
# exactly when the fix lands.
CONTROL_JOURNAL = {116: "08-29", 125: "08-30", 143: "09-01", 152: "09-02", 107: "08-28"}


def table_rows(text: str) -> dict[int, str]:
    """The ANGLE INDEX rows, keyed by row number.

    Same identification rule as w155a and w156a: the ten-row block whose cells carry the
    handing counts. A bare row number appears in half a dozen other tables in this file.
    """
    out: dict[int, str] = {}
    for line in text.split("\n"):
        m = ROW.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 10 and "w117a_handcount" in line and n not in out:
            out[n] = line
    return out


def entries(cell: str) -> list[tuple[int, str, bool]]:
    """(run, date, marker) for every trail entry in one cell, in the order written."""
    return [(int(a), d, bool(c)) for a, d, c in ENTRY.findall(cell)]


def journal_dates() -> dict[int, str]:
    """The date on each run's JOURNAL.md header, by run id.

    First header wins: a run's entry is written once, and later runs quote earlier ids in
    prose. w156a's journal_run_ids collects every id on a header line for its own purposes;
    here only the id the header is ABOUT can be dated, so this takes the first.
    """
    out: dict[int, str] = {}
    for line in JOURNAL.read_text(encoding="utf-8").split("\n"):
        if not RUN_HDR.match(line):
            continue
        d = re.search(r"20\d\d-(\d\d-\d\d)", line)
        ids = re.findall(r"\bw(\d+)\b", line)
        if d and ids:
            out.setdefault(int(ids[0]), d.group(1))
    return out


def deadline_defects(rows: dict[int, str]) -> list[tuple[int, int, str]]:
    """Trail entries whose `(closed)` marker disagrees with their own date.

    Needs nothing but the cell and the deadline -- no journal, no network. A run dated after
    the deadline could not submit and must be marked; one dated on or before it must not be.

    ⚠ MONOTONICITY WAS TRIED FIRST AND DOES NOT CATCH THIS. "Once marked, all later entries
    marked" is true but useless here: the marker did not vanish, it MOVED FORWARD onto w152
    09-02, so the trail still reads as monotone in date order and the unmarked w143 09-01
    sits legitimately before the first marked entry. Only the deadline distinguishes them.
    """
    out = []
    for n in sorted(rows):
        for run, date, mark in entries(rows[n]):
            want = date > DEADLINE_DATE
            if want != mark:
                out.append((n, run, f"dated {date}, marker={mark}, deadline says {want}"))
    return out


def live_deadline() -> tuple[str | None, str]:
    """The competition deadline, read through the interpreter that has `kaggle` installed.

    `kaggle` is not importable from this venv; w153a_openreadguard re-execs into the uv tool
    python for it. Re-execing a doc-parsing guard for one cross-check is heavier than shelling
    out for it, so this does the latter and treats an unavailable read as UNKNOWN, not FAIL.
    """
    import subprocess

    if not pathlib.Path(KAGGLE_PY).exists():
        return None, f"{KAGGLE_PY} not present"
    prog = (
        "from kaggle.api.kaggle_api_extended import KaggleApi;"
        "from kagglesdk.competitions.types.competition_api_service import "
        "ApiGetCompetitionRequest as R;"
        "a=KaggleApi();a.authenticate();r=R();"
        f"r.competition_name={COMP!r};"
        "c=a.build_kaggle_client().__enter__().competitions.competition_api_client"
        ".get_competition(r);print(c.deadline)"
    )
    try:
        p = subprocess.run([KAGGLE_PY, "-c", prog], capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["non-zero exit"])[-1]
    return p.stdout.strip(), "read live"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    control = args.control
    fails = 0

    def fail(msg: str) -> None:
        nonlocal fails
        fails += 1
        print(f"  FAIL {msg}")

    print("w157a -- the (closed) column of the ANGLE INDEX trail, against the deadline and "
          "the journal")
    print()

    text = RESEARCH.read_text(encoding="utf-8")
    rows = table_rows(text)
    if control:
        rows = dict(rows)
        rows[5] = CONTROL_CELL

    print("C1 the trail entries, extracted as (run, date, marker)")
    if len(rows) != 10:
        fail(f"expected 10 handing-count rows, found {len(rows)}: {sorted(rows)}")
    total = sum(len(entries(c)) for c in rows.values())
    marked = sum(1 for c in rows.values() for e in entries(c) if e[2])
    print(f"  {len(rows)} rows, {total} trail entries, {marked} carrying (closed)")
    for n in sorted(rows):
        es = entries(rows[n])
        print(f"    row {n:>2}  " + "  ".join(
            f"w{r} {d}{'(closed)' if m else ''}" for r, d, m in es))
    if total < 40:
        fail(f"only {total} trail entries extracted; the matcher has stopped tracking the text")

    print("C1b the matcher tracks the text rather than agreeing by luck")
    probe = rows.get(6, "")
    base = len(entries(probe))
    renamed = len(entries(probe.replace("w153 09-02 (closed)", "w999 09-02 (closed)")))
    stripped = len(entries(probe.replace("→ w", "→ x")))
    demarked = sum(1 for e in entries(probe.replace(" (closed)", "")) if e[2])
    ok = base > 0 and renamed == base and stripped == 0 and demarked == 0
    print(f"  row 6: {base} entries; rename a run -> {renamed}; strip the arrows -> {stripped}; "
          f"strip the markers -> {demarked} marked  {'OK' if ok else 'FAIL'}")
    fails += not ok
    got = {r for r, _, _ in entries(probe.replace("w153 09-02 (closed)", "w999 09-02 (closed)"))}
    if 999 not in got:
        fail("the renamed run did not appear; C1's extraction is not reading these cells")

    print("C2 the deadline, read live off the competition object")
    if control:
        print(f"  --control: frozen at {DEADLINE_FULL}, no live read")
    else:
        got, how = live_deadline()
        if got is None:
            # Reported loudly and NOT counted: the suite already has network-dependent checks
            # and a second one would make an outage look like a defect. C6 says so out loud.
            print(f"  live read UNAVAILABLE ({how})")
            print(f"  falling back to the recorded {DEADLINE_FULL} -- NOT counted as a "
                  "failure, see C6")
        else:
            agree = got == DEADLINE_FULL
            print(f"  live deadline {got}   this guard compares against {DEADLINE_FULL}  "
                  f"{'OK' if agree else 'FAIL'}")
            fails += not agree

    print("C3a THE DEADLINE RULE -- marker vs the entry's own date, cell text only")
    dd = deadline_defects(rows)
    for n, run, why in dd:
        print(f"  row {n:>2}  w{run}: {why}")
    if dd:
        print(f"  {len(dd)} entr(ies) whose marker disagrees with their date. "
              + ("FIRES, as it must on --control. OK" if control else "FAIL"))
        fails += not control
    else:
        print("  every marker agrees with its date. "
              + ("FAIL -- control did not fire" if control else "OK"))
        fails += control

    print("C3b AGREEMENT -- trail date vs the journal header, marker vs the deadline")
    # Under --control the frozen journal map covers row 5's runs and no others, so C3b is
    # scoped to row 5 too. Otherwise it would report 38 "no journal header" lines for the 37
    # runs the map simply does not carry, and the control would be FIRING FOR THE WRONG
    # REASON -- green for a defect it is not testing, which is w156's lesson inverted.
    jd = CONTROL_JOURNAL if control else journal_dates()
    scope = {5: CONTROL_CELL} if control else rows
    bad = []
    for n in sorted(scope):
        for run, date, mark in entries(scope[n]):
            have = jd.get(run)
            if have is None:
                bad.append((n, run, "no JOURNAL.md run header"))
                continue
            if have != date:
                bad.append((n, run, f"trail says {date}, its journal header says {have}"))
            want = have > DEADLINE_DATE
            if want != mark:
                bad.append((n, run, f"({have}) marker={mark}, deadline says {want}"))
    for n, run, why in bad:
        print(f"  row {n:>2}  w{run}: {why}")
    if bad:
        print(f"  {len(bad)} disagreement(s). "
              + ("FIRES, as it must on --control. OK" if control else "FAIL"))
        fails += not control
    else:
        print(f"  all {sum(len(entries(c)) for c in scope.values())} entries agree with their journal headers and the deadline. "
              + ("FAIL -- control did not fire" if control else "OK"))
        fails += control

    print("C4 the anchors this guard copies rather than imports")
    hc = HANDCOUNT.read_text(encoding="utf-8")
    anchors = [
        (r'RUN_HDR = re\.compile\(r"\^#\{1,2\} \(20\\d\\d-\\d\\d-\\d\\d\|w\\d\+\[a-z\]\? —\|wave \|\\\(w\\d\+\[a-z\]\?,\)"\)',
         "w117a_handcount's RUN_HDR, copied verbatim into this file"),
    ]
    for pat, what in anchors:
        hit = re.search(pat, hc) is not None
        print(f"  {'OK  ' if hit else 'FAIL'} {what}")
        fails += not hit
    recguard = (HERE / "w156a_recordguard.py").read_text(encoding="utf-8")
    shared = '"w117a_handcount" in line' in recguard
    print(f"  {'OK  ' if shared else 'FAIL'} the row-identification rule #65/#66/#67 all "
          "share, still in w156a_recordguard")
    fails += not shared
    mine = RUN_HDR.pattern
    theirs = re.search(r'RUN_HDR = re\.compile\(r"([^"]+)"\)', hc)
    same = theirs is not None and theirs.group(1) == mine
    print(f"  {'OK  ' if same else 'FAIL'} the copy is byte-identical to the original")
    fails += not same

    print("C5 the control differs from the shipped text on the defect and nothing else")
    ship = table_rows(text)
    ship5 = entries(ship.get(5, ""))
    ctrl5 = entries(CONTROL_CELL)
    same_runs = [r for r, _, _ in ship5] == [r for r, _, _ in ctrl5]
    same_dates = [d for _, d, _ in ship5] == [d for _, d, _ in ctrl5]
    diff = [(r, a, b) for (r, _, a), (_, _, b) in zip(ship5, ctrl5) if a != b]
    ok = same_runs and same_dates and len(diff) == 1 and diff[0][0] == 143
    print(f"  shipped row 5 {[f'w{r}{d}{int(m)}' for r, d, m in ship5]}")
    print(f"  frozen  row 5 {[f'w{r}{d}{int(m)}' for r, d, m in ctrl5]}")
    print(f"  same runs {same_runs}, same dates {same_dates}, marker differs on "
          f"{[f'w{r}' for r, _, _ in diff]}  {'OK' if ok else 'FAIL'}")
    fails += not ok

    print("C6 what this guard is blind to, measured rather than assumed")
    print("  it reads the MARKER, not the run's actual submission behaviour. A run dated after")
    print("  the deadline that somehow submitted would be marked (closed) and pass here.")
    print("  the live deadline read is a CROSS-CHECK, not a dependency: an outage prints")
    print("  UNAVAILABLE and the recorded date is used, so a network failure cannot make a")
    print("  clean index look defective -- and a wrong constant survives only until the next")
    print("  run with network.")
    hist = 0
    for doc in (JOURNAL, ROOT / "LEADERBOARD.md"):
        t = doc.read_text(encoding="utf-8")
        hist += sum(1 for line in t.split("\n") if ENTRY.search(line))
    print(f"  {hist} line(s) in JOURNAL.md / LEADERBOARD.md carry trail-shaped text -- NOT")
    print("  enforced; both are append-only history and quote the index as it was that day.")

    print()
    print(f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
