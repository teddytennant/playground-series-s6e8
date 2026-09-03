"""w156a -- the ANGLE INDEX trail can name a run that left no journal entry, and the census
that is supposed to notice stays green because it is crediting that same run itself.

w155 was handed row 9, ran `w128a_row9.py`, shipped `w155a_poolguard.py`, wrote a full block
at the head of RESEARCH.md, bumped row 9 from x16 to x17 and extended its trail with
`-> w155 09-02 (closed)`. Then the run ended without ever writing its JOURNAL.md entry. The
suite log it left stops at [53/65].

🔴 AND `w117a_handcount` (#50) READ THAT STATE AS CLEAN. Its census counts JOURNAL.md run
headers against the index's handing counts, and at w117a_handcount.py:324-325 it adds one to
CURRENT_ROW when CURRENT_RUN's header is not in the corpus yet:

    if not any(re.search(CURRENT_RUN, r["header"]) for r in runs):
        measured[CURRENT_ROW] = measured.get(CURRENT_ROW, 0) + 1

That compensation is right, and its comment reasons carefully about TWO states -- before the
entry lands and after. There is a third. If the entry NEVER lands, the corpus is short by one
forever and the compensation covers it forever, so the run that vanishes is exactly the run
the census cannot see.

🎯 THE DEBT IS REAL, IT IS JUST DEFERRED, AND IT LANDS ON THE WRONG ROW. Measured live this
run by pointing CURRENT_RUN at w156/row 10 and re-running the census unchanged:

    FAIL: live row  9 (error analysis): index claims x17, corpus has x16
    FAIL: live row 10 (consolidation): index claims x16, corpus has x17

Two red rows, neither of them the defect. Row 10's is this run's ordinary bump; row 9's is
w155's missing entry surfacing a run late, attributed to a row nobody touched today. A future
reader gets a census failure pointing at error analysis for a fault in record-keeping.

⚠ THE GENUS, AND WHY IT IS NOT THIS WEEK'S. w150-w154 were all reads that LOOK complete and
are not, and w155 was a record that looks inconsistent and is not. This is neither: the read
is complete, the record is consistent, and the check is CORRECT -- it is just correct about a
window it cannot tell it has left. A self-cancelling adjustment hides the thing it adjusts for.

    C1  the trail entries in the ANGLE INDEX handing-count cells, extracted, with C1b proving
        the matcher tracks the text rather than agreeing by luck.
    C2  the assertion: every run named in a trail has a JOURNAL.md run header, matched with
        w117a_handcount's own RUN_HDR so the two agree on what a run is.
    C3  the compensation is BOUNDED. At most one trail run may be uncovered, and it must be
        the one w117a_handcount is currently crediting. That is the whole fix -- the +1 is
        legitimate for the in-flight run and for no other.
    C4  the anchors. RUN_HDR and the compensation must still read the way this guard assumes;
        if either moves, C4 goes red BEFORE C3 goes quietly wrong.
    C5  --control over a frozen trail cell naming a run with no entry -- w155's actual pre-fix
        state -- asserting both directions itself.
    C6  scope and blindness, measured rather than assumed.

    .venv/bin/python experiments/w156a_recordguard.py            # rc 0 = clean
    .venv/bin/python experiments/w156a_recordguard.py --control  # exits 0, having fired

Exits 0 when every assertion holds, 1 otherwise, and prints FAILURES: n as its last line.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"
JOURNAL = ROOT / "JOURNAL.md"
HANDCOUNT = ROOT / "experiments" / "w117a_handcount.py"

# Copied from w117a_handcount.py:137 on purpose, not imported. Importing would make the two
# agree by construction and this guard would have nothing to say about drift; C4 asserts the
# copy is still faithful, which is the check the import would have thrown away.
RUN_HDR = re.compile(r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,|══ 20\d\d-\d\d-\d\d)")

# The anchors C4 stands on. Each is a line that must still be in w117a_handcount.py, and each
# is the line that DECIDES something this guard assumes.
ANCHORS = {
    "the RUN_HDR this guard copied":
        r'RUN_HDR = re.compile(r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,|══ 20\d\d-\d\d-\d\d)")',
    "the compensation C3 bounds":
        "measured[CURRENT_ROW] = measured.get(CURRENT_ROW, 0) + 1",
    "the condition that makes it self-cancelling":
        'if not any(re.search(CURRENT_RUN, r["header"]) for r in runs):',
}

# C5's specimen: row 9's count cell exactly as w155 left it, naming a run with no entry.
CONTROL_CELL = ("| 9 | *error analysis: find where the best model is wrong* | **x17, from 08-11 "
                "-> w128 08-30 -> w137 08-31 -> w146 09-01 (closed) -> w155 09-02 (closed) * "
                "artefacts verified** (count from `w117a_handcount`) | none |")

TRAIL = re.compile(r"→ w(\d+) \d\d-\d\d")
ROW = re.compile(r"^\|\s*(\d+)\s*\|")
# The control cell is frozen ASCII, so it needs the ASCII arrow too.
TRAIL_ANY = re.compile(r"(?:→|->) w(\d+) \d\d-\d\d")


def table_rows(text: str) -> dict[int, str]:
    """The ANGLE INDEX rows, keyed by row number.

    Same identification rule as w155a: the ten-row block whose cells carry the handing counts.
    A row number alone appears in half a dozen other tables in this file.
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


def journal_run_ids() -> tuple[set[int], int]:
    """Every wNNN named by a JOURNAL.md run header, and how many headers there are.

    A header can name more than one run id (`wave w26, slot 4 of 10` names w26 once; some
    headers quote a second wave in prose), so this collects all of them rather than the first.
    """
    ids: set[int] = set()
    n = 0
    for line in JOURNAL.read_text(encoding="utf-8").split("\n"):
        if not RUN_HDR.match(line):
            continue
        n += 1
        ids |= {int(m.group(1)) for m in re.finditer(r"\bw(\d+)\b", line)}
    return ids, n


def current_run() -> tuple[int | None, int | None]:
    """The run w117a_handcount is currently crediting, read out of its source.

    Read, not hardcoded: the whole point of C3 is that the exemption follows whatever that
    file says, so that moving CURRENT_RUN without writing the entry cannot widen it.
    """
    m = re.search(r'CURRENT_RUN, CURRENT_ROW = r"([^"]+)", (\d+)', HANDCOUNT.read_text())
    if not m:
        return None, None
    w = re.search(r"w(\d+)", m.group(1))
    return (int(w.group(1)) if w else None), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C2 over a frozen trail cell naming a run with no journal entry")
    a = ap.parse_args()
    fails = 0

    def fail(msg: str) -> None:
        nonlocal fails
        fails += 1
        print(f"  FAIL: {msg}")

    text = RESEARCH.read_text()
    rows = table_rows(text)
    ids, n_hdr = journal_run_ids()
    cur_run, cur_row = current_run()

    if a.control:
        # w72 5.3's shape: the control asserts BOTH directions itself and returns 0. A control
        # that only makes the shipped run red says nothing about whether the check has power.
        print("w156a --control: C2 against the state w156 actually found\n")
        # The state, reconstructed rather than borrowed from the live one: w155 is named in a
        # trail, w155 has no journal header, and w117a_handcount has already moved on to w156
        # so the C3 exemption no longer covers it. Freezing all three keeps the control's
        # power independent of the fix -- borrowing the live `ids` would make it silent again
        # the moment w155's entry landed, which is exactly when it must still work.
        ids_pre, cur_pre = ids - {155}, 156
        frozen = sorted({int(m.group(1)) for m in TRAIL_ANY.finditer(CONTROL_CELL)})
        shipped = sorted({int(m.group(1)) for m in TRAIL.finditer(rows.get(9, ""))})
        f_miss = [w for w in frozen if w not in ids_pre and w != cur_pre]
        s_miss = [w for w in shipped if w not in ids and w != cur_run]
        print(f"  frozen trail {frozen} against the pre-fix corpus, credit w{cur_pre}")
        print(f"    uncovered {f_miss}  {'FIRES' if f_miss else 'SILENT (BAD)'}")
        print(f"  shipped trail {shipped} against the live corpus, credit w{cur_run}")
        print(f"    uncovered {s_miss}  {'SILENT' if not s_miss else 'FIRES (BAD)'}")
        print(f"\n  the control {'WORKS' if f_miss and not s_miss else 'DOES NOT WORK'}")
        return 0

    print("C1 the trail entries in the ANGLE INDEX handing-count cells")
    if len(rows) != 10:
        fail(f"the table resolved to {len(rows)} rows, not 10")
    trail: dict[int, list[int]] = {}
    for n in sorted(rows):
        trail[n] = sorted({int(m.group(1)) for m in TRAIL.finditer(rows[n])})
        print(f"  row {n:2d}  trail {trail[n]}")
    total = sum(len(v) for v in trail.values())
    print(f"  {total} trail entries across {len(trail)} rows")
    if total < 30:
        fail(f"only {total} trail entries parsed -- the cell format has moved and this guard "
             f"is no longer reading it")

    print("C1b the matcher tracks the text rather than agreeing by luck")
    probe = rows.get(9, "").replace("→ w128", "→ w9999")
    got = sorted({int(m.group(1)) for m in TRAIL.finditer(probe)})
    if 9999 in got and 128 not in got:
        print("  renaming one trail entry moves the extraction: 128 -> 9999  OK")
    else:
        fail(f"perturbing row 9's trail changed nothing -- extraction reads {got}")
    if not TRAIL.search(rows.get(9, "").replace("→ w", "→ x")):
        print("  removing the arrow form yields nothing at all  OK")
    else:
        fail("the matcher still fires on a cell with no trail arrows")

    print("C2 every run named in a trail has a JOURNAL.md run header")
    print(f"  {n_hdr} run header(s) in the corpus, naming {len(ids)} distinct run id(s)")
    uncovered: list[tuple[int, int]] = []
    for n in sorted(trail):
        miss = [w for w in trail[n] if w not in ids]
        for w in miss:
            uncovered.append((n, w))
    for n, w in uncovered:
        if w == cur_run:
            print(f"  row {n:2d}  w{w} has no entry yet -- it is the in-flight run, exempt "
                  f"under C3")
        else:
            fail(f"row {n} names w{w} as a handing and JOURNAL.md has no run header for it -- "
                 f"the index counts a run that left no record")
    if not uncovered:
        print("  every trail entry is backed by a run header  OK")

    print("C3 the compensation w117a_handcount applies is bounded to ONE run")
    if cur_run is None or cur_row is None:
        fail("could not read CURRENT_RUN/CURRENT_ROW out of w117a_handcount.py")
    else:
        print(f"  w117a_handcount credits w{cur_run} to row {cur_row}")
        strays = [(n, w) for n, w in uncovered if w != cur_run]
        if strays:
            fail(f"{len(strays)} trail run(s) uncovered that are NOT the in-flight run: "
                 f"{[f'row {n}/w{w}' for n, w in strays]}")
        else:
            print(f"  uncovered trail runs: {len(uncovered)}, all of them w{cur_run}  OK")
        # The exemption must be spendable at most once. Two rows both claiming the current run
        # would let a second unwritten entry ride in behind the first.
        rows_claiming = sorted({n for n, w in uncovered if w == cur_run})
        if len(rows_claiming) > 1:
            fail(f"w{cur_run} is exempt on {len(rows_claiming)} rows at once: {rows_claiming}")
        elif rows_claiming and rows_claiming[0] != cur_row:
            fail(f"w{cur_run} is uncovered on row {rows_claiming[0]} but credited to row "
                 f"{cur_row} -- the compensation lands on the wrong row")

    print("C4 the anchors this guard reasons about are still in w117a_handcount.py")
    src = HANDCOUNT.read_text()
    for what, needle in ANCHORS.items():
        if needle in src:
            print(f"  OK   {what}")
        else:
            fail(f"{what}: the line this guard assumes is gone from w117a_handcount.py")

    print("C6 scope and blindness, measured")
    print(f"  {total} trail entries checked; {len(uncovered)} uncovered, "
          f"{len([1 for _, w in uncovered if w == cur_run])} of them exempt")
    print("  it reads the TRAIL, not the count. A row whose x-count is wrong but whose trail")
    print("    is fully backed passes here -- that is w117a_handcount's job and it keeps it.")
    print("  a run that leaves artefacts but is never named in any trail is invisible: the")
    print("    index is the only place this guard looks.")
    print("  it cannot tell a reconstructed entry from one written live. C2 asks whether a")
    print("    header exists, not whether the run that would have written it did the work.")

    print(f"\nFAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
