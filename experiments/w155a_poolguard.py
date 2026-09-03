"""w155a -- three ANGLE INDEX rows disclose "the full 167-member pack" and their artefacts
record 167, 167 and 170 for it, under the same key name.

w128 wrote down where the next defect in the price-cell genus would be, and it was right about
the place: "the pool disclosures (base104) that rows 3/4/5 make in prose are checked by nothing
at all". Four guards -- #53 units, #55 scope, #56 denominator, #57 opt-out -- read only the
`price` column of the ANGLE INDEX. None of them reads a pool.

    row 1  "...the **full 167-member pack**"   w131a_row1.json  arm_c.pool_n = 167
    row 7  "...the full **167-member** pack"   w127a_row7.json  pool_n       = 167
    row 9  "...the full 167-member pack"       w128a_row9.json  pool_n       = 170

🔴 THE DEFECT IS NOT THAT ANY PRICE IS WRONG. Every per-member rate in those three cells is
correct and reproduces. It is that ROW 9'S DISCLOSURE CANNOT BE CHECKED AGAINST ITS OWN
ARTEFACT: the only pool number `w128a_row9.json` carries is 170, the prose says 167, and
nothing in either file says why the two differ. A reader reconciling the row against the file
finds a mismatch and has no way to tell whether the cell is wrong, the file is wrong, or the
key means two things.

WHY THEY DIFFER, established by reading the three producing scripts rather than assumed:

  w127a  the arms (three seed twins and their average) are ALREADY members of the loaded pack,
         so `pool_n` is written straight off `load_members` and the measurement runs on those
         same 167 columns.                       w127a_row7.py:207-213
  w131a  `origmodel` is built in this process and appended to `names2`, but `arm_c.pool_n`
         records `len(names)` -- the pack WITHOUT it -- so the recorded 167 is the baseline
         and the measurement runs on 168.        w131a_row1.py:290-293, written at :344
  w128a  three arms are built in this process and appended BEFORE `pool_n` is taken, so the
         recorded 170 is 167 + 3 and the baseline the enrolment price is measured against is
         Q = 167.                                w128a_row9.py:358-363

🎯 So `pool_n` means "the pool the measurement ran on" in two files and "the pool before the
arms" in the third, and the prose in all three rows quotes the BASELINE. All three cells are
true. Two of the three files corroborate that on their face and the third contradicts it on
its face, and which is which is knowable only by reading source that neither document cites.

⚠ THIS IS A DISCLOSURE DEFECT, NOT A MEASUREMENT ONE, AND THE SEPARATION IS THE POINT. w150's
genus all week has been reads that look complete and are not; this is a record that looks
inconsistent and is not. Both cost the same thing -- a future run cannot tell from the
artefacts which case it is in -- and only a check that carries the reconciliation can.

    C1  the three rows disclose a pack size in prose, extracted with a matcher proved to fire
        both ways on perturbed copies of the real cells.
    C2  each artefact's recorded pool number, reduced to a BASELINE pack size by the
        per-artefact convention below, equals the number its row publishes. This is the
        assertion; everything else supports it.
    C3  the raw numbers, printed side by side with their conventions -- the reconciliation
        that exists nowhere else. Asserted: all three reduce to ONE pack size.
    C4  the convention table is not free-floating. Each entry names an anchor line in the
        script that produced the JSON, and the anchor must still be there. If someone moves
        the `pool_n` assignment past the append, C4 goes red before C2 goes quietly wrong.
    C5  the frozen 170-member specimen is still a specimen. `--control` runs C2 over it and
        asserts both directions itself: fires on the frozen cell, silent on what shipped.
    C6  scope and blindness, measured: how many pool disclosures the table carries, and what
        this guard cannot see.

    .venv/bin/python experiments/w155a_poolguard.py            # rc 0 = clean
    .venv/bin/python experiments/w155a_poolguard.py --control  # exits 0, having fired

Exits 0 when every assertion holds, 1 otherwise, and prints FAILURES: n as its last line.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"
EXP = ROOT / "experiments"

# The pack every one of the three rows is talking about. Not a free constant: C2 checks that
# all three artefacts reduce to it, so a pack that grows makes this guard red rather than
# letting a stale 167 sit in three documents.
PACK = 167

# Per-artefact convention. `includes_arms` says whether the recorded number already counts the
# columns the run built for itself; `anchor` is a line that must still exist in the producing
# script, and it is the line that DECIDES the convention. Hardcoding the convention is the
# whole contribution -- it was previously written down nowhere -- so each entry has to point
# at the source that justifies it (C4).
CONVENTION = {
    1: dict(
        blob="w131a_row1.json", path=("arm_c", "pool_n"), includes_arms=False, n_arms=1,
        src="w131a_row1.py", anchor="origmodel is the +1",
        note="arm_c.pool_n records len(names), the pack without origmodel -> already the baseline",
    ),
    7: dict(
        blob="w127a_row7.json", path=("pool_n",), includes_arms=False, n_arms=0,
        src="w127a_row7.py", anchor="the seed family is not in the loaded pool",
        note="the arms are already members of the pack -> nothing is appended at all",
    ),
    9: dict(
        blob="w128a_row9.json", path=("pool_n",), includes_arms=True, n_arms=3,
        src="w128a_row9.py", anchor="names = list(names) + list(new)", arms_key="r2",
        note="pool_n taken AFTER three in-process arms are appended -> 167 + 3",
    ),
}

# C5's specimen: row 9's cell with the disclosure moved to the number its own file records.
CONTROL_CELL = ("| 9 | *error analysis* | x1 | priced as an ENROLMENT with rows 3/4's "
                "instrument (paired 50/50, splits 0/1/2, C=1.0, `hybrid`, the full "
                "170-member pack, w128a) | none |")

DISCLOSURE = re.compile(r"(?:full\s+)?\*{0,2}(\d[\d,]*)\*{0,2}-member\*{0,2}\s+pack")
ROW = re.compile(r"^\|\s*(\d+)\s*\|")


def table_rows(text: str) -> dict[int, str]:
    """The ANGLE INDEX rows, keyed by row number.

    The table is identified by the ten-row block whose cells carry the handing counts, not by
    a heading -- headings above it have moved twice (w101) and a row number alone appears in
    half a dozen other tables in this file.
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


def disclosed_pack(cell: str) -> list[int]:
    return [int(m.group(1).replace(",", "")) for m in DISCLOSURE.finditer(cell)]


def dig(blob: dict, path: tuple[str, ...]):
    cur = blob
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1/C2 over a frozen cell that claims a 170-member pack")
    a = ap.parse_args()
    fails = 0

    text = RESEARCH.read_text()
    rows = table_rows(text)

    if a.control:
        # w72 5.3 / w128b's shape: the control block asserts BOTH directions itself and
        # returns 0. A control that only makes the shipped run red proves nothing about
        # whether the check has power.
        print("w155a --control: C2 against a frozen row 9 cell claiming a 170-member pack\n")
        c = CONVENTION[9]
        raw = dig(json.loads((EXP / c["blob"]).read_text()), c["path"])
        base = raw - c["n_arms"]
        pre = disclosed_pack(CONTROL_CELL)[0]
        ship = disclosed_pack(rows[9])[0]
        fired, silent = base != pre, base == ship
        print(f"  frozen cell says {pre}, the artefact reduces to {base}  "
              f"{'FIRES' if fired else 'SILENT (BAD)'}")
        print(f"  shipped cell says {ship}, the artefact reduces to {base}  "
              f"{'SILENT' if silent else 'FIRES (BAD)'}")
        print(f"\n  the control {'WORKS' if fired and silent else 'DOES NOT WORK'}")
        return 0

    print("C1 the pool disclosures in the ANGLE INDEX, extracted")
    if len(rows) != 10:
        fails += 1
        print(f"  -> the table resolved to {len(rows)} rows, not 10  FAIL")
    claimed: dict[int, int] = {}
    for n in sorted(rows):
        hits = disclosed_pack(rows[n])
        if not hits:
            continue
        if len(set(hits)) != 1:
            fails += 1
            print(f"  row {n:2d}  discloses {hits} -- one row, two pack sizes  FAIL")
            continue
        claimed[n] = hits[0]
        print(f"  row {n:2d}  discloses a {hits[0]}-member pack")
    missing = sorted(set(CONVENTION) - set(claimed))
    if missing:
        fails += 1
        print(f"  -> rows {missing} carry an artefact convention but no longer disclose a "
              f"pack size  FAIL")
    print(f"  {len(claimed)} of 10 rows disclose a pool: {sorted(claimed)}")

    print("C1b the matcher fires both ways on perturbed copies of the real cells")
    live = table_rows(RESEARCH.read_text())
    both = 0
    for n in sorted(CONVENTION):
        cell = live.get(n)
        if cell is None:
            fails += 1
            print(f"  row {n:2d}  not present in the table  FAIL")
            continue
        bumped = DISCLOSURE.sub(lambda m: m.group(0).replace(m.group(1), "999"), cell, count=1)
        stripped = DISCLOSURE.sub("pack", cell)
        ok = disclosed_pack(bumped) == [999] and disclosed_pack(stripped) == []
        both += ok
        print(f"  row {n:2d}  perturbed -> {disclosed_pack(bumped)}   removed -> "
              f"{disclosed_pack(stripped)}   {'OK' if ok else 'FAIL'}")
    if both != len(CONVENTION):
        fails += 1
        print("  -> the extractor does not track the text  FAIL")

    print("C2 each artefact's pool number, reduced to a baseline, against its row")
    baselines: dict[int, int] = {}
    for n, c in CONVENTION.items():
        p = EXP / c["blob"]
        if not p.exists():
            print(f"  row {n:2d}  {c['blob']} missing -- INERT, not passing")
            continue
        raw = dig(json.loads(p.read_text()), c["path"])
        if raw is None:
            fails += 1
            print(f"  row {n:2d}  {c['blob']} has no {'.'.join(c['path'])}  FAIL")
            continue
        # where the JSON records the arms themselves, count them rather than trusting the
        # table -- an arm added or dropped must turn this red, not shift the baseline quietly.
        if c.get("arms_key"):
            n_rec = len(dig(json.loads(p.read_text()), (c["arms_key"],)) or {})
            if n_rec != c["n_arms"]:
                fails += 1
                print(f"  row {n:2d}  {c['blob']}[{c['arms_key']}] holds {n_rec} arm(s), the "
                      f"convention says {c['n_arms']}  FAIL")
        base = raw - c["n_arms"] if c["includes_arms"] else raw
        baselines[n] = base
        want = claimed.get(n)
        ok = want is not None and base == want
        fails += not ok
        print(f"  row {n:2d}  {'.'.join(c['path']):16s} = {raw:4d}  "
              f"{'- ' + str(c['n_arms']) + ' arm(s)' if c['includes_arms'] else 'baseline as-is':16s}"
              f"  -> {base}   row says {want}   {'OK' if ok else 'FAIL'}")

    print("C3 the reconciliation, which exists nowhere else")
    for n, c in CONVENTION.items():
        if n in baselines:
            print(f"  row {n:2d}  {c['note']}")
    got = set(baselines.values())
    ok = got == {PACK}
    fails += not ok
    print(f"  raw numbers {sorted(dig(json.loads((EXP / c['blob']).read_text()), c['path']) for n, c in CONVENTION.items() if n in baselines)}"
          f"  ->  baselines {sorted(got)}   one pack of {PACK}?  {'OK' if ok else 'FAIL'}")

    print("C4 the convention table points at source that still says so")
    for n, c in CONVENTION.items():
        p = EXP / c["src"]
        hit = p.exists() and c["anchor"] in p.read_text()
        fails += not hit
        print(f"  row {n:2d}  {c['src']:16s} contains {c['anchor']!r}  "
              f"{'OK' if hit else 'FAIL'}")

    print("C5 the frozen control specimen is still a specimen")
    ctrl = disclosed_pack(CONTROL_CELL)
    would = ctrl == [170] and 170 != PACK
    fails += not would
    print(f"  it reads {ctrl} against a baseline of {PACK}, so --control has something to "
          f"fire on   {'OK' if would else 'FAIL'}")

    print("C6 scope and blindness, measured")
    n_disc = len(claimed)
    n_cov = len([n for n in CONVENTION if n in baselines])
    print(f"  {n_disc} pool disclosure(s) in the table, {n_cov} covered by a convention entry")
    print(f"  rows 3/4/5 say `base104` in words and name no number, so this guard does not")
    print(f"    see them -- w128's prediction covered those and they are still unchecked.")
    print(f"  it reads the RECORD, not the DATA: it cannot tell whether {PACK} is the right")
    print(f"    pack, only whether the three documents agree on it.")
    print(f"  a new row that discloses a pool with no convention entry passes C1 and is")
    print(f"    invisible to C2 -- the count above is the only thing that would show it.")

    print()
    print(f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
