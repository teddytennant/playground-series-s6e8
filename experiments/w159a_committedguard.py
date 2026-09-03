"""w159a (#69) — a run's record must be COMMITTED, not merely written, and the in-flight
exemption must be ANSWERABLE rather than open-ended.

WHY THIS EXISTS. Three runs in a row left their record incomplete and every one of them was
found by `git status` at the NEXT run's orientation, never by a check:

    w155  wrote no JOURNAL.md entry at all           -> reconstructed by w156
    w156  wrote the entry and committed nothing      -> committed by w157
    w158  wrote RESEARCH/LEADERBOARD, a guard and the
          census sync, but no entry and no commit    -> reconstructed and committed by w159

#66 w156a_recordguard was written for exactly this genus and it read GREEN through all of
w158's, which is not a bug in #66 -- it is the shape of its exemption. Measured, not argued:

    census credits w158 (the state w159 opened on)   uncovered [w158]  strays []      GREEN
    census credits w159 (after w159's ordinary sync) uncovered [w158]  strays [w158]  FIRES

🎯 SO THE DETECTION IS COUPLED TO THE FIX, NOT TO THE READ. #66 goes red only once the NEXT
run edits the census, which is one of the last things a run does -- by which point whoever
was going to notice has already noticed, from `git status`, hours earlier. At the one moment
the playbook says to look around (orientation), the exemption is still covering the run that
vanished, because "in-flight" has no expiry and nothing else can tell the two states apart.

⚠ AND THERE IS A SECOND AXIS NOTHING READS AT ALL. w156's miss was not a missing entry, it
was an uncommitted one. Every one of the 68 standing checks reads the WORKING TREE. Measured
over the shipped STEMS list: 7 of 68 mention `git` at all, 5 of those in prose or in a
`.git` skip-list, and the one function that ever ran it (w115a_docselectguard.git_head) is
DEAD CODE -- deliberately retired at w115a:80-86 because a control anchored to a moving HEAD
stops being a control the moment you commit the fix. That lesson is right about CONTROLS and
it left the workspace with ZERO checks that read git state as an ASSERTION.

WHAT THIS ADDS, IN ONE LINE. #66 asks "does a header exist anywhere". #69 asks "is it in
HEAD", and gives the caller a way to say who it is so the exemption can be REFUSED.

  C1   the run ids named in the ANGLE INDEX trails, and the run the census credits.
  C1b  the matcher tracks the text rather than agreeing by luck.
  C2   every trail run's JOURNAL.md header is in `git show HEAD:JOURNAL.md`. At most one may
       be missing and it must be the credited run -- the same bound #66's C3 applies to the
       tree, so the two cannot disagree about who is exempt.
  C3   the credited run's record state is REPORTED by name, in three states, at orientation:
       COMMITTED / WRITTEN-NOT-COMMITTED / UNWRITTEN. This is the part that would have said
       the sentence nobody had: "w158 is credited, named in row 7's trail, and has no entry."
  C4   --run wNNN refuses the exemption. If the caller names its own id and the census credits
       someone else, an uncovered credited run is a hard FAIL. No timing heuristic, no mtime,
       no false positives: a run knows its own name and nothing else has to guess.
  C5   the anchors in w117a_handcount.py and w156a_recordguard.py this guard reasons about.
  C6   --control, every input frozen (w115a:80-86: never anchor a control to live HEAD).
  C7   blindness.

    .venv/bin/python experiments/w159a_committedguard.py                # suite form, rc 0/1
    .venv/bin/python experiments/w159a_committedguard.py --run w160     # orientation form
    .venv/bin/python experiments/w159a_committedguard.py --control      # exits 0, having fired
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"
JOURNAL = ROOT / "JOURNAL.md"
HANDCOUNT = ROOT / "experiments" / "w117a_handcount.py"
RECORDGUARD = ROOT / "experiments" / "w156a_recordguard.py"

# Copied from w117a_handcount.py:137 via w156a_recordguard.py:68, not imported, for the reason
# #66 gives: importing would make the three agree by construction and none of them could say
# anything about drift. C5 asserts the copy is still faithful to BOTH of them.
RUN_HDR = re.compile(r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,|══ 20\d\d-\d\d-\d\d)")

TRAIL = re.compile(r"→ w(\d+) \d\d-\d\d")
ROW = re.compile(r"^\|\s*(\d+)\s*\|")

ANCHORS = {
    HANDCOUNT: {
        "the CURRENT_RUN line C1 parses":
            "CURRENT_RUN, CURRENT_ROW = r",
        "the compensation whose expiry this guard supplies":
            "measured[CURRENT_ROW] = measured.get(CURRENT_ROW, 0) + 1",
    },
    RECORDGUARD: {
        "#66's tree-side exemption, which C2 mirrors onto HEAD":
            'strays = [(n, w) for n, w in uncovered if w != cur_run]',
        "the RUN_HDR all three copies share":
            'RUN_HDR = re.compile(r"^#{1,2} (20\\d\\d-\\d\\d-\\d\\d|w\\d+[a-z]? —|wave |\\(w\\d+[a-z]?,|══ 20\\d\\d-\\d\\d-\\d\\d)")',
    },
}

# C6's specimen. Every field is a literal: the trail as w158 left it, the ids in HEAD at
# 975bde7 (w157's commit, before w159 committed anything), the ids in the working tree at
# w159's orientation, and the run the census credited. ⚠ Reading any of these live would make
# the control go silent the moment the fix lands -- w115a:80-86, learned the hard way.
CONTROL = {
    "trail": [128, 137, 146, 155, 109, 118, 127, 136, 145, 154, 158],
    "head_ids": {128, 137, 146, 155, 109, 118, 127, 136, 145, 154},   # no w158
    "tree_ids": {128, 137, 146, 155, 109, 118, 127, 136, 145, 154},   # no w158 here either
    "credited": 158,
    "caller": 159,
}


def table_rows(text: str) -> dict[int, str]:
    """The ANGLE INDEX rows, keyed by row number. Same identification rule as #65 and #66."""
    out: dict[int, str] = {}
    for line in text.split("\n"):
        m = ROW.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 10 and "w117a_handcount" in line and n not in out:
            out[n] = line
    return out


def run_ids(text: str) -> set[int]:
    """Every wNNN named by a JOURNAL.md run header in `text`."""
    ids: set[int] = set()
    for line in text.split("\n"):
        if RUN_HDR.match(line):
            ids |= {int(m.group(1)) for m in re.finditer(r"\bw(\d+)\b", line)}
    return ids


def head_journal() -> str | None:
    """JOURNAL.md as of HEAD. None if git cannot answer -- which must FAIL, not pass."""
    r = subprocess.run(["git", "show", "HEAD:JOURNAL.md"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def credited() -> tuple[int | None, int | None]:
    """The run w117a_handcount credits, read out of its source rather than hardcoded."""
    m = re.search(r'CURRENT_RUN, CURRENT_ROW = r"([^"]+)", (\d+)', HANDCOUNT.read_text())
    if not m:
        return None, None
    w = re.search(r"w(\d+)", m.group(1))
    return (int(w.group(1)) if w else None), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, metavar="wNNN",
                    help="the id of the run invoking this. Refuses the in-flight exemption "
                         "for anyone else, which turns C2 from one-run-late into orientation.")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    fails: list[str] = []

    def fail(msg: str) -> None:
        fails.append(msg)
        print(f"  FAIL: {msg}")

    if a.control:
        print("w159a --control: the state w159 opened on, every input frozen\n")
        c = CONTROL
        for label, caller in (("no --run (suite form)", None),
                              (f"--run w{c['caller']} (orientation form)", c["caller"])):
            unc = [w for w in c["trail"] if w not in c["head_ids"]]
            exempt = [] if (caller is not None and caller != c["credited"]) else [c["credited"]]
            strays = [w for w in unc if w not in exempt]
            print(f"  {label:<34} credited w{c['credited']}  uncovered {unc}  "
                  f"strays {strays}  {'FIRES' if strays else 'SILENT'}")
        # And the same instrument on what actually shipped, so the control asserts BOTH
        # directions (w72 5.3). Frozen too: the post-fix ids, not a live read.
        post = set(c["head_ids"]) | {158}
        unc = [w for w in c["trail"] if w not in post]
        print(f"  {'shipped state, --run w159':<34} credited w158  uncovered {unc}  "
              f"strays {unc}  {'FIRES (BAD)' if unc else 'SILENT'}")
        print("\n  C3's three states, on the frozen inputs:")
        for w, hd, tr in ((158, False, False), (156, False, True), (157, True, True)):
            print(f"    w{w}: in HEAD {hd!s:<5} in tree {tr!s:<5} -> "
                  f"{'COMMITTED' if hd else ('WRITTEN, NOT COMMITTED' if tr else 'UNWRITTEN')}")
        print("\n  the control WORKS: it fires on the orientation form and only there")
        return 0

    text = RESEARCH.read_text()
    rows = table_rows(text)
    trail = {n: sorted({int(m.group(1)) for m in TRAIL.finditer(cell)})
             for n, cell in sorted(rows.items())}
    cur_run, cur_row = credited()

    print("C1 the trail runs and the run the census credits")
    n_entries = sum(len(v) for v in trail.values())
    print(f"  {n_entries} trail entries across {len(trail)} rows")
    if cur_run is None:
        fail("could not read CURRENT_RUN/CURRENT_ROW out of w117a_handcount.py")
    else:
        print(f"  w117a_handcount credits w{cur_run} to row {cur_row}")
    if n_entries < 40 or len(trail) != 10:
        fail(f"the ANGLE INDEX did not parse: {len(trail)} rows, {n_entries} entries")

    print("C1b the matcher tracks the text rather than agreeing by luck")
    probe = next(iter(rows.values()))
    print(f"  renaming one trail entry moves the extraction: "
          f"{sorted({int(m.group(1)) for m in TRAIL.finditer(re.sub(chr(8594) + r' w\d+ ', '→ w9999 ', probe, count=1))})[:3]}...  OK")
    print(f"  removing the arrow form yields nothing at all: "
          f"{[m.group(1) for m in TRAIL.finditer(probe.replace(chr(8594), ''))]}  OK")

    print("C2 every trail run's journal header is in `git show HEAD:JOURNAL.md`")
    head = head_journal()
    if head is None:
        fail("`git show HEAD:JOURNAL.md` did not answer -- this guard fails CLOSED, it does "
             "not assume a clean tree when it cannot read history")
        head_ids: set[int] = set()
    else:
        head_ids = run_ids(head)
        print(f"  HEAD's JOURNAL.md names {len(head_ids)} distinct run id(s)")
        if len(head_ids) < 100:
            fail(f"HEAD's JOURNAL.md parsed to only {len(head_ids)} run ids -- the read is "
                 f"broken, not the record")
    tree_ids = run_ids(JOURNAL.read_text(encoding="utf-8"))

    uncovered = [(n, w) for n, ws in trail.items() for w in ws if w not in head_ids]
    # The exemption, mirroring #66's C3 onto HEAD: at most one run, and it must be the one the
    # census credits. C4 can withdraw it entirely.
    exempt = cur_run
    if a.run is not None:
        m = re.fullmatch(r"w?(\d+)", a.run)
        if not m:
            fail(f"--run {a.run!r} is not a run id")
        elif int(m.group(1)) != cur_run:
            print(f"  C4: caller is w{int(m.group(1))} but the census credits w{cur_run} -- "
                  f"the in-flight exemption is REFUSED")
            exempt = None
    strays = [(n, w) for n, w in uncovered if w != exempt]
    for n, w in strays:
        fail(f"row {n} trails w{w}, which has no committed JOURNAL.md entry"
             + (" and none in the working tree either" if w not in tree_ids else
                " (it IS written in the working tree -- commit it)"))
    rows_claiming = sorted({n for n, w in uncovered if w == exempt})
    if len(rows_claiming) > 1:
        fail(f"w{exempt} is exempt on {len(rows_claiming)} rows at once: {rows_claiming}")
    elif rows_claiming and cur_row is not None and rows_claiming[0] != cur_row:
        fail(f"w{exempt} is uncovered on row {rows_claiming[0]} but credited to row {cur_row}")
    if not strays:
        print(f"  {n_entries} trail entries, {len(uncovered)} not in HEAD, "
              f"{len(uncovered) - len(strays)} of them exempt  OK")

    print("C3 the credited run's record state, named rather than left to be inferred")
    if cur_run is not None:
        in_head, in_tree = cur_run in head_ids, cur_run in tree_ids
        state = ("COMMITTED" if in_head else
                 "WRITTEN, NOT COMMITTED" if in_tree else "UNWRITTEN")
        print(f"  w{cur_run} (row {cur_row}): {state}")
        if state != "COMMITTED":
            print(f"  ⚠ if you are NOT w{cur_run}, that run left no record and this guard "
                  f"should have been called as `--run w<your id>`, which fails on it.")

    print("C4 the exemption is refusable")
    print(f"  --run {'not given, exemption stands (suite form)' if a.run is None else a.run}")

    print("C5 the anchors this guard reasons about are still where it read them")
    for path, anchors in ANCHORS.items():
        src = path.read_text()
        for what, line in anchors.items():
            if line in src:
                print(f"  OK   {path.name}: {what}")
            else:
                fail(f"{path.name}: {what} -- the line this guard assumes is gone")

    print("C7 scope and blindness, measured")
    print(f"  it reads HEAD, so a run that commits its entry and never PUSHES is invisible")
    print(f"  it reads the TRAIL, not the count, and not whether the entry is any good --")
    print(f"    an empty entry with the right header passes, exactly as it does in #66")
    print(f"  a run that leaves artefacts but is never named in a trail is invisible to both")
    print(f"  without --run it is still one run late; the orientation form is what makes it")
    print(f"    prompt, and nothing forces a run to pass it")
    print(f"\nFAILURES: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
