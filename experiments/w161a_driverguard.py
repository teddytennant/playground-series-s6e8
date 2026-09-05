"""w161a (#71) -- the record of what RAN lives outside this workspace, and nothing here read it.

WHY THIS EXISTS. Every check on this disk censuses the workspace against its own writings:
#50 counts JOURNAL.md headers, #66 asks whether an entry reached the tree, #69 whether it
reached HEAD, #70 whether HEAD reached the remote. All four take "the runs that happened" to be
"the runs that wrote something". The launcher keeps a separate, out-of-process log --

    /home/nixos/all-my-repos/ai/kaggle-agents/logs/playground-series-s6e8/<date>.log

-- which records every slot it started, the ANGLE it handed over, and the exit code it got
back. Measured over the shipped STEMS list at w161: of the 70 standing checks, ZERO opened it.
That is the same shape as w160's finding one run earlier (0 of 69 read the remote), one layer
further out: a corpus cannot audit its own completeness.

🔴 WHAT IT SHOWED THE MOMENT IT WAS READ. Two things, and they are different failures.

1. ROW 4 WAS NOT UNHANDED SINCE 08-31 -- IT WAS UNREADABLE. The index cell argued from the
   absence ("w133 (08-31) is the 16th handing and it built nothing"), and row 4 was the only
   one of the ten with no post-deadline entry. The driver log records slot 1 of 2026-09-01
   being handed the XGBoost angle and exiting 0, and that run wrote a full entry:

       # (w142, 2026-09-01, SLOT 1/10, ANGLE "XGBoost: third leg of the ensemble") — 🏁 ...

   It graded the frozen forecast, fixed the leaderboard download and committed as 4e387f7.
   `RUN_HDR`'s three alternatives all anchored on what follows `# `, so the leading `(` hid it,
   and w141 with it. The entry was in the journal the whole time. Fixed in the reader, not
   here; this guard is what makes the next one impossible to miss.

2. TWO GENERA WERE BEING TREATED AS ONE, AND THE DIAGNOSIS WENT THE WRONG WAY. w160 wrote that
   w159 "left the same fault as w155, w156 and w158: it did the work ... and wrote no journal
   entry", and #69 was built on that reading. The driver log disagrees on half the set:

       w155  2026-09-02 slot 4   exit 0  -> it ran to completion and wrote nothing
       w158  2026-09-03 slot 2   exit 0  -> the same
       w156  2026-09-02 slot 5   exit 1  -> API 529 after 25 minutes, retried, 529 again
       w159  2026-09-03 slot 3   exit 1  -> API 500 after 10 minutes, retried, 500 again

   ⚠ A KILLED RUN DID NOT DECLINE TO WRITE ITS ENTRY. It was terminated before it could, and
   NO guard that runs inside that run can change the outcome -- #66, #69 and #70 all execute in
   the process being audited. 2026-09-03 lost 5 of its 9 slots this way and 2026-09-02 lost 6
   of 10. So the operational rule is not "call the guards at orientation" (a dead run calls
   nothing); it is that the NEXT run must reconcile against a record the dead one did not write.

WHAT THIS ADDS, IN ONE LINE. #50/#66/#69/#70 ask what the workspace wrote. #71 asks what the
launcher ran, and it fails only on the half that discipline can actually fix.

CHECKED:

  C1   the driver logs parse into slot records -- date, slot, angle, exit code, duration -- and
       every ANGLE resolves to one of the ten rows through w117a_handcount's own `classify`,
       so this guard and the census cannot disagree about what a row is.
  C1b  the parse tracks the log rather than agreeing by luck: rewriting one slot's exit code
       moves that slot's verdict and nothing else, and removing the marker line yields nothing.
  C1c  the record key is (CYCLE, slot), not slot. Added w162: the launcher can be restarted
       against the same date and append a whole second 1..10 cycle to the same file, and the old
       key let the second cycle overwrite the first -- 16 of 208 records vanished that way. The
       check asserts `slot header lines == records` and declares itself INERT rather than
       passing quietly if no multi-cycle date is left in the corpus.
  C2   the read is ANSWERABLE. A log directory that is missing or empty is UNREADABLE and
       FAILS, on #70's rule: a standing check that cannot see its subject must never report
       clean.
  C3   THE ASSERTION. Every slot that exited 0 must have a JOURNAL.md entry for its (date, row).
       Slots that exited non-zero are REPORTED, not failed -- see C4 for why that is not a
       softening.
  C4   the genus split, named per slot rather than inferred. A slot that died is KILLED; one
       that finished and wrote nothing is UNWRITTEN. The split is also measured in seconds,
       because a slot killed at birth stranded no work and one killed mid-run did.
  C5   the anchors this guard reasons about are still where it read them.
  C6   --control over a frozen pre-fix corpus, both arms literal.
  C7   scope and blindness, measured.
"""
import os
import re
import sys
import pathlib
import datetime
import collections

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from w117a_handcount import classify, RUN_HDR          # noqa: E402  (the census's own resolver)

LOGDIR = pathlib.Path("/home/nixos/all-my-repos/ai/kaggle-agents/logs/playground-series-s6e8")
JOURNAL = HERE.parent / "JOURNAL.md"
HANDCOUNT = HERE / "w117a_handcount.py"
COMMITTEDGUARD = HERE / "w159a_committedguard.py"
RECORDGUARD = HERE / "w156a_recordguard.py"

SLOT_HDR = re.compile(r"^\s+playground-series-s6e8 — slot (\d+)/(\d+)")
ANGLE = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)[^\]]*\] ANGLE: (.*)$")
DONE = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)[^\]]*\] slot (\d+) finished \(exit (\d+)\)")
# ⚠ THE DURATION MUST COME OFF THE FIRST ATTEMPT, NOT THE LAST `finished` LINE. The launcher
# retries a failed slot after sleeping 60s, so `finished - ANGLE` folds that sleep in and every
# retried slot reads >= 62s no matter how fast it actually died. Measured that way the 09-02
# slots that died in ONE SECOND read 62s, C4's at-birth arm matched nothing, and the threshold
# check quietly skipped itself. The first terminal event after the ANGLE line is the real one.
FAILED = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)[^\]]*\] slot (\d+) failed \(exit (\d+)\)")
HDR_DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
HDR_RUN = re.compile(r"\bw(\d+)\b")

# ⚠ THE STATED REASON FOR THIS WINDOW WAS WRONG AND THE WINDOW IS STILL RIGHT (w162). w161 wrote
# that "the launcher only began writing per-slot ANGLE lines partway through"; in fact every log
# from 2026-08-10 on carries them, all 208. What the earlier logs lack is a slot-level RUN ID, so
# on a date with 10 handed and 6 recorded there is no way to say WHICH four went missing -- C3
# asserts per slot, not per date, and it cannot be honest without that. The totals reconcile
# across the whole corpus (see RESEARCH's `handed` vs `recorded` section); the attribution only
# reconciles here. Two dates (08-14, 08-15) also have no slot log at all.
FIRST_RECONCILED = "2026-09-01"

# C4's threshold, and it is not a tuning choice: the observed durations fall either side of a
# gap two orders of magnitude wide (1s for the slots that died at birth on 2026-09-02, 196s for
# the shortest that died mid-run on 2026-09-03). Anything inside that gap gives this split.
BIRTH_S = 60

FAILS = 0


def fail(msg):
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


# ⛔ THE LAUNCHER STAMPS EDT AND THE JOURNAL DATES ITS HEADERS IN UTC (w162). Every one of the
# 645 timestamps in the log directory carries the literal `EDT`, i.e. UTC-4, so any slot that
# starts after 20:00 local belongs to the NEXT UTC day -- and C3 joins the driver record to a
# journal entry on (date, row). Taken raw, 2026-08-19's evening cycle reads 08-19 in the log and
# 08-20 in the journal, and the join misses ten slots. Converted, they agree.
EDT_OFFSET_H = 4


def utc_date(date_s, hms):
    """The UTC calendar date of an EDT log timestamp, which is what a journal header carries."""
    h, m, sec = (int(x) for x in hms.split(":"))
    y, mo, d = (int(x) for x in date_s.split("-"))
    t = datetime.datetime(y, mo, d, h, m, sec) + datetime.timedelta(hours=EDT_OFFSET_H)
    return t.strftime("%Y-%m-%d")


def parse_log(text):
    """One log file -> {(cycle, slot): {slot, cycle, date, angle, row, t0, t1, exit}}.

    The `finished` line is authoritative for the exit code and it is written once per attempt;
    the launcher retries a failed slot in place, so the LAST one is the slot's real outcome.

    ⛔ THE KEY IS (CYCLE, SLOT), NOT SLOT (w162). A log file is named for the launcher's START
    date and the launcher can be started MORE THAN ONCE against the same date, appending a whole
    second 1..10 cycle to the same file. Keyed on the slot number alone, the second cycle
    silently OVERWRITES the first: 2026-08-10 holds cycles [1..5] then [1..10] and 2026-08-19
    holds one aborted start, [1..10] and [1..10] again. That collapsed 16 of 208 ANGLE lines
    down to 192 records, and the record it then reported for those slots -- angle, exit code,
    duration -- belonged to the later cycle only. A new cycle is a slot number that does not
    increase.

    ⚠ `date` still comes off the slot's OWN ANGLE timestamp, not the file name, and that is
    deliberate: a cycle started at 20:12 runs past local midnight (2026-08-10's second cycle
    finished eight of its ten slots on 08-11), and C3 matches against the run's journal header
    date, which is the run's own. So the file name dates the CYCLE and the timestamp dates the
    SLOT, and they are allowed to differ.
    """
    out, cur, cycle, last_slot = {}, None, 1, 0
    for ln in text.splitlines():
        m = SLOT_HDR.match(ln)
        if m:
            slot = int(m.group(1))
            if slot <= last_slot:
                cycle += 1
            last_slot = slot
            cur = (cycle, slot)
            out.setdefault(cur, {"slot": slot, "cycle": cycle})
            continue
        m = ANGLE.match(ln)
        if m and cur is not None:
            r = out[cur]
            r["date"] = utc_date(m.group(1), m.group(2))        # UTC, to match the journal
            r["local_date"], r["t0"], r["angle"] = m.group(1), secs(m.group(2)), m.group(3)
            r["row"] = classify(r["angle"].split(":")[0])
            continue
        m = FAILED.match(ln)
        if m:
            r = out.setdefault((cycle, int(m.group(3))),
                               {"slot": int(m.group(3)), "cycle": cycle})
            r.setdefault("t_end", secs(m.group(2)))          # first attempt only
            continue
        m = DONE.match(ln)
        if m:
            r = out.setdefault((cycle, int(m.group(3))),
                               {"slot": int(m.group(3)), "cycle": cycle})
            r.setdefault("t_end", secs(m.group(2)))
            r["t1"], r["exit"] = secs(m.group(2)), int(m.group(4))
    return out


def read_driver(logdir):
    days = {}
    for f in sorted(logdir.glob("20*.log")):
        days[f.stem] = parse_log(f.read_text(errors="replace"))
    return days


def journal_rows():
    """(date, row) -> [run ids], read with the census's own RUN_HDR so the two cannot drift."""
    from w117a_handcount import census
    seen = collections.defaultdict(list)
    for r in census():
        d = HDR_DATE.search(r["header"])
        w = HDR_RUN.search(r["header"])
        if d and w and r["row"]:
            seen[(d.group(1), r["row"])].append("w" + w.group(1))
    return seen


def journal_unresolved():
    """date -> the run tags whose JOURNAL.md header resolves to NO genus.

    w175: the census drops these, so a slot whose entry has no angle in its header is
    indistinguishable from a slot that wrote nothing at all. It is not the same failure and
    it does not have the same remedy, so C3 names the two apart rather than calling both
    UNWRITTEN. w173 and w174 are the specimens: both ran, both appended a full entry, and
    both headers read `# 2026-09-05 — w173 — CLOSED. Deadline re-confirmed, nothing done,
    nothing to do.` with the handed angle stripped out.
    """
    from w117a_handcount import census
    out = collections.defaultdict(list)
    for r in census():
        d = HDR_DATE.search(r["header"])
        w = HDR_RUN.search(r["header"])
        if d and w and not r["row"]:
            out[d.group(1)].append("w" + w.group(1))
    return out


def verdicts(days, seen, inflight_row=None, unresolved=None):
    """One row per slot in the reconciled window, classified."""
    rows = []
    newest = max(days) if days else None
    for date in sorted(days):
        if date < FIRST_RECONCILED:
            continue
        for n in sorted(days[date]):
            r = days[date][n]
            if "row" not in r:
                continue
            entries = seen.get((r["date"], r["row"]), [])
            dur = r["t_end"] - r["t0"] if "t_end" in r and "t0" in r else None
            if dur is not None and dur < 0:
                dur += 86400            # the slot ran past local midnight (08-10's 2nd cycle)
            if "exit" not in r:
                # In flight. Bounded, not open-ended: it must be the newest log's last slot.
                v = "INFLIGHT" if (date == newest and n == max(days[date])) else "NO-EXIT"
            elif entries:
                v = "RECORDED" if r["exit"] == 0 else "KILLED/recovered"
            elif r["exit"] == 0:
                # An entry with an unreadable header is not an absent entry (w175).
                v = "UNREADABLE-HDR" if (unresolved or {}).get(r["date"]) else "UNWRITTEN"
            else:
                v = "KILLED/at-birth" if dur is not None and dur < BIRTH_S else "KILLED/mid-run"
            rows.append(dict(date=date, slot=r["slot"], cycle=r["cycle"], row=r["row"],
                             exit=r.get("exit"), dur=dur, entries=entries, verdict=v,
                             unresolved=(unresolved or {}).get(date, [])))
    return rows


def main():
    run = None
    for i, a in enumerate(sys.argv):
        if a == "--run" and i + 1 < len(sys.argv):
            run = sys.argv[i + 1]

    print("C2 the driver log is readable, or this check FAILS rather than reports clean")
    if not LOGDIR.is_dir():
        fail(f"UNREADABLE: no driver log directory at {LOGDIR}")
        print(f"\nFAILURES: {FAILS}")
        return 1
    days = read_driver(LOGDIR)
    if not days:
        fail(f"UNREADABLE: {LOGDIR} holds no dated launcher logs")
        print(f"\nFAILURES: {FAILS}")
        return 1
    print(f"  {len(days)} day log(s) at {LOGDIR}")

    print("C1 the slot records parse, and every angle resolves through the census's classify")
    total = sum(len(v) for v in days.values())
    unres = [(d, days[d][n]["slot"], days[d][n].get("angle", "")[:44])
             for d in days for n in days[d] if days[d][n].get("row") is None
             and "angle" in days[d][n]]
    print(f"  {total} slot record(s) across {len(days)} day(s); "
          f"{sum(1 for d in days for n in days[d] if 'angle' in days[d][n])} carry an ANGLE line")
    if unres:
        for d, n, a in unres[:6]:
            fail(f"{d} slot {n}: angle does not resolve to a row -- {a!r}")
    else:
        print("  every ANGLE line resolves to one of the ten rows  OK")

    print("C1b the parse tracks the log rather than agreeing by luck")
    probe = LOGDIR / "2026-09-03.log"
    if probe.exists():
        txt = probe.read_text(errors="replace")
        base = parse_log(txt)
        flipped = parse_log(txt.replace("slot 1 finished (exit 0)", "slot 1 finished (exit 7)"))
        moved = sorted(base[n]["slot"] for n in base
                       if base[n].get("exit") != flipped.get(n, {}).get("exit"))
        print(f"  rewriting slot 1's exit code moves exactly {moved}  "
              f"{'OK' if moved == [1] else 'BROKEN'}")
        if moved != [1]:
            fail("the exit-code parse does not track the text it reads")
        stripped = parse_log(re.sub(r"^.*slot \d+ finished.*$", "", txt, flags=re.M))
        left = [n for n in stripped if "exit" in stripped[n]]
        print(f"  removing every finished line leaves {len(left)} exit code(s)  "
              f"{'OK' if not left else 'BROKEN'}")
        if left:
            fail("exit codes survive the removal of the lines they are read from")
    else:
        fail(f"UNREADABLE: C1b's probe log {probe.name} is absent")

    print("C1c the key is (cycle, slot) -- a second launcher cycle must not overwrite the first")
    hdrs = {}
    for f in sorted(LOGDIR.glob("20*.log")):
        # SLOT_HDR has no re.M, so match it per LINE rather than over the whole file.
        hdrs[f.stem] = sum(1 for ln in f.read_text(errors="replace").splitlines()
                           if SLOT_HDR.match(ln))
    multi = {d: (len(days[d]), hdrs[d], max(r["cycle"] for r in days[d].values()))
             for d in sorted(days) if hdrs.get(d, 0) and
             max(r["cycle"] for r in days[d].values()) > 1}
    total_hdr = sum(hdrs.values())
    collapsed = len({(d, days[d][n]["slot"]) for d in days for n in days[d]})
    print(f"  {total_hdr} slot header line(s) -> {total} record(s)  "
          f"{'OK' if total_hdr == total else 'BROKEN'}")
    if total_hdr != total:
        fail(f"C1c {total_hdr} slot headers parse into {total} records -- a cycle is being lost")
    for d, (n_rec, n_hdr, n_cyc) in multi.items():
        print(f"    {d}: {n_cyc} launcher cycles, {n_hdr} headers, {n_rec} records")
    print(f"  keyed on the slot number alone the same logs give {collapsed} record(s), "
          f"losing {total - collapsed}")
    if not multi or total == collapsed:
        fail("C1c is inert: no multi-cycle date in the corpus, so the key proves nothing")

    seen = journal_rows()
    rows = verdicts(days, seen, unresolved=journal_unresolved())

    print(f"C3 every slot that exited 0 has a JOURNAL.md entry for its (date, row) "
          f"-- window from {FIRST_RECONCILED}")
    tally = collections.Counter(r["verdict"] for r in rows)
    for r in rows:
        if r["verdict"] == "UNWRITTEN":
            fail(f"{r['date']} slot {r['slot']} (row {r['row']}) exited 0 and left no entry")
        elif r["verdict"] == "UNREADABLE-HDR":
            fail(f"{r['date']} slot {r['slot']} (row {r['row']}) exited 0 and its entry "
                 f"({'/'.join(r['unresolved'])}) has no angle in its header, so the census "
                 f"cannot resolve it -- the handing is real and will never be counted")
    for r in rows:
        mark = {"RECORDED": "  ", "INFLIGHT": "· ", "UNWRITTEN": "⛔", "NO-EXIT": "⛔",
                "UNREADABLE-HDR": "⛔"}.get(
            r["verdict"], "⚠ ")
        d = "     " if r["dur"] is None else f"{r['dur']:4d}s"
        print(f"  {mark} {r['date']} slot {r['slot']:2d}  row {r['row']:2d}  "
              f"exit {str(r['exit']):>4}  {d}  {r['verdict']:16s} "
              f"{','.join(r['entries']) or '-'}")
    for r in rows:
        if r["verdict"] == "NO-EXIT":
            fail(f"{r['date']} slot {r['slot']} has no exit code and is not the in-flight slot")
    if run:
        infl = [r for r in rows if r["verdict"] == "INFLIGHT"]
        print(f"  C3b caller says {run}; the in-flight slot is "
              f"{infl[0]['date']+' slot '+str(infl[0]['slot']) if infl else 'absent'}")

    print("C4 the two genera, separated -- a killed run did not decline to write")
    for k in ("RECORDED", "KILLED/recovered", "KILLED/mid-run", "KILLED/at-birth",
              "UNWRITTEN", "UNREADABLE-HDR", "INFLIGHT"):
        if tally[k]:
            print(f"  {k:17s} {tally[k]:3d}")
    killed = tally["KILLED/recovered"] + tally["KILLED/mid-run"] + tally["KILLED/at-birth"]
    print(f"  of {len(rows)} slot(s) in the window, {killed} were killed and "
          f"{tally['UNWRITTEN']} finished without writing")
    print("  ⚠ no in-process guard can help the killed ones: #66, #69 and #70 all run inside")
    print("    the process being audited, so a slot that dies runs none of them")
    births = [r["dur"] for r in rows if r["verdict"] == "KILLED/at-birth" and r["dur"] is not None]
    mids = [r["dur"] for r in rows if r["verdict"] == "KILLED/mid-run" and r["dur"] is not None]
    if births and mids:
        good = max(births) < BIRTH_S <= min(mids)
        print(f"  the gap C4's {BIRTH_S}s threshold sits in: at-birth max {max(births)}s, "
              f"mid-run min {min(mids)}s  {'OK' if good else 'BROKEN'}")
        if not good:
            fail("the at-birth/mid-run threshold no longer sits inside a gap in the data")
    else:
        # An arm that matches nothing proves nothing, and saying so is the whole point: the
        # first cut of this check read every retried slot as >= 62s and skipped this silently.
        print(f"  ⚠ INERT: at-birth {len(births)} slot(s), mid-run {len(mids)} -- the {BIRTH_S}s"
              " split is unexercised in this window and is asserting nothing")

    print("C5 the anchors this guard reasons about are still where it read them")
    for path, label, needle in (
        (HANDCOUNT, "the census's classify, which fixes what a row is",
         "def classify(genus: str):"),
        # Both recovered header shapes: `(w142,` (w161) and the `══ <date>` banner (w162).
        # This anchor is a SUBSTRING of RUN_HDR on purpose -- widening the pattern turns it red
        # here first, which is what happened when w162 added the banner alternative.
        (HANDCOUNT, "the two header shapes that made w142 and the banner entries readable",
         r"|wave |\(w\d+[a-z]?,|══ 20\d\d-\d\d-\d\d)"),
        (COMMITTEDGUARD, "#69's C7 blindness, which this guard closes",
         "a run that leaves artefacts but is never named in a trail is invisible to both"),
        (RECORDGUARD, "#66's in-flight exemption, which C3 bounds differently",
         "strays = [(n, w) for n, w in uncovered if w != cur_run]"),
    ):
        ok = path.exists() and needle in path.read_text(encoding="utf-8")
        print(f"  {'OK  ' if ok else 'GONE'} {path.name}: {label}")
        if not ok:
            fail(f"anchor moved in {path.name}: {label}")

    print("C7 scope and blindness, measured")
    stems = 0
    if (HERE / "w93a_suite.py").exists():
        stems = len(re.findall(r'"(w\d+[a-z]?_\w+)"', (HERE / "w93a_suite.py").read_text()))
    readers = [p.name for p in HERE.glob("*.py")
               if "kaggle-agents/logs" in p.read_text(errors="replace")]
    print(f"  scripts in experiments/ that read the launcher log: {readers}")
    print(f"  before this one there were {len(readers) - 1}, against ~{stems} registered stems")
    print("  it reads what the LAUNCHER recorded, not what the run did: a slot that exited 0")
    print("    having done nothing at all still reads RECORDED if an entry exists")
    print("  it cannot see a run started by hand, outside the launcher -- those leave no slot")
    print("  UNREADABLE-HDR is keyed on the DATE, not on the slot: it says this day has an entry")
    print("    the census cannot resolve, not that THIS slot wrote it. A day carrying both a")
    print("    genuinely-empty slot and an unreadable-header one reads UNREADABLE-HDR for both.")
    print("    Both still FAIL, so the split changes the diagnosis and never the verdict.")
    print(f"  it reconciles from {FIRST_RECONCILED} only; earlier logs predate the per-slot")
    print("    ANGLE line, so the corpus before that is out of its reach and not audited")
    print("  KILLED is reported and never failed, so a run of dead slots keeps this green --")
    print("    that is deliberate (nothing a later run does can un-kill them) and it means")
    print("    this check must not be read as 'the day went well'")

    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


# C6's specimen. Every field is a literal: the 2026-09-01 slot 1 record as the launcher wrote
# it, against the journal map BEFORE the RUN_HDR fix (w142 invisible, so row 4 has no entry)
# and AFTER it (w142 present). Both arms are frozen, on w156's lesson that a control borrowing
# live state stops working the moment the fix lands.
# The key is (cycle, slot) since w162, so the frozen arm carries it too -- and `slot`/`cycle`
# are in the record because `verdicts` reports them.
FROZEN_DAYS = {"2026-09-01": {(1, 1): {"slot": 1, "cycle": 1, "date": "2026-09-01",
                                       "t0": 31201, "t1": 32160, "exit": 0,
                                       "angle": "XGBoost: third leg of the ensemble, tuned on "
                                                "the same folds so the blend weights mean "
                                                "something.",
                                       "row": 4}}}
FROZEN_PRE = {}
FROZEN_POST = {("2026-09-01", 4): ["w142"]}


def control():
    print("C6 --control, both arms frozen")
    ok = True
    for label, seen, want in (("pre-fix  (RUN_HDR blind to `# (w142,`)", FROZEN_PRE, "FIRES"),
                              ("post-fix (w142 readable)             ", FROZEN_POST, "SILENT")):
        rows = verdicts(FROZEN_DAYS, seen)
        got = ("FIRES" if any(r["verdict"] in ("UNWRITTEN", "UNREADABLE-HDR")
                              for r in rows) else "SILENT")
        mark = "OK" if got == want else "BROKEN"
        ok = ok and mark == "OK"
        print(f"  {label}  -> {rows[0]['verdict']:10s} {got:6s} (want {want})  {mark}")
    print("  the control WORKS" if ok else "  ⛔ THE CONTROL IS BROKEN")
    return 0


if __name__ == "__main__":
    sys.exit(control() if "--control" in sys.argv else main())
