"""w162a (#72) -- the OTHER half of w161's reader defect: eleven run entries wrote their
header as a `══` banner, and `RUN_HDR` could not see any of them.

WHY THIS EXISTS. w161 found that `RUN_HDR`'s alternatives all anchored on the character
straight after `# `, so the two runs whose header opens with `(` were invisible to the census,
to #66, to #67 and to #69. It fixed those two and MEASURED, without fixing, a second population
with the same cause: the 08-16 -> 08-18 window wrote entry headers as

    # ══ 2026-08-17 (UTC) — WAVE w17, SLOT 1 of 10 ══

Eleven of them. Every one is a real per-slot entry that states its handed angle in the body
(`**Handed angle:** "Error analysis: ..."`), and every one resolved to a genus the moment the
line was matched at all. w161 left it because the resync is nine rows wide and it could not
verify that in the time it had; this run did the resync and this guard is what holds it.

🔴 WHAT IT COST WHILE IT WAS BLIND. The eleven entries were not merely uncounted -- they were
ABSORBED. Before the fix, ONE header at L9048 (`## 2026-08-16 — w16e/w16v/w16w, slot 10/10`)
owned every line from 9048 to 10957, which is the 08-16 wave summary plus seven whole 08-17
slot entries; a second at L10957 (w22, 08-17 slot 8) owned four more. So two entries were
carrying twelve runs' worth of body between them, and the census reported 181 runs against a
corpus of 192. Nine of the ten index rows were undercounted, by +1 or +2 each.

⚠ ROW 5 IS THE ONE ROW THE RESYNC DOES NOT MOVE, AND THAT IS AN ACCIDENT OF PUNCTUATION.
08-17's feature-engineering slot (w22, slot 8) wrote its header in the dated form the reader
already understood, so it was never hidden. This run was handed row 5, which means the run that
performed the nine-row resync is the one run with no stake in its own row's count.

🎯 THE RULE, AND WHY IT IS NOT `══`. Two `══` lines in the corpus are NOT entry starts:

    L 9273  # ══ WAVE SUMMARY — Kaggle day UTC 2026-08-16, slots 1–10, written by slot 10 ══
    L10889  ## ══ w21 ADDENDUM — SHIP 3, and a CORRECTION TO §4 ABOVE ══

The first is a summary written BY 08-16 slot 10 and belongs inside that slot's entry; the
second is a sub-header inside w21's. An entry-starting banner names its DATE right after the
`══`, and both of these name something else. Widening to a bare `══` alternative instead takes
both: the corpus goes 192 -> 194, the summary arrives as a phantom row-6 run on `body-bare` (the
weakest path the resolver has) and the addendum as an `unresolved` one, and row 6 would have to
publish ×18 for a run that never existed. C3 keeps that arm live.

CHECKED:

  C1   the eleven banner entries are matched by the LIVE `RUN_HDR`, each resolves to one of the
       ten genera, and the slots they claim are distinct within their date -- so they are per-
       slot entries and not one entry seen eleven times.
  C1b  the match tracks the text rather than agreeing by luck: rewriting one banner's `══`
       drops exactly that entry and leaves the other ten.
  C2   THE ASSERTION, and it is additivity MEASURED per run, not asserted. Adding a header
       shortens the PRECEDING entry's body, and `resolve` reads bodies, so the blast radius is
       not bounded by the lines that changed. Census under the frozen pre-fix pattern vs the
       live one: 11 added, 0 removed, 0 pre-existing runs whose (row, how) moves.
  C3   the control that says why the rule is dated and not bare `══`, both arms frozen.
  C4   all SEVEN sites of the pattern moved together -- four `RUN_HDR` definitions compared
       byte-for-byte across the four modules, and three escaped literal copies counted in
       source. #67 compares its copy to #50's and #66/#69 check a literal each; nothing before
       this compared all four at once.
  C5   the anchors this guard reasons about are still where it read them.
  C6   --control over the frozen pre-fix pattern: C1 must FIRE pre-fix and be SILENT live.
  C7   scope and blindness, measured.
"""
import io
import re
import sys
import pathlib
import importlib.util
import collections

HERE = pathlib.Path(__file__).resolve().parent
JOURNAL = HERE.parent / "JOURNAL.md"

HANDCOUNT = HERE / "w117a_handcount.py"
RECORDGUARD = HERE / "w156a_recordguard.py"
CLOSEDGUARD = HERE / "w157a_closedguard.py"
COMMITTEDGUARD = HERE / "w159a_committedguard.py"
COPIES = [HANDCOUNT, RECORDGUARD, CLOSEDGUARD, COMMITTEDGUARD]

# ── FROZEN LITERALS ─────────────────────────────────────────────────────────────────────────
# w156's lesson: a control arm that borrows live state stops working the moment the fix lands.
# All three of these are literals and none is derived from the file being audited.
PRE_FIX = r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,)"
LIVE = r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,|══ 20\d\d-\d\d-\d\d)"
BLANKET = r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave |\(w\d+[a-z]?,|══)"

WAVE_ALT = "══ 20"          # the fragment every one of the seven sites must now carry
N_SITES = 7                 # 4 definitions + 3 escaped literal copies
N_BANNERS = 11              # entry-starting banners in the corpus
NOT_ENTRIES = 2             # `══` lines that are a summary / an addendum, not an entry

BANNER = re.compile(r"^#{1,2} ══ (20\d\d-\d\d-\d\d) \(UTC\) — WAVE (w\d+), SLOT (\d+) of (\d+) ══")
ANY_WAVE = re.compile(r"^#{1,2} ══")

FAILS = 0


def fail(msg):
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def load_census():
    """A FRESH module object each time, so patching RUN_HDR on one cannot leak into another."""
    spec = importlib.util.spec_from_file_location("hc_%d" % load_census.n, HANDCOUNT)
    load_census.n += 1
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


load_census.n = 0


def census_with(pattern, text=None):
    """{line: run} under `pattern`. `text` overrides JOURNAL.md for C1b."""
    m = load_census()
    m.RUN_HDR = re.compile(pattern)
    if text is not None:
        m.JOURNAL = _tmp_journal(text)
    return {r["line"]: r for r in m.census()}


def _tmp_journal(text):
    p = HERE / "w162a_tmp_journal.md"
    io.open(p, "w", encoding="utf-8").write(text)
    return str(p)


def counts(cen):
    d = collections.Counter()
    for r in cen.values():
        if r["row"]:
            d[r["row"]] += 1
    return d


def c1_banners(pattern, label, quiet=False):
    """Returns the banner entries the census sees under `pattern`. Fails if not all eleven
    are there and resolved. This is the body C6 re-runs against the pre-fix literal."""
    bad = 0
    lines = io.open(JOURNAL, encoding="utf-8").read().split("\n")
    want = [(i + 1, BANNER.match(l)) for i, l in enumerate(lines) if BANNER.match(l)]
    cen = census_with(pattern)
    if not quiet:
        print(f"C1 the {len(want)} banner entries, under {label}")
    if len(want) != N_BANNERS:
        bad += 1
        if not quiet:
            fail(f"C1 vacuous: {len(want)} banner-shaped lines in the corpus, expected {N_BANNERS}")
    per_date = collections.defaultdict(list)
    for ln, m in want:
        r = cen.get(ln)
        date, wid, slot = m.group(1), m.group(2), int(m.group(3))
        if r is None:
            bad += 1
            if not quiet:
                fail(f"C1 L{ln} {date} {wid} slot {slot}: not matched as a run header")
            continue
        if not r["row"]:
            bad += 1
            if not quiet:
                fail(f"C1 L{ln} {date} {wid} slot {slot}: matched but {r['how']}, no genus")
            continue
        per_date[date].append(slot)
        if not quiet:
            print(f"     L{ln:6d}  {date} {wid:5s} slot {slot:2d}  -> row {r['row']:2d} "
                  f"{r['genus'][:26]:26s} ({r['how']})")
    for date, slots in sorted(per_date.items()):
        if len(set(slots)) != len(slots):
            bad += 1
            if not quiet:
                fail(f"C1 {date}: duplicate slots {sorted(slots)} -- these are not per-slot entries")
        elif not quiet:
            print(f"     {date}: slots {sorted(slots)}, all distinct  OK")
    return bad


def main():
    control = "--control" in sys.argv
    print(f"w162a -- the `══` banner headers  ({'CONTROL' if control else 'live'})\n")

    # ── C6 / C1 ─────────────────────────────────────────────────────────────────────────────
    if control:
        print("C6 the control: C1 re-run against the FROZEN pre-fix pattern\n")
        bad_pre = c1_banners(PRE_FIX, "the frozen pre-fix pattern")
        print(f"\n  pre-fix -> {bad_pre} C1 failure(s), want > 0   "
              f"{'OK' if bad_pre else 'BROKEN'}")
        bad_live = c1_banners(LIVE, "the live pattern", quiet=True)
        print(f"  live    -> {bad_live} C1 failure(s), want 0       "
              f"{'OK' if not bad_live else 'BROKEN'}")
        if not bad_pre or bad_live:
            fail("C6 the control does not separate the two arms")
        else:
            global FAILS
            FAILS -= bad_pre          # the pre-fix arm is SUPPOSED to fail
            print("  the control WORKS")
        return 0 if FAILS == 0 else 1

    live_pat = load_census().RUN_HDR.pattern
    if live_pat != LIVE:
        fail(f"C1 the live RUN_HDR is not the pattern this guard was written against:\n"
             f"       live {live_pat!r}\n       want {LIVE!r}")
    c1_banners(live_pat, "the live RUN_HDR")

    # ── C1b ─────────────────────────────────────────────────────────────────────────────────
    print("\nC1b the match tracks the text, not luck")
    text = io.open(JOURNAL, encoding="utf-8").read()
    lines = text.split("\n")
    victim = next(i for i, l in enumerate(lines) if BANNER.match(l))
    hacked = list(lines)
    hacked[victim] = hacked[victim].replace("══ 20", "==== 20", 1)
    cen_h = census_with(live_pat, "\n".join(hacked))
    still = sum(1 for i, l in enumerate(lines) if BANNER.match(l) and (i + 1) in cen_h)
    print(f"  rewrite L{victim + 1}'s `══` -> {still} of {N_BANNERS} banners still matched  "
          f"{'OK' if still == N_BANNERS - 1 else 'BROKEN'}")
    if still != N_BANNERS - 1:
        fail(f"C1b dropping one banner left {still} matched, expected {N_BANNERS - 1}")
    (HERE / "w162a_tmp_journal.md").unlink(missing_ok=True)

    # ── C2 ──────────────────────────────────────────────────────────────────────────────────
    print("\nC2 THE ASSERTION -- the widening is purely additive, measured run by run")
    pre, now = census_with(PRE_FIX), census_with(LIVE)
    added = sorted(set(now) - set(pre))
    removed = sorted(set(pre) - set(now))
    moved = [(l, pre[l]["row"], pre[l]["how"], now[l]["row"], now[l]["how"])
             for l in sorted(set(pre) & set(now))
             if (pre[l]["row"], pre[l]["how"]) != (now[l]["row"], now[l]["how"])]
    print(f"  corpus {len(pre)} -> {len(now)} runs")
    print(f"  headers added   : {len(added)}  {added}")
    print(f"  headers removed : {len(removed)}")
    print(f"  pre-existing runs whose (row, how) CHANGED: {len(moved)}")
    for m in moved:
        print(f"    ~ L{m[0]}  ({m[1]}, {m[2]}) -> ({m[3]}, {m[4]})")
    if len(added) != N_BANNERS:
        fail(f"C2 {len(added)} headers recovered, expected exactly {N_BANNERS}")
    if removed:
        fail(f"C2 the widening REMOVED {len(removed)} header(s): {removed}")
    if moved:
        fail(f"C2 the widening RE-ASSIGNED {len(moved)} pre-existing run(s)")
    cp, cn = counts(pre), counts(now)
    print("  the resync this forced, per row:")
    for row in range(1, 11):
        mark = "MOVE" if cp[row] != cn[row] else "    "
        print(f"    row {row:2d}  ×{cp[row]:2d} -> ×{cn[row]:2d}  {mark}")
    if cp[5] != cn[5]:
        fail("C2 row 5 moved -- the run doing the resync was handed row 5 and must not "
             "be the one it benefits")
    if sum(1 for row in range(1, 11) if cp[row] != cn[row]) != 9:
        fail("C2 the resync no longer covers exactly nine rows")

    # ── C3 ──────────────────────────────────────────────────────────────────────────────────
    print("\nC3 the control for the RULE: dated, not bare `══`")
    not_entries = [(i + 1, l) for i, l in enumerate(lines)
                   if ANY_WAVE.match(l) and not BANNER.match(l)]
    print(f"  `══` lines that are NOT entry starts: {len(not_entries)}")
    for ln, l in not_entries:
        taken = "TAKEN" if ln in now else "not matched"
        print(f"    L{ln:6d}  {taken:11s} | {l[:72]}")
        if ln in now:
            fail(f"C3 L{ln} is not an entry start and the live pattern takes it")
    if len(not_entries) != NOT_ENTRIES:
        fail(f"C3 vacuous: {len(not_entries)} non-entry `══` lines, expected {NOT_ENTRIES}")
    blanket = census_with(BLANKET)
    extra = sorted(set(blanket) - set(now))
    print(f"  a bare `══` alternative instead: corpus {len(now)} -> {len(blanket)}, "
          f"and it takes")
    for ln in extra:
        r = blanket[ln]
        print(f"    L{ln:6d}  row {str(r['row']):4s} via {r['how']:11s} | {r['header'][:56]}")
    cb = counts(blanket)
    phantom = [row for row in range(1, 11) if cb[row] != cn[row]]
    print(f"  which would make {len(phantom)} row(s) publish a count for a run that does not "
          f"exist: {[(row, f'×{cn[row]}->×{cb[row]}') for row in phantom]}")
    if len(extra) != NOT_ENTRIES or not phantom:
        fail("C3 the bare-`══` control no longer shows the harm it was written for")
    else:
        print("  the control WORKS")

    # ── C4 ──────────────────────────────────────────────────────────────────────────────────
    print("\nC4 all seven sites of the pattern, moved together")
    pats = {}
    for p in COPIES:
        spec = importlib.util.spec_from_file_location("cp_" + p.stem, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        pats[p.name] = m.RUN_HDR.pattern
    for name, pat in pats.items():
        ok = pat == LIVE
        print(f"  {'OK  ' if ok else 'FAIL'} {name:28s} definition")
        if not ok:
            fail(f"C4 {name}'s RUN_HDR is not byte-identical to the other copies")
    sites = sum(io.open(p, encoding="utf-8").read().count(WAVE_ALT) for p in COPIES)
    print(f"  source sites carrying {WAVE_ALT!r}: {sites}, expected {N_SITES} "
          f"(4 definitions + 3 escaped literal copies)  {'OK' if sites == N_SITES else 'FAIL'}")
    if sites != N_SITES:
        fail(f"C4 {sites} of {N_SITES} sites carry the wave alternative -- a widening that "
             f"misses a copy leaves #66/#67/#69 reading a stale literal")

    # ── C5 ──────────────────────────────────────────────────────────────────────────────────
    print("\nC5 the anchors this guard reasons about are still where it read them")
    anchors = [
        (HANDCOUNT, f'RUN_HDR = re.compile(r"{LIVE}")',
         "w117a_handcount.py: the definition every copy is compared against"),
        (HANDCOUNT, "def census():",
         "w117a_handcount.py: the entry-splitting this guard patches RUN_HDR into"),
        (CLOSEDGUARD, "the copy is byte-identical to the original",
         "w157a_closedguard.py: #67's one-to-one copy check, which C4 generalises"),
        (COMMITTEDGUARD, "the RUN_HDR all three copies share",
         "w159a_committedguard.py: #69's literal copy, one of C4's three"),
    ]
    for path, needle, why in anchors:
        ok = needle in io.open(path, encoding="utf-8").read()
        print(f"  {'OK  ' if ok else 'FAIL'} {why}")
        if not ok:
            fail(f"C5 anchor moved: {why}")

    # ── C7 ──────────────────────────────────────────────────────────────────────────────────
    print("\nC7 scope and blindness, measured")
    print(f"  it audits ONE header shape. The corpus has {len(now)} runs and this guard has")
    print(f"    an opinion about {N_BANNERS} of them; a twelfth shape nobody has written yet is")
    print("    invisible to it exactly as the banners were to #50 for eighteen days.")
    print("  the ELEVEN is a frozen constant. JOURNAL.md is append-only and no future run")
    print("    writes a banner header, so a twelfth appearing means the corpus was EDITED, and")
    print("    C1 goes red rather than adapting -- that is deliberate.")
    print("  it proves the banners are COUNTED, not that their entries are any good: an entry")
    print("    that resolves by `body` and did nothing all run still counts as a handing.")
    print("  C2 measures re-assignment against the pre-fix pattern only. A change to `resolve`")
    print("    itself, or to the genus table, moves rows without moving any header, and #50's")
    print("    own C2/C3 are what catch that.")
    print("  the nine-row resync it holds is a COUNT. The trail text in each index cell still")
    print("    begins `from 08-1x` and names no banner run, so the eleven are inside the")
    print("    unnamed prefix of nine trails -- #67 checks the named entries and cannot see")
    print("    whether the prefix is complete.")

    print(f"\nFAILURES: {FAILS}")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
