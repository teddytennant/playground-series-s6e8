"""w178a (#72) -- a journal entry can name its own ANGLE INDEX row, and four in a row named it wrong.

WHY THIS EXISTS. Every existing check reads the INDEX side of the handing: #50 counts the
trails in RESEARCH.md against JOURNAL.md headers, #67 checks the `(closed)` markers on those
trails, #71 checks the launcher log against them. All three treat the trail as the statement of
which row a run was handed -- and the trail is in RESEARCH.md, which is editable, so a wrong one
gets fixed.

Nothing read the OTHER statement. A journal body routinely writes the row out in prose:

    Handed angle was slot 7, *"CatBoost: ... identical folds"* -- row 4, closed in week 1.

That sentence is a second, independent claim about the same fact, and JOURNAL.md is append-only,
so a wrong one is permanent. It had never been compared against the first.

🔴 WHAT IT SHOWED THE MOMENT IT WAS READ. Over 208 entries, 28 carry a self-attribution and 24
agree. The four that do not are CONSECUTIVE, all on 2026-09-04, and every one is off by exactly
+1:

    w167   genus `the original dataset`  -> row 1    body says row 2
    w168   genus `LightGBM`              -> row 2    body says row 3
    w169   genus `CatBoost`              -> row 3    body says row 4
    w170   genus `XGBoost`               -> row 4    body says row 5

w163 before them is right and w171 after them is right, so it is a bounded block of four, not a
drift. ⚠ IT IS NOT A RENUMBERING. The index table is byte-identical on the row-number column at
`6361962` (09-03) and `18f9c6c` (09-04), so the rows did not move under those runs; the prose is
simply wrong. And it is confined to prose: the trails put w167..w171 on rows 1,2,3,4,5 -- the
census was right about all five while four bodies said otherwise. A reader of JOURNAL.md alone
gets the wrong row four times; a reader of RESEARCH.md never does.

⛔ THE FOUR CANNOT BE CORRECTED. JOURNAL.md is append-only (the standing rule, and #71 already
carries two permanent reds of its own for the same reason). So this guard does NOT fail on them.
It FREEZES them: C3 requires exactly that set, so a fifth one is a new defect and fails, and one
of them vanishing is an append-only violation and also fails. C2 is the live half -- any entry
dated on or after the cutoff must agree, and that is the half a future run can still get right.

CHECKED:

  C1   the corpus is ANSWERABLE -- the journal parses, and enough entries carry a
       self-attribution to make silence meaningful. A read that covers nothing FAILS rather
       than passing quietly (#70's rule).
  C2   LIVE: every self-attribution dated >= CUTOFF agrees with `classify`. This is the only
       arm that can go red on new work.
  C3   FROZEN: the historical mismatches are exactly KNOWN_BAD, by (run, true row, claimed row).
  C4   the reader tracks its evidence, six ways -- a correct entry is silent, the same entry
       with the claim bumped fires, a quote naming no genus is not covered, an uncued quote
       (program output, the w129/w131 shape) is not covered, a claim beyond the window is not
       covered, and a claim before the quote is not covered.
  C5   it cannot disagree with the census about what a row is: the genus->row map is imported
       from w117a_handcount, not re-implemented, and every covered row is one of the ten.

    .venv/bin/python experiments/w178a_rowclaimguard.py

Deterministic: pure text, no API call, no fit.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JOURNAL = os.path.join(ROOT, "JOURNAL.md")

# C5: the same `classify` the census uses, imported rather than copied, so this guard and
# w117a can never disagree about which row a genus names.
_spec = importlib.util.spec_from_file_location("_w117a", os.path.join(HERE, "w117a_handcount.py"))
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)

# A quoted angle. Straight and curly, long enough to exclude a bare word.
QUOTE = re.compile(r'["“]([^"”]{12,300})["”]')
# `row 4`, `row **4**`, `row  4`.
ROWCLAIM = re.compile(r"\brow\s*\*{0,2}\s*(\d{1,2})\b", re.I)
# The quote must be introduced as the handed angle. Without this, program output quoted in a
# body counts as an angle: w129 and w131 both quote `w26g_send.py` and both are followed by
# prose naming an unrelated row. Measured: the cue drops exactly those two and no true one.
CUE = re.compile(r"\b(handed|angle|as issued|slot \d+)\b", re.I)
CUE_BACK = 90    # chars before the quote searched for the cue
WINDOW = 130     # chars after the quote searched for the claim

DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
RUNID = re.compile(r"\b(w\d{1,3})\b")

# C2's live/frozen boundary. Entries dated on or after this must agree; earlier ones are
# history and are judged by C3.
CUTOFF = "2026-09-05"

# C3's frozen control: (run, true row, claimed row). Four bodies on 2026-09-04, each +1.
# JOURNAL.md is append-only, so this set can never shrink by correction -- only by a rewrite,
# which is itself the defect this arm would then be reporting.
KNOWN_BAD = {("w167", 1, 2), ("w168", 2, 3), ("w169", 3, 4), ("w170", 4, 5)}

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL {msg}")


def attributions(text: str):
    """Every (run, date, true_row, claimed_row, quote) the journal text self-attributes.

    One per entry: the FIRST cued quote that classifies. Entries discuss other rows further
    down and those are commentary, not the run's statement of its own handing.
    """
    lines = text.split("\n")
    hdr = [i for i, l in enumerate(lines) if _hc.RUN_HDR.match(l)]
    out = []
    for k, i in enumerate(hdr):
        end = hdr[k + 1] if k + 1 < len(hdr) else len(lines)
        body = " ".join(lines[i:end])
        dm = DATE.search(lines[i])
        rm = RUNID.search(lines[i])
        for m in QUOTE.finditer(body):
            row = _hc.classify(m.group(1))
            if not row:
                continue
            if not CUE.search(body[max(0, m.start() - CUE_BACK):m.start()]):
                continue
            cm = ROWCLAIM.search(body[m.end():m.end() + WINDOW])
            if cm:
                out.append({"run": rm.group(1) if rm else "?",
                            "date": dm.group(1) if dm else "?",
                            "true": row, "claimed": int(cm.group(1)),
                            "quote": m.group(1)[:40], "line": i + 1})
            break
    return out


def main() -> int:
    text = open(JOURNAL, encoding="utf-8").read()
    hits = attributions(text)
    bad = [h for h in hits if h["true"] != h["claimed"]]

    print("C1 the corpus is answerable")
    n_entries = len(_hc.RUN_HDR.findall(text)) if hasattr(_hc.RUN_HDR, "findall") else 0
    n_entries = sum(1 for l in text.split("\n") if _hc.RUN_HDR.match(l))
    print(f"  {n_entries} journal entries, {len(hits)} carry a self-attribution, "
          f"{len(hits) - len(bad)} agree, {len(bad)} do not")
    if n_entries < 100:
        fail(f"only {n_entries} entries parsed -- the journal or RUN_HDR moved")
    if len(hits) < 20:
        fail(f"only {len(hits)} self-attributions found -- the reader has gone blind, "
             f"and a blind check must not pass")

    print("C2 LIVE: self-attributions dated >= %s" % CUTOFF)
    live = [h for h in hits if h["date"] >= CUTOFF]
    live_bad = [h for h in live if h["true"] != h["claimed"]]
    print(f"  {len(live)} live, {len(live_bad)} wrong")
    for h in live_bad:
        fail(f"L{h['line']} {h['run']} says row {h['claimed']}, genus "
             f"{h['quote']!r} resolves to row {h['true']}")
    if not live:
        print("  no live entry carries one yet -- INERT, not passing")

    print("C3 FROZEN: the historical mismatches are exactly the known four")
    hist = {(h["run"], h["true"], h["claimed"]) for h in bad if h["date"] < CUTOFF}
    for t in sorted(hist):
        print(f"  {t[0]}  genus row {t[1]}  body says row {t[2]}"
              f"{'' if t in KNOWN_BAD else '   <- NOT IN THE FROZEN SET'}")
    for t in sorted(hist - KNOWN_BAD):
        fail(f"new historical mismatch {t} -- not one of the four permanent ones")
    for t in sorted(KNOWN_BAD - hist):
        fail(f"frozen mismatch {t} is GONE -- JOURNAL.md is append-only, so it was rewritten")

    print("C4 the reader tracks its evidence")
    good = ('# 2026-09-09 — w900 — CLOSED. ANGLE: x\n\nHanded angle was slot 2, '
            '*"CatBoost: it usually handles categoricals better"* — row 3, closed.\n')
    cases = [
        ("a correct entry is silent", good, 0),
        ("the claim bumped fires", good.replace("row 3", "row 4"), 1),
        ("a quote naming no genus is not covered",
         good.replace("CatBoost: it usually handles categoricals better",
                      "some phrase naming none of the ten rows here"), None),
        ("an uncued quote is not covered",
         '# 2026-09-09 — w900 — x\n\nThe reader printed '
         '*"CatBoost: it usually handles categoricals better"* — row 4 was borrowed.\n', None),
        ("a claim beyond the window is not covered",
         good.replace("— row 3", "— " + "z" * (WINDOW + 20) + " row 3"), None),
        ("a claim before the quote is not covered",
         '# 2026-09-09 — w900 — x\n\nrow 4 — Handed angle was slot 2, '
         '*"CatBoost: it usually handles categoricals better"*.\n', None),
    ]
    for name, blob, want in cases:
        h = attributions(blob)
        if want is None:
            got = "not covered" if not h else f"covered ({h[0]['claimed']})"
            ok = not h
        else:
            n = len([x for x in h if x["true"] != x["claimed"]])
            got = f"{len(h)} covered, {n} wrong"
            ok = len(h) == 1 and n == want
        print(f"  {name}: {got}  {'OK' if ok else 'BROKEN'}")
        if not ok:
            fail(f"C4 {name}")
    # The uncued case must be covered once the cue is present, or the cue is just suppressing
    # everything rather than discriminating.
    if len(attributions('# 2026-09-09 — w900 — x\n\nThe handed angle was '
                        '*"CatBoost: it usually handles categoricals better"* — row 4.\n')) != 1:
        fail("C4 the cue suppresses a genuinely cued quote -- it is not discriminating")

    print("C5 no disagreement with the census about what a row is")
    print(f"  classify imported from w117a_handcount ({_hc.__file__.split('/')[-1]})")
    off = sorted({h["true"] for h in hits} - set(range(1, 11)))
    print(f"  covered rows {sorted({h['true'] for h in hits})}")
    if off:
        fail(f"rows outside 1..10: {off}")

    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
