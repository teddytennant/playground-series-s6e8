"""w180a -- THE RUNNING "Nth CONSECUTIVE RUN" COUNTER IN JOURNAL.md IS A CENSUS, AND NOTHING
EVER COUNTED IT (2026-09-05).

🎯 THE DEFECT THIS EXISTS BECAUSE OF. Every closed-board entry since w143 opens with an ordinal:

    w154   **Twelfth consecutive run with no submission.**
    w156   THIRTEENTH CONSECUTIVE RUN WITH NO SUBMISSION.
    w179   **Thirty-sixth consecutive run with no submission, seventh of the second full pass.**

That ordinal is a statement about the corpus -- "count the runs behind me" -- exactly like the
`×N` on an ANGLE INDEX trail cell, which #50 re-derives every single run. But it lives in
JOURNAL.md, it is never re-derived, and each run computes it the cheapest way available: read
the number in the entry directly above, add one. So it is a CHAIN, and a chain has the property
that one bad link is inherited by every link after it, permanently, because JOURNAL.md is
append-only.

🔴 THERE IS ONE BAD LINK AND IT HAS BEEN COPIED 24 TIMES.

    w154   ord 184   claims 12   measured 12   OK
    w155   ord 185   claims --   measured 13   <- LEFT NO ENTRY, so no ordinal was written
    w156   ord 186   claims 13   measured 14   <- read w154's 12, added one, SKIPPED w155
    ...    every run after it inherits the same -1
    w179   ord 209   claims 36   measured 37

⚠ AND THE RUN THAT BROKE IT IS THE RUN THAT FIXED THE SAME DEFECT ON THE OTHER SURFACE, IN THE
SAME ENTRY. w155's session died before it wrote anything to JOURNAL.md. w156 found the work on
disk and reconstructed w155's entry precisely so that the corpus would contain the run the ANGLE
INDEX already claimed -- its own header reads "THE CENSUS THAT WATCHES THE ANGLE INDEX STAYED
GREEN OVER A RUN THAT LEFT NO RECORD". It repaired the census over w155 and, four paragraphs
later, incremented its own counter off w154 as though w155 had never happened. One run, two
censuses over the same missing entry, one fixed and one broken.

🎯 WHY IT WAS INVISIBLE, WHICH IS THE PART THAT GENERALISES. The reconstructed w155 entry
deliberately carries no ordinal -- w156 would have had to invent a number w155 never wrote, and
inventing one is the thing the reconstruction was careful not to do. So the series READS as
unbroken: 12, 13, 14, ... with no gap in the numbers, only a gap in the runs. A chain checked
for self-consistency is green here. Only a chain checked against the corpus is not, and no
reader had ever fetched the corpus. This is #72's spread mechanism (four runs copying a
predecessor's wrong row) at 24 runs and a different quantity.

⚠ A SECOND, INDEPENDENT DEFECT ON THE SAME PHRASE: THE LABEL CHANGED AND THE NUMBER DID NOT.
w143 opened the series as `NO WORK DONE, BY DESIGN`, and w144..w149 continue it as `Nth
CONSECUTIVE NO-WORK RUN`. At w150 the wording silently became `Nth CONSECUTIVE RUN WITH NO
SUBMISSION` while the number carried straight on from the no-work series. Those are two
different quantities: the last run that did no work is w142 (ord 172), but the last run that
actually SENT anything is w132 (ord 163), which drained ten files at 12:36-12:37Z on 08-31 --
confirmed against the Kaggle API, whose newest row is dated 2026-08-31 12:37:26. The nine runs
between them (w133, w135..w142) did substantial work and sent nothing, so under the words
actually on the page every ordinal since w150 is understated by a further 9. This is #53's
shape -- a column that changed quantity without changing its number -- and C6 measures it rather
than arguing about it.

WHICH READING THIS GUARD ENFORCES, AND WHY. The no-WORK one, anchored at w142. Not because it
is the better English -- it is not -- but because it is the series' actual construction, and
under it the defect is exactly one skipped link rather than a diffuse disagreement about
wording. Enforcing the literal reading instead would paint all 35 historical claims red and
tell a future run nothing about which of them was a mistake. C6 still publishes the literal
number so the disagreement is on the record and not quietly resolved by fiat.

⛔ THE 24 CANNOT BE CORRECTED -- append-only, the same permanence that gives #71 its two
standing reds and #72 its frozen four. So this guard does not fail on them; it FREEZES them,
and splits live from historical so the arm that can go red is the one a future run can act on.

    .venv/bin/python experiments/w180a_countguard.py     # rc 0 = green

Deterministic: reads JOURNAL.md only. No API call, no model fit, no git.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# C5: what a RUN is comes from #50 and is never re-implemented here, so the two censuses cannot
# disagree about the denominator. Importing the compiled pattern, not a copy of its source.
from w117a_handcount import RUN_HDR  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(HERE, "JOURNAL.md")

MIN_RUNS, MIN_CLAIMS = 180, 20

# The last run that did competition WORK: w142 graded the finished forecast on 09-01. w143 is
# the series' own first link ("NO WORK DONE, BY DESIGN"), so measured(k) = k - ANCHOR_WORK_ORD.
ANCHOR_WORK, ANCHOR_WORK_ORD = "w142", 172

# The last run that actually SUBMITTED: w132 drained ten files at 12:36-12:37Z on 2026-08-31.
# Frozen as a literal on w115's rule -- a control anchored to a moving reference (the live API)
# stops being a control. C1 re-derives it from the corpus and requires the two to agree.
ANCHOR_SEND, ANCHOR_SEND_ORD = "w132", 163

# The 24 inherited copies of the w155 skip, verbatim as they stand in the append-only file.
# A 25th is a NEW defect; one of these disappearing is an append-only violation. Both FAIL.
KNOWN_BAD = {
    "w156": 13, "w157": 14, "w158": 15, "w159": 16, "w160": 17, "w161": 18,
    "w162": 19, "w163": 20, "w164": 21, "w165": 22, "w166": 23, "w167": 24,
    "w168": 25, "w169": 26, "w170": 27, "w171": 28, "w172": 29, "w173": 30,
    "w174": 31, "w175": 32, "w176": 33, "w177": 34, "w178": 35, "w179": 36,
}

# The run whose missing entry broke the chain, and the run that wrote its reconstruction.
SKIPPED, SKIPPED_BY = "w155", "w156"

_ONES = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
         "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
         "seventeenth", "eighteenth", "nineteenth", "twentieth"]
WORDS = {w: n for n, w in enumerate(_ONES, 1)}
for _t, _v, _tw in (("twenty", 20, "twentieth"), ("thirty", 30, "thirtieth"),
                    ("forty", 40, "fortieth"), ("fifty", 50, "fiftieth")):
    WORDS[_tw] = _v
    for _n, _w in enumerate(_ONES[:9], 1):
        WORDS[f"{_t}-{_w}"] = _v + _n

# Both surface forms of the counter. The NO-WORK arm is the series' first six links; the
# NO SUBMISSION arm is every link from w150. Matching only the second loses the anchor.
CLAIM = re.compile(
    r"\b([A-Za-z]+(?:-[a-z]+)?)\s+CONSECUTIVE\s+(?:NO-WORK\s+RUN|RUNS?\s+WITH\s+NO\s+SUBMISSION)",
    re.I)
RID = re.compile(r"\b(w\d+[a-z]?)\b")

# ⚠ A QUOTED COUNTER IS NOT THE RUN'S OWN COUNTER, AND THE FIRST VERSION OF THIS READER GOT THAT
# WRONG ON THE FIRST ENTRY THAT DISCUSSED THE DEFECT. w180's body quotes w179's "Thirty-sixth ..."
# as evidence, several paragraphs above its own "Thirty-eighth ...", and taking the first match in
# the body read the quotation as w180's claim. Same trap #50 documents for ANGLE_Q: a quote is not
# a declaration. A claim is DISOWNED when another run's id appears within DISOWN_WIN characters
# before it, and the run's own claim is the last one left standing.
#
# THE WINDOW IS MEASURED, NOT GUESSED. Over the whole corpus the nearest preceding other-run id
# for a GENUINE claim is 62-64 characters away, and it is always an incidental citation -- the
# stock "(w143: a write against a closed competition is not a free read). **Nth ...". For the one
# real quotation it is 8. Anything in [9, 61] separates them; 30 is the midpoint by order of
# magnitude and matches #50's own ANGLE_Q_NARROW. C4(g) re-measures both distances every run, so
# the margin cannot quietly close.
DISOWN_WIN = 30

# ⚠ AND THE DISTANCE RULE ALONE IS NOT ENOUGH, WHICH THIS ENTRY ALSO PROVED. w180's addendum
# quotes the same counter a second time as "the diagnosis section above quotes *"Thirty-sixth
# ..."" -- no run id within 30 characters, so the distance rule kept it, and because it sits
# AFTER the run's real claim the "last one standing" selector picked the quotation again.
# A quote mark immediately before the ordinal is a direct syntactic signal rather than a
# heuristic about distance, so it goes first. Measured over the corpus: 38 raw matches, exactly
# 2 preceded by a quote mark, and both are w180 quoting w179. No genuine claim opens on a quote.
QUOTED = re.compile(r"[“”\"‘’'][*_]{0,2}\s*$")

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  FAIL: {msg}")


def runs(text: str):
    """Every run in the corpus, in order, as (ordinal, run_id, flattened_body)."""
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if RUN_HDR.match(l)]
    out = []
    for k, i in enumerate(idx):
        end = idx[k + 1] if k + 1 < len(idx) else len(lines)
        rid = RID.search(lines[i])
        body = re.sub(r"\s+", " ", "\n".join(lines[i:end]))
        out.append((k, rid.group(1) if rid else f"?@{i}", body))
    return out


def own_claims(rid: str, body: str):
    """Every counter in this body that the run states about ITSELF, quotations dropped."""
    keep = []
    for m in CLAIM.finditer(body):
        if QUOTED.search(body[max(0, m.start() - 6):m.start()]):
            continue  # opens on a quote mark -- someone else's sentence, verbatim
        pre = body[max(0, m.start() - DISOWN_WIN):m.start()]
        if any(x != rid for x in RID.findall(pre)):
            continue  # attributed to another run -- evidence, not a declaration
        keep.append(m)
    return keep


def claims(corpus):
    """(ordinal, run_id, claimed_n, surface_form) for every run carrying a counter."""
    out = []
    for k, rid, body in corpus:
        keep = own_claims(rid, body)
        if not keep:
            continue
        m = keep[-1]
        n = WORDS.get(m.group(1).lower())
        form = "no-work" if "NO-WORK" in m.group(0).upper() else "no-sub"
        out.append((k, rid, n, form))
    return out


def main() -> int:
    text = open(JOURNAL, encoding="utf-8").read()
    corpus = runs(text)
    cl = claims(corpus)

    print("C1 the read is answerable, and a read that covers nothing FAILS")
    print(f"  {len(corpus)} run(s) in the corpus, floor {MIN_RUNS}")
    if len(corpus) < MIN_RUNS:
        fail(f"only {len(corpus)} runs parse; the corpus reader is broken, not the counter")
    print(f"  {len(cl)} run(s) carry a counter, floor {MIN_CLAIMS}")
    if len(cl) < MIN_CLAIMS:
        fail(f"only {len(cl)} counters parse; passing on this would be passing blind")
    unparsed = [(k, r) for k, r, n, _ in cl if n is None]
    if unparsed:
        fail(f"ordinal word(s) not in WORDS at {unparsed} -- extend WORDS, do not drop them")
    by_id = {r: k for k, r, _ in corpus}
    for name, want in ((ANCHOR_WORK, ANCHOR_WORK_ORD), (ANCHOR_SEND, ANCHOR_SEND_ORD)):
        got = by_id.get(name)
        ok = got == want
        print(f"  anchor {name} sits at ordinal {got}, frozen literal says {want}  "
              f"{'OK' if ok else 'MOVED'}")
        if not ok:
            fail(f"anchor {name} moved to {got}; every measured number below is off by the same")

    print("\nC2 LIVE -- every counter not in the frozen set must equal the corpus")
    live = [(k, r, n) for k, r, n, _ in cl if r not in KNOWN_BAD]
    bad = [(k, r, n, k - ANCHOR_WORK_ORD) for k, r, n in live if n != k - ANCHOR_WORK_ORD]
    for k, r, n, m in bad:
        fail(f"{r} (ord {k}) claims {n} consecutive, the corpus measures {m}")
    print(f"  {len(live)} live counter(s), {len(bad)} disagreeing with the corpus  "
          f"{'OK' if not bad else 'RED'}")
    if not bad and live:
        span = f"{live[0][1]}..{live[-1][1]}"
        print(f"  {span} all agree; the newest is {live[-1][1]} at {live[-1][2]}")

    print("\nC3 FROZEN -- the inherited copies are exactly KNOWN_BAD, no more and no fewer")
    seen = {r: n for k, r, n, _ in cl if r in KNOWN_BAD}
    missing = sorted(set(KNOWN_BAD) - set(seen))
    if missing:
        fail(f"frozen counter(s) {missing} no longer in JOURNAL.md -- append-only was violated")
    changed = {r: (KNOWN_BAD[r], seen[r]) for r in seen if seen[r] != KNOWN_BAD[r]}
    if changed:
        fail(f"frozen counter(s) rewritten in place: {changed} -- append-only was violated")
    newly = [(k, r, n) for k, r, n, _ in cl
             if r not in KNOWN_BAD and n is not None and n != k - ANCHOR_WORK_ORD]
    print(f"  {len(seen)}/{len(KNOWN_BAD)} frozen counters present and unchanged  "
          f"{'OK' if not missing and not changed else 'RED'}")
    print(f"  {len(newly)} counter(s) outside the frozen set disagree (a 25th copy would show "
          f"here)  {'OK' if not newly else 'RED'}")

    print("\nC4 the reader tracks its evidence rather than asserting it")
    # (a) the skipped run is really in the corpus, and really carries no counter
    in_corpus = SKIPPED in by_id
    has_claim = any(r == SKIPPED for _, r, _, _ in cl)
    print(f"  {SKIPPED} is in the corpus: {in_corpus}; carries a counter: {has_claim}  "
          f"{'OK' if in_corpus and not has_claim else 'BROKEN'}")
    if not in_corpus or has_claim:
        fail(f"{SKIPPED} is not the shape the diagnosis claims")
    # (b) the break is a single point, not a drift: exactly one ordinal where the offset changes.
    # Scoped to the FROZEN era, which is immutable and where the skip is the whole story. A live
    # run that repays the skip necessarily moves the offset back to 0, and that step is the fix
    # landing, not a second defect -- reading it as one is what made this arm red on a correct
    # entry the first time it was perturbed.
    last_frozen = max(by_id[r] for r in KNOWN_BAD if r in by_id)
    offsets = [(r, n - (k - ANCHOR_WORK_ORD)) for k, r, n, _ in cl
               if n is not None and k <= last_frozen]
    steps = [(offsets[i][0], offsets[i - 1][1], offsets[i][1])
             for i in range(1, len(offsets)) if offsets[i][1] != offsets[i - 1][1]]
    print(f"  offset changes over the frozen era (to ord {last_frozen}): "
          f"{[(r, a, b) for r, a, b in steps]}  "
          f"{'OK -- one point' if len(steps) == 1 else 'NOT A SINGLE SKIP'}")
    if len(steps) != 1 or steps[0][0] != SKIPPED_BY:
        fail(f"the break is not the single skip at {SKIPPED_BY} the diagnosis claims")
    repaid = [(r, o) for r, o in
              [(r, n - (k - ANCHOR_WORK_ORD)) for k, r, n, _ in cl
               if n is not None and k > last_frozen] if o == 0]
    if repaid:
        print(f"  live run(s) carrying the repaid offset 0: {[r for r, _ in repaid]}  OK")
    # (c) the pre-break links are green, so the anchor is not being fitted to make them so
    pre = [(r, n) for k, r, n, _ in cl if n is not None and n == k - ANCHOR_WORK_ORD]
    print(f"  {len(pre)} counter(s) before the break already agree with the corpus: "
          f"{[r for r, _ in pre][:12]}  {'OK' if len(pre) >= 6 else 'THIN'}")
    if len(pre) < 6:
        fail("too few links agree with the anchor for it to be the series' real origin")
    # (d) a synthetic correct next link passes and a synthetic off-by-one does not
    nxt = len(corpus) - ANCHOR_WORK_ORD
    probe_ok = nxt == (len(corpus) - 1 - ANCHOR_WORK_ORD) + 1
    print(f"  the next run's correct counter is {nxt}; naive increment off {cl[-1][1]} gives "
          f"{cl[-1][2] + 1}  {'DIFFER -- the skip must be repaid' if nxt != cl[-1][2] + 1 else 'same'}")
    if not probe_ok:
        fail("the arithmetic of the next link does not close")
    # (e) the reader is not a blanket matcher: a body with no counter is not covered
    quiet = len(claims([(0, "wSYNTH", "a run that says nothing about consecutive anything")]))
    print(f"  a run claiming no counter is not covered: {quiet} covered  "
          f"{'OK' if quiet == 0 else 'BROKEN'}")
    if quiet:
        fail("the reader covers a run that makes no counter claim")
    # (f) both surface forms are actually exercised, or the regex's second arm is dead code
    forms = {f for *_, f in cl}
    print(f"  surface forms matched: {sorted(forms)}  "
          f"{'OK' if forms == {'no-work', 'no-sub'} else 'ONE ARM IS DEAD'}")
    if forms != {"no-work", "no-sub"}:
        fail(f"CLAIM matched only {sorted(forms)}; the other arm is untested")

    # (g) the disown window discriminates, and by how much. A cue that merely suppresses
    # everything is not a reader, so both sides of the margin are measured on live text.
    kept_d, dropped_d, quoted = [], [], []
    for k, rid, body in corpus:
        for m in CLAIM.finditer(body):
            if QUOTED.search(body[max(0, m.start() - 6):m.start()]):
                quoted.append(rid)
                continue
            pre = body[max(0, m.start() - 200):m.start()]
            others = [x for x in RID.findall(pre) if x != rid]
            d = len(pre) - pre.rfind(others[-1]) - len(others[-1]) if others else 10 ** 6
            (dropped_d if d < DISOWN_WIN else kept_d).append((rid, d))
    raw = sum(len(CLAIM.findall(b)) for _, _, b in corpus)
    print(f"  {raw} raw match(es); {len(quoted)} dropped on a quote mark {sorted(set(quoted))}, "
          f"{len(dropped_d)} on attribution {sorted({r for r, _ in dropped_d})}")
    near = min((d for _, d in kept_d if d < 10 ** 6), default=None)
    far = max((d for _, d in dropped_d), default=None)
    print(f"  disown window {DISOWN_WIN}: nearest other-run id on a KEPT claim is {near}, "
          f"furthest on a DROPPED one is {far}")
    print(f"  {len(dropped_d)} quotation(s) dropped: {sorted({r for r, _ in dropped_d})}  "
          f"{'OK' if dropped_d else 'nothing to drop yet'}")
    if near is not None and not (far or 0) < DISOWN_WIN <= near:
        fail(f"the disown window does not separate quotations ({far}) from claims ({near})")

    print("\nC5 no disagreement with #50 about what a run is")
    print("  RUN_HDR imported from w117a_handcount -- one implementation, not two")
    print(f"  denominator {len(corpus)} runs, the same list #50 counts the ANGLE INDEX against")

    print("\nC6 the label and the number are different quantities, measured not argued")
    work_forms = [r for _, r, _, f in cl if f == "no-work"]
    sub_forms = [r for _, r, _, f in cl if f == "no-sub"]
    print(f"  {len(work_forms)} counter(s) say NO-WORK ({work_forms[0]}..{work_forms[-1]}), "
          f"{len(sub_forms)} say NO SUBMISSION ({sub_forms[0]}..{sub_forms[-1]})")
    gap = ANCHOR_WORK_ORD - ANCHOR_SEND_ORD
    k_last, r_last, n_last, _ = cl[-1]
    print(f"  anchors differ by {gap} run(s): {ANCHOR_SEND} (ord {ANCHOR_SEND_ORD}, last to SEND) "
          f"-> {ANCHOR_WORK} (ord {ANCHOR_WORK_ORD}, last to WORK)")
    print(f"  newest counter {r_last}: claims {n_last} · no-work {k_last - ANCHOR_WORK_ORD} · "
          f"literal no-submission {k_last - ANCHOR_SEND_ORD}")
    print(f"  so every counter since {sub_forms[0]} is understated by {gap} under the words on "
          f"the page, and by a further 1 from the {SKIPPED} skip -- {gap + 1} in total  "
          f"(RECORDED, not enforced; see the docstring)")
    if gap <= 0:
        fail("the two anchors do not bracket the way the diagnosis describes")

    print(f"\nFAILURES: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
