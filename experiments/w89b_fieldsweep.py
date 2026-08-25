"""w89b — sweep BOTH public indices and print only what this workspace has never named.

w84, w86 and w88 each wrote "the field is unchanged" on the strength of
`kernels list --sort-by dateRun`. That call does not cover the DATASETS index, which is where
every external OOF pack this workspace has ever imported came from (beicicc, najiama,
szymonkapiski, ravi20076, stephentarter). Swept on 2026-08-25 it returned SIX refs the journal
and RESEARCH have never named, the oldest published 2026-08-18. The conclusion was right and
the instrument did not cover the claim.

w90 found the same shape one level down: the DATASETS queries covered the spellings `s6e8` and
`smartphone-addiction` but not `s06e08`, and `stephentarter/ps-s06e08-nn-tabular-predictions`
(20.7 MB of NN OOF + test probabilities, uploaded 2026-08-25) is reachable by NO other query in
the set. Right index, wrong spelling. `REACH` below is the standing regression test for it.

Membership is decided by whether the ref's owner/slug appears anywhere in JOURNAL.md or
RESEARCH.md -- the workspace's own memory is the manifest, so there is no third file to keep
in sync and nothing can be "seen" without being written down.

    .venv/bin/python experiments/w89b_fieldsweep.py            # both indices, new only
    .venv/bin/python experiments/w89b_fieldsweep.py --all      # show everything
    .venv/bin/python experiments/w89b_fieldsweep.py --selftest # offline, no API
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMP = "playground-series-s6e8"
MEMORY = ["JOURNAL.md", "RESEARCH.md", "LEADERBOARD.md"]


def memory_text():
    t = []
    for f in MEMORY:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            t.append(open(p, errors="replace").read().lower())
    if not t:
        raise SystemExit("no memory files found -- refusing to call everything new")
    return "\n".join(t)


def named(token, mem):
    """Is `token` written down as a WHOLE name, not as a substring of a longer one?

    ⚠ PLAIN `in` IS NOT ENOUGH, and this cost the first cut of this file its best find.
    `nhtquyn/s6e8-addiction` was marked KNOWN because `s6e8-addiction` is a substring of
    najiama's `s6e8-addiction-lb-0-97092`, which is a different artefact by a different
    author. A ref boundary is any character that cannot continue a slug.
    """
    t = token.lower()
    if not t:
        return False
    for m in re.finditer(re.escape(t), mem):
        a, b = m.start(), m.end()
        before = mem[a - 1] if a else " "
        after = mem[b] if b < len(mem) else " "
        if not re.match(r"[\w-]", before) and not re.match(r"[\w-]", after):
            return True
    return False


def seen(ref, mem):
    """('slug' | 'owner' | None) -- HOW the ref is known, not just whether.

    Two tiers on purpose. An unknown OWNER is a genuinely new participant and is the strong
    signal. A known owner with an unnamed slug is the weak one: hboyang is dispositioned in
    the ledger, but `hboyang/s6e8-catstrall-member` is still a pack nobody has opened, and
    collapsing the two tiers is what lets a known author's new upload go unlooked-at.
    """
    owner, _, slug = ref.partition("/")
    if named(slug, mem):
        return "slug"
    # The stripped slug must still be SPECIFIC: `s6e8-addiction` strips to `addiction`, a
    # word on nearly every line of the memory. Require a hyphen and 10 characters, which
    # keeps `150-fusion-local-members` (the journal really does write that short form).
    stripped = re.sub(r"^(pg-)?s6e8[-_]", "", slug.lower())
    if len(stripped) >= 10 and "-" in stripped and named(stripped, mem):
        return "slug"
    if named(owner, mem):
        return "owner"
    return None


def cli(args):
    out = subprocess.run(["kaggle"] + args, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        print(f"  ⚠ `kaggle {' '.join(args)}` rc={out.returncode}: "
              f"{(out.stderr or out.stdout).strip()[:200]}")
        return []
    rows = []
    for ln in out.stdout.splitlines():
        m = re.match(r"^([A-Za-z0-9][\w.-]*/[\w.-]+)\s+(.*)$", ln.strip())
        if m:
            rows.append((m.group(1), " ".join(m.group(2).split())[:78]))
    return rows


REACH = {
    # Refs that are KNOWN TO EXIST and are each reachable only through one query in the set
    # below. They are the sweep's own regression test: if the union of the queries stops
    # returning one of these, the sweep's COVERAGE has narrowed and it will go on printing
    # "nothing new" from a hole. w89 fixed the wrong INDEX; this catches the wrong SPELLING.
    #
    # ⛔ Do not delete a ref from this set to make the check pass. Confirm with
    #    `kaggle datasets files <ref>` that the owner really removed it, and say so here.
    "stephentarter/ps-s06e08-nn-tabular-predictions":
        "reachable ONLY as s06e08; invisible to -s s6e8 and -s smartphone-addiction (w90)",
    "nhtquyn/s6e8-addiction":
        "the pack w89 had to find by hand; -s s6e8 (w89)",
    "atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124":
        "-s s6e8; the 0.97124 anchor behind the 0.97127 cluster (w90)",
}


def main():
    mem = memory_text()
    show_all = "--all" in sys.argv
    total_new = total_weak = 0
    reached = set()
    for label, args in [
        ("DATASETS  (-s s6e8, by updated)", ["datasets", "list", "-s", "s6e8",
                                            "--sort-by", "updated", "--page-size", "50"]),
        ("DATASETS  (-s smartphone-addiction)", ["datasets", "list", "-s", "smartphone-addiction",
                                                 "--sort-by", "updated", "--page-size", "50"]),
        ("DATASETS  (-s s06e08, THE OTHER SPELLING)", ["datasets", "list", "-s", "s06e08",
                                                       "--sort-by", "updated", "--page-size", "50"]),
        ("KERNELS   (by dateRun)", ["kernels", "list", "--competition", COMP,
                                    "--sort-by", "dateRun", "--page-size", "50"]),
        ("KERNELS   (by voteCount)", ["kernels", "list", "--competition", COMP,
                                      "--sort-by", "voteCount", "--page-size", "50"]),
    ]:
        rows = cli(args)
        reached.update(r[0] for r in rows)
        tier = {r[0]: seen(r[0], mem) for r in rows}
        fresh = [r for r in rows if tier[r[0]] is None]
        weak = [r for r in rows if tier[r[0]] == "owner"]
        total_new += len(fresh)
        total_weak += len(weak)
        print(f"\n{label}: {len(rows)} rows | {len(fresh)} UNKNOWN OWNER | "
              f"{len(weak)} known owner, unnamed slug")
        for ref, rest in (rows if show_all else fresh + weak):
            mark = {None: ">>>", "owner": " ~ ", "slug": "   "}[tier[ref]]
            print(f"  {mark}{ref:62s} {rest}")

    print(f"\n>>> UNKNOWN OWNER: {total_new}    ~ known owner, unnamed slug: {total_weak}")
    print("A ref here is UNREAD, not USEFUL. Triage it in the journal either way, so the next"
          "\nsweep does not re-surface it -- writing it down is what makes it 'seen'.")

    missing = [r for r in REACH if r not in reached]
    print(f"\nCOVERAGE  {len(REACH) - len(missing)}/{len(REACH)} known-live refs returned by the query set")
    for r in missing:
        print(f"  ⛔ NOT REACHED  {r}\n                 {REACH[r]}")
    if missing:
        print("A ref that exists and is not returned means the QUERY SET has a hole, which reads"
              "\nexactly like a quiet field. Widen the queries; do not shrink REACH.")
        return 1
    return 0


def selftest():
    """Offline. The parser and the membership rule are the two things that can silently rot."""
    mem = ("beicicc szymonkapiski hboyang najiama 150-fusion-local-members "
           "najiama/s6e8-addiction-lb-0-97092 "
           "atakanaldemir/s6e8-regime-calibrated-rank-fusion-lb-0-97127").lower()
    cases = [
        ("beicicc/s6e8-fixed900-structural-lgbm-artifacts", "owner", "owner named, slug never opened"),
        ("hboyang/s6e8-150-fusion-local-members",           "slug",  "journal writes the short slug"),
        ("nobody/s6e8-brand-new-pack",                      None,    "genuinely new"),
        ("atakanaldemir/s6e8-v13-diversity-anchor",         "owner", "owner named via the 0.97127 read"),
        ("nobody/pg-s6e8-exp999",                           None,    "prefix strip must not match ''"),
        ("nhtquyn/s6e8-addiction",                          None,    "substring of najiama's slug -- must NOT count"),
        # ⚠ NOT a control: `nobody/s6e8-beicicc` (an unknown owner re-hosting a known pack)
        # SHOULD surface as new. The specificity rule drops the 7-char stripped slug and that
        # is the wanted behaviour, not a gap -- a re-upload is a thing to look at.
    ]
    fails = 0
    for ref, want, why in cases:
        got = seen(ref, mem)
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} seen({ref:48s}) = {got!s:5s} want {want!s:5s}  {why}")
    for tok, want, why in [("addiction", False, "a bare word inside a longer slug is not a name"),
                           ("beicicc", True, "a whole token is"),
                           ("eicicc", False, "a suffix of a name is not the name")]:
        got = named(tok, mem)
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} named({tok:20s}) = {got!s:5s} want {want!s:5s}  {why}")

    # The row parser must survive the header, the rule line and a ref with dots/dashes.
    sample = ("ref                     title      size\n"
              "----------------------  ---------  ----\n"
              "a.b-c/d_e-f             Some Title 123\n"
              "not-a-ref-line\n")
    import io
    rows = [m.group(1) for ln in sample.splitlines()
            for m in [re.match(r"^([A-Za-z0-9][\w.-]*/[\w.-]+)\s+(.*)$", ln.strip())] if m]
    ok = rows == ["a.b-c/d_e-f"]
    fails += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} parser skips header/rule/junk, keeps {rows}")

    # An empty memory must RAISE, not declare the whole field new.
    global MEMORY
    keep, MEMORY = MEMORY, ["does-not-exist.md"]
    try:
        memory_text()
        print("  FAIL empty memory did not raise")
        fails += 1
    except SystemExit:
        print("  ok   empty memory raises instead of calling everything new")
    finally:
        MEMORY = keep

    print(f"\nselftest failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
