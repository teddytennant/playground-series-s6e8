"""w109 (2026-08-28) — GUARD #44: the enrolled member matrix has no duplicate columns.

THE FINDING THIS EXISTS TO STOP COMING BACK
-------------------------------------------
`bolt_xgb_d7_alt1` and `bolt_xgb_d7_alt2` are the SAME ARRAY -- byte-identical on the OOF
side and on the test side. That was found on 2026-08-11, written into RESEARCH.md under
"are the SAME ARRAY -- drop one", and acted on ONCE: `blend150sx` dropped `_alt2` as a
build argument and measured the delta (a null, sign-flipping across transforms).

⚠ AND THEN IT CAME BACK, because the fix lived in one build's argument list and never
entered a standing drop list. `stack.DEFAULT_DROP` is ("golem_a","golem_f");
`blend_lab.HONEST_DROP` adds two `lgbm_tuned_lat*`; `w36b_run.sh` -- the recipe that built
the DEADLINE PICK -- adds `lat_ctraw_r400`, `lat_ctfixte_r400`, `om_cat`. None of them names
a bolt member. So the shipped 199-member pack enrols the duplicate, and `load_members` has
no opinion about it: it takes every `oof_*/test_*` pair it finds.

🎯 THE SHAPE, and it is the reason this is a check and not a paragraph: **a fix applied as a
build argument is not applied to the workspace.** The next build does not inherit it. The
same document that says "drop one" is the document the build never reads.

WHY THE DROP LISTS ARE NOT SIMPLY EDITED. Every shipping chain passes `--drop` explicitly,
so editing `HONEST_DROP` would change nothing that ships while making a future default-drop
run non-comparable with every past one -- a reproducibility cost for no gain, three days
out, with the deadline pick settled. The duplicate's price is measured in
`w109a_dupscan.json` (arm D), not assumed. This guard freezes the census instead.

THE EXEMPTION, AND WHY IT IS ALLOWED TO EXIST
---------------------------------------------
Exactly one pair is exempt, by name. The exemption surface of the underlying rule is ZERO --
two byte-identical member columns are never legitimate -- so this is a single dated carve-out
for a known, measured instance, not a judgement call the guard launders as a check (w108 §6).
And per w108 §4 the carve-out must PROVE IT IS COVERING SOMETHING: C2 fails if the exempt
pair stops being duplicated, so a stale exemption cannot sit here reporting itself present.

  C1  the shipped pack loads at its expected size and both exempt members are in it
  C2  the exempt pair is STILL byte-identical -- otherwise the exemption is stale, FAIL
  C3  no OTHER group of byte-identical columns exists
  C4  no OTHER group of rank-identical columns exists (AUC sees nothing but ranks, so two
      columns with different values and identical ranks are one member to this metric)
  C5  the census function reproduces the finding: an injected duplicate that is not the
      exempt pair must be caught, and the exempt pair alone must come back clean

C5 runs against the extracted function in-process, so the guard has no flag or env var that
could repoint it at a synthetic pack in production (w108 §4).

Exit 0 = the enrolled matrix is one member per column, plus one known and still-real pair.
"""
from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend_lab as BL  # noqa: E402,F401  -- pins the BLAS thread count on import

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members  # noqa: E402

from w109a_dupscan import BASEDIRS, DROP, EXPECT, XDIRS  # noqa: E402

# The one known duplicate, found 2026-08-11, priced at 199 members in w109a_dupscan.json.
EXEMPT = frozenset({"bolt_xgb_d7_alt1", "bolt_xgb_d7_alt2"})


def dupe_groups(names, digests, exempt=EXEMPT):
    """Groups of columns sharing a digest, minus the one exempt pair.

    Returns (unexpected_groups, exempt_group_seen). `exempt_group_seen` is what lets the
    caller fail a carve-out that has stopped covering anything.
    """
    by = {}
    for n, d in zip(names, digests):
        by.setdefault(d, []).append(n)
    groups = [sorted(v) for v in by.values() if len(v) > 1]
    seen = any(set(g) == set(exempt) for g in groups)
    return [g for g in groups if set(g) != set(exempt)], seen


def _sha(a):
    return hashlib.sha1(np.ascontiguousarray(a)).hexdigest()


def main() -> int:
    print("=" * 92)
    print("w109b  MEMBER-COLUMN DUPLICATE GUARD")
    print("=" * 92)
    fail = []

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = tuple(os.path.join(DATA, d)
                  for d in BASEDIRS + tuple(XDIRS.split(",")))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(DROP.split(",")))

    # --- C1 the premise -------------------------------------------------------------
    if len(names) != EXPECT:
        fail.append(f"C1 pack is {len(names)} members, w109a_dupscan.EXPECT is {EXPECT}")
    missing = sorted(EXEMPT - set(names))
    if missing:
        fail.append(f"C1 exempt member(s) absent from the pack: {missing} -- the "
                    "exemption names something that is no longer enrolled, remove it")
    print(f"C1 pack {len(names)} members (expected {EXPECT}); "
          f"exempt pair enrolled: {not missing}")

    # --- C2 the exemption must still be covering something --------------------------
    if not missing:
        a, b = sorted(EXEMPT)
        ia, ib = names.index(a), names.index(b)
        same_o = np.array_equal(O[:, ia], O[:, ib])
        same_t = np.array_equal(T[:, ia], T[:, ib])
        if not (same_o and same_t):
            fail.append(f"C2 {a} and {b} are NO LONGER identical "
                        f"(oof {same_o}, test {same_t}) -- the exemption is stale, "
                        "delete it rather than carrying a carve-out that excuses nothing")
        print(f"C2 {a} == {b}: oof {same_o}, test {same_t}")

    # --- C3 byte-level census -------------------------------------------------------
    dig = [(_sha(O[:, j]), _sha(T[:, j])) for j in range(len(names))]
    bad, seen = dupe_groups(names, dig)
    if bad:
        fail.append(f"C3 byte-identical column group(s) beyond the exemption: {bad}")
    print(f"C3 byte-identical groups: exempt pair seen {seen}, unexpected {bad or 'none'}")

    # --- C4 rank-level census -------------------------------------------------------
    rdig = [(_sha(np.argsort(O[:, j], kind="stable")),
             _sha(np.argsort(T[:, j], kind="stable"))) for j in range(len(names))]
    rbad, rseen = dupe_groups(names, rdig)
    if rbad:
        fail.append(f"C4 rank-identical column group(s) beyond the exemption: {rbad}")
    print(f"C4 rank-identical groups: exempt pair seen {rseen}, unexpected {rbad or 'none'}")

    print(f"   distinct members: {len(set(dig))} of {len(names)} enrolled")

    # --- C5 the census reproduces the finding ---------------------------------------
    a, b = sorted(EXEMPT)
    only = dupe_groups([a, b, "x", "z"], ["h", "h", "p", "q"])
    inj = dupe_groups([a, b, "x", "z"], ["h", "h", "p", "p"])
    two = dupe_groups(["x", "z"], ["p", "p"])
    ok5 = (only == ([], True) and inj == ([["x", "z"]], True)
           and two == ([["x", "z"]], False))
    if not ok5:
        fail.append(f"C5 census function does not reproduce the finding: "
                    f"exempt-only {only}, injected {inj}, no-exempt {two}")
    print(f"C5 census reproduces the finding: {ok5}")

    print("-" * 92)
    if fail:
        for f in fail:
            print("FAIL " + f)
        return 1
    print("PASS  one member per column, plus one known and still-real duplicate pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
