# Negative controls for w105a_liveguard C4's exemption locator (w108, 2026-08-28).
#
# The defect being armed against: C4 used to find its exempt span with
#   next(i for i,l in ... if l.startswith("# ") and "w105" in l.lower())
# so ANY later `# ` header merely MENTIONING w105 captured the exemption. w108's block opened
# "# (w108, ...) — ... after w105/row 3, w106/row 4" and did exactly that: the span jumped to the
# top of the file, the real w105 section went red, and w108's own block silently gained cover.
set -u
cd /home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
PY=.venv/bin/python
say(){ printf '%-62s %s\n' "$1" "$2"; }
$PY - <<'PY'
import sys, io
sys.path.insert(0, 'experiments')
from w105a_liveguard import c4_scan, C4_HDR

DOC = io.open('RESEARCH.md', encoding='utf-8').read()
LIVE = f"# {C4_HDR} 2026-08-28) — use `experiments/alive.py` instead"
assert LIVE.split(')')[0] in DOC, "the w105 header moved; these arms are testing nothing"
ok = lambda n, c: print(f"{n:<62s} {'PASS' if c else 'FAIL'}")

# C1-  the live document is green: exactly one header, the anti-patterns all covered, none loose.
h, lo, hi, bad, cov, anti = c4_scan(DOC)
ok("C1- live document: one header, all anti-patterns covered", len(h) == 1 and not bad and cov)

# C1+  NON-VACUITY. If the live document held no anti-pattern at all, C1- would pass for a broken
#      scanner. Assert the thing being exempted actually exists.
ok("C1+ non-vacuous: the live document DOES contain the pattern", len(anti) >= 3)

# C2   THE REGRESSION, REPRODUCED. The live document ALREADY carries the trigger -- w108's own
#      subtitle -- which is why the suite went red today. So build a CLEAN baseline by dropping
#      that one line, then re-add it, and compare the two locators across the same pair.
DECOYS = [l for l in DOC.splitlines()
          if l.startswith("# ") and "w105" in l.lower() and C4_HDR not in l]
assert DECOYS, "no decoy header in the live document -- C2/C2b would be testing nothing"
TRIG = DECOYS[0]
CLEAN = "\n".join(l for l in DOC.splitlines() if l not in DECOYS) + "\n"
assert not any(l.startswith("# ") and "w105" in l.lower() and C4_HDR not in l
               for l in CLEAN.splitlines()), "CLEAN still holds a decoy header"
# The decoy must be PREPENDED: the old locator took the FIRST matching header, so a decoy after
# the real one changed nothing. Putting it first is what moved the span.
POISON = TRIG + "\n\n" + CLEAN
hc, loc_, _, _, _, _ = c4_scan(CLEAN)
h2, lo2, hi2, bad2, cov2, _ = c4_scan(POISON)
ok("C2  a header MENTIONING w105 does not capture the exemption",
   len(hc) == 1 and len(h2) == 1 and lo2 == loc_ + 2 and not bad2 and cov2)

# C2b  ...and the OLD locator really did break on that same input, so C2 is a live control and not
#      a test of something that never failed.
def old_locator(txt):
    L = txt.splitlines()
    lo = next((i for i, l in enumerate(L) if l.startswith("# ") and "w105" in l.lower()), None)
    hi = next((j for j in range(lo + 1, len(L))
               if L[j].startswith("# ") and "w105" not in L[j].lower()), len(L)) if lo is not None else len(L)
    ex = range(lo, hi) if lo is not None else range(0)
    return [L[i].strip()[:60] for i, l in enumerate(L) if "pgrep" in l and r"\|" in l and i not in ex]
ok("C2b the OLD locator DOES break on that input (control separates)",
   len(old_locator(CLEAN)) == 0 and len(old_locator(POISON)) >= 3)

# C3   A GENUINELY LOOSE PRESCRIPTION IS STILL CAUGHT. The fix must not have made C4 toothless.
h3, _, _, bad3, _, _ = c4_scan(CLEAN + "\n\nCheck it with `pgrep -af 'w96c_build\\|w26i_value'`.\n")
ok("C3  a loose pgrep BRE prescription is still caught", len(bad3) == 1)

# C4   A RENAMED HEADER FAILS LOUDLY (0 hits) instead of silently drifting onto another match.
h4, lo4, _, _, _, _ = c4_scan(CLEAN.replace(C4_HDR, "RESEARCH SUGGESTS (w105,"))
ok("C4  renamed section header -> 0 hits, span refused", len(h4) == 0 and lo4 is None)

# C5   A DUPLICATED HEADER ALSO FAILS. Two candidate spans must never be resolved by picking one.
h5, lo5, _, _, _, _ = c4_scan(CLEAN + "\n" + LIVE + "\n")
ok("C5  duplicated section header -> 2 hits, span refused", len(h5) == 2 and lo5 is None)

# C6   THE MISPLACEMENT DETECTOR. A span that covers no anti-pattern must report covered == [],
#      which is what makes main() fail. This is the arm that would have caught w108's bug directly.
moved = CLEAN.replace(LIVE, "# RESEARCH PRESCRIBES (w105, relocated) — decoy\n\nnothing here\n\n# X", 1)
h6, _, _, _, cov6, _ = c4_scan(moved)
ok("C6  an exemption covering nothing reports covered == [] (misplacement)",
   len(h6) == 1 and cov6 == [])
PY
