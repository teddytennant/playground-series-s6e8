"""w40b — APPLY the w38d pre-registered promotion rule to ARM 202, mechanically.

Written and committed BEFORE experiments/w38d_build.log contains `w38d done`. Verify with
`git log` on this file against that log's mtime.

WHY THIS EXISTS. The rule is already fixed in experiments/w38d_prereg.txt:

    WANTED does not move to the new file unless the corrected object clears 0.9701440
    (leader w36_ad199stdcorr 0.9701400060 + the ~4e-6 rebuild reproducibility floor).

but applying it was left to whichever run wakes up at 00:00 UTC, from a blank context, with
ten submissions to fire and the queue head at stake. That run has to find the number, find
the bar, and not talk itself into a promotion on a near-miss. w36 §2 already records one
sign gate that fired and then disagreed with the workspace's own standing criterion; the
only reason it was not resolved in favour of the preferred answer is that the rule was
fixed first. This script fixes the APPLICATION as well, so the 00:00 run reads a verdict
rather than re-deriving one.

    .venv/bin/python experiments/w40b_promote.py          # verdict only, writes nothing
    .venv/bin/python experiments/w40b_promote.py --go     # apply, only if PROMOTE

It refuses to promote on anything it cannot check: no `w38d done`, no corrected object, a
submission artefact that is not row/id-identical to sample_submission.csv, or a CV at or
below the bar. A near-miss is NOT a promotion, and the margin is printed either way.
"""
import json, os, sys, hashlib
import numpy as np, pandas as pd

W    = "/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8/"
BAR  = 0.9701440          # w38d_prereg.txt, fixed before the build existed
LEADER_CV   = 0.9701400059625017
LEADER_FILE = "w36_ad199stdcorr.csv"
NEW_TAG     = "w38_ad202stdcorr"
GO = "--go" in sys.argv

def bail(msg):
    print(f"\n⛔ NOT PROMOTED — {msg}")
    print(f"   {LEADER_FILE} keeps the head of the queue and keeps its WANTED slot.")
    sys.exit(0)

# ---- 1. the build must have finished, on its own say-so ------------------------------
log = W + "experiments/w38d_build.log"
txt = open(log).read() if os.path.exists(log) else ""
print(f"w38d_build.log: {len(txt)} bytes, 'w38d done' present: {'w38d done' in txt}")
if "w38d done" not in txt:
    bail("the ARM 202 build has not printed `w38d done`. Nothing to price yet.")

# ---- 2. the corrected object, and its SHIPPED arm (never the argmax) ------------------
jp = W + f"experiments/w21a_{NEW_TAG}.json"
if not os.path.exists(jp):
    bail(f"the w21a correction step left no {os.path.basename(jp)} — the build did not finish it.")
d = json.load(open(jp))
shipped = d["shipped"]
cv = float(d["combos"][shipped]["cv"])
argmax = d.get("cv_argmax_combo")
print(f"\nARM 202  shipped arm : {shipped}")
print(f"         shipped CV  : {cv:.10f}")
print(f"         argmax arm  : {argmax}  (NOT used — w34/w36 ship the pre-registration)")
print(f"         h3 base AUC : {d['base_auc']:.10f}")
print(f"\nbar (w38d_prereg) : {BAR:.7f}")
print(f"leader ARM 199    : {LEADER_CV:.10f}")
print(f"delta vs leader   : {(cv-LEADER_CV)*1e6:+.2f}e-6   "
      f"(registered expectation was +2 to +12e-6 on the h3 base)")
print(f"margin vs bar     : {(cv-BAR)*1e6:+.2f}e-6")

if cv <= BAR:
    bail(f"CV {cv:.10f} does not clear the bar {BAR:.7f} "
         f"({(cv-BAR)*1e6:+.2f}e-6). This is the registered outcome for a null or a "
         f"sub-floor gain, and a near-miss is not a promotion.")

# ---- 3. the artefact has to be submittable before it can head the queue ---------------
sub = W + f"submissions/{NEW_TAG}.csv"
if not os.path.exists(sub):
    bail(f"CV clears the bar but submissions/{NEW_TAG}.csv does not exist.")
s  = pd.read_csv(sub)
ss = pd.read_csv(W + "data/sample_submission.csv")
col = [c for c in s.columns if c != "id"][0]
checks = {
    "rows == 296302":        len(s) == 296302,
    "id order == sample":    len(s) == len(ss) and bool((s.id.values == ss.id.values).all()),
    "no NaN":                bool(s[col].notna().all()),
    "finite":                bool(np.isfinite(s[col].values).all()),
    "non-degenerate":        float(s[col].std()) > 0,
}
print("\nartefact audit (w39a rules):")
for k, v in checks.items():
    print(f"   {'PASS' if v else 'FAIL'}  {k}")
if not all(checks.values()):
    bail("the artefact failed the w39a audit. A file that cannot be scored cannot lead the day.")
print(f"   md5 {hashlib.md5(open(sub,'rb').read()).hexdigest()}")

# ---- 4. verdict ----------------------------------------------------------------------
print(f"""
✅ PROMOTE. ARM 202 clears the pre-registered bar by {(cv-BAR)*1e6:+.2f}e-6.

Two edits follow, and the prereg requires them in ONE commit:
  a) check_selection.py  WANTED: {LEADER_FILE} -> {NEW_TAG}.csv
     (the second WANTED slot, w23_ad187stdcorr.csv, is NOT touched — it is the already-sent
      diversification pick and the prereg says nothing about it.)
  b) w37d_order.py       ORDER: insert {NEW_TAG}.csv at slot 1, ARM 199 falls to slot 2,
     and the 08-21 day becomes 202 -> 199 -> the eight already ordered behind them. The
     13th file falls off the ordered list and back to the pricer.

⚠ ARM 202 IS NOT SELECTABLE UNTIL IT IS SENT. Promotion is worthless without the drain.
""")
if not GO:
    print("DRY RUN. Re-run with --go to apply, then commit both files together.")
    sys.exit(0)

# ---- 5. apply ------------------------------------------------------------------------
p = W + "experiments/check_selection.py"
s_cs = open(p).read()
old = f'WANTED = {{"{LEADER_FILE}", "w23_ad187stdcorr.csv"}}'
assert old in s_cs, "WANTED line not found verbatim — refusing to guess at it."
s_cs = s_cs.replace(old, f'WANTED = {{"{NEW_TAG}.csv", "w23_ad187stdcorr.csv"}}\n'
                         f'# w40b: moved off {LEADER_FILE} on the w38d_prereg rule — ARM 202\n'
                         f'# shipped CV {cv:.10f}, clearing the 0.9701440 bar by {(cv-BAR)*1e6:+.2f}e-6.')
open(p, "w").write(s_cs)
print(f"check_selection.py: WANTED -> {NEW_TAG}.csv")

p = W + "experiments/w37d_order.py"
s_or = open(p).read()
anchor = ' ("w36_ad199stdcorr.csv",'
assert anchor in s_or, "ORDER anchor not found — refusing to guess at it."
ins = (f' ("{NEW_TAG}.csv",       "ARM 202: the 199 pack plus the three zhukovoleksiy members.\\n'
       f'                              "Shipped CV {cv:.10f}, clearing the w38d_prereg bar of\\n'
       f'                              "0.9701440 by {(cv-BAR)*1e6:+.2f}e-6. New CV leader and\\n'
       f'                              "check_selection.WANTED, so unsent = NOT SELECTABLE."),\\n')
s_or = s_or.replace(anchor, ins.replace("\\n", "\n") + anchor, 1)
open(p, "w").write(s_or)
print("w37d_order.py: ARM 202 inserted at slot 1")
print("\nNOW RUN:  .venv/bin/python experiments/w37d_order.py   then re-check w26g_send.py --n 10")
