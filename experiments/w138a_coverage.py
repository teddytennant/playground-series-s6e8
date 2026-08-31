#!/usr/bin/env python
"""w138 (2026-08-31, slot 7, ANGLE: consolidation) -- COVERAGE AUDIT OF THE PICK.

The pick is defended as "CV rank 1 of 169" (w137 s4).  The denominator is
`w48a_cv_recomputed.csv`, which is built from a REGISTRY: four hand-maintained
CSV/JSON files list a stem, and only then is `oof_<stem>.npy` looked up.  A file
that was submitted but never registered is absent from the table, and therefore
cannot lose to the pick -- it is never compared at all.

This scores every submitted stem that has an OOF vector on disk, registry or
not, on the same y and the same metric w48a uses, and asks one question:

    does anything beat the pick's true_cv?

Prints the full ranking so the answer is checkable, not asserted.
"""
import os, sys, json, csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET                       # noqa: E402

PICK1, PICK2 = "w36_ad199stdcorr", "w23_ad187stdcorr"

subs = json.load(open("/tmp/subs_all.json"))
sub_stems = sorted({r["file"][:-4] if r["file"].endswith(".csv") else r["file"] for r in subs})
best_pub = {}
for r in subs:
    st = r["file"][:-4] if r["file"].endswith(".csv") else r["file"]
    if r["public"] is not None:
        best_pub[st] = max(best_pub.get(st, 0.0), float(r["public"]))

table = {}
for r in csv.DictReader(open(os.path.join(HERE, "w48a_cv_recomputed.csv"))):
    if r["true_cv"] and r["true_cv"] != "":
        try: table[r["stem"]] = float(r["true_cv"])
        except ValueError: pass

y = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[TARGET])[TARGET].values

def score(stem):
    f = os.path.join(SUB, f"oof_{stem}.npy")
    if not os.path.exists(f): return None, "NO OOF ON DISK"
    v = np.load(f).ravel()
    if len(v) != len(y): return None, f"LENGTH {len(v)} != {len(y)}"
    if np.isnan(v).any(): return None, "NaN IN OOF"
    return float(roc_auc_score(y, v)), "ok"

rows = []
for st in sub_stems:
    cv, why = score(st)
    rows.append(dict(stem=st, cv=cv, why=why, in_table=st in table,
                     table_cv=table.get(st), public=best_pub.get(st)))

pick_cv = table[PICK1]
print(f"submitted stems           : {len(sub_stems)}")
print(f"  in w48a CV table        : {sum(r['in_table'] for r in rows)}")
print(f"  NOT in table            : {sum(not r['in_table'] for r in rows)}")
scored = [r for r in rows if r["cv"] is not None]
print(f"  scorable from OOF here  : {len(scored)}")
newly = [r for r in scored if not r["in_table"]]
print(f"  NEWLY scorable (untabled but OOF on disk): {len(newly)}")
print(f"\npick1 {PICK1} table_cv = {pick_cv:.10f}")
print(f"pick2 {PICK2} table_cv = {table[PICK2]:.10f}\n")

# consistency: recomputed vs table, for stems in both
both = [r for r in scored if r["in_table"]]
d = np.array([r["cv"] - r["table_cv"] for r in both])
print(f"recompute vs table, {len(both)} stems: max|diff| = {np.abs(d).max()*1e6:.4f}e-6")

beat = sorted([r for r in scored if r["cv"] > pick_cv], key=lambda r: -r["cv"])
print(f"\n=== stems BEATING the pick on recomputed CV: {len(beat)} ===")
for r in beat:
    tag = "IN TABLE" if r["in_table"] else "*** UNTABLED ***"
    print(f"  {r['cv']:.10f}  (+{(r['cv']-pick_cv)*1e6:8.4f}e-6)  pub={r['public']}  {r['stem']:34s} {tag}")
if not beat:
    print("  NONE. The pick is the CV argmax over every submitted stem with an OOF on disk.")

print(f"\n=== top 12 untabled stems by recomputed CV ===")
for r in sorted(newly, key=lambda r: -r["cv"])[:12]:
    print(f"  {r['cv']:.10f}  ({(r['cv']-pick_cv)*1e6:+9.4f}e-6)  pub={r['public']}  {r['stem']}")

nooof = [r for r in rows if r["cv"] is None]
print(f"\n=== submitted stems with NO usable OOF: {len(nooof)} ===")
for r in sorted(nooof, key=lambda r: -(r["public"] or 0))[:15]:
    print(f"  pub={r['public']}  {r['stem']:34s} {r['why']}  in_table={r['in_table']}")

json.dump({"n_submitted": len(sub_stems), "n_in_table": sum(r["in_table"] for r in rows),
           "n_scored": len(scored), "n_newly_scorable": len(newly),
           "pick_cv": pick_cv, "n_beating_pick": len(beat),
           "beating": [{"stem": r["stem"], "cv": r["cv"], "in_table": r["in_table"],
                        "public": r["public"]} for r in beat],
           "max_table_recompute_diff_e6": float(np.abs(d).max()*1e6)},
          open(os.path.join(HERE, "w138a_coverage.json"), "w"), indent=1)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "w138a_coverage.csv"), index=False)
