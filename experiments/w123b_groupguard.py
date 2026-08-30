"""w123b -- STANDING CHECK #52. A GROUP'S LABEL MUST NAME THE PREDICATE THAT BUILT IT.
(w123, 2026-08-30)

WHAT WENT WRONG. RESEARCH.md prices the `rest` group at **5.9e-6/member** and describes it,
in four places, as *"35 ordinary XGB/LGBM/CatBoost members"*. ANGLE INDEX row 3 (`CatBoost`)
sells that per-member average as THE CATBOOST PRICE. But `rest` is not a family group at all:
`member_value2.py` builds it as a RESIDUAL --

    rest = [n for n in new if n not in lookup2 and n not in decorr and n not in bei]

-- and `decorr` is cut on `maxcorr < 0.97`, which is a correlation test, not a family test.
Reconstructed from the arrays on disk (w123a_row3.py), the 35 hold **8** CatBoost-named
members and **4** neural nets: `bolt_lookup_v1` (the author's FIRST Lookup-Transformer, whose
six v2 seeds sit in the group explicitly labelled as transformers), `bolt_realmlp_lattice`,
`bolt_tabm_missing`, `bolt_tabm_rank1`. They are in `rest` because they correlate with the
pack, not because anyone thought they were GBDTs.

🎯 THIS IS THE SAME DEFECT AS w120 §4 (`priority` naming an input), w121 §3 (a prose cause
beside a derived count) and w122 §2 (`slot` printed for `tier`), in a FOURTH place: a label
that describes what the run EXPECTED the predicate to select rather than what it selects. The
arithmetic was never wrong -- 0.000206/35 really is 5.9e-6 -- so no numeric check reaches it.

WHAT THIS GUARD ENFORCES.

  C1  the `rest` group still reconstructs from the arrays on disk, member-for-member, against
      the frozen list below. If the pool moves, the published price stops describing a
      reconstructible group and the disclosure below is about the wrong 35 members.
  C2  the composition is what the disclosure claims: 8 CatBoost-named, 4 neural-named.
  C3  DISCLOSURE. Every occurrence of the family label in RESEARCH.md must carry the word
      RESIDUAL within +/-2 lines. This is the check that would have caught the defect, and it
      is a check on a WORD, which is the class of defect the last four runs kept finding.
  C4  ANGLE INDEX row 3's price cell must name BOTH prices. 5.9e-6 is a mixed-group average;
      the only PURE-CatBoost group ever measured here (w20d's `cat`, adarsh1077, n=4) reads
      +10.3e-6/member, 1.75x, and it sat two bullets below the row's own citation.
  C5  --control: the same C3 predicate over the FROZEN pre-fix lines, which must FIRE. If it
      does not, the guard reports INERT rather than green -- a fix with no alternative
      behaviour to select from is not a fix (w121 §5, w122 §5).

⚠ JOURNAL.md is append-only and is NOT checked: its four occurrences are history and correcting
them would be a rewrite. Only RESEARCH.md, the live document, is in scope.

    .venv/bin/python experiments/w123b_groupguard.py            # rc 0 = clean
    .venv/bin/python experiments/w123b_groupguard.py --control  # exits 0, having fired

Deterministic: reads filenames and two documents. No fit, no API call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA          # noqa: E402
from stack import LIB, DEFAULT_DROP  # noqa: E402

RESEARCH = os.path.join(ROOT, "RESEARCH.md")
OOFDIR = os.path.join(ROOT, "oof")
EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
NTR, NTE = 691369, 296302        # the frozen train/test row counts
DECORR_MAX = 0.97                # member_value2's own cut. NOT settable from here.

LABEL = "ordinary XGB/LGBM/CatBoost"
DISCLOSURE = "RESIDUAL"
WINDOW = 2                       # lines either side of the label

# The group as it reconstructs on 2026-08-30, frozen so C1 is a COMPARISON.
REST = [
    "bolt_cat_cpu5", "bolt_cat_dual_seed81", "bolt_cat_dual_view", "bolt_cat_nested_te",
    "bolt_cat_pair_evidence", "bolt_cat_unique", "bolt_foldsafe_te_cat",
    "bolt_foldsafe_te_multi", "bolt_foldsafe_te_wide", "bolt_foldsafe_te_xgb",
    "bolt_foldsafe_te_xgb_10f", "bolt_histgb_5fold", "bolt_lgb_driver_recon",
    "bolt_lgb_missing_global", "bolt_lgb_pair_lattice", "bolt_lgb_raw_d4", "bolt_lgb_raw_d6",
    "bolt_lgb_te_5fold", "bolt_lookup_v1", "bolt_realmlp_lattice", "bolt_repr_lgb_global",
    "bolt_tabm_missing", "bolt_tabm_rank1", "bolt_xgb_d7_alt1", "bolt_xgb_d7_alt2",
    "bolt_xgb_dd_d4", "bolt_xgb_dd_d5", "bolt_xgb_dd_d6", "bolt_xgb_hpo_d7",
    "bolt_xgb_raw_bag", "bolt_xgb_te_4fold", "bolt_xgb_te_5fold",
    "mkt_cat", "mkt_lgb", "mkt_xgb",
]
CAT = ["bolt_cat_cpu5", "bolt_cat_dual_seed81", "bolt_cat_dual_view", "bolt_cat_nested_te",
       "bolt_cat_pair_evidence", "bolt_cat_unique", "bolt_foldsafe_te_cat", "mkt_cat"]
NEURAL = ["bolt_lookup_v1", "bolt_realmlp_lattice", "bolt_tabm_missing", "bolt_tabm_rank1"]

# The pre-fix lines, frozen verbatim for the --control. Each is a real RESEARCH.md line as it
# stood at 2026-08-30 13:0xZ, with the two lines of context the window would have seen.
PREFIX_CONTROL = [
    ["group — 35 ordinary XGB/LGBM/CatBoost members worth +0.000206 ± 0.000011 in total, i.e.",
     "**5.9e-6 each** — which does not depend on those two files existing. And the unbuilt arms' own"],
    ["* the `rest` group — **35** ordinary XGB/LGBM/CatBoost members — is worth +0.000206 ± 0.000011",
     "  **in total**, i.e. **5.9e-6 each**;"],
    ["hyperparameters.** This resolves the slot-3 puzzle: 35 ordinary XGB/LGBM/CatBoost members",
     "from `boltuzamaki` were the largest single share of that day's +0.000340, while GBDT"],
    ["| `rest` — ordinary XGB/LGBM/CatBoost | 35 | +0.000206 ± 0.000011 | 5.9e-6 |",
     "| **all 63 together** | 63 | **+0.000330 ± 0.000011** | 5.2e-6 |"],
]

ROW3_PRICES = ["5.9e-6", "10.3e-6"]
ANGLE_HEAD = "# 📇 THE ANGLE INDEX"


def reconstruct():
    """member_value2's group cut, by filename and vetting sheet only. No arrays loaded."""
    drop, names = set(DEFAULT_DROP), []
    for d in [os.path.join(LIB, "oof"), OOFDIR, EXT, EXT2]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.startswith("oof_") or not fn.endswith(".npy"):
                continue
            nm = fn[4:-4]
            if nm in drop or nm in names:
                continue
            tp = os.path.join(d, f"test_{nm}.npy")
            if not os.path.exists(tp):
                continue
            if os.path.getsize(os.path.join(d, fn)) != NTR * 8 + 128:
                continue
            if os.path.getsize(tp) != NTE * 8 + 128:
                continue
            names.append(nm)
    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    new = [n for n in names if n in vet.index]
    bei = [n for n in new if n.startswith("bei_")]
    lookup2 = [n for n in new if n.startswith("bolt_lookup_v2")]
    decorr = [n for n in new
              if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
    return [n for n in new if n not in lookup2 and n not in decorr and n not in bei]


def undisclosed(lines):
    """Occurrences of LABEL with no DISCLOSURE within +/-WINDOW lines. The C3 predicate."""
    bad = []
    for i, ln in enumerate(lines):
        if LABEL not in ln:
            continue
        ctx = lines[max(0, i - WINDOW): i + WINDOW + 1]
        if not any(DISCLOSURE in c for c in ctx):
            bad.append((i + 1, ln.strip()[:90]))
    return bad


def row3_cell(txt):
    k = txt.find(ANGLE_HEAD)
    if k < 0:
        return None
    for ln in txt[k:].splitlines():
        if ln.startswith("| 3 |"):
            return ln
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run the C3 predicate over the frozen PRE-FIX lines; it must fire")
    a = ap.parse_args()

    if a.control:
        print("w123b --control: the C3 predicate against the frozen PRE-FIX RESEARCH.md lines\n")
        fired = 0
        for block in PREFIX_CONTROL:
            bad = undisclosed(block)
            fired += len(bad)
            print(f"  {'FIRES ' if bad else 'silent'} {block[0].strip()[:82]}")
        shipped = undisclosed(open(RESEARCH, encoding="utf-8").read().splitlines())
        print(f"\n  pre-fix text: {fired} undisclosed occurrence(s)")
        print(f"  shipped text: {len(shipped)} undisclosed occurrence(s)")
        if fired == 0:
            print("\n  ** INERT ** the pre-fix predicate finds nothing, so the guard has no "
                  "alternative behaviour to select from and is not evidence of a fix.")
        elif fired == len(shipped):
            print("\n  ** INERT ** both predicates agree; the fix changed nothing observable.")
        else:
            print(f"\n  ✅ the pre-fix label was undisclosed in {fired} place(s) and the "
                  f"shipped one in {len(shipped)}.")
        return 0

    fails, out = [], {}

    def fail(m):
        fails.append(m)
        print("  FAIL " + m)

    print("w123b -- a group's label must name the predicate that built it\n")

    # C1 ------------------------------------------------------------------
    rest = reconstruct()
    out["rest_n"] = len(rest)
    if rest != REST:
        miss, extra = sorted(set(REST) - set(rest)), sorted(set(rest) - set(REST))
        fail(f"C1 the `rest` group no longer reconstructs to the frozen 35: n={len(rest)}, "
             f"missing={miss[:5]}, unexpected={extra[:5]}")
    else:
        print(f"  C1 OK `rest` reconstructs member-for-member at n={len(rest)}")

    # C2 ------------------------------------------------------------------
    cat = [n for n in rest if "cat" in n.split("_")]
    neu = [n for n in rest
           if {"lookup", "tabm", "realmlp", "mlp"} & set(n.split("_"))]
    if sorted(cat) != sorted(CAT) or sorted(neu) != sorted(NEURAL):
        fail(f"C2 composition drifted: catboost={sorted(cat)}, neural={sorted(neu)}")
    else:
        print(f"  C2 OK composition holds: {len(cat)} CatBoost-named, {len(neu)} neural-named "
              f"({len(cat)}/{len(rest)} = {len(cat)/len(rest):.0%} CatBoost)")
    out["cat"], out["neural"] = sorted(cat), sorted(neu)

    # C3 ------------------------------------------------------------------
    txt = open(RESEARCH, encoding="utf-8").read()
    lines = txt.splitlines()
    total = sum(1 for ln in lines if LABEL in ln)
    bad = undisclosed(lines)
    out["label_occurrences"], out["undisclosed"] = total, bad
    if bad:
        for lineno, snip in bad:
            fail(f"C3 RESEARCH.md:{lineno} calls the group '{LABEL}' with no '{DISCLOSURE}' "
                 f"within {WINDOW} lines. It is a residual, and 4 of the 35 are neural nets. "
                 f">> {snip}")
    else:
        print(f"  C3 OK all {total} occurrence(s) of the family label disclose the residual")

    # C4 ------------------------------------------------------------------
    cell = row3_cell(txt)
    if cell is None:
        fail("C4 ANGLE INDEX row 3 not found")
    else:
        missing = [p for p in ROW3_PRICES if p not in cell]
        if missing:
            fail(f"C4 ANGLE INDEX row 3 sells one price and omits {missing}. 5.9e-6 is a "
                 f"mixed-group average; the only pure-CatBoost group measured here is "
                 f"+10.3e-6/member.")
        else:
            print(f"  C4 OK row 3 names both {ROW3_PRICES}")
    out["row3_cell"] = cell

    print(f"\nFAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    if not fails:
        print("✅ CLEAN — the residual group is labelled as one, and row 3 sells both prices")
    out["failures"] = fails
    with open(os.path.join(HERE, "w123b_groupguard.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
