"""w123a -- ANGLE INDEX row 3 re-verified at the ARTEFACT level, not quoted (2026-08-30).

The handed ANGLE was *"CatBoost: it usually handles categoricals better than the others on
survey-style data. Tune and compare on identical folds."* Genus `CatBoost` -> row 3, closed,
price **5.9e-6/member**.

Row 3 is one of only TWO rows (with row 1) never marked `artefacts verified`, and it is the
row with the worst record: w105 found its closure citing two members that were never built.
w106's standard is that a row's closure is not its evidence. So this reads the evidence.

WHAT ROW 3's CLOSURE CLAIMS, and what each claim is checked against here:

  A  "the `rest` group -- 35 ordinary XGB/LGBM/CatBoost members -- is worth +0.000206 +/-
     0.000011 in total, i.e. 5.9e-6 each".
     -> reconstruct the group from the arrays on disk and re-run the paired 50/50 test that
        produced it (`member_value2.py`'s procedure: 3 splits, C=1.0, hybrid transform).
  B  the same sentence's LABEL: "35 ordinary XGB/LGBM/CatBoost members".
     -> classify all 35 by family. `rest` is defined by RESIDUAL ("everything not decorr, not
        lookup2, not bei"), which is not a family predicate, so the label is a claim about the
        membership and can be false while the arithmetic is right.
  C  "the CatBoost function class is already in the pack -- three ways: cat_lat, cat_native,
     cat_raw", and w105's correction that `cat_native_ctr2` / `cat_natlat` were never built.
     -> ls, and a disk-wide search for the two unbuilt stems.
  D  "the one place foreign CatBoosts paid (+10.3e-6 each, adarsh1077) was a property of the
     PIPELINE, not the function class".
     -> that number is the only PURE-CatBoost measurement in the record, and it is nearly
        double the price row 3 actually sells. Check it is on record and compare the two.
  E  arithmetic: 0.000206 / 35 -> 5.9e-6, and 5.9e-6 is "~12%" of the 5e-5 noise floor.

This is a READ. It fits a meta-combiner to re-measure a published paired delta -- it builds no
member, enrols nothing, sweeps no hyperparameter, and ships no file. Re-measuring a price is
not re-opening it (w122 6).

    .venv/bin/python experiments/w123a_row3.py            # rc 0 = every claim reproduces
    .venv/bin/python experiments/w123a_row3.py --quick    # skip the paired fit (C1/B/C/D/E only)

Deterministic: fixed splits (StratifiedShuffleSplit random_state=rep), no API call, no
member fit, no fold rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA, TARGET, load_raw           # noqa: E402
from stack import DEFAULT_DROP, load_members, transform  # noqa: E402

EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
OOFDIR = os.path.join(ROOT, "oof")
DECORR_MAX = 0.97          # member_value2's own cut. NOT settable from here.
REPS = 3                   # the published run used 3 splits
CVAL = 1.0
TRANSFORM = "hybrid"

# Published, frozen as literals so this is a COMPARISON and not a recomputation that agrees
# with itself. Sources named per claim in the closure quoted above.
PUBLISHED = {
    "rest_n": 35,                 # claim A
    "rest_delta": 0.000206,       # claim A
    "rest_sd": 0.000011,          # claim A
    "rest_per_member": 5.9e-6,    # claim A / the row 3 price cell
    "base_n": 86,                 # the base the published delta was measured against
    "noise_floor": 5e-5,          # claim E
    "floor_pct": 0.12,            # claim E, "~12% of the 5e-5 noise floor"
    "foreign_cat_per_member": 10.3e-6,   # claim D, w20d `cat` group, 4 members
    "in_pack": ["cat_lat", "cat_native", "cat_raw"],       # claim C
    "never_built": ["cat_native_ctr2", "cat_natlat"],      # claim C, w105's correction
}

# Claim B. Family is read off the member NAME, which is the only family evidence these
# foreign arrays carry -- boltuzamaki/mohankrishnathalla ship no manifest of model types.
# `_` boundaries only, so `cat` does not match `lattice` (w121's identifier-boundary rule).
FAMILY = {
    "catboost": ("cat",),
    "xgboost": ("xgb",),
    "lightgbm": ("lgb",),
    "sklearn-gbdt": ("histgb",),
    "neural": ("lookup", "tabm", "realmlp", "mlp", "ft", "dcnv2", "gandalf"),
}


def classify(name):
    """Family tags present in a member name, matched on `_`-delimited tokens."""
    toks = set(name.split("_"))
    hits = [fam for fam, keys in FAMILY.items() if toks & set(keys)]
    return hits


def paired(Z, y, idx, variants, reps=REPS):
    rows = []
    for rep in range(reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        r = {}
        for k, mem in variants.items():
            cols = [idx[m] for m in mem]
            m = LogisticRegression(max_iter=3000, C=CVAL).fit(Z[np.ix_(iA, cols)], y[iA])
            r[k] = roc_auc_score(y[iB], m.predict_proba(Z[np.ix_(iB, cols)])[:, 1])
        rows.append(r)
        print("    split %d: " % rep + "  ".join(f"{k}={v:.6f}" for k, v in r.items()),
              flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the paired re-measurement (claim A)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w123a -- ANGLE INDEX row 3 (CatBoost) against its artefacts\n")

    # ---------------------------------------------------------------- claim C, cheap first
    print("[C] the pack-membership citation, and w105's correction")
    on_disk = sorted(f[4:-4] for f in os.listdir(OOFDIR)
                     if f.startswith("oof_cat_") and f.endswith(".npy"))
    print(f"    oof/oof_cat_*: {on_disk}")
    if on_disk != sorted(PUBLISHED["in_pack"]):
        fail(f"the three-ways citation no longer resolves: {on_disk} != "
             f"{sorted(PUBLISHED['in_pack'])}")
    else:
        print(f"    OK all three of {PUBLISHED['in_pack']} are on disk, and nothing else is")
    for stem in PUBLISHED["never_built"]:
        r = subprocess.run(["find", ROOT, "-name", f"*{stem}*"],
                           capture_output=True, text=True)
        hits = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if hits:
            fail(f"w105 says `{stem}` was never built, but {len(hits)} path(s) now match: "
                 f"{hits[:3]}")
        else:
            print(f"    OK `{stem}` still resolves to nothing anywhere on disk")
    out["oof_cat_on_disk"] = on_disk

    # ---------------------------------------------------------------- reconstruct the groups
    print("\n[A/B] reconstructing member_value2's groups from the arrays on disk")
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, T = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
    idx = {n: i for i, n in enumerate(names)}
    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    new = [n for n in names if n in vet.index]
    base = [n for n in names if n not in vet.index]
    bei = [n for n in new if n.startswith("bei_")]
    lookup2 = [n for n in new if n.startswith("bolt_lookup_v2")]
    decorr = [n for n in new
              if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
    rest = [n for n in new if n not in lookup2 and n not in decorr and n not in bei]
    print(f"    base {len(base)} | new {len(new)} = decorr {len(decorr)} + "
          f"lookup2 {len(lookup2)} + rest {len(rest)} + bei {len(bei)}")
    out["group_sizes"] = {"base": len(base), "new": len(new), "decorr": len(decorr),
                          "lookup2": len(lookup2), "rest": len(rest), "bei": len(bei)}

    if len(rest) != PUBLISHED["rest_n"]:
        fail(f"the `rest` group reconstructs at n={len(rest)}, published n="
             f"{PUBLISHED['rest_n']}")
    else:
        print(f"    OK `rest` reconstructs at exactly n={PUBLISHED['rest_n']}")
    if len(base) != PUBLISHED["base_n"]:
        note(f"the BASE has grown: published base{PUBLISHED['base_n']}, on disk today "
             f"base{len(base)}. The re-measurement below is against a DIFFERENT POOL, so a "
             f"gap against +{PUBLISHED['rest_delta']:.6f} is pool drift, not a contradiction.")

    # ---------------------------------------------------------------- claim B, the label
    print("\n[B] the LABEL: '35 ordinary XGB/LGBM/CatBoost members'")
    fam = {n: classify(n) for n in rest}
    cat = sorted(n for n in rest if "catboost" in fam[n])
    neural = sorted(n for n in rest if "neural" in fam[n])
    gbdt = sorted(n for n in rest
                  if {"catboost", "xgboost", "lightgbm", "sklearn-gbdt"} & set(fam[n]))
    unknown = sorted(n for n in rest if not fam[n])
    print(f"    CatBoost-named {len(cat):2d}: {' '.join(cat)}")
    print(f"    XGB-named      {len([n for n in rest if 'xgboost' in fam[n]]):2d}")
    print(f"    LGBM-named     {len([n for n in rest if 'lightgbm' in fam[n]]):2d}")
    print(f"    NEURAL-named   {len(neural):2d}: {' '.join(neural)}")
    print(f"    unclassifiable {len(unknown):2d}: {' '.join(unknown)}")
    out["rest_members"] = rest
    out["rest_families"] = {"catboost": cat, "neural": neural, "gbdt": gbdt,
                            "unclassifiable": unknown}
    if neural:
        fail(f"the label says 'ordinary XGB/LGBM/CatBoost' but {len(neural)} member(s) of "
             f"`rest` are neural nets by name: {neural}. `rest` is a RESIDUAL "
             f"('not decorr, not lookup2, not bei'), and maxcorr >= {DECORR_MAX} is not a "
             f"family test -- these four correlate with the pack, so they fell through.")
    if cat:
        share = len(cat) / len(rest)
        print(f"    -> CatBoost is {len(cat)}/{len(rest)} = {share:.0%} of the group whose "
              f"per-member average row 3 sells as THE CATBOOST PRICE")
        out["cat_share"] = share

    # ---------------------------------------------------------------- claim E, arithmetic
    print("\n[E] arithmetic")
    per = PUBLISHED["rest_delta"] / PUBLISHED["rest_n"]
    print(f"    {PUBLISHED['rest_delta']:.6f} / {PUBLISHED['rest_n']} = {per*1e6:.3f}e-6 "
          f"(published {PUBLISHED['rest_per_member']*1e6:.1f}e-6)")
    if abs(per - PUBLISHED["rest_per_member"]) > 0.05e-6:
        fail(f"5.9e-6 does not multiply out: {per*1e6:.3f}e-6")
    pct = PUBLISHED["rest_per_member"] / PUBLISHED["noise_floor"]
    print(f"    {PUBLISHED['rest_per_member']*1e6:.1f}e-6 / {PUBLISHED['noise_floor']*1e6:.0f}e-6"
          f" = {pct:.1%} of the noise floor (published ~{PUBLISHED['floor_pct']:.0%})")
    if abs(pct - PUBLISHED["floor_pct"]) > 0.01:
        fail(f"the '~12% of the noise floor' gloss is off: {pct:.1%}")
    out["arith"] = {"per_member": per, "floor_pct": pct}

    # ---------------------------------------------------------------- claim D, the other price
    print("\n[D] the OTHER CatBoost price in the same closure")
    hits = subprocess.run(["grep", "-c", "10.3e-6", os.path.join(ROOT, "RESEARCH.md")],
                          capture_output=True, text=True).stdout.strip()
    print(f"    RESEARCH.md carries '10.3e-6' on {hits} line(s) -- w20d's `cat` group, 4 PURE "
          f"CatBoost arrays from adarsh1077")
    ratio = PUBLISHED["foreign_cat_per_member"] / PUBLISHED["rest_per_member"]
    print(f"    {PUBLISHED['foreign_cat_per_member']*1e6:.1f}e-6 vs the row 3 cell "
          f"{PUBLISHED['rest_per_member']*1e6:.1f}e-6 = {ratio:.2f}x")
    if hits == "0":
        fail("the +10.3e-6 foreign-CatBoost measurement is no longer in RESEARCH.md")
    else:
        note(f"the only PURE-CatBoost group ever measured here reads "
             f"{PUBLISHED['foreign_cat_per_member']*1e6:.1f}e-6/member, {ratio:.2f}x the "
             f"price row 3 publishes -- and it sits two bullets BELOW it in the same closure.")
    out["foreign_ratio"] = ratio

    # ---------------------------------------------------------------- claim A, the fit
    if a.quick:
        print("\n[A] --quick: paired re-measurement skipped")
    else:
        print(f"\n[A] paired 50/50, {REPS} splits, C={CVAL}, transform={TRANSFORM} "
              f"(member_value2's own procedure)")
        Z, _ = transform(O, T, TRANSFORM)
        rest_nocat = [n for n in rest if n not in cat]
        variants = {
            "base": base,
            "+rest": base + rest,
            "+cat_only": base + cat,
            "+rest_nocat": base + rest_nocat,
        }
        rob = paired(Z, y, idx, variants)
        print(f"\n    PAIRED DELTA vs base{len(base)} (same rows; split noise cancels)")
        res = {}
        for c in rob.columns:
            if c == "base":
                continue
            d = rob[c] - rob["base"]
            n = len(variants[c]) - len(base)
            sign = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
            res[c] = {"n": n, "delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                      "per_member": float(d.mean() / n), "sign": sign}
            print(f"      {c:<14s} n={n:2d}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  "
                  f"{d.mean()/n*1e6:+7.2f}e-6/member  [{sign}]")
        out["paired"] = res

        r = res["+rest"]
        if r["sign"] != "consistent":
            fail("the `rest` delta no longer holds its sign across splits")
        gap = r["delta"] - PUBLISHED["rest_delta"]
        print(f"\n    `rest` re-measures {r['delta']:+.6f} against published "
              f"+{PUBLISHED['rest_delta']:.6f} (gap {gap:+.6f}) on a base that has grown "
              f"{PUBLISHED['base_n']} -> {len(base)}")
        if r["per_member"] <= 0:
            fail(f"the `rest` group no longer pays: {r['per_member']*1e6:+.2f}e-6/member")
        c8, cn = res["+cat_only"], res["+rest_nocat"]
        print(f"\n    -> the CatBoost-only subgroup of `rest`: n={c8['n']}, "
              f"{c8['per_member']*1e6:+.2f}e-6/member [{c8['sign']}]")
        print(f"    -> `rest` with the CatBoosts removed: n={cn['n']}, "
              f"{cn['per_member']*1e6:+.2f}e-6/member [{cn['sign']}]")
        note(f"row 3's price cell is a per-member average over a group that is "
             f"{len(cat)}/{len(rest)} CatBoost; measured alone the CatBoost members read "
             f"{c8['per_member']*1e6:+.2f}e-6/member here.")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    for m in notes:
        print("  note " + m)
    print(f"FAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    out["failures"] = fails
    out["notes"] = notes
    with open(os.path.join(HERE, "w123a_row3.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
