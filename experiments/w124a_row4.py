"""w124a -- ANGLE INDEX row 4 (XGBoost) read at the ARTEFACT level, and given the price it
has never had (2026-08-30).

The handed ANGLE was *"XGBoost: third leg of the ensemble, tuned on the same folds so the
blend weights mean something."* Genus `XGBoost` -> row 4, closed, price **+4e-7**.

w106 already verified row 4's artefacts once. This is NOT that check repeated. w123 found
that row 3's published price is a RESIDUAL group's average and that the CatBoost members
measured alone read 1.8x it. Row 4 has the mirror problem and it is worse:

  * row 4's **+4e-7 is a TUNING price** -- the marginal value of tuning a GBDT you already
    hold, derived as 3e-5 (solo tuning gain) x 1.4% (solo->stack pass-through).
  * row 3's **5.9e-6 / 10.04e-6 is an ENROLMENT price** -- the marginal value of ADDING a
    member to the pack.
  These are different quantities in the same column of the same table, and nothing says so.
  Read side by side they invite "CatBoost is 15x XGBoost", which is not a claim either
  measurement supports.

So this run measures the quantity row 4 is missing: the **enrolment price of the XGBoosts**,
on the same base104 pool, the same paired 50/50 procedure and the same three splits w123 used
for the CatBoosts -- which is exactly what the handed angle asks for ("the same folds so the
blend weights mean something"). CatBoost is re-measured in the same process as a reproduction
control: if `+cat_only` does not come back at ~+10.04e-6, the XGB number is not comparable to
anything and the run says so instead of publishing it.

CLAIMS CHECKED, each against an artefact rather than a quotation:

  R1  "the pack already holds `xgb_lat`, `xgb_latcat` (x3 seeds), `xgb_cat_lattice`,
      `xgb_raw_nan` and the `bolt_xgb_*` family"      -> ls, per name.
  R2  "`latr1_xgb` at 0.96780 is the best GBDT of any family here" (LGBM 0.96768 `lattri_lgbm`
      / `latmax_lgbm`, CatBoost 0.96718 `latwide_cat`)
      -> recompute OOF AUC for every array on disk and check the three family maxima.
  R3  the derivation "+3e-5 x 1.4% = ~4e-7", and "1.4%" itself from "+138e-6 solo -> +2e-6
      stack"                                          -> arithmetic.
  R4  NEW. enrolment price of the 12 XGBoosts of `rest`, measured alone, vs LGBM (8) and
      CatBoost (8) on identical folds.
  R5  NEW. `bolt_xgb_d7_alt1` == `bolt_xgb_d7_alt2` byte-identical (w109a_dupscan relation 1).
      Both are in the XGB subgroup, so its per-member DENOMINATOR is 11 distinct arrays, not
      12 names -- and the same duplicate sits inside the 35 that the published 5.9e-6 divides
      by. Verified on the arrays, and measured: `+xgb_dedup` drops the duplicate column.

This is a READ. It fits a meta-combiner to measure a price; it builds no member, enrols
nothing, sweeps no hyperparameter and ships no file. Re-measuring a price is not re-opening
it (w122 6).

    .venv/bin/python experiments/w124a_row4.py            # rc 0 = every claim reproduces
    .venv/bin/python experiments/w124a_row4.py --quick    # skip the paired fit (R1/R2/R3/R5)

Deterministic: fixed splits (StratifiedShuffleSplit random_state=rep), no API call, no member
fit, no fold rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
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
EXTRA_OOF = os.path.join(DATA, "oof", "oof")   # where the latr1_*/latwide_* baselines live
DECORR_MAX = 0.97          # member_value2's own cut. NOT settable from here.
REPS = 3                   # w123 used 3 splits; same number, same seeds, same procedure
CVAL = 1.0
TRANSFORM = "hybrid"

# Published, frozen as literals so this is a COMPARISON, not a recomputation that agrees with
# itself. Sources named per claim in the docstring above.
PUBLISHED = {
    # R1 -- the pack-membership citation in the row 4 closure
    "in_pack": ["xgb_lat", "xgb_latcat", "xgb_latcat_s17", "xgb_latcat_s23",
                "xgb_cat_lattice", "xgb_raw_nan"],
    # R2 -- the "best single models by family" table
    "best_by_family": {
        "xgboost":  ("latr1_xgb", 0.96780),
        "lightgbm": ("lattri_lgbm", 0.96768),
        "catboost": ("latwide_cat", 0.96718),
    },
    "auc_tol": 5e-5,          # the table publishes 5 decimal places
    "noise_floor": 5e-5,      # the workspace's standing noise floor
    # R3 -- the derivation of the row 4 price cell
    "tuning_solo_gain": 3e-5,     # "tuning a GBDT for solo AUC was measured at +3e-5"
    "passthrough": 0.014,         # "solo->stack pass-through is 1.4%"
    "row4_price": 4e-7,           # the row 4 PRICE cell
    "seedavg_solo": 138e-6,       # "+138e-6 solo"
    "seedavg_stack": 2e-6,        # "moved the stack +2e-6"
    # R4 -- w123's CatBoost readings, used as the reproduction control and the comparand
    "cat_only_per_member": 10.041599941293389e-06,   # w123a_row3.json, base104, 3 splits
    "cat_only_n": 8,
    "rest_per_member_pub": 5.9e-6,   # the row 3 PRICE cell (a RESIDUAL group average)
    "rest_n": 35,
    # R5 -- w109a_dupscan relation 1
    "dup_pair": ("bolt_xgb_d7_alt1", "bolt_xgb_d7_alt2"),
}

# w109a_dupscan relation 2: `xgb_latcat_avg3` = mean(xgb_latcat, _s17, _s23), max|diff| EXACTLY
# 0.0. It is a seed average, not a model, so it cannot answer a "best SINGLE model" table. The
# table is checked both ways rather than picking one silently.
DERIVED = {"xgb_latcat_avg3"}

# Family read off the member NAME -- the only family evidence these foreign arrays carry.
# `_`-delimited tokens only, so `cat` does not match `lattice` (w121's identifier-boundary rule).
FAMILY = {
    "catboost": ("cat",),
    "xgboost": ("xgb",),
    "lightgbm": ("lgb", "lgbm"),
    "sklearn-gbdt": ("histgb",),
    "neural": ("lookup", "tabm", "realmlp", "mlp", "ft", "dcnv2", "gandalf"),
}


def classify(name):
    """Family tags present in a member name, matched on `_`-delimited tokens."""
    toks = set(name.split("_"))
    return [fam for fam, keys in FAMILY.items() if toks & set(keys)]


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
                    help="skip the paired re-measurement (R4)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w124a -- ANGLE INDEX row 4 (XGBoost) against its artefacts\n")

    # ------------------------------------------------------------------ R1, cheap first
    print("[R1] the pack-membership citation")
    on_disk = sorted(f[4:-4] for f in os.listdir(OOFDIR)
                     if f.startswith("oof_xgb_") and f.endswith(".npy"))
    print(f"    oof/oof_xgb_*: {on_disk}")
    missing = [n for n in PUBLISHED["in_pack"] if n not in on_disk]
    if missing:
        fail(f"row 4 cites XGB members that are not on disk: {missing}")
    else:
        print(f"    OK all {len(PUBLISHED['in_pack'])} cited XGB members resolve")
    out["oof_xgb_on_disk"] = on_disk

    # ------------------------------------------------------------------ R2, the family table
    print("\n[R2] 'best single models by family' -- recomputed from the arrays")
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    solo = {}
    for d in (OOFDIR, EXTRA_OOF):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not (f.startswith("oof_") and f.endswith(".npy")):
                continue
            nm = f[4:-4]
            if nm in solo:
                continue
            v = np.load(os.path.join(d, f))
            if v.shape[0] != len(y) or not np.isfinite(v).all():
                continue
            solo[nm] = float(roc_auc_score(y, v))
    print(f"    scored {len(solo)} OOF arrays across {OOFDIR} and {EXTRA_OOF}")
    fam_best = {}
    for fam in ("xgboost", "lightgbm", "catboost"):
        cand = {n: s for n, s in solo.items() if fam in classify(n)}
        if not cand:
            fail(f"no {fam} arrays found on disk at all")
            continue
        single = {n: s_ for n, s_ in cand.items() if n not in DERIVED}
        bn = max(cand, key=cand.get)
        sn = max(single, key=single.get) if single else bn
        fam_best[fam] = (bn, cand[bn], len(cand), sn, single.get(sn, cand[bn]))
        pn, ps = PUBLISHED["best_by_family"][fam]
        mark = "OK " if bn == pn and abs(cand[bn] - ps) <= PUBLISHED["auc_tol"] else "-> "
        print(f"    {mark}{fam:<9s} best array {bn:<22s} {cand[bn]:.5f}  | best SINGLE "
              f"{sn:<22s} {single.get(sn, cand[bn]):.5f}  (published {pn} {ps:.5f}, "
              f"over {len(cand)} arrays)")
        if bn != pn:
            note(f"the '{fam}' row of the family table names {pn} at {ps:.5f}; the best "
                 f"{fam}-named array on disk today is {bn} at {cand[bn]:.5f} "
                 f"({(cand[bn]-ps)*1e6:+.0f}e-6). The table is STALE, not wrong at the time.")
        elif abs(cand[bn] - ps) > PUBLISHED["auc_tol"]:
            fail(f"{pn} recomputes at {cand[bn]:.5f}, published {ps:.5f}")
        # the published name must at least still resolve and still hold its published score
        if pn in cand and abs(cand[pn] - ps) > PUBLISHED["auc_tol"]:
            fail(f"the published {fam} entry {pn} recomputes at {cand[pn]:.5f}, "
                 f"published {ps:.5f}")
        elif pn not in cand:
            fail(f"the published {fam} entry {pn} is not on disk")
    out["family_best"] = {k: {"name": v[0], "auc": v[1], "n": v[2],
                              "single_name": v[3], "single_auc": v[4]}
                          for k, v in fam_best.items()}
    if {"xgboost", "lightgbm", "catboost"} <= set(fam_best):
        xa, la, ca = (fam_best[f][1] for f in ("xgboost", "lightgbm", "catboost"))
        if xa >= la and xa >= ca:
            print("    OK 'XGBoost is not a missing leg' holds on today's arrays: the best "
                  "XGB array beats the best LGBM and the best CatBoost")
        else:
            fail("the 'best GBDT of any family' claim no longer holds on the arrays")
        # the claim can survive and still stop MEANING anything
        margin = xa - la
        pub_margin = (PUBLISHED["best_by_family"]["xgboost"][1]
                      - PUBLISHED["best_by_family"]["lightgbm"][1])
        print(f"    XGB - LGBM margin: published {pub_margin*1e6:+.0f}e-6, "
              f"on today's arrays {margin*1e6:+.0f}e-6  (noise floor "
              f"{PUBLISHED['noise_floor']*1e6:.0f}e-6)")
        if abs(margin) < PUBLISHED["noise_floor"] <= abs(pub_margin) or (
                abs(margin) < PUBLISHED["noise_floor"] and abs(margin) < abs(pub_margin) / 2):
            note(f"'best GBDT of any family' is now decided by {margin*1e6:+.0f}e-6, INSIDE "
                 f"the {PUBLISHED['noise_floor']*1e6:.0f}e-6 noise floor. It survives as an "
                 f"ordering and no longer survives as a separation.")
        if la > PUBLISHED["best_by_family"]["xgboost"][1]:
            note(f"and the best LGBM array today ({fam_best['lightgbm'][0]} "
                 f"{la:.5f}) BEATS the XGB number the table publishes "
                 f"({PUBLISHED['best_by_family']['xgboost'][1]:.5f}). Anyone reading the "
                 f"published table against today's disk gets the opposite ordering; the "
                 f"claim only holds because BOTH rows are stale.")

    # ------------------------------------------------------------------ R3, the derivation
    print("\n[R3] the derivation behind the +4e-7 PRICE cell")
    derived = PUBLISHED["tuning_solo_gain"] * PUBLISHED["passthrough"]
    print(f"    {PUBLISHED['tuning_solo_gain']*1e6:.0f}e-6 x {PUBLISHED['passthrough']:.1%} "
          f"= {derived*1e6:.2f}e-6  (published {PUBLISHED['row4_price']*1e6:.1f}e-6)")
    if abs(derived - PUBLISHED["row4_price"]) > 0.15e-6:
        fail(f"the +4e-7 cell does not multiply out: {derived*1e6:.2f}e-6")
    pt = PUBLISHED["seedavg_stack"] / PUBLISHED["seedavg_solo"]
    print(f"    and the 1.4% itself: {PUBLISHED['seedavg_stack']*1e6:.0f}e-6 / "
          f"{PUBLISHED['seedavg_solo']*1e6:.0f}e-6 = {pt:.2%}")
    if abs(pt - PUBLISHED["passthrough"]) > 0.002:
        fail(f"the 1.4% pass-through does not multiply out: {pt:.2%}")
    note("the +4e-7 is a TUNING price (value of tuning a GBDT already held). Row 3's "
         "5.9e-6/10.04e-6 is an ENROLMENT price (value of ADDING a member). Same column of "
         "the same table, different quantities, and the table does not say so.")
    out["derivation"] = {"derived": derived, "passthrough": pt}

    # ------------------------------------------------------------------ groups
    print("\n[R4/R5] reconstructing member_value2's groups from the arrays on disk")
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
    fam = {n: classify(n) for n in rest}
    xgb = sorted(n for n in rest if "xgboost" in fam[n])
    lgb = sorted(n for n in rest if "lightgbm" in fam[n])
    cat = sorted(n for n in rest if "catboost" in fam[n])
    print(f"    base {len(base)} | rest {len(rest)} = xgb {len(xgb)} + lgb {len(lgb)} + "
          f"cat {len(cat)} + other {len(rest)-len(xgb)-len(lgb)-len(cat)}")
    print(f"    XGB subgroup: {' '.join(xgb)}")
    if len(rest) != PUBLISHED["rest_n"]:
        fail(f"`rest` reconstructs at n={len(rest)}, published n={PUBLISHED['rest_n']}")
    if len(cat) != PUBLISHED["cat_only_n"]:
        fail(f"the CatBoost subgroup reconstructs at n={len(cat)}, w123 measured "
             f"n={PUBLISHED['cat_only_n']} -- the control is not comparable")
    out["groups"] = {"base": len(base), "rest": rest, "xgb": xgb, "lgb": lgb, "cat": cat}

    # ------------------------------------------------------------------ R5, the duplicate
    print("\n[R5] the duplicate inside the XGB subgroup")
    d1, d2 = PUBLISHED["dup_pair"]
    dup_ok = False
    if d1 in idx and d2 in idx:
        same_o = bool(np.array_equal(O[:, idx[d1]], O[:, idx[d2]]))
        same_t = bool(np.array_equal(T[:, idx[d1]], T[:, idx[d2]]))
        dup_ok = same_o and same_t
        print(f"    np.array_equal({d1}, {d2}): OOF {same_o}  TEST {same_t}")
        if not dup_ok:
            fail(f"w109a_dupscan relation 1 says {d1} == {d2} byte-identical; they are not")
    else:
        fail(f"the duplicate pair {d1}/{d2} is no longer in the loaded pack")
    dups_in_xgb = [n for n in (d1, d2) if n in xgb]
    xgb_distinct = len(xgb) - (1 if dup_ok and len(dups_in_xgb) == 2 else 0)
    if dup_ok and len(dups_in_xgb) == 2:
        note(f"both halves of the duplicate are in the XGB subgroup, so its per-member "
             f"denominator is {xgb_distinct} DISTINCT arrays, not {len(xgb)} names -- and the "
             f"same pair sits inside the {PUBLISHED['rest_n']} that the published "
             f"{PUBLISHED['rest_per_member_pub']*1e6:.1f}e-6 divides by "
             f"({PUBLISHED['rest_per_member_pub']*1e6:.2f}e-6 over 35 names becomes "
             f"{PUBLISHED['rest_per_member_pub']*35/34*1e6:.2f}e-6 over 34 arrays).")
    out["dup"] = {"identical": dup_ok, "in_xgb": dups_in_xgb, "xgb_distinct": xgb_distinct}

    # ------------------------------------------------------------------ R4, the fit
    if a.quick:
        print("\n[R4] --quick: paired re-measurement skipped")
    else:
        print(f"\n[R4] paired 50/50, {REPS} splits, C={CVAL}, transform={TRANSFORM} "
              f"(w123's procedure, same seeds, same base pool)")
        Z, _ = transform(O, T, TRANSFORM)
        xgb_dedup = [n for n in xgb if n != d2] if dup_ok else list(xgb)
        variants = {
            "base": base,
            "+xgb_only": base + xgb,
            "+xgb_dedup": base + xgb_dedup,
            "+lgb_only": base + lgb,
            "+cat_only": base + cat,
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
            print(f"      {c:<12s} n={n:2d}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  "
                  f"{d.mean()/n*1e6:+7.2f}e-6/member  [{sign}]")
        out["paired"] = res

        # the control: does w123's CatBoost number come back?
        cc = res["+cat_only"]
        gap = cc["per_member"] - PUBLISHED["cat_only_per_member"]
        print(f"\n    CONTROL  `+cat_only` re-measures {cc['per_member']*1e6:+.2f}e-6/member "
              f"against w123's {PUBLISHED['cat_only_per_member']*1e6:+.2f}e-6 "
              f"(gap {gap*1e6:+.3f}e-6)")
        if abs(gap) > 0.05e-6:
            fail(f"the CatBoost control does not reproduce w123 (gap {gap*1e6:+.3f}e-6); the "
                 f"XGB reading below is NOT comparable to it and must not be published")
        else:
            print("    OK the control reproduces to 0.05e-6, so XGB and CatBoost below are "
                  "measured on the same footing")

        xo, xd, lo = res["+xgb_only"], res["+xgb_dedup"], res["+lgb_only"]
        if xo["sign"] != "consistent":
            fail("the XGB subgroup delta does not hold its sign across splits")
        gd = xd["delta"] - xo["delta"]
        # Report the per-split differences, do not assert. Under L2 a duplicated column is not
        # obviously inert -- the pair shares the weight, so the ridge penalty on that direction
        # is halved -- so whether it "carries nothing" is a measurement, not a fact about rank.
        per_split = (rob["+xgb_dedup"] - rob["+xgb_only"]).to_numpy()
        dsign = ("consistent" if (per_split > 0).all() or (per_split < 0).all()
                 else "SIGN FLIPS -> not distinguishable from zero")
        print(f"\n    the duplicate: dropping `{d2}` moves the GROUP delta by {gd*1e6:+.3f}e-6 "
              f"[{dsign}] " + " ".join(f"{v*1e6:+.1f}e-6" for v in per_split) +
              f"; the per-member price moves from {xo['per_member']*1e6:+.2f}e-6 to "
              f"{xd['per_member']*1e6:+.2f}e-6, which is where the denominator bites")
        out["dup_effect"] = {"group_delta_shift": float(gd),
                             "per_split": [float(v) for v in per_split], "sign": dsign}
        print(f"\n    ENROLMENT PRICE, identical folds, identical base{len(base)}:")
        for lbl, r in (("XGBoost (distinct)", xd), ("LightGBM", lo), ("CatBoost", cc)):
            print(f"      {lbl:<20s} n={r['n']:2d}  {r['per_member']*1e6:+6.2f}e-6/member "
                  f"[{r['sign']}]")
        note(f"row 4 has never published an enrolment price. On identical folds the XGBoosts "
             f"read {xd['per_member']*1e6:+.2f}e-6/member over {xd['n']} distinct arrays, "
             f"against CatBoost {cc['per_member']*1e6:+.2f}e-6 and LightGBM "
             f"{lo['per_member']*1e6:+.2f}e-6.")

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    for m in notes:
        print("  note " + m)
    print(f"FAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    out["failures"] = fails
    out["notes"] = notes
    with open(os.path.join(HERE, "w124a_row4.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
