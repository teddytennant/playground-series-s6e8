"""w125a -- ANGLE INDEX row 5 (feature engineering) read at the ARTEFACT level, and given the
price it has never had: a STACK-layer one (2026-08-30).

The handed ANGLE was *"feature engineering: interactions, in-fold target and count encodings,
and careful categorical treatment. Measure every feature on CV, keep only what pays."* Genus
`feature engineering` -> row 5, closed, price **negative**. Sixteenth handing, the most-handed
row in the index.

w107 already verified row 5's artefacts once. This is NOT that check repeated. w123 found row
3's price was a residual group's average; w124 found row 4's column mixed two quantities and
gave it the enrolment number it lacked. Row 5 has the third variant of the same defect, one
level down:

  * every other magnitude in the `price` column is a STACK-layer number -- e-6 of blend CV.
  * row 5's **negative** is carried by MEMBER-layer numbers: -19.26e-6 (xgb), -82.68e-6 (cat)
    solo fold AUC, on top of the LightGBM null.
  These are not the same layer, and the workspace's own standing rule says so in terms:
  *"member-level AUC is not evidence about stack value ... the sign is not even guaranteed"*
  (w106, and RESEARCH's own "quote a member's fold AUC as a member number and never as a
  blend number").  #53 gave every price cell a QUANTITY.  None of them carries a LAYER.

So this run measures the quantity row 5 is missing: the **stack-layer enrolment price of the
feature blocks themselves**, using the ablation ladder that is already on disk as matched
pairs (`w27r_blockdrop`: same model, same PARAMS, same rounds, same seed, same frozen SKF5
folds -- only the column set differs). Each arm is enrolled ALONE into the same base104 pool,
paired 50/50, same three splits and the same seeds w123/w124 used, with CatBoost re-measured
in the same process as a reproduction control.

That also puts a SECOND point on the workspace's `1.4%` solo->stack pass-through, which is
quoted in nine places as a conversion constant and was measured on exactly one thing (seed-
averaging `xgb_latcat`, +138e-6 solo -> +2e-6 stack). The encoding channel is a member-level
effect of +13,253e-6 -- two orders of magnitude past anything the constant was fitted on.

CLAIMS CHECKED, each against an artefact rather than a quotation:

  R1  row 5's citations -- `agent/features.py:te_block`, the "A SECOND skew in the same block"
      section, and the "measured negative in three model classes" anchor with its two
      magnitudes                                       -> read the files.
  R2  the feature-block ablation ladder recomputes from the arrays on disk, to the published
      10 dp, and `tedrop` -- published as "*below encdrop*" with no number -- gets one, which
      also prices the published "~ -2,700e-6" marginal exactly.
  R3  NEW. stack-layer enrolment price of each ablation arm, enrolled alone into base104.
  R4  NEW. the solo->stack pass-through implied by R2/R3, against the published 1.4%.

This is a READ. It fits a meta-combiner to measure a price; it builds no member, enrols
nothing, sweeps no hyperparameter and ships no file. Re-measuring a price is not re-opening
it (w122 6).

    .venv/bin/python experiments/w125a_row5.py            # rc 0 = every claim reproduces
    .venv/bin/python experiments/w125a_row5.py --quick    # skip the paired fit (R1/R2)

Deterministic: fixed splits (StratifiedShuffleSplit random_state=rep), no API call, no member
fit, no fold rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
RESEARCH = os.path.join(ROOT, "RESEARCH.md")
FEATURES = os.path.join(ROOT, "agent", "features.py")
DECORR_MAX = 0.97          # member_value2's own cut. NOT settable from here.
REPS = 3                   # w123/w124 used 3 splits; same number, same seeds, same procedure
CVAL = 1.0
TRANSFORM = "hybrid"

# The ablation ladder, and WHERE each arm is read from. `ctdrop`/`encdrop` are taken from the
# PINNED directory: `data/ext_members7` is an export target w27r writes into while it runs, and
# `ext_members7pin` holds hardlinks to exactly the registered bytes. Same inodes, checked in R2.
ARMS = {
    "lat_ctraw_r400":  (os.path.join(DATA, "ext_members6"), "control, 184 cols"),
    "lat_ctfix_r400":  (os.path.join(DATA, "ext_members6"), "CT_ x 4/3 skew fix, 184 cols"),
    "lat_ctdrop_r400": (os.path.join(DATA, "ext_members7pin"), "drop CT_, 112 cols"),
    "lat_tedrop_r400": (os.path.join(DATA, "ext_members7"), "drop TE_, 112 cols"),
    "lat_encdrop_r400": (os.path.join(DATA, "ext_members7pin"), "drop TE_+CT_, 40 cols"),
    "lat_rawdrop_r400": (os.path.join(DATA, "ext_members7"), "drop raw/derived, 144 cols"),
}
PINNED = ("lat_ctdrop_r400", "lat_encdrop_r400")

# Published, frozen as literals so this is a COMPARISON, not a recomputation that agrees with
# itself. Sources named per claim in the docstring above.
PUBLISHED = {
    # R1 -- row 5's citations
    "te_block_line": 175,
    "section_anchor": "A SECOND skew in the same block",
    "member_anchor": "THE CT SKEW IS A PROPERTY OF THE MATRIX",
    "member_level": {"xgb": -19.26e-6, "cat": -82.68e-6},
    # R2 -- the feature-block ablation ladder, "pooled OOF" column
    "ladder": {
        "lat_ctraw_r400":   0.9654813306,
        "lat_ctfix_r400":   0.9657751945,
        "lat_ctdrop_r400":  0.9656895129,
        "lat_encdrop_r400": 0.9522288823,
        # `lat_tedrop_r400` is published only as "*below encdrop*" -- no number. R2 gives it one.
        # `lat_rawdrop_r400` never appears in the ladder table at all.
    },
    "ladder_tol": 1e-9,           # the table publishes 10 dp; these are the same arrays
    "ct_on_raw_marginal": -2700e-6,   # "raw + CT_ (tedrop) ... ~ -2,700e-6"
    "ct_on_raw_tol": 200e-6,          # published with a "~"
    "cols": {"lat_ctraw_r400": 184, "lat_ctfix_r400": 184, "lat_ctdrop_r400": 112,
             "lat_tedrop_r400": 112,
             "lat_encdrop_r400": 40, "lat_rawdrop_r400": 144},
    "noise_floor": 5e-5,          # the workspace's standing noise floor
    # R3 -- w123's CatBoost reading, used as the reproduction control
    "cat_only_per_member": 10.041599941293389e-06,   # w123a_row3.json, base104, 3 splits
    "cat_only_n": 8,
    "base_n": 104,
    # R4 -- the pass-through constant row 4's price is derived from
    "passthrough": 0.014,
    "passthrough_solo": 138e-6,   # seed-averaging xgb_latcat, solo
    "passthrough_stack": 2e-6,    # ... and what it moved the stack
}

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
                    help="skip the paired re-measurement (R3/R4)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w125a -- ANGLE INDEX row 5 (feature engineering) against its artefacts\n")

    # ------------------------------------------------------------------ R1, the citations
    print("[R1] row 5's citations")
    src = open(FEATURES).read().splitlines()
    hits = [i + 1 for i, ln in enumerate(src) if re.search(r"\bdef te_block\b", ln)]
    if not hits:
        fail("row 5 re-derives from `agent/features.py:te_block`; there is no such function")
    else:
        print(f"    OK agent/features.py:te_block at line {hits[0]} "
              f"(row 5's evidence table says {PUBLISHED['te_block_line']})")
        if hits[0] != PUBLISHED["te_block_line"]:
            note(f"the te_block line number has moved {PUBLISHED['te_block_line']} -> "
                 f"{hits[0]}; the FUNCTION is the citation, the line number is not.")
    research = open(RESEARCH).read()
    for key in ("section_anchor", "member_anchor"):
        anc = PUBLISHED[key]
        n = research.count(anc)
        print(f"    {'OK ' if n else 'MISSING '}anchor {anc!r}: {n} occurrence(s)")
        if not n:
            fail(f"row 5's anchor {anc!r} does not resolve in RESEARCH.md")
    # the two member-level magnitudes must actually be at the anchor they are cited from
    i = research.find(PUBLISHED["member_anchor"])
    win = research[i:i + 4000] if i >= 0 else ""
    for k, v in PUBLISHED["member_level"].items():
        lit = f"{abs(v)*1e6:.2f}e-6"
        ok = lit in win
        print(f"    {'OK ' if ok else 'MISSING '}the {k} member-level figure {lit} at that anchor")
        if not ok:
            fail(f"row 5 cites {lit} ({k}) at {PUBLISHED['member_anchor']!r}; not found there")
    out["r1"] = {"te_block_line": hits[0] if hits else None}

    # ------------------------------------------------------------------ R2, the ladder
    print("\n[R2] the feature-block ablation ladder, recomputed from the arrays")
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    for nm in PINNED:
        p7 = os.path.join(DATA, "ext_members7", f"oof_{nm}.npy")
        pp = os.path.join(DATA, "ext_members7pin", f"oof_{nm}.npy")
        same = os.path.exists(p7) and os.stat(p7).st_ino == os.stat(pp).st_ino
        print(f"    pin  {nm}: ext_members7pin is the same inode as ext_members7 -> {same}")
        if not same:
            fail(f"{nm} in ext_members7pin is NOT a hardlink of the export; the pin is broken")
    arm_oof, arm_test, solo = {}, {}, {}
    for nm, (d, what) in ARMS.items():
        o = np.load(os.path.join(d, f"oof_{nm}.npy")).astype(np.float64)
        t = np.load(os.path.join(d, f"test_{nm}.npy")).astype(np.float64)
        if o.shape != (len(y),) or t.shape != (len(te),):
            fail(f"{nm} has shapes {o.shape}/{t.shape}, expected ({len(y)},)/({len(te)},)")
            continue
        arm_oof[nm], arm_test[nm] = o, t
        solo[nm] = float(roc_auc_score(y, o))
        pub = PUBLISHED["ladder"].get(nm)
        mark = "OK " if pub is None or abs(solo[nm] - pub) <= PUBLISHED["ladder_tol"] else "-> "
        extra = f"published {pub:.10f}" if pub is not None else "NEVER PUBLISHED AS A NUMBER"
        print(f"    {mark}{nm:<19s} {PUBLISHED['cols'].get(nm, '?'):>4} cols  "
              f"solo {solo[nm]:.10f}  ({extra}, {what})")
        if pub is not None and abs(solo[nm] - pub) > PUBLISHED["ladder_tol"]:
            fail(f"{nm} recomputes at {solo[nm]:.10f}, the ladder publishes {pub:.10f}")
    out["solo"] = solo

    # the one cell the ladder left as prose, and the marginal it is quoted for
    if {"lat_tedrop_r400", "lat_encdrop_r400"} <= set(solo):
        gap = solo["lat_tedrop_r400"] - solo["lat_encdrop_r400"]
        print(f"\n    `tedrop` is published as '*below encdrop*' with no number. It is "
              f"{solo['lat_tedrop_r400']:.10f}, which is {gap*1e6:+.1f}e-6 below `encdrop` "
              f"-- so the ladder's '~{PUBLISHED['ct_on_raw_marginal']*1e6:.0f}e-6' marginal "
              f"value of CT_ on a raw frame is exactly {gap*1e6:+.1f}e-6")
        if abs(gap - PUBLISHED["ct_on_raw_marginal"]) > PUBLISHED["ct_on_raw_tol"]:
            fail(f"the '~{PUBLISHED['ct_on_raw_marginal']*1e6:.0f}e-6' CT_-on-raw marginal "
                 f"measures {gap*1e6:+.1f}e-6")
        note(f"`lat_tedrop_r400` now has a number ({solo['lat_tedrop_r400']:.10f}); the "
             f"ladder carried it as prose. `lat_rawdrop_r400` is not in the ladder at all "
             f"and is {solo.get('lat_rawdrop_r400', float('nan')):.10f}.")
        out["ct_on_raw"] = gap
    # the member-level size of the whole encoding channel -- the denominator R4 divides by
    if {"lat_ctraw_r400", "lat_encdrop_r400"} <= set(solo):
        enc_solo = solo["lat_ctraw_r400"] - solo["lat_encdrop_r400"]
        print(f"    the whole ENCODING channel (TE_+CT_) is worth {enc_solo*1e6:+.0f}e-6 at "
              f"MEMBER level -- {enc_solo/PUBLISHED['passthrough_solo']:.0f}x the +"
              f"{PUBLISHED['passthrough_solo']*1e6:.0f}e-6 the 1.4% pass-through was fitted on")
        out["enc_solo"] = enc_solo

    # ------------------------------------------------------------------ the pool
    print(f"\n[R3] reconstructing member_value2's base pool from the arrays on disk")
    names, O, T = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
    clash = [n for n in ARMS if n in names]
    if clash:
        fail(f"ablation arms are ALREADY in the base pool, so enrolling them measures "
             f"nothing: {clash}")
    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    new = [n for n in names if n in vet.index]
    base = [n for n in names if n not in vet.index]
    bei = [n for n in new if n.startswith("bei_")]
    lookup2 = [n for n in new if n.startswith("bolt_lookup_v2")]
    decorr = [n for n in new
              if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
    rest = [n for n in new if n not in lookup2 and n not in decorr and n not in bei]
    cat = sorted(n for n in rest if "catboost" in classify(n))
    print(f"    pool {len(names)} | base {len(base)} | rest {len(rest)} | "
          f"CatBoost subgroup {len(cat)}")
    if len(base) != PUBLISHED["base_n"]:
        fail(f"the base pool reconstructs at n={len(base)}, w123/w124 measured on "
             f"base{PUBLISHED['base_n']} -- the control is not comparable")
    if len(cat) != PUBLISHED["cat_only_n"]:
        fail(f"the CatBoost subgroup reconstructs at n={len(cat)}, w123 measured "
             f"n={PUBLISHED['cat_only_n']} -- the control is not comparable")
    out["groups"] = {"pool": len(names), "base": len(base), "rest": len(rest), "cat": cat}

    # ------------------------------------------------------------------ R3, the fit
    if a.quick:
        print("\n[R3/R4] --quick: paired re-measurement skipped")
    else:
        arms = [n for n in ARMS if n in arm_oof]
        Ofull = np.column_stack([O] + [arm_oof[n] for n in arms])
        Tfull = np.column_stack([T] + [arm_test[n] for n in arms])
        allnames = list(names) + arms
        idx = {n: i for i, n in enumerate(allnames)}
        print(f"\n    transform={TRANSFORM} over {Ofull.shape[1]} columns "
              f"({len(names)} pool + {len(arms)} arms). `transform` is per-column, so the "
              f"base columns are bit-identical to w123/w124's -- the control below proves it.")
        Z, _ = transform(Ofull, Tfull, TRANSFORM)
        variants = {"base": base, "+cat_only": base + cat}
        for n in arms:
            variants["+" + n.replace("lat_", "").replace("_r400", "")] = base + [n]
        print(f"\n[R3] paired 50/50, {REPS} splits, C={CVAL} "
              f"(w123/w124's procedure, same seeds, same base pool)")
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
                      "per_member": float(d.mean() / n), "sign": sign,
                      "per_split": [float(v) for v in d]}
            print(f"      {c:<14s} n={n:2d}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  "
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
                 f"ablation readings below are NOT comparable to it and must not be published")
        else:
            print("    OK the control reproduces to 0.05e-6, so the arms below sit on the "
                  "same footing as row 3's and row 4's enrolment prices")

        # ------------------------------------------------------------- R4, the pass-through
        print(f"\n[R4] MEMBER layer vs STACK layer, arm by arm "
              f"(both measured here, on the same arrays)")
        print(f"      {'arm':<20s} {'solo':>13s} {'solo d vs encdrop':>19s} "
              f"{'STACK enrolment':>17s} {'pass-through':>13s}")
        ref = "lat_encdrop_r400"
        pt_rows = {}
        for n in arms:
            key = "+" + n.replace("lat_", "").replace("_r400", "")
            ds = solo[n] - solo[ref]
            dk = res[key]["delta"] - res["+" + ref.replace("lat_", "").replace("_r400", "")]["delta"]
            pt = (dk / ds) if abs(ds) > 1e-9 else float("nan")
            pt_rows[n] = {"solo": solo[n], "solo_vs_ref": ds, "stack_vs_ref": dk,
                          "passthrough": pt}
            print(f"      {n:<20s} {solo[n]:13.7f} {ds*1e6:+16.0f}e-6 "
                  f"{res[key]['delta']*1e6:+14.2f}e-6 "
                  + (f"{pt*100:12.4f}%" if np.isfinite(pt) and abs(ds) > 1e-6 else f"{'--':>13s}"))
        out["passthrough_rows"] = pt_rows

        full = pt_rows.get("lat_ctraw_r400")
        if full and abs(full["solo_vs_ref"]) > 1e-6:
            implied = full["solo_vs_ref"] * PUBLISHED["passthrough"]
            # The contrast is a DIFFERENCE OF TWO NOISY n=1 ENROLMENTS, so report whether it
            # holds its sign before reading a ratio off it. w124 §5: an assertion where a
            # measurement was available is this run's own defect genus one level up.
            ref_key = "+" + ref.replace("lat_", "").replace("_r400", "")
            cs = (np.array(res["+ctraw"]["per_split"])
                  - np.array(res[ref_key]["per_split"]))
            csign = ("consistent" if (cs > 0).all() or (cs < 0).all()
                     else "SIGN FLIPS -> NOT DISTINGUISHABLE FROM ZERO")
            print(f"\n    THE ENCODING CHANNEL, both ways:")
            print(f"      member layer   {full['solo_vs_ref']*1e6:+.0f}e-6  "
                  f"(`ctraw` 184 cols vs `encdrop` 40 cols, same model/seed/folds)")
            print(f"      stack  layer   {full['stack_vs_ref']*1e6:+.2f}e-6 "
                  f"[{csign}] per split " + " ".join(f"{v*1e6:+.2f}" for v in cs) +
                  f"  (same two arrays, each enrolled alone into base{len(base)})")
            print(f"      the published 1.4% predicts {implied*1e6:+.0f}e-6 into the stack. "
                  f"The largest single split puts the channel at "
                  f"{max(abs(cs))*1e6:.2f}e-6, i.e. a pass-through of at most "
                  f"{max(abs(cs))/full['solo_vs_ref']*100:.3f}% -- and the mean has the "
                  f"other sign.")
            note(f"the 1.4% solo->stack pass-through is quoted as a conversion constant and "
                 f"was fitted on ONE point (+{PUBLISHED['passthrough_solo']*1e6:.0f}"
                 f"e-6 solo -> +{PUBLISHED['passthrough_stack']*1e6:.0f}e-6 stack, seed-"
                 f"averaging xgb_latcat). The encoding channel is a second point "
                 f"{full['solo_vs_ref']/PUBLISHED['passthrough_solo']:.0f}x further out: at "
                 f"the stack layer it is NOT DISTINGUISHABLE FROM ZERO ({csign}), against a "
                 f"1.4%-implied {implied*1e6:+.0f}e-6. Bounded by the largest split it is "
                 f"under {max(abs(cs))/full['solo_vs_ref']*100:.3f}%. It is not a constant, "
                 f"and it is not safe to extrapolate.")
        note(f"row 5's price cell reads `negative` and is carried by MEMBER-layer numbers "
             f"({PUBLISHED['member_level']['xgb']*1e6:.2f}e-6 xgb, "
             f"{PUBLISHED['member_level']['cat']*1e6:.2f}e-6 cat). Every other magnitude in "
             f"that column is a STACK-layer number. Measured at the stack layer the arms "
             f"read " + " ".join(
                 f"{n.replace('lat_','').replace('_r400','')} "
                 f"{res['+' + n.replace('lat_','').replace('_r400','')]['delta']*1e6:+.2f}e-6"
                 for n in arms) + f", all inside the {PUBLISHED['noise_floor']*1e6:.0f}e-6 "
             f"floor. The closure does not change; its price now names its layer.")

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    for m in notes:
        print("  note " + m)
    print(f"FAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    out["failures"] = fails
    out["notes"] = notes
    with open(os.path.join(HERE, "w125a_row5.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
