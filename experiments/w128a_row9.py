"""w128a -- ANGLE INDEX row 9 (error analysis) against its artefacts and its PRICE CELL.

THE HANDED ANGLE. "Error analysis: find where the current best model is wrong. Segment the
out-of-fold errors and look for structure a feature could capture" -- row 9, closed
2026-08-14 by four instruments and re-verified at the artefact level by w110 and w119.
Nothing here re-opens it. This prices the row's own PRICE CELL in the units the rest of the
table uses.

🔴 THE DEFECT. Row 9's price cell reads, in full:

    **0 / negative**

That is the entire cell. Rows 1..7 each name a QUANTITY (CONCAT / TUNING / ENROLMENT /
SEARCH), a MAGNITUDE with a unit, and -- since w124, w125, w126 and w127 each fixed one --
a LAYER, a SCOPE and a DENOMINATOR. Row 9 names none of the six. Two consequences:

  (a) NO BASELINE, AND THE ROW'S OWN INSTRUMENT PUBLISHES BOTH SIGNS. `w14d_bandmap`'s
      cross-fitted per-cell isotonic read -118e-6 against the uncorrected stack and
      +6e-6 against a size-matched permuted-cell control. "negative" is true of the first
      and FALSE of the second, and the cell does not say which one it quotes. Nine runs
      have now each found one missing facet of a price; this is the ninth: A PRICE IS A
      NUMBER, A QUANTITY, A LAYER, A SCOPE, A DENOMINATOR -- AND A BASELINE.

  (b) 🎯 NO MAGNITUDE, THEREFORE NO GUARD, AND THIS IS THE GENERAL LESSON. Standing checks
      #53 `w124b_priceunitguard` (units), #55 `w126c_scopeguard` (scope) and #56
      `w127b_rateguard` (denominator) all begin by selecting cells that carry an `e-6` or
      `e-7` token. Row 9 carries none, so all three skip it in silence. THE MOST
      UNDER-SPECIFIED CELL IN THE TABLE IS THE ONE CELL THE ENTIRE GUARD FAMILY IS
      STRUCTURALLY UNABLE TO SEE. A check that triggers on a number cannot see a cell that
      declines to give one -- and eight consecutive runs looked only where the numbers were.
      w127 predicted the next defect would be in row 1 or row 2, on exactly that reasoning,
      and it was wrong for exactly that reason: it searched the numbered cells too.

WHAT THIS MEASURES, pre-registered in `w128_prereg.txt` before any number for this run
existed. Row 9's manoeuvre priced as an ENROLMENT rate with rows 3/4's instrument (paired
50/50 StratifiedShuffleSplit at random_state 0,1,2; LogisticRegression C=1.0; transform
`hybrid`), ⚠ ON THE FULL LOADED PACK, not on rows 3/4's `base104`: the arms are columns
built in this process and base104 is a frozen subset that cannot contain them. The base104
CatBoost control anchors the INSTRUMENT to w123/w124; it does not make the arms' pool theirs.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or
decoration).
  R1 +   THE PUBLISHED ARTEFACTS, recomputed not cited. errormap's four segment figures,
         its two per-level endpoints, the hard band's row count (250,188, NOT 274k -- that
         is the adjacent pair), and bandmap's within/cross-cell deficit split.
  R2 +-  THE NEW COLUMNS ARE REAL COLUMNS. Each must differ from the stack it is built from
         (else the arm measures nothing) and the permuted control must have the same cell
         SIZES as the real one (else it is not size-matched).
  R3 +-  THE ENROLMENT ARMS, with the base104 CatBoost control re-measured in-process. If
         the control does not reproduce w123's +10.0416e-6/member the arms are not in rows
         3/4's units and are not published in them.
  R4 +   THE ARITHMETIC OF THE MISSING MAGNITUDE. The span the word "negative" covers across
         row 9's own four instruments, printed against the 50e-6 floor, on the record either
         way.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import CAT, COMP, DAILY, NUM, SUB, TARGET, get_folds, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, transform                     # noqa: E402
from w14d_bandmap import cells, decompose                                   # noqa: E402

DATA = os.path.join(ROOT, "data")
EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
OUT = os.path.join(HERE, "w128a_row9.json")

BLEND = "blend158_h3"
REPS = 3
CVAL = 1.0
TRANSFORM = "hybrid"
DECORR_MAX = 0.97
PERM_SEED = 20260830

# Published, frozen as literals: this is a COMPARISON, not a recomputation that agrees with
# itself. Sources named per claim.
PUBLISHED = {
    # R1 -- w110's I1, the errormap figures, quoted from "WHERE THE ERROR-ANALYSIS ANGLE WAS
    # ALREADY CLOSED"
    "global": 0.970048,
    "seg": {"n_missing_all": 0.974025, "n_screen_missing": 0.974071,
            "other_screen_band": 0.961145, "daily_band": 0.933423},
    "seg_tol": 1e-5,
    "lvl_0missing": 0.977541,
    "lvl_5missing": 0.913295,
    "rate_lo": 0.2421,
    "rate_hi": 0.9997,
    "hard_band_n": 250_188,          # daily_band 2 and 3, i.e. 4-8h.  274,034 is bands 3+4.
    # R1 -- w110's I2, the bandmap decomposition
    "within_deficit": 0.006964,
    "cross_deficit": 0.022987,
    "cross_share": 0.767,
    "deficit_tol": 2e-5,
    # R3 -- the control, from w123a_row3.json via w124a/w125a/w127a
    "cat_only_per_member": 10.041599941293389e-06,
    "cat_only_n": 8,
    "control_tol": 0.05e-6,
    # R4 -- the span the word "negative" covers, per instrument
    "negative_span": {
        "I2 per-cell isotonic vs the uncorrected stack": -118e-6,
        "I2 the same arm vs its permuted-cell control": +6e-6,
        "I4 residual booster vs control, round 100 (best)": -74e-6,
        "I4 residual booster vs control, round 25 (worst)": -526e-6,
        "I3 cell-local LightGBM, BAND 50 rounds (best)": -307e-6,
        "I3 cell-local LightGBM, G 400 rounds (worst)": -2043e-6,
    },
    "noise_floor": 50e-6,
    "pack_k": 104,
    # the registered intervals, from w128_prereg.txt
    "pred_enrol_lo": -30e-6,
    "pred_enrol_hi": +30e-6,
}


def pooled_within(y, s, g):
    """AUC over pairs inside the same segment, weighted by pair count. errormap.py's."""
    num = den = 0.0
    for v in np.unique(g):
        m = g == v
        yy, ss = y[m], s[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(yy, ss)
        den += w
    return num / den


def oof_cell_isotonic(p_tr, y, g_tr, p_te, g_te, folds):
    """Per-cell isotonic, fitted OUT OF FOLD on the frozen SKF5 for the train side and on
    the full train for the test side. Cells with one class in the fit part pass through."""
    out = np.empty_like(p_tr)
    for itr, iva in folds:
        for c in np.unique(g_tr):
            fit = itr[g_tr[itr] == c]
            app = iva[g_tr[iva] == c]
            if app.size == 0:
                continue
            if fit.size < 50 or len(np.unique(y[fit])) < 2:
                out[app] = p_tr[app]
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p_tr[fit], y[fit])
            out[app] = iso.predict(p_tr[app])
    ote = np.empty_like(p_te)
    for c in np.unique(g_te):
        fit = np.where(g_tr == c)[0]
        app = np.where(g_te == c)[0]
        if app.size == 0:
            continue
        if fit.size < 50 or len(np.unique(y[fit])) < 2:
            ote[app] = p_te[app]
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_tr[fit], y[fit])
        ote[app] = iso.predict(p_te[app])
    return np.clip(out, 1e-6, 1 - 1e-6), np.clip(ote, 1e-6, 1 - 1e-6)


def oof_cell_residual(p_tr, y, g_tr, p_te, g_te, folds):
    """The stack shifted by the OUT-OF-FOLD per-cell mean residual (y - p)."""
    out = np.empty_like(p_tr)
    for itr, iva in folds:
        for c in np.unique(g_tr):
            fit = itr[g_tr[itr] == c]
            app = iva[g_tr[iva] == c]
            if app.size == 0:
                continue
            off = float((y[fit] - p_tr[fit]).mean()) if fit.size else 0.0
            out[app] = p_tr[app] + off
    ote = np.empty_like(p_te)
    for c in np.unique(g_te):
        fit = np.where(g_tr == c)[0]
        app = np.where(g_te == c)[0]
        if app.size == 0:
            continue
        off = float((y[fit] - p_tr[fit]).mean()) if fit.size else 0.0
        ote[app] = p_te[app] + off
    return np.clip(out, 1e-6, 1 - 1e-6), np.clip(ote, 1e-6, 1 - 1e-6)


def paired(Z, y, cols_by_arm, reps=REPS):
    rows = []
    for rep in range(reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        r = {}
        for k, cols in cols_by_arm.items():
            m = LogisticRegression(max_iter=3000, C=CVAL).fit(Z[np.ix_(iA, cols)], y[iA])
            r[k] = roc_auc_score(y[iB], m.predict_proba(Z[np.ix_(iB, cols)])[:, 1])
        rows.append(r)
        print("    split %d: " % rep + "  ".join(f"{k}={v:.6f}" for k, v in r.items()),
              flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the paired arms (R3)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w128a -- ANGLE INDEX row 9 (error analysis) against its artefacts and its units\n")

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    p_tr = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    p_te = pd.read_csv(os.path.join(SUB, f"{BLEND}.csv"))[TARGET].to_numpy("float64")

    # ------------------------------------------------------------------ R1, the artefacts
    print("[R1] w110's I1 (errormap) and I2 (bandmap), recomputed from the saved vectors")
    g_auc = float(roc_auc_score(y, p_tr))
    print(f"    {BLEND} global OOF AUC {g_auc:.6f}   (published {PUBLISHED['global']:.6f})")
    if abs(g_auc - PUBLISHED["global"]) > PUBLISHED["seg_tol"]:
        fail(f"the global OOF AUC recomputes at {g_auc:.6f}")

    d = tr[DAILY].to_numpy("float64")
    comp = tr[COMP].to_numpy("float64")
    segs = {
        "n_missing_all": np.minimum(tr[NUM + CAT].isna().sum(1).to_numpy(), 5),
        "n_screen_missing": tr[[DAILY] + COMP].isna().sum(1).to_numpy(),
        "daily_band": np.where(np.isfinite(d), np.clip(np.floor(d / 2), 0, 6), -1),
        "other_screen_band": np.where(
            np.isfinite(d) & np.isfinite(comp).all(1),
            np.clip(np.floor(d - np.nansum(comp, axis=1)), 0, 5), -1),
    }
    got = {}
    for name, want in PUBLISHED["seg"].items():
        pw = pooled_within(y, p_tr, np.asarray(segs[name]).astype(str))
        got[name] = pw
        ok = abs(pw - want) <= PUBLISHED["seg_tol"]
        print(f"    {'OK ' if ok else '-> '}{name:<20s} pooled-within {pw:.6f} "
              f"({pw - g_auc:+.6f})   published {want:.6f}")
        if not ok:
            fail(f"errormap's {name} recomputes at {pw:.6f}, w110 published {want:.6f}")

    nm = np.asarray(segs["n_missing_all"])
    l0 = float(roc_auc_score(y[nm == 0], p_tr[nm == 0]))
    l5 = float(roc_auc_score(y[nm == 5], p_tr[nm == 5]))
    db = np.asarray(segs["daily_band"])
    r_lo, r_hi = float(y[db == 0].mean()), float(y[db == 6].mean())
    for label, v, want in (("0 missing within-level AUC", l0, PUBLISHED["lvl_0missing"]),
                           ("5+ missing within-level AUC", l5, PUBLISHED["lvl_5missing"]),
                           ("base rate, 0-2h band", r_lo, PUBLISHED["rate_lo"]),
                           ("base rate, 12h+ band", r_hi, PUBLISHED["rate_hi"])):
        ok = abs(v - want) <= 1e-4
        print(f"    {'OK ' if ok else '-> '}{label:<28s} {v:.6f}   published {want:.6f}")
        if not ok:
            fail(f"{label} recomputes at {v:.6f}, published {want:.6f}")

    hard = int(((db == 2) | (db == 3)).sum())
    adj = int(((db == 3) | (db == 4)).sum())
    ok = hard == PUBLISHED["hard_band_n"]
    print(f"    {'OK ' if ok else '-> '}hard band (4-8h) rows {hard:,}   published "
          f"{PUBLISHED['hard_band_n']:,}   [the adjacent pair 6-10h is {adj:,}, which is the "
          f"274k w110 corrected]")
    if not ok:
        fail(f"the hard band recomputes at {hard:,}")

    g_tr, order = cells(tr)
    U, M, _, _ = decompose(y, p_tr, g_tr, order)
    tot = M.sum()
    auc_chk = U.sum() / tot
    diag = np.eye(len(order), dtype=bool)
    within = float((M - U)[diag].sum() / tot)
    cross = float((M - U)[~diag].sum() / tot)
    share = cross / (within + cross)
    print(f"    bandmap: AUC from the U decomposition {auc_chk:.6f} (vs roc_auc "
          f"{g_auc:.6f})")
    for label, v, want in (("within-cell deficit", within, PUBLISHED["within_deficit"]),
                           ("cross-cell deficit", cross, PUBLISHED["cross_deficit"])):
        ok = abs(v - want) <= PUBLISHED["deficit_tol"]
        print(f"    {'OK ' if ok else '-> '}{label:<22s} {v:.6f}   published {want:.6f}")
        if not ok:
            fail(f"bandmap's {label} recomputes at {v:.6f}, published {want:.6f}")
    ok = abs(share - PUBLISHED["cross_share"]) <= 2e-3
    print(f"    {'OK ' if ok else '-> '}cross-cell SHARE       {share:.4f}   published "
          f"{PUBLISHED['cross_share']:.3f}   <- no within-segment feature can reach it")
    if not ok:
        fail(f"the cross-cell share recomputes at {share:.4f}")
    out["r1"] = {"global": g_auc, "seg": got, "within": within, "cross": cross,
                 "share": share, "hard_band_n": hard, "adjacent_n": adj}

    # ------------------------------------------------- R2, the columns row 9's angle asks for
    print("\n[R2] the columns the angle asks for, built out of fold on the frozen SKF5")
    g_te, _ = cells(te)
    folds = [(np.asarray(i), np.asarray(j)) for i, j in get_folds(y)]
    rng = np.random.default_rng(PERM_SEED)
    gp_tr = g_tr[rng.permutation(len(g_tr))]
    gp_te = g_te[rng.permutation(len(g_te))]

    iso_o, iso_t = oof_cell_isotonic(p_tr, y, g_tr, p_te, g_te, folds)
    res_o, res_t = oof_cell_residual(p_tr, y, g_tr, p_te, g_te, folds)
    isp_o, isp_t = oof_cell_isotonic(p_tr, y, gp_tr, p_te, gp_te, folds)

    new = {"cellIso": (iso_o, iso_t), "cellRes": (res_o, res_t), "cellIsoP": (isp_o, isp_t)}
    for nmm, (o_, _t) in new.items():
        auc = float(roc_auc_score(y, o_))
        diff = float(np.max(np.abs(o_ - p_tr)))
        print(f"    {nmm:<9s} solo OOF AUC {auc:.6f} ({(auc-g_auc)*1e6:+8.1f}e-6 vs the "
              f"stack)   max|col - stack| {diff:.4f}")
        if diff == 0.0:
            fail(f"{nmm} is byte-identical to the stack; the arm would measure nothing")
    sz_real = pd.Series(g_tr).value_counts().sort_index().to_numpy()
    sz_perm = pd.Series(gp_tr).value_counts().sort_index().to_numpy()
    ok = np.array_equal(sz_real, sz_perm)
    print(f"    {'OK ' if ok else '-> '}the permuted control is SIZE-MATCHED: "
          f"{list(sz_real)} vs {list(sz_perm)}")
    if not ok:
        fail("the permuted control is not size-matched; it is not an admissible null")
    ok = not np.array_equal(g_tr, gp_tr)
    print(f"    {'OK ' if ok else '-> '}...and it is actually permuted "
          f"({(g_tr != gp_tr).mean()*100:.1f}% of rows moved cell)")
    if not ok:
        fail("the permutation is the identity")
    out["r2"] = {n: {"solo": float(roc_auc_score(y, o_)),
                     "vs_stack": float(roc_auc_score(y, o_)) - g_auc}
                 for n, (o_, _t) in new.items()}

    # ------------------------------------------------------------- R3, the enrolment arms
    if a.quick:
        print("\n[R3] --quick: paired arms skipped")
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"\nFAILURES: {len(fails)}")
        return 1 if fails else 0

    print("\n[pool] the FULL loaded pack via load_members(drop=DEFAULT_DROP). ⚠ NOT base104 "
          "-- these columns were built in this process and base104 is a frozen subset. The "
          "base104 CatBoost control below anchors the INSTRUMENT only.")
    names, O, T = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
    print(f"    {len(names)} members loaded")
    O = np.column_stack([O] + [new[n][0] for n in new])
    T = np.column_stack([T] + [new[n][1] for n in new])
    names = list(names) + list(new)
    idx = {n: i for i, n in enumerate(names)}
    out["pool_n"] = len(names)

    Z, _ = transform(O, T, TRANSFORM)

    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    newv = [n for n in names if n in vet.index]
    bei = [n for n in newv if n.startswith("bei_")]
    lookup2 = [n for n in newv if n.startswith("bolt_lookup_v2")]
    decorr = [n for n in newv
              if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
    rest = [n for n in newv if n not in lookup2 and n not in decorr and n not in bei]
    cat = sorted(n for n in rest if "cat" in set(n.split("_")))
    catbase = [n for n in names if n not in vet.index and n not in new]
    if len(cat) != PUBLISHED["cat_only_n"]:
        fail(f"the CatBoost control group reconstructs at n={len(cat)}, w123 measured "
             f"n={PUBLISHED['cat_only_n']}")

    Q = [n for n in names if n not in new]
    arms_def = {
        "Q": Q,
        "R": Q + ["cellIso"],
        "S": Q + ["cellRes"],
        "T": Q + ["cellIsoP"],
        "catbase": catbase,
        "catbase+cat": catbase + cat,
    }
    cols_by_arm = {k: [idx[n] for n in v] for k, v in arms_def.items()}
    print(f"\n[R3] paired 50/50, {REPS} splits, C={CVAL}, transform={TRANSFORM} "
          f"(w123/w124's procedure and seeds).  Q={len(Q)}")
    rob = paired(Z, y, cols_by_arm)

    print("\n    THE CONTROL FIRST -- w123's CatBoost rate, in this process, these splits")
    dc = rob["catbase+cat"] - rob["catbase"]
    cpm = float(dc.mean() / len(cat))
    gap = cpm - PUBLISHED["cat_only_per_member"]
    print(f"      +cat_only n={len(cat)}  {cpm*1e6:+.4f}e-6/member  against w123's "
          f"{PUBLISHED['cat_only_per_member']*1e6:+.4f}e-6  (gap {gap*1e6:+.4f}e-6)")
    comparable = abs(gap) <= PUBLISHED["control_tol"]
    if not comparable:
        fail(f"the CatBoost control does not reproduce w123 (gap {gap*1e6:+.4f}e-6); the "
             f"arms below are NOT in rows 3/4's units and must not be published in them")
    out["control"] = {"per_member": cpm, "gap": gap, "n": len(cat),
                      "comparable": comparable}

    print("\n    THE ENROLMENT ARMS (paired; split noise cancels).  k=1 on every arm.")
    arms = {
        "per-cell isotonic      (R-Q)": rob["R"] - rob["Q"],
        "per-cell residual      (S-Q)": rob["S"] - rob["Q"],
        "the PERMUTED null      (T-Q)": rob["T"] - rob["Q"],
        "real minus null        (R-T)": rob["R"] - rob["T"],
    }
    res = {}
    for label, dd in arms.items():
        sign = "consistent" if (dd > 0).all() or (dd < 0).all() else "SIGN FLIPS"
        res[label] = {"delta": float(dd.mean()), "sd": float(dd.std(ddof=1)), "sign": sign}
        print(f"      {label:<30s} {dd.mean():+.6f} +/- {dd.std(ddof=1):.6f}  "
              f"{dd.mean()*1e6:+8.2f}e-6  [{sign}]")
    out["arms"] = res

    print("\n    THE REGISTERED PREDICTIONS, SCORED")
    for key in ("per-cell isotonic      (R-Q)", "per-cell residual      (S-Q)"):
        v = res[key]["delta"]
        ok = PUBLISHED["pred_enrol_lo"] <= v <= PUBLISHED["pred_enrol_hi"]
        print(f"      2  {key} in [{PUBLISHED['pred_enrol_lo']*1e6:+.0f}, "
              f"{PUBLISHED['pred_enrol_hi']*1e6:+.0f}]e-6 : {v*1e6:+.2f}e-6  "
              f"{'HELD' if ok else 'MISSED'}")
    rt = res["real minus null        (R-T)"]
    indist = abs(rt["delta"]) < 2 * rt["sd"] or rt["sign"] == "SIGN FLIPS"
    print(f"      3  (R-T) indistinguishable from zero : {rt['delta']*1e6:+.2f}e-6 "
          f"+/- {rt['sd']*1e6:.2f}e-6 [{rt['sign']}]  {'HELD' if indist else 'MISSED'}")
    reopen = (rt["delta"] > PUBLISHED["noise_floor"]) and rt["sign"] == "consistent"
    print(f"      4  DOES ROW 9 RE-OPEN?  (R-T) positive, 3/3, above the "
          f"{PUBLISHED['noise_floor']*1e6:.0f}e-6 floor : "
          f"{'*** YES, SAY SO ***' if reopen else 'NO'}")
    out["reopen"] = bool(reopen)

    # ------------------------------------------------------------- R4, the missing magnitude
    print("\n[R4] the span the word `negative` covers, on the record either way")
    sp = PUBLISHED["negative_span"]
    for label, v in sorted(sp.items(), key=lambda kv: kv[1]):
        side = "UNDER" if abs(v) < PUBLISHED["noise_floor"] else "OVER "
        print(f"    {v*1e6:+9.1f}e-6  {side} the {PUBLISHED['noise_floor']*1e6:.0f}e-6 "
              f"floor   {label}")
    vals = list(sp.values())
    rng_ = max(abs(min(vals)), abs(max(vals))) / min(abs(v) for v in vals)
    print(f"    ⟹ one word, {len(vals)} instrument readings, a {rng_:.0f}x span, and the two "
          f"ends fall on OPPOSITE sides of the floor.")
    print(f"    ⟹ signs present under the word `negative`: "
          f"{sorted(set('+' if v > 0 else '-' for v in vals))}")
    out["r4"] = {"span_ratio": rng_, "readings": sp}

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nFAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
