"""w126 -- ANGLE INDEX row 6 (blending / OOF weight search) read at the ARTEFACT level, and the
one thing its price cell does not say: WHICH SEARCH, AND HOW BIG (2026-08-30).

The handed ANGLE was *"Blending: rank-average or weight the tuned models by out-of-fold
performance. Search blend weights on OOF predictions, never on the public leaderboard."*
Genus `blending` -> row 6, closed, price **-1.07e-6**.

w108 already verified row 6's artefacts once and found the row's anchor pointed at the
RESTATEMENT rather than the evidence; it fixed that and split the handed string into its three
clauses. This is not that check repeated. It is the seventh instance of the genus w120-w125
have been walking through, one level further in:

  w120  `priority` named an input, read as an output
  w121  a prose cause sat beside a derived count
  w122  `slot` was printed where `tier` was meant
  w123  a family label sat on a residual group
  w124  a magnitude carried no unit                 -> #53 gave every price a QUANTITY
  w125  a magnitude carried no layer                -> #54 gave row 5 its LAYER
  w126  a SEARCH price carries no SCOPE.

Row 6's cell reads:  a **SEARCH** price (re-weighting members already in) -- **-1.07e-6**

The number and the parenthetical are about DIFFERENT SEARCHES, at different levels, with
different k:

  * -1.07e-6 is w36d's cross-arm number: honestly cross-fitted `all4` (k=4 TRANSFORM arms)
    against the zero-parameter equal-weight `h3`. It is clause 3 of the handed string.
  * "re-weighting members already in" is clause 1, and w108 recorded its status in terms:
    *"ALREADY THE SHIPPED ARCHITECTURE ... the incumbent IS an OOF-fitted weighted blend"*
    -- 104+ MEMBER weights, fitted inside the frozen folds by `agent/stack.py`.

So the cell hands a reader a k=4 top-level refusal wearing a k>=104 member-level label. And the
workspace's own scaling rule for search cost, `optimism ~ 0.55(k-1) e-6`, was fitted at k=3 and
k=4 ONLY. Reading the cell's parenthetical literally and multiplying that rule out to k=104
gives +57e-6, which is ABOVE the 50e-6 floor -- i.e. the mislabelling is not cosmetic, it flips
which side of the floor the row lands on. That is exactly the extrapolation w125 4 had just
finished warning about for the 1.4% pass-through, on a different constant.

CLAIMS CHECKED, each against an artefact rather than a quotation:

  R1  row 6's two anchors resolve, the evidence anchor carries -1.07e-6 and the three-clause
      table exists                                            -> read RESEARCH.md.
  R2  the published w36d table recomputes from the OOF arrays on disk with `fit_w` VERBATIM:
      equal / in-sample / cross-fitted / optimism for h3 and all4, and the cross-arm
      -1.0710e-6                                              -> to 1e-9 against w36d_wsearch.json.
  R3  NEW. the optimism of an OOF weight search as a function of k, measured at the MEMBER
      layer over the same base104 pool w123/w124/w125 priced on, with the shipped combiner.
      This prices the search row 6's parenthetical actually names.
  R4  NEW. `0.55(k-1) e-6` evaluated against that ladder. The rule is a two-point fit at
      k=3,4; the ladder reaches k=104, a 26x extrapolation.
  R5  NEW. an INSTRUMENT BRIDGE. R2's estimator (Nelder-Mead on the simplex, AUC objective)
      and R3's (LogisticRegression, the shipped stacker) are different search spaces, so their
      optimisms are not interchangeable by assumption. Both are run at the same k on the same
      columns and the ratio is reported rather than assumed to be 1.

This is a READ. It fits combiners to measure a price; it builds no member, enrols nothing,
sweeps no hyperparameter and ships no file. Re-measuring a price is not re-opening it (w122 6).

    .venv/bin/python experiments/w126a_row6.py            # rc 0 = every claim reproduces
    .venv/bin/python experiments/w126a_row6.py --quick    # R1/R2/R5 only, skip the k ladder

Deterministic: frozen SKF5 folds, fixed member order (seed 42 shuffle of base104), no API call,
no member fit, no fold rebuild.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA, SUB, TARGET, get_folds, load_raw          # noqa: E402
from stack import DEFAULT_DROP, load_members                        # noqa: E402

EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
RESEARCH = os.path.join(ROOT, "RESEARCH.md")

BASE = "w34_ad195std"
ALL4 = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")

# The k ladder. Nested by construction: a fixed seed-42 shuffle of base104, first k taken, so
# every rung contains every smaller rung and the curve is not a subset-selection artefact.
KS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 104)
KS_NM = (2, 3, 4, 6, 8)          # R5: Nelder-Mead is run on the member columns only this far
MEMBER_SEED = 42
CVAL = 1.0                        # member_value2 / w123-w125's C. NOT settable from here.

# Published, frozen as literals so this is a COMPARISON and not a recomputation that agrees
# with itself. Sources named per claim in the docstring.
PUBLISHED = {
    # R1 -- row 6's cell and its anchors
    "anchor_evidence": "THE PRICE OF A TOP-LEVEL SEARCH",
    "anchor_restatement": "BLENDING / OOF WEIGHT SEARCH / HILL CLIMBING",
    "row6_price": -1.07e-6,
    "cross_arm": -1.0710e-6,
    "clause_table_anchor": "ALREADY THE SHIPPED ARCHITECTURE",
    # R2 -- w36d_wsearch.json, verbatim
    "w36d": {
        "h3":   {"equal": 0.9701205752950482, "insample": 0.9701208766029016,
                 "xfit": 0.9701196117472624, "optimism": 1.2648556392269583e-06},
        "all4": {"equal": 0.9701181652381732, "insample": 0.9701213202375878,
                 "xfit": 0.9701195043323168, "optimism": 1.8159052710409185e-06},
    },
    "w36d_tol": 1e-9,
    # R4 -- the scaling rule, and the two points it was fitted on
    "rule_slope": 0.55e-6,        # per FREE parameter, i.e. per (k-1)
    "rule_fitted_at": (3, 4),
    "noise_floor": 5e-5,
    "base_n": 104,
}


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def fast_auc(y, s):
    """w36d's, verbatim. Verified against sklearn in main() before any search runs."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_w(R, y, w0):
    """w36d's simplex search, VERBATIM -- same method, same options, same objective. Copied
    rather than imported because w36d's module writes submission files at import-time scope;
    R2 asserts it reproduces w36d's published numbers, which is what makes the copy checkable."""
    def neg(w):
        w = np.abs(w)
        s = w.sum()
        if s <= 0:
            return 0.0
        return -fast_auc(y, R @ (w / s))
    r = minimize(neg, w0, method="Nelder-Mead",
                 options=dict(xatol=2e-4, fatol=2e-10, maxiter=150, disp=False))
    w = np.abs(r.x)
    return w / w.sum()


def search_simplex(R, y, folds):
    """equal / in-sample / cross-fitted for the simplex searcher."""
    n = R.shape[1]
    w0 = np.full(n, 1.0 / n)
    eq = roc_auc_score(y, R @ w0)
    ins = roc_auc_score(y, R @ fit_w(R, y, w0))
    pred = np.empty(len(y))
    for tr_i, va_i in folds:
        pred[va_i] = R[va_i] @ fit_w(R[tr_i], y[tr_i], w0)
    xf = roc_auc_score(y, pred)
    return dict(equal=eq, insample=ins, xfit=xf, optimism=ins - xf, d_xfit_vs_equal=xf - eq)


def search_logreg(R, y, folds, C=CVAL):
    """Same three numbers for the SHIPPED combiner. `agent/stack.py` fits exactly this object
    on member OOF inside exactly these folds, so at the member layer this is not an analogue of
    the search row 6's parenthetical names -- it IS that search."""
    eq = roc_auc_score(y, R.mean(1))
    m = LogisticRegression(max_iter=3000, C=C).fit(R, y)
    ins = roc_auc_score(y, m.decision_function(R))
    pred = np.empty(len(y))
    for tr_i, va_i in folds:
        mf = LogisticRegression(max_iter=3000, C=C).fit(R[tr_i], y[tr_i])
        pred[va_i] = mf.decision_function(R[va_i])
    xf = roc_auc_score(y, pred)
    return dict(equal=eq, insample=ins, xfit=xf, optimism=ins - xf, d_xfit_vs_equal=xf - eq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the k ladder (R3/R4)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w126a -- ANGLE INDEX row 6 (blending / OOF weight search) against its artefacts\n")

    # ------------------------------------------------------------------ R1, the citations
    print("[R1] row 6's citations")
    research = open(RESEARCH).read()
    for key in ("anchor_evidence", "anchor_restatement", "clause_table_anchor"):
        anc = PUBLISHED[key]
        n = research.count(anc)
        print(f"    {'OK ' if n else 'MISSING '}anchor {anc!r}: {n} occurrence(s)")
        if not n:
            fail(f"row 6's anchor {anc!r} does not resolve in RESEARCH.md")
    i = research.find(PUBLISHED["anchor_evidence"])
    win = research[i:i + 6000] if i >= 0 else ""
    ok = "1.07e-6" in win
    print(f"    {'OK ' if ok else 'MISSING '}the -1.07e-6 cross-arm figure at the evidence anchor")
    if not ok:
        fail("row 6's price is cited at 'THE PRICE OF A TOP-LEVEL SEARCH'; it is not there")

    # ------------------------------------------------------------------ R2, the artefact
    print("\n[R2] the w36d top-level table, recomputed from the OOF arrays on disk")
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    _p = np.random.default_rng(0).random(len(y))
    assert abs(fast_auc(y, _p) - roc_auc_score(y, _p)) < 1e-12, "fast_auc disagrees with sklearn"

    O = {}
    for k in ALL4:
        p = os.path.join(SUB, f"oof_{BASE}_{k}.npy")
        if not os.path.exists(p):
            fail(f"row 6's evidence array {os.path.basename(p)} is not on disk")
            continue
        O[k] = rk(np.load(p))
    t0 = time.time()
    r2 = {}
    for tag, keys in (("h3", H3), ("all4", ALL4)):
        if not all(k in O for k in keys):
            continue
        R = np.column_stack([O[k] for k in keys])
        r2[tag] = search_simplex(R, y, folds)
        pub = PUBLISHED["w36d"][tag]
        print(f"    {tag:<5s} k={len(keys)}")
        for f in ("equal", "insample", "xfit", "optimism"):
            got, want = r2[tag][f], pub[f]
            mark = "OK " if abs(got - want) <= PUBLISHED["w36d_tol"] else "-> "
            unit = "e-6" if f == "optimism" else ""
            show = f"{got*1e6:+.4f}e-6" if f == "optimism" else f"{got:.10f}"
            shw = f"{want*1e6:+.4f}e-6" if f == "optimism" else f"{want:.10f}"
            print(f"      {mark}{f:<9s} {show}   (published {shw})")
            if abs(got - want) > PUBLISHED["w36d_tol"]:
                fail(f"w36d {tag}.{f} recomputes at {got!r}, the artefact publishes {want!r}")
    if {"h3", "all4"} <= set(r2):
        cross = r2["all4"]["xfit"] - r2["h3"]["equal"]
        mark = "OK " if abs(cross - PUBLISHED["cross_arm"]) <= 1e-9 else "-> "
        print(f"    {mark}CROSS-ARM  fitted-all4 (k=4) vs equal-h3 (k=3, zero parameters): "
              f"{cross*1e6:+.4f}e-6   (row 6 publishes {PUBLISHED['row6_price']*1e6:+.2f}e-6)")
        if abs(cross - PUBLISHED["cross_arm"]) > 1e-9:
            fail(f"the cross-arm number recomputes at {cross*1e6:+.4f}e-6, "
                 f"published {PUBLISHED['cross_arm']*1e6:+.4f}e-6")
        out["cross_arm"] = cross
        note(f"row 6's -1.07e-6 is a k={len(ALL4)} TRANSFORM-arm number. Its cell's parenthetical "
             f"says 're-weighting members already in', which is a k>={PUBLISHED['base_n']} "
             f"MEMBER-layer search. R3 prices that one.")
    out["r2"] = r2
    print(f"    ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ the pool
    print(f"\n[R3] reconstructing the base pool w123/w124/w125 priced on")
    names, Om, Tm = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    base = [n for n in names if n not in vet.index]
    idx = {n: i for i, n in enumerate(names)}
    print(f"    pool {len(names)} | base {len(base)}")
    if len(base) != PUBLISHED["base_n"]:
        fail(f"the base pool reconstructs at n={len(base)}, w123-w125 measured on "
             f"base{PUBLISHED['base_n']} -- not the same object")
    order = list(np.random.default_rng(MEMBER_SEED).permutation(base))
    # rank-transform each member column: the SAME representation R2's arms are in, so the
    # ladder and the published two points are on one scale.
    Rm = np.column_stack([rk(Om[:, idx[n]]) for n in order])
    print(f"    member columns rank-transformed ({Rm.shape[1]}), nested order seed={MEMBER_SEED}")
    out["order"] = order

    # ------------------------------------------------------------------ R5, the bridge
    print("\n[R5] instrument bridge -- both searchers, same k, same member columns")
    bridge = {}
    for k in KS_NM:
        R = Rm[:, :k]
        s = search_simplex(R, y, folds)
        l = search_logreg(R, y, folds)
        bridge[k] = {"simplex": s, "logreg": l,
                     "ratio": (l["optimism"] / s["optimism"]) if s["optimism"] else None}
        rr = bridge[k]["ratio"]
        shown = "n/a (simplex optimism is exactly 0)" if rr is None else f"{rr:.2f}"
        print(f"    k={k:<3d} simplex optimism {s['optimism']*1e6:+.4f}e-6   "
              f"logreg optimism {l['optimism']*1e6:+.4f}e-6   logreg/simplex {shown}",
              flush=True)
    out["bridge"] = bridge

    # ------------------------------------------------------------------ R3/R4, the ladder
    if a.quick:
        print("\n[R3/R4] --quick: the k ladder is skipped")
    else:
        print(f"\n[R3] the optimism of a MEMBER-layer OOF weight search vs k "
              f"(shipped combiner, C={CVAL}, frozen SKF5)")
        print(f"    {'k':>4}  {'equal':>14}  {'in-sample':>14}  {'cross-fitted':>14}  "
              f"{'optimism':>12}  {'xfit-equal':>12}  {'0.55(k-1)':>10}  {'meas/rule':>9}")
        ladder = {}
        for k in KS:
            t = time.time()
            r = search_logreg(Rm[:, :k], y, folds)
            rule = PUBLISHED["rule_slope"] * (k - 1)
            r["rule"] = rule
            r["ratio"] = r["optimism"] / rule if rule else None
            ladder[k] = r
            print(f"    {k:>4}  {r['equal']:.10f}  {r['insample']:.10f}  {r['xfit']:.10f}  "
                  f"{r['optimism']*1e6:>+10.3f}e-6  {r['d_xfit_vs_equal']*1e6:>+10.2f}e-6  "
                  f"{rule*1e6:>8.2f}e-6  {r['ratio']:>9.3f}   ({time.time()-t:.0f}s)",
                  flush=True)
        out["ladder"] = ladder

        # ---------------------------------------------------------- R4, the rule
        print("\n[R4] `optimism ~ 0.55(k-1) e-6` against the ladder it was never fitted on")
        fit_at = [k for k in PUBLISHED["rule_fitted_at"] if k in ladder]
        big = max(ladder)
        r_big = ladder[big]
        print(f"    the rule is a TWO-POINT fit at k={PUBLISHED['rule_fitted_at']}. "
              f"The ladder reaches k={big}, a {big/max(PUBLISHED['rule_fitted_at']):.0f}x "
              f"extrapolation.")
        print(f"    at k={big}: rule predicts {r_big['rule']*1e6:+.1f}e-6, "
              f"measured {r_big['optimism']*1e6:+.1f}e-6, "
              f"ratio {r_big['ratio']:.2f}")
        side_rule = "ABOVE" if r_big["rule"] > PUBLISHED["noise_floor"] else "below"
        side_meas = "ABOVE" if r_big["optimism"] > PUBLISHED["noise_floor"] else "below"
        print(f"    against the {PUBLISHED['noise_floor']*1e6:.0f}e-6 floor: "
              f"the RULE puts k={big} {side_rule} it, the MEASUREMENT puts it {side_meas} it.")
        if side_rule != side_meas:
            note(f"the rule and the measurement land on OPPOSITE SIDES of the floor at k={big}. "
                 f"Extrapolating a two-point fit {big/max(PUBLISHED['rule_fitted_at']):.0f}x "
                 f"past its range decides the closure by itself.")
        # is the search itself worth anything at the member layer?
        pays = [k for k in ladder if ladder[k]["d_xfit_vs_equal"] > 0]
        print(f"    honest search beats equal weights at {len(pays)}/{len(ladder)} rungs; "
              f"at k={big} by {r_big['d_xfit_vs_equal']*1e6:+.1f}e-6")
        note(f"at the MEMBER layer the honestly cross-fitted search beats the equal-weight "
             f"baseline by {r_big['d_xfit_vs_equal']*1e6:+.1f}e-6 at k={big}. Row 6's price is "
             f"{PUBLISHED['row6_price']*1e6:+.2f}e-6. The two searches do not share a sign, and "
             f"the positive one is the incumbent, not a candidate.")

    out["fails"], out["notes"] = fails, notes
    json.dump(out, open(os.path.join(HERE, "w126a_row6.json"), "w"), indent=1, default=float)
    print(f"\nFAILURES: {len(fails)}   notes: {len(notes)}")
    for m in fails:
        print("  FAIL " + m)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
