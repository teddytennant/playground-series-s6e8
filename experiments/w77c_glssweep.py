"""w77c — WHY THE ENLARGED DESIGN BROKE: the GLS common gap leaves its own data.

Pre-registration: experiments/w77_prereg_b.txt, committed e475ffb BEFORE this file existed.
⚠ POST-HOC DIAGNOSTIC, prompted by w77a's ARM L output. The prereg separates what was already
known (gh = -4598 against z ~ +1050) from what is genuinely open (where it starts, and whether
the LIVE 8-file estimate is safe).

`fit` estimates one common (LB - CV) gap by GLS over the scored files:

    iS = inv(Sxx[js,js]);  Vg = inv(1' iS 1);  gh = Vg (1' iS z)

That is a weighted mean whose weights w = Vg (1' iS) sum to 1 but are NOT constrained to be
non-negative. With an ESTIMATED, strongly-coupled Sxx the weights can be enormous and of both
signs, and the "mean" can land far outside the data. w77a caught it landing 5,600e-6 outside.

THE SWEEP IS EXACT, NOT APPROXIMATE. St and Sp are entrywise functions of cov(A) and cov(B),
so Sxx[js,js] for a subset IS the Sxx that sub-design would have had. GATE B proves it by
reproducing w63a's stored `gap` from the 122-file matrix.

⛔ WRITES NO BAR AND DOES NOT TOUCH `fit`. Q5 measures the equal-weight alternative and stops.

    .venv/bin/python experiments/w77c_glssweep.py     # ~7 min first run, seconds after (npz cache)
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import SUB, TARGET, load_raw                       # noqa: E402
import w63a_setprice as W63A                                   # noqa: E402
import w76a_addtest as W76A                                    # noqa: E402

U = 1e-6
CACHE = os.path.join(HERE, "w77c_gls.npz")
FAILURES, FALSIFIED = 0, []


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def falsify(tag: str, msg: str) -> None:
    FALSIFIED.append(tag)
    print(f"  🔴 {tag} FALSIFIED — {msg}")


def gls(Sxx, js, z):
    """`fit`'s estimator, verbatim, plus the weights it never names."""
    iS = np.linalg.inv(Sxx[np.ix_(js, js)])
    D1 = np.ones((len(js), 1))
    Vg = float(np.linalg.inv(D1.T @ iS @ D1)[0, 0])
    w = np.ravel(Vg * (D1.T @ iS))
    return float(w @ z), Vg, w


def main() -> None:
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    plan = list(ref["plan_0824"])
    T1, T2 = list(ref["tiers"]["slot1"]), list(ref["tiers"]["slot2"])
    S_NAMES = sorted(set(T1 + T2 + list(ref["wanted"]) + plan))
    lb_all, lt1, _, _, _ = W76A.board(exclude=plan)
    assert sorted(lt1) == sorted(T1), "the board did not rewind"

    have = [k for k in sorted(lb_all) if os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))]
    L_NAMES = sorted(set(S_NAMES) | set(have))

    if os.path.exists(CACHE):
        c = np.load(CACHE, allow_pickle=True)
        NAMES, Sxx, cvv = list(c["names"]), c["Sxx"], c["cv"]
        print(f"loaded cached 122-file Sxx from {os.path.basename(CACHE)}")
        assert NAMES == L_NAMES, "the cached design is not the live one -- delete the npz"
    else:
        beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))
                     ["coupling_beta_median"])
        tr, _ = load_raw()
        y = tr[TARGET].astype(int).to_numpy()
        V, cvd = W63A._load(L_NAMES, y)
        LB_L = {k: v for k, v in lb_all.items() if k in set(L_NAMES)}
        E = W63A.fit(L_NAMES, LB_L, cvd, V, y, beta)
        NAMES, Sxx = list(E["names"]), E["Sxx"]
        cvv = np.array([cvd[k] for k in NAMES])
        np.savez(CACHE, names=np.array(NAMES, dtype=object), Sxx=Sxx, cv=cvv)
        print(f"fitted and cached the 122-file Sxx (gh from `fit` itself: {E['gap']:+.4f}e-6)")

    col = {k: i for i, k in enumerate(NAMES)}
    cv = {k: float(v) for k, v in zip(NAMES, cvv)}
    z = np.array([(lb_all[k] - cv[k]) / U if k in lb_all else np.nan for k in NAMES])

    # ---------------------------------------------------------------------------- Q1 GATE B
    print("\n" + "=" * 94)
    print("GATE B (Q1) — the submatrix of the 122-file Sxx must reproduce w63a's stored `gap`")
    print("=" * 94)
    s_scored = [k for k in S_NAMES if k in lb_all]
    js_s = [col[k] for k in s_scored]
    gh_s, Vg_s, w_s = gls(Sxx, js_s, z[js_s])
    want = float(ref["gap"])
    print(f"  {len(s_scored)} scored files in ARM S: {', '.join(s_scored)}")
    print(f"  GLS from the big matrix {gh_s:+.10f}   w63a's stored gap {want:+.10f}"
          f"   |d| {abs(gh_s-want):.3e}")
    if abs(gh_s - want) > 1e-6:
        fail("the submatrix identity is FALSE -- every sweep number below is meaningless")
        print("\n⛔ GATE B FAILED. REFUSING.")
        sys.exit(1)
    print("  ✅ GATE B PASSED (Q1). Sub-designs can be read off the big matrix exactly.")

    # ---------------------------------------------------------------------------- Q2, the live one
    lo_s, hi_s = float(z[js_s].min()), float(z[js_s].max())
    print(f"\n=== Q2: is the LIVE gh inside the data it averages? ===")
    print(f"  z over ARM S's 8 scored files: [{lo_s:+.2f}, {hi_s:+.2f}]e-6, mean {z[js_s].mean():+.2f}")
    print(f"  gh = {gh_s:+.2f}e-6   sum|w| = {np.abs(w_s).sum():.3f}   "
          f"min w {w_s.min():+.3f}  max w {w_s.max():+.3f}")
    q2 = bool(lo_s <= gh_s <= hi_s)
    if q2:
        print("  ✅ Q2 CONFIRMED — the live estimate is inside its own hull. The pathology is a "
              "property of\n     the ENLARGED design, not of the artefact the sender reads today.")
    else:
        falsify("Q2", "THE LIVE BAR ALREADY RESTS ON AN OUT-OF-HULL GLS ESTIMATE. This is an "
                      "operational finding about a live artefact.")

    # ---------------------------------------------------------------------------- Q3/Q4 sweep
    order = sorted([k for k in NAMES if k in lb_all], key=lambda k: -cv[k])   # fixed, mechanical
    print(f"\n=== Q3/Q4: add scored files in DESCENDING CV, {len(order)} of them ===")
    print(f"  {'k':>4s} {'gh':>12s} {'[min z':>9s} {'max z]':>9s} {'in hull':>8s} "
          f"{'sum|w|':>10s} {'min w':>9s} {'cond':>10s} {'sqrt(Vg)':>9s}")
    recs, kstar = [], None
    for k in range(2, len(order) + 1):
        js = [col[s] for s in order[:k]]
        zk = z[js]
        g, Vg, w = gls(Sxx, js, zk)
        cn = float(np.linalg.cond(Sxx[np.ix_(js, js)]))
        inh = bool(zk.min() <= g <= zk.max())
        recs.append(dict(k=k, gh=g, zmin=float(zk.min()), zmax=float(zk.max()), in_hull=inh,
                         l1=float(np.abs(w).sum()), wmin=float(w.min()), wmax=float(w.max()),
                         cond=cn, sd_gap=float(np.sqrt(Vg)), added=order[k - 1],
                         ols=float(zk.mean())))
        if kstar is None and not inh:
            kstar = k
        if k <= 12 or k % 10 == 0 or k == len(order) or k in (kstar, (kstar or 0) - 1):
            print(f"  {k:4d} {g:+12.2f} {zk.min():+9.2f} {zk.max():+9.2f} {str(inh):>8s} "
                  f"{np.abs(w).sum():10.3f} {w.min():+9.3f} {cn:10.2e} {np.sqrt(Vg):9.3f}"
                  f"   +{order[k-1]}")
    df = pd.DataFrame(recs)
    q3 = kstar is not None and 8 < kstar <= len(order)
    print(f"\n  first k with gh OUTSIDE [min z, max z]: {kstar}"
          f"  (added {order[kstar-1] if kstar else '-'})")
    if q3:
        print("  ✅ Q3 CONFIRMED — the estimator is fine small and breaks as the design grows.")
    elif kstar is None:
        falsify("Q3", "gh never leaves the hull in this ordering, so out-of-hull is not the "
                      "mechanism and the -4598 must come from somewhere else")
    else:
        falsify("Q3", f"gh is ALREADY out of hull at k = {kstar} <= 8, so the pathology is not "
                      "specific to the enlarged design")

    l1_8 = float(df.loc[df.k == 8, "l1"].iloc[0]) if (df.k == 8).any() else float("nan")
    l1_max = float(df["l1"].iloc[-1])
    cond_star = float(df.loc[df.k == kstar, "cond"].iloc[0]) if kstar else float("nan")
    q4 = bool(l1_max > 10 * l1_8 and cond_star > 1e6)
    print(f"\n=== Q4: sum|w| {l1_8:.3f} at k=8 -> {l1_max:.3f} at k={len(order)} "
          f"({l1_max/l1_8:.1f}x);  cond at k* = {cond_star:.2e}")
    if q4:
        print("  ✅ Q4 CONFIRMED — negative weights of growing magnitude on an ill-conditioned\n"
              "     estimated covariance. That is the whole mechanism.")
    else:
        falsify("Q4", f"sum|w| grew {l1_max/l1_8:.1f}x and cond at k* is {cond_star:.1e}; the "
                      "registered mechanism does not account for it")

    # ---------------------------------------------------------------------------- Q5, the rescue
    print("\n=== Q5: the equal-weight alternative, MEASURED AND NOT ADOPTED ===")
    js_L = [col[s] for s in order]
    print(f"  full design k = {len(order)}:  GLS {df['gh'].iloc[-1]:+.2f}e-6   "
          f"OLS mean(z) {z[js_L].mean():+.2f}e-6   hull [{z[js_L].min():+.2f}, {z[js_L].max():+.2f}]")
    print(f"  ARM S      k = {len(js_s)}:   GLS {gh_s:+.2f}e-6   OLS {z[js_s].mean():+.2f}e-6")
    print(f"  the two estimators differ by {abs(gh_s - z[js_s].mean()):.2f}e-6 on the LIVE design "
          f"and by {abs(df['gh'].iloc[-1] - z[js_L].mean()):.2f}e-6 on the enlarged one")
    print("  ⛔ NOT ADOPTED. Swapping the estimator inside `fit` moves the live bar, and the run\n"
          "     that finds a defect does not also get to ship the fix (w59a). Registered for the\n"
          "     successor, which must pre-register the swap and re-derive the bar through w63a.")

    out = dict(prereg="experiments/w77_prereg_b.txt @ e475ffb", post_hoc=True,
               n_design=len(NAMES), n_scored=len(order),
               gate_b=dict(passed=True, gh_from_submatrix=gh_s, w63a_gap=want,
                           dev=abs(gh_s - want)),
               live=dict(k=len(js_s), gh=gh_s, zmin=lo_s, zmax=hi_s, in_hull=q2,
                         l1=float(np.abs(w_s).sum()), ols=float(z[js_s].mean())),
               kstar=kstar, kstar_added=(order[kstar - 1] if kstar else None),
               l1_at_8=l1_8, l1_at_max=l1_max, cond_at_kstar=cond_star,
               full=dict(gh=float(df["gh"].iloc[-1]), ols=float(z[js_L].mean()),
                         zmin=float(z[js_L].min()), zmax=float(z[js_L].max())),
               predictions=dict(q1=True, q2=q2, q3=bool(q3), q4=q4),
               # ⚠ THE FIELD THAT STOPS w77a's P6 BEING READ AS A FINDING. w77a reported the
               # ARM L cost curve crossing `base` eleven times. It does not. `fit` gives an
               # unscored file xh0 = 0 and a scored one xh0 = z - gh, so a gh that is 5,634e-6
               # wrong opens a crevasse of that size between the two groups, and ARM L's rows
               # alternate between them: every ARM L unscored row reads only() ~ -496 and every
               # scored one +1.5..+5.6. w77d, holding the GLS fit at w63a's eight files, gets
               # ONE crossing in both readings. P6 is an INSTRUMENT FAILURE, not a result.
               p6_void=True,
               p6_void_reason=("w77a's ARM L P6 (11 crossings) is an artefact of the "
                               "scored/unscored xh0 crevasse opened by the pathological gh, "
                               "NOT evidence that the cost curve is non-monotone; w77d gets "
                               "exactly one crossing with the GLS fit held fixed"),
               falsified=FALSIFIED, failures=FAILURES, writes_a_bar=False)
    json.dump(out, open(os.path.join(HERE, "w77c_glssweep.json"), "w"), indent=1)
    df.to_csv(os.path.join(HERE, "w77c_glssweep.csv"), index=False)
    print(f"\nwrote w77c_glssweep.json / .csv ({len(df)} sweep rows)")
    print(f"\nFAILURES {FAILURES} · FALSIFIED {FALSIFIED or 'none'}")


if __name__ == "__main__":
    main()
