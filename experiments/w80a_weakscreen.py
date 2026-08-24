"""w80a -- screen the szymonkapiski "50 weakest OOF models" pack (data/ext_members17).

Pre-registration: experiments/w80_prereg.txt, commit 88fbaf1, written before this file
existed. Every threshold below is READ from that file's text, not re-typed from memory,
and the reader is a gate: if the prereg is edited the constants move with it and the
assertions here fail loudly rather than drifting silently.

⚠ THIS SCREEN IS POWERED TO KILL, NOT TO ADOPT. w38c's gate 4 (foreign-partition test) is
unrunnable on this pack -- the author publishes solo overall AUC per column and nothing per
fold, and withholds the recipes while the competition runs. w38c's rule "a declaration is
not a measurement" therefore binds and cannot be discharged. No branch of the decision rule
adopts anything; the best available outcome is "worth the full vetting a later run owes".
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "data", "ext_members17")
PREREG = os.path.join(ROOT, "experiments", "w80_prereg.txt")
OUT = os.path.join(ROOT, "experiments", "w80a_weakscreen.json")
OUT_POSTHOC = os.path.join(ROOT, "experiments", "w80a_posthoc.json")

N_TR, N_TE, N_COL = 691369, 296302, 50

BASE_A = os.path.join(ROOT, "submissions", "oof_w36_ad199stdcorr.npy")
BASE_B = os.path.join(ROOT, "submissions", "oof_w40_ad211std_h3.npy")

# P6: the send path. Nothing here writes to any of these; the md5s prove it.
SEND_PATH = [
    os.path.join(ROOT, "experiments", "w79a_barfill.json"),
    os.path.join(ROOT, "experiments", "w63a_setprice.json"),
    os.path.join(ROOT, "experiments", "w26d_queueprice.csv"),
]

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  ❌ {msg}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def read_prereg() -> dict:
    """Pull the registered constants OUT of the prereg text. Never re-type a threshold."""
    txt = open(PREREG).read()
    got = {}
    pats = {
        "p2_tol":    r"max_j \|roc_auc_score\(y, oof\[:,j\]\) - members\.csv\[j\]\| < ([\d.e+-]+)",
        "p3_bound":  r"xfit_cv\(\[BASE_A\]\)\] < \+([\d.e+-]+)e-6",
        "p4_point":  r"Registered point prediction: G50 < \+([\d.e+-]+)e-6",
        "p4_refuse": r"G50 > \+([\d.e+-]+)e-6\s+==>\s+REFUSE THE PACK",
        "p5_bound":  r"G50_null < \+([\d.e+-]+)e-6",
        "floor":     r"FLOOR  = ([\d.e+-]+)e-6",
        "seed":      r"SEED   = (\d+)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        assert m, f"prereg no longer states {k} -- refusing to guess it"
        got[k] = float(m.group(1))
    got["seed"] = int(got["seed"])
    for k in ("p3_bound", "p4_point", "p4_refuse", "p5_bound", "floor"):
        got[k] *= 1e-6
    return got


def rank01(v: np.ndarray) -> np.ndarray:
    return (rankdata(v) - 0.5) / v.size


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def xfit_cv(cols, y, folds) -> float:
    """w56a.xfit_cv, verbatim. Cross-fitted logistic stack; pooled OOF AUC on frozen folds."""
    X = np.column_stack([logit(rank01(c)) for c in cols])
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m.fit(X[tr], y[tr])
        oof[va] = m.decision_function(X[va])
    return roc_auc_score(y, oof)


def main(posthoc: bool = False) -> dict:
    global FAILURES
    FAILURES = 0
    reg = read_prereg()
    print("=" * 78)
    print("w80a -- SCREEN: szymonkapiski 50-weakest OOF pack (ext_members17)")
    print("=" * 78)
    print("  constants read from w80_prereg.txt, not re-typed:")
    for k, v in reg.items():
        print(f"    {k:10s} {v}")

    before = {p: md5(p) for p in SEND_PATH if os.path.exists(p)}

    tr, te = common.load_raw()
    y = tr[common.TARGET].astype(int).to_numpy()
    assert len(y) == N_TR and len(te) == N_TE
    folds = list(common.get_folds(y))

    # ---------------------------------------------------------------- P1
    print("\n" + "=" * 78)
    print("P1 (GATING) -- shape, dtype, finiteness, range")
    print("=" * 78)
    rng_bad: dict = {}
    O = np.load(os.path.join(PACK, "oof.npy"))
    T = np.load(os.path.join(PACK, "test.npy"))
    p1 = True
    for nm, A, shape in (("oof", O, (N_TR, N_COL)), ("test", T, (N_TE, N_COL))):
        if A.shape != shape:
            fail(f"{nm}.npy shape {A.shape}, expected {shape}"); p1 = False; continue
        if A.dtype != np.float64:
            fail(f"{nm}.npy dtype {A.dtype}, expected float64"); p1 = False; continue
        if not np.isfinite(A).all():
            fail(f"{nm}.npy has non-finite values"); p1 = False; continue
        lo, hi = float(A.min()), float(A.max())
        if not (0.0 <= lo and hi <= 1.0):
            fail(f"{nm}.npy range [{lo!r}, {hi!r}] leaves [0,1] by "
                 f"{max(0.0 - lo, hi - 1.0):.3e}"); p1 = False
            rng_bad[nm] = [lo, hi]; continue
        ok(f"{nm}.npy {A.shape} float64, finite, range [{lo:.6f}, {hi:.6f}]")
    print(f"  P1 {'✅ CONFIRMED' if p1 else '❌ FALSIFIED'}")
    if not p1 and not posthoc:
        return finish(reg, before, dict(p1=False, range_violation=rng_bad),
                      "REFUSE -- malformed (P1)")
    if not p1:
        print()
        print("!" * 78)
        print("!!  POST-HOC ARM. P1 IS FALSIFIED AND THE REGISTERED RULE SAYS REFUSE-AND-STOP.")
        print("!!  That verdict stands: it is in w80a_weakscreen.json and is NOT amended.")
        print("!!")
        print("!!  This arm continues anyway because the falsified clause does not bind the")
        print("!!  measurement -- P2..P5 read oof.npy only and never touch test.npy -- and")
        print("!!  because the overshoot is 1.5e-8 on a quantity AUC cannot see. That is a")
        print("!!  judgement made AFTER seeing the number, which is exactly what a prereg")
        print("!!  exists to stop, so the output goes to its own file and NOTHING here may")
        print("!!  be quoted as a registered finding. It licenses no adoption -- and could")
        print("!!  not have, since w38c gate 4 is unrunnable on this pack either way.")
        print("!" * 78)

    # ---------------------------------------------------------------- P2
    print("\n" + "=" * 78)
    print("P2 (GATING) -- row alignment: recompute every solo AUC against members.csv")
    print("=" * 78)
    mem = pd.read_csv(os.path.join(PACK, "members.csv"))
    assert len(mem) == N_COL and list(mem.columns) == ["id", "solo_oof_auc"]
    solo = np.array([roc_auc_score(y, O[:, j]) for j in range(N_COL)])
    claimed = mem["solo_oof_auc"].to_numpy()
    d = np.abs(solo - claimed)
    jmax = int(np.argmax(d))
    print(f"  worst column   : {mem['id'][jmax]}  ours {solo[jmax]:.9f}  "
          f"claimed {claimed[jmax]:.6f}  |d| {d[jmax]:.3e}")
    print(f"  median |d|     : {np.median(d):.3e}")
    print(f"  registered tol : {reg['p2_tol']:.1e}")
    p2 = bool(d.max() < reg["p2_tol"])
    (ok if p2 else fail)(f"P2 max |d| = {d.max():.3e}")
    print(f"  P2 {'✅ CONFIRMED' if p2 else '❌ FALSIFIED'}")
    if not p2:
        return finish(reg, before, dict(p1=True, p2=False, p2_maxd=float(d.max())),
                      "REFUSE -- misaligned (P2)", posthoc)

    # ---------------------------------------------------------------- baselines
    print("\n" + "=" * 78)
    print("BASELINES -- the cross-fitted 1-column controls")
    print("=" * 78)
    bA = np.load(BASE_A)
    bB = np.load(BASE_B)
    assert bA.shape == y.shape and bB.shape == y.shape
    cv0A = xfit_cv([bA], y, folds)
    cv0B = xfit_cv([bB], y, folds)
    print(f"  BASE_A w36_ad199stdcorr  standalone {roc_auc_score(y, bA):.10f}   "
          f"xfit 1-col {cv0A:.10f}")
    print(f"  BASE_B w40_ad211std_h3   standalone {roc_auc_score(y, bB):.10f}   "
          f"xfit 1-col {cv0B:.10f}")

    # ---------------------------------------------------------------- P3
    print("\n" + "=" * 78)
    print("P3 (RECORDED) -- best single weak column against BASE_A")
    print("=" * 78)
    per = np.array([xfit_cv([bA, O[:, j]], y, folds) - cv0A for j in range(N_COL)])
    order = np.argsort(-per)
    for j in order[:5]:
        print(f"    {mem['id'][j]}  solo {solo[j]:.6f}   marginal {per[j] * 1e6:+8.2f}e-6")
    print(f"    ...")
    for j in order[-2:]:
        print(f"    {mem['id'][j]}  solo {solo[j]:.6f}   marginal {per[j] * 1e6:+8.2f}e-6")
    best = float(per.max())
    p3 = bool(best < reg["p3_bound"])
    print(f"\n  best single column : {best * 1e6:+.2f}e-6   "
          f"(registered bound < {reg['p3_bound'] * 1e6:+.0f}e-6)")
    print(f"  w56a's hboyang_mix, for scale                     :   +89.32e-6")
    print(f"  P3 {'✅ CONFIRMED' if p3 else '❌ FALSIFIED'}  (recorded, not gating)")

    # ---------------------------------------------------------------- P4
    print("\n" + "=" * 78)
    print("P4 (GATING) -- the full 50-column joint stack. LARGE IS THE BAD BRANCH.")
    print("=" * 78)
    colsA = [bA] + [O[:, j] for j in range(N_COL)]
    cv50A = xfit_cv(colsA, y, folds)
    g50 = cv50A - cv0A
    cv50B = xfit_cv([bB] + [O[:, j] for j in range(N_COL)], y, folds)
    g50B = cv50B - cv0B
    print(f"  BASE_A + 50 : {cv50A:.10f}   G50   = {g50 * 1e6:+.2f}e-6")
    print(f"  BASE_B + 50 : {cv50B:.10f}   G50_B = {g50B * 1e6:+.2f}e-6")
    print(f"  registered point prediction : G50 < {reg['p4_point'] * 1e6:+.0f}e-6")
    print(f"  registered REFUSE cut       : G50 > {reg['p4_refuse'] * 1e6:+.0f}e-6"
          f"  (a large G50 is a leak signature, not value)")
    p4 = bool(g50 < reg["p4_point"])
    print(f"  P4 {'✅ CONFIRMED' if p4 else '❌ FALSIFIED'}")

    # ---------------------------------------------------------------- P5
    print("\n" + "=" * 78)
    print("P5 (GATING) -- NULL CONTROL: same marginals, row alignment destroyed")
    print("=" * 78)
    rng = np.random.default_rng(reg["seed"])
    shuf = [O[rng.permutation(N_TR), j] for j in range(N_COL)]
    cv50n = xfit_cv([bA] + shuf, y, folds)
    g50n = cv50n - cv0A
    print(f"  BASE_A + 50 permuted : {cv50n:.10f}   G50_null = {g50n * 1e6:+.2f}e-6")
    print(f"  registered bound     : G50_null < {reg['p5_bound'] * 1e6:+.0f}e-6")
    p5 = bool(g50n < reg["p5_bound"])
    (ok if p5 else fail)(f"null control {g50n * 1e6:+.2f}e-6")
    print(f"  P5 {'✅ CONFIRMED' if p5 else '❌ FALSIFIED'}")
    print(f"\n  ⇒ signal-to-null ratio: G50 / |G50_null| = "
          f"{g50 / abs(g50n) if g50n else float('inf'):.1f}x")

    # ------------------------------------------------- the registered decision rule
    print("\n" + "=" * 78)
    print("DECISION -- the prereg's rule, applied in its registered order")
    print("=" * 78)
    if not p5:
        verdict = "VOID -- the instrument gains on a null; P3/P4 mean nothing"
    elif g50 > reg["p4_refuse"]:
        verdict = "REFUSE -- CONTAMINATION SUSPECTED (G50 over the registered cut)"
    elif g50 < reg["floor"]:
        verdict = "REFUSE -- inside the rebuild floor; worthless to this stack"
    else:
        verdict = "SURVIVES SCREENING -- worth full vetting. NOT ADOPTED (gate 4 unrunnable)."
    print(f"  {verdict}")
    print("\n  ⛔ No branch of this rule adopts the pack, edits the queue, touches a build,")
    print("     or sends. Zero submission slots exist today and none was attempted.")

    stats = dict(p1=p1, range_violation=rng_bad, p2=p2, p3=p3, p4=p4, p5=p5,
                 p2_maxd=float(d.max()), cv0A=cv0A, cv0B=cv0B,
                 cv50A=cv50A, cv50B=cv50B, g50=float(g50), g50B=float(g50B),
                 g50_null=float(g50n), best_single=best,
                 best_single_id=str(mem["id"][int(order[0])]),
                 solo_min=float(solo.min()), solo_max=float(solo.max()),
                 per_column={str(mem["id"][j]): float(per[j]) for j in range(N_COL)})
    return finish(reg, before, stats, verdict, posthoc)


def finish(reg, before, stats, verdict, posthoc: bool = False) -> dict:
    print("\n" + "=" * 78)
    print("P6 (GATING) -- the send path is byte-identical")
    print("=" * 78)
    p6 = True
    for p, h in before.items():
        now = md5(p)
        if now != h:
            fail(f"{os.path.basename(p)} CHANGED {h} -> {now}"); p6 = False
        else:
            ok(f"{os.path.basename(p)} unchanged  {h}")
    stats["p6"] = p6
    print(f"  P6 {'✅ CONFIRMED' if p6 else '❌ FALSIFIED'}")

    res = dict(prereg=reg, verdict=verdict, stats=stats, failures=FAILURES)
    dest = OUT_POSTHOC if posthoc else OUT
    if posthoc:
        res["REGISTERED"] = False
        res["note"] = ("POST-HOC. P1 falsified; the registered verdict is in "
                       "w80a_weakscreen.json and stands. Not a registered finding.")
    with open(dest, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"\n  wrote {os.path.relpath(dest, ROOT)}")
    print(f"  FAILURES {FAILURES}")
    return res


if __name__ == "__main__":
    r = main(posthoc="--posthoc" in sys.argv)
    sys.exit(1 if r["failures"] else 0)
