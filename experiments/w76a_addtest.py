"""w76a — the REFINED P5 of w63 §4, tested OUT OF SAMPLE on a population that has power.

Pre-registration: experiments/w76_prereg.txt, committed 97486e1 BEFORE this file existed.

THE CLAIM (w63 §4, verbatim):
    "the additive price is optimistic iff the two candidates fall on the SAME side of the
     status quo, and pessimistic iff they straddle it"

w63a wrote that partition AFTER its strict P5 was falsified, on one 15-file design, and the
cell that distinguishes it from the simpler rival ("optimistic iff NEITHER helps") held ONE
observation. This file re-asks it on the live 08-24 board over a mechanically-defined
population, with the decisive cell populated.

⚠ THE ESTIMATOR IS IMPORTED, NEVER RE-IMPLEMENTED. `fit` and `_load` come from w63a itself, so
a drift in the pricer cannot leave this test agreeing with a version of it that no longer runs.
GATE A is a COMPARISON, not a stamp (w62's lesson): it rebuilds w63a's OWN 08-23 design and
re-derives its stored numbers rather than reading a `passed: true` field.

⛔ THIS FILE NEVER WRITES w63a_setprice.json AND NEVER CALLS w63a.main(). That artefact is the
live send bar.

    .venv/bin/python experiments/w76a_addtest.py             # ~3 min, writes only its own two files
    .venv/bin/python experiments/w76a_addtest.py --no-write  # what-if
"""
from __future__ import annotations

import io, itertools, json, os, subprocess, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import SUB, TARGET, load_raw                       # noqa: E402
import w63a_setprice as W63A                                   # noqa: E402 — the instrument itself

COMP = "playground-series-s6e8"
U = 1e-6
TIE = 1e-9          # |gap| at or below this is a TIE and counts against neither side (prereg)
N_TOPCV = 24        # prereg: the design takes the 24 highest-CV stems with a stored OOF
RESOLUTION_BAR = 0.01   # prereg P4, in e-6

FAILURES, FALSIFIED = 0, []


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


# ----------------------------------------------------------------------------------------------
# The three quantities under test, written ONCE and used for both designs. These are w63a's own
# definitions (w63a_setprice.py:330 `only`, :455 `add`/`joint`); GATE A is what proves it.
# ----------------------------------------------------------------------------------------------
def quantities(price, TIER1):
    base = price(sorted(TIER1))

    def only(x):
        return float(np.mean([price([x, d]) for d in TIER1 if d != x]))

    def gap(x, y, ox=None, oy=None):
        ox = only(x) if ox is None else ox
        oy = only(y) if oy is None else oy
        return float(price([x, y]) - (ox + oy - base))

    return base, only, gap


def board(exclude=()):
    """The live board, optionally with named stems removed to reconstruct an earlier one."""
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert len(sub) < 500, "hit the page size -- the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree -- scores are not deterministic"
    lb = {k: float(v) for k, v in agg["max"].items() if k not in set(exclude)}
    top = sorted(set(lb.values()), reverse=True)
    t1 = sorted([k for k, v in lb.items() if v == top[0]])
    t2 = sorted([k for k, v in lb.items() if v == top[1]])
    return lb, t1, t2, top[0], top[1]


def partition(pairs, helps):
    """The registered three cells. `same side` == helps(X) == helps(Y)."""
    cells = {"both help": [], "exactly one helps": [], "neither helps": []}
    for d in pairs:
        hx, hy = helps[d["x"]], helps[d["y"]]
        nm = "both help" if (hx and hy) else ("neither helps" if not (hx or hy)
                                              else "exactly one helps")
        cells[nm].append(d)
    return cells


def report(cells) -> None:
    print(f"  {'cell':22s} {'n':>5s} {'gap>0':>9s} {'gap<0':>9s} {'ties':>6s} "
          f"{'min|gap|':>10s} {'mean gap':>10s}")
    for nm in ("both help", "exactly one helps", "neither helps"):
        g = np.array([d["gap"] for d in cells[nm]])
        if not g.size:
            print(f"  {nm:22s} {0:5d}   -- EMPTY --")
            continue
        pos, neg = int((g > TIE).sum()), int((g < -TIE).sum())
        print(f"  {nm:22s} {g.size:5d} {pos:9d} {neg:9d} {int(g.size-pos-neg):6d} "
              f"{np.abs(g).min()/1:10.4f} {g.mean():+10.4f}")


# ----------------------------------------------------------------------------------------------
def gate_a(y, beta):
    """P1 — rebuild w63a's OWN 08-23 design and re-derive its stored numbers. Comparison, not stamp."""
    print("=" * 96)
    print("GATE A (P1) — reproduce w63a_setprice.json's design, prices and P5 partition")
    print("=" * 96)
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    T1, T2 = list(ref["tiers"]["slot1"]), list(ref["tiers"]["slot2"])
    plan = list(ref["plan_0824"])
    CAND = [c for c in plan if c not in T1]
    NAMES = sorted(set(T1 + T2 + list(ref["wanted"]) + CAND))

    # w63a ran BEFORE the 08-24 window, so its board is the live one minus exactly that day's ten.
    lb, lt1, lt2, _, _ = board(exclude=plan)
    LB = {k: v for k, v in lb.items() if k in NAMES}
    if sorted(lt1) != sorted(T1):
        fail(f"reconstructed tier1 {sorted(lt1)} != w63a's {sorted(T1)} -- the board did not rewind")
    print(f"  reconstructed 08-23 board: tier1 {lt1}\n  {len(NAMES)} design files, "
          f"{sum(k in LB for k in NAMES)} of them scored then ({len(CAND)} candidates unscored)")

    V, cv = W63A._load(NAMES, y)
    E = W63A.fit(NAMES, LB, cv, V, y, beta)
    base, only, gap = quantities(E["price"], T1)

    worst, nchk = 0.0, 0

    def cmp(label, got, want, bar=1e-9):
        nonlocal worst, nchk
        nchk += 1
        worst = max(worst, abs(got - want))
        if abs(got - want) > bar:
            fail(f"GATE A: {label} {got:+.12f} != w63a's {want:+.12f}")

    cmp("base", base, float(ref["base"]))
    ou = {}
    for r in ref["rows"]:
        ou[r["stem"]] = only(r["stem"])
        cmp(f"only[{r['stem']}]", ou[r["stem"]], float(r["uncond"]))

    helps = {r["stem"]: bool(r["helps"]) for r in ref["rows"]}
    for s, v in helps.items():                 # the classifier itself must reproduce, not be trusted
        if bool(ou[s] < base) != v:
            fail(f"GATE A: helps[{s}] recomputes to {ou[s] < base}, artefact says {v}")

    # w63a's pair set: combinations of CAND + TIER2, ordered by -cv. Order does not change the set.
    pairs = []
    for x, yq in itertools.combinations(sorted(set(CAND + T2), key=lambda k: -cv[k]), 2):
        pairs.append(dict(x=x, y=yq, gap=gap(x, yq, ou[x], ou[yq])))
    g = np.array([d["gap"] for d in pairs])
    na = ref["nonadditivity"]
    cmp("n_pairs", float(len(pairs)), float(na["n_pairs"]), 0.0)
    cmp("n_joint_gt_add", float((g > 0).sum()), float(na["n_joint_gt_add"]), 0.0)
    cmp("mean_gap", float(g.mean()), float(na["mean_gap"]))
    cmp("min_gap", float(g.min()), float(na["min_gap"]))
    cmp("max_gap", float(g.max()), float(na["max_gap"]))

    cells = partition(pairs, helps)
    print(f"\n  w63a's post-hoc partition, re-derived:")
    report(cells)
    for nm, n_want, pos_want in (("neither helps", 78, 78), ("exactly one helps", 26, 0),
                                 ("both help", 1, 1)):
        gg = np.array([d["gap"] for d in cells[nm]])
        cmp(f"cell[{nm}].n", float(gg.size), float(n_want), 0.0)
        cmp(f"cell[{nm}].gap>0", float((gg > 0).sum()), float(pos_want), 0.0)

    print(f"\n  {nchk} recorded quantities re-derived; worst absolute deviation {worst:.3e}  (bar 1e-9)")
    if FAILURES:
        print("\n⛔ GATE A FAILED — this is not w63a's instrument. Claiming nothing. REFUSING.")
        sys.exit(1)
    print("  ✅ GATE A PASSED (P1). The only/add/gap below are w63a's, on a board it never saw.")
    return worst, nchk


def main() -> None:
    write = "--no-write" not in sys.argv
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    worst_a, nchk_a = gate_a(y, beta)

    # ------------------------------------------------------------------ THE FRESH DESIGN
    # The population rule is w76_prereg's, fixed before the board was read.
    print("\n" + "=" * 96)
    print("THE FRESH DESIGN — the live 08-24 board, population fixed in the prereg")
    print("=" * 96)
    LB, T1, T2, t1v, t2v = board()
    have = {f[4:-4] for f in os.listdir(SUB) if f.startswith("oof_") and f.endswith(".npy")}
    scored_with_oof = [k for k in LB if k in have]

    # CV is recomputed from the stored OOF vector, never parsed from a description (w75 §5).
    from w16b_cellweight import fast_auc
    cvs = {}
    for k in scored_with_oof:
        cvs[k] = fast_auc(y, np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64))
    TOPCV = sorted(cvs, key=lambda k: -cvs[k])[:N_TOPCV]
    NAMES = sorted(set(T1 + T2 + list(W63A.WANTED) + TOPCV))
    miss = [k for k in NAMES if k not in have]
    assert not miss, f"no OOF vector for {miss} -- do not guess"
    print(f"  auto tier public {t1v:.5f}: {len(T1)} files {T1}")
    print(f"  next tier  public {t2v:.5f}: {len(T2)} files")
    print(f"  {len(scored_with_oof)} scored stems have an OOF; top {N_TOPCV} by CV taken")
    print(f"  design = {len(NAMES)} files (was 15 in w63a)")
    assert len(T1) == 2, f"tier 1 holds {len(T1)} files -- the determined-pair pricer does not apply"

    V, cv = W63A._load(NAMES, y)
    E = W63A.fit(NAMES, LB, cv, V, y, beta)
    base, only, gap = quantities(E["price"], T1)
    print(f"  GLS common gap {E['gap']:+.2f}e-6, gamma {E['gamma']:+.6f}, base {base:+.4f}e-6")

    cand = sorted([k for k in NAMES if k not in T1], key=lambda k: -cv[k])
    ou = {k: only(k) for k in cand}
    helps = {k: bool(ou[k] < base) for k in cand}
    print(f"  {sum(helps.values())} of {len(cand)} design files HELP (only(X) < base)")

    pairs = [dict(x=x, y=yq, only_x=ou[x], only_y=ou[yq], gap=gap(x, yq, ou[x], ou[yq]))
             for x, yq in itertools.combinations(cand, 2)]
    cells = partition(pairs, helps)
    print(f"\n  {len(pairs)} pairs on the fresh design:")
    report(cells)

    # ------------------------------------------------------------------ P2 POWER
    n_both = len(cells["both help"])
    p2 = n_both >= 10
    print(f"\nP2 POWER — the decisive 'both help' cell holds {n_both} pairs (bar 10, w63a had 1): "
          f"{'PASS' if p2 else 'UNDERPOWERED'}")
    if not p2:
        FALSIFIED.append("P2 the decisive cell is underpowered")

    # ------------------------------------------------------------------ P3 THE CLAIM
    same = cells["both help"] + cells["neither helps"]
    strad = cells["exactly one helps"]
    bad_same = [d for d in same if d["gap"] < -TIE]
    bad_strad = [d for d in strad if d["gap"] > TIE]
    p3 = not bad_same and not bad_strad
    print(f"\nP3 THE REGISTERED CLAIM, OUT OF SAMPLE — optimistic iff SAME side, pessimistic iff STRADDLE")
    print(f"  same side  n {len(same):4d}   counterexamples (gap < 0) {len(bad_same)}")
    print(f"  straddle   n {len(strad):4d}   counterexamples (gap > 0) {len(bad_strad)}")
    for d in (bad_same + bad_strad)[:8]:
        print(f"    counterexample {d['x']} + {d['y']}  gap {d['gap']:+.6f}e-6")
    print(f"  ==> P3 {'CONFIRMED' if p3 else 'FALSIFIED'}")
    if not p3:
        FALSIFIED.append("P3 the same-side/straddle iff")

    # ------------------------------------------------------------------ P4 RESOLUTION
    gs = np.abs([d["gap"] for d in strad]) if strad else np.array([np.nan])
    p4 = bool(strad) and float(gs.min()) > RESOLUTION_BAR
    print(f"\nP4 RESOLUTION — min|gap| in the straddling cell {gs.min():.6f}e-6 "
          f"(bar {RESOLUTION_BAR}e-6): {'PASS' if p4 else 'FAIL'}")
    if not p4:
        FALSIFIED.append("P4 the straddling sign is resolvable")

    # ------------------------------------------------------------------ P5 THE RIVAL
    gb = np.array([d["gap"] for d in cells["both help"]])
    both_pos = bool(gb.size) and bool((gb > TIE).all())
    p5 = both_pos
    print(f"\nP5 THE RIVAL SEPARATED — 'optimistic iff NEITHER helps' predicts the 'both help' cell")
    print(f"  is NEGATIVE; the registered rule predicts POSITIVE.")
    if gb.size:
        print(f"  both-help gaps: n {gb.size}  min {gb.min():+.4f}  mean {gb.mean():+.4f}  "
              f"max {gb.max():+.4f} e-6  ->  {'REGISTERED RULE' if both_pos else 'RIVAL'} wins")
    else:
        print("  cell EMPTY -- the two rules are indistinguishable on this design")
    if not p5:
        FALSIFIED.append("P5 the both-help cell is positive (the registered rule's distinctive half)")

    # ------------------------------------------------------------------ what it is worth
    # The reason anyone cares: the additive price is what w60 applied twice and got wrong by
    # 5.5e-6. Quantify the error the rule predicts the sign of, on this design.
    print("\n=== THE SIZE OF THE THING WHOSE SIGN IS BEING PREDICTED ===")
    for nm in ("both help", "exactly one helps", "neither helps"):
        g = np.array([d["gap"] for d in cells[nm]])
        if g.size:
            print(f"  {nm:22s} |gap| median {np.median(np.abs(g)):8.4f}e-6   max {np.abs(g).max():8.4f}e-6")

    print("\n" + "=" * 96)
    print(f"FAILURES {FAILURES}   FALSIFIED {FALSIFIED if FALSIFIED else 'none'}")
    print("=" * 96)

    out = dict(
        prereg="experiments/w76_prereg.txt (97486e1)",
        gate_a=dict(passed=True, worst=worst_a, n_checks=nchk_a, against="w63a_setprice.json"),
        design=dict(names=NAMES, n=len(NAMES), tier1=T1, n_tier2=len(T2), top_cv=N_TOPCV,
                    tier1_public=t1v, base=base, gls_gap=E["gap"], gamma=E["gamma"]),
        helps={k: helps[k] for k in cand},
        cells={nm: dict(n=len(v), n_pos=int(sum(d["gap"] > TIE for d in v)),
                        n_neg=int(sum(d["gap"] < -TIE for d in v)),
                        min_gap=float(min(d["gap"] for d in v)) if v else None,
                        mean_gap=float(np.mean([d["gap"] for d in v])) if v else None,
                        max_gap=float(max(d["gap"] for d in v)) if v else None)
               for nm, v in cells.items()},
        predictions=dict(p1=True, p2=bool(p2), p3=bool(p3), p4=bool(p4), p5=bool(p5)),
        falsified=FALSIFIED,
        note=("tests w63 §4's refined P5 out of sample. w63a's own partition was post-hoc with "
              "n=1 in the decisive cell; this design populates it. Nothing here feeds the "
              "sender: the live bar is w63a_setprice.json, untouched."),
    )
    if write:
        with open(os.path.join(HERE, "w76a_addtest.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
        pd.DataFrame(pairs).to_csv(os.path.join(HERE, "w76a_addtest.csv"), index=False)
        print("wrote experiments/w76a_addtest.json and .csv")
    # The write rule is bound to GATE A alone (prereg): a falsification must not be punished.


if __name__ == "__main__":
    main()
