"""Is the 53-84e-6 "cross-team noise floor" a fact about CROSS-TEAM-NESS, or about the
partner being WORSE?  The fourth wrong-population audit.  Pre-registered in w17h_prereg.txt.

WHY THIS EXISTS
---------------
RESEARCH.md line 2287 publishes the law

    sd(gap) = sd(single) * sqrt(2 * (1 - rho))                              [LAW-RHO]

a table of six measured pairs, and a LOOKUP TABLE over rho used to price teams whose
predictions we cannot see.  Two standing conclusions ride on it and sit on six do-not-spend
lists: the 18e-5 gap to public #1 is "2.2-3.4 sigma", and the 5-11e-5 gaps to the teams at
0.97113-0.97117 are "0.8-1.7 sigma -- not a difference".  The second is the reason this
workspace does not chase the top of the board.

But the six pairs were all {our best} x {a partner that is worse}, and their sd_gap is
monotone in the partner's CV DEFICIT (3/35/88/193/549/577e-6 -> 6.0/19.3/27.7/53.0/74.8/
83.9e-6).  The teams the number gets applied to have deficit ~0.  Meanwhile the stored
json already contains a counterexample to LAW-RHO itself: BOLT:rankavg_top12 sits at
rho_test 0.99736, HIGHER than OURS:blend158_logit's 0.99715, with 3.03x the sd_gap.

WHAT THIS DOES
--------------
1. GATE.  Reproduces w15a_crossteam.json's six sd_gap values at its own seed 15 / 400 reps.
   The rng stream is consumed only by one permutation per rep, so scoring EXTRA vectors on
   the same masks leaves the gate exact.
2. Builds a pair population spanning deficit and rho as independently as they can be made
   to span, including the cell the original had no members in: DISJOINT-HALF rank averages,
   which are matched in quality to each other by construction and diverse by construction.
   Pairs are taken over ALL vectors, not just against REF, which is what makes the
   deficit~0 / diversity-high cell reachable at all.
3. Tests three laws against the simulation: LAW-RHO, deficit, and

     d_i = F0^a(s_i^a) - F0^b(s_i^b)          over pool positives
     e_j = F1^b(s_j^b) - F1^a(s_j^a)          over pool negatives
     Var(gap) = Var_pos(d)/n1 + Var_neg(e)/n0                               [LAW-IF]

   the AUC influence function, which costs no simulation at all.

    w17h_floorpop.py --reps 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
EXT = os.path.join(ROOT, "data", "ext")
LIB = os.path.join(ROOT, "data", "oof")
N_TEST = 296_302
F_PUBLIC = 0.20
REF = "blend159av_h3"

# the six competitors of w15a_crossteam, in its own construction order -- the gate
GATE_ORDER = ["OURS:blend158_logit", "OURS:blend150fx_hybrid", "OURS:blend156_h3",
              "NAJI:18_blend", "NAJI:12_blend", "BOLT:rankavg_top12"]


def prep(v):
    o = np.argsort(v, kind="stable")
    return o, np.ascontiguousarray(v[o])


def _auc_sorted(s, yy):
    n = s.size
    n1 = float(yy.sum())
    n0 = float(n) - n1
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    b = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    ends = np.concatenate((b[1:], [n]))
    avg = (b + ends + 1) * 0.5
    pos = np.add.reduceat(yy, b)
    return (float((avg * pos).sum()) - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def subset_auc(o, vs, ys, mask):
    sel = mask[o]
    return _auc_sorted(vs[sel], ys[sel])


def rank01(v):
    return rankdata(v) / (len(v) + 1.0)


def midrank_cdf(sorted_ref, s):
    """F(s) with ties at midpoint, against a sorted reference sample."""
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


# ------------------------------------------------------------------ vector construction
def build_vectors(y, n_pool):
    """{name: (vector, kind)} -- everything is an OOF vector on the 691,369 labelled rows."""
    V = {}
    V["REF:" + REF] = (np.load(os.path.join(SUB, f"oof_{REF}.npy")), "ours")

    # -- w15a's own three internal controls, loaded first so the gate order is exact
    for nm in ["blend158_logit", "blend150fx_hybrid", "blend156_h3"]:
        p = os.path.join(SUB, f"oof_{nm}.npy")
        if os.path.exists(p):
            V[f"OURS:{nm}"] = (np.load(p), "ours")

    # -- najiama, then boltuzamaki: the two real other teams
    nd = os.path.join(EXT, "najiama_predicting-smartphone-addiction-oof-submission-csv")
    for tag in ["18", "12"]:
        fo = os.path.join(nd, f"{tag}_blend_oof_predictions.csv")
        if os.path.exists(fo):
            do = pd.read_csv(fo)
            col = [c for c in do.columns if c != "id"][0]
            v = do.sort_values("id")[col].to_numpy() if "id" in do.columns else do[col].to_numpy()
            if len(v) == n_pool:
                V[f"NAJI:{tag}_blend"] = (v, "foreign")

    bd = os.path.join(EXT, "boltuzamaki_s6e8-oof-prediction-library")
    fo = os.path.join(bd, "oof_predictions.parquet")
    if os.path.exists(fo):
        do = pd.read_parquet(fo)
        cols = [c for c in do.columns if c not in ("id", "row_id", "target")]
        dl = pd.read_parquet(os.path.join(bd, "train_labels.parquet"))
        y_ = dl[[c for c in dl.columns if c not in ("id", "row_id")][0]].to_numpy()
        aucs = {}
        for c in cols:
            v = do[c].to_numpy()
            if len(v) != n_pool or not np.isfinite(v).all():
                continue
            o, vs = prep(v)
            aucs[c] = _auc_sorted(vs, y_[o])
        top = sorted(aucs, key=aucs.get, reverse=True)[:12]
        V["BOLT:rankavg_top12"] = (np.mean([rank01(do[c].to_numpy()) for c in top], axis=0), "foreign")
        # extra foreign vectors from the SAME library, disjoint from the top-12 above,
        # so foreign-ness is held fixed while quality varies
        rest = sorted(aucs, key=aucs.get, reverse=True)[12:36]
        if len(rest) >= 12:
            V["BOLT:rankavg_next12"] = (np.mean([rank01(do[c].to_numpy()) for c in rest[:12]], axis=0), "foreign")
    return V


def add_our_spread(V, n_pool, k=10):
    """Our own stored files spanning CV, to fill the high-rho / varying-deficit axis."""
    have = {n.split(":", 1)[1] for n in V}
    cand = []
    for fn in sorted(os.listdir(SUB)):
        if not fn.startswith("oof_") or not fn.endswith(".npy"):
            continue
        nm = fn[4:-4]
        if nm in have:
            continue
        v = np.load(os.path.join(SUB, fn))
        if v.shape != (n_pool,) or not np.isfinite(v).all():
            continue
        cand.append((nm, v))
    # spread by pooled AUC rather than taking the top k, so deficit varies
    aucs = []
    for nm, v in cand:
        o, vs = prep(v)
        aucs.append(_auc_sorted(vs, YGLOB[o]))
    order = np.argsort(aucs)
    picks = np.unique(np.linspace(0, len(order) - 1, k).astype(int))
    for i in picks:
        nm, v = cand[order[i]]
        V[f"OURS:{nm}"] = (v, "ours")
    return V


def add_disjoint(V, y, n_pool, rng):
    """THE DECISIVE CELL.  Rank-averages over DISJOINT halves of the member pool: matched in
    quality to each other by construction, and neither is a perturbation of the other."""
    names, oofs = [], []
    for d in [os.path.join(LIB, "oof"), os.path.join(ROOT, "data", "ext_members")]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.startswith("oof_") or not fn.endswith(".npy"):
                continue
            nm = fn[4:-4]
            if nm in names:
                continue
            v = np.load(os.path.join(d, fn))
            if v.shape != (n_pool,) or not np.isfinite(v).all():
                continue
            names.append(nm)
            oofs.append(v)
    aucs = []
    for v in oofs:
        o, vs = prep(v)
        aucs.append(_auc_sorted(vs, y[o]))
    aucs = np.array(aucs)
    order = np.argsort(-aucs)              # strongest first
    print(f"  member pool for disjoint halves: {len(names)} members, "
          f"AUC {aucs.min():.5f}..{aucs.max():.5f}")

    R = {}                                  # cache rank01 per member index
    def rk(i):
        if i not in R:
            R[i] = rank01(oofs[i])
        return R[i]

    # Interleaving by strength quality-matches the halves; several seeds per size, because
    # one split is one draw and the deficit between halves is itself a random quantity.
    for size in [16, 24, 40, min(86, len(order))]:
        take = order[:size]
        if len(take) < size:
            continue
        V[f"SYN:iv{size}s0a"] = (np.mean([rk(i) for i in take[0::2]], axis=0), "synth")
        V[f"SYN:iv{size}s0b"] = (np.mean([rk(i) for i in take[1::2]], axis=0), "synth")
        p = rng.permutation(take)
        V[f"SYN:rd{size}s0a"] = (np.mean([rk(i) for i in p[: size // 2]], axis=0), "synth")
        V[f"SYN:rd{size}s0b"] = (np.mean([rk(i) for i in p[size // 2:]], axis=0), "synth")
    return V


def add_stack_halves(V, n_pool):
    """w17i's cross-fitted logistic stacks over DISJOINT member halves -- the closest thing
    available to two strong teams working the same public pool independently."""
    n = 0
    for fn in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
        if not (fn.startswith("w17i_syn_") and fn.endswith(".npy")):
            continue
        v = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn))
        if v.shape != (n_pool,) or not np.isfinite(v).all():
            print(f"  [skip] {fn}: {v.shape}")
            continue
        V[f"STK:{fn[9:-4]}"] = (v, "stackhalf")
        n += 1
    print(f"  {n} disjoint-half logistic stacks loaded from w17i")
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--gate-reps", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "w17h_floorpop.json"))
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)
    global YGLOB
    YGLOB = y

    n_pub = int(round(N_TEST * F_PUBLIC))
    print(f"pool {n_pool:,} labelled rows | public slice {n_pub:,} | "
          f"prevalence {y.mean():.6f}\n")

    V = build_vectors(y, n_pool)
    gate_names = ["REF:" + REF] + GATE_ORDER
    missing = [g for g in gate_names if g not in V]
    if missing:
        raise SystemExit(f"gate vectors missing: {missing}")
    V = add_our_spread(V, n_pool, k=10)
    V = add_disjoint(V, y, n_pool, np.random.default_rng(1717))
    V = add_stack_halves(V, n_pool)
    names = list(V)
    print(f"\n{len(names)} vectors: "
          f"{sum(1 for n in names if V[n][1]=='ours')} ours, "
          f"{sum(1 for n in names if V[n][1]=='foreign')} foreign, "
          f"{sum(1 for n in names if V[n][1]=='synth')} rank-avg halves, "
          f"{sum(1 for n in names if V[n][1]=='stackhalf')} stack halves\n")

    # ---- correctness gate on the fast AUC, the same one w15a/w14b use
    o0, vs0 = prep(V[gate_names[0]][0])
    m = np.zeros(n_pool, bool)
    m[np.random.default_rng(0).permutation(n_pool)[:50_000]] = True
    n1m = int(y[m].sum())
    n0m = int(m.sum()) - n1m
    scipy_ref = (rankdata(V[gate_names[0]][0][m])[y[m] == 1].sum() - n1m * (n1m + 1) / 2) / (n1m * n0m)
    assert abs(scipy_ref - subset_auc(o0, vs0, y[o0], m)) < 1e-12
    print(f"fast AUC checked against scipy: {scipy_ref:.12f}\n")

    PRE, pooled = {}, {}
    for nm in names:
        o, vs = prep(V[nm][0])
        PRE[nm] = (o, vs, y[o])
        pooled[nm] = _auc_sorted(vs, y[o])

    # ---------------------------------------------------------------- the simulation
    rng = np.random.default_rng(a.seed)
    A = np.empty((a.reps, len(names)))
    for r in range(a.reps):
        perm = rng.permutation(n_pool)
        pub = np.zeros(n_pool, bool)
        pub[perm[:n_pub]] = True
        for j, nm in enumerate(names):
            A[r, j] = subset_auc(*PRE[nm], pub)
        if (r + 1) % 200 == 0:
            print(f"  rep {r+1}/{a.reps}")
    sd_single = {nm: float(A[:, j].std(ddof=1)) for j, nm in enumerate(names)}

    # ---- GATE: w15a used the first `gate_reps` draws of this same stream
    G = A[: a.gate_reps]
    ji = {nm: j for j, nm in enumerate(names)}
    pub_json = os.path.join(ROOT, "experiments", "w15a_crossteam.json")
    gate_rows, gate_drift = [], 0.0
    if os.path.exists(pub_json):
        ref_pub = json.load(open(pub_json))
        for nm in GATE_ORDER:
            g = G[:, ji["REF:" + REF]] - G[:, ji[nm]]
            sd = float(g.std(ddof=1))
            want = ref_pub["competitors"][nm]["sd_gap"]
            gate_drift = max(gate_drift, abs(sd - want))
            gate_rows.append(dict(pair=nm, got=sd, want=want, drift=abs(sd - want)))
        print("\nGATE vs w15a_crossteam.json (seed 15, first 400 draws)")
        for r in gate_rows:
            print(f"  {r['pair']:26} got {r['got']*1e6:7.2f}e-6  want {r['want']*1e6:7.2f}e-6  "
                  f"drift {r['drift']*1e12:8.3f}e-12")
        print(f"  MAX DRIFT {gate_drift*1e12:.3f}e-12  "
              f"-> {'PASS' if gate_drift < 1e-12 else 'FAIL'}  (registered < 1e-12)")

    # ------------------------------------------------------- LAW-IF inputs, per vector
    pos = y == 1
    neg = ~pos
    n1_pub = n_pub * float(y.mean())
    n0_pub = n_pub - n1_pub
    fpc = 1.0 - n_pub / n_pool
    Fp, Fn = {}, {}                       # F0(s) on positives, F1(s) on negatives
    for nm in names:
        v = V[nm][0]
        sn = np.sort(v[neg])
        sp = np.sort(v[pos])
        Fp[nm] = midrank_cdf(sn, v[pos])   # fraction of negatives below each positive
        Fn[nm] = midrank_cdf(sp, v[neg])   # fraction of positives below each negative

    RK = {nm: rankdata(V[nm][0]) for nm in names}

    # ------------------------------------------------------------------- all the pairs
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            g = A[:, i] - A[:, j]
            sd = float(g.std(ddof=1))
            rho = float(np.corrcoef(RK[na], RK[nb])[0, 1])
            deficit = abs(pooled[na] - pooled[nb])
            sds = 0.5 * (sd_single[na] + sd_single[nb])
            law_rho = sds * np.sqrt(max(2.0 * (1.0 - rho), 0.0))
            d = Fp[na] - Fp[nb]
            e = Fn[nb] - Fn[na]
            var_if = d.var(ddof=1) / n1_pub + e.var(ddof=1) / n0_pub
            law_if = float(np.sqrt(var_if))
            rows.append(dict(a=na, b=nb, kind_a=V[na][1], kind_b=V[nb][1],
                             auc_a=pooled[na], auc_b=pooled[nb], deficit=float(deficit),
                             rho=rho, sd_gap=sd, mean_gap=float(g.mean()),
                             law_rho=float(law_rho), law_if=law_if,
                             law_if_fpc=float(law_if * np.sqrt(fpc))))
    df = pd.DataFrame(rows)
    df["r_rho"] = df["sd_gap"] / df["law_rho"]
    df["r_if"] = df["sd_gap"] / df["law_if"]
    df["r_if_fpc"] = df["sd_gap"] / df["law_if_fpc"]
    df.to_csv(os.path.join(ROOT, "experiments", "w17h_pairs.csv"), index=False)
    print(f"\n{len(df)} pairs over {len(names)} vectors")

    # ------------------------------------------------------------------------ P2
    lo, hi = df["r_rho"].min(), df["r_rho"].max()
    print(f"\nP2  LAW-RHO ratio observed/predicted: min {lo:.3f} max {hi:.3f} "
          f"spread {hi/lo:.2f}x   -> {'CONFIRMED' if hi/lo >= 3 else 'FALSIFIED'} "
          f"(registered >= 3x)")

    # ------------------------------------------------------------------------ P3
    def partial(xcol):
        """partial corr of log sd_gap with xcol, controlling for the other covariate."""
        Ldf = df[(df["deficit"] > 0) & (df["rho"] < 1)]
        Y = np.log(Ldf["sd_gap"].to_numpy())
        X1 = np.log(1.0 - Ldf["rho"].to_numpy())
        X2 = np.log(Ldf["deficit"].to_numpy())
        tgt, ctl = (X1, X2) if xcol == "rho" else (X2, X1)
        rt = Y - np.polyval(np.polyfit(ctl, Y, 1), ctl)
        rx = tgt - np.polyval(np.polyfit(ctl, tgt, 1), ctl)
        return float(np.corrcoef(rt, rx)[0, 1])
    p_rho, p_def = partial("rho"), partial("deficit")
    print(f"P3  partial corr of log sd_gap with log(1-rho) | deficit = {p_rho:+.3f}")
    print(f"    partial corr of log sd_gap with log deficit | rho     = {p_def:+.3f}"
          f"   -> {'CONFIRMED' if abs(p_def) > abs(p_rho) else 'FALSIFIED'}")

    # ------------------------------------------------------------------------ P4
    med_if = float(np.median(np.abs(df["r_if"] - 1)))
    med_iff = float(np.median(np.abs(df["r_if_fpc"] - 1)))
    med_rho = float(np.median(np.abs(df["r_rho"] - 1)))
    ok4 = med_if < 0.10 and med_rho / med_if > 5
    print(f"P4  median |LAW-IF/sim - 1|      {med_if*100:6.2f}%   (with fpc {med_iff*100:.2f}%)")
    print(f"    median |LAW-RHO/sim - 1|     {med_rho*100:6.2f}%   ratio {med_rho/med_if:.1f}x"
          f"   -> {'CONFIRMED' if ok4 else 'FALSIFIED'} (registered <10% and >5x)")

    # ------------------------------------------------------------------------ P5
    # sibling pairs: the 'a' and 'b' halves of one and the same disjoint split
    sib = df[df.apply(lambda r: r["a"].endswith("a") and r["b"].endswith("b")
                      and r["a"][:-1] == r["b"][:-1]
                      and r["a"].split(":")[0] in ("SYN", "STK"), axis=1)]
    print("\nP5  THE DECISIVE CELL -- matched-quality, genuinely diverse partners")
    print(f"    {'pair':30} {'auc_a':>10} {'deficit':>10} {'rho':>9} {'sd_gap':>11}")
    viol = []
    for _, r in sib.sort_values(["kind_a", "deficit"]).iterrows():
        flag = ""
        if r["deficit"] < 40e-6 and r["sd_gap"] >= 25e-6:
            viol.append(r["a"])
            flag = "  <-- VIOLATES P5"
        elif r["deficit"] >= 40e-6:
            flag = "  (not matched; excluded)"
        print(f"    {r['a'].split(':')[1]+' vs '+r['b'].split(':')[1]:30} "
              f"{r['auc_a']:>10.6f} {r['deficit']*1e6:9.1f}e-6 "
              f"{r['rho']:>9.5f} {r['sd_gap']*1e6:10.2f}e-6{flag}")
    matched = sib[sib["deficit"] < 40e-6]
    print(f"    {len(matched)} of {len(sib)} sibling pairs are quality-matched (<40e-6)")
    print(f"    -> {'CONFIRMED' if not viol else 'FALSIFIED'} "
          f"(registered: deficit<40e-6 pairs all sd_gap < 25e-6)")
    stk = sib[(sib["kind_a"] == "stackhalf") & (sib["deficit"] < 40e-6)]

    # ------------------------------------------------------------------------ P6
    print("\nP6  READOUT -- what the board gaps are actually worth")
    ours = df[(df["kind_a"] == "ours") & (df["kind_b"] == "ours")]
    tight = ours[ours["deficit"] < 40e-6]
    print(f"    our own pairs, deficit<40e-6:  n {len(tight)}  "
          f"sd_gap median {tight['sd_gap'].median()*1e6:.1f}e-6 "
          f"range {tight['sd_gap'].min()*1e6:.1f}-{tight['sd_gap'].max()*1e6:.1f}e-6")
    if len(stk):
        print(f"    disjoint STACK halves, matched: n {len(stk)}  "
              f"sd_gap median {stk['sd_gap'].median()*1e6:.1f}e-6 "
              f"range {stk['sd_gap'].min()*1e6:.1f}-{stk['sd_gap'].max()*1e6:.1f}e-6")
    # the honest upper bound for an unseen rival: the most disagreeing matched pair we built
    ub = float(matched["sd_gap"].max()) if len(matched) else float("nan")
    print(f"    UPPER BOUND from the most-disagreeing matched pair: {ub*1e6:.1f}e-6")
    cols = [("incumbent 53-84e-6", 68.5e-6),
            ("our own matched pairs", float(tight["sd_gap"].median()))]
    if len(stk):
        cols.append(("disjoint stack halves", float(stk["sd_gap"].median())))
    cols.append(("upper bound", ub))
    print(f"    {'board gap':34}" + "".join(f"{t:>26}" for t, _ in cols))
    for label, gap in [("MILANFX 0.97124", 18e-5), ("the 0.97117 team", 11e-5),
                       ("the 0.97113 team", 5e-5)]:
        print(f"    {label:20} {gap*1e5:5.1f}e-5      "
              + "".join(f"{gap/s:>24.1f}s" for _, s in cols))

    json.dump(dict(reps=a.reps, seed=a.seed, n_pub=n_pub, n_pairs=len(df),
                   vectors={nm: dict(kind=V[nm][1], pooled_auc=pooled[nm],
                                     sd_single=sd_single[nm]) for nm in names},
                   gate=dict(rows=gate_rows, max_drift=gate_drift),
                   P2=dict(ratio_min=float(lo), ratio_max=float(hi), spread=float(hi / lo)),
                   P3=dict(partial_rho=p_rho, partial_deficit=p_def),
                   P4=dict(med_if=med_if, med_if_fpc=med_iff, med_rho=med_rho),
                   P5=dict(violations=viol, n_matched=int(len(matched)),
                           pairs=sib[["a", "b", "kind_a", "auc_a", "deficit", "rho",
                                      "sd_gap"]].to_dict("records")),
                   tight_median=float(tight["sd_gap"].median()),
                   stackhalf_median=float(stk["sd_gap"].median()) if len(stk) else None,
                   upper_bound=ub),
              open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
