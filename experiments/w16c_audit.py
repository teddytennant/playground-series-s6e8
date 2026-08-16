"""w16c: the consolidation audit — is the file at the top of CV the right deadline pick?

WHY THIS RUN EXISTS
-------------------
w15i settled "which pair to select" over the 11-file CV-top cluster: all 55 pairs within
0.5e-6 of each other on E[max], current pick (blend159av_h3 + blend160origm_h3) ranks 3/55,
and every one of those files fits ZERO parameters above the stack.

Two files have since been added that are NOT in that cluster and were never simulated:

    w15f_antistudent_avg   base + c_avg with ONE global fitted weight      CV 0.9700528
    w16b_cellweight        base + c_avg with SEVEN per-rule-cell weights   CV 0.9700554

Both print 0.97107 on the public slice — the account best — and w16b holds the best CV number
in the workspace, +6.2e-6 above the current first pick. The brief says select on CV. So either
the pick moves to w16b, or there is a number that says it should not. Nobody has produced that
number: w15i's simulation predates both files.

WHAT THIS SCRIPT DOES
---------------------
1. THE LADDER, re-verified against the live API on all scored files. Recompute every CV from
   the file's own stored OOF, join to the live public score, refit w15i's within-family fixed
   effects slope with the two new readings folded in, and check slot-1's three pre-registration
   rules (ens4 high/low shelf, h3 cluster) against every row rather than against the subset
   they were induced from.

2. THE PAIRED PRIVATE-SLICE SIMULATION, extended to the corrected files. Their OOF vectors do
   not exist on disk, so they are RECONSTRUCTED CROSS-FITTED from w16b_cellweight.json's stored
   per-fold weights: fold f's validation rows get the weights coordinate-ascended on the other
   four folds. That is the honest out-of-sample object and is exactly what their CV number
   measures; the shipped test files use full-data weights and are, if anything, slightly better.
   Every file is scored on the SAME slice each rep, so the sd of the DIFFERENCE is the quantity
   the decision needs, and it is far tighter than the fold-level spread.

3. THE FOLD-LEVEL UNCERTAINTY of the corrected arms, which the journal reports as point
   estimates. Five per-fold deltas, paired against the arm below, t on 4 df.

Nothing here refits a model, nothing is chosen with reference to the public slice, and no file
is written for submission.

    w16c_audit.py --reps 500
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DAILY, TARGET, get_folds, load_raw  # noqa: E402

from w15i_cvlb import classify, full_auc, prep, subset_auc  # noqa: E402
from w16b_cellweight import CELL_A, rule_cells  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
HERE = os.path.join(ROOT, "experiments")
N_TEST = 296_302
BASE = "blend159av_h3"
PICK = ("blend159av_h3", "blend160origm_h3")
CLUSTER_MIN_CV = 0.970046

# the three pre-registration rules slot 1 (w16a section 0c) wrote into the journal
LADDER = [
    ("ens4 high  CV >= 0.9700416", lambda k, cv: k == "ens4" and cv >= 0.9700416, 0.97106),
    ("ens4 low   CV <= 0.9700343", lambda k, cv: k == "ens4" and cv <= 0.9700343, 0.97104),
    ("h3 / w cluster", lambda k, cv: k in ("h3", "w"), 0.97105),
]


def pct(x):
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


def corrected_oof(br, c, assign, fold_ws, folds):
    """Cross-fitted corrected OOF: fold f's rows use weights fitted WITHOUT fold f."""
    z = br.copy()
    for (_, iva), w in zip(folds, fold_ws):
        for lvl, wv in w.items():
            m = iva[assign[iva] == lvl]
            z[m] = br[m] + wv * c[m]
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1616)
    ap.add_argument("--f", type=float, default=0.20)
    ap.add_argument("--lbdir", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "w16c_audit.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {}

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)
    folds = get_folds(y)

    # ================================================================ 1. the ladder
    print("=" * 80)
    print("1. CV -> LB, EVERY SCORED FILE, LIVE API")
    print("=" * 80)
    s = pd.read_csv(os.path.join(HERE, "w16c_subs_raw.csv"))
    s = s[s["status"].astype(str).str.endswith("COMPLETE")].copy()
    s["stem"] = s["fileName"].str.replace(".csv", "", regex=False)
    rows = []
    for _, r in s.iterrows():
        p = os.path.join(SUB, f"oof_{r['stem']}.npy")
        if os.path.exists(p):
            rows.append(dict(name=r["stem"], lb=float(r["publicScore"]),
                             date=str(r["date"])[:10], cv=full_auc(np.load(p), y)))
    t = pd.DataFrame(rows).drop_duplicates("name")
    t[["mset", "kind"]] = t["name"].apply(lambda n: pd.Series(classify(n)))
    t["big"] = ~t["name"].str.startswith("stack_pub")
    big = t[t["big"]].copy()
    print(f"\n  {len(s)} scored submissions on the API, {len(t)} have a stored OOF, "
          f"{len(big)} are pack files")
    print(f"  files with a public score but NO stored OOF (cannot enter the map): "
          f"{sorted(set(s['stem']) - set(t['name']))}")

    # --- the pre-registration rules, checked on every row rather than the inducing subset
    print("\n  slot-1's three pre-registration rules, re-checked on all scored pack files:")
    ladder_rec = {}
    for lab, cond, want in LADDER:
        m = big.apply(lambda r: cond(r["kind"], r["cv"]), axis=1)
        hit = int((big.loc[m, "lb"] == want).sum())
        tot = int(m.sum())
        bad = big.loc[m & (big["lb"] != want), ["name", "cv", "lb"]]
        print(f"    {lab:32s} -> {want:.5f}   {hit}/{tot}"
              + ("" if bad.empty else "   MISSES: "
                 + ", ".join(f"{r['name']}({r['lb']:.5f})" for _, r in bad.iterrows())))
        ladder_rec[lab] = [hit, tot]
    out["ladder"] = ladder_rec

    # --- within-family fixed effects, w15i's instrument, with today's readings
    g = big.groupby("kind")
    big["cv_d"] = big["cv"] - g["cv"].transform("mean")
    big["lb_d"] = big["lb"] - g["lb"].transform("mean")
    keep = big[g["cv"].transform("count") >= 2]
    beta = float((keep["cv_d"] * keep["lb_d"]).sum() / (keep["cv_d"] ** 2).sum())
    resid = keep["lb_d"] - beta * keep["cv_d"]
    dfree = len(keep) - keep["kind"].nunique() - 1
    se = float(np.sqrt((resid ** 2).sum() / dfree / (keep["cv_d"] ** 2).sum()))
    print(f"\n  within-family FE slope {beta:+.3f} +/- {se:.3f}  t {beta/se:+.2f}  "
          f"n={len(keep)}  resid sd {resid.std(ddof=1):.2e} "
          f"({resid.std(ddof=1)/1e-5:.2f} LB grid steps)")
    conc = disc = 0
    for _, gg in big.groupby("kind"):
        for i, j in itertools.combinations(range(len(gg)), 2):
            dc = gg["cv"].iloc[i] - gg["cv"].iloc[j]
            dl = gg["lb"].iloc[i] - gg["lb"].iloc[j]
            if dl == 0 or dc == 0:
                continue
            conc += int(np.sign(dc) == np.sign(dl))
            disc += int(np.sign(dc) != np.sign(dl))
    print(f"  within-family resolvable pairs {conc + disc}: {conc} agree / {disc} disagree "
          f"= {conc/(conc+disc):.1%} concordant")
    out["fe"] = dict(slope=beta, se=se, n=int(len(keep)), conc=conc, disc=disc)

    print("\n  top of the CV table (recomputed CV, live LB):")
    for _, r in big.sort_values("cv", ascending=False).head(8).iterrows():
        print(f"    {r['name']:24s} CV {r['cv']:.8f}  LB {r['lb']:.5f}  "
              f"kind {r['kind']:8s} {r['date']}")

    # ================================================== 2. reconstruct the corrected files
    print("\n" + "=" * 80)
    print("2. THE TWO CORRECTED FILES, RECONSTRUCTED CROSS-FITTED")
    print("=" * 80)
    j = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    cell = rule_cells(tr)
    a_only = np.where(cell == CELL_A, "A", "rest").astype(object)
    one = np.zeros(n_pool, dtype=object)
    one_s = np.array([str(v) for v in one], dtype=object)

    built = {}
    for tag, assign, key in (("w15f_antistudent_avg", one_s, "glob"),
                             ("w16c_aonly2", a_only, "a_only"),
                             ("w16b_cellweight", cell, "per_cell")):
        fw = [{k: float(v) for k, v in d.items()} for d in j["fold_weights"][key]]
        built[tag] = corrected_oof(br, c, assign, fw, folds)
        auc = full_auc(built[tag], y)
        rep = j["arms"][{"glob": "global", "a_only": "a_only",
                         "per_cell": "per_cell"}[key]]["cv"]
        print(f"  {tag:24s} reconstructed CV {auc:.8f}   w16b reported {rep:.8f}   "
              f"delta {(auc-rep)*1e6:+.3f}e-6")
    base_auc = full_auc(br, y)
    print(f"  {BASE + ' (base, 0 params)':24s} CV {base_auc:.8f}")

    # --- fold-level uncertainty on the arms, which the journal reports as point estimates
    print("\n  per-fold deltas and their paired uncertainty (the journal quotes only the mean):")
    arms = {k: np.array(j["arms"][k]["per_fold"]) for k in
            ("global", "a_only", "per_cell", "ctrl7", "ctrl2")}
    for k, v in arms.items():
        se_f = v.std(ddof=1) / np.sqrt(len(v))
        print(f"    {k:10s} mean {v.mean()*1e6:+7.3f}e-6  sd {v.std(ddof=1)*1e6:6.3f}e-6  "
              f"se {se_f*1e6:6.3f}e-6  t(4df) {v.mean()/se_f:+5.2f}")
    for lab, x, z in (("per_cell - global", arms["per_cell"], arms["global"]),
                      ("per_cell - a_only", arms["per_cell"], arms["a_only"]),
                      ("per_cell - ctrl7 ", arms["per_cell"], arms["ctrl7"]),
                      ("a_only   - ctrl2 ", arms["a_only"], arms["ctrl2"])):
        d = x - z
        se_f = d.std(ddof=1) / np.sqrt(len(d))
        print(f"    {lab} paired mean {d.mean()*1e6:+7.3f}e-6  se {se_f*1e6:6.3f}e-6  "
              f"t(4df) {d.mean()/se_f:+5.2f}")
    out["arm_folds"] = {k: dict(mean=float(v.mean()), se=float(v.std(ddof=1)/np.sqrt(len(v))))
                        for k, v in arms.items()}

    # ============================================== 3. paired private-slice simulation
    print("\n" + "=" * 80)
    print("3. PAIRED PRIVATE-SLICE SIMULATION -- CV-top cluster PLUS the corrected files")
    print("=" * 80)
    cl = big[big["cv"] >= CLUSTER_MIN_CV].sort_values("cv", ascending=False)
    names = cl["name"].tolist()
    vecs = {n: np.load(os.path.join(SUB, f"oof_{n}.npy")).astype(np.float64) for n in names}
    for tag in ("w15f_antistudent_avg", "w16c_aonly2", "w16b_cellweight"):
        vecs[tag] = built[tag]
        names.append(tag)
    npar = {n: 0 for n in names}
    npar.update({"w15f_antistudent_avg": 1, "w16c_aonly2": 2, "w16b_cellweight": 7})
    print(f"\n  {len(names)} files: {len(cl)} zero-parameter cluster members + 3 corrected")
    pk = {n: prep(vecs[n], y) for n in names}
    n_pub = int(round(N_TEST * a.f))
    print(f"  {a.reps} reps, {N_TEST}-row pseudo-test from the {n_pool}-row pool, "
          f"f={a.f} cut away, private = {N_TEST - n_pub} rows, SAME slice for every file")
    A = np.zeros((a.reps, len(names)))
    for i in range(a.reps):
        idx = rng.choice(n_pool, size=N_TEST, replace=False)
        m = np.zeros(n_pool, dtype=bool)
        m[idx[n_pub:]] = True
        for k, n in enumerate(names):
            A[i, k] = subset_auc(pk[n], m)
    mean = A.mean(axis=0)
    ipick = names.index(PICK[0])
    print(f"\n  mean private AUC, and the PAIRED difference against the current first pick "
          f"({PICK[0]}):")
    base_col = A[:, ipick]
    solo = {}
    for k, n in enumerate(names):
        d = A[:, k] - base_col
        sed = d.std(ddof=1) / np.sqrt(a.reps)
        win = float((d > 0).mean())
        solo[n] = dict(mean=float(mean[k]), d=float(d.mean()), sd=float(d.std(ddof=1)),
                       se=float(sed), p_better=win, params=npar[n])
        print(f"    {n:24s} p{npar[n]:<2d} mean {mean[k]:.8f}  d {d.mean()*1e6:+7.2f}e-6 "
              f"+/-{sed*1e6:5.2f}  sd(d) {d.std(ddof=1)*1e6:6.2f}e-6  P(beats pick) {win:.3f}")
    out["solo"] = solo

    print(f"\n  every pair by E[max] -- the quantity Kaggle scores, best first:")
    pairs = []
    for i, k in itertools.combinations(range(len(names)), 2):
        mx = np.maximum(A[:, i], A[:, k])
        pairs.append((float(mx.mean()), names[i], names[k],
                      float((A[:, i] - A[:, k]).std(ddof=1))))
    pairs.sort(reverse=True)
    best_solo = max(mean)
    for e, n1, n2, sdd in pairs[:12]:
        tag = "  <-- current recommendation" if {n1, n2} == set(PICK) else ""
        print(f"    E[max] {e:.8f}  (+{(e-best_solo)*1e6:5.2f}e-6 over best single)  "
              f"sd(d) {sdd*1e6:5.2f}e-6  p{npar[n1]}+p{npar[n2]}  {n1} + {n2}{tag}")
    rank_cur = 1 + [i for i, p in enumerate(pairs) if {p[1], p[2]} == set(PICK)][0]
    cur = [p for p in pairs if {p[1], p[2]} == set(PICK)][0]
    print(f"\n  current recommendation ranks {rank_cur} of {len(pairs)}; "
          f"E[max] {cur[0]:.8f}; best pair beats it by "
          f"{(pairs[0][0]-cur[0])*1e6:+.2f}e-6")
    out["pairs"] = [dict(e_max=e, a=n1, b=n2, sd_diff=sdd) for e, n1, n2, sdd in pairs[:15]]
    out["current_pair_rank"] = int(rank_cur)
    out["n_pairs"] = int(len(pairs))

    # the specific decision: swap the FIRST pick for the CV leader, keep the second
    print("\n  THE DECISION -- candidate pairs against the incumbent, paired:")
    cand = [PICK,
            ("w16b_cellweight", PICK[1]),
            ("w16b_cellweight", PICK[0]),
            ("w16b_cellweight", "w15f_antistudent_avg"),
            ("w15f_antistudent_avg", PICK[0])]
    inc = np.maximum(A[:, names.index(PICK[0])], A[:, names.index(PICK[1])])
    dec = {}
    for p in cand:
        mx = np.maximum(A[:, names.index(p[0])], A[:, names.index(p[1])])
        d = mx - inc
        sed = d.std(ddof=1) / np.sqrt(a.reps)
        dec[" + ".join(p)] = dict(e_max=float(mx.mean()), d=float(d.mean()),
                                  se=float(sed), p_better=float((d > 0).mean()))
        print(f"    {' + '.join(p):50s} E[max] {mx.mean():.8f}  "
              f"vs incumbent {d.mean()*1e6:+6.2f}e-6 +/-{sed*1e6:4.2f}  "
              f"P(better) {(d > 0).mean():.3f}")
    out["decision"] = dec

    # ------------------------------------------------ places, at the live board density
    lbdir = a.lbdir or ""
    cands = sorted(glob.glob(os.path.join(lbdir, "*publicleaderboard*.csv"))) if lbdir else []
    if cands:
        lb = pd.read_csv(cands[-1])
        ours = lb[lb["TeamMemberUserNames"].astype(str).str.contains("thtennant", na=False)]
        sc = float(ours["Score"].iloc[0])
        dens = int(((lb["Score"] >= sc - 2e-5) & (lb["Score"] <= sc + 2e-5)).sum()) / 4e-5
        print(f"\n  board {len(lb)} teams, leader {lb['Score'].max():.5f}, ours {sc:.5f} at "
              f"rank {int(ours['Rank'].iloc[0])}; local density "
              f"{dens*1e-5:.1f} teams per 1e-5")
        for k, v in dec.items():
            print(f"    {k:50s} {v['d']*dens:+5.2f} places")
        out["density_per_1e5"] = float(dens * 1e-5)
        out["rank"] = int(ours["Rank"].iloc[0])
        out["teams"] = int(len(lb))

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
