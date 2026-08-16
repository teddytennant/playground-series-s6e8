"""The deadline defence: every (file, cross-fitted CV, public LB) triple, the CV->LB map
as of 2026-08-15, and the priced recommendation for what to select.

WHY THIS EXISTS
---------------
The competition closes 2026-08-31 and `check_selection.py` still exits 1: nothing is
selected.  If that holds, Kaggle auto-selects on best PUBLIC score.  Three journal entries
have priced that risk and each priced it differently (-88e-6 "CV gap", -111e-6 "predicted
private", ~-10e-6 "the 4-way tie bounds it").  Nobody has yet written the number a human
needs in order to decide whether the click is worth making, with its uncertainty and with
the assumptions it rests on made explicit.  That is the deliverable here.

Four things, in the order they matter:

1.  ASSEMBLE.  Recompute the cross-fitted CV of every submitted file from its own stored
    `submissions/oof_*.npy` rather than trusting `audit_results.csv`, and join it to the
    LIVE public score off the API.  `audit_results.csv` is 8 readings stale as of today.
2.  THE MAP, AND WHETHER IT IS DRIFTING.  Regress LB on CV over the >=150-member family.
    Then split by submission epoch (08-10/11, 08-13, 08-14/15), fit on the early epoch and
    predict the later ones OUT OF SAMPLE.  A relationship that is stable predicts; one that
    is drifting does not.  w14a fitted this in-sample only.
3.  THE LOGIT DISPLACEMENT, ARITHMETIC CHECKED FROM SCRATCH.  RESEARCH.md records it as
    "measured on three member sets", and w14b then showed the three readings correlate
    +0.992 on a shared slice so the effective independent count is 1.01.  Re-derive the
    three displacements from the recomputed CVs and the live LB, re-run the shared-slice
    correlation from the stored OOF, and state whether it is an effect or one observation
    reported three times.
4.  THE RECOMMENDATION, PRICED.  Two named files, and the expected private-AUC cost of the
    default auto-selection against them, with its uncertainty and a sensitivity sweep over
    the two assumptions that are not readable from the API (the final-submission limit and
    the public-slice fraction f).

    w15i_cvlb.py --reps 400
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
HERE = os.path.join(ROOT, "experiments")
N_TEST = 296_302
TRANSFORMS = ("logit", "hybrid", "rankraw", "rescale")

# The four files tied for best public score (0.97106).  This is the auto-selection draw.
TIE = ["blend156", "blend158", "blend158_logit", "blend159av"]
# The deadline pick this workspace has recommended since 2026-08-13.
PICK = ["blend159av_h3", "blend160origm_h3"]
REF = "blend159av_h3"


# --------------------------------------------------------------------- fast AUC
def prep(v, y):
    """Global stable argsort once; subset AUCs are then O(n) instead of O(n log n).

    The labels must be carried in the SAME permutation as the values -- every vector has
    its own sort order, so a single shared `y` silently scores the wrong rows.
    """
    o = np.argsort(v, kind="stable")
    return o, np.ascontiguousarray(v[o]), np.ascontiguousarray(y[o])


def _auc_sorted(s, yy):
    """Mann-Whitney AUC on values already sorted ascending, average ranks for ties."""
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


def subset_auc(pk, mask):
    o, vs, ys = pk
    sel = mask[o]
    return _auc_sorted(vs[sel], ys[sel])


def full_auc(v, y):
    _, vs, ys = prep(v, y)
    return _auc_sorted(vs, ys)


def classify(name):
    """(member-set tag, transform tag) for a submission file stem."""
    for t in TRANSFORMS:
        if name.endswith("_" + t):
            return name[: -len(t) - 1], t
    if name.endswith("_wh3"):
        return name[:-4], "h3"          # wh3 is an h3 rank-average with two fitted weights
    if name.endswith("_h3"):
        return name[:-3], "h3"
    if name == "blendtop3":
        return name, "h3"               # rank-average of three h3 files
    if name.endswith("_w"):
        return name[:-2], "w"
    if name in ("blend156w", "blend156w2"):
        return name, "w"
    return name, "ens4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--subs", default=os.path.join(HERE, "w15i_subs_raw.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "w15i_cvlb.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)
    out = {}

    # ------------------------------------------------------------- 1. assemble
    print("=" * 78)
    print("1. THE TRIPLES -- CV recomputed from stored OOF, LB read live off the API")
    print("=" * 78)
    s = pd.read_csv(a.subs)
    s = s[s["status"].astype(str).str.endswith("COMPLETE")].copy()
    s["stem"] = s["fileName"].str.replace(".csv", "", regex=False)
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date")

    rows = []
    for _, r in s.iterrows():
        p = os.path.join(SUB, f"oof_{r['stem']}.npy")
        if not os.path.exists(p):
            print(f"  ! {r['stem']:28s} LB {r['publicScore']:.5f}  NO STORED OOF -- excluded")
            continue
        v = np.load(p)
        assert v.shape[0] == n_pool, (r["stem"], v.shape)
        rows.append(dict(name=r["stem"], date=r["date"], lb=float(r["publicScore"]),
                         cv=full_auc(v, y)))
    t = pd.DataFrame(rows)
    t[["mset", "kind"]] = t["name"].apply(lambda n: pd.Series(classify(n)))
    t["day"] = t["date"].dt.strftime("%m-%d")
    t["big"] = ~t["name"].str.startswith("stack_pub7") & ~t["name"].str.startswith("stack_pub8")
    print(f"\n  {len(t)} triples assembled ({int(t['big'].sum())} in the >=150-member family)")

    # cross-check against the workspace's own record
    au = pd.read_csv(os.path.join(HERE, "audit_results.csv")).set_index("name")
    dif = []
    for _, r in t.iterrows():
        if r["name"] in au.index:
            dif.append(abs(r["cv"] - float(au.loc[r["name"], "cv"])))
    dif = np.array(dif)
    print(f"  cross-check vs audit_results.csv on {len(dif)} shared names: "
          f"max |dCV| {dif.max():.2e}  (0 = the recomputation reproduces the record)")
    stale = t[~t["name"].isin(au.index) | au.reindex(t["name"])["lb"].isna().to_numpy()]
    print(f"  readings NOT in audit_results.csv's lb column (it is stale by these): "
          + ", ".join(stale["name"].tolist()))

    print(f"\n  {'file':24s} {'day':>6s} {'kind':>7s} {'CV':>10s} {'LB':>9s} {'LB-CV':>10s}")
    for _, r in t.sort_values("cv", ascending=False).iterrows():
        print(f"  {r['name']:24s} {r['day']:>6s} {r['kind']:>7s} {r['cv']:10.6f} "
              f"{r['lb']:9.5f} {r['lb']-r['cv']:+10.6f}")
    out["n_triples"] = int(len(t))
    out["max_cv_recompute_diff"] = float(dif.max())

    # ------------------------------------------------------ 2. the map, and drift
    print("\n" + "=" * 78)
    print("2. THE CV->LB MAP, AND WHETHER IT IS STABLE")
    print("=" * 78)
    big = t[t["big"]].copy()
    x, yy = big["cv"].to_numpy(), big["lb"].to_numpy()
    sl, ic = np.polyfit(x, yy, 1)
    res = yy - (sl * x + ic)
    r2 = 1 - res.var() / yy.var()
    print(f"\n  full >=150 family, n={len(big)}")
    print(f"    slope {sl:+.4f}   intercept {ic:+.5f}   R^2 {r2:.4f}   "
          f"pearson {np.corrcoef(x, yy)[0,1]:+.3f}   spearman "
          f"{pd.Series(x).corr(pd.Series(yy), method='spearman'):+.3f}")
    print(f"    residual sd {res.std(ddof=2):.3e} = {res.std(ddof=2)/1e-5:.2f} LB grid steps")
    print(f"    gap (LB-CV): mean {(yy-x).mean():+.6f}  sd {(yy-x).std(ddof=1):.6f}")
    out["fit_all"] = dict(n=int(len(big)), slope=float(sl), intercept=float(ic),
                          r2=float(r2), resid_sd=float(res.std(ddof=2)),
                          gap_mean=float((yy - x).mean()), gap_sd=float((yy - x).std(ddof=1)))

    print("\n  by epoch -- is the relationship the same object each day?")
    print(f"    {'epoch':>12s} {'n':>3s} {'slope':>9s} {'R^2':>7s} {'gap mean':>10s} "
          f"{'gap sd':>9s} {'resid sd':>9s}")
    epochs = {"08-10/11": ("08-10", "08-11"), "08-13": ("08-13",), "08-14/15": ("08-14", "08-15")}
    ep_stats = {}
    for lab, days in epochs.items():
        g = big[big["day"].isin(days)]
        if len(g) < 2:
            continue
        gx, gy = g["cv"].to_numpy(), g["lb"].to_numpy()
        gs, gi = np.polyfit(gx, gy, 1)
        gr = gy - (gs * gx + gi)
        gr2 = 1 - gr.var() / gy.var() if gy.var() > 0 else float("nan")
        print(f"    {lab:>12s} {len(g):3d} {gs:+9.3f} {gr2:7.3f} {(gy-gx).mean():+10.6f} "
              f"{(gy-gx).std(ddof=1):9.6f} {gr.std(ddof=2) if len(g)>2 else float('nan'):9.2e}")
        ep_stats[lab] = dict(n=int(len(g)), slope=float(gs), r2=float(gr2),
                             gap_mean=float((gy - gx).mean()), gap_sd=float((gy - gx).std(ddof=1)))
    out["epochs"] = ep_stats

    print("\n  OUT-OF-SAMPLE: fit on 08-10/11 only, predict every later reading")
    tr_ep = big[big["day"].isin(("08-10", "08-11"))]
    te_ep = big[~big["day"].isin(("08-10", "08-11"))]
    s0, i0 = np.polyfit(tr_ep["cv"], tr_ep["lb"], 1)
    pr = s0 * te_ep["cv"].to_numpy() + i0
    er = te_ep["lb"].to_numpy() - pr
    # the null a regression has to beat: predict the training-epoch mean LB
    nul = te_ep["lb"].to_numpy() - tr_ep["lb"].mean()
    # and the constant-gap null: LB = CV + mean training gap
    cg = te_ep["lb"].to_numpy() - (te_ep["cv"].to_numpy() + (tr_ep["lb"] - tr_ep["cv"]).mean())
    print(f"    train n={len(tr_ep)} slope {s0:+.3f} | test n={len(te_ep)}")
    print(f"    regression   RMSE {np.sqrt((er**2).mean()):.3e}  bias {er.mean():+.3e}")
    print(f"    mean-LB null RMSE {np.sqrt((nul**2).mean()):.3e}  bias {nul.mean():+.3e}")
    print(f"    const-gap    RMSE {np.sqrt((cg**2).mean()):.3e}  bias {cg.mean():+.3e}")
    out["oos"] = dict(n_train=int(len(tr_ep)), n_test=int(len(te_ep)), slope_train=float(s0),
                      rmse_reg=float(np.sqrt((er**2).mean())),
                      rmse_meannull=float(np.sqrt((nul**2).mean())),
                      rmse_constgap=float(np.sqrt((cg**2).mean())),
                      bias_reg=float(er.mean()))

    # ------------------------------------------- 3. residual per transform family
    print("\n" + "=" * 78)
    print("3. RESIDUAL PER TRANSFORM FAMILY (against the common fit)")
    print("=" * 78)
    big = big.assign(res=res)
    print(f"\n  {'kind':8s} {'n':>3s} {'mean resid':>11s} {'sd':>10s} {'sem':>10s}  LB values")
    fam = {}
    for k, g in big.groupby("kind"):
        sd = g["res"].std(ddof=1) if len(g) > 1 else float("nan")
        sem = sd / np.sqrt(len(g)) if len(g) > 1 else float("nan")
        print(f"  {k:8s} {len(g):3d} {g['res'].mean():+11.3e} {sd:10.3e} {sem:10.3e}  "
              + " ".join(f"{v:.5f}" for v in sorted(g['lb'])))
        fam[k] = dict(n=int(len(g)), mean=float(g["res"].mean()), sd=float(sd))
    out["family_resid"] = fam

    print("\n  h3-family LB invariance (w14a's rule, extended with 08-14/15):")
    h3 = big[big["kind"] == "h3"].sort_values("cv", ascending=False)
    print(f"    n={len(h3)}  CV spans {h3['cv'].max()-h3['cv'].min():.2e}  "
          f"LB values {sorted(set(h3['lb']))}  sd(LB) {h3['lb'].std(ddof=1):.2e}")

    # -------------------------------- 4. the logit displacement, checked from scratch
    print("\n" + "=" * 78)
    print("4. THE LOGIT DISPLACEMENT -- IS IT AN EFFECT OR ONE OBSERVATION x3?")
    print("=" * 78)
    panel = []
    for ms, g in big.groupby("mset"):
        d = g.set_index("kind")
        if {"logit", "hybrid"} <= set(d.index):
            panel.append((ms, d.loc["logit", "cv"], d.loc["hybrid", "cv"],
                          d.loc["logit", "lb"], d.loc["hybrid", "lb"]))
    print(f"\n  {'set':8s} {'dCV':>10s} {'dLB':>10s} {'displacement':>14s}")
    disp = []
    for ms, lc, hc, ll, hl in panel:
        dcv, dlb = lc - hc, ll - hl
        disp.append(dlb - dcv)
        print(f"  {ms:8s} {dcv:+10.2e} {dlb:+10.2e} {dlb-dcv:+14.2e}")
    disp = np.array(disp)
    print(f"  mean {disp.mean():+.2e}   naive sd/sqrt(n) with n={len(disp)}: "
          f"{disp.std(ddof=1)/np.sqrt(len(disp)):.2e}")

    print("\n  Shared-slice simulation: how much of each reading is the SAME draw?")
    print(f"  (pseudo-test {N_TEST} rows from the {n_pool}-row labelled pool, f=0.20 public")
    print(f"   slice, all three sets scored on the IDENTICAL slice, {a.reps} reps)")
    vecs = {}
    for ms, *_ in panel:
        for tf in ("logit", "hybrid"):
            nm = f"{ms}_{tf}"
            vecs[nm] = prep(np.load(os.path.join(SUB, f"oof_{nm}.npy")), y)
    pool_gap = {}
    for ms, *_ in panel:
        pool_gap[ms] = (full_auc(np.load(os.path.join(SUB, f"oof_{ms}_logit.npy")), y)
                        - full_auc(np.load(os.path.join(SUB, f"oof_{ms}_hybrid.npy")), y))
    sets = [p[0] for p in panel]
    devs = np.zeros((a.reps, len(sets)))
    f = 0.20
    n_slice = int(round(N_TEST * f))
    for i in range(a.reps):
        idx = rng.choice(n_pool, size=N_TEST, replace=False)
        sl_idx = idx[:n_slice]
        m = np.zeros(n_pool, dtype=bool)
        m[sl_idx] = True
        for j, ms in enumerate(sets):
            g = subset_auc(vecs[f"{ms}_logit"], m) - subset_auc(vecs[f"{ms}_hybrid"], m)
            devs[i, j] = g - pool_gap[ms]
    C = np.corrcoef(devs.T)
    off = C[np.triu_indices(len(sets), 1)]
    rho = off.mean()
    n_eff = len(sets) / (1 + (len(sets) - 1) * rho)
    sd1 = devs.std(axis=0, ddof=1)
    sd_mean = devs.mean(axis=1).std(ddof=1)
    print(f"\n    per-set sd of the slice deviation: "
          + "  ".join(f"{s}={v:.2e}" for s, v in zip(sets, sd1)))
    print(f"    pairwise correlations of the deviations: "
          + "  ".join(f"{v:+.4f}" for v in off) + f"   mean rho {rho:+.4f}")
    print(f"    => effective independent reads  n_eff = {len(sets)}/(1+{len(sets)-1}*rho) "
          f"= {n_eff:.3f}  of {len(sets)}")
    print(f"    sd of the MEAN of the three:  {sd_mean:.2e}   "
          f"(if independent it would be {sd1.mean()/np.sqrt(3):.2e})")
    z = disp.mean() / sd_mean
    print(f"\n    observed mean displacement {disp.mean():+.2e} against the shared-slice sd "
          f"of the mean {sd_mean:.2e}  =>  z = {z:+.2f}")
    # LB quantisation: each dLB is a 1e-5 grid reading, uniform +-5e-6 -> sd 1e-5/sqrt(12)
    q = 1e-5 / np.sqrt(12) * np.sqrt(2)          # two quantised readings per dLB
    sd_tot = np.sqrt(sd_mean**2 + (q / np.sqrt(n_eff))**2)
    print(f"    folding in LB quantisation (each dLB is two 1e-5 grid reads, "
          f"sd {q:.2e} each, n_eff {n_eff:.2f}): total sd {sd_tot:.2e}  =>  z = "
          f"{disp.mean()/sd_tot:+.2f}")
    print(f"    P(a single shared draw >= the observed mean) = "
          f"{float((devs.mean(axis=1) >= disp.mean()).mean()):.4f}")
    out["logit_disp"] = dict(sets=sets, disp=[float(v) for v in disp],
                             mean=float(disp.mean()), rho=float(rho), n_eff=float(n_eff),
                             sd_mean=float(sd_mean), z=float(z),
                             z_with_quant=float(disp.mean() / sd_tot),
                             p_shared=float((devs.mean(axis=1) >= disp.mean()).mean()))

    # ------------------------------------------------- 5. the recommendation, priced
    print("\n" + "=" * 78)
    print("5. THE RECOMMENDATION, PRICED")
    print("=" * 78)
    cv = t.set_index("name")["cv"].to_dict()
    lb = t.set_index("name")["lb"].to_dict()

    print("\n  The public-LB tie that the default draws from (best public = "
          f"{max(lb.values()):.5f}):")
    for n in TIE:
        print(f"    {n:18s} CV {cv[n]:.6f}  LB {lb[n]:.5f}  kind {classify(n)[1]}")
    print(f"  The CV pick: " + ", ".join(f"{n} (CV {cv[n]:.6f}, LB {lb[n]:.5f})" for n in PICK))

    # sigma on the full-test gap.  The CV gap is measured on the 691,369 labelled rows;
    # the real full-test gap lives on 296,302 DIFFERENT rows drawn from the same generator.
    # Both estimate one population quantity, so
    #     Var(g_test - g_cv) = sigma_pop^2 * (1/296302 + 1/691369).
    # sigma_pop^2 is recovered from subsampling WITHOUT replacement inside the pool: a
    # subsample of n rows has Var(g_sub - g_pool) = sigma_pop^2 * (1/n - 1/N) (the finite-
    # population correction), so the scale factor between what we can measure and what we
    # want is exactly (1/n + 1/N)/(1/n - 1/N) = 2.500 at n = N_TEST.  No bootstrap: a
    # with-replacement resample of an AUC needs tie-weighted ranks and the coverage-mask
    # shortcut silently draws only ~63% of the rows.
    FPC = (1.0 / N_TEST + 1.0 / n_pool) / (1.0 / N_TEST - 1.0 / n_pool)
    print(f"\n  sigma on transferring a CV gap to the full test set "
          f"({a.reps} subsamples of {N_TEST} rows, variance scaled by "
          f"(1/n+1/N)/(1/n-1/N) = {FPC:.3f}):")
    sig = {}
    ref_o = prep(np.load(os.path.join(SUB, f"oof_{REF}.npy")), y)
    cand = TIE + [p for p in PICK if p != REF]
    subs = {n: [] for n in cand}
    cand_o = {n: prep(np.load(os.path.join(SUB, f"oof_{n}.npy")), y) for n in cand}
    for i in range(a.reps):
        idx = rng.choice(n_pool, size=N_TEST, replace=False)
        m = np.zeros(n_pool, dtype=bool)
        m[idx] = True
        rb = subset_auc(ref_o, m)
        for n in cand:
            subs[n].append(subset_auc(cand_o[n], m) - rb)
    for n in cand:
        sg = float(np.std(subs[n], ddof=1) * np.sqrt(FPC))
        sig[n] = sg
        print(f"    {n:18s} sd(subsample gap) {np.std(subs[n], ddof=1):.2e}  "
              f"=> sigma(g_test - g_cv) = {sg:.2e}")
    out["sigma_transfer"] = sig

    print("\n  Private gap vs the pick, by the partition identity "
          "private = (g - f*public)/(1-f):")
    print(f"    {'file':18s} {'CV gap':>10s} {'LB gap':>9s} " +
          "".join(f"{'f=' + str(fv):>12s}" for fv in (0.20, 0.25, 0.50)))
    priv = {}
    for n in TIE + PICK:
        g = cv[n] - cv[REF]
        p = lb[n] - lb[REF]
        row = {}
        for fv in (0.20, 0.25, 0.50):
            row[fv] = (g - fv * p) / (1 - fv)
        priv[n] = row
        print(f"    {n:18s} {g:+10.2e} {p:+9.1e} " +
              "".join(f"{row[fv]:+12.2e}" for fv in (0.20, 0.25, 0.50)))
    out["private_gap"] = {n: {str(k): float(v) for k, v in r.items()} for n, r in priv.items()}

    print("\n  Scenario table -- expected private AUC LOST by the default, vs the pick.")
    print("  (private score of a selection = MAX over the selected entries, Kaggle-standard)")
    pick_priv = {fv: max(priv[p][fv] for p in PICK) for fv in (0.20, 0.25, 0.50)}
    scen = {}
    for fv in (0.20, 0.25, 0.50):
        # limit 2, uniform 2-of-4 from the tie
        vals2 = [max(priv[a_][fv], priv[b_][fv]) for a_, b_ in itertools.combinations(TIE, 2)]
        # limit 1, uniform from the tie
        vals1 = [priv[n][fv] for n in TIE]
        scen[fv] = dict(
            lim2_mean=float(np.mean(vals2)), lim2_worst=float(np.min(vals2)),
            lim1_mean=float(np.mean(vals1)), lim1_worst=float(np.min(vals1)),
            pick=float(pick_priv[fv]))
    print(f"\n    {'f':>5s} {'limit2 E[cost]':>15s} {'limit2 worst':>13s} "
          f"{'limit1 E[cost]':>15s} {'limit1 worst':>13s}")
    for fv in (0.20, 0.25, 0.50):
        d = scen[fv]
        print(f"    {fv:5.2f} {d['pick']-d['lim2_mean']:15.2e} {d['pick']-d['lim2_worst']:13.2e} "
              f"{d['pick']-d['lim1_mean']:15.2e} {d['pick']-d['lim1_worst']:13.2e}")
    out["scenarios"] = {str(k): v for k, v in scen.items()}

    print("\n    the 6 two-of-four draws at f=0.20, cost against the pick:")
    for a_, b_ in itertools.combinations(TIE, 2):
        c = pick_priv[0.20] - max(priv[a_][0.20], priv[b_][0.20])
        print(f"      {a_:16s} + {b_:16s}  cost {c:+.2e}")

    # uncertainty on the headline number, propagated from sigma(g_test - g_cv)
    print("\n  Uncertainty on the headline, by Monte Carlo over the transfer noise "
          f"({a.reps*10} draws, f=0.20):")
    fv = 0.20
    mc2, mc1 = [], []
    for _ in range(a.reps * 10):
        pv = {}
        for n in TIE + PICK:
            g = cv[n] - cv[REF] + rng.normal(0.0, sig.get(n, 0.0))
            p = lb[n] - lb[REF] + rng.uniform(-5e-6, 5e-6) - rng.uniform(-5e-6, 5e-6)
            pv[n] = (g - fv * p) / (1 - fv)
        pk = max(pv[p] for p in PICK)
        mc2.append(pk - np.mean([max(pv[x], pv[z]) for x, z in itertools.combinations(TIE, 2)]))
        mc1.append(pk - np.mean([pv[n] for n in TIE]))
    mc2, mc1 = np.array(mc2), np.array(mc1)
    print(f"    limit 2:  cost {mc2.mean():+.2e}  sd {mc2.std(ddof=1):.2e}  "
          f"90% CI [{np.percentile(mc2,5):+.2e}, {np.percentile(mc2,95):+.2e}]  "
          f"P(cost>0) {float((mc2>0).mean()):.3f}")
    print(f"    limit 1:  cost {mc1.mean():+.2e}  sd {mc1.std(ddof=1):.2e}  "
          f"90% CI [{np.percentile(mc1,5):+.2e}, {np.percentile(mc1,95):+.2e}]  "
          f"P(cost>0) {float((mc1>0).mean()):.3f}")
    out["headline"] = dict(
        lim2_mean=float(mc2.mean()), lim2_sd=float(mc2.std(ddof=1)),
        lim2_ci=[float(np.percentile(mc2, 5)), float(np.percentile(mc2, 95))],
        lim2_p=float((mc2 > 0).mean()),
        lim1_mean=float(mc1.mean()), lim1_sd=float(mc1.std(ddof=1)),
        lim1_ci=[float(np.percentile(mc1, 5)), float(np.percentile(mc1, 95))],
        lim1_p=float((mc1 > 0).mean()))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
