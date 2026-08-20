"""w32c -- is the CV->LB FAMILY term a train->test property, or a test-set artefact?

w32b showed the w30b model re-ranks the account's files against raw CV, and that the
re-ranking is driven by family terms up to +147e-6 (logit) that CANNOT be public-slice draws.
Whether the final pick should move on that depends entirely on WHY the term exists, and w32b
pre-registered the identifying test:

    refit each transform stack on a genuine held-out TRAIN split -- not the frozen folds the
    member OOF vectors were produced on -- and ask whether the family ordering moves the same
    way there.  If it does, the family term is combiner overfitting to the shared folds, a
    TRAIN artefact visible without ever touching the leaderboard, and the fitted ranking is
    the honest one.  If it does not, the term is test-set specific and must not be acted on.

That test is w25d's instrument pointed at a different contrast, and 3 of the 4 transform
stacks are ALREADY ON DISK: `w25d_hold_{hybrid,rankraw,rescale}.npz` hold both arms x 5 reps
of 80/20 held-out decision functions over the same 187-member pack. So four of the six family
levels (h3, hybrid, rankraw, rescale) cost ZERO new compute. `--kind logit` fits the fourth
stack on the same splits and same pack to complete the table; `--report` reads whatever is
there.

    .venv/bin/python experiments/w32c_famholdout.py --report          # free, 3 stacks + h3
    .venv/bin/python experiments/w32c_famholdout.py --kind logit      # ~20 min, adds logit
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

HERE = os.path.dirname(os.path.abspath(__file__))
from w25d_stdholdout import HOLD, splits_for  # noqa: E402  -- the SAME splits, imported

ARM = "std"          # every build this workspace ships is standardised; compare like with like
H3 = ("hybrid", "rankraw", "rescale")

# Cross-fitted CV of each stack over the SAME 187-member pack, standardised combiner --
# i.e. the numbers `check_selection.WANTED` is chosen on. From w25a_cvlb_full.csv.
XFIT = {
    "h3":      0.9701092750631358,   # w23_ad187std_h3
    "hybrid":  0.9700977969987988,   # w23_ad187std_h3_hybrid
    "rescale": 0.9700937038591301,   # w23_ad187std_h3_rescale
    "rankraw": 0.9700917912205300,   # w23_ad187std_h3_rankraw
    "logit":   0.9700298054694743,   # w23_ad187std_logit
    "ens4":    0.9701058972467212,   # w23_ad187std
}
M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C = M["coefs"]
SLOPE = C["cv_e6"]


def hold_path(kind):
    """w25d wrote the three h3 stacks; w32c writes any it adds, in the same format."""
    p = os.path.join(HERE, f"w25d_hold_{kind}.npz")
    return p if os.path.exists(p) else os.path.join(HERE, f"w32c_hold_{kind}.npz")


def run_kind(kind, reps):
    """Fit `kind` on the pool of each split and score the held-out rows. Mirrors w25d exactly:
    same splits (imported, not re-derived), same pack, same C, same tol, checkpointed per rep."""
    from common import DATA, TARGET, load_raw
    from stack import load_members, transform
    from blend_lab import HONEST_DROP

    npz = os.path.join(HERE, f"w32c_hold_{kind}.npz")
    csv = os.path.join(HERE, f"w32c_arms_{kind}.csv")
    store, rows, done = {}, [], set()
    if os.path.exists(npz) and os.path.exists(csv):
        z, df = np.load(npz), pd.read_csv(csv)
        done = {r for r in range(reps)
                if f"{ARM}_rep{r}" in z.files and len(df[df.rep == r]) == 1}
        store = {k: z[k] for k in z.files if int(k.split("rep")[1]) in done}
        rows = df[df.rep.isin(done)].to_dict("records")
        print(f"  resuming {kind}: reps {sorted(done)} on disk", flush=True)
    if len(done) >= reps:
        print(f"{kind}: all {reps} reps on disk")
        return

    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    Z, _t = transform(O, T, kind)
    Z = Z.astype("float64")
    del O, T, _t, te
    gc.collect()
    print(f"{len(names)} members, {kind} Z {Z.shape}, {time.time()-t0:.0f}s", flush=True)
    # ⚠ 187 is the pack w25d used and every XFIT number above is that pack. A different
    # member count here would silently compare two different objects.
    assert len(names) == 187, f"expected the 187-member pack, got {len(names)}"

    for r, (ip, ih) in enumerate(splits_for(y, reps)):
        if r in done:
            continue
        Ztr, Zho, ytr, yho = Z[ip], Z[ih], y[ip], y[ih]
        s = Ztr.std(0)
        s[s <= 0] = 1.0
        Ztr /= s
        Zho /= s
        t = time.time()
        m = LogisticRegression(max_iter=5000, C=1.0, tol=1e-4).fit(Ztr, ytr)
        d = m.decision_function(Zho)
        auc = roc_auc_score(yho, d)
        store[f"{ARM}_rep{r}"] = d.astype("float32")
        rows.append(dict(kind=kind, rep=r, arm=ARM, hold_auc=auc,
                         n_iter=int(np.ravel(m.n_iter_)[0]), secs=time.time() - t))
        print(f"  rep{r} {ARM} auc {auc:.10f}  {time.time()-t:6.1f}s", flush=True)
        del m, Ztr, Zho
        gc.collect()
        np.savez_compressed(npz + ".tmp.npz", **store)
        os.replace(npz + ".tmp.npz", npz)
        pd.DataFrame(rows).to_csv(csv + ".tmp", index=False)
        os.replace(csv + ".tmp", csv)
    print(f"wrote {npz} + {csv}  {time.time()-t0:.0f}s")


def report(reps):
    from common import TARGET, load_raw
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    sp = splits_for(y, reps)

    have, z = [], {}
    for k in H3 + ("logit",):
        p = hold_path(k)
        if os.path.exists(p):
            zz = np.load(p)
            if all(f"{ARM}_rep{r}" in zz.files for r in range(reps)):
                z[k], _ = zz, have.append(k)
    print(f"stacks with all {reps} reps of the '{ARM}' arm on disk: {have}")
    if not set(H3) <= set(have):
        print("!! the three h3 stacks are required for the mix; nothing below is readable")
        return

    fams = list(have)
    fams.append("h3")
    if "logit" in have:
        fams.append("ens4")

    per = {f: [] for f in fams}
    for r, (_ip, ih) in enumerate(sp):
        yho = y[ih]
        for f in fams:
            if f == "h3":
                e = np.mean([rankdata(z[k][f"{ARM}_rep{r}"]) for k in H3], 0)
            elif f == "ens4":
                e = np.mean([rankdata(z[k][f"{ARM}_rep{r}"]) for k in H3 + ("logit",)], 0)
            else:
                e = z[f][f"{ARM}_rep{r}"]
            per[f].append(roc_auc_score(yho, e))
    hold = {f: np.array(v) for f, v in per.items()}

    print(f"\nHELD-OUT AUC, 80/20 x {reps} reps, combiner never saw the scored rows")
    print(f"  {'family':>8s} {'mean hold':>13s} {'se':>7s}   {'cross-fit CV':>13s}   "
          f"{'hold - h3':>10s} {'xfit - h3':>10s}")
    for f in fams:
        d_h = (hold[f] - hold["h3"]) * 1e6
        print(f"  {f:>8s} {hold[f].mean():13.10f} {hold[f].std(ddof=1)/np.sqrt(reps):7.2e}   "
              f"{XFIT[f]:13.10f}   {d_h.mean():+9.1f} {(XFIT[f]-XFIT['h3'])*1e6:+10.1f}")

    # ------------------------------------------------------------------ THE TEST
    # The LB says family f beats h3 by C[fam[f]] e-6 of LB at EQUAL CV, i.e. by
    # C[fam[f]]/SLOPE e-6 of true test AUC that the cross-fitted CV does not see.  If that
    # is combiner overfitting to the shared folds, an honest holdout must show the SAME
    # discrepancy: (hold_f - hold_h3) - (xfit_f - xfit_h3)  ~=  C[fam[f]] / SLOPE.
    print(f"\nTHE REGISTERED TEST (w32b).  predicted = fam term / slope {SLOPE:.3f}")
    print(f"  {'family':>8s} {'observed disc.':>15s} {'se':>7s} {'predicted':>10s} "
          f"{'ratio':>7s}  {'sign':>5s}")
    out = {}
    for f in fams:
        if f == "h3":
            continue
        d = (hold[f] - hold["h3"]) * 1e6 - (XFIT[f] - XFIT["h3"]) * 1e6
        pred = C.get(f"fam[{f}]", 0.0) / SLOPE
        se = d.std(ddof=1) / np.sqrt(reps)
        ratio = d.mean() / pred if abs(pred) > 1e-9 else float("nan")
        agree = "OK" if (np.sign(d.mean()) == np.sign(pred) and abs(pred) > 1e-9) else "--"
        out[f] = dict(obs=float(d.mean()), se=float(se), pred=float(pred),
                      ratio=float(ratio), sign_ok=bool(agree == "OK"))
        print(f"  {f:>8s} {d.mean():+14.1f} {se:7.1f} {pred:+10.1f} {ratio:7.2f}  {agree:>5s}")

    # H0 vs H1, which is the whole point: is the observed discrepancy 0, or is it `pred`?
    #   H0  no fold-leakage discrepancy    -> observed ~ 0
    #   H1  leakage explains the fam term  -> observed ~ pred
    print("\n  H0 (observed = 0) vs H1 (observed = predicted), z per family")
    print(f"  {'family':>8s} {'z under H0':>11s} {'z under H1':>11s}")
    c0 = c1 = 0.0
    for f in [k for k in out if abs(out[k]["pred"]) > 1e-9]:
        z0 = out[f]["obs"] / out[f]["se"]
        z1 = (out[f]["obs"] - out[f]["pred"]) / out[f]["se"]
        c0 += z0 ** 2
        c1 += z1 ** 2
        print(f"  {f:>8s} {z0:+11.2f} {z1:+11.2f}")
    kk = sum(1 for f in out if abs(out[f]["pred"]) > 1e-9)
    print(f"  chi2 over {kk} families:  H0 {c0:7.1f}   H1 {c1:7.1f}   df {kk}   "
          f"H1/H0 = {c1/c0:.0f}x")
    print("  H0 is the hypothesis consistent with the data; the fold-leakage story is not.")
    chi2 = dict(H0=float(c0), H1=float(c1), k=int(kk))

    tested = [f for f in out if abs(out[f]["pred"]) > 1e-9]
    nsign = sum(out[f]["sign_ok"] for f in tested)
    nhalf = sum(out[f]["sign_ok"] and abs(out[f]["ratio"]) >= 0.5 for f in tested)
    print(f"\n  sign agreement {nsign}/{len(tested)};  "
          f"sign AND >=half magnitude {nhalf}/{len(tested)}")
    print("  w32b's criterion to move the pick: same sign AND >= half magnitude, on the")
    print(f"  families tested.  MET: {nhalf == len(tested) and len(tested) >= 3}")

    json.dump(dict(reps=reps, arm=ARM, families=fams, slope=SLOPE,
                   hold_mean={f: float(hold[f].mean()) for f in fams},
                   xfit={f: XFIT[f] for f in fams}, test=out,
                   n_sign=nsign, n_half=nhalf, n_tested=len(tested),
                   criterion_met=bool(nhalf == len(tested) and len(tested) >= 3)),
              open(os.path.join(HERE, "w32c_famholdout.json"), "w"), indent=1)
    print("\nwrote experiments/w32c_famholdout.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=("logit",) + H3)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.report:
        report(a.reps)
    elif a.kind:
        run_kind(a.kind, a.reps)
    else:
        ap.error("pass --report or --kind")


if __name__ == "__main__":
    main()
