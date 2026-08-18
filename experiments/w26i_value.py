"""w26i: what are the two new CatBoost members worth INTO THE PACK, and are they duplicates?

Pre-registered at experiments/w26_prereg.txt §E, before either member was built.

WHAT THIS IS FOR
----------------
Solo AUC does not decide anything here -- solo-to-stack pass-through in this workspace is
~1.4%, and the member with the best solo number in the whole 22-member import was NOT the
one that paid. The two quantities that have ever predicted marginal value are (a) maxcorr
against the members already held and (b) the paired 50/50 delta itself. Both run here, in
that order, per prereg R-E3.

THE INSTRUMENT IS w21b's / w20d's, DELIBERATELY UNCHANGED
---------------------------------------------------------
Paired 50/50 stratified splits, the same rows scored with and without the member, hybrid
transform, C=1.0, 5 reps. Split noise is ~2e-4 and the effects being looked for are ~2e-6,
so nothing here is readable except as a within-rep difference. A delta whose SIGN FLIPS
across reps is a null whatever its mean -- w20d's `nn` group is the worked example.

TWO BASES, ON PURPOSE
---------------------
  base165  members whose name does not start with `ad_`, i.e. w21b's exact base. The ONLY
           thing measured against it is the `cat4` REPRODUCTION GATE, which must return
           w20d's +0.000041 or this is not the same instrument and no row below is
           comparable to anything in RESEARCH.md (prereg R-E2).
  pack     every member currently in the default pack, i.e. what a real build would stack.
           The new members are measured against THIS. Measuring them against base165 would
           overstate them, because base165 is missing the 22 imports that already span some
           of the same directions.

The new members live OUTSIDE oof/ (see --new-dir). That is not tidiness: every reproduction
gate on file is stated against a fixed member COUNT, and dropping a new member into oof/
would change the pack under w26f, w26h and blend_lab all at once.

    w26i_value.py --new-dir data/ext_members4 --reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, ROOT, TARGET, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402

HONEST_DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac")
EXP = os.path.join(ROOT, "experiments")
CAT4_W20D = ["ad_catnative", "ad_gcatlr02", "ad_gcatd8", "ad_gcatseed7"]
W20D_CAT4 = 0.000041          # the published value the gate must return
NEW = ["cat_native_ctr2", "cat_natlat"]      # default; --new-names overrides

# w26 slot 6 made this reusable for the XGBoost pair without changing ANY w26i behaviour:
# `--new-dir` still defaults to ext_members4 and `--new-names` still defaults to the two
# CatBoosts, and ext_members4 is only added to the base pack when it is NOT the new-dir,
# so w26i's own invocation loads the identical member set it always did.
PREREG_E3 = ("PREREG \u00a7E3 said: each member 0 to +4e-6 (ctr2) / 0 to +3e-6 (natlat),\n"
             "the pair together +1 to +7e-6 on the combiner, modal +2.5e-6.")


def fit_score(Z_tr, y_tr, Z_te, y_te, C):
    m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(Z_te)[:, 1])


def main():
    global NEW
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default=os.path.join(DATA, "ext_members4"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--transform", default="hybrid")
    ap.add_argument("--out", default="w26i_value")
    ap.add_argument("--new-names", default=",".join(NEW),
                    help="comma-separated member names expected in --new-dir")
    ap.add_argument("--prereg-note", default=PREREG_E3,
                    help="the registered prior, echoed above the verdicts so the table "
                         "is always read against what was written before the run")
    a = ap.parse_args()

    NEW = [n for n in a.new_names.split(",") if n]

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = [os.path.join(DATA, d)
             for d in ("ext_members", "ext_members2", "ext_members3", "ext_members4")]
    extra = [d for d in extra if os.path.isdir(d)]
    if os.path.abspath(a.new_dir) not in {os.path.abspath(d) for d in extra}:
        extra.append(a.new_dir)
    names, O, T = load_members(y, len(te), extra_dirs=tuple(extra), drop=set(HONEST_DROP))
    idx = {n: i for i, n in enumerate(names)}

    present = [n for n in NEW if n in idx]
    if not present:
        raise SystemExit(f"neither of {NEW} is in {a.new_dir} -- nothing to measure")
    pack = [n for n in names if n not in NEW]
    base165 = [n for n in pack if not n.startswith("ad_")]
    print(f"{len(names)} loaded = {len(pack)} pack + {len(present)} new {present}", flush=True)
    print(f"  base165 for the gate: {len(base165)} members", flush=True)
    for m in CAT4_W20D:
        if m not in idx:
            raise SystemExit(f"gate member {m} missing -- prereg R-E2 cannot be honoured")

    # ---- (1) maxcorr screen, BEFORE the paired test (prereg R-E3) -------------------
    # Spearman on the OOF column, which is what the AUC-side redundancy actually is.
    from scipy.stats import rankdata
    R = np.empty_like(O, dtype=np.float32)
    for j in range(O.shape[1]):
        R[:, j] = rankdata(O[:, j])
    R -= R.mean(0)
    R /= (np.linalg.norm(R, axis=0) + 1e-12)
    corr = {}
    for nm in present:
        c = R[:, idx[nm]] @ R
        c[idx[nm]] = -1.0
        for other in NEW:                      # a new member's twin is not "the pack"
            if other != nm and other in idx:
                c[idx[other]] = -1.0
        k = int(np.argmax(c))
        corr[nm] = dict(maxcorr=float(c[k]), nearest=names[k],
                        solo=float(roc_auc_score(y, O[:, idx[nm]])))
        flag = "NEAR-DUPLICATE (prereg R-E3)" if c[k] > 0.999 else "ok"
        print(f"  {nm:16s} solo {corr[nm]['solo']:.6f}  maxcorr {c[k]:.6f} "
              f"vs {names[k]}  [{flag}]", flush=True)
    del R

    t0 = time.time()
    Z, _ = transform(O, T, a.transform)
    del O, T
    print(f"transform {a.transform} in {time.time()-t0:.0f}s", flush=True)

    cfg = {"base165": base165, "gate_cat4": base165 + CAT4_W20D, "pack": pack}
    for nm in present:
        cfg[nm] = pack + [nm]
    if len(present) == 2:
        cfg["both"] = pack + present

    rows = []
    for rep in range(a.reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        r = {"rep": rep}
        for cname, mem in cfg.items():
            cols = [idx[m] for m in mem]
            t1 = time.time()
            r[cname] = fit_score(Z[np.ix_(iA, cols)], y[iA],
                                 Z[np.ix_(iB, cols)], y[iB], a.C)
            print(f"  rep {rep} {cname:16s} n_mem {len(mem):3d} AUC {r[cname]:.6f} "
                  f"({time.time()-t1:.0f}s)", flush=True)
        rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(EXP, a.out + ".csv"), index=False)

    df = pd.DataFrame(rows)
    print("\nheld-out AUC per 50/50 split")
    print(df.to_string(index=False, float_format="%.6f"))

    # ---- (2) the reproduction gate decides whether anything below is readable -------
    dg = (df["gate_cat4"] - df["base165"]).to_numpy()
    sd = max(dg.std(ddof=1), 1e-6)
    gate_ok = abs(dg.mean() - W20D_CAT4) < 3 * sd
    print(f"\nREPRODUCTION GATE (prereg R-E2) -- w20d's cat4 row was {W20D_CAT4:+.6f}")
    print(f"  here {dg.mean():+.6f} +/- {dg.std(ddof=1):.6f} over {len(dg)} reps"
          f"  -> {'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        print("  !! GATE FAILED. Registered in advance: a miss invalidates the whole table")
        print("     below and none of it is comparable to RESEARCH.md. Do not quote it.")

    print("\nPAIRED vs the pack (same rows with and without the member)")
    out = {}
    for c in df.columns:
        if c in ("rep", "base165", "gate_cat4", "pack"):
            continue
        d = (df[c] - df["pack"]).to_numpy()
        n_new = len(cfg[c]) - len(pack)
        ok = (d > 0).all() or (d < 0).all()
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[c] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se),
                      n=n_new, per_member=float(d.mean() / n_new), consistent=bool(ok),
                      t=float(d.mean() / se) if se > 0 else float("nan"))
        print(f"  {c:16s} n {n_new}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}"
              f"  per member {d.mean()/n_new:+.2e}  t {out[c]['t']:+5.2f}"
              f"  [{'consistent' if ok else 'SIGN FLIPS'}]")

    print("\n" + a.prereg_note)
    print("Anything above +10e-6 is disbelieved on sight and re-run on fresh reps.")
    for c, v in out.items():
        verdict = ("DISBELIEVE, re-run on fresh reps (§E5)" if v["mean"] > 10e-6
                   else "clears the null" if v["consistent"] and v["mean"] > 0
                   else "NULL")
        print(f"  {c:16s} {verdict}")

    json.dump(dict(gate=dict(mean=float(dg.mean()), sd=float(dg.std(ddof=1)),
                             published=W20D_CAT4, passed=bool(gate_ok)),
                   maxcorr=corr, paired=out, reps=a.reps, C=a.C,
                   transform=a.transform, n_pack=len(pack), new=present),
              open(os.path.join(EXP, a.out + ".json"), "w"), indent=2)
    print(f"\nwrote {a.out}.csv / .json")


if __name__ == "__main__":
    main()
