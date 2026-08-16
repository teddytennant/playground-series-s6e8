"""w15h: the anti-student mechanism instantiated from members we ALREADY OWN, at zero
training cost, over many (sharp teacher, smooth student) pairs.

WHY THIS EXISTS
---------------
`w15h_rebuild.py` builds one teacher/student pair and can therefore be dismissed as "your
rebuild was bad". This script removes that escape route. The mechanism's content is:

    take a sharp strong model, subtract a smoother model's ranking of the same rows,
    signed-square the residual, add 10% of it to a stack

and the pack already contains both halves. `lookup`, `naji03`, `xgb_latcat` and the
`bolt_lookup_v2` family are exact-value lattice readers -- as sharp as anything rayk's
teacher can be. `golem_c` (spline GAM), `logreg`, `linlat`, `knn` and the RealMLPs are
genuinely smooth. w15e measured that the members most aligned with rayk's own correction
are exactly this smooth set, all at rho ~= -0.07, which is the reason to expect these pairs
to span the same direction.

So every (sharp, smooth) pair gives a correction of the same family for free, and the
question "does a sharp-minus-smooth residual help our saturated stack" can be asked
~40 times instead of once.

CONTROLS
--------
Each pair's correction is scored against its OWN matched permutation null, because w15e
section 6 measured that a signal-free correction of this size costs about -1.1e-5 -- the
null is displaced, so a raw delta is uninterpretable on its own. The reported statistic is
the z of the real delta against that pair's permutation distribution.

NO TRAINING. Pure numpy over stored OOF vectors on the frozen folds.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, LIB, OOF, SUB, TARGET, get_folds  # noqa: E402

NPZ = os.path.join(DATA, "w15e", "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
OUT = os.path.join(ROOT, "experiments", "w15h_pairs.json")
DIRS = {"lib": os.path.join(LIB, "oof"), "own": OOF,
        "ext": os.path.join(DATA, "ext_members"), "ext2": os.path.join(DATA, "ext_members2")}

SHARP = ["lib:lookup", "lib:naji03", "lib:naji05", "own:xgb_latcat",
         "ext2:bolt_lookup_v2_s42", "lib:lat_lgbm", "own:lgbm_tuned_lat"]
SMOOTH = ["ext:golem_c", "lib:logreg", "lib:knn", "own:linlat", "lib:pub_rmlp",
          "lib:hgb", "own:lgbm_stump_lat_frac", "ext2:bolt_lgb_raw_d4"]
ANCHORS = ["blend159av_h3", "blend158_logit", "blend150fx_hybrid"]
W_ANTI = 0.10
N_PERM = 60


def pct(v):
    v = np.asarray(v, np.float64)
    return rankdata(v) / len(v)


def zr(v):
    r = rankdata(np.asarray(v, np.float64))
    return (r - r.mean()) / r.std()


def build_anti(residual, reference):
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    reference_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    return (test_sq - reference_sq.mean()) * reference.std() / reference_sq.std()


def load(spec):
    tag, nm = spec.split(":", 1)
    p = os.path.join(DIRS[tag], f"oof_{nm}.npy")
    return np.load(p) if os.path.exists(p) else None


def main():
    t0 = time.time()
    y = pd.read_csv(os.path.join(DATA, "train.csv"))[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)

    sharp = {s: load(s) for s in SHARP}
    smooth = {s: load(s) for s in SMOOTH}
    sharp = {k: v for k, v in sharp.items() if v is not None and v.shape == y.shape}
    smooth = {k: v for k, v in smooth.items() if v is not None and v.shape == y.shape}
    print("sharp teachers:")
    for k, v in sharp.items():
        print(f"  {k:32s} OOF {roc_auc_score(y, v):.6f}")
    print("smooth students:")
    for k, v in smooth.items():
        print(f"  {k:32s} OOF {roc_auc_score(y, v):.6f}")

    anchors = {a: np.load(os.path.join(SUB, f"oof_{a}.npy")) for a in ANCHORS}
    base_pct = {a: pct(v) for a, v in anchors.items()}
    base_auc = {a: roc_auc_score(y, v) for a, v in anchors.items()}

    # One shared permutation null per anchor: the toll depends on the anchor and on the
    # correction's marginal, and every pair's correction has the same marginal by
    # construction (build_anti renormalises to the same reference statistics).
    rg = np.random.default_rng(1234)
    probe = build_anti(pct(sharp[SHARP[0]]) - pct(smooth[SMOOTH[0]]), reference)
    null = {}
    for a in ANCHORS:
        d = np.array([roc_auc_score(y, base_pct[a] + W_ANTI * rg.permutation(probe)) - base_auc[a]
                      for _ in range(N_PERM)])
        null[a] = (float(d.mean()), float(d.std(ddof=1)))
        print(f"\nnull for {a:20s}: {d.mean()*1e6:+7.2f} +/- {d.std(ddof=1)*1e6:.2f} e-6"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for ts, tv in sharp.items():
        for ss, sv in smooth.items():
            resid = pct(tv) - pct(sv)
            anti = build_anti(resid, reference)
            r = dict(teacher=ts, student=ss,
                     rho_teacher_student=float(zr(tv) @ zr(sv) / n),
                     rho_anti_vs_rayk_family=None)
            for a in ANCHORS:
                cor = base_pct[a] + W_ANTI * anti
                d = roc_auc_score(y, cor) - base_auc[a]
                m, s = null[a]
                r[f"delta_{a}"] = float(d)
                r[f"z_{a}"] = float((d - m) / s)
                if a == "blend159av_h3":
                    per = [roc_auc_score(y[va], cor[va]) - roc_auc_score(y[va], base_pct[a][va])
                           for _, va in folds]
                    r["folds_pos"] = int(sum(x > 0 for x in per))
            rows.append(r)
            print(f"  {ts:28s} - {ss:26s}  rho {r['rho_teacher_student']:.4f}   "
                  + "  ".join(f"{a.split('_')[0][:8]} z {r['z_'+a]:+6.2f}" for a in ANCHORS),
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "experiments", "w15h_pairs.csv"), index=False)
    summary = dict(
        n_pairs=len(df), null={a: null[a] for a in ANCHORS},
        z_main_mean=float(df["z_blend159av_h3"].mean()),
        z_main_max=float(df["z_blend159av_h3"].max()),
        n_pos_z2=int((df["z_blend159av_h3"] > 2).sum()),
        n_pos_delta=int((df["delta_blend159av_h3"] > 0).sum()),
        delta_main_max=float(df["delta_blend159av_h3"].max()),
        best=df.loc[df["z_blend159av_h3"].idxmax()].to_dict(),
    )
    json.dump(summary, open(OUT, "w"), indent=2, default=str)
    print(f"\n=== {len(df)} pairs, anchor blend159av_h3 ===")
    print(f"  z: mean {summary['z_main_mean']:+.2f}  max {summary['z_main_max']:+.2f}   "
          f"pairs with z>2: {summary['n_pos_z2']}/{len(df)}   "
          f"raw delta > 0: {summary['n_pos_delta']}/{len(df)}")
    print(f"  best pair: {summary['best']['teacher']} - {summary['best']['student']}  "
          f"delta {float(summary['best']['delta_blend159av_h3'])*1e6:+.2f}e-6")
    print(f"wrote {OUT}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
