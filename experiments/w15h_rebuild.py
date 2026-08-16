"""w15h: rebuild the anti-student MECHANISM on our own frozen folds, with the ablation
raykkretzschmar never ran.

WHY A SECOND INSTRUMENT
-----------------------
`w15h_transfer.py` moves rayk's own published correction onto labelled rows by fitting
f(X) from (X_test, residual) -- label-free, so leakage-free -- and scoring it on the
691,369 labelled rows. Its weakness is measured attenuation: the reconstruction is not
perfect, so a real effect arrives shrunk.

This script has the opposite trade. It rebuilds the mechanism from scratch, so there is no
attenuation at all, at the cost of not being rayk's exact vector. The two together bracket
the question, and each one's weakness is the other's strength.

THE CONSTRUCTION, from his notebook cell 28 and the markdown above it
--------------------------------------------------------------------
  teacher   a strong sharp model.  Ours is an existing library member with honest OOF, so
            it costs nothing and cannot leak: its OOF value on a row comes from a fit that
            never saw that row.
  student   "a second LightGBM regresses the teacher's percentile ranks with a smoother
            configuration", trained on the outer-training rows' teacher OOF targets AND
            "the unlabeled outer validation rows with their teacher predictions at weight
            0.5".  Since our teacher's value on a held-out row IS its out-of-fold
            prediction, that is exactly: train on all rows, held-out fold downweighted to
            0.5.  No labels enter the student at any point.
  residual  pct(teacher) - pct(student), signed-squared and renormalised to his published
            `reference_contrast` statistics -- the same map, transcribed from his cell 29.

THE ABLATION HE NEVER RAN  (arm `inductive`)
--------------------------------------------
The entire case for this object in w15e §3 is that it is TRANSDUCTIVE and therefore
"orthogonal to our 168 inductive members by construction". That argument rests on the
carried unlabeled rows doing real work. So the student is also fitted with those rows
removed entirely -- a strictly inductive student, everything else identical. If the two
corrections agree, the transductive ingredient is decorative and the object is an ordinary
inductive function of the 12 columns, which is a class this workspace has already searched
under matched controls (w15b section 6).

CONTROLS
--------
  * matched permutation of the correction (the -1.1e-5 toll null from w15e section 6);
  * an anchor ladder from a single member up to the cross-fitted best blend, because the
    documented failure mode here is a gain that does not survive transfer to a saturated
    representation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "3")
import lightgbm as lgb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import DATA, LIB, OOF, SUB, TARGET, get_folds  # noqa: E402
from w15h_transfer import build_anti, pct, target_free_matrix, zr  # noqa: E402

NPZ = os.path.join(DATA, "w15e", "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
NTHREADS = 3
W_ANTI = 0.10
N_TEST = 296_302

# Deliberately smoother than any teacher in the pack: few leaves, large leaves, and a
# coarse histogram so it CANNOT read the exact quantisation lattice.
STUDENT = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=5.0, max_bin=63, verbose=-1, num_threads=NTHREADS, seed=42)
ROUNDS = 500

TEACHER_DIRS = {"lib": os.path.join(LIB, "oof"), "own": OOF,
                "ext": os.path.join(DATA, "ext_members"),
                "ext2": os.path.join(DATA, "ext_members2")}


def load_teacher(spec):
    tag, nm = spec.split(":", 1)
    d = TEACHER_DIRS[tag]
    return (np.load(os.path.join(d, f"oof_{nm}.npy")),
            np.load(os.path.join(d, f"test_{nm}.npy")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="lib:lookup")
    ap.add_argument("--tag", default="lookup")
    a = ap.parse_args()

    t0 = time.time()
    res = {"teacher": a.teacher}
    Xtr, Xte, names, y, folds = target_free_matrix()
    n = len(y)

    t_oof, t_test = load_teacher(a.teacher)
    print(f"teacher {a.teacher}: OOF AUC {roc_auc_score(y, t_oof):.6f}", flush=True)
    res["teacher_auc"] = float(roc_auc_score(y, t_oof))

    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)
    resid_pub = pct(z["test_teacher"].astype(np.float64)) - pct(z["test_student"].astype(np.float64))

    tgt_tr = pct(t_oof)            # the teacher's percentile ranks on the labelled rows
    tgt_te = pct(t_test)

    # ------------------------------------------------------------------ students
    s_pred = {}
    for arm in ("transductive", "inductive"):
        sp = np.empty(n, np.float64)
        for k, (tr_idx, va_idx) in enumerate(folds):
            if arm == "transductive":
                # every row, held-out fold carried at half weight with its (out-of-fold)
                # teacher value as target -- rayk's "unlabeled outer validation rows"
                w = np.ones(n)
                w[va_idx] = 0.5
                ds = lgb.Dataset(Xtr, label=tgt_tr, weight=w, feature_name=names)
            else:
                ds = lgb.Dataset(Xtr[tr_idx], label=tgt_tr[tr_idx], feature_name=names)
            bst = lgb.train(STUDENT, ds, num_boost_round=ROUNDS)
            sp[va_idx] = bst.predict(Xtr[va_idx])
            print(f"  {arm:12s} fold {k} {time.time()-t0:.0f}s", flush=True)
        s_pred[arm] = sp
        rho = float(zr(sp) @ zr(t_oof) / n)
        print(f"  {arm:12s} student vs teacher spearman {rho:.5f}   "
              f"student solo AUC {roc_auc_score(y, sp):.6f}", flush=True)
        res[f"{arm}_student_rho_teacher"] = rho
        res[f"{arm}_student_auc"] = float(roc_auc_score(y, sp))

    r_trans = tgt_tr - pct(s_pred["transductive"])
    r_ind = tgt_tr - pct(s_pred["inductive"])
    res["ablation_rho_trans_vs_ind"] = float(zr(r_trans) @ zr(r_ind) / n)
    print(f"\nABLATION: spearman(transductive residual, inductive residual) = "
          f"{res['ablation_rho_trans_vs_ind']:.5f}", flush=True)

    # ------------------------------------------------- faithfulness gate on test rows
    print("\n=== faithfulness: same construction on the test rows, vs rayk's own ===",
          flush=True)
    mass = 0.2450247
    Xall = np.vstack([Xtr, Xte])
    lab = np.concatenate([tgt_tr, tgt_te])
    w = np.concatenate([np.ones(n), np.full(N_TEST, mass)])
    bst = lgb.train(STUDENT, lgb.Dataset(Xall, label=lab, weight=w, feature_name=names),
                    num_boost_round=ROUNDS)
    s_te = bst.predict(Xte)
    r_te = pct(t_test) - pct(s_te)
    gate = float(zr(r_te) @ zr(resid_pub) / N_TEST)
    print(f"  spearman(our test residual, rayk's published test residual) = {gate:.4f}",
          flush=True)
    res["faithfulness_rho"] = gate
    np.save(os.path.join(ROOT, "experiments", f"w15h_rebuild_{a.tag}_resid_test.npy"), r_te)

    # ------------------------------------------------------------------ anchor ladder
    anti = {arm: build_anti(r, reference) for arm, r in
            (("transductive", r_trans), ("inductive", r_ind))}
    np.save(os.path.join(ROOT, "experiments", f"w15h_rebuild_{a.tag}_anti_train.npy"),
            anti["transductive"])

    anchors = []
    for nm in ["lib:lookup", "lib:naji03", "lib:tabm_seed3"]:
        tag, base = nm.split(":", 1)
        p = os.path.join(TEACHER_DIRS[tag], f"oof_{base}.npy")
        if os.path.exists(p) and nm != a.teacher:
            anchors.append((nm, np.load(p)))
    for nm in ["blend150fx_logit", "blend158_logit", "blend150fx_hybrid", "blend153",
               "blend159av_rankraw", "blend156_h3", "blend159av_h3"]:
        p = os.path.join(SUB, f"oof_{nm}.npy")
        if os.path.exists(p):
            anchors.append((f"blend:{nm}", np.load(p)))

    print("\n=== anchor ladder ===", flush=True)
    rg = np.random.default_rng(11)
    ladder = []
    for nm, v in anchors:
        b = pct(v)
        a0 = roc_auc_score(y, b)
        row = dict(anchor=nm, base_auc=float(a0))
        for arm in ("transductive", "inductive"):
            cor = b + W_ANTI * anti[arm]
            d = roc_auc_score(y, cor) - a0
            per = [roc_auc_score(y[va], cor[va]) - roc_auc_score(y[va], b[va])
                   for _, va in folds]
            row[f"delta_{arm}"] = float(d)
            row[f"folds_pos_{arm}"] = int(sum(x > 0 for x in per))
            row[f"per_fold_{arm}"] = [float(x) for x in per]
        draws = np.array([roc_auc_score(y, b + W_ANTI * rg.permutation(anti["transductive"])) - a0
                          for _ in range(120)])
        row.update(null_mean=float(draws.mean()), null_sd=float(draws.std(ddof=1)),
                   z=float((row["delta_transductive"] - draws.mean()) / draws.std(ddof=1)),
                   p_greater=float((draws >= row["delta_transductive"]).mean()))
        ladder.append(row)
        print(f"  {nm:26s} base {a0:.6f}  trans {row['delta_transductive']*1e6:+7.2f}e-6 "
              f"({row['folds_pos_transductive']}/5)  ind {row['delta_inductive']*1e6:+7.2f}e-6"
              f"  null {draws.mean()*1e6:+6.2f}+/-{draws.std(ddof=1)*1e6:.2f}"
              f"  z {row['z']:+.2f}  p {row['p_greater']:.3f}", flush=True)
    res["ladder"] = ladder

    res["elapsed_s"] = time.time() - t0
    out = os.path.join(ROOT, "experiments", f"w15h_rebuild_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}   {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
