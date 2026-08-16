"""w15h: put raykkretzschmar's transductive anti-student correction onto LABELLED rows.

THE CLAIM UNDER ATTACK
----------------------
w15e §3-4 (and w15b's next-run item 3, and w15a's "only live direction") rest on one
positive claim: the teacher-minus-student rank residual published as
`raykkretzschmar/s6e8-transductive-anti-student-signals` is a genuinely orthogonal signal
worth +0.000018 to +0.000036 on an OOF stack, and it is *unmeasurable here* because it is
defined only on the 296,302 test rows.

That last clause is what makes the claim unfalsifiable in this workspace, and it is false.

WHY IT IS FALSE
---------------
`anti_student` is a deterministic function of the 12 predictor columns: the teacher is a
LightGBM on X, the student is a LightGBM on X, and their rank difference is therefore a
fixed function f(X) evaluated at the test rows. Fitting f from (X_test, anti_student) uses
NO LABELS AT ALL -- neither train labels nor test labels -- so the fitted f may be
evaluated on the 691,369 labelled train rows without any leakage whatsoever, and the
resulting correction scored against the frozen folds.

The reconstruction is validated out of fold WITHIN the test rows before it is transferred,
so the attenuation is measured rather than assumed.

WHAT IS MEASURED
----------------
1. How well f(X) reconstructs the published correction (5-fold within test, held out).
2. The DeltaAUC of `pct(anchor) + 0.10 * anti_hat` on 691,369 labelled rows, per fold.
3. A matched permutation control -- the same anti_hat values with rows shuffled -- because
   w15e §6 measured that a signal-free correction of this size costs about -1.1e-5, so the
   null for this move is NOT centred on zero and the raw delta cannot be read directly.
4. An ANCHOR LADDER from a single member up to our cross-fitted best blend. The workspace's
   documented failure mode is that a gain measured on a weaker representation does not
   survive transfer to a saturated one; if that is what is happening here, the delta falls
   with anchor strength and crosses zero somewhere below 0.97005.
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

os.environ.setdefault("OMP_NUM_THREADS", "3")
import lightgbm as lgb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, LIB, OOF, SUB, TARGET, get_folds, load_raw  # noqa: E402

NPZ = os.path.join(DATA, "w15e", "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
OUT = os.path.join(ROOT, "experiments", "w15h_transfer.json")
NTHREADS = 3
W_ANTI = 0.10
N_TEST = 296_302


def pct(v):
    v = np.asarray(v, np.float64)
    return rankdata(v) / len(v)


def zr(v):
    r = rankdata(np.asarray(v, np.float64))
    return (r - r.mean()) / r.std()


def target_free_matrix():
    """Assemble the 112 target-free cache columns for train (original order) and test.

    The TE_ blocks are fold-dependent and carry the label; everything else in the cache
    (raw columns, constrained imputation, missingness, generator identities, and the CT_
    exact-value / pair / triple count families) is computed on train+test together and is
    target-free. Fold 0's (Xa, Xb) partition covers every train row exactly once.
    """
    cols = json.load(open(os.path.join(ROOT, "cache", "cols.json")))
    keep = np.array([i for i, c in enumerate(cols) if not c.startswith("TE_")])
    names = [cols[i] for i in keep]
    Xa = np.load(os.path.join(ROOT, "cache", "f0_Xa.npy"), mmap_mode="r")
    Xb = np.load(os.path.join(ROOT, "cache", "f0_Xb.npy"), mmap_mode="r")
    Xt = np.load(os.path.join(ROOT, "cache", "f0_Xt.npy"), mmap_mode="r")
    n_train = Xa.shape[0] + Xb.shape[0]
    Xtr = np.empty((n_train, len(keep)), np.float32)
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    a_idx, b_idx = folds[0]
    Xtr[a_idx] = np.asarray(Xa[:, keep])
    Xtr[b_idx] = np.asarray(Xb[:, keep])
    Xte = np.asarray(Xt[:, keep], np.float32)
    return Xtr, Xte, names, y, folds


def build_anti(residual, reference):
    """raykkretzschmar's renormalisation, transcribed from his cell 29.

    `reference` is his own two-holdout `reference_contrast`; all of its statistics are
    fixed constants, so the same map can be applied to any residual vector.
    """
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    reference_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    return (test_sq - reference_sq.mean()) * reference.std() / reference_sq.std()


def delta_auc(y, base, anti, w=W_ANTI):
    b = pct(base)
    return roc_auc_score(y, b + w * anti) - roc_auc_score(y, b)


def main():
    t0 = time.time()
    res = {}

    print("=== loading ===", flush=True)
    Xtr, Xte, names, y, folds = target_free_matrix()
    print(f"target-free design: train {Xtr.shape}  test {Xte.shape}", flush=True)

    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)
    teacher = pct(z["test_teacher"].astype(np.float64))
    student = pct(z["test_student"].astype(np.float64))
    resid_true = teacher - student
    anti_true = build_anti(resid_true, reference)
    print(f"published residual sd {resid_true.std():.6f}   "
          f"anti_student sd {anti_true.std():.6f}", flush=True)
    res["anti_true_sd"] = float(anti_true.std())

    # ---------------------------------------------------------------- 1. reconstruct
    # Target: the raw rank residual (smoother and better conditioned than the
    # signed-square). NO LABELS ANYWHERE IN THIS FIT.
    params = dict(objective="l2", learning_rate=0.05, num_leaves=255,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_bin=1023, verbose=-1,
                  num_threads=NTHREADS, seed=42)
    ROUNDS = 2000

    print("\n=== 1. reconstruct f(X) from (X_test, residual), 5-fold within test ===",
          flush=True)
    rng = np.random.default_rng(42)
    fold_of = rng.permutation(np.arange(N_TEST) % 5)
    r_hat_oof = np.empty(N_TEST, np.float64)
    for k in range(5):
        m = fold_of == k
        ds = lgb.Dataset(Xte[~m], label=resid_true[~m], feature_name=names)
        bst = lgb.train(params, ds, num_boost_round=ROUNDS)
        r_hat_oof[m] = bst.predict(Xte[m])
        print(f"  fold {k} done  {time.time()-t0:.0f}s", flush=True)
    ss = 1.0 - ((resid_true - r_hat_oof) ** 2).sum() / ((resid_true - resid_true.mean()) ** 2).sum()
    rho = float(zr(r_hat_oof) @ zr(resid_true) / N_TEST)
    anti_hat_oof = build_anti(r_hat_oof, reference)
    rho_anti = float(zr(anti_hat_oof) @ zr(anti_true) / N_TEST)
    print(f"  held-out R^2 {ss:.4f}   spearman(r_hat, r) {rho:.4f}   "
          f"spearman(anti_hat, anti) {rho_anti:.4f}", flush=True)
    res.update(recon_r2=float(ss), recon_rho=rho, recon_rho_anti=rho_anti)

    # How much of the RANKING ACTION survives reconstruction? Apply both corrections to
    # our own test-space base and compare the two perturbed rankings.
    base_test = pd.read_csv(os.path.join(SUB, "blend159av_h3.csv"))[TARGET].to_numpy()
    br = pct(base_test)
    p_true = br + W_ANTI * anti_true
    p_hat = br + W_ANTI * anti_hat_oof
    d_true = zr(p_true) - zr(br)
    d_hat = zr(p_hat) - zr(br)
    action = float(np.corrcoef(d_true, d_hat)[0, 1])
    print(f"  correlation of the two RANK PERTURBATIONS on test: {action:.4f}", flush=True)
    res["perturbation_corr"] = action

    # ---------------------------------------------------------------- 2. transfer
    print("\n=== 2. fit on all test rows, evaluate on the 691,369 labelled rows ===",
          flush=True)
    ds = lgb.Dataset(Xte, label=resid_true, feature_name=names)
    bst = lgb.train(params, ds, num_boost_round=ROUNDS)
    imp = pd.Series(bst.feature_importance("gain"), index=names).sort_values(ascending=False)
    print("  what the residual is made of (top 15 by gain):")
    for nm, g in imp.head(15).items():
        print(f"    {nm:52s} {g / imp.sum() * 100:5.2f}%", flush=True)
    res["recon_importance"] = {k: float(v / imp.sum()) for k, v in imp.head(25).items()}
    r_hat_train = bst.predict(Xtr)
    anti_hat_train = build_anti(r_hat_train, reference)
    np.save(os.path.join(ROOT, "experiments", "w15h_anti_hat_train.npy"), anti_hat_train)
    np.save(os.path.join(ROOT, "experiments", "w15h_anti_hat_test_oof.npy"), anti_hat_oof)
    print(f"  transferred anti_hat sd {anti_hat_train.std():.6f} "
          f"(published test vector {anti_true.std():.6f})   {time.time()-t0:.0f}s",
          flush=True)

    # ---------------------------------------------------------------- 3. anchor ladder
    print("\n=== 3. anchor ladder ===", flush=True)
    anchors = []
    man = pd.read_csv(os.path.join(LIB, "manifest.csv")).set_index("model")["oof_auc"]
    for nm in ["lgbm_tuned_lat", "xgb_latcat", "cat_lat"]:
        p = os.path.join(OOF, f"oof_{nm}.npy")
        if os.path.exists(p):
            anchors.append((f"member:{nm}", np.load(p)))
    for nm in ["lookup", "naji03", "tabm_seed3"]:
        p = os.path.join(LIB, "oof", f"oof_{nm}.npy")
        if os.path.exists(p):
            anchors.append((f"lib:{nm}", np.load(p)))
    for nm in ["blend150fx_logit", "blend158_logit", "blend150fx_hybrid",
               "blend153", "blend159av_rankraw", "blend156_h3", "blend159av_h3"]:
        p = os.path.join(SUB, f"oof_{nm}.npy")
        if os.path.exists(p):
            anchors.append((f"blend:{nm}", np.load(p)))

    ladder = []
    for nm, v in anchors:
        base_auc = roc_auc_score(y, v)
        d = delta_auc(y, v, anti_hat_train)
        per = []
        b = pct(v)
        cor = b + W_ANTI * anti_hat_train
        for _, va in folds:
            per.append(roc_auc_score(y[va], cor[va]) - roc_auc_score(y[va], b[va]))
        ladder.append(dict(anchor=nm, base_auc=float(base_auc), delta=float(d),
                           folds_pos=int(sum(x > 0 for x in per)),
                           per_fold=[float(x) for x in per]))
        print(f"  {nm:28s} base {base_auc:.6f}   delta {d*1e6:+8.2f}e-6   "
              f"folds+ {ladder[-1]['folds_pos']}/5", flush=True)
    res["ladder"] = ladder

    # ---------------------------------------------------------------- 4. null control
    print("\n=== 4. matched permutation control (the toll null) ===", flush=True)
    nulls = {}
    for nm in ["blend159av_h3", "blend150fx_logit"]:
        v = np.load(os.path.join(SUB, f"oof_{nm}.npy"))
        b = pct(v)
        a0 = roc_auc_score(y, b)
        rg = np.random.default_rng(7)
        draws = []
        for i in range(200):
            perm = rg.permutation(anti_hat_train)
            draws.append(roc_auc_score(y, b + W_ANTI * perm) - a0)
        draws = np.asarray(draws)
        real = delta_auc(y, v, anti_hat_train)
        nulls[nm] = dict(null_mean=float(draws.mean()), null_sd=float(draws.std(ddof=1)),
                         q05=float(np.quantile(draws, .05)),
                         q95=float(np.quantile(draws, .95)),
                         real=float(real),
                         z=float((real - draws.mean()) / draws.std(ddof=1)),
                         p_greater=float((draws >= real).mean()))
        print(f"  {nm:22s} null {draws.mean()*1e6:+7.2f} +/- {draws.std(ddof=1)*1e6:.2f} e-6"
              f"   real {real*1e6:+7.2f}e-6   z {nulls[nm]['z']:+.2f}"
              f"   P(null>=real) {nulls[nm]['p_greater']:.3f}", flush=True)
    res["nulls"] = nulls

    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}   {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
