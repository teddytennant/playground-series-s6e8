"""w15f stage 3: price the rebuilt transductive correction on the frozen folds.

Every number here is against a matched control, because w15e measured that a small additive
rank correction is NOT a zero-mean move: it degrades a strict ranking wherever it is
uninformative, at -1.1e-5 for a perturbation of this size. So "the AUC went down" and "the
correction is worthless" are different statements and only the control separates them.

FOUR INSTRUMENTS, in increasing order of sensitivity:

 1. Pooled OOF AUC of base + w*c over all 691,369 rows, against the permuted-c control.
    This is the number that is comparable to the workspace's 0.970049 and the one a
    submission decision has to be made on. It is also the LEAST sensitive, because it pays
    the toll.
 2. Per-fold AUC delta, 5 folds, sign count -- the format the author's own 60/60 evidence
    is in, so it is the like-for-like comparison against his claim.
 3. Conditional AUC of c given the base score: bin the base OOF into 200 quantile bins and
    pool the within-bin Mann-Whitney of c against y. This asks the question directly -- does
    c order the label among rows the stack already scores the same? -- and it does not pay
    the toll at all. Control: c permuted WITHIN each bin, so the marginal and the bin
    structure survive and only the row correspondence dies. This is w15c's instrument.
 4. Correlation of c against our pack members, the OOF-space analogue of w15e's max |rho|
    0.0772 test-space reading.

TRANS vs INDUC is the mechanism control: if the transductive student and the inductive one
give the same answer, then carrying the unlabeled rows does nothing and the object is just
teacher-minus-smooth-student, which is an inductive function of the 12 columns and therefore
already inside w15b's power bound.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, OOF, SUB, TARGET, load_raw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"          # CV 0.970049, the deadline pick, chosen on CV never on LB
W_ANTI = 0.10                   # the author's published constant. NOT searched.
N_CTRL = 200
N_BOOT = 200
RNG = np.random.default_rng(20260815)


def pct(v):
    v = np.asarray(v, np.float64)
    return rankdata(v) / len(v)


def zr(v):
    r = rankdata(np.asarray(v, np.float64))
    return (r - r.mean()) / r.std()


def make_correction(teacher_r, student_r, fold_id):
    """sign(r)|r|^2 renormalised to sd(r), built INDEPENDENTLY INSIDE EACH FOLD.

    The author renormalises against a `reference_contrast` from two held-out blocks of his
    own data. We do not have his blocks, and we do not need them: his formula multiplies by
    reference.std()/reference_sq.std(), i.e. it restores the signed-square to the scale of
    the raw residual. Doing that with our own fold's residual sd is the same operation with
    one less borrowed constant, and it leaves nothing fitted.
    """
    c = np.zeros(len(teacher_r))
    for k in np.unique(fold_id):
        m = fold_id == k
        r = teacher_r[m] - student_r[m]
        sq = np.sign(r) * np.abs(r) ** 2
        c[m] = (sq - sq.mean()) * (r.std() / sq.std())
    return c


def cond_auc(score, c, y, n_bins=200, seed=0):
    """Pooled within-bin Mann-Whitney of c against y, binned on the base score."""
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, score, "right") - 1, 0, n_bins - 1)
    num = den = 0.0
    for k in range(n_bins):
        m = b == k
        yy = y[m]
        if yy.min() == yy.max():
            continue
        cc = c[m]
        r = rankdata(cc)
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        u = r[yy == 1].sum() - npos * (npos + 1) / 2.0
        num += u
        den += npos * nneg
    return num / den


def cond_auc_ctrl(score, c, y, n_bins=200, seeds=24):
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, score, "right") - 1, 0, n_bins - 1)
    idx = [np.flatnonzero(b == k) for k in range(n_bins)]
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(9000 + s)
        cp = c.copy()
        for ix in idx:
            cp[ix] = c[rng.permutation(ix)]
        out.append(cond_auc(score, cp, y, n_bins))
    return np.array(out)


def main():
    z = np.load(os.path.join(HERE, "w15f_nested.npz"))
    teacher_r, fold_id = z["teacher_r"], z["fold_id"]
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    assert base_p.shape == (n,)
    base_rank = pct(base_p)
    base_auc = roc_auc_score(y, base_p)
    print(f"base {BASE}: pooled OOF AUC {base_auc:.7f}")
    print(f"teacher held-out AUC per fold {z['teacher_auc']}  "
          f"mean {z['teacher_auc'].mean():.6f}")
    print(f"teacher vs base, rank corr: {float(zr(teacher_r) @ zr(base_p) / n):+.5f}"
          "   (a sane teacher sits near the pack)\n")

    results = {}
    for arm, key in [("trans", "student_t"), ("induc", "student_i")]:
        c = make_correction(teacher_r, z[key], fold_id)
        print(f"================ arm {arm} ================")
        print(f"  correction sd {c.std():.5f}  "
              f"[{c.min():+.4f}, {c.max():+.4f}]  solo AUC {roc_auc_score(y, c):.6f}")

        # --- 1. pooled AUC vs the permuted-c control -------------------------------
        real = roc_auc_score(y, base_rank + W_ANTI * c)
        ctrl = np.empty(N_CTRL)
        for i in range(N_CTRL):
            cp = np.empty(n)
            for k in np.unique(fold_id):
                m = np.flatnonzero(fold_id == k)
                cp[m] = c[RNG.permutation(m)]
            ctrl[i] = roc_auc_score(y, base_rank + W_ANTI * cp)
        toll = ctrl.mean() - base_auc
        print(f"  pooled: real {real:.7f}  ({real-base_auc:+.2e} vs base)")
        print(f"          control mean {ctrl.mean():.7f} sd {ctrl.std():.2e}  "
              f"=> toll {toll:+.2e}")
        print(f"          REAL - CONTROL = {real-ctrl.mean():+.3e}  "
              f"z = {(real-ctrl.mean())/ctrl.std():+.2f}  "
              f"P(ctrl >= real) = {(ctrl >= real).mean():.3f}")

        # --- 2. per-fold, the author's own evidence format --------------------------
        deltas = []
        for k in np.unique(fold_id):
            m = fold_id == k
            br = pct(base_p[m])
            deltas.append(roc_auc_score(y[m], br + W_ANTI * c[m])
                          - roc_auc_score(y[m], br))
        deltas = np.array(deltas)
        print(f"  per-fold delta: {' '.join(f'{d:+.2e}' for d in deltas)}")
        print(f"          mean {deltas.mean():+.3e}  positive {int((deltas>0).sum())}/5")

        # --- 3. conditional AUC given the base score (does not pay the toll) --------
        ca = cond_auc(base_p, c, y)
        cc = cond_auc_ctrl(base_p, c, y)
        print(f"  cond AUC(c | base) = {ca:.6f}   control {cc.mean():.6f} "
              f"sd {cc.std():.2e}   z = {(ca-cc.mean())/cc.std():+.2f}")

        # --- 4. how orthogonal is it to our pack, in OOF space? ---------------------
        za = zr(c)
        rows = []
        for d, tag in [(os.path.join(DATA, "oof", "oof"), "lib"),
                       (os.path.join(DATA, "ext_members"), "ext"),
                       (os.path.join(DATA, "ext_members2"), "ext2"),
                       (OOF, "own")]:
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.startswith("oof_") or not fn.endswith(".npy"):
                    continue
                v = np.load(os.path.join(d, fn))
                if v.shape != (n,):
                    continue
                rows.append((f"{tag}:{fn[4:-4]}", float(za @ zr(v) / n)))
        cdf = pd.DataFrame(rows, columns=["member", "corr"])
        print(f"  vs {len(cdf)} pack members (OOF-space): min {cdf['corr'].min():+.4f}  "
              f"median {cdf['corr'].median():+.4f}  max {cdf['corr'].max():+.4f}  "
              f"max|.| {cdf['corr'].abs().max():.4f}")
        print(f"  vs the base blend itself: {float(za @ zr(base_p) / n):+.5f}")

        results[arm] = dict(sd=float(c.std()), solo=float(roc_auc_score(y, c)),
                            pooled_real=float(real), pooled_ctrl=float(ctrl.mean()),
                            pooled_ctrl_sd=float(ctrl.std()),
                            net=float(real - ctrl.mean()),
                            perfold=[float(d) for d in deltas],
                            cond_auc=float(ca), cond_ctrl=float(cc.mean()),
                            cond_ctrl_sd=float(cc.std()),
                            maxabs_corr=float(cdf["corr"].abs().max()))
        np.save(os.path.join(HERE, f"w15f_c_{arm}.npy"), c)

    # --- how much do the two arms even differ? -------------------------------------
    ct = np.load(os.path.join(HERE, "w15f_c_trans.npy"))
    ci = np.load(os.path.join(HERE, "w15f_c_induc.npy"))
    results["trans_vs_induc_corr"] = float(zr(ct) @ zr(ci) / n)
    print(f"\ncorr(trans correction, induc correction) = "
          f"{results['trans_vs_induc_corr']:+.5f}")
    print("  -> if this is ~1.0, carrying the unlabeled rows changed nothing and the "
          "object is\n     an ordinary inductive function of the 12 columns.")

    results["base"] = BASE
    results["base_auc"] = float(base_auc)
    results["weight"] = W_ANTI
    json.dump(results, open(os.path.join(HERE, "w15f_eval.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'w15f_eval.json')}")


if __name__ == "__main__":
    main()
