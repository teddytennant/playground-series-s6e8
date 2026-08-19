"""w29a -- members from FUNCTION CLASSES the 190-pack does not contain.

Registered at experiments/w29_prereg_slot10.txt SS M14.

w27q and w27r closed the feature-set route: perturbing the column list of a pipeline the
pack already holds relocates the member inside the pack's span (lat_ctdrop_r400 maxcorr
0.99941, lat_encdrop_r400 0.98607 with all five nearest neighbours being raw-frame library
members). What the pack actually pays for is a new DIRECTION, and w16c's PCA bounds that:
190 members span ~55 usable directions carrying 4.6% of the variance, so a member inside
the span is worth ~0 however good its solo AUC.

Enumerating the pack by function class leaves two holes:

  KERNEL MACHINE.  Every smooth global member in the pack (realmlp, tabm*, nn2,
  bolt_fttransformer) is built from half-space units -- a unit fires on one side of a
  hyperplane. A radial basis fires on a NEIGHBOURHOOD. `rff_raw`/`rff_lat` are random
  Fourier features (Rahimi-Recht): z(x) = sqrt(2/D) cos(Wx + b) with W ~ N(0, 2*gamma),
  whose inner products approximate the RBF kernel, then a plain logistic fit on z. The
  approximation is what makes it affordable at 553k x 184; an exact kernel is not.

  GENERATIVE.  All 190 members are discriminative. `qda_raw` models p(x | y) as two
  Gaussians and classifies on the density ratio. It cannot represent anything the trees
  cannot, but it makes a completely different error -- it is wrong wherever the class
  conditionals are non-Gaussian, which is not where a tree is wrong -- and a linear
  stacker can spend a negative coefficient on exactly that.

`poly2_raw` is the registered CONTROL for the kernel arms, not a candidate in its own
right: an explicit degree-2 polynomial logistic is the same order of nonlinearity as the
RBF arms but anchored to the coordinate axes rather than isotropic. It separates "radial
geometry decorrelates" from "anything that is not a tree decorrelates".

Nothing here is tuned. These are deliberately weak members; the gate is maxcorr < 0.985
(the pack's 10th percentile), fixed in the prereg, and solo AUC is explicitly not a gate.
"""
from __future__ import annotations

import os
# w28 SS8: pin BLAS threads BEFORE numpy is imported, so the reproducibility floor
# does not grow between builds.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, N_SPLITS, ROOT, TARGET, get_folds, load_raw, save_preds  # noqa: E402

EPS = 1e-6
ARMS = ("rff_raw", "rff_lat", "qda_raw", "poly2_raw",
        # w29 second batch: the generative direction, opened by qda_raw landing at
        # maxcorr 0.88773 -- decorrelation rank 1 of 191, BELOW the pack's previous
        # minimum (orig_binm, 0.91797). These extend it along the two axes that
        # actually change the model rather than its hyperparameters: the FRAME
        # (qda_lat sees the encodings) and the DENSITY (gnb_raw is QDA with the
        # covariance forced diagonal, gmm_raw replaces one Gaussian per class with a
        # K-component mixture, so it can represent the lattice's multi-modality).
        "qda_lat", "gnb_raw", "gmm_raw")


def col_index():
    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    te = np.array([i for i, c in enumerate(cols) if c.startswith("TE_")])
    ct = np.array([i for i, c in enumerate(cols) if c.startswith("CT_")])
    base = np.array([i for i, c in enumerate(cols) if not c.startswith(("TE_", "CT_"))])
    return cols, te, ct, base


def raw_design(X, base_idx, med):
    """The 40 raw/derived columns, median-filled from the training part."""
    b = X[:, base_idx].astype(np.float32, copy=True)
    bad = ~np.isfinite(b)
    if bad.any():
        b[bad] = np.take(med, np.where(bad)[1])
    return b


def lat_design(X, te_idx, ct_idx, base_idx, med):
    """run_linear's design: logit(TE) + log1p(CT) + median-filled base."""
    te = np.clip(X[:, te_idx], EPS, 1 - EPS)
    return np.hstack([np.log(te / (1 - te)),
                      np.log1p(np.maximum(X[:, ct_idx], 0.0)),
                      raw_design(X, base_idx, med)]).astype(np.float32)


def zscore_fit(D):
    mu = D.mean(0)
    sd = D.std(0)
    sd[sd < 1e-9] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def rff_map(D, W, b, out=None):
    """sqrt(2/D) cos(x W + b), streamed in row blocks so peak memory stays bounded."""
    n, d = D.shape[0], W.shape[1]
    Z = np.empty((n, d), dtype=np.float32) if out is None else out
    scale = np.float32(np.sqrt(2.0 / d))
    step = 100_000
    for s in range(0, n, step):
        e = min(s + step, n)
        blk = D[s:e] @ W
        blk += b
        np.cos(blk, out=blk)
        blk *= scale
        Z[s:e] = blk
    return Z


def poly2_map(D, keep):
    """[x, x_i * x_j for i<=j over the kept columns]."""
    n = D.shape[0]
    k = len(keep)
    P = D[:, keep]
    out = [D]
    for i in range(k):
        out.append(P[:, i:i + 1] * P[:, i:])
    return np.hstack(out).astype(np.float32)


def fit_arm(arm, Da, ya, makers, a):
    """Returns a callable score(D_raw_design) -> decision function."""
    if arm in ("rff_raw", "rff_lat"):
        rng = np.random.default_rng(a.seed)
        d = Da.shape[1]
        W = rng.normal(0.0, np.sqrt(2.0 * a.gamma), size=(d, a.dim)).astype(np.float32)
        b = rng.uniform(0.0, 2.0 * np.pi, size=a.dim).astype(np.float32)
        Z = rff_map(Da, W, b)
        m = LogisticRegression(max_iter=a.max_iter, C=a.C).fit(Z, ya)
        del Z
        return lambda D: m.decision_function(rff_map(D, W, b))
    if arm in ("qda_raw", "qda_lat"):
        m = QuadraticDiscriminantAnalysis(reg_param=a.reg).fit(Da, ya)
        return lambda D: m.decision_function(D)
    if arm == "gnb_raw":
        m = GaussianNB().fit(Da, ya)
        return lambda D: m.predict_log_proba(D)[:, 1] - m.predict_log_proba(D)[:, 0]
    if arm == "gmm_raw":
        # One mixture per class, then the log density ratio. The mixtures are fitted on a
        # subsample (EM on 553k x 40 buys nothing a 200k fit does not already have) and
        # then SCORE every row, so no row is left without a density.
        rng = np.random.default_rng(a.seed)
        parts = []
        for c in (0, 1):
            Dc = Da[ya == c]
            if len(Dc) > a.gmm_fit:
                Dc = Dc[rng.choice(len(Dc), a.gmm_fit, replace=False)]
            parts.append(GaussianMixture(n_components=a.k, covariance_type="full",
                                         reg_covar=1e-4, max_iter=100,
                                         random_state=a.seed).fit(Dc))
        return lambda D: parts[1].score_samples(D) - parts[0].score_samples(D)
    if arm == "poly2_raw":
        keep = np.arange(Da.shape[1])
        m = LogisticRegression(max_iter=a.max_iter, C=a.C).fit(poly2_map(Da, keep), ya)
        return lambda D: m.decision_function(poly2_map(D, keep))
    raise SystemExit(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--name", default="")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "ext_members8"),
                    help="NEVER oof/ -- writing there moves the whole pack")
    ap.add_argument("--dim", type=int, default=512, help="random-feature count")
    ap.add_argument("--gamma", type=float, default=0.0, help="0 = 1/(2*d) heuristic")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--reg", type=float, default=0.05, help="QDA shrinkage")
    ap.add_argument("--k", type=int, default=6, help="gmm_raw components per class")
    ap.add_argument("--gmm-fit", type=int, default=200_000,
                    help="rows per class the mixtures are FITTED on; all rows are scored")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", default="")
    ap.add_argument("--sub", type=int, default=0, help="probe: subsample train rows")
    a = ap.parse_args()
    name = a.name or a.arm
    if os.path.abspath(a.outdir) == os.path.abspath(os.path.join(ROOT, "oof")):
        raise SystemExit("refusing to write into oof/ -- that moves the shipped pack")
    os.makedirs(a.outdir, exist_ok=True)

    cols, te_idx, ct_idx, base_idx = col_index()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    t0 = time.time()
    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xa[:, base_idx], axis=0)
        med = np.nan_to_num(med).astype(np.float32)
        lat = a.arm in ("rff_lat", "qda_lat")
        mk = lat_design if lat else raw_design
        args = (te_idx, ct_idx, base_idx, med) if lat else (base_idx, med)
        Da = mk(Xa, *args)
        del Xa
        if a.sub:
            rs = np.random.default_rng(a.seed).choice(len(Da), a.sub, replace=False)
            Da, ya = Da[rs], ya[rs]
        mu, sd = zscore_fit(Da)
        Da -= mu
        Da /= sd
        if a.gamma <= 0:
            a.gamma = 1.0 / (2.0 * Da.shape[1])
        print(f"[{name}] fold {f}: design {Da.shape} gamma {a.gamma:.5g}", flush=True)
        score = fit_arm(a.arm, Da, ya, mk, a)
        del Da
        prep = lambda X: ((mk(X, *args) - mu) / sd)
        oof[iva] = score(prep(Xb))
        tp += score(prep(Xt)) / N_SPLITS
        print(f"[{name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xb, Xt

    if len(want) < N_SPLITS or a.sub:
        print(f"[{name}] PROBE run ({len(want)} folds, sub={a.sub}) -- nothing saved",
              flush=True)
        return
    cv = roc_auc_score(y, oof)
    print(f"\n[{name}] FULL OOF AUC = {cv:.8f}  ({time.time()-t0:.0f}s)", flush=True)
    # Stored as probabilities: load_members/transform expect the member scale and the
    # stacker's to_logit inverts a decision function exactly.
    #
    # ⚠ QDA's decision function runs to |z| ~ 1e3, and expit saturates to exactly 1.0 in
    # float64 above z ~ 36.7. The first build of qda_raw put 8.0% of its OOF rows on
    # p == 1.0, which is precisely the plateau RESEARCH documents for rf/et/naji03: the
    # ordering inside that block is destroyed, so `rankraw` averages it into one tied
    # value and the member loses its whole top tail. Rescaling by ONE constant fixes it.
    # The constant is a pure monotone map applied identically to OOF and test, so no
    # member's ranking moves, its solo AUC is unchanged, and the two sides stay on a
    # common scale by construction. No labels enter it.
    zmax = max(np.abs(oof).max(), np.abs(tp).max())
    s = 1.0 if zmax <= 30.0 else 30.0 / zmax
    if s != 1.0:
        print(f"[{name}] monotone rescale: max|z| {zmax:.4g} -> scale {s:.6g}", flush=True)
    save_preds(name, 1 / (1 + np.exp(-s * oof)), 1 / (1 + np.exp(-s * tp)),
               len(y), len(te), out=a.outdir)
    json.dump(dict(name=name, arm=a.arm, cv=float(cv), dim=a.dim, gamma=a.gamma,
                   C=a.C, reg=a.reg, k=a.k, seed=a.seed, zscale=float(s)),
              open(os.path.join(a.outdir, f"summary_{name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
