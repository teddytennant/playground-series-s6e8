"""M12 -- ensemble disagreement as a segmentation and as a signal.

Pre-registered at experiments/w27_prereg_slot6.txt §M12 BEFORE this file was written.

The slot-8 ANGLE asks to segment the best model's OOF errors and look for structure a
feature could capture. Every segmentation of the DATA is already closed here with matched
controls (errormap's daily_band / n_missing / other_screen_band, the generator rule cells,
per-cell isotonic, cell-local LightGBM, resid_boost2 in both modes, per-cell member
weights) and the ceiling-from-our-own-OOF route is a tautology. The one axis never used is
the ensemble's own internal dispersion.

Why it is not covered by those nulls: the combiner is s = w.m over 190 member columns, so
anything in the linear span of {m_j} is already used at its fitted weight -- that is w16c
§5's lesson. d_sd = sd_j(m_ij) is a SECOND MOMENT and is outside that span.

The rankraw view is the right scale and the choice is not free: it maps every member to
ndtri(rank pct), so all 190 columns have identical N(0,1) marginals by construction and
sd across them is pure ranking disagreement with no scale confound. On the logit or
hybrid views a member with a wider spread would dominate d_sd for reasons that have
nothing to do with disagreement.

    w27y_disagree.py            # full readout, ~5 min after the load
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
sys.path.insert(0, HERE)
from common import DATA, SUB, TARGET, load_raw  # noqa: E402
from blend_lab import load_all  # noqa: E402

# byte-identical to w27w / w27t so the member set IS the shipped one.
# R-M11g: load_all takes ABSOLUTE paths and load_members silently skips a directory
# that is not there, so join DATA here and assert existence BEFORE the load.
XDIRS = tuple(os.path.join(DATA, d) for d in
              ("ext_members3", "ext_members4", "ext_members6", "ext_members7pin"))
# ⚠ NOT blend_lab.HONEST_DROP. The shipped 188/190 packs drop SIX members; HONEST_DROP
# holds only the first four and using it loads 192, silently and with no error. The
# R-M12e count gate caught exactly that on this script's first launch.
DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "lat_ctraw_r400", "lat_ctfixte_r400")
STACK = "w27_ad190stdcorr"
STACK_CV = 0.9701181344          # the registered on-disk number, R-M12e gates on it
N_MEMBERS = 190
NSLICE = 200                     # thin slices of the stack score for the conditional AUC
NPERM = 50


def pooled_within(y, s, g):
    """Concordant-pair-weighted AUC pooled over segments (errormap.py's definition)."""
    num = den = 0.0
    for v in np.unique(g):
        m = g == v
        yy, ss = y[m], s[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        if npos == 0 or nneg == 0:
            continue
        num += roc_auc_score(yy, ss) * npos * nneg
        den += npos * nneg
    return num / den, den


def cond_auc(y, d, sl):
    """Pooled within-slice AUC of `d` against the label, slices given by `sl`."""
    return pooled_within(y, d, sl)[0]


def main():
    t0 = time.time()
    for p in XDIRS:
        assert os.path.isdir(p), f"R-M11g: missing extra dir {p}"

    tr, _te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    s = np.load(os.path.join(SUB, f"oof_{STACK}.npy")).astype(np.float64)
    assert len(s) == len(y), (len(s), len(y))

    # ---- R-M12e gate 1: the stack file reproduces its registered CV -----------------
    pooled = roc_auc_score(y, s)
    print(f"R-M12e gate 1: pooled OOF AUC of {STACK} = {pooled:.10f}  "
          f"registered {STACK_CV:.10f}  d={pooled - STACK_CV:+.2e}")
    gate1 = abs(pooled - STACK_CV) < 5e-8
    print(f"             -> {'PASS' if gate1 else 'FAIL'}")

    names, y2, mats, _ = load_all(("rankraw",), DROP, extra_dirs=XDIRS,
                                  dtype="float32")
    # ---- R-M12e gate 2: member count -------------------------------------------------
    print(f"R-M12e gate 2: {len(names)} members, expected {N_MEMBERS} -> "
          f"{'PASS' if len(names) == N_MEMBERS else 'FAIL'}")
    assert len(names) == N_MEMBERS, f"R-M11g: got {len(names)} members, not {N_MEMBERS}"
    assert np.array_equal(y, y2)
    Z = mats["rankraw"][0]                      # (n, 190) rank-gauss, N(0,1) marginals
    del mats
    n = Z.shape[0]
    print(f"loaded {Z.shape} in {time.time()-t0:.0f}s", flush=True)

    # ---- dispersion statistics -------------------------------------------------------
    d_sd = Z.std(axis=1).astype(np.float64)
    q = np.percentile(Z, [10, 25, 50, 75, 90], axis=1).astype(np.float64)
    d_iqr = q[3] - q[1]
    d_p8 = q[4] - q[0]
    med = q[2]
    lo, hi = q[0], q[4]
    trim = np.where((Z >= lo[:, None]) & (Z <= hi[:, None]), Z, np.nan)
    trim = np.nanmean(trim, axis=1).astype(np.float64)
    del q
    print(f"d_sd  mean {d_sd.mean():.4f}  sd {d_sd.std():.4f}  "
          f"range {d_sd.min():.4f}..{d_sd.max():.4f}", flush=True)

    # =================================================================================
    # ARM A -- segmentation readout
    # =================================================================================
    print("\n=== ARM A: within-decile-of-d_sd AUC of the stack ===")
    dec = np.searchsorted(np.percentile(d_sd, np.arange(10, 100, 10)), d_sd)
    print(f"{'dec':>3s} {'n':>8s} {'d_sd range':>17s} {'base':>7s} {'within AUC':>11s}"
          f" {'|med rank|':>10s}")
    for v in range(10):
        m = dec == v
        yy = y[m]
        print(f"{v:3d} {m.sum():8d} {d_sd[m].min():8.4f}..{d_sd[m].max():-7.4f}"
              f" {yy.mean():7.4f} {roc_auc_score(yy, s[m]):11.6f} {np.abs(med[m]).mean():10.4f}")
    pw, den = pooled_within(y, s, dec)
    tot = int(y.sum()) * int((1 - y).sum())
    print(f"pooled within d_sd decile {pw:.6f}  vs global {pooled:.6f} "
          f"({pw - pooled:+.6f})  pair share {den/tot:.3f}")

    # crosstab against daily_band, the segmentation where the AUC actually lives
    dd = tr["daily_screen_time_hours"].to_numpy("float64") if \
        "daily_screen_time_hours" in tr.columns else None
    if dd is None:
        cand = [c for c in tr.columns if "daily" in c and "hour" in c]
        dd = tr[cand[0]].to_numpy("float64")
        print(f"(daily column resolved to {cand[0]})")
    band = np.where(np.isfinite(dd), np.clip(np.floor(dd / 2), 0, 6), -1)
    print("\n  crosstab: mean d_sd and within-cell AUC by daily_band x d_sd tercile")
    ter = np.searchsorted(np.percentile(d_sd, [33.333, 66.667]), d_sd)
    print(f"{'band':>5s} {'n':>8s} " + " ".join(f"{'T'+str(t):>16s}" for t in range(3)))
    for b in sorted(np.unique(band)):
        mb = band == b
        cells = []
        for t in range(3):
            m = mb & (ter == t)
            yy = y[m]
            if m.sum() < 200 or yy.sum() == 0 or (1 - yy).sum() == 0:
                cells.append(f"{m.sum():7d}    --   ")
            else:
                cells.append(f"{m.sum():7d} {roc_auc_score(yy, s[m]):8.5f}")
        print(f"{b:5.0f} {mb.sum():8d} " + " ".join(f"{c:>16s}" for c in cells))

    # =================================================================================
    # ARM B -- THE GATE. does d_sd carry label info the stack does not use?
    # =================================================================================
    print("\n=== ARM B (GATE): conditional AUC of d_sd within thin slices of the stack ===")
    order = np.argsort(s, kind="mergesort")
    sl = np.empty(n, np.int32)
    sl[order] = np.minimum((np.arange(n) * NSLICE) // n, NSLICE - 1)
    rng = np.random.default_rng(2708)
    for tag, vec in (("d_sd", d_sd), ("d_iqr", d_iqr), ("d_p8", d_p8),
                     ("CTRL uniform", rng.random(n)),
                     ("CTRL d_sd shuffled", rng.permutation(d_sd))):
        a = cond_auc(y, vec, sl)
        # null: permute the LABEL within slice, which is the exact null of "d carries
        # nothing about y at fixed s" and preserves every slice's class balance.
        null = np.empty(NPERM)
        for b in range(NPERM):
            yp = y.copy()
            for v in range(NSLICE):
                m = sl == v
                yp[m] = rng.permutation(y[m])
            null[b] = cond_auc(yp, vec, sl)
        z = (a - null.mean()) / (null.std(ddof=1) + 1e-18)
        print(f"  {tag:20s} cond AUC {a:.6f}   null {null.mean():.6f} "
              f"+/- {null.std(ddof=1):.6f}   z {z:+7.2f}")

    # =================================================================================
    # ARM C -- robust aggregates
    # =================================================================================
    print("\n=== ARM C: unfitted robust aggregates vs the fitted stack ===")
    mean_r = Z.mean(axis=1).astype(np.float64)
    for tag, v in (("stack (fitted)", s), ("mean rank", mean_r),
                   ("median rank", med), ("10-90 trimmed", trim)):
        row = [f"  {tag:16s} global {roc_auc_score(y, v):.6f}"]
        for lbl, mask in (("botdec", dec == 0), ("topdec", dec == 9)):
            yy = y[mask]
            row.append(f"{lbl} {roc_auc_score(yy, v[mask]):.6f}")
        print("   ".join(row))

    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
