"""Does the windowed-local TE prior survive into the REAL 184-column frame? (w94)

`w94a_teprior` measured the windowed prior worth +520e-6 on fold 0 of a 12-column
exact-value-TE-only frame, with both controls passing. That frame is a WEAKER
representation than the one we ship, and RESEARCH's standing lesson is that a gain
measured on a weaker representation need not survive transfer to a saturated one.

Here the saturation is concrete and identifiable rather than vague: `make_frames` already
emits `{c}_r1` and `{c}_fl` coarsenings for every numeric column. A value rounded to one
decimal, or floored, pools roughly 10 or 100 adjacent lattice levels -- that is the same
"neighbouring values share something" statement the windowed prior makes, in steps rather
than smoothly. If the GBDT can already read it off the coarsening columns, the smooth
version is a new VIEW, not a new CHANNEL, and it should collapse.

This runs the paired fold-0 comparison on the real frame. Control arm is `te_block`
itself, imported, not a reimplementation. The variant differs only in the prior used for
the 9 full-resolution single-column NUM keys -- every other one of the 72 keys is
byte-identical between the arms.

Nothing here writes to cache/, oof/ or submissions/. Fold-0 only; a full-frame OOF
rebuild is a separate, much larger decision and this is the number that decides whether
it is worth making.

  C1  WIN=1e7 must collapse the variant onto the control EXACTLY (max|diff| == 0).
      If C1 does not collapse, no other number in this file means anything.
"""
from __future__ import annotations
import os, sys, time, gc
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from common import TARGET, get_folds, load_raw, SEED  # noqa: E402
from features import NUM, make_frames, te_block  # noqa: E402

SMOOTH, WIN = 20.0, 16
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.035, num_leaves=96,
              min_child_samples=40, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=5.0, max_depth=7,
              verbosity=-1, n_jobs=16, random_state=SEED)


def _axis(full_keys, c):
    """Integer codes into the NUMERICALLY sorted level set of a single-column key."""
    v = pd.to_numeric(full_keys[c], errors="coerce")
    lev = pd.Index(np.sort(v.dropna().unique()))
    return lev.get_indexer(v).astype(np.int32), len(lev)


def te_block_win(Ktr, y_tr, Kva, Kte, axes, smooth=SMOOTH, win=WIN):
    """te_block with a windowed local prior on the keys named in `axes`.

    `axes` maps key -> (codes_tr, codes_va, codes_te, n_lev). Keys absent from `axes`
    take exactly te_block's global-mean prior, via the identical groupby path.
    """
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 101)
    gm = float(y_tr.mean())
    otr, ova, ote = {}, {}, {}
    splits = list(inner.split(Ktr, y_tr))
    yv = np.asarray(y_tr, dtype=np.float64)

    def table(codes, rows, n_lev, p0):
        cd = codes[rows]; ok = cd >= 0
        n = np.bincount(cd[ok], minlength=n_lev).astype(np.float64)
        k = np.bincount(cd[ok], weights=yv[rows][ok], minlength=n_lev)
        loc = np.full(n_lev, p0)
        if n_lev > 3 * win:
            cn = np.concatenate([[0.], np.cumsum(n)]); ck = np.concatenate([[0.], np.cumsum(k)])
            i = np.arange(n_lev); lo, hi = np.maximum(0, i - win), np.minimum(n_lev, i + win + 1)
            loc = ((ck[hi] - ck[lo] - k) + smooth * p0) / ((cn[hi] - cn[lo] - n) + smooth)
        return (k + smooth * loc) / (n + smooth), n

    for c in Ktr.columns:
        if c in axes:
            ctr, cva, cte, n_lev = axes[c]
            oof_te = np.full(len(Ktr), gm, "float32"); oof_ct = np.zeros(len(Ktr), "float32")
            for fi, hi in splits:
                # te_block shrinks every inner fold toward the OUTER training mean `gm`,
                # not toward the inner fit part's own mean. Matching that exactly is what
                # makes the C1 collapse meaningful -- using the inner mean here is a real
                # 5.4e-7 discrepancy, small enough to hide under a loose tolerance.
                te_, n_ = table(ctr, fi, n_lev, gm)
                cd = ctr[hi]; m = cd >= 0
                oof_te[hi] = np.where(m, te_[np.clip(cd, 0, n_lev - 1)], gm)
                oof_ct[hi] = np.where(m, n_[np.clip(cd, 0, n_lev - 1)], 0.0)
            te_, n_ = table(ctr, np.arange(len(Ktr)), n_lev, gm)
            otr[f"TE_{c}"], otr[f"CT_{c}"] = oof_te, oof_ct
            for dst, cd in ((ova, cva), (ote, cte)):
                m = cd >= 0
                dst[f"TE_{c}"] = np.where(m, te_[np.clip(cd, 0, n_lev - 1)], gm).astype("float32")
                dst[f"CT_{c}"] = np.where(m, n_[np.clip(cd, 0, n_lev - 1)], 0.0).astype("float32")
        else:
            col = Ktr[c].to_numpy()
            oof_te = np.full(len(Ktr), gm, "float32"); oof_ct = np.zeros(len(Ktr), "float32")
            for fi, hi in splits:
                g = pd.DataFrame({"k": col[fi], "y": yv[fi]}).groupby("k").y.agg(["sum", "count"])
                mp = (g["sum"] + smooth * gm) / (g["count"] + smooth)
                s = pd.Series(col[hi])
                oof_te[hi] = s.map(mp).fillna(gm).to_numpy()
                oof_ct[hi] = s.map(g["count"]).fillna(0.0).to_numpy()
            g = pd.DataFrame({"k": col, "y": yv}).groupby("k").y.agg(["sum", "count"])
            mp = (g["sum"] + smooth * gm) / (g["count"] + smooth)
            otr[f"TE_{c}"], otr[f"CT_{c}"] = oof_te, oof_ct
            ova[f"TE_{c}"] = Kva[c].map(mp).fillna(gm).to_numpy("float32")
            ova[f"CT_{c}"] = Kva[c].map(g["count"]).fillna(0.0).to_numpy("float32")
            ote[f"TE_{c}"] = Kte[c].map(mp).fillna(gm).to_numpy("float32")
            ote[f"CT_{c}"] = Kte[c].map(g["count"]).fillna(0.0).to_numpy("float32")
    return (pd.DataFrame(otr, index=Ktr.index), pd.DataFrame(ova, index=Kva.index),
            pd.DataFrame(ote, index=Kte.index))


def main():
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int)
    Xtr, Xte, Ktr, Kte = make_frames(tr, te)
    ntr = len(Ktr)
    print(f"frame: {Xtr.shape[1]} base feats, {Ktr.shape[1]} lattice keys ({time.time()-t0:.0f}s)")

    full_keys = pd.concat([Ktr, Kte], ignore_index=True)
    axes_all = {}
    for c in NUM:
        cd, n_lev = _axis(full_keys, c)
        axes_all[c] = (cd[:ntr], cd[ntr:], n_lev)
    print("windowed keys (full-resolution single-column NUM):")
    for c, (_, _, n_lev) in axes_all.items():
        print(f"  {c:28s} {n_lev:5d} levels   "
              f"{'WINDOWED' if n_lev > 3*WIN else 'global (too few levels)'}")

    itr, iva = get_folds(y)[0]
    ya = y.iloc[itr]

    ax = {c: (axes_all[c][0][itr], axes_all[c][0][iva], axes_all[c][1][:1], axes_all[c][2])
          for c in axes_all}

    res = {}
    for tag, win in (("global", None), ("windowed", WIN), ("C1_win1e7", 10 ** 7)):
        t1 = time.time()
        if win is None:
            t_tr, t_va, _ = te_block(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte.iloc[:1], SMOOTH)
        else:
            t_tr, t_va, _ = te_block_win(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte.iloc[:1],
                                         ax, SMOOTH, win)
        Xa = pd.concat([Xtr.iloc[itr].reset_index(drop=True), t_tr.reset_index(drop=True)],
                       axis=1).to_numpy("float32")
        Xb = pd.concat([Xtr.iloc[iva].reset_index(drop=True), t_va.reset_index(drop=True)],
                       axis=1).to_numpy("float32")
        if tag == "C1_win1e7":
            d = float(np.max(np.abs(Xa - res["global"]["Xa"])))
            print(f"\nC1 WIN=1e7 vs global: max|Xa diff| = {d:.3e}  "
                  f"{'PASS' if d == 0.0 else 'FAIL'}")
            del Xa, Xb; gc.collect(); break
        m = lgb.LGBMClassifier(n_estimators=4000, **PARAMS)
        m.fit(Xa, ya.to_numpy(), eval_set=[(Xb, y.iloc[iva].to_numpy())],
              eval_metric="auc", callbacks=[lgb.early_stopping(150, verbose=False)])
        a = roc_auc_score(y.iloc[iva], m.predict_proba(Xb)[:, 1])
        print(f"fold-0 AUC, {tag:9s} prior: {a:.10f}   "
              f"({m.best_iteration_} trees, {time.time()-t1:.0f}s)")
        res[tag] = {"auc": a, "Xa": Xa if tag == "global" else None}
        del Xb; gc.collect()

    d = (res["windowed"]["auc"] - res["global"]["auc"]) * 1e6
    print(f"\nREAL FRAME, fold 0: WINDOWED - GLOBAL = {d:+.3f}e-6")
    print(f"w94a's 12-column frame said +520.313e-6")
    print(f"done ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
