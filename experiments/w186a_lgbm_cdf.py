"""w186: does the original's class-conditional CDF survive next to target encoding?

w185 measured the original-as-COLUMN route on a BARE LightGBM (12 raw columns) and got
+1077.9e-6 OOF from one group, `cdfd_*` -- F1(x) - F0(x) from the 7,500-row original's
class-conditional empirical CDFs. It closed with the honest caveat that a large gain on a
0.9633 baseline routinely shrinks against a strong one, and asked the next run to price it
inside a real stack member.

This is that measurement, on the strongest single GBDT this workspace has built:

    lgbm_tuned_lat_frac   OOF 0.9678206   184 lattice cols + 12 decimal-lattice cols
                          lr 0.025, leaves 63, depth 7, max_bin 511, mcs 250, l2 80

THE PREDICTION, WRITTEN BEFORE THE FIT
--------------------------------------
Near-null, and for a reason w185's own mechanism supplies. `cdfd_c` is a univariate
function of column c estimated from ~2.1k positive / ~5.4k negative original rows. The
lattice already carries `TE_c`, which is a univariate function of the SAME column
estimated from 553k in-fold rows -- 70x the sample, the same axis, at exact-value
resolution. w185 won because its baseline had no target encoding at all, so `cdfd_c` was
the only univariate label-conditional summary in the frame. Here it is the second one,
and the incumbent is better estimated.

The one channel that could still pay: `cdfd_c` comes from a DIFFERENT sample of the
generator, so it is a denoised prior immune to the competition frame's own sampling noise,
where TE at an exact value with few rows is not. If that is what carries it, the gain
should concentrate on rare values -- which the per-missingness split below will not see,
so if the arm wins I do not get to claim I predicted the mechanism.

Falsifier: cdf arm beats ctrl by more than +50e-6 OOF (the noise floor this workspace
uses for full-OOF LightGBM deltas).

Arms are paired: identical folds, identical params, identical seed, identical code path.
Only the 9 appended columns differ.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, N_SPLITS, ROOT, TARGET, get_folds, load_raw, save_preds  # noqa: E402

# NOT `oof/`. That directory is the curated pack every downstream module censuses
# (w65c_subsetcheck pins it at 202, w109b_colguard at 199), and a member joins it by a
# selection decision, not by a file landing in it. These two are measurements.
OOF_W186 = os.path.join(ROOT, "oof_w186")
os.makedirs(OOF_W186, exist_ok=True)

ORIG = "data/orig/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"
NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
FRAC_COLS = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
             "work_study_hours", "sleep_hours", "weekend_screen_time"]

PARAMS = dict(learning_rate=0.025, num_leaves=63, max_depth=7, max_bin=511,
              min_child_samples=250, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=80.0)


def lattice(df):
    """The decimal lattice, verbatim from agent/run_lgbm.py."""
    o = {}
    for c in FRAC_COLS:
        v = df[c].to_numpy("float64")
        o[f"frac_{c}"] = v - np.floor(v)
        o[f"d1_{c}"] = np.floor(v * 10) % 10
    return pd.DataFrame(o).to_numpy("float32")


def cdfd(df):
    """F1(x) - F0(x) from the original's class-conditional CDFs, verbatim from w185a.

    Fit on the 7,500 original rows only. The original is disjoint from the competition
    frame and carries its own labels, so this map is constant across folds and cannot
    leak fold-held-out labels -- same argument as the decimal lattice's.
    """
    o = pd.read_csv(ORIG)
    oy = o["addicted_label"].values
    out = {}
    for c in NUM:
        a = np.sort(o.loc[oy == 1, c].dropna().values)
        b = np.sort(o.loc[oy == 0, c].dropna().values)
        x = df[c].to_numpy("float64")
        f1 = np.searchsorted(a, x, "right") / max(len(a), 1)
        f0 = np.searchsorted(b, x, "right") / max(len(b), 1)
        out[f"cdfd_{c}"] = np.where(np.isnan(x), np.nan, f1 - f0)
    return pd.DataFrame(out).to_numpy("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ctrl", "cdf"])
    ap.add_argument("--n-estimators", type=int, default=8000)
    ap.add_argument("--stopping", type=int, default=200)
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    want = [int(f) for f in a.folds.split(",")]
    name = f"w186_lgbmfrac_{a.arm}"

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)

    Ftr, Fte = lattice(tr), lattice(te)
    if a.arm == "cdf":
        Ftr = np.hstack([Ftr, cdfd(tr)])
        Fte = np.hstack([Fte, cdfd(te)])
    print(f"[{name}] appended {Ftr.shape[1]} cols  params={PARAMS}", flush=True)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    iters, t0 = [], time.time()
    for f in want:
        itr, iva = folds[f]
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa = np.hstack([g("Xa"), Ftr[itr]])
        Xb = np.hstack([g("Xb"), Ftr[iva]])
        ya, yb = g("ya"), g("yb")
        m = lgb.LGBMClassifier(n_estimators=a.n_estimators, random_state=42,
                               verbose=-1, n_jobs=int(os.environ.get("LGB_JOBS", 8)),
                               **PARAMS)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgb.early_stopping(a.stopping, verbose=False)])
        del Xa
        oof[iva] = m.predict_proba(Xb)[:, 1]
        del Xb
        Xt = np.hstack([g("Xt"), Fte])
        tp += m.predict_proba(Xt)[:, 1] / N_SPLITS
        del Xt
        iters.append(int(m.best_iteration_ or a.n_estimators))
        print(f"[{name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.6f}  "
              f"iters {iters[-1]}  ({time.time()-t0:.0f}s)", flush=True)

    if len(want) == N_SPLITS:
        cv = roc_auc_score(y, oof)
        print(f"\n[{name}] FULL OOF AUC = {cv:.7f}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"  stored lgbm_tuned_lat_frac = 0.9678206", flush=True)
        save_preds(name, oof, tp, len(y), len(te), out=OOF_W186)
        json.dump(dict(name=name, cv=float(cv), params=PARAMS, arm=a.arm,
                       ncols_appended=int(Ftr.shape[1]), iters=iters),
                  open(os.path.join(OOF_W186, f"summary_{name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
