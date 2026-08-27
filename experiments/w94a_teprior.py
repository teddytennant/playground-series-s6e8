"""Price the one untested detail in abhirajhiwale/s6e8-what-the-generator-remembers-honest-cv.

Their exact-value target encoding smooths each lattice cell toward a WINDOWED LOCAL TREND
over the value axis; `agent/features.py:te_block` smooths toward the GLOBAL MEAN. Everything
else about the two constructions matches (leave-self-out inner folds, smooth=20), and
RESEARCH already records exact-value TE itself as this workspace's single biggest win, so
the prior is the only genuinely untested piece of their claim.

Their stated rationale is that in steep regions a global prior "flips residual signs". This
measures that on OUR labels and OUR frozen folds, read-only: nothing is trained for ship,
no artefact on disk is written or changed.

Both TE variants are built out-of-fold with the same inner loop, then compared three ways:
univariate OOF AUC per column, and a fold-0 LightGBM over the 12 single-column TE frames --
single-column keys are exactly where a value-axis prior is defined, so this is the setting
most favourable to their construction.

Controls:
  C1  WIN >= n_levels degenerates the windowed prior INTO the global prior, so the two
      frames must agree to ~0. If C1 does not collapse, the implementation is wrong and
      no other number here means anything.
  C2  shuffled labels must send both univariate AUCs to ~0.5.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import get_folds, load_raw, SEED  # noqa: E402

NUM = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
       'work_study_hours','sleep_hours','notifications_per_day',
       'app_opens_per_day','weekend_screen_time']
CAT = ['gender','stress_level','academic_work_impact']
COLS = NUM + CAT
SMOOTH, WIN = 20.0, 16


def _codes(tr, te, c):
    cats = pd.Index(sorted(pd.concat([tr[c], te[c]], ignore_index=True).dropna().unique()))
    return cats.get_indexer(tr[c]).astype(np.int32), len(cats)


def _curves(n, k, p0, n_lev, windowed):
    """Return the per-level TE. `windowed` picks the prior each cell shrinks toward."""
    loc = np.full(n_lev, p0)
    if windowed and n_lev > 3 * WIN:
        cn = np.concatenate([[0.], np.cumsum(n)])
        ck = np.concatenate([[0.], np.cumsum(k)])
        i = np.arange(n_lev)
        lo, hi = np.maximum(0, i - WIN), np.minimum(n_lev, i + WIN + 1)
        # leave-self-out neighbourhood, exactly as their notebook writes it
        loc = ((ck[hi] - ck[lo] - k) + SMOOTH * p0) / ((cn[hi] - cn[lo] - n) + SMOOTH)
    return (k + SMOOTH * loc) / (n + SMOOTH)


def build_oof(codes, n_lev, y, folds, windowed, win_override=None):
    """OOF TE for one column under one prior. Same inner-fold rule as te_block."""
    global WIN
    keep = WIN
    if win_override is not None:
        WIN = win_override
    try:
        out = np.full(len(y), np.nan, dtype=np.float64)
        for itr, iva in folds:
            inner = StratifiedKFold(4, shuffle=True, random_state=SEED + 101)
            # valid rows encode from the whole outer training part
            cd = codes[itr]; ok = cd >= 0
            n = np.bincount(cd[ok], minlength=n_lev).astype(np.float64)
            k = np.bincount(cd[ok], weights=y[itr][ok].astype(np.float64), minlength=n_lev)
            te = _curves(n, k, float(y[itr].mean()), n_lev, windowed)
            cv = codes[iva]
            out[iva] = np.where(cv >= 0, te[np.clip(cv, 0, n_lev - 1)], float(y[itr].mean()))
        return out
    finally:
        WIN = keep


def main():
    tr, te = load_raw()
    y = tr['addicted_label'].to_numpy()
    folds = get_folds(y)
    print(f"train {len(tr)} rows, base rate {y.mean():.10f}, {len(folds)} frozen folds")

    codes, levs = {}, {}
    for c in COLS:
        codes[c], levs[c] = _codes(tr, te, c)

    rows = []
    G, L = {}, {}
    for c in COLS:
        g = build_oof(codes[c], levs[c], y, folds, windowed=False)
        l = build_oof(codes[c], levs[c], y, folds, windowed=True)
        G[c], L[c] = g, l
        ag, al = roc_auc_score(y, g), roc_auc_score(y, l)
        rows.append((c, levs[c], ag, al, (al - ag) * 1e6))
        print(f"  {c:28s} {levs[c]:5d} lev   global {ag:.10f}   windowed {al:.10f}"
              f"   delta {(al-ag)*1e6:+9.3f}e-6")

    best = max(rows, key=lambda r: r[4])
    tot = sum(r[4] for r in rows)
    print(f"\nunivariate OOF AUC, windowed - global: sum {tot:+.3f}e-6, "
          f"best column {best[0]} {best[4]:+.3f}e-6")

    # ---- C1: WIN wide enough to cover every level must collapse windowed -> global ----
    c1 = 'notifications_per_day'
    wide = build_oof(codes[c1], levs[c1], y, folds, windowed=True, win_override=10 ** 7)
    d = float(np.max(np.abs(wide - G[c1])))
    print(f"\nC1 WIN=1e7 on {c1}: max|windowed - global| = {d:.3e}  "
          f"{'PASS' if d < 1e-9 else 'FAIL'}")

    # ---- C2: shuffled labels ----
    rng = np.random.default_rng(0)
    ys = y.copy(); rng.shuffle(ys)
    c2 = 'daily_screen_time_hours'
    gs = build_oof(codes[c2], levs[c2], ys, folds, windowed=False)
    ls = build_oof(codes[c2], levs[c2], ys, folds, windowed=True)
    ag, al = roc_auc_score(ys, gs), roc_auc_score(ys, ls)
    ok2 = abs(ag - 0.5) < 5e-3 and abs(al - 0.5) < 5e-3
    print(f"C2 shuffled labels on {c2}: global {ag:.6f} windowed {al:.6f}  "
          f"{'PASS' if ok2 else 'FAIL'}")

    # ---- the combined read: fold-0 LightGBM over each 12-column TE frame ----
    import lightgbm as lgb
    itr, iva = folds[0]
    params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=4, seed=SEED)
    out = {}
    for tag, F in (("global", G), ("windowed", L)):
        X = pd.DataFrame({f"TE_{c}": F[c] for c in COLS})
        b = lgb.train(params, lgb.Dataset(X.iloc[itr], y[itr]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[iva], y[iva])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        a = roc_auc_score(y[iva], b.predict(X.iloc[iva], num_iteration=b.best_iteration))
        out[tag] = a
        print(f"fold-0 LGBM on the 12 single-column TE frame, {tag:8s} prior: "
              f"{a:.10f}  ({b.best_iteration} trees)")
    print(f"\nWINDOWED - GLOBAL on fold 0: {(out['windowed']-out['global'])*1e6:+.3f}e-6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
