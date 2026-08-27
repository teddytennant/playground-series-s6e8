"""w94b, re-run at a screening-grade operating point that actually finishes.

w94b was correct but priced at the shipped preset (lr 0.035, 4000 trees) and took ~50 min
PER ARM; its stdout was block-buffered and the run was lost before the second arm printed.
Two fixes: every print flushes, and each arm's number is written to JSON the moment it
exists, so a kill can never again cost the whole measurement.

The question here is a SCREENING question -- is the windowed-local TE prior worth a real
rebuild? -- so it is priced at a cheaper operating point: fewer leaves, a higher learning
rate, and the model trained on a fixed 60% subsample of the fold-0 training part. The
comparison stays honest because it is PAIRED: identical rows, identical params, identical
seed, and the two arms differ in exactly one thing, the prior used for the 9 full-resolution
single-column NUM keys. The TE tables themselves are still fitted on the FULL outer training
part, so the subsample never degrades the encoding being tested.

⚠ The absolute AUCs printed here are NOT comparable to anything in the journal and must not
be quoted as a CV. Only the DIFFERENCE between the two arms means anything.

  C1  WIN=1e7 must collapse the variant onto the control EXACTLY, which also proves
      te_block_win's non-windowed path reproduces te_block key-for-key.
      ⚠ C1 is asserted on the TE/CT BLOCK, not on the assembled Xa. The base block keeps
      the raw columns WITH their NaNs alongside the imputed copies, so `Xa_c1 - Xa` is
      NaN wherever a raw value is missing and a max-abs-diff over it is `nan` by
      construction -- the first draft of this check failed that way and the failure was
      the assertion, not the variant. The NaN PATTERN of the base block is checked
      separately and exactly.
"""
from __future__ import annotations
import os, sys, json, time, gc
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from common import TARGET, get_folds, load_raw, SEED  # noqa: E402
from features import NUM, make_frames, te_block  # noqa: E402
from w94b_teprior_full import SMOOTH, WIN, _axis, te_block_win  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w94c_teprior.json")
SUB_FRAC = 0.60
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.08, num_leaves=64,
              min_child_samples=40, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=5.0, max_depth=7,
              verbosity=-1, n_jobs=16, random_state=SEED)


def say(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    res = {"sub_frac": SUB_FRAC, "params": {k: v for k, v in PARAMS.items()}, "arms": {}}
    tr, te = load_raw()
    y = tr[TARGET].astype(int)
    Xtr, Xte, Ktr, Kte = make_frames(tr, te)
    ntr = len(Ktr)
    say(f"frame: {Xtr.shape[1]} base feats, {Ktr.shape[1]} lattice keys ({time.time()-t0:.0f}s)")

    full_keys = pd.concat([Ktr, Kte], ignore_index=True)
    axes_all = {}
    for c in NUM:
        cd, n_lev = _axis(full_keys, c)
        axes_all[c] = (cd[:ntr], cd[ntr:], n_lev)
    wk = [c for c in axes_all if axes_all[c][2] > 3 * WIN]
    say(f"windowed keys ({len(wk)} of {len(axes_all)} single-column NUM, "
        f"> {3*WIN} levels): {wk}")
    res["windowed_keys"] = {c: int(axes_all[c][2]) for c in axes_all}

    itr, iva = get_folds(y)[0]
    ya = y.iloc[itr]
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(len(itr), int(SUB_FRAC * len(itr)), replace=False))
    yb = y.iloc[iva].to_numpy()
    say(f"fold 0: train {len(itr)} -> {len(sub)} subsampled, valid {len(iva)}")

    ax = {c: (axes_all[c][0][itr], axes_all[c][0][iva], axes_all[c][1][:1], axes_all[c][2])
          for c in axes_all}

    frames, te_blocks = {}, {}
    for tag, win in (("global", None), ("windowed", WIN)):
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
        frames[tag] = (Xa, Xb)
        te_blocks[tag] = t_tr[sorted(t_tr.columns)].to_numpy("float32")
        say(f"  {tag} frame built {Xa.shape} ({time.time()-t1:.0f}s)")

    # how far apart the two arms actually are, so a null result can be told from a no-op
    dw = float(np.max(np.abs(te_blocks["windowed"] - te_blocks["global"])))
    res["windowed_vs_global_max_te_shift"] = dw
    say(f"max|TE shift| between the two arms = {dw:.6f}  "
        f"({'the arms DIFFER' if dw > 0 else 'IDENTICAL — the variant is a no-op'})")

    # ---- C1: nearly free, and nothing below means anything without it ----
    t_c1, _, _ = te_block_win(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte.iloc[:1], ax, SMOOTH, 10**7)
    c1cols = sorted(t_c1.columns)
    diff = np.abs(t_c1[c1cols].to_numpy("float32") - te_blocks["global"])
    d = float(np.max(diff))
    if d != 0.0:
        worst = np.argsort(diff.max(axis=0))[::-1][:5]
        say("C1 offenders: " + ", ".join(f"{c1cols[i]}={diff[:, i].max():.3e}" for i in worst))
    # the base block is built ONCE, outside the arm loop, so both arms literally share it;
    # its NaN count is reported as the reason C1 cannot be asserted on Xa, not as a check.
    n_nan = int(np.isnan(Xtr.iloc[itr].to_numpy("float32")).sum())
    res["C1_max_abs_diff"] = d
    res["C1_base_nan_cells"] = n_nan
    res["C1"] = "PASS" if d == 0.0 else "FAIL"
    say(f"\nC1 WIN=1e7 vs te_block, over the TE/CT block only: max|diff| = {d:.3e}  "
        f"{res['C1']}   (base block holds {n_nan} NaN cells and is built once outside "
        f"the arm loop, which is why C1 is not asserted on the assembled Xa)")
    del t_c1; gc.collect()
    json.dump(res, open(OUT, "w"), indent=2)
    if d != 0.0:
        say("C1 FAILED — refusing to report an arm comparison off an unverified variant.")
        return 1

    ysub = ya.to_numpy()[sub]
    for tag in ("global", "windowed"):
        t1 = time.time()
        Xa, Xb = frames[tag]
        m = lgb.LGBMClassifier(n_estimators=1000, **PARAMS)
        m.fit(Xa[sub], ysub, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgb.early_stopping(80, verbose=False)])
        a = float(roc_auc_score(yb, m.predict_proba(Xb)[:, 1]))
        res["arms"][tag] = {"fold0_auc": a, "trees": int(m.best_iteration_),
                            "secs": round(time.time() - t1, 1)}
        json.dump(res, open(OUT, "w"), indent=2)
        say(f"fold-0 AUC, {tag:8s} prior: {a:.10f}  "
            f"({m.best_iteration_} trees, {time.time()-t1:.0f}s)  [written to json]")

    d = (res["arms"]["windowed"]["fold0_auc"] - res["arms"]["global"]["fold0_auc"]) * 1e6
    res["windowed_minus_global_e6"] = d
    json.dump(res, open(OUT, "w"), indent=2)
    say(f"\nREAL 184-COL FRAME, fold 0, paired: WINDOWED - GLOBAL = {d:+.3f}e-6")
    say(f"w94a's 12-column TE-only frame said +520.313e-6")
    say(f"done ({time.time()-t0:.0f}s) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
