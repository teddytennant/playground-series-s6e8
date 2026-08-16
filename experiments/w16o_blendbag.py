"""w16o -- how much of the +662.6e-6 OOF/test BAGGING asymmetry survives BLENDING.

This is w16a section 6 item 3 / w15g section 5's own stated open number, deferred by four
consecutive entries.  w16m section 4 corrected the cost estimate that three of those entries
repeated: it is NOT "two extra configurations inside w15g_cvgap.py's existing loop", because
that loop trains six LightGBMs per outer split for ONE member and the quantity in question is
a property of a BLEND, which the loop has no representation of.  This is the scaled stack.

THE QUANTITY
------------
agent/run_lgbm.py:116-118 -- and the standard 5-fold convention, so this holds for every
member in the 160-member pack and for every public library member too:

    oof[iva] = pb                 # ONE model, trained on 80%
    tp += pt / N_SPLITS           # the MEAN of FIVE such models

So every member's OOF column is a single model and its test column is a five-model bag.
w15g measured what that asymmetry is worth for ONE member, in a TRAIN/HOLD geometry built
inside the labelled rows so that w15c's missingness confound is switched off by construction:

    gap_bagging  +662.6e-6  se 6.9   [+660.4, +652.0, +675.5]   (3 outer splits)

w15g then flagged, in its own words, that it had NOT measured how much of that survives
160-member blending, and gave an argument for why most of it should: fold f's training
subsample is COMMON TO EVERY MEMBER, so fold-model idiosyncrasy does not average out across
members the way independent seed noise would.  That is an argument, not a measurement.

THE GEOMETRY (identical to w15g_cvgap.py, deliberately, so the numbers are comparable)
--------------------------------------------------------------------------------------
    outer split: 691,369 labelled rows -> TRAIN 70% / HOLD 30%, stratified, random.
                 TRAIN and HOLD therefore have identical missingness distributions.
    inner      : StratifiedKFold(5, shuffle, random_state=42) inside TRAIN.
    members    : M diverse learners, ALL fitted on the SAME five inner partitions.
    -> M x 5 models.  NONE of them saw ANY HOLD row, so every one of the M x 5 is a valid
       predictor of the whole HOLD set.  That fact is what makes the control below exact.

Blend operator: plain rank-average across members.  Our stack consumes ranks
(blend159av_h3, w16i_schemeavg, ...), and an unweighted average is the version with no
fitted parameter in it, so nothing here can be an argmax over a selection.

    bag_m           = mean_f p[m,f]                        the member's TEST-side column
    BLEND_BAG       = AUC(rankavg_m bag_m)                 the test-side blend
    BLEND_ALIGNED_f = AUC(rankavg_m p[m,f])                the OOF-side blend for the rows
                                                           that fall in fold f: every member
                                                           contributes ITS fold-f model, all
                                                           five trained on the same 80%.
    gap_bagging_blend = BLEND_BAG - mean_f BLEND_ALIGNED_f
    gap_bagging_m     = AUC(bag_m) - mean_f AUC(p[m,f])    the w15g single-member quantity

    SURVIVAL = gap_bagging_blend / mean_m gap_bagging_m

THE CONTROL -- matched, not permuted
------------------------------------
The mechanism w15g proposed is specifically the COMMON fold partition.  Break exactly that
and nothing else: score the same blend with the members' fold indices MISALIGNED, i.e.
member m contributes its fold-sigma(m) model, over all sigma in {0..4}^M that are not
constant.  Every one of those combinations uses the SAME trained models, the SAME rows, the
SAME blend operator and the SAME number of models; only the alignment of the index tuple
differs, and it is legal precisely because no model saw any HOLD row.  Nothing is resampled,
reshuffled or refitted, so this is an exactly matched control rather than a permuted one.

    gap_bagging_misaligned = BLEND_BAG - mean_sigma BLEND_MIS_sigma

If w15g's argument is right, the aligned blend is WORSE than the misaligned blend (the
misaligned one already gets some cross-subsample averaging for free), so
gap_bagging_blend > gap_bagging_misaligned, and the difference between the two IS the
coupling the argument names.  If the two are equal, the common partition is irrelevant and
the aligned/misaligned distinction was never the mechanism.

SCALING IN M
------------
The real pack has 160 members, this has M.  So the primary output is not one number but the
curve of gap_bagging_blend against blend size k = 1..M, computed over ALL C(M,k) subsets
(and, inside each subset, all 5^k alignments).  A flat curve says essentially all of the
single-member gap survives arbitrary blending; a decaying curve can be extrapolated.

PRE-REGISTERED, WRITTEN BEFORE THIS SCRIPT WAS RUN
--------------------------------------------------
1. Report gap_bagging_blend at k=M against BOTH the single-member baseline measured in this
   same run (mean_m gap_bagging_m) and w15g's +662.6e-6, whatever the sign or size, and
   report the misaligned control beside it in the same table.  No arm is dropped.
2. The headline is the ALIGNED number, because the aligned geometry is the one the real
   pipeline has.  The misaligned arm is the control, not an alternative headline.
3. SHIP DECISION, fixed before any number here exists: this measurement produces a
   diagnostic, not a test-set prediction, so it CANNOT yield a submission candidate.  The
   slot ships submissions/w16h_h3av6.csv unconditionally -- the highest-CV file in
   submissions/ that this account has never sent (CV 0.97004873, recomputed here from
   submissions/oof_w16h_h3av6.npy, rank-checked distinct from all 24 sent files, minimum
   295,107 of 296,302 rows differing in rank).  It ships whatever this script prints.
   The higher-CV unsent file w16m_widegrid.csv is excluded because w16m section 1 measured
   it rank-IDENTICAL to the already-sent w16i_schemeavg.csv.
4. DEADLINE PICKS: nothing in this run can move them -- it builds no prediction object over
   the competition test set.  They stay {w16i_schemeavg.csv, blend159av_h3.csv} and
   experiments/check_selection.py is not touched.
5. HARNESS GATE: member 0 is w15g_cvgap.py's exact LightGBM+TE configuration.  Its
   gap_bagging_m must land inside w15g's [+652.0, +675.5] per-split range, or the geometry
   has drifted and the blend numbers mean nothing.  Reported either way.

Usage:  python experiments/w16o_blendbag.py [n_outer_splits] [n_estimators]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, load_raw  # noqa: E402
from features import make_frames, te_block  # noqa: E402

import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "w16o_blendbag.json")
NTHREAD = 14
HOLD_FRAC = 0.30
NFOLD = 5

# Member 0 is w15g_cvgap.py's PARAMS verbatim -- the comparability anchor and the gate.
LGB_TE = dict(objective="binary", learning_rate=0.06, num_leaves=63, max_bin=511,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=NTHREAD)
# Members 1-3 exist to make the blend a BLEND. They are deliberately drawn from three
# different learner families and two different feature sets, because the question is what
# averaging across members does and averaging near-copies would answer a different question.
LGB_RAW = dict(objective="binary", learning_rate=0.05, num_leaves=127, max_bin=255,
               min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
               bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=NTHREAD)
XGB_TE = dict(objective="binary:logistic", tree_method="hist", max_depth=8,
              learning_rate=0.06, subsample=0.8, colsample_bytree=0.6,
              min_child_weight=40, reg_lambda=5.0, nthread=NTHREAD, eval_metric="auc")
XGB_RAW = dict(objective="binary:logistic", tree_method="hist", max_depth=10,
               learning_rate=0.05, subsample=0.7, colsample_bytree=0.8,
               min_child_weight=20, reg_lambda=2.0, nthread=NTHREAD, eval_metric="auc")


def fit_members(Fa, ya, Fb, Fh, n_est):
    """Return {name: (pred_on_Fb, pred_on_Fh)} for every member, on one inner fold."""
    out = {}
    ncol_x = Fa["nx"]
    A, B, H = Fa["F"], Fb, Fh

    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=n_est, random_state=42, **LGB_TE)
    m.fit(A, ya)
    out["lgbm_te"] = (m.predict_proba(B)[:, 1], m.predict_proba(H)[:, 1])
    print(f"      lgbm_te   {time.time() - t0:5.0f}s", flush=True)

    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=n_est, random_state=7, **LGB_RAW)
    m.fit(A[:, :ncol_x], ya)
    out["lgbm_raw"] = (m.predict_proba(B[:, :ncol_x])[:, 1],
                       m.predict_proba(H[:, :ncol_x])[:, 1])
    print(f"      lgbm_raw  {time.time() - t0:5.0f}s", flush=True)

    t0 = time.time()
    m = xgb.XGBClassifier(n_estimators=n_est, random_state=17, **XGB_TE)
    m.fit(A, ya, verbose=False)
    out["xgb_te"] = (m.predict_proba(B)[:, 1], m.predict_proba(H)[:, 1])
    print(f"      xgb_te    {time.time() - t0:5.0f}s", flush=True)

    t0 = time.time()
    m = CatBoostClassifier(iterations=n_est, depth=8, learning_rate=0.08, l2_leaf_reg=6.0,
                           random_seed=23, thread_count=NTHREAD, verbose=0,
                           allow_writing_files=False)
    m.fit(A, ya)
    out["cat_te"] = (m.predict_proba(B)[:, 1], m.predict_proba(H)[:, 1])
    print(f"      cat_te    {time.time() - t0:5.0f}s", flush=True)

    t0 = time.time()
    m = xgb.XGBClassifier(n_estimators=n_est, random_state=31, **XGB_RAW)
    m.fit(A[:, :ncol_x], ya, verbose=False)
    out["xgb_raw"] = (m.predict_proba(B[:, :ncol_x])[:, 1],
                      m.predict_proba(H[:, :ncol_x])[:, 1])
    print(f"      xgb_raw   {time.time() - t0:5.0f}s", flush=True)
    return out


def one_split(X, K, y, seed, n_est):
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y > 0.5)
    idx_neg = np.flatnonzero(y < 0.5)
    hold = []
    for idx in (idx_pos, idx_neg):
        idx = idx.copy()
        rng.shuffle(idx)
        hold.append(idx[:int(round(HOLD_FRAC * len(idx)))])
    hold = np.sort(np.concatenate(hold))
    mask = np.zeros(len(y), bool)
    mask[hold] = True
    train = np.flatnonzero(~mask)
    Xh, Kh, yh = X.iloc[hold], K.iloc[hold], y[hold]
    Xt, Kt, yt = X.iloc[train], K.iloc[train], y[train]
    print(f"  outer seed {seed}: TRAIN {len(train)}  HOLD {len(hold)}", flush=True)

    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)
    names = None
    hold_p = {}          # name -> list of NFOLD arrays over HOLD
    oof = {}             # name -> array over TRAIN
    Xh_np = Xh.to_numpy("float32")

    for f, (ia, ib) in enumerate(skf.split(np.zeros(len(yt)), yt)):
        t0 = time.time()
        TEa, TEb, TEh = te_block(Kt.iloc[ia], yt[ia], Kt.iloc[ib], Kh)
        nx = Xt.shape[1]
        Fa = np.hstack([Xt.iloc[ia].to_numpy("float32"), TEa.to_numpy("float32")])
        Fb = np.hstack([Xt.iloc[ib].to_numpy("float32"), TEb.to_numpy("float32")])
        Fh = np.hstack([Xh_np, TEh.to_numpy("float32")])
        print(f"    fold {f}: encoded {Fa.shape} ({time.time() - t0:.0f}s)", flush=True)
        res = fit_members(dict(F=Fa, nx=nx), yt[ia], Fb, Fh, n_est)
        if names is None:
            names = list(res)
            for n in names:
                hold_p[n] = []
                oof[n] = np.zeros(len(yt))
        for n in names:
            pb, ph = res[n]
            oof[n][ib] = pb
            hold_p[n].append(ph)
        print("    fold %d oof/hold AUC: %s" % (f, "  ".join(
            f"{n} {roc_auc_score(yt[ib], res[n][0]):.6f}/{roc_auc_score(yh, res[n][1]):.6f}"
            for n in names)), flush=True)
        del Fa, Fb, Fh, TEa, TEb, TEh, res

    M = len(names)
    # ---- rank every one of the M*NFOLD hold predictions once ----
    R = {n: [rankdata(p) for p in hold_p[n]] for n in names}
    BAG = {n: np.mean(hold_p[n], axis=0) for n in names}
    RBAG = {n: rankdata(BAG[n]) for n in names}

    def auc(v):
        return float(roc_auc_score(yh, v))

    # ---- per-member: the w15g quantity, member by member ----
    per_member = {}
    for n in names:
        singles = [auc(p) for p in hold_p[n]]
        per_member[n] = dict(singles=singles, single_mean=float(np.mean(singles)),
                             bagged=auc(BAG[n]),
                             gap_bagging=auc(BAG[n]) - float(np.mean(singles)),
                             cv=auc_train(yt, oof[n]))

    # ---- the curve: every subset of every size, aligned and misaligned ----
    curve = {}
    for k in range(1, M + 1):
        rows = []
        for sub in itertools.combinations(range(M), k):
            sn = [names[i] for i in sub]
            bag = auc(np.mean([RBAG[n] for n in sn], axis=0))
            aligned, misaligned = [], []
            for sig in itertools.product(range(NFOLD), repeat=k):
                a = auc(np.mean([R[sn[j]][sig[j]] for j in range(k)], axis=0))
                (aligned if len(set(sig)) == 1 else misaligned).append(a)
            rows.append(dict(members=sn, bagged=bag,
                             aligned_mean=float(np.mean(aligned)),
                             misaligned_mean=float(np.mean(misaligned))
                             if misaligned else None,
                             gap_aligned=bag - float(np.mean(aligned)),
                             gap_misaligned=(bag - float(np.mean(misaligned)))
                             if misaligned else None))
        g_a = np.array([r["gap_aligned"] for r in rows])
        g_m = np.array([r["gap_misaligned"] for r in rows
                        if r["gap_misaligned"] is not None])
        curve[k] = dict(n_subsets=len(rows), rows=rows,
                        gap_aligned_mean=float(g_a.mean()),
                        gap_aligned_sd=float(g_a.std(ddof=1)) if len(g_a) > 1 else 0.0,
                        gap_misaligned_mean=float(g_m.mean()) if len(g_m) else None)
        print(f"    k={k}: {len(rows):2d} subsets   gap_aligned "
              f"{g_a.mean() * 1e6:+8.1f}e-6   gap_misaligned "
              f"{(g_m.mean() * 1e6 if len(g_m) else float('nan')):+8.1f}e-6", flush=True)

    # ---- the full-M blend, spelled out, plus its CV analogue on TRAIN ----
    full = curve[M]["rows"][0]
    oof_blend = np.mean([rankdata(oof[n]) for n in names], axis=0)
    cv_blend = auc_train(yt, oof_blend)
    # member diversity, so the reader can judge how representative this blend is
    div = []
    for i in range(M):
        for j in range(i + 1, M):
            div.append(float(np.corrcoef(RBAG[names[i]], RBAG[names[j]])[0, 1]))
    print(f"    blend CV(TRAIN) {cv_blend:.6f}  bagged HOLD {full['bagged']:.6f}  "
          f"gap_total {(full['bagged'] - cv_blend) * 1e6:+.1f}e-6", flush=True)
    print(f"    member pairwise rank corr on HOLD: min {min(div):.4f} "
          f"median {float(np.median(div)):.4f} max {max(div):.4f}", flush=True)

    return dict(seed=int(seed), n_train=int(len(train)), n_hold=int(len(hold)),
                names=names, per_member=per_member,
                curve={str(k): v for k, v in curve.items()},
                cv_blend=cv_blend, gap_total_blend=full["bagged"] - cv_blend,
                diversity=dict(min=min(div), median=float(np.median(div)), max=max(div)))


def auc_train(yt, v):
    return float(roc_auc_score(yt, v))


def main() -> None:
    nsplit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_est = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    tr, te = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    # same key set as w15g_cvgap.py: singles + the four hand-picked pairs.
    X, _, K, _ = make_frames(tr, te, wide_pairs=False, triples=False)
    print(f"design {X.shape}, {K.shape[1]} lattice keys, {n_est} rounds, "
          f"{nsplit} outer splits, {NTHREAD} threads", flush=True)

    rows = [one_split(X, K, y, 1000 + 7 * i, n_est) for i in range(nsplit)]
    names = rows[0]["names"]
    M = len(names)
    print("\n" + "=" * 78)

    print("PER-MEMBER gap_bagging (the w15g single-member quantity), e-6")
    for n in names:
        v = np.array([r["per_member"][n]["gap_bagging"] for r in rows])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"  {n:10s} {v.mean() * 1e6:+9.1f}   se {se * 1e6:6.1f}   "
              f"[{', '.join(f'{x * 1e6:+.1f}' for x in v)}]")
    base = np.array([np.mean([r["per_member"][n]["gap_bagging"] for n in names])
                     for r in rows])
    print(f"  {'MEAN':10s} {base.mean() * 1e6:+9.1f}   se "
          f"{(base.std(ddof=1) / np.sqrt(len(base)) if len(base) > 1 else float('nan')) * 1e6:6.1f}")
    print("  w15g_cvgap.py single-member reference: +662.6e-6 se 6.9  "
          "[+660.4, +652.0, +675.5]  (member lgbm_te is that exact configuration)")

    print("\nBLEND SIZE CURVE -- gap_bagging as a function of members blended, e-6")
    print(f"{'k':>2} {'ALIGNED (real geometry)':>26} {'MISALIGNED (control)':>22} "
          f"{'aligned-misaligned':>20} {'survival vs k=1':>16}")
    summ = {}
    k1 = np.array([r["curve"]["1"]["gap_aligned_mean"] for r in rows]).mean()
    for k in range(1, M + 1):
        a = np.array([r["curve"][str(k)]["gap_aligned_mean"] for r in rows])
        mm = [r["curve"][str(k)]["gap_misaligned_mean"] for r in rows]
        m_ = np.array(mm) if mm[0] is not None else None
        sea = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        if m_ is not None:
            d = a - m_
            sed = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            ms, ds = f"{m_.mean() * 1e6:+9.1f}", f"{d.mean() * 1e6:+9.1f} +- {sed * 1e6:.1f}"
        else:
            ms, ds = "        -", "        -"
        print(f"{k:>2} {a.mean() * 1e6:+13.1f} +- {sea * 1e6:<9.1f} {ms:>22} {ds:>20} "
              f"{a.mean() / k1 * 100:14.1f}%")
        summ[k] = dict(aligned=float(a.mean()), aligned_se=float(sea),
                       misaligned=float(m_.mean()) if m_ is not None else None,
                       per_split_aligned=[float(x) for x in a])

    print("\nBLEND-LEVEL TOTALS")
    gt = np.array([r["gap_total_blend"] for r in rows])
    cvb = np.array([r["cv_blend"] for r in rows])
    print(f"  blend CV on TRAIN      {cvb.mean():.6f}")
    print(f"  gap_total (bagged-CV)  {gt.mean() * 1e6:+9.1f}e-6   se "
          f"{(gt.std(ddof=1) / np.sqrt(len(gt)) if len(gt) > 1 else float('nan')) * 1e6:.1f}e-6"
          f"   [{', '.join(f'{x * 1e6:+.1f}' for x in gt)}]")
    print(f"  w15g single-member reference: gap_total +432.4e-6 se 53.8")
    dv = [r["diversity"] for r in rows]
    print(f"  member pairwise rank corr on HOLD: min {np.mean([d['min'] for d in dv]):.4f} "
          f"median {np.mean([d['median'] for d in dv]):.4f} "
          f"max {np.mean([d['max'] for d in dv]):.4f}   "
          f"(real pack: median maxcorr 0.9949, decorrelated pair 0.9746/0.9762)")

    with open(OUT, "w") as fh:
        json.dump(dict(n_est=n_est, n_outer=nsplit, names=names,
                       summary={str(k): v for k, v in summ.items()}, splits=rows), fh,
                  indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
