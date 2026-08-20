"""w35c: vet the candidates the KERNEL-OUTPUT sweep turned up, with w34b's seven gates.

WHY THIS SCAN EXISTS AND WHY IT IS NOT THE w34 ONE
---------------------------------------------------
w34 §3 established that `kaggle kernels output` and `kaggle datasets download` are DIFFERENT
ENDPOINTS RETURNING DIFFERENT FILES, and that a ref excluded on one has not been excluded on
the other. w35 acted on that. The `dataset_sources` bibliography route is now EXHAUSTED --
every dataset named by `adarsh1077` (19), `thisray` (7) and `riponce` (2) is already imported
or excluded on record, `beicicc` included (10 imported, `sixmember_*` correctly excluded as a
level-2 stack). So the route that was 2-for-2 has saturated, and the open route is the other
endpoint: kernels whose OUTPUT ships an OOF that no dataset publishes.

19 refs off the vote list were swept. Eight ship an OOF+test pair. Those are what is vetted
here. Six of the eight are single-model baselines titled as such -- the same blind spot that
hid `donmarch14` from three earlier scans, which were all looking for LIBRARIES.

THE SEVEN GATES ARE w34b's, UNCHANGED
--------------------------------------
1. SHAPE + ID ORDER.  (691369,) / (296302,), ids 0..N_TR-1 and N_TR..N_TR+N_TE-1.
2. LABEL IDENTITY.    Where the author ships a label column. Neither redamountassir file
                      does, so gate 4 is the binding test for both.
3. PUBLISHED AUC.     Recomputed vs the author's own printed overall OOF AUC.
4. THE FOLD GATE.     Per-fold AUC under OUR frozen SKF5, in order, vs the author's printed
                      per-fold numbers. This is the one that proves the partition, the fold
                      LABELLING and the row indexing are ours, not merely "compatible".
5. CREDIBILITY.       OOF AUC above ~0.9720 is not reachable here (reported, not enforced).
6. ES-ON-VAL.         Read off each author's own source, never guessed.
7. ALREADY-HELD.      Rank-maxcorr ~1.000 against the pack = a re-publish; importing it
                      double-counts.

⚠ THE TWO CLEAN CANDIDATES ARE A DIFFERENT PROFILE FROM w34's, AND THAT IS THE POINT
-------------------------------------------------------------------------------------
w34's clean pair (`ravi_xgb1c`, `ravi_lgbm1c`) sit at solo 0.9642, BELOW the pack median
0.966, at maxcorr ~0.992 -- w34 §6 registered them as a probable null for exactly that
reason. These two sit at solo 0.9680 and 0.9683, ABOVE the pack median, and one of them is
a `HistGradientBoostingClassifier`, a sklearn implementation family the pack does not hold
(it bins to `max_bins=255` with its own split finder and its own categorical handling; it
is not a re-parameterised LightGBM). That is closer to the `om_ftt` profile -- a pipeline
class the pack did not hold -- which is the only profile that has ever paid here.

That is a REASON TO MEASURE, NOT A PREDICTION. Solo-to-stack pass-through in this workspace
is ~1.4% and RESEARCH prices another tuned GBDT into this stack at ~4e-7. The registered
expectation is in experiments/w35d_run.sh, written before the instrument was started.

    .venv/bin/python experiments/w35c_vet.py            # vet + report, writes nothing
    .venv/bin/python experiments/w35c_vet.py --write    # vet, then import the passers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_D = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_D, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402

N_TR, N_TE = 691369, 296302
SRC = os.path.join(ROOT_D, "notebooks", "w35", "out")
OUT = os.path.join(DATA, "ext_members12")
OUT_ES = os.path.join(DATA, "ext_members12es")
AUC_TOL = 5e-5
FOLD_TOL = 1e-4          # the logs print 5-6 decimals; 1e-4 is the printing floor

# GATE 6 -- model selection ON THE FOLD THAT BECOMES THE OOF, read off each author's source
# (notebooks/w35/<ref>/*.ipynb), never guessed. w34 §5 is the standing warning this applies:
# grepping for `early_stopping_rounds` is NOT sufficient, because CatBoost turns
# `use_best_model` on by default the moment an eval set is supplied. Here every dirty member
# declares itself explicitly, so no inference was needed.
ES_ON_VAL = {
    # -- CLEAN. `CONFIG` carries `early_stopping=False` LITERALLY, and sklearn's
    # HistGradientBoosting cannot early-stop against an external eval set at all: its
    # `early_stopping='auto'` carves an internal `validation_fraction` out of the TRAINING
    # rows, never the scored fold. So this one is clean twice over -- by declaration and by
    # construction. Its per-fold TargetEncoder is fit on `y[tr]` only, so no label leak either.
    "ram_hgb": "",
    # -- CLEAN. `model.fit(A_tr, y[tr], categorical_feature=STR_CATS)` -- NO `eval_set`
    # argument is passed at all, so LightGBM has nothing to early-stop against. Absence of an
    # eval set is a stronger clearance than absence of `early_stopping_rounds`.
    "ram_lgb": "",
    "kava_cat": "CatBoost `use_best_model=True` explicit, with early_stopping_rounds, on the scored fold",
    "dkv_cb":  "`use_best_model=True` explicit; `best_iter` printed per fold",
    "dkv_lgb": "`early_stopping_rounds=150` on the scored fold; `best_iter` printed per fold",
    "dkv_xgb": "`early_stopping_rounds=150` on the scored fold; `best_iter` printed per fold",
    "zwr_realmlp": "keeps the best EPOCH by val AUC on the scored fold (`-> best AUC: .. (epoch k)`)",
}

# GATE 4 pre-empted by the author's own declared partition. A foreign partition matters MORE
# than es-on-val: the member's prediction on one of our validation rows comes from a model
# trained on other rows of that same validation fold, so it is partly in-sample with respect
# to the fold our stacker is scored on -- inflating CV in the same direction as selection.
FOREIGN_PARTITION = {
    "evg_lgb": "notebook sets StratifiedKFold(n_splits=10), not our SKF5",
    "ern_spline": "notebook sets OUTER_SPLIT_SEED = 21, not our random_state=42",
}


def _log_text(path):
    return open(path, encoding="utf-8", errors="replace").read()


def _idpair(d, oof_file, test_file, col=TARGET):
    o = pd.read_csv(os.path.join(d, oof_file))
    t = pd.read_csv(os.path.join(d, test_file))
    return (o[col].to_numpy(np.float64), t[col].to_numpy(np.float64),
            o["id"].to_numpy(), t["id"].to_numpy())


def load_candidates():
    """-> {name: dict(oof, test, label, fold_log, overall_log, source)}.

    Explicit per source rather than glob-driven: a glob that silently picks up the wrong
    column is exactly how a member lands in the pack under the wrong name.
    """
    C = {}

    # --- redamountassir, two kernels, same harness: csv `id,addicted_label`, no label column.
    # Log line is `Fold k | AUC: 0.xxxxxx` and `Overall OOF AUC: 0.xxxxxx` (ANSI-bolded, so
    # the overall regex must tolerate the escape).
    for nm, kern, stem in (
        ("ram_hgb", "redamountassir_s6e8-histgradientboosting-lb-0-96945", "tehgbc"),
        ("ram_lgb", "redamountassir_s6e8-lgbm-lb-0-96965", "lgbm"),
    ):
        d = os.path.join(SRC, kern)
        o, t, oid, tid = _idpair(d, f"{stem}_oof_preds.csv", f"{stem}_test_preds.csv")
        raw = _log_text(os.path.join(d, f"{kern.split('_', 1)[1]}.log"))
        C[nm] = dict(
            oof=o, test=t, oof_id=oid, test_id=tid, label=None,
            fold_log=[float(x) for x in re.findall(r"Fold \d \| AUC: (0\.\d+)", raw)],
            overall_log=float(re.findall(r"Overall OOF AUC: \\u001b\[1m(0\.\d+)", raw)[-1]),
            source=f"redamountassir/{kern.split('_', 1)[1]}")

    # --- kava1: a single CatBoost .npy pair. Log prints `Fold k Best AUC: 0.xxxxxx` -- the
    # word "Best" IS the finding; it is the early-stopped score, not a clean fold score.
    d = os.path.join(SRC, "kava1_predicting-smartphone-addiction-lightgbm-fe")
    raw = _log_text(os.path.join(d, "predicting-smartphone-addiction-lightgbm-fe.log"))
    C["kava_cat"] = dict(
        oof=np.load(os.path.join(d, "oof_preds_catboost_0.npy")).astype(np.float64),
        test=np.load(os.path.join(d, "test_preds_catboost_0.npy")).astype(np.float64),
        oof_id=None, test_id=None, label=None,
        fold_log=[float(x) for x in re.findall(r"Fold \d Best AUC: (0\.\d+)", raw)],
        overall_log=np.nan, source="kava1/...-lightgbm-fe:catboost_0")

    # --- danushkumarv: three .npy pairs, our partition (SEED=42, N_FOLDS=5) but all three
    # early-stop. Per-fold line is `fold k  auc 0.xxxxxx  best_iter N`, printed once per
    # model in lgb/xgb/cb order; slice rather than match loosely.
    d = os.path.join(SRC, "danushkumarv_smartphone-addiction-gbm-rank-blend-nb01")
    raw = _log_text(os.path.join(d, "smartphone-addiction-gbm-rank-blend-nb01.log"))
    allf = [float(x) for x in re.findall(r"fold \d  auc (0\.\d+)  best_iter", raw)]
    for j, (nm, stem) in enumerate((("dkv_lgb", "lgb"), ("dkv_xgb", "xgb"), ("dkv_cb", "cb"))):
        C[nm] = dict(
            oof=np.load(os.path.join(d, f"oof_{stem}.npy")).astype(np.float64),
            test=np.load(os.path.join(d, f"pred_{stem}.npy")).astype(np.float64),
            oof_id=None, test_id=None, label=None,
            fold_log=allf[5 * j: 5 * j + 5], overall_log=np.nan,
            source=f"danushkumarv/...-rank-blend-nb01:{stem}")

    # --- zhenruiweng RealMLP: csv pair. Our partition (FOLDS=5) but the epoch is chosen on
    # the scored fold, so it is quarantined however well it reproduces.
    d = os.path.join(SRC, "zhenruiweng_s6e8-public-lb-0-97009-single-model-realmlp")
    raw = _log_text(os.path.join(d, "s6e8-public-lb-0-97009-single-model-realmlp.log"))
    o = pd.read_csv(os.path.join(d, "oof.csv"))
    t = pd.read_csv(os.path.join(d, "test_pred.csv"))
    ocol = [c for c in o.columns if c != "id"][-1]
    tcol = [c for c in t.columns if c != "id"][-1]
    C["zwr_realmlp"] = dict(
        oof=o[ocol].to_numpy(np.float64), test=t[tcol].to_numpy(np.float64),
        oof_id=o["id"].to_numpy() if "id" in o else None,
        test_id=t["id"].to_numpy() if "id" in t else None,
        label=None,
        fold_log=[float(x) for x in re.findall(r"Fold \d \| AUC: (0\.\d+)", raw)],
        overall_log=np.nan, source="zhenruiweng/...-single-model-realmlp")

    # --- evgendvorkin and ern711 are excluded on FOREIGN PARTITION before any artifact is
    # read, so they are recorded in FOREIGN_PARTITION and not loaded. evgendvorkin is a
    # 10-fold split AND early-stops; ern711 is seed 21 AND restores the best-val checkpoint.
    # ern711 is the loss worth naming: a Contextualized Deep Univariate Spline Transformer is
    # a function class the pack does not hold anywhere, i.e. the om_ftt profile exactly, and
    # it is unusable purely because the author picked seed 21 for the outer split.
    return C


def per_fold_auc(v, y, folds):
    return [roc_auc_score(y[va], v[va]) for _, va in folds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    tr, _te = load_raw()
    y = tr[TARGET].to_numpy(np.int8)
    folds = list(get_folds(y))
    assert len(y) == N_TR and len(folds) == 5

    C = load_candidates()
    print(f"{len(C)} candidates from the w35 kernel-output sweep\n")
    rows, keep = [], {}

    for nm, c in sorted(C.items()):
        o, t = c["oof"], c["test"]
        r = {"member": nm, "source": c["source"]}

        g1 = o.shape == (N_TR,) and t.shape == (N_TE,)
        if g1 and c["oof_id"] is not None:
            g1 = np.array_equal(c["oof_id"], np.arange(N_TR)) and \
                 np.array_equal(c["test_id"], np.arange(N_TR, N_TR + N_TE))
        r["g1_shape_ids"] = bool(g1)

        if c["label"] is None:
            r["g2_label_mismatch"] = None
        else:
            r["g2_label_mismatch"] = int((c["label"] != y).sum())

        auc = roc_auc_score(y, o) if g1 else np.nan
        r["oof_auc"] = float(auc)
        r["log_auc"] = float(c["overall_log"])
        r["g3_auc_ok"] = bool(np.isnan(c["overall_log"]) or abs(auc - c["overall_log"]) < AUC_TOL)
        r["g5_credible"] = bool(auc < 0.9720)

        if g1 and len(c["fold_log"]) == 5:
            ours = per_fold_auc(o, y, folds)
            d = float(np.max(np.abs(np.array(ours) - np.array(c["fold_log"]))))
            r["g4_maxdiff"] = d
            r["g4_ok"] = bool(d < FOLD_TOL)
            r["ours_per_fold"] = [round(x, 6) for x in ours]
            r["their_per_fold"] = c["fold_log"]
        else:
            r["g4_maxdiff"] = None
            r["g4_ok"] = False
            r["ours_per_fold"] = None
            r["their_per_fold"] = c["fold_log"]

        r["foreign_partition"] = FOREIGN_PARTITION.get(nm, "")
        r["es_on_val"] = ES_ON_VAL.get(nm, "?")

        hard = r["g1_shape_ids"] and r["g3_auc_ok"] and \
            (r["g2_label_mismatch"] in (None, 0)) and r["g4_ok"] and not r["foreign_partition"]
        r["hard_gates"] = bool(hard)
        rows.append(r)
        if hard:
            keep[nm] = (o, t)

        fl = ("[%s]" % " ".join(f"{x:.5f}" for x in c["fold_log"])) if c["fold_log"] else "[none]"
        g4s = "--" if r["g4_maxdiff"] is None else f"{r['g4_maxdiff']:.2e}"
        print(f"  {nm:14s} auc {auc:.6f}  g1 {int(r['g1_shape_ids'])}  "
              f"g2 {r['g2_label_mismatch']}  g3 {int(r['g3_auc_ok'])}  g4 {g4s}"
              f"  logfolds {fl}")
        if r["ours_per_fold"]:
            print(f"      ours    [{' '.join(f'{x:.5f}' for x in r['ours_per_fold'])}]")
        if r["foreign_partition"]:
            print(f"      FOREIGN PARTITION: {r['foreign_partition']}")
        if r["es_on_val"]:
            print(f"      es-on-val: {r['es_on_val']}")

    # -- gate 7: maxcorr against the pack we already hold.
    if keep:
        from stack import load_members  # noqa: E402
        # ⚠ w34 §8: blend_lab/load_members PREPENDS ext_members and ext_members2 before
        # whatever --extra-dirs names, so a maxcorr computed against the --extra-dirs list
        # alone is against a 92-member SUBSET and an already-held member could walk straight
        # through. Reproduce the shipping build's member set exactly and ASSERT the count.
        base_dirs = ("ext_members", "ext_members2",
                     "ext_members3", "ext_members4", "ext_members6", "ext_members7pin",
                     "ext_members8", "ext_members10")
        drop = {"golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
                "lat_ctraw_r400", "lat_ctfixte_r400", "om_cat"}
        bn, BO, _BT = load_members(
            y, N_TE, extra_dirs=tuple(os.path.join(DATA, d) for d in base_dirs), drop=drop)
        assert len(bn) == 195, f"expected the 195-member w34 pack, got {len(bn)}"
        print(f"\nmaxcorr against the {len(bn)}-member pack (rank corr on OOF):")
        from scipy.stats import rankdata
        BR = np.column_stack([rankdata(BO[:, j]) for j in range(BO.shape[1])])
        BR = (BR - BR.mean(0)) / BR.std(0)
        med = float(np.median([roc_auc_score(y, BO[:, j]) for j in range(BO.shape[1])]))
        print(f"  pack median solo AUC {med:.6f}")
        mc, near = {}, {}
        for nm in sorted(keep):
            rk = rankdata(keep[nm][0]); rk = (rk - rk.mean()) / rk.std()
            cc = np.abs(BR.T @ rk) / len(rk)
            mc[nm] = float(cc.max()); near[nm] = bn[int(cc.argmax())]
            flag = "  <-- ALREADY HELD" if cc.max() > 0.9999 else ""
            print(f"  {nm:14s} maxcorr {cc.max():.5f}  nearest {near[nm]}{flag}")
        for r in rows:
            if r["member"] in mc:
                r["maxcorr"] = mc[r["member"]]
                r["nearest"] = near[r["member"]]
                r["pack_median_solo"] = med
                if mc[r["member"]] > 0.9999:
                    r["hard_gates"] = False
                    r["already_held"] = True

    out_csv = os.path.join(HERE, "w35c_vet.csv")
    pd.DataFrame(rows).drop(columns=["ours_per_fold", "their_per_fold"]).to_csv(out_csv, index=False)
    json.dump(rows, open(os.path.join(HERE, "w35c_vet.json"), "w"), indent=1, default=str)
    print(f"\nwrote {out_csv}")

    passers = [r for r in rows if r["hard_gates"]]
    clean = [r for r in passers if not r["es_on_val"]]
    dirty = [r for r in passers if r["es_on_val"]]
    print(f"\n{len(passers)} pass the hard gates: "
          f"{len(clean)} clean {[r['member'] for r in clean]}, "
          f"{len(dirty)} es-on-val {[r['member'] for r in dirty]}")

    if a.write:
        os.makedirs(OUT, exist_ok=True); os.makedirs(OUT_ES, exist_ok=True)
        for r in passers:
            nm = r["member"]
            d = OUT_ES if r["es_on_val"] else OUT
            o, t = keep[nm]
            # A stored member must be a probability in (0,1); rescale by ONE constant applied
            # identically to both sides -- monotone, so no ranking and no solo AUC moves.
            if o.min() < 0.0 or o.max() > 1.0:
                s = 30.0 / max(abs(o).max(), abs(t).max())
                o, t = 1.0 / (1.0 + np.exp(-s * o)), 1.0 / (1.0 + np.exp(-s * t))
            np.save(os.path.join(d, f"oof_{nm}.npy"), o.astype(np.float64))
            np.save(os.path.join(d, f"test_{nm}.npy"), t.astype(np.float64))
            print(f"  wrote {nm} -> {os.path.basename(d)}/")


if __name__ == "__main__":
    main()
