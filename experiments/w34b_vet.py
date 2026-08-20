"""w34b -- vet nine S6E8 base members found by the LEVEL-2-AS-BIBLIOGRAPHY route.

WHY THIS EXISTS
---------------
w33 established the supply route that re-opened the member pool: a level-2 stacker notebook
is a BIBLIOGRAPHY of importable level-1 members, because it has to name every base model it
loads. w33 applied it to `omidbaghchehsaraei` and got `om_ftt` (+7.5e-6 paired, the largest
per-member value measured here since the adarsh import).

This slot applies it to `ravi20076/playgrounds6e8-public-l2stack-v1`, whose loads point at
`ravi20076/playgrounds6e8-datacollation-v1` -- and THAT notebook is a curated bibliography
of the whole public base-model pool, naming six kernels that publish OOF as kernel output:

    donmarch14/s6e8-catboost                    oof_preds.csv     + test_preds.csv
    donmarch14/s6e8-lgbm                        lgb_oof.npy       + lgb_test.npy
    ravi20076/playgrounds6e8-public-baseline-v1 OOF_Preds_MLV1_1  + Mdl_Preds_MLV1_1  (3 cols)
    ravi20076/playgrounds6e8-public-baseline-v2 OOF_Preds_MLV2_1  + Mdl_Preds_MLV2_1  (1 col)
    tamerlanomralinov/s6e8-lookup-transformer   oof_{lkup,cat,lgb}.npy + test_*.npy
    mhamza0810/s6e8-single-model-fe-cv-0-96947  -- the .npy files were NOT retained in the
                                                   kernel output, only the log. Not importable.

⚠ Two of these were previously dispositioned WRONG in RESEARCH and this corrects the record:
`tamerlanomralinov` was filed as "submission only, no OOF" -- that was true of its DATASET
and false of its KERNEL OUTPUT, which ships six .npy files. `donmarch14`'s two notebooks sit
at 23 and 21 votes on the front page and were never opened at all, because they look like
plain single-model baselines. They are -- and they publish OOF anyway.

THE GATES -- same instrument as w33a, a member failing 1-4 is not written
------------------------------------------------------------------------
1. SHAPE + ID ORDER.  691,369 OOF rows / 296,302 test rows; ids contiguous where shipped.
2. LABEL IDENTITY.    Where the author ships `addicted_label` next to the prediction it must
                      equal OUR y elementwise. Stronger than AUC reproduction: AUC only
                      proves the rows are in SOME consistent order, an exact label match
                      proves the indexing is ours.
3. PUBLISHED-AUC.     Computed OOF AUC vs the overall figure in their kernel log, tol 5e-5.
4. THE FOLD GATE.     Score their OOF PER FOLD under our frozen SKF5(5, shuffle, seed 42)
                      and compare, IN ORDER, with the per-fold AUCs their log prints. If the
                      partition and the fold LABELLING are both ours the five agree to the
                      printing floor. Declaring `random_state=42` and having produced the
                      file with it are different claims; only this test separates them.
5. CREDIBILITY.       OOF AUC above ~0.9720 is not reachable here (reported, not enforced).
6. ES-ON-VAL.         Read off each author's own source, never guessed. See ES_ON_VAL below.
7. ALREADY-HELD.      Rank-maxcorr ~1.000 against the pack means the member is a re-publish
                      of something the combiner already has, and importing it double-counts.

Clean passers go to data/ext_members11/; anything with an optimistic OOF goes to
data/ext_members11es/, which is on NO build's --extra-dirs list. Measured out, not assumed
out -- available and separable, but unable to enter a pack silently.

    .venv/bin/python experiments/w34b_vet.py            # vet + report, writes nothing
    .venv/bin/python experiments/w34b_vet.py --write    # vet, then import the passers
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
SRC = os.path.join(ROOT_D, "notebooks", "w34", "out")
OUT = os.path.join(DATA, "ext_members11")
OUT_ES = os.path.join(DATA, "ext_members11es")
AUC_TOL = 5e-5
FOLD_TOL = 1e-4          # the logs print 5-6 decimals; 1e-4 is the printing floor

# GATE 6 -- model selection ON THE FOLD THAT BECOMES THE OOF. Read off each author's own
# source (notebooks/w34/src/), not guessed. This is the `golem_a`/`golem_f` defect that
# agent/stack.py:DEFAULT_DROP already excludes: the member's OOF is optimistic, so it earns
# stacker weight it has not earned, which inflates CV without moving the leaderboard.
# Selection here is on CV, so this is the one bias that must not enter the pack silently.
ES_ON_VAL = {
    "dm_cat":   "CatBoost od_type + `best iteration: N` printed per fold; eval_set is the scored fold",
    "dm_lgb":   "`Fold k: auc | Best Iter = N` -- LightGBM early stopping on valid_0 = the scored fold",
    # ravi20076's training.py line 115 passes eval_set=[(Xdev,ydev)] for EVERY model, but the
    # v1 fit_params carry only `verbose: 0` -- no early_stopping_rounds and no callback. For
    # XGBoost 2.x and the LightGBM sklearn wrapper an eval_set without either is pure logging
    # and selects nothing, so those two are CLEAN. CatBoost is the exception: `use_best_model`
    # DEFAULTS TO TRUE the moment an eval set is supplied, and no `use_best_model=False`
    # appears anywhere in the notebook. Absent proof otherwise the conservative call is the
    # one that keeps a possibly-optimistic member out of the pack.
    "ravi_xgb1c":  "",
    "ravi_lgbm1c": "",
    "ravi_cb1c":   "CatBoost use_best_model defaults True with eval_set=[(Xdev,ydev)]; no explicit False in source",
    "ravi_realmlp1c": "CFG early_stopping=True with additive patience 15, on the scored fold",
    "tam_lkup": "`fold_k ep_n valAUC=.. best=..` keeps the best epoch by val AUC",
    "tam_cat":  "use_best_model / od_type on the scored fold",
    "tam_lgb":  "early stopping on the scored fold",
}

# GATE 4 pre-empted by the author's own declared partition. tamerlanomralinov's notebook sets
# N_FOLDS = 3, so its three members were never produced under our SKF5 at all -- a FOREIGN
# PARTITION, the same ground `factualexplorer` was excluded on (RESEARCH). It matters more
# than es-on-val does: under a foreign partition the member's prediction on one of our
# validation rows comes from a model trained on other rows of that same validation fold, so
# it is partly in-sample with respect to the fold our stacker is scored on. That inflates CV
# in the same direction as the selection criterion.
FOREIGN_PARTITION = {
    "tam_lkup": "notebook sets N_FOLDS = 3, not our SKF5",
    "tam_cat":  "notebook sets N_FOLDS = 3, not our SKF5",
    "tam_lgb":  "notebook sets N_FOLDS = 3, not our SKF5",
}


def _log_text(path):
    return open(path, encoding="utf-8", errors="replace").read()


def load_candidates():
    """-> {name: dict(oof, test, label, fold_log, overall_log, source)}.

    Every author ships a different container, so the loaders are explicit per source rather
    than glob-driven: a glob that silently picks up the wrong column is exactly how a member
    lands in the pack under the wrong name.
    """
    C = {}

    # --- donmarch14/s6e8-catboost: csv with id + addicted_label + oof_pred
    d = os.path.join(SRC, "donmarch14_s6e8-catboost")
    o = pd.read_csv(os.path.join(d, "oof_preds.csv"))
    t = pd.read_csv(os.path.join(d, "test_preds.csv"))
    raw = _log_text(os.path.join(d, "s6e8-catboost.log"))
    C["dm_cat"] = dict(
        oof=o["oof_pred"].to_numpy(np.float64), test=t["test_pred"].to_numpy(np.float64),
        oof_id=o["id"].to_numpy(), test_id=t["id"].to_numpy(),
        label=o[TARGET].to_numpy(np.int8),
        fold_log=[float(x) for x in re.findall(r"Fold \d+ AUC: (0\.\d+)", raw)],
        overall_log=float(re.findall(r"OOF AUC: (0\.\d+)", raw)[-1]),
        source="donmarch14/s6e8-catboost")

    # --- donmarch14/s6e8-lgbm: same csv shape, plus raw .npy of the same vectors
    d = os.path.join(SRC, "donmarch14_s6e8-lgbm")
    o = pd.read_csv(os.path.join(d, "lgbm_oof.csv"))
    t = pd.read_csv(os.path.join(d, "lgbm_test_preds.csv"))
    raw = _log_text(os.path.join(d, "s6e8-lgbm.log"))
    # ⚠ `Fold k: 0.9xxxxx | Best Iter` is the per-fold line. The `[250]\tvalid_0's auc:` lines
    # are per-ITERATION traces of the same folds; matching `0\.\d+` loosely would collect
    # dozens of them and the gate would compare noise.
    C["dm_lgb"] = dict(
        oof=o["oof_pred"].to_numpy(np.float64), test=t["test_pred"].to_numpy(np.float64),
        oof_id=o["id"].to_numpy(), test_id=t["id"].to_numpy(),
        label=o[TARGET].to_numpy(np.int8),
        fold_log=[float(x) for x in re.findall(r"Fold \d+: (0\.\d+) \| Best Iter", raw)],
        overall_log=float(re.findall(r"OOF AUC: (0\.\d+)", raw)[-1])
        if re.findall(r"OOF AUC: (0\.\d+)", raw) else np.nan,
        source="donmarch14/s6e8-lgbm")

    # --- ravi20076 baselines v1 (3 models) and v2 (1 model), parquet, no id column and no
    # label: the frames are written with `df.index = range(len(df))` so row order IS id order,
    # which gate 4 is what actually tests.
    for tag, kern, cols in (
        ("MLV1", "ravi20076_playgrounds6e8-public-baseline-v1", ("XGB1C", "LGBM1C", "CB1C")),
        ("MLV2", "ravi20076_playgrounds6e8-public-baseline-v2", ("REALMLP1C",)),
    ):
        d = os.path.join(SRC, kern)
        O = pd.read_parquet(os.path.join(d, f"OOF_Preds_{tag}_1.parquet"))
        T = pd.read_parquet(os.path.join(d, f"Mdl_Preds_{tag}_1.parquet"))
        raw = _log_text(os.path.join(d, f"{kern.split('_', 1)[1]}.log"))
        # The log prints `---> OOF score = X | Fold k` five times PER MODEL, in the order the
        # models appear in Mdl_Master, i.e. the same order as the parquet columns.
        allf = [float(x) for x in re.findall(r"OOF score = (0\.\d+) \| Fold \d", raw)]
        for j, c in enumerate(cols):
            nm = "ravi_" + c.lower()
            C[nm] = dict(
                oof=O[c].to_numpy(np.float64), test=T[c].to_numpy(np.float64),
                oof_id=None, test_id=None, label=None,
                fold_log=allf[5 * j: 5 * j + 5],
                overall_log=np.nan,
                source=f"ravi20076/{kern.split('_', 1)[1]}:{c}")

    # --- tamerlanomralinov: three .npy pairs. The lookup transformer's vector is a LOGIT
    # (range about -6..+10), not a probability -- stored as-is here because every gate below
    # is rank-based; the rescale happens at import.
    d = os.path.join(SRC, "tamerlanomralinov_s6e8-lookup-transformer-insights-lb-0-97041")
    raw = _log_text(os.path.join(d, "s6e8-lookup-transformer-insights-lb-0-97041.log"))
    for nm, stem in (("tam_lkup", "lookup_transformer"), ("tam_cat", "catboost"), ("tam_lgb", "lightgbm")):
        C[nm] = dict(
            oof=np.load(os.path.join(d, f"oof_{stem}.npy")).astype(np.float64),
            test=np.load(os.path.join(d, f"test_{stem}.npy")).astype(np.float64),
            oof_id=None, test_id=None, label=None,
            fold_log=[float(x) for x in re.findall(r"fold \d done AUC=(0\.\d+)", raw)],
            overall_log=np.nan,
            source=f"tamerlanomralinov/...:{stem}")
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
    print(f"{len(C)} candidates from the w34 bibliography scan\n")
    rows, keep = [], {}

    for nm, c in sorted(C.items()):
        o, t = c["oof"], c["test"]
        r = {"member": nm, "source": c["source"]}

        # gate 1 -- shape + id order
        g1 = o.shape == (N_TR,) and t.shape == (N_TE,)
        if g1 and c["oof_id"] is not None:
            g1 = np.array_equal(c["oof_id"], np.arange(N_TR)) and \
                 np.array_equal(c["test_id"], np.arange(N_TR, N_TR + N_TE))
        r["g1_shape_ids"] = bool(g1)

        # gate 2 -- label identity, where the author ships one
        if c["label"] is None:
            r["g2_label_mismatch"] = None
        else:
            r["g2_label_mismatch"] = int((c["label"] != y).sum())

        # gate 3 + 5 -- overall AUC, published vs computed
        auc = roc_auc_score(y, o) if g1 else np.nan
        r["oof_auc"] = float(auc)
        r["log_auc"] = float(c["overall_log"])
        r["g3_auc_ok"] = bool(np.isnan(c["overall_log"]) or abs(auc - c["overall_log"]) < AUC_TOL)
        r["g5_credible"] = bool(auc < 0.9720)

        # gate 4 -- the fold gate
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
        print(f"  {nm:16s} auc {auc:.6f}  g1 {int(r['g1_shape_ids'])}  "
              f"g2 {r['g2_label_mismatch']}  g3 {int(r['g3_auc_ok'])}  g4 {g4s}"
              f"  logfolds {fl}")
        if r["foreign_partition"]:
            print(f"      FOREIGN PARTITION: {r['foreign_partition']}")
        if r["es_on_val"]:
            print(f"      es-on-val: {r['es_on_val']}")

    # -- gate 7: maxcorr against the pack we already hold. This is what says whether a member
    # is genuinely new or a re-publish of something the combiner already sums over.
    if keep:
        sys.path.insert(0, os.path.join(ROOT_D, "agent"))
        from stack import load_members  # noqa: E402
        # ⚠ blend_lab PREPENDS ext_members and ext_members2 before whatever --extra-dirs
        # names, so a maxcorr computed against the --extra-dirs list alone is against a
        # 92-member SUBSET and would miss an already-held member entirely. Reproduce the
        # shipping build's member set exactly, or gate 7 is not the gate it claims to be.
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
        mc, near = {}, {}
        for nm in sorted(keep):
            rk = rankdata(keep[nm][0]); rk = (rk - rk.mean()) / rk.std()
            cc = np.abs(BR.T @ rk) / len(rk)
            mc[nm] = float(cc.max()); near[nm] = bn[int(cc.argmax())]
            flag = "  <-- ALREADY HELD" if cc.max() > 0.9999 else ""
            print(f"  {nm:16s} maxcorr {cc.max():.5f}  nearest {near[nm]}{flag}")
        for r in rows:
            if r["member"] in mc:
                r["maxcorr"] = mc[r["member"]]
                r["nearest"] = near[r["member"]]
                if mc[r["member"]] > 0.9999:
                    r["hard_gates"] = False
                    r["already_held"] = True

    out_csv = os.path.join(HERE, "w34b_vet.csv")
    pd.DataFrame(rows).drop(columns=["ours_per_fold", "their_per_fold"]).to_csv(out_csv, index=False)
    json.dump(rows, open(os.path.join(HERE, "w34b_vet.json"), "w"), indent=1, default=str)
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
            # A stored member must be a probability in (0,1): `tam_lkup` is a raw logit and a
            # saturating expit would flatten its tail into a tied block, the qda_raw defect at
            # RESEARCH w29 §6. Rescale by one constant applied identically to BOTH sides --
            # monotone, so no ranking and no solo AUC moves.
            if o.min() < 0.0 or o.max() > 1.0:
                s = 30.0 / max(abs(o).max(), abs(t).max())
                o, t = 1.0 / (1.0 + np.exp(-s * o)), 1.0 / (1.0 + np.exp(-s * t))
            np.save(os.path.join(d, f"oof_{nm}.npy"), o.astype(np.float64))
            np.save(os.path.join(d, f"test_{nm}.npy"), t.astype(np.float64))
            print(f"  wrote {nm} -> {os.path.basename(d)}/")


if __name__ == "__main__":
    main()
