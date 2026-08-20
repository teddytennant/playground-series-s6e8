"""w33a -- vet and import `omidbaghchehsaraei`'s 12 S6E8 base members.

WHY THIS EXISTS
---------------
w32 §5 re-enumerated the pool and concluded "the last importable material is still
adarsh1077's 22 members from 08-15 ... blocked on SUPPLY, not on method", and priced the
gold line at +10.4e-6 with foreign members historically worth +2.1e-6 each -- i.e. "roughly
five more foreign members is a medal".

That conclusion was wrong, and it was wrong because the scan only ever looked at Kaggle
DATASETS plus the first page of kernels. `omidbaghchehsaraei` publishes twelve SEPARATE
single-model S6E8 notebooks, each of which writes `oof.csv` + `submission.csv` as kernel
output. They were invisible to a dataset scan and they are not one library, they are twelve
kernels. Seven of the twelve are neural (TabM, TabNet, FT-Transformer, RealMLP, ResNet,
CNN, TabTransformer) -- which is the class RESEARCH 6323 says actually pays, because the
operational rule here is not "prefer CatBoost" but "prefer a pipeline you do not hold".

`stephentarter`'s five base notebooks were checked at the same time and are EXCLUDED: they
publish `submission.csv` only, no OOF, so they cannot be honestly weighted.

THE GATES -- a member that fails any of 1-4 is not written
----------------------------------------------------------
1. SHAPE + ID ORDER. 691,369 OOF rows with `id` exactly 0..691368; 296,302 test rows with
   `id` exactly 691369..987670.
2. LABEL IDENTITY. Their `oof.csv` ships `addicted_label` alongside the prediction. It must
   equal OUR y elementwise on all 691,369 rows. This is stronger than the usual
   published-AUC check: AUC reproduction only proves the rows are in SOME consistent order,
   whereas an exact label match proves the row indexing is ours.
3. PUBLISHED-AUC REPRODUCTION. Computed OOF AUC vs the "ROC-AUC SCORE" printed in their
   kernel log, tolerance 5e-5 (the tolerance the 74-lib manifest gate uses).
4. THE FOLD GATE -- the strongest one available, per RESEARCH's RealMLP worked example.
   Score their OOF vector PER FOLD under our frozen SKF5(5, shuffle, seed 42) and compare
   against the five per-fold AUCs their log prints. If the partition AND the fold labelling
   both match ours, the five numbers agree IN ORDER. Their notebooks declare
   `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, so they should -- but
   declaring it and having produced the file with it are different claims, and only this
   test separates them.
5. CREDIBILITY (reported, not enforced). An OOF AUC above ~0.9720 is not achievable here.
6. EARLY-STOPPING-ON-THE-VALIDATION-FOLD (reported, not enforced). Read off the notebook
   source, not guessed: a member whose fit calls `eval_set=[(X_val, y_val)]` /
   `use_best_model=True` has an OPTIMISTIC OOF and earns undeserved stacker weight. This is
   the `golem_a`/`golem_f` defect that `agent/stack.py:DEFAULT_DROP` already excludes. It
   is recorded per member as `es_on_val` so the affected ones stay separable downstream --
   measured out, not assumed out.

Members are written to data/ext_members10/ as `om_<name>` and NEVER into oof/: every
reproduction gate on file is stated against a fixed member COUNT, so dropping a new member
into oof/ would silently change the pack under w26f, w26h and blend_lab at once.

    .venv/bin/python experiments/w33a_vet.py            # vet + report, writes nothing
    .venv/bin/python experiments/w33a_vet.py --write    # vet, then import the passers
"""
from __future__ import annotations

import argparse
import glob
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
SRC = "/tmp/w33pool2"
OUT = os.path.join(DATA, "ext_members10")
AUC_TOL = 5e-5
FOLD_TOL = 1e-4          # their log prints 5 decimals, so 1e-4 is the printing floor

# dirname-fragment -> short member name. Kept explicit so a new download cannot silently
# rename an existing member and land twice in the pack under two names.
MEMBERS = {
    "cnn-for-predicting":            "cnn",
    "tabtransformer-predicting":     "tabtrans",
    "flaml-xgboost-for-predicting":  "flamlxgb",
    "flaml-lgbm-for-predicting":     "flamllgb",
    "tabnet-for-predicting":         "tabnet",
    "xgboost-v2-for-predicting":     "xgb2",
    "realmlp-for-predicting":        "realmlp",
    "catboost-for-predicting":       "cat",
    "ft-transformer-for-predicting": "ftt",
    "tabm-for-predicting":           "tabm",
    "resnet-for-predicting":         "resnet",
    "xgboost-for-predicting":        "xgb",
}

# GATE 6 -- early stopping / model selection ON THE FOLD THAT BECOMES THE OOF.
# Read off each notebook's own source (notebooks/w33_omid_base/), not guessed. This is the
# `golem_a`/`golem_f` defect that agent/stack.py:DEFAULT_DROP already excludes: the member's
# OOF is optimistic, so it earns stacker weight it has not earned, which inflates CV without
# moving the leaderboard. Final selection here is on CV, so that is the one bias that must
# not enter the pack silently. Value: the exact evidence string, or "" for a clean member.
ES_ON_VAL = {
    "cnn":      "best_val_auc on X_va/y_va keeps best_weights -- checkpoint selected on the scored fold",
    "tabtrans": "best_val_auc on X_va/y_va keeps best_ema_weights -- same, on the EMA copy",
    "xgb2":     "XGBClassifier(early_stopping_rounds=200), eval_set=[(X_tr,y_tr),(X_va,y_va)]",
    "flamllgb": "automl.fit(X_val=X_val, y_val=y_val) -- FLAML selects HYPERPARAMETERS on the scored fold",
    "flamlxgb": "automl.fit(X_val=X_val, y_val=y_val) -- same, and `xgb` is a byte-identical copy",
    "xgb":      "XGBClassifier(early_stopping_rounds=150), eval_set=[(X_val,y_val)]",
    "tabm":     "patience on the scored fold (already held as pub_tabm, so moot)",
    "cat":      "",     # CatBoost fitted on a fixed iteration count, no eval_set
    "ftt":      "",     # fixed epoch schedule, no val-based checkpointing
    "realmlp":  "",
    "resnet":   "",
    "tabnet":   "",
}


def log_aucs(logpath):
    """(five per-fold AUCs in printed order, overall OOF AUC) from the kernel log.

    The log is a JSON stream, so the numbers arrive with escaped newlines around them;
    matching on the literal label is more robust than trying to parse the stream.
    """
    raw = open(logpath, encoding="utf-8", errors="replace").read()
    folds = [float(m) for m in re.findall(r"ROC-AUC: (0\.\d+)", raw)]
    # `cnn` and `tabtrans` emit each fold line TWICE, so their logs carry ten numbers where
    # the rest carry five. Collapsing CONSECUTIVE repeats is the obvious fix and it is
    # wrong: `flamlxgb`, `xgb` and `ftt` each have two adjacent folds that genuinely print
    # the same 5-decimal value, and blind collapsing eats one and leaves four. Detect the
    # doubling structurally instead -- ten numbers that are exactly a pairwise repeat.
    if len(folds) == 10 and folds[::2] == folds[1::2]:
        folds = folds[::2]
    # The overall line is `ROC-AUC SCORE (WITH TE & FREQ): 0.96751` in ten of the
    # twelve logs. That literal `&` carries DIGITS, so a `[^0-9]` run never reaches the
    # number; stopping at the colon instead is what makes this match all twelve.
    overall = re.findall(r"ROC-AUC SCORE[^:]*:\s*(0\.\d+)", raw)
    return folds, (float(overall[-1]) if overall else np.nan)


def per_fold_auc(v, y, folds):
    return [roc_auc_score(y[va], v[va]) for _, va in folds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="import the members that pass")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    tr, _te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    assert len(y) == N_TR, f"train is {len(y)} rows, expected {N_TR}"
    folds = get_folds(y)
    del tr, _te

    rows, keep = [], {}
    for frag, short in sorted(MEMBERS.items(), key=lambda kv: kv[1]):
        hits = [d for d in glob.glob(os.path.join(a.src, "*")) if frag in os.path.basename(d)]
        if not hits:
            rows.append(dict(member=short, verdict="MISSING", note="no download dir"))
            continue
        d = hits[0]
        fo, fs = os.path.join(d, "oof.csv"), os.path.join(d, "submission.csv")
        if not (os.path.exists(fo) and os.path.exists(fs)):
            rows.append(dict(member=short, verdict="NO_OOF",
                             note="submission only -- not honestly weightable"))
            continue

        o = pd.read_csv(fo)
        s = pd.read_csv(fs)
        r = dict(member=short, n_oof=len(o), n_test=len(s))

        # -- gate 1: shape and id order
        g1 = (len(o) == N_TR and len(s) == N_TE
              and o["id"].is_monotonic_increasing and s["id"].is_monotonic_increasing
              and int(o["id"].iloc[0]) == 0 and int(o["id"].iloc[-1]) == N_TR - 1
              and int(s["id"].iloc[0]) == N_TR and int(s["id"].iloc[-1]) == N_TR + N_TE - 1)
        r["g1_ids"] = bool(g1)

        # -- gate 2: their shipped label column must BE our y
        if TARGET in o.columns:
            r["g2_label_mismatch"] = int((o[TARGET].to_numpy().astype(int) != y).sum())
        else:
            r["g2_label_mismatch"] = -1          # column absent -> cannot verify
        r["g2_labels"] = r["g2_label_mismatch"] == 0

        v = o["oof_pred"].to_numpy(np.float64)
        t = s[TARGET].to_numpy(np.float64)

        # -- gate 3: published-AUC reproduction
        lf, lo = log_aucs(glob.glob(os.path.join(d, "*.log"))[0])
        auc = roc_auc_score(y, v)
        r["solo_auc"], r["log_auc"] = auc, lo
        r["g3_auc_diff"] = auc - lo if np.isfinite(lo) else np.nan
        r["g3_auc"] = bool(np.isfinite(lo) and abs(auc - lo) < AUC_TOL)

        # -- gate 4: the fold gate, in order
        ours = per_fold_auc(v, y, folds)
        theirs = lf[:5]
        r["fold_ours"] = " ".join(f"{x:.5f}" for x in ours)
        r["fold_theirs"] = " ".join(f"{x:.5f}" for x in theirs)
        if len(theirs) == 5:
            r["g4_maxdiff"] = float(np.max(np.abs(np.array(ours) - np.array(theirs))))
            r["g4_folds"] = r["g4_maxdiff"] < FOLD_TOL
        else:
            r["g4_maxdiff"], r["g4_folds"] = np.nan, False

        # -- gate 5: credibility
        r["g5_credible"] = bool(auc < 0.9720)

        r["test_min"], r["test_max"] = float(t.min()), float(t.max())
        ok = r["g1_ids"] and r["g2_labels"] and r["g3_auc"] and r["g4_folds"] and r["g5_credible"]
        r["es_on_val"] = bool(ES_ON_VAL.get(short))
        r["verdict"] = "PASS" if ok else "FAIL"
        rows.append(r)
        if ok:
            keep[short] = (v, t)

    # -- gate 6: duplicate arrays WITHIN the import. RESEARCH 1361 has the precedent --
    # `bolt_xgb_d7_alt1` and `bolt_xgb_d7_alt2` were the same array and one had to be
    # dropped. Two members here print an identical solo AUC to all 8 digits, so check the
    # vectors themselves rather than trusting the summary statistic.
    dup_of, held = {}, []
    ks = sorted(keep)
    for i, ni in enumerate(ks):
        for nj in ks[i + 1:]:
            if nj in dup_of:
                continue
            if np.array_equal(keep[ni][0], keep[nj][0]) and np.array_equal(keep[ni][1], keep[nj][1]):
                dup_of[nj] = ni
    for nm, src_nm in dup_of.items():
        keep.pop(nm, None)
        print(f"  [dup] {nm} is byte-identical to {src_nm} on BOTH oof and test -- dropped")

    # -- maxcorr against the pack we already hold. This, not solo AUC, is what has
    # predicted marginal value here: solo-to-stack pass-through is ~1.4%.
    if keep:
        from stack import load_members  # noqa: E402
        base_dirs = [os.path.join(DATA, d) for d in
                     ("ext_members", "ext_members2", "ext_members3",
                      "ext_members4", "ext_members6")]
        base_dirs = [d for d in base_dirs if os.path.isdir(d)]
        bn, BO, _BT = load_members(y, N_TE, extra_dirs=tuple(base_dirs),
                                   drop=("golem_a", "golem_f"))
        print(f"\nmaxcorr against the {len(bn)}-member pack (rank corr on OOF):")
        from scipy.stats import rankdata
        BR = np.column_stack([rankdata(BO[:, j]) for j in range(BO.shape[1])])
        BR = (BR - BR.mean(0)) / BR.std(0)
        mc = {}
        for nm in sorted(keep):
            r = rankdata(keep[nm][0]); r = (r - r.mean()) / r.std()
            c = np.abs(BR.T @ r) / len(r)
            mc[nm] = float(c.max())
            print(f"  {nm:9s} maxcorr {c.max():.5f}  nearest {bn[int(c.argmax())]}")
        # -- gate 7: ALREADY HELD. szymonkapiski's 74-model library (built 2026-08-04)
        # had already re-published four of this author's notebooks under a `pub_` prefix,
        # and they come back at rank-maxcorr 1.00000 against `pub_rmlp`, `pub_resnet`,
        # `pub_tabm`, `pub_tabnet`. Spot-checked against the stored arrays: tabnet is
        # bit-exact, the other three agree to 3e-08 on test, i.e. the library stored a
        # float32 round-trip of the same vector. Importing them would double-count a member
        # the combiner already has, which inflates its weight for nothing.
        held = [nm for nm, v in mc.items() if v > 0.9999]
        for nm in held:
            keep.pop(nm, None)
            print(f"  [held] {nm} is already in the pack as its maxcorr partner -- dropped")

        for r in rows:
            if r.get("member") in mc:
                r["maxcorr"] = mc[r["member"]]
            if r.get("member") in dup_of:
                r["verdict"] = "DUP"
            elif r.get("member") in held:
                r["verdict"] = "ALREADY_HELD"

    df = pd.DataFrame(rows)
    cols = ["member", "verdict", "solo_auc", "log_auc", "g3_auc_diff",
            "g1_ids", "g2_label_mismatch", "g4_maxdiff", "g5_credible", "maxcorr",
            "es_on_val"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print("\nper-fold, ours vs theirs (gate 4):")
    for _, r in df.iterrows():
        if isinstance(r.get("fold_ours"), str):
            print(f"  {r['member']:9s} ours   {r['fold_ours']}")
            print(f"  {'':9s} theirs {r['fold_theirs']}   maxdiff {r['g4_maxdiff']:.2e}")
    df.to_csv(os.path.join(HERE, "w33a_vet.csv"), index=False)
    print(f"\n{len(keep)} of {len(MEMBERS)} pass all four hard gates")

    # The two groups go to SEPARATE dirs. The clean members can join the pack; the
    # optimistic ones must stay measurable but must never be picked up by a build that
    # globs the usual ext_members* list, because their bias runs the same direction as the
    # selection criterion.
    clean = {k: v for k, v in keep.items() if not ES_ON_VAL.get(k)}
    dirty = {k: v for k, v in keep.items() if ES_ON_VAL.get(k)}
    print(f"\n{len(keep)} new members pass the hard gates: "
          f"{len(clean)} clean {sorted(clean)}, {len(dirty)} early-stopped-on-val {sorted(dirty)}")
    for nm in sorted(dirty):
        print(f"  [es_on_val] {nm}: {ES_ON_VAL[nm]}")

    if a.write:
        for sub, grp in (("", clean), ("es", dirty)):
            if not grp:
                continue
            d = a.out + sub
            os.makedirs(d, exist_ok=True)
            for nm, (v, t) in grp.items():
                np.save(os.path.join(d, f"oof_om_{nm}.npy"), v.astype(np.float64))
                np.save(os.path.join(d, f"test_om_{nm}.npy"), t.astype(np.float64))
            print(f"wrote {2*len(grp)} arrays to {d}")
    else:
        print("(dry run -- pass --write to import)")

    json.dump({"n_pass": len(keep), "clean": sorted(clean), "es_on_val": sorted(dirty),
               "already_held": sorted(held),
               "dup": dup_of},
              open(os.path.join(HERE, "w33a_vet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
