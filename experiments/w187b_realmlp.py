"""w187: the first RealMLP this workspace has ever TRAINED, on the frozen folds.

w184's post-mortem: 1st place won with a single tuned RealMLP (CV 0.97070) and this
workspace never fitted one -- it only ever HARVESTED other people's OOF columns, whose
best honest number here is ~0.9647. The gap is ~+440e-6, the largest known number in the
workspace, and it was a classification error rather than a search failure.

WHAT IS HONEST HERE AND WHAT IS NOT
-----------------------------------
The source notebook (`kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only`,
public 0.97016) calls `model.fit(X_train, y_train, X_valid, y_valid)`, i.e. it early-stops
on the very rows that become its OOF. That is es-on-val, the exact defect this workspace
drops members for, and it inflates the OOF (not the public score, which is clean).

    --es val    reproduces the notebook, for comparability. OOF is OPTIMISTIC.
    --es inner  the honest path and the default: RealMLP carves `val_fraction` out of the
                outer TRAINING part for its own early stopping and never sees a fold
                validation row. Only this arm's OOF is comparable to our 0.9679083.

Both arms are run so the size of the es-on-val optimism is measured rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
from common import SEED, TARGET, get_folds, load_raw  # noqa: E402

FEAT = os.path.join(HERE, "w187_feat")
RAW = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time",
       "gender", "stress_level", "academic_work_impact"]

# the notebook's tuned vector, verbatim except device
PARAMS = dict(
    n_epochs=100, batch_size=128, n_ens=8, val_metric_name="1-auc_ovr",
    use_early_stopping=True, early_stopping_additive_patience=20,
    early_stopping_multiplicative_patience=1, act="mish", embedding_size=8,
    first_layer_lr_factor=0.5962121993798933, hidden_sizes="rectangular",
    hidden_width=384, lr=0.04, ls_eps=0.011498317194338772, ls_eps_sched="coslog4",
    max_one_hot_cat_size=18, n_hidden_layers=4, p_drop=0.07301419697186451,
    p_drop_sched="flat_cos", plr_hidden_1=16, plr_hidden_2=8,
    plr_lr_factor=0.1151437622270563, plr_sigma=2.3316811282666916,
    scale_lr_factor=2.244801835541429, sq_mom=1.0 - 0.011834054955582318,
    wd=0.02369230879235962,
)


def build_fold(cont_tr, cont_te, cat_tr, cat_te, cont_cols, itr, iva, y, seed):
    """Exact-value target encoding inside the fold, then the model's DataFrames."""
    Ktr = pd.DataFrame(cat_tr[itr]).astype("category")
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=seed,
                        target_type="binary")
    Ztr = enc.fit_transform(Ktr, y[itr]).astype(np.float32)
    Zva = enc.transform(pd.DataFrame(cat_tr[iva]).astype("category")).astype(np.float32)
    Zte = enc.transform(pd.DataFrame(cat_te).astype("category")).astype(np.float32)

    def frame(cont, cat, Z):
        d = pd.DataFrame(cont, columns=cont_cols)
        for i, c in enumerate(RAW):
            d[f"{c}_te"] = Z[:, i]
        for i, c in enumerate(RAW):
            d[c] = pd.Categorical(cat[:, i])
        return d

    return (frame(cont_tr[itr], cat_tr[itr], Ztr),
            frame(cont_tr[iva], cat_tr[iva], Zva),
            frame(cont_te, cat_te, Zte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--es", choices=["inner", "val"], default="inner")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--nens", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--rows", type=int, default=0, help="subsample the training part")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-test", action="store_true", help="skip test prediction")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from pytabkit import RealMLP_TD_Classifier

    params = dict(PARAMS)
    for k, v in (("n_epochs", a.epochs), ("batch_size", a.batch), ("n_ens", a.nens),
                 ("hidden_width", a.width)):
        if v is not None:
            params[k] = v

    cont_cols = json.load(open(os.path.join(FEAT, "cont_cols.json")))
    Xc = np.load(os.path.join(FEAT, "Xc_train.npy"))
    Xct = np.load(os.path.join(FEAT, "Xc_test.npy"))
    cat_tr = np.load(os.path.join(FEAT, "cat_train.npy"))
    cat_te = np.load(os.path.join(FEAT, "cat_test.npy"))

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = list(range(5)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]

    print(f"[{a.name}] es={a.es} rows={a.rows or 'all'} threads={a.threads}\n"
          f"  params={ {k: params[k] for k in ('n_epochs','batch_size','n_ens','hidden_width')} }",
          flush=True)

    oof = np.full(len(y), np.nan)
    tp = np.zeros(len(te))
    aucs = {}
    for f in want:
        itr, iva = folds[f]
        if a.rows:
            rng = np.random.default_rng(SEED + f)
            itr = np.sort(rng.choice(itr, size=min(a.rows, len(itr)), replace=False))
        t0 = time.time()
        Xa, Xb, Xt = build_fold(Xc, Xct, cat_tr, cat_te, cont_cols, itr, iva, y, SEED + f)
        tfe = time.time() - t0
        m = RealMLP_TD_Classifier(device="cpu", random_state=SEED + f, verbosity=1,
                                  n_threads=a.threads, **params)
        t1 = time.time()
        if a.es == "val":
            m.fit(Xa, y[itr], Xb, y[iva], cat_col_names=RAW)
        else:
            m.fit(Xa, y[itr], cat_col_names=RAW)
        tfit = time.time() - t1
        p = m.predict_proba(Xb)[:, 1]
        oof[iva] = p
        auc = roc_auc_score(y[iva], p)
        aucs[f] = auc
        if not a.no_test:
            tp += m.predict_proba(Xt)[:, 1] / len(want)
        print(f"  fold {f}  AUC {auc:.8f}   feat {tfe:.0f}s  fit {tfit:.0f}s", flush=True)
        # checkpoint per fold: a 5-fold run is hours, and writing only at the end means a
        # kill costs everything. w187 paid 100 minutes to learn that.
        ck = os.path.join(HERE, "..", "oof_w187")
        os.makedirs(ck, exist_ok=True)
        np.save(os.path.join(ck, f"oof_{a.name}.npy"), oof)
        if not a.no_test:
            np.save(os.path.join(ck, f"test_{a.name}_partial.npy"), tp)
        json.dump({"done": sorted(aucs), "aucs": aucs},
                  open(os.path.join(HERE, f"{a.name}_progress.json"), "w"), indent=1)

    print(f"[{a.name}] mean fold AUC {np.mean(list(aucs.values())):.8f}", flush=True)
    if len(want) == 5:
        print(f"[{a.name}] OOF AUC {roc_auc_score(y, oof):.8f}", flush=True)
    out = os.path.join(HERE, "..", "oof_w187")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"oof_{a.name}.npy"), oof)
    if not a.no_test:
        np.save(os.path.join(out, f"test_{a.name}.npy"), tp)
    json.dump({"name": a.name, "es": a.es, "folds": want, "rows": a.rows,
               "params": {k: v for k, v in params.items()}, "aucs": aucs},
              open(os.path.join(HERE, f"{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
