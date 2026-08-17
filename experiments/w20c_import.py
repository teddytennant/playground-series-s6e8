"""Materialise the vetted new members into data/ext_members3/, and explain cat_str.

TWO JOBS

1. Import `adarsh1077/s6e8-adarsh-oof-library` (22 members) as `ad_*` after the three
   gates: published-AUC reproduction (row order), credibility, and w20a's fold-signature
   gate. Nothing is written for a member that fails.

2. Explain `masayakawamata/s6e8-catstr-aug16`. Its fold statistic is not merely weak, it
   is 7.5x BELOW the null median and below every one of 200 null draws' median -- our
   partition explains LESS between-fold variance in it than a random partition does.
   That needs a mechanism, and there are only two:

     (a) the author normalised predictions WITHIN each fold. Doing that on partition P
         drives P's between-fold mean variance to ~0 while leaving any other partition at
         null. Suppression specific to OUR folds would then be positive evidence that
         P = ours, i.e. the same signature with the sign flipped -- and NOT a leak.
     (b) the OOF is averaged across models that each saw some of the rows they score
         (bagging over folds/seeds), which shrinks between-fold spread and IS a leak.

   These are separable. Under (a) the per-fold MEANS collapse but the per-fold SDs stay
   ordinary and the fold-mean pattern is flat to a precision no random partition reaches.
   Under (b) nothing in particular happens to our folds specifically; the vector just has
   low between-fold variance under every partition, ours included -- so the *ratio* to the
   null is what separates them, and the null is already computed.

   The tell-tale printed below is the z of our fold-mean spread against the null draws,
   plus the same quantity for the member's per-fold SD and per-fold AUC.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, ROOT, TARGET, get_folds  # noqa: E402
from stack import to_logit  # noqa: E402

N_TR, N_TE = 691369, 296302
SRC = os.path.join(DATA, "w20_new")
OUT = os.path.join(DATA, "ext_members3")
EXP = os.path.join(ROOT, "experiments")


def fold_labels(folds, n):
    f = np.empty(n, np.int8)
    for k, (_, va) in enumerate(folds):
        f[va] = k
    return f


def fold_stats(z, lab, y, k=5):
    """(spread of fold means, mean of fold sds, spread of fold AUCs)."""
    mu, sd, au = [], [], []
    for j in range(k):
        m = lab == j
        mu.append(z[m].mean())
        sd.append(z[m].std())
        au.append(roc_auc_score(y[m], z[m]))
    return float(np.std(mu)), float(np.mean(sd)), float(np.std(au))


def main():
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
    ours = fold_labels(get_folds(y), N_TR)
    gate = pd.read_csv(os.path.join(EXP, "w20a_foldgate.csv")).set_index("member")
    screen = pd.read_csv(os.path.join(EXP, "w20b_screen.csv")).set_index("member")

    # ---------------- 2. the cat_str anomaly ----------------
    print("=== cat_str: which mechanism? ===")
    o = np.load(os.path.join(SRC, "catstr", "oof_cat_str.npy")).ravel()
    z = to_logit(o)
    nulls = [fold_labels(list(StratifiedKFold(5, shuffle=True, random_state=90210 + b)
                              .split(np.zeros(N_TR), y)), N_TR) for b in range(60)]
    obs = fold_stats(z, ours, y)
    nl = np.array([fold_stats(z, l, y) for l in nulls])
    lab = ("sd of fold MEANS", "mean of fold SDs", "sd of fold AUCs")
    anom = {}
    for i, nm in enumerate(lab):
        zz = (obs[i] - nl[:, i].mean()) / nl[:, i].std()
        anom[nm] = dict(obs=obs[i], null_mean=float(nl[:, i].mean()),
                        null_sd=float(nl[:, i].std()), z=float(zz))
        print(f"  {nm:18s} ours {obs[i]:.6g}   null {nl[:, i].mean():.6g}"
              f" +/- {nl[:, i].std():.3g}   z {zz:+.2f}")
    # a reference member known to be on our partition, for the same three numbers
    ref = to_logit(np.load(os.path.join(SRC, "oof_catnative.npy")).ravel())
    ro = fold_stats(ref, ours, y)
    rn = np.array([fold_stats(ref, l, y) for l in nulls[:20]])
    print("  reference ad_catnative (same partition, no normalisation):")
    for i, nm in enumerate(lab):
        print(f"  {nm:18s} ours {ro[i]:.6g}   null {rn[:, i].mean():.6g}"
              f"   z {(ro[i] - rn[:, i].mean()) / rn[:, i].std():+.2f}")

    # ---------------- 1. the import ----------------
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for po in sorted(glob.glob(os.path.join(SRC, "oof_*.npy"))):
        nm = os.path.basename(po)[4:-4]
        pt = os.path.join(SRC, f"test_{nm}.npy")
        if not os.path.exists(pt):
            print(f"[skip] {nm}: no test array")
            continue
        oo, tt = np.load(po).ravel(), np.load(pt).ravel()
        if oo.shape != (N_TR,) or tt.shape != (N_TE,):
            print(f"[skip] {nm}: shapes")
            continue
        g1 = abs(float(screen.loc[nm, "repro"])) < 5e-5
        g2 = bool(screen.loc[nm, "credible"])
        g3 = float(gate.loc[nm, "log10_ratio"])
        if not (g1 and g2):
            print(f"[REJECT] {nm}: gate1 {g1} gate2 {g2}")
            continue
        np.save(os.path.join(OUT, f"oof_ad_{nm}.npy"), oo.astype(np.float64))
        np.save(os.path.join(OUT, f"test_ad_{nm}.npy"), tt.astype(np.float64))
        rows.append(dict(member=f"ad_{nm}", oof_auc=float(screen.loc[nm, "oof_auc"]),
                         maxcorr=float(screen.loc[nm, "maxcorr"]), foldgate_log10=g3,
                         foldgate_pass=bool(gate.loc[nm, "exceeds_null_max"])))
    df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    df.to_csv(os.path.join(OUT, "_vetting_w20.csv"), index=False)
    print(f"\nimported {len(df)} members -> {OUT}")
    print(df.to_string(index=False, float_format="%.6f"))

    json.dump(dict(imported=df.member.tolist(), catstr_anomaly=anom),
              open(os.path.join(EXP, "w20c_import.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
