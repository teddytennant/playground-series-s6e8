"""Screen the two new public libraries against the pack: AUC, credibility, maxcorr.

The operational rule this applies is RESEARCH.md's, not a new one:

  screen on the PAIR -- low maxcorr to the pack AND solo >= ~0.966 -- never on
  correlation alone, and never reject a group for being GBDT-shaped if it comes from a
  pipeline we do not already hold.

Candidates:
  data/w20_new/           adarsh1077/s6e8-adarsh-oof-library, 22 members
  data/w20_new/catstr/    masayakawamata/s6e8-catstr-aug16, 1 member (CatBoost, raw
                          string categoricals) -- the slot's handed angle in the only
                          form the record says ever pays.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, ROOT, TARGET  # noqa: E402
from stack import DEFAULT_DROP, load_members  # noqa: E402

N_TR, N_TE = 691369, 296302
CREDIBLE_MAX = 0.9720
OUT = os.path.join(ROOT, "experiments")
# published OOF AUCs, from the library READMEs -- gate 1 (row order)
PUBLISHED = {
    "catnative": 0.968601, "gxgbnote": 0.967042, "glgbnote2": 0.966381,
    "gcatnote": 0.968405, "xgbte": 0.968372, "logregte": 0.958910,
    "lgbnote": 0.966491, "lgbte": 0.968165, "gxgbcs4": 0.968451,
    "gxgbd8": 0.968302, "gnn_note": 0.942090, "glgbd4": 0.968193,
    "lgbs7": 0.968155, "gnn_te": 0.963941, "glgb127": 0.968091,
    "gxgbd4": 0.968337, "gnn_wide": 0.964050, "gcatlr02": 0.968275,
    "gcatd8": 0.967943, "gcatseed7": 0.968093, "glgbcs3": 0.968301,
    "hgbte": 0.967468,
}


def candidates():
    out = {}
    for d, pref in ((os.path.join(DATA, "w20_new"), ""),
                    (os.path.join(DATA, "w20_new", "catstr"), "")):
        for p in sorted(glob.glob(os.path.join(d, "oof_*.npy"))):
            nm = os.path.basename(p)[4:-4]
            # the two libraries use different test-file prefixes
            for tp in (os.path.join(d, f"test_{nm}.npy"),
                       os.path.join(d, f"tep_{nm}.npy")):
                if os.path.exists(tp):
                    out[pref + nm] = (p, tp)
                    break
            else:
                print(f"[skip] {nm}: no test array")
    return out


def main():
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
    names, O, T = load_members(y, N_TE,
                               extra_dirs=(os.path.join(DATA, "ext_members"),
                                           os.path.join(DATA, "ext_members2")),
                               drop=DEFAULT_DROP)
    print(f"pack: {len(names)} members")
    # rank-transform the pack once; correlation is measured on ranks throughout here
    Rp = np.column_stack([rankdata(O[:, j]) for j in range(O.shape[1])])
    Rp = (Rp - Rp.mean(0)) / Rp.std(0)

    rows = []
    for nm, (po, pt) in sorted(candidates().items()):
        o, t = np.load(po).ravel(), np.load(pt).ravel()
        if o.shape != (N_TR,) or t.shape != (N_TE,):
            print(f"[skip] {nm}: shapes {o.shape} {t.shape}")
            continue
        auc = roc_auc_score(y, o)
        r = rankdata(o)
        r = (r - r.mean()) / r.std()
        corr = np.abs(Rp.T @ r) / len(r)
        j = int(np.argmax(corr))
        pub = PUBLISHED.get(nm, np.nan)
        rows.append(dict(member=nm, oof_auc=auc, published=pub, repro=auc - pub,
                         maxcorr=float(corr[j]), nearest=names[j],
                         credible=bool(auc < CREDIBLE_MAX),
                         in_test_range=bool(t.min() >= 0 and t.max() <= 1)))
        print(f"{nm:12s} auc {auc:.6f}  pub {pub if pub == pub else float('nan'):.6f}"
              f"  d {auc - pub:+.2e}  maxcorr {corr[j]:.4f} ({names[j]})", flush=True)

    df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    df.to_csv(os.path.join(OUT, "w20b_screen.csv"), index=False)
    print("\n" + df.to_string(index=False, float_format="%.6f"))

    bad = df[df.repro.abs() > 5e-5].dropna(subset=["published"])
    print(f"\ngate 1 (row order): {len(df.dropna(subset=['published'])) - len(bad)}"
          f"/{len(df.dropna(subset=['published']))} reproduce published AUC to <5e-5")
    if len(bad):
        print(bad.to_string(index=False, float_format="%.6f"))
    print(f"gate 2 (credibility < {CREDIBLE_MAX}): {int(df.credible.sum())}/{len(df)}")
    json.dump(dict(n=len(df), pack=len(names),
                   gate1_fail=bad.member.tolist(),
                   gate2_fail=df[~df.credible].member.tolist(),
                   maxcorr_min=float(df.maxcorr.min()),
                   best=df.iloc[0].member, best_auc=float(df.iloc[0].oof_auc)),
              open(os.path.join(OUT, "w20b_screen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
