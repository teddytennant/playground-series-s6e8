"""Import and VET three further public OOF libraries into data/ext_members2/.

The point of this run is not "more GBDTs" -- two days of measurement say another
GBDT member is worth ~2e-6. The point is that `boltuzamaki/s6e8-oof-prediction-library`
contains function classes that exist nowhere in the 86 members we already stack:
a retrieval model (TabR), an explainable boosting machine (a GAM), GANDALF, DCNv2,
DeepFM, an FT-Transformer, and -- most interesting -- six seeds of a *second*
Lookup-Transformer implementation. `lookup` is the single most decorrelated member we
have and takes the largest stacker coefficient; a second, independently written one is
exactly the shape the journal says is missing.

VETTING, because an OOF array from a stranger is a liability until proven otherwise:

1. row alignment -- `id` must be 0..n-1 in order (these libraries ship an id column).
2. published-vs-computed OOF AUC -- must reproduce, or the rows are permuted.
3. credibility -- OOF AUC above ~0.9720 is not achievable here; anything above that
   is early stopping on the validation fold or a mixed partition, and it will earn
   undeserved stacker weight.
4. sd(logit(test))/sd(logit(oof)) -- should sit at 1.000. Our own honest 5-fold model
   is the control at 1.0018. Far below 1 means the array is clipped or otherwise
   compressed asymmetrically between the two splits.
5. max |corr| against every member we already stack, on the logit scale. This is the
   number that decides whether a member is worth having at all: at 0.987-0.999
   correlation the blend's ranking is pinned by the consensus.

Nothing is added to the stack here. This writes the arrays and prints the report;
experiments/member_value.py makes the paired decision.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, LIB, OOF, TARGET, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, to_logit  # noqa: E402

EXT = os.path.join(DATA, "ext")
OUT = os.path.join(DATA, "ext_members2")
N_TR, N_TE = 691369, 296302

# An OOF AUC above this is not credible on this dataset: the best honest member in
# three independent public libraries is 0.96881, and our own cross-fitted 88-member
# stack only reaches 0.96968.
CREDIBLE_MAX = 0.9720


def collect():
    """Return {name: (oof, test)} for every candidate member in the new libraries."""
    out = {}

    # --- boltuzamaki: 47 streams in two parquets, with an id column ---
    d = os.path.join(EXT, "boltuzamaki_s6e8-oof-prediction-library")
    o = pd.read_parquet(os.path.join(d, "oof_predictions.parquet"))
    t = pd.read_parquet(os.path.join(d, "test_predictions.parquet"))
    tr, te = load_raw()
    assert (o["id"].to_numpy() == tr["id"].to_numpy()).all(), "bolt oof id order differs"
    assert (t["id"].to_numpy() == te["id"].to_numpy()).all(), "bolt test id order differs"
    published = pd.read_csv(os.path.join(d, "stream_index.csv")).set_index("stream")["oof_auc"]
    for c in o.columns:
        if c == "id":
            continue
        out[f"bolt_{c}"] = (o[c].to_numpy(np.float64), t[c].to_numpy(np.float64),
                            float(published.get(c, np.nan)))

    # --- mohankrishnathalla: one model per dataset, bare npy in row order ---
    for repo, tag in [("s6e8-xgb-oof", "xgb"), ("s6e8-lgb-dart-oof", "lgb"),
                      ("s6e8-cat-mlp-oof", "cat"), ("s6e8-cat-mlp-oof", "nn")]:
        d = os.path.join(EXT, f"mohankrishnathalla_{repo}")
        po, pt = os.path.join(d, f"oof_{tag}.npy"), os.path.join(d, f"test_{tag}.npy")
        if not (os.path.exists(po) and os.path.exists(pt)):
            continue
        out[f"mkt_{tag}"] = (np.load(po).ravel().astype(np.float64),
                             np.load(pt).ravel().astype(np.float64), np.nan)

    # --- najiama: numbered csv pairs. 01-05 are single models (already in the 74-lib
    # as naji01..naji05); 07+ are the author's own blends of them, so they are not new
    # information -- collected anyway so the paired test can say so with a number.
    d = os.path.join(EXT, "najiama_predicting-smartphone-addiction-oof-submission-csv")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith("_oof_predictions.csv"):
            continue
        stem = fn[: -len("_oof_predictions.csv")]
        sub = os.path.join(d, f"{stem}_submission.csv")
        if not os.path.exists(sub):
            continue
        oo, tt = pd.read_csv(os.path.join(d, fn)), pd.read_csv(sub)
        oc = [c for c in oo.columns if c != "id"][-1]
        tc = [c for c in tt.columns if c != "id"][-1]
        out[f"njm_{stem}"] = (oo[oc].to_numpy(np.float64), tt[tc].to_numpy(np.float64), np.nan)
    return out


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    # the pack we already stack, for the correlation column
    names, O, T = load_members(y, len(te),
                               extra_dirs=(os.path.join(DATA, "ext_members"),),
                               drop=set(DEFAULT_DROP))
    Zpack = to_logit(O)
    Zpack = (Zpack - Zpack.mean(0)) / Zpack.std(0)
    print(f"existing pack: {len(names)} members\n")

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for nm, (o, t, pub) in sorted(collect().items()):
        if o.shape != (N_TR,) or t.shape != (N_TE,):
            rows.append(dict(member=nm, verdict=f"SHAPE {o.shape} {t.shape}"))
            continue
        auc = roc_auc_score(y, o)
        zo, zt = to_logit(o), to_logit(t)
        sd_ratio = zt.std() / zo.std()
        z = (zo - zo.mean()) / zo.std()
        maxcorr = float(np.abs(Zpack.T @ z / len(z)).max())

        verdict = "ok"
        if np.isfinite(pub) and abs(auc - pub) > 5e-5:
            verdict = f"MISALIGNED pub={pub:.6f}"
        elif auc > CREDIBLE_MAX:
            verdict = "NOT CREDIBLE"
        elif nm in names or f"{nm}" in names:
            verdict = "dup"
        rows.append(dict(member=nm, auc=auc, pub=pub, sd_ratio=sd_ratio,
                         maxcorr=maxcorr, verdict=verdict))
        if verdict == "ok":
            np.save(os.path.join(OUT, f"oof_{nm}.npy"), o)
            np.save(os.path.join(OUT, f"test_{nm}.npy"), t)

    df = pd.DataFrame(rows).sort_values("maxcorr")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format="%.6f"))
    print(f"\nwrote {(df['verdict'] == 'ok').sum()} vetted members to {OUT}")
    df.to_csv(os.path.join(OUT, "_vetting.csv"), index=False)


if __name__ == "__main__":
    main()
