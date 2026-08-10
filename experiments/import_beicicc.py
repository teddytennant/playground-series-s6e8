"""Import the ten `beicicc/s6e8-*-artifacts` libraries into data/ext_members2/.

These are the best-documented public artifacts in this competition and the only ones
that ship `fold_id.npy`, which turns "trust me, same folds" into something checkable.

THE FOLD GATE
-------------
Every other library is accepted on the strength of reproducing a published OOF AUC,
which only proves the rows are in the right ORDER -- it says nothing about whether the
member was cross-validated on our partition. These ship the partition itself, so the
check is exact. Note the fold *labels* are permuted relative to ours (raw agreement is
0.0000), so compare the induced PARTITION, not the integers: cross-tabulate their fold
id against ours and require that every one of their folds maps entirely into one of
ours. All ten pass. A library that fails this gate is not importable at any AUC.

WHAT IS DELIBERATELY LEFT OUT
-----------------------------
`sixmember_*` are level-2 outputs -- a cross-fitted logistic stack over six members we
already hold, plus its equal-rank reference. Feeding a stack's own predictions back in
as a member of another stack is the thing agent/stack.py warns about, and this author
discloses the caveat himself: "a different meta-training row can come from a base model
that used labels from the current meta-validation fold ... may be optimistic". Excluded
on mechanism, not on CV, because CV is exactly what an optimistic OOF fools.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, to_logit  # noqa: E402

EXT = os.path.join(DATA, "ext")
OUT = os.path.join(DATA, "ext_members2")
N_TR, N_TE = 691369, 296302
CREDIBLE_MAX = 0.9720
# level-2 outputs -- see module docstring
SKIP_STEMS = ("sixmember_equal_rank", "sixmember_meta", "sixmember_meta_perp")


def fold_gate(path, ours):
    """True iff the shipped fold assignment induces exactly our partition."""
    a = np.load(path)
    if a.shape != ours.shape:
        return False
    ct = pd.crosstab(a, ours)
    return bool(ct.max(axis=1).sum() == len(ours))


def collect(ours):
    """{name: (oof, test)} for every *_oof.npy/_test.npy pair that clears the gate."""
    out, gate = {}, {}
    for d in sorted(glob.glob(os.path.join(EXT, "beicicc_*"))):
        fid = glob.glob(os.path.join(d, "*fold_id.npy"))
        ok = bool(fid) and fold_gate(fid[0], ours)
        gate[os.path.basename(d)] = ok
        if not ok:
            print(f"[FOLD GATE FAILED] {os.path.basename(d)} -- not importable")
            continue
        for po in sorted(glob.glob(os.path.join(d, "*_oof.npy"))):
            stem = os.path.basename(po)[: -len("_oof.npy")]
            pt = os.path.join(d, f"{stem}_test.npy")
            if not os.path.exists(pt) or stem in SKIP_STEMS:
                continue
            o, t = np.load(po).ravel(), np.load(pt).ravel()
            if o.shape != (N_TR,) or t.shape != (N_TE,):
                print(f"[skip] {stem}: shapes {o.shape} {t.shape}")
                continue
            out[f"bei_{stem}"] = (o.astype(np.float64), t.astype(np.float64))
    print(f"fold gate: {sum(gate.values())}/{len(gate)} libraries pass\n")
    return out


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ours = np.empty(len(y), int)
    for k, (_, iva) in enumerate(get_folds(y)):
        ours[iva] = k

    names, O, T = load_members(y, len(te),
                               extra_dirs=(os.path.join(DATA, "ext_members"),
                                           os.path.join(DATA, "ext_members2")),
                               drop=set(DEFAULT_DROP))
    Zpack = to_logit(O)
    Zpack = (Zpack - Zpack.mean(0)) / Zpack.std(0)
    print(f"existing pack: {len(names)} members")

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for nm, (o, t) in sorted(collect(ours).items()):
        auc = roc_auc_score(y, o)
        zo, zt = to_logit(o), to_logit(t)
        z = (zo - zo.mean()) / zo.std()
        maxcorr = float(np.abs(Zpack.T @ z / len(z)).max())
        verdict = "NOT CREDIBLE" if auc > CREDIBLE_MAX else "ok"
        rows.append(dict(member=nm, auc=auc, sd_ratio=zt.std() / zo.std(),
                         maxcorr=maxcorr, verdict=verdict))
        if verdict == "ok":
            np.save(os.path.join(OUT, f"oof_{nm}.npy"), o)
            np.save(os.path.join(OUT, f"test_{nm}.npy"), t)

    df = pd.DataFrame(rows).sort_values("maxcorr")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format="%.6f"))
    print(f"\nwrote {(df['verdict'] == 'ok').sum()} vetted members to {OUT}")
    old = os.path.join(OUT, "_vetting.csv")
    prev = pd.read_csv(old) if os.path.exists(old) else pd.DataFrame()
    pd.concat([prev, df], ignore_index=True).to_csv(old, index=False)


if __name__ == "__main__":
    main()
