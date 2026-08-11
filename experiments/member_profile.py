"""Profile a new member against the pack, IN THE SPACE THE STACKER IS FITTED IN.

WHY THIS SCRIPT EXISTS
----------------------
Two statistics have now been read wrong in this workspace by being measured in a space
the model does not use:

  * `sd_ratio` was computed on the raw logit scale and read as a defect detector. It is
    not one -- every saturating member reads low because its OOF is one model's output
    while its test array is a mean of five.
  * `bolt_extratrees_support`'s headline "maxcorr 0.811" came from `import_ext2.py`,
    which correlates on the raw `to_logit` scale, the scale whose clip flattens the tails
    of saturating members. In the hybrid space the stacker actually sees, the same member
    reads 0.9701.

Both errors have the same shape. So: any statistic used to judge a member is computed
here on the transformed matrix, and reported for more than one transform, because a
number that moves between transforms is a property of the transform.

    member_profile.py xgb_raw_nan xgb_cat_lattice
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, OOF, TARGET, load_raw  # noqa: E402
from stack import transform  # noqa: E402


def corr_against(z, Z):
    """Pearson correlation of one transformed column against every pack column."""
    a = (z - z.mean()) / (z.std() + 1e-12)
    B = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    return (B.T @ a) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--kinds", default="hybrid,rankraw")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    p = os.path.join(CACHE, "meta_hybrid.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run stack_lab.py build first")
    d = np.load(p, allow_pickle=True)
    pack_names = [n for n in d["names"]]
    print(f"pack: {len(pack_names)} members from cache/meta_hybrid.npz")

    for nm in a.members:
        op = os.path.join(OOF, f"oof_{nm}.npy")
        tp = os.path.join(OOF, f"test_{nm}.npy")
        if not os.path.exists(op):
            print(f"\n[{nm}] not built yet")
            continue
        O, T = np.load(op)[:, None], np.load(tp)[:, None]
        print(f"\n[{nm}] solo OOF AUC {roc_auc_score(y, O[:, 0]):.6f}")
        for kind in a.kinds.split(","):
            # the pack matrix on disk is the hybrid one; for any other kind the pack
            # columns would have to be rebuilt, so only the member side is re-transformed
            # and the comparison stays hybrid-vs-hybrid. Reported so the two agree or the
            # disagreement is visible.
            z, _ = transform(O, T, kind)
            c = np.abs(corr_against(z[:, 0].astype("float32"), d["Z"]))
            order = np.argsort(-c)
            top = ", ".join(f"{pack_names[i]} {c[i]:.4f}" for i in order[:a.top])
            print(f"  {kind:8s} vs hybrid pack: maxcorr {c.max():.4f}  "
                  f"median {np.median(c):.4f}  below 0.97: {(c < 0.97).sum()}/{len(c)}")
            print(f"           nearest: {top}")

    # what a member has to beat to count as decorrelated here: the pack's own profile
    Z = d["Z"]
    if len(pack_names) <= 400:
        Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        R = np.abs(Zs.T @ Zs) / len(Zs)
        np.fill_diagonal(R, 0.0)
        mx = R.max(1)
        print(f"\npack's own maxcorr: min {mx.min():.4f} "
              f"({pack_names[int(np.argmin(mx))]}), median {np.median(mx):.4f}, "
              f"{(mx < 0.97).sum()} of {len(mx)} below 0.97")


if __name__ == "__main__":
    main()
