"""w27l -- where the CatBoost lineage actually sits in the 188-member pack.

The slot's angle is "CatBoost handles categoricals better on survey data; tune and compare on
identical folds". The workspace has already answered the naive form of that: `cat_lat`,
`cat_native`, `cat_raw` and `mkt_cat` are in the pack, and w26's finding is that where a member
lands is set by its PIPELINE, not by its model class or its hyperparameters. So tuning another
CatBoost is a measured null before it starts.

The question that is NOT answered is whether the CatBoost lineage is carrying the thing that
actually pays in this pack, which is DECORRELATION rather than solo strength (adarsh1077's
leave-one-author-out and our own w20d per-member value both say so). This reads that straight
off the stored OOF vectors -- no fits, no combiner, seconds of arithmetic -- and it is the
cheapest honest test of the angle's premise available.

Rank-transform then Pearson, which is the workspace's standing member-profile convention.

⚠ The member list comes from `stack.load_members`, NOT from a directory list written out here.
A first version of this file enumerated the dirs by hand and silently missed `data/oof` -- the
74-model public library -- so it profiled 114 members and called them 188. The canonical loader
is the only thing that knows the real pack, and every count printed below is its count.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# w27h's exact pack definition: the drop list and the extra dirs, verbatim.
DROP = {"golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "lat_ctraw_r400", "lat_ctfixte_r400"}
EXTRA = ("ext_members", "ext_members2", "ext_members3", "ext_members4", "ext_members6")


def fastrank(v):
    """argsort-of-argsort. Ties are broken arbitrarily rather than averaged; with 691k
    continuous scores that is a difference far below any number reported here, and it is
    ~8x faster than scipy.rankdata on this shape."""
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype="float32")
    r[o] = np.arange(len(v), dtype="float32")
    return r / len(v)


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, _ = load_members(y, len(te),
                               extra_dirs=[os.path.join(DATA, d) for d in EXTRA],
                               drop=DROP)
    assert len(names) == 188, f"expected the 188-member pack, got {len(names)}"
    M = np.empty((len(names), len(y)), dtype="float32")
    for i in range(len(names)):
        M[i] = fastrank(O[:, i])
    del O
    print(f"{len(names)} members loaded via stack.load_members (w27h's exact pack)", flush=True)

    solo = np.array([roc_auc_score(y, M[i]) for i in range(len(names))])
    Mc = M - M.mean(1, keepdims=True)
    Mc /= np.linalg.norm(Mc, axis=1, keepdims=True)
    R = Mc @ Mc.T
    np.fill_diagonal(R, -1.0)
    mx = R.max(1)
    who = R.argmax(1)
    np.fill_diagonal(R, np.nan)
    med = np.nanmedian(R, axis=1)

    # Print every name containing "cat" so the lineage is CLASSIFIED BY EYE rather than by a
    # substring rule -- `xgb_latcat` and `xgb_cat_lattice` are XGBoost members whose *features*
    # are categorical, not CatBoost members, and a substring rule quietly mixes the two.
    print("  names containing 'cat': " + ", ".join(n for n in names if "cat" in n.lower()),
          flush=True)
    # ⚠ CLASSIFIED BY EYE from the list printed above, because no substring rule works here.
    # `xgb_latcat`, `xgb_latcat_avg3/s17/s23` and `xgb_cat_lattice` are XGBoost members whose
    # FEATURES are categorical, not CatBoost members; a `"cat" in name` rule sweeps them in and
    # a `startswith("cat_")` rule catches only 6 of the 29 that are really CatBoost. The first
    # version of this file used the latter and reported a "CatBoost-lineage median" over 6.
    CATLIN = [n for n in (
        "cat", "cat_tuned", "cat_lat", "cat_native", "cat_raw",
        "digit_cat", "imp_cat", "lat_cat", "latwide_cat",
        "pub_cat", "pubfe_cat", "pubmk_cat", "mkt_cat",
        "view_bounds_cat", "view_rank_cat", "view_resid_cat",
        "bei_exact_value_catboost_fixed4000",
        "bolt_cat_cpu5", "bolt_cat_dual_seed81", "bolt_cat_dual_view",
        "bolt_cat_nested_te", "bolt_cat_pair_evidence", "bolt_cat_unique",
        "bolt_foldsafe_te_cat",
        "ad_catnative", "ad_gcatd8", "ad_gcatlr02", "ad_gcatnote", "ad_gcatseed7",
    ) if n in names]
    NOTCAT = [n for n in names if "cat" in n.lower() and n not in CATLIN]
    REF = [n for n in ("lgbm_fixed_lat", "xgb_lat", "xgb_latcat", "lat_ctfix_r400",
                       "bolt_tabr_retrieval", "fmdeep", "mkt_nn") if n in names]
    order = np.argsort(mx)
    rk = {names[k]: p + 1 for p, k in enumerate(order)}

    print(f"\n=== the CatBoost lineage, and reference members, in the 188-pack ===")
    print(f"{'member':>22s} {'solo AUC':>10s} {'maxcorr':>9s} {'closest':>22s} "
          f"{'medcorr':>9s} {'decorr rank':>12s}")
    for n in CATLIN + ["--"] + REF:
        if n == "--":
            print("  " + "-" * 84)
            continue
        i = names.index(n)
        print(f"{n:>22s} {solo[i]:10.6f} {mx[i]:9.5f} {names[who[i]]:>22s} "
              f"{med[i]:9.5f} {rk[n]:>7d}/{len(names)}")

    print(f"\n=== the 12 most DECORRELATED members in the pack ===")
    for k in order[:12]:
        print(f"  {names[k]:>22s}  maxcorr {mx[k]:.5f}  solo {solo[k]:.6f}")

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w27l_profile.npz"),
             names=np.array(names), solo=solo, maxcorr=mx, medcorr=med,
             closest=np.array([names[j] for j in who]))
    print(f"\n=== the CatBoost lineage is {len(CATLIN)} of {len(names)} members "
          f"({100*len(CATLIN)/len(names):.0f}% of the pack) ===")
    print("  NOT counted as CatBoost (categorical FEATURES, other model class): "
          + ", ".join(NOTCAT))
    print(f"\n=== pack summary ===")
    print(f"  median maxcorr over all {len(names)} members : {np.median(mx):.5f}")
    ci = [names.index(n) for n in CATLIN]
    oi = [i for i in range(len(names)) if i not in set(ci)]
    print(f"  CatBoost lineage ({len(ci):>3d})  median maxcorr {np.median(mx[ci]):.5f}   "
          f"median solo {np.median(solo[ci]):.6f}   median decorr rank "
          f"{int(np.median([rk[n] for n in CATLIN]))}")
    print(f"  everything else  ({len(oi):>3d})  median maxcorr {np.median(mx[oi]):.5f}   "
          f"median solo {np.median(solo[oi]):.6f}")
    print(f"  whole pack       ({len(names):>3d})  median maxcorr {np.median(mx):.5f}   "
          f"median solo {np.median(solo):.6f}")
    print("\n  Read: LOWER maxcorr = more decorrelated = the property w20d and adarsh1077's "
          "leave-one-author-out both say actually pays in a saturated stack.")


if __name__ == "__main__":
    main()
