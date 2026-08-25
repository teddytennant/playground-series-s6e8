"""w89a — is an externally published `fold_id.npy` OUR frozen partition, bit for bit?

Every external OOF pack this workspace has ever weighed asserts its fold scheme in prose
(`StratifiedKFold(5, shuffle=True, random_state=42)` on train.csv in original row order) and
the workspace has taken that assertion on trust, because prose is all szymonkapiski shipped.
`nhtquyn/s6e8-addiction` and beicicc ship the VECTOR. A vector can be checked.

This is a read of the field, NOT a step toward adoption. w81 closed ext_members17 on w51's
es-on-val clause, which no fold instrument can see, and nothing here reopens it.

    .venv/bin/python experiments/w89a_foldid.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
from common import get_folds, DATA  # noqa: E402

PACKS = {
    "nhtquyn/s6e8-addiction":                              "data/w89_nhtquyn/fold_id.npy",
    "beicicc/fixed900-structural-lgbm":                    "data/ext/beicicc_s6e8-fixed900-structural-lgbm-artifacts/fold_id.npy",
    "beicicc/sixmember-crossfit-logitlr":                  "data/ext/beicicc_s6e8-sixmember-crossfit-logitlr-artifacts/fold_id.npy",
    "beicicc/fixed4000-catboost-screen-relation":          "data/ext/beicicc_s6e8-fixed4000-catboost-screen-relation-artifacts/fold_id.npy",
}


def ours():
    """Our frozen partition, flattened to a per-row fold id in ORIGINAL row order."""
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["addicted_label"])["addicted_label"].values
    f = np.full(len(y), -1, dtype=np.int64)
    for k, (_, va) in enumerate(get_folds(y)):
        f[va] = k
    assert (f >= 0).all(), "our own partition left rows unassigned"
    return f, y


def compare(theirs, mine):
    """(exact, relabelled, agreement, label_map). See the empty-class note in main()."""
    ti = np.asarray(theirs).astype(np.int64)
    exact = bool(np.array_equal(ti, mine))
    tlab, mlab = np.unique(ti), np.unique(mine)
    conf = np.zeros((len(tlab), len(mlab)), dtype=np.int64)
    tpos = {v: i for i, v in enumerate(tlab.tolist())}
    mpos = {v: i for i, v in enumerate(mlab.tolist())}
    for a, b in zip(ti, mine):
        conf[tpos[a], mpos[b]] += 1
    perm = mlab[conf.argmax(axis=1)]
    bij = len(tlab) == len(mlab) and len(set(perm.tolist())) == len(mlab)
    rel = bool(bij and np.array_equal(np.array([perm[tpos[v]] for v in ti]), mine))
    return exact, rel, float((ti == mine).mean()), perm, tlab, conf


def selftest(mine, y):
    """The test is worthless if it cannot tell a relabelling from a repartition."""
    rng = np.random.default_rng(89)
    from sklearn.model_selection import StratifiedKFold
    fails = 0

    def chk(tag, arr, want_rel):
        nonlocal fails
        _, rel, _, _, _, _ = compare(arr, mine)
        ok = rel == want_rel
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {tag:52s} relabelled={rel} (want {want_rel})")

    # C1 our partition, labels rotated -- MUST pass. This is the beicicc case.
    chk("C1 ours, labels +1 (beicicc's 1-based convention)", mine + 1, True)
    # C2 our partition under an arbitrary label permutation -- MUST pass.
    chk("C2 ours, labels permuted [3,0,4,1,2]", np.array([3, 0, 4, 1, 2])[mine], True)
    # C3 a DIFFERENT seed on the same stratification -- MUST fail. Without this the test
    #    could be passing everything, and a foreign split is the thing it exists to catch.
    other = np.full(len(mine), -1, dtype=np.int64)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=999).split(np.zeros(len(y)), y)):
        other[va] = k
    chk("C3 StratifiedKFold(5, seed=999) -- a foreign split", other, False)
    # C4 the SAME foreign split, 1-based -- MUST still fail. Relabelling tolerance must not
    #    launder a repartition; C1 and C4 together pin that the tolerance is exactly a bijection.
    chk("C4 the seed=999 split, 1-based", other + 1, False)
    # C5 ours with 200 rows moved between folds -- a near-miss MUST fail, not round to pass.
    near = mine.copy()
    idx = rng.choice(len(near), 200, replace=False)
    near[idx] = (near[idx] + 1) % 5
    chk("C5 ours with 200 of 691,369 rows moved", near, False)
    # C6 ours with the row ORDER shuffled -- same multiset of labels, different assignment.
    chk("C6 ours, row order shuffled", mine[rng.permutation(len(mine))], False)
    # C7 a 4-fold split -- different K must fail on the bijection, not crash.
    four = np.full(len(mine), -1, dtype=np.int64)
    for k, (_, va) in enumerate(StratifiedKFold(4, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        four[va] = k
    chk("C7 a 4-fold split (different K)", four, False)
    print(f"\nselftest failures: {fails}")
    return fails


def main():
    mine, y = ours()
    if "--selftest" in sys.argv:
        print("SELFTEST — can this test tell a relabelling from a repartition?\n")
        return 1 if selftest(mine, y) else 0
    n = len(mine)
    print(f"our frozen partition: n={n}  fold sizes={np.bincount(mine).tolist()}")
    print(f"positives per fold  : {[int(y[mine == k].sum()) for k in range(mine.max() + 1)]}")

    fail = 0
    for name, path in PACKS.items():
        if not os.path.exists(path):
            print(f"\n{name}: MISSING {path}")
            continue
        theirs = np.load(path).ravel()
        print(f"\n{name}")
        print(f"  dtype={theirs.dtype} n={len(theirs)} folds={np.bincount(theirs.astype(np.int64)).tolist()}")
        if len(theirs) != n:
            print("  ⛔ DIFFERENT LENGTH — not comparable")
            fail += 1
            continue

        exact, relabelled, agree, perm, tlab, conf = compare(theirs, mine)
        K = len(np.unique(mine))

        print(f"  identical as labelled      : {exact}")
        print(f"  identical up to relabelling: {relabelled}  (their {tlab.tolist()} -> ours {perm.tolist()})")
        print(f"  raw per-row agreement      : {agree:.6f}   (chance = {1.0 / K:.6f})")
        if not (exact or relabelled):
            # How far off is it? Largest cell share tells you whether it is a near-miss
            # (e.g. a different seed on the same stratification) or a wholly foreign split.
            print(f"  ⚠ NOT our partition. max overlap per their-fold: "
                  f"{(conf.max(axis=1) / conf.sum(axis=1)).round(4).tolist()}")
            fail += 1
        else:
            print("  ✅ our frozen partition")

    print(f"\npacks failing the identity test: {fail}")
    # A real guard, not a report: a pack that stops being our partition is a pack
    # whose published OOF AUCs are no longer comparable with ours.
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
