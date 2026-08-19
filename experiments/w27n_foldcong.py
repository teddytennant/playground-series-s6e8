"""w27n -- is every member of the 188-pack actually fitted on OUR fold split?

WHY THIS HAS NEVER BEEN RUN, AND WHY IT SHOULD HAVE BEEN
--------------------------------------------------------
The whole cross-fitted CV instrument -- every number this workspace selects on -- assumes each
member's OOF vector was produced by holding out the SAME rows we hold out. 188 members come from
seven sources and only some publish fold ids. A member trained on a foreign 5-fold split averages
five models into each of our folds; its OOF is then partly in-sample for our validation rows,
which inflates the stack CV in a way no amount of cross-fitting the COMBINER can detect.

@adarsh1077's `s6e8-my-best-cv-model-scored-worse-on-the-lb` (§7) publishes the cheap indirect
test and reports median ~+0.98 on his pool. This is that test on ours. Credit is his; the
implementation and the numbers here are ours.

THE TEST
--------
Some folds are intrinsically easier than others -- fold 3 is easier than fold 0 for EVERY honest
member, because of what happens to sit in it. So the shape of (per-fold AUC, centred on the
member's own mean) is a property of the SPLIT and should be shared by everything fitted on it.
A member trained on a foreign split has each of our folds scored by a blend of models that saw
some of those rows, which washes the shape out toward flat.

For each member: a = [AUC on fold k for k in 0..4], centred. Score = corr(a, ref) where ref is
the pack's median centred shape. Report the tail.

⚠ HOW TO READ A LOW SCORE. Low means "unusual or unstable", NOT "foreign split" -- adarsh's own
weakest model scored lowest on his pool, and a deliberate residual-direction array scored
negative because it is not trying to predict the target at all. This is a diagnostic that says
where to look, and it is not evidence on its own. Nothing is dropped on this number.
"""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members  # noqa: E402

DROP = {"golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "lat_ctraw_r400", "lat_ctfixte_r400"}
EXTRA = ("ext_members", "ext_members2", "ext_members3", "ext_members4", "ext_members6")


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, _ = load_members(y, len(te),
                               extra_dirs=[os.path.join(DATA, d) for d in EXTRA], drop=DROP)
    assert len(names) == 188, f"expected the 188-member pack, got {len(names)}"
    folds = get_folds(y)
    print(f"{len(names)} members, {len(folds)} folds", flush=True)

    A = np.zeros((len(names), len(folds)))
    for k, (_, iva) in enumerate(folds):
        yv = y[iva]
        for i in range(len(names)):
            A[i, k] = roc_auc_score(yv, O[iva, i])
        print(f"  fold {k}: mean member AUC {A[:, k].mean():.6f}", flush=True)

    C = A - A.mean(1, keepdims=True)
    ref = np.median(C, axis=0)
    print(f"\npack median centred fold shape (fold0..4), x1e6: "
          + "  ".join(f"{v*1e6:+8.1f}" for v in ref))
    print("  (a positive entry = that fold is intrinsically EASIER than the member's own mean)")

    cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    score = cn @ (ref / np.linalg.norm(ref))
    o = np.argsort(score)
    print(f"\nfold-congruence score: median {np.median(score):+.4f}   "
          f"10th pct {np.percentile(score, 10):+.4f}   min {score.min():+.4f}")
    print(f"  members below +0.90: {(score < 0.90).sum()} of {len(names)}")
    print(f"  members below +0.50: {(score < 0.50).sum()}")
    print(f"  members below  0.00: {(score < 0.00).sum()}")
    print("\n=== 15 LOWEST (look here first; low = unusual or unstable, NOT proof of a foreign split) ===")
    print(f"  {'member':>34s} {'score':>8s} {'solo AUC':>10s}   per-fold centred AUC x1e6")
    for i in o[:15]:
        print(f"  {names[i]:>34s} {score[i]:+8.4f} {roc_auc_score(y, O[:, i]):10.6f}   "
              + " ".join(f"{v*1e6:+8.0f}" for v in C[i]))
    np.savez(os.path.join(HERE, "w27n_foldcong.npz"),
             names=np.array(names), fold_auc=A, score=score, ref=ref)
    print(f"\nwrote experiments/w27n_foldcong.npz")


if __name__ == "__main__":
    main()
