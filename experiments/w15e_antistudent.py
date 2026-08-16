"""w15e: graft raykkretzschmar's transductive teacher-minus-student residual onto our
CV-best blend, at his published weight, and measure what it actually is.

WHY THIS OBJECT AND NOT A PUBLIC SUBMISSION
-------------------------------------------
`w15e_extcorr.py` measured every downloadable public submission against our pack. The
entire >=0.9709 public cluster sits at Spearman 0.9980-0.9988 against `blend159av_h3`,
and the members of that cluster are 0.9999+ against EACH OTHER -- several are the same
file re-published. There is no decorrelated-and-strong public ranking to blend with.

The one object in the public space that our pack cannot produce is not a ranking at all.
It is a TRANSDUCTIVE correction vector: a LightGBM teacher, a deliberately smoother
student fitted on the teacher's out-of-fold ranks WITH the unlabeled test rows and their
teacher predictions carried at weight 0.245, and the teacher-minus-student rank residual
as the correction. Every one of our 168 members is inductive -- fitted on train rows
only, applied to test rows blind. A signal defined by what a smoother model fails to
reconstruct on the TEST distribution is orthogonal to that by construction, and it is
the only thing here that is.

WHAT IS AND IS NOT VALIDATED
----------------------------
Not by us: the residual exists only on the 296,302 test rows, so it has no OOF and
cannot be scored on our frozen folds. That is a property of the object, not an oversight.
By its author, nested: five outer folds x four inner teacher folds regenerated the
correction for every labelled row, and adding the same 10% signed-square residual moved
four independent OOF anchors by +0.000018 / +0.000019 / +0.000036 / +0.000025, positive
in 60 of 60 anchor-by-fold comparisons. His notebook states the leaderboard chose nothing
and reports no LB delta as evidence.

NO WEIGHT IS FITTED HERE. The 0.10 is his published constant and the residual scale comes
from `reference_contrast` inside his own NPZ. Nothing in this file is tuned against any
leaderboard feedback -- that is the Rogii failure the brief names, and it is why the
weight is copied rather than searched.

BASE = `blend159av_h3`, our joint-best CV file (0.970049, zero fitted parameters above
the stack) and one of the two deadline picks. The base is chosen on CV, never on LB.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, OOF, SUB, TARGET  # noqa: E402

W15E = os.path.join(DATA, "w15e")
NPZ = os.path.join(W15E, "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
BASE = "blend159av_h3"          # CV 0.970049, LB 0.97105, ref 55488344
W_ANTI = 0.10                   # published constant, NOT searched here
N_TEST = 296_302


def pct(v):
    """Percentile rank in [0,1], ties averaged -- the author's `pct`."""
    return rankdata(np.asarray(v, np.float64)) / len(v)


def zr(v):
    r = rankdata(np.asarray(v, np.float64))
    return (r - r.mean()) / r.std()


def main():
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    assert (test["id"].to_numpy() == ids).all()

    base_df = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id").reindex(ids)
    base_p = base_df[TARGET].to_numpy(np.float64)
    assert np.isfinite(base_p).all() and len(base_p) == N_TEST
    base_rank = pct(base_p)

    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)      # his 2 x 120k holdouts
    teacher = pct(z["test_teacher"].astype(np.float64))
    student = pct(z["test_student"].astype(np.float64))
    residual = teacher - student

    # --- the author's construction, transcribed verbatim ---
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    reference_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    anti_student = ((test_sq - reference_sq.mean())
                    * reference.std() / reference_sq.std())
    print(f"residual scale {scale:.9f}   "
          f"anti_student sd {anti_student.std():.6f} "
          f"[{anti_student.min():+.4f}, {anti_student.max():+.4f}]")

    pred = base_rank + W_ANTI * anti_student

    # --- 1. is the correction ORTHOGONAL to what we already own? ---
    # This is the mission's question. A correction that our pack can already express is
    # worthless however well its author validated it.
    za = zr(anti_student)
    n = N_TEST
    corr_rows = []
    for d, tag in [(os.path.join(DATA, "oof", "oof"), "lib"),
                   (os.path.join(DATA, "ext_members"), "ext"),
                   (os.path.join(DATA, "ext_members2"), "ext2"),
                   (OOF, "own")]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.startswith("test_") or not fn.endswith(".npy"):
                continue
            v = np.load(os.path.join(d, fn))
            if v.shape != (N_TEST,):
                continue
            corr_rows.append((f"{tag}:{fn[5:-4]}", float(za @ zr(v) / n)))
    cdf = pd.DataFrame(corr_rows, columns=["member", "corr"]).sort_values("corr")
    print(f"\n=== anti_student vs our {len(cdf)} pack members (test-space Spearman) ===")
    print(f"  min {cdf['corr'].min():+.4f}  median {cdf['corr'].median():+.4f}  "
          f"max {cdf['corr'].max():+.4f}  max|.| {cdf['corr'].abs().max():.4f}")
    print("  most aligned members:")
    print(cdf.reindex(cdf["corr"].abs().sort_values(ascending=False).index)
          .head(6).to_string(index=False, float_format="%.4f"))
    print(f"\n  anti_student vs the base blend itself: "
          f"{float(za @ zr(base_p) / n):+.5f}")
    print(f"  anti_student vs teacher: {float(za @ zr(z['test_teacher']) / n):+.4f}   "
          f"vs student: {float(za @ zr(z['test_student']) / n):+.4f}")
    print(f"  teacher vs base blend:   {float(zr(z['test_teacher']) @ zr(base_p) / n):+.4f}"
          f"   (a sane teacher should sit near the pack)")

    # --- 2. how big a perturbation is this? ---
    rho = float(zr(pred) @ zr(base_p) / n)
    moved = int((rankdata(pred) != rankdata(base_p)).sum())
    print(f"\n=== perturbation size ===")
    print(f"  Spearman(corrected, base) = {rho:.7f}")
    print(f"  rows whose rank moved at all: {moved:,} / {N_TEST:,}")
    for q in (0.5, 0.9, 0.99, 1.0):
        d = np.abs(pct(pred) - base_rank)
        print(f"  |percentile shift| q{q:<5}: {np.quantile(d, q):.5f}")

    # --- 3. write the file: strict unique ranks, base_rank as the tie-breaker ---
    order = np.lexsort((ids, base_rank, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})

    assert sub.shape == (N_TEST, 2)
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    out = os.path.join(SUB, "w15e_antistudent.csv")
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}  rows={len(sub):,}  "
          f"range [{strict.min():.3e}, {strict.max():.6f}]")
    print(f"base = {BASE} (CV 0.970049, LB 0.97105)   weight = {W_ANTI} (published)")


if __name__ == "__main__":
    main()
