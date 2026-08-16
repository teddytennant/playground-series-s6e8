"""w15f stage 5: is the rebuilt correction the SAME OBJECT as the author's?

This is the load-bearing validity check for the whole run and it has to be stated before
the CV number means anything. The author published no training code -- notebook cell 29
only recomposes his submission from a saved NPZ of test-space vectors -- so the teacher and
student hyperparameters here are a reconstruction from his prose, not a transcription.

If my correction rank-correlates strongly with his on the 296,302 test rows, then the CV
number stage 3 produces is a measurement OF HIS OBJECT and the run answers the mission.
If it correlates near zero, I have measured a different object that happens to be built the
same way, and the honest report is that his remains unmeasured -- which is a materially
weaker result and must not be dressed up as a refutation.

Both directions are reported. Nothing here is tuned toward agreement.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, SUB, TARGET  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(DATA, "w15e", "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
N_TEST = 296_302


def pct(v):
    v = np.asarray(v, np.float64)
    return rankdata(v) / len(v)


def zr(v):
    r = rankdata(np.asarray(v, np.float64))
    return (r - r.mean()) / r.std()


def main():
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()

    mine = np.load(os.path.join(HERE, "w15f_c_test.npy"))
    my_teacher = np.load(os.path.join(HERE, "w15f_teacher_test.npy"))
    assert mine.shape == (N_TEST,)

    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)
    teacher = pct(z["test_teacher"].astype(np.float64))
    student = pct(z["test_student"].astype(np.float64))
    residual = teacher - student
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    reference_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    his = (test_sq - reference_sq.mean()) * reference.std() / reference_sq.std()

    n = N_TEST
    out = {}
    print("=== the two TEACHERS ===")
    out["teacher_corr"] = float(zr(my_teacher) @ zr(z["test_teacher"]) / n)
    print(f"  mine vs his, rank corr: {out['teacher_corr']:+.5f}")

    print("\n=== the two CORRECTIONS ===")
    out["corr_corr"] = float(zr(mine) @ zr(his) / n)
    out["corr_pearson"] = float(np.corrcoef(mine, his)[0, 1])
    print(f"  mine sd {mine.std():.5f}  his sd {his.std():.5f}  "
          f"ratio {mine.std()/his.std():.3f}")
    print(f"  rank corr  {out['corr_corr']:+.5f}")
    print(f"  pearson    {out['corr_pearson']:+.5f}")

    # A correction of sd s at weight w is the same perturbation as sd s' at weight w*s/s'.
    out["matched_weight"] = float(0.10 * his.std() / mine.std())
    print(f"\n  weight that matches HIS perturbation size: {out['matched_weight']:.4f}"
          f"   (his published 0.10 x {his.std()/mine.std():.3f})")

    base_p = pd.read_csv(os.path.join(SUB, "blend159av_h3.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    out["mine_vs_base"] = float(zr(mine) @ zr(base_p) / n)
    out["his_vs_base"] = float(zr(his) @ zr(base_p) / n)
    print(f"\n  mine vs base blend {out['mine_vs_base']:+.5f}   "
          f"his vs base blend {out['his_vs_base']:+.5f}")

    json.dump(out, open(os.path.join(HERE, "w15f_fidelity.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'w15f_fidelity.json')}")


if __name__ == "__main__":
    main()
