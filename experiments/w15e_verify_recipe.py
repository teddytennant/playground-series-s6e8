"""w15e: verify the anti-student transcription against the author's own scored output.

The graft in `w15e_antistudent.py` is only worth a slot if the recipe was transcribed
correctly. There is a free check: the author's published file (`rayk_mixmeta`, LB 0.97100)
IS his 0.97099 base plus this correction plus three much smaller band-local terms. So:

    if the transcription is right, applying MY anti_student to HIS base must move that
    base measurably TOWARDS his published output.

Prediction, registered before running: corr(his_base + anti, his_output) must exceed
corr(his_base, his_output). If the transcription were wrong the correction would be a
random direction and would move it AWAY.

His 0.97099 three-source rank blend is `daniilkrasnovvv/s6e8-top-1-public-0-97099`
(public LB 0.97099, and rank-correlated 0.99993 with his output -- the closest published
file to it that is not itself a copy).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET  # noqa: E402

W15E = os.path.join(DATA, "w15e")
N = 296_302
HIS_BASE = os.path.join(W15E, "kout", "daniilkrasnovvv_s6e8-top-1-public-0-97099",
                        "submission.csv")
HIS_OUT = os.path.join(W15E, "kout",
                       "raykkretzschmar_mix-the-meta-models-then-learn-what-they-miss",
                       "submission.csv")


def pct(v):
    return rankdata(np.asarray(v, np.float64)) / len(v)


def zr(v):
    r = rankdata(np.asarray(v, np.float64))
    return (r - r.mean()) / r.std()


def load(p, ids):
    d = pd.read_csv(p)
    c = [x for x in d.columns if x != "id"][0]
    return d.set_index("id").reindex(ids)[c].to_numpy(np.float64)


def main():
    ids = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))["id"].to_numpy()
    z = np.load(os.path.join(W15E,
                             "raykkretzschmar_s6e8-transductive-anti-student-signals",
                             "transductive_signals.npz"))
    reference = z["reference_contrast"].astype(np.float64)
    residual = pct(z["test_teacher"]) - pct(z["test_student"])
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    reference_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    anti = (test_sq - reference_sq.mean()) * reference.std() / reference_sq.std()

    hb, ho = load(HIS_BASE, ids), load(HIS_OUT, ids)
    zho = zr(ho)
    before = float(zr(hb) @ zho / N)
    after = float(zr(pct(hb) + 0.10 * anti) @ zho / N)
    print(f"corr(his_base,        his_output) = {before:.7f}")
    print(f"corr(his_base + anti, his_output) = {after:.7f}")
    print(f"                             move = {after - before:+.7f}  "
          f"({'TOWARDS -- transcription confirmed' if after > before else 'AWAY -- WRONG'})")

    # A random direction of the same size is the control: the move must not be something
    # any perturbation of this magnitude would produce.
    rng = np.random.default_rng(15)
    ctrl = []
    for _ in range(20):
        sham = rng.permutation(anti)
        ctrl.append(float(zr(pct(hb) + 0.10 * sham) @ zho / N))
    ctrl = np.array(ctrl)
    print(f"\nshuffled-correction control (n=20, same values, permuted rows):")
    print(f"  mean {ctrl.mean():.7f}  sd {ctrl.std(ddof=1):.2e}  "
          f"vs base {before:.7f}  -> move {ctrl.mean() - before:+.7f}")
    print(f"  z of the real correction vs the control: "
          f"{(after - ctrl.mean()) / ctrl.std(ddof=1):+.1f}")

    # And how much of his base->output distance does the anti-student term alone explain?
    print(f"\nresidual distance to his output after the anti term: "
          f"{1 - after:.3e} vs {1 - before:.3e} before "
          f"({100 * (after - before) / (1 - before):.1f}% of the gap closed)")


if __name__ == "__main__":
    main()
