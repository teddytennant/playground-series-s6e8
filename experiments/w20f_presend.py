"""Pre-send verification for w20_ad187_h3, and the paired LB prediction, registered.

Checks, in the order a failure would matter:

  1. file sanity          -- 296,302 rows, ids identical to sample_submission, finite,
                             inside (0,1), no ties introduced by a bad write
  2. rank-distinctness    -- against EVERY submission CSV on disk, not a remembered list.
                             w18 found `w16m_widegrid` to be a rank-duplicate of an
                             already-sent file this way; a duplicate scores by
                             construction and wastes the slot.
  3. CV, recomputed       -- from the stored OOF vector, never from a journal number
  4. the paired LAW-IF prediction against the best-CV reference in the same family,
     using w17h/w18a's closed form (no simulation), so the registered distribution over
     grid cells exists BEFORE the file is uploaded.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, ROOT, SUB, TARGET  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
GRID = 1e-5
N_TE = 296302
# public slice fraction -- RESEARCH "The public slice, quantified once and for all"
PUB_FRAC = 0.20


def rhash(v):
    return hashlib.blake2b(rankdata(v).astype(np.int64).tobytes(), digest_size=6).hexdigest()


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def lawif_pair_sd(o_a, o_b, y, n_pub, m_t=N_TE):
    """Paired sd of (AUC_a - AUC_b) on the public slice -- LAW-IF, verbatim from w19a.

    S_x = S_t + S_p: the pseudo-test draw and the public-slice draw are the same
    operation at two sizes. Gated in w18a at median 0.51% against w17d's 6,000-draw
    simulation, so this is the workspace's standard instrument and is reused rather than
    re-derived -- a second implementation of a registered instrument is a second chance
    to get it wrong.
    """
    pos, neg = y == 1, y == 0
    n1, n0, n = int(pos.sum()), int(neg.sum()), len(y)
    A = np.empty((2, n1))
    B = np.empty((2, n0))
    for i, v in enumerate((o_a, o_b)):
        A[i] = midrank_cdf(np.sort(v[neg]), v[pos])
        B[i] = 1.0 - midrank_cdf(np.sort(v[pos]), v[neg])
    C1, C0 = np.cov(A), np.cov(B)
    pi1 = n1 / n
    S_t = (1.0 - m_t / n) * (C1 / (m_t * pi1) + C0 / (m_t * (1 - pi1)))
    S_p = (1.0 - n_pub / m_t) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1)))
    S = S_t + S_p
    u = np.array([1.0, -1.0])
    return float(np.sqrt(u @ S @ u))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "w20_ad187_h3"
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = pd.read_csv(os.path.join(SUB, f"{name}.csv"))

    print(f"=== 1. file sanity: {name}.csv ===")
    v = sub[TARGET].to_numpy()
    ok = dict(rows=len(sub) == N_TE,
              ids=bool(np.array_equal(sub["id"].to_numpy(), ss["id"].to_numpy())),
              finite=bool(np.isfinite(v).all()),
              in_unit=bool(v.min() > 0 and v.max() < 1),
              distinct=int(len(np.unique(v))))
    for k, x in ok.items():
        print(f"  {k:9s} {x}")
    assert ok["rows"] and ok["ids"] and ok["finite"], "FILE IS NOT SENDABLE"

    print("\n=== 2. rank-distinctness against every CSV on disk ===")
    h = rhash(v)
    dup = []
    for p in sorted(glob.glob(os.path.join(SUB, "*.csv"))):
        s = os.path.basename(p)[:-4]
        if s == name:
            continue
        w = pd.read_csv(p)
        if len(w) != N_TE or TARGET not in w:
            continue
        if rhash(w[TARGET].to_numpy()) == h:
            dup.append(s)
    print(f"  rhash {h}   duplicates: {dup if dup else 'NONE'}")
    assert not dup, f"rank-identical to {dup} -- would score by construction"

    print("\n=== 3. CV, recomputed from the stored OOF ===")
    oof = np.load(os.path.join(SUB, f"oof_{name}.npy"))
    cv = roc_auc_score(y, oof)
    inv = pd.read_csv(os.path.join(EXP, "w18_inventory.csv"))
    pick = inv[inv.stem == "w16i_schemeavg"].iloc[0]
    ref_name = sys.argv[2] if len(sys.argv) > 2 else "blend159av_h3"
    ref = inv[inv.stem == ref_name].iloc[0].copy()
    # w18_inventory.csv was built at 55 submissions and RESEARCH already flags the LB
    # column as the one that goes stale. An explicit override beats a silent NaN.
    if len(sys.argv) > 3:
        ref["lb"] = float(sys.argv[3])
    assert ref["lb"] == ref["lb"], f"{ref_name} has no LB on file -- pass it as argv[3]"
    print(f"  {name:18s} CV {cv:.10f}")
    print(f"  {'w16i_schemeavg':18s} CV {pick.cv:.10f}  (the standing deadline pick)"
          f"  -> d {cv - pick.cv:+.3e}")
    print(f"  {ref_name:18s} CV {ref.cv:.10f}  LB {ref.lb}"
          f"  -> d {cv - ref.cv:+.3e}")

    print("\n=== 4. registered paired LB prediction (LAW-IF, closed form) ===")
    o_ref = np.load(os.path.join(SUB, f"oof_{ref_name}.npy"))
    n_pub = int(round(N_TE * PUB_FRAC))
    sd = lawif_pair_sd(oof, o_ref, y, n_pub)
    dcv = cv - ref.cv
    centre = ref.lb + dcv
    print(f"  reference {ref_name}: LB {ref.lb}, dCV {dcv:+.4e}, paired sd {sd:.3e}")
    print(f"  centre {centre:.7f}")
    cells = {}
    # the window must straddle the CENTRE, not the reference: this file's dCV is +52e-6,
    # i.e. five grid steps, so a window centred on the reference registers almost none of
    # the mass and would make the prediction look falsified whatever landed.
    m0 = int(round((centre - ref.lb) / GRID))
    for m in range(m0 - 8, m0 + 9):
        c = round(ref.lb + m * GRID, 5)
        lo, hi = c - GRID / 2, c + GRID / 2
        p = norm.cdf((hi - centre) / sd) - norm.cdf((lo - centre) / sd)
        if p > 0.005:
            cells[f"{c:.5f}"] = round(float(p), 3)
    print(f"  registered distribution: {cells}")
    print(f"  P(strictly above the account best 0.97108) = "
          f"{1 - norm.cdf((0.971085 - centre) / sd):.3f}")

    json.dump(dict(name=name, cv=float(cv), rhash=h,
                   d_vs_pick=float(cv - pick.cv), d_vs_ref=float(dcv),
                   reference=ref_name, ref_lb=float(ref.lb), paired_sd=float(sd),
                   centre=float(centre), cells=cells,
                   p_above_account_best=float(1 - norm.cdf((0.971085 - centre) / sd))),
              open(os.path.join(EXP, "w20f_presend.json"), "w"), indent=1)
    print("\nwrote w20f_presend.json")


if __name__ == "__main__":
    main()
