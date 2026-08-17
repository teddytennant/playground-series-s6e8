"""w17c — turn this slot's send into an OUT-OF-SAMPLE test of the instrument w17a just built.

w17a estimates, from simulation alone, how far two files' public scores can sit from their CV
difference. Nothing in the workspace has ever checked that estimate against a real public
score, because every check available so far reused files whose LB was already known when the
estimate was made. This slot has a free submission and can buy one honest out-of-sample point.

THE CANDIDATE
-------------
`w14a_repro159av_h3` is a REBUILD of `blend159av_h3` — same recipe, independent run — and has
never been sent. `blend159av_h3` was sent and scored 0.97105. So the pair has:
  - a tiny, known CV difference (both are plain h3 stacks, no fitted correction),
  - one LB already observed,
  - and the second LB not yet observed by anyone.

That is the only configuration in this workspace that can falsify the instrument rather than
be fitted by it. It is also, separately, the best-CV file this account has never sent whose
rank vector is distinct from an already-sent file (`w16m_widegrid` has a higher CV but is
rank-IDENTICAL to the already-sent `w16i_schemeavg`, so it would score 0.97107 by
construction and would be the forbidden re-send of an identical file).

THE PRE-REGISTERED PREDICTION is printed by this script BEFORE the file is submitted, and is
copied verbatim into the submission message and the journal.

    .venv/bin/python experiments/w17c_repropair.py --reps 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
HERE = os.path.join(ROOT, "experiments")
N_TEST, F_PUBLIC, GRID = 296_302, 0.20, 1e-5
N_PUB = int(round(N_TEST * F_PUBLIC))

NEW = "w14a_repro159av_h3"
REF = "blend159av_h3"
REF_LB = 0.97105                       # observed, from the live submission list
GATE_REF = 0.9700491721182696          # audit_results.csv


def auc(y, v):
    o = np.argsort(v, kind="stable")
    s, yy = v[o], y[o]
    n1 = float(yy.sum())
    n0 = float(s.size) - n1
    b = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    ends = np.concatenate((b[1:], [s.size]))
    avg = (b + ends + 1) * 0.5
    pos = np.add.reduceat(yy, b)
    return (float((avg * pos).sum()) - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260817)
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    v_new = np.load(os.path.join(SUB, f"oof_{NEW}.npy")).astype("float64")
    v_ref = np.load(os.path.join(SUB, f"oof_{REF}.npy")).astype("float64")
    assert len(v_new) == len(v_ref) == n

    cv_ref, cv_new = auc(y, v_ref), auc(y, v_new)
    d = cv_ref - GATE_REF
    print(f"GATE  {REF} pooled {cv_ref:.16f} vs stored {GATE_REF:.16f}  drift {d:+.3e}")
    assert abs(d) < 1e-12
    dcv = cv_new - cv_ref
    print(f"\n{NEW:22s} cross-fitted CV {cv_new:.10f}")
    print(f"{REF:22s} cross-fitted CV {cv_ref:.10f}   (LB {REF_LB})")
    print(f"  dCV (new - ref) {dcv*1e6:+.3f}e-6")
    rho = float(np.corrcoef(np.argsort(np.argsort(v_new)),
                            np.argsort(np.argsort(v_ref)))[0, 1])
    print(f"  Spearman rank correlation between the two vectors {rho:.6f}")

    rng = np.random.default_rng(a.seed)
    dsl = np.empty(a.reps)
    for r in range(a.reps):
        idx = rng.permutation(n)[:N_PUB]
        yy = y[idx]
        dsl[r] = auc(yy, v_new[idx]) - auc(yy, v_ref[idx])
    sd = float((dsl - dcv).std(ddof=1))
    print(f"\npaired slice sd over {a.reps} draws: {sd*1e6:.2f}e-6")

    # LB_new = LB_ref + dCV + eps, then rounded to the 1e-5 grid.
    lo = REF_LB + dcv - 1.96 * sd
    hi = REF_LB + dcv + 1.96 * sd
    centre = REF_LB + dcv
    print(f"\nPRE-REGISTERED PREDICTION for {NEW}")
    print(f"  point estimate  {centre:.7f}  ->  prints {round(centre, 5):.5f}")
    print(f"  95% interval    [{lo:.7f}, {hi:.7f}]  ->  grid values "
          f"{sorted({round(x, 5) for x in np.linspace(lo, hi, 400)})}")
    p_same = float((np.abs(np.round(REF_LB + dcv + (dsl - dcv), 5) - REF_LB) < 1e-9).mean())
    print(f"  P(prints exactly {REF_LB}, i.e. identical to {REF}) = {p_same:.3f}")
    print("\nWHAT WOULD FALSIFY THE INSTRUMENT")
    print(f"  A print outside the 95% interval above. With sd {sd*1e6:.1f}e-6 the interval")
    print(f"  spans {(hi-lo)/GRID:.1f} grid steps, so this is a real two-sided test: the")
    print("  outcome is NOT guaranteed to land inside, which is exactly the property")
    print("  wave-summary §4.2 says the nine ladder 'confirmations' lacked.")

    out = dict(new=NEW, ref=REF, ref_lb=REF_LB, cv_new=cv_new, cv_ref=cv_ref, dcv=dcv,
               spearman=rho, paired_sd=sd, reps=int(a.reps), seed=int(a.seed),
               point=centre, lo=lo, hi=hi, p_same_as_ref=p_same)
    with open(os.path.join(HERE, "w17c_repropair.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("\nwrote w17c_repropair.json")


if __name__ == "__main__":
    main()
