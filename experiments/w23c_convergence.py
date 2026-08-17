"""w23c: is the standardisation gain a BETTER OPTIMUM or a BETTER PRIOR? One statistic
decides it, and the reading is registered here before the run.

WHAT PROVOKED THIS
------------------
w23a Part B, rep 0, K=187, n=553,095, same holdout rows:

    std=0  lam=0 (C=1, the SHIPPED setting)   holdout AUC 0.970360   45s
    std=1  lam=0 (C=1)                        holdout AUC 0.970375   10s

+15e-6 and 4.5x faster. My own prereg (§3 H4) registered standardisation as a NULL
worth "0 to +5e-6", so this is a miss and it is the largest single blending effect in
the workspace since the 22-member import. But +15e-6 is worthless until the MECHANISM
is known, because the two candidate mechanisms have opposite consequences:

  (i) UNDER-CONVERGENCE. `LogisticRegression` stops on `tol=1e-4` compared against the
      max gradient, and gradients carry the columns' scale. Hybrid member sds run
      1.81..27.57 (measured), so one fixed `tol` is a ~15x looser stopping rule on the
      narrow columns than on the wide ones. Standardising makes the metric isotropic
      and lbfgs then walks further downhill. If this is it, the shipped stacks --
      EVERY blend CV number in the journal -- come from a fit that stopped short, the
      fix is free and mechanical, and it applies to the fold fits and the full test
      fit alike.

  (ii) A BETTER PRIOR. At C=1 the L2 penalty is not scale-invariant, so standardising
      changes which members get shrunk (the 08-13 note: "at any C the widest member is
      shrunk ~230x less than the narrowest"). If this is it, the gain is a genuine
      regularisation effect, it should track the lam sweep, and it is a modelling
      choice rather than a bug.

THE DISCRIMINATOR
-----------------
The UNPENALISED mean log-loss on the FIT rows, which is a single objective both
parameterisations are trying to minimise and which is invariant to column scaling.

    (i) under-convergence  =>  std=1 has LOWER fit log-loss (it found a better point
                               on the same surface) AND higher holdout AUC
    (ii) better prior      =>  std=1 has HIGHER fit log-loss (it deliberately fits the
                               training rows worse) AND higher holdout AUC

REGISTERED BEFORE RUNNING: I expect (i), because the C sweep in w23a is FLAT across a
1000x range of penalty (0.970358..0.970362 for C from 181 down to 0.18), and a penalty
that does nothing when varied 1000x cannot be worth +15e-6 when merely re-metricised.
Registered prediction: std=1 fit log-loss is lower by >1e-6 nats, and tightening `tol`
on the unstandardised fit recovers most of the +15e-6.

Third and fourth arms make that last clause testable rather than rhetorical: both
parameterisations are refitted at a 1000x tighter tol. Under (i) the four arms collapse
toward two numbers that agree; under (ii) the tight-tol arms stay apart by roughly the
original gap.

PART 2 — AND IT REPAIRS w23a's OWN PART A
-----------------------------------------
w23a's learning surface was measured entirely with std=0, i.e. with whichever fit this
module is about to indict. Its headline D_187 = +15.20e-6 (se 5.58) therefore mixes two
things: real learning-curve gain from 442k -> 553k rows, and optimiser noise, which
also grows with K. w23a's own GATE 2 FAILED at exactly that scale (C=181 and C=1 should
give the same answer and differed by 5.28e-6), so the confound is not hypothetical.

Part 2 re-runs only the K=187 row of the learning surface with std=1, same three
holdouts, same nested row subsets, same seeds. Twelve fits, ~10s each, because that is
the other half of the finding: the standardised fit is 4.5x faster. Registered
prediction: D_187 SHRINKS and its se shrinks more, because part of +15.20e-6 is
optimiser wobble rather than rows.

    w23c_convergence.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, SUB, TARGET, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402
from blend_lab import HONEST_DROP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = 0.20
REP = 0                      # w23a rep 0 exactly: StratifiedShuffleSplit random_state 1000


def logloss(y, z):
    """Mean unpenalised logistic loss from the decision function, computed stably."""
    z = np.asarray(z, np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def main():
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    Z, _ = transform(O, T, "hybrid")
    Z = Z.astype("float64")
    del O, T, _
    print(f"{len(names)} members, Z {Z.shape}, loaded {time.time()-t0:.0f}s", flush=True)

    ipool, ihold = next(StratifiedShuffleSplit(1, test_size=HOLD, random_state=1000 + REP)
                        .split(np.zeros(len(y)), y))
    ipool, ihold = np.sort(ipool), np.sort(ihold)
    Ztr, Zho = Z[ipool], Z[ihold]
    ytr, yho = y[ipool], y[ihold]
    s = Ztr.std(0)
    s[s <= 0] = 1.0
    print(f"pool {len(ipool):,}  holdout {len(ihold):,}  col sd {s.min():.2f}..{s.max():.2f}",
          flush=True)

    # tol=1e-7 is a 1000x tightening, which is enough to separate the two mechanisms;
    # 1e-9 was in the docstring's first draft and was cut because lbfgs would spend
    # thousands of iterations chasing rounding noise for no extra discrimination.
    arms = [("std0_tol1e-4", False, 1e-4, 5000),     # the SHIPPED setting
            ("std1_tol1e-4", True,  1e-4, 5000),     # w23a's winner
            ("std0_tol1e-7", False, 1e-7, 8000),
            ("std1_tol1e-7", True,  1e-7, 8000)]

    rows = []
    for tag, std, tol, mi in arms:
        A, B = (Ztr / s, Zho / s) if std else (Ztr, Zho)
        t = time.time()
        m = LogisticRegression(max_iter=mi, C=1.0, tol=tol).fit(A, ytr)
        el = time.time() - t
        # everything below is reported in the ORIGINAL column metric so the four arms
        # are directly comparable: w_orig = w_std / s.
        w = (m.coef_.ravel() / s) if std else m.coef_.ravel()
        b = float(m.intercept_[0])
        ztr = Ztr @ w + b
        zho = Zho @ w + b
        rows.append(dict(arm=tag, std=std, tol=tol,
                         n_iter=int(np.ravel(m.n_iter_)[0]), secs=round(el, 1),
                         fit_logloss=logloss(ytr, ztr),
                         hold_auc=roc_auc_score(yho, zho),
                         wnorm=float(np.linalg.norm(w)),
                         wnorm_std=float(np.linalg.norm(w * s))))
        print(f"  {tag:<14s} iters {rows[-1]['n_iter']:>6d}  {el:5.1f}s  "
              f"fit_logloss {rows[-1]['fit_logloss']:.9f}  "
              f"hold_auc {rows[-1]['hold_auc']:.6f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "w23c_convergence.csv"), index=False)
    print("\n" + df.to_string(index=False, float_format="%.9f"))

    g = {r["arm"]: r for r in rows}
    d_ll = g["std1_tol1e-4"]["fit_logloss"] - g["std0_tol1e-4"]["fit_logloss"]
    d_auc = g["std1_tol1e-4"]["hold_auc"] - g["std0_tol1e-4"]["hold_auc"]
    tight = g["std0_tol1e-7"]["hold_auc"] - g["std0_tol1e-4"]["hold_auc"]
    resid = g["std1_tol1e-7"]["hold_auc"] - g["std0_tol1e-7"]["hold_auc"]
    verdict = ("(i) UNDER-CONVERGENCE" if d_ll < 0 else "(ii) BETTER PRIOR")
    print("\n" + "=" * 78)
    print(f"std1 - std0 at the shipped tol:  fit_logloss {d_ll:+.3e} nats, "
          f"holdout AUC {d_auc*1e6:+.2f}e-6")
    print(f"  -> {verdict}   (lower fit log-loss = a better point on the SAME surface)")
    print(f"tightening tol on the UNSTANDARDISED fit alone recovers "
          f"{tight*1e6:+.2f}e-6 of that {d_auc*1e6:+.2f}e-6")
    print(f"residual std1-std0 gap once BOTH are converged: {resid*1e6:+.2f}e-6  "
          f"(near 0 => purely a metric/stopping artefact; ~{d_auc*1e6:+.0f}e-6 => a real prior)")
    print(f"coefficient norms in the standardised metric: "
          f"std0 {g['std0_tol1e-4']['wnorm_std']:.3f} vs std1 "
          f"{g['std1_tol1e-4']['wnorm_std']:.3f}")

    # ---------------- PART 2: w23a's K=187 learning row, redone standardised -------
    from w23a_bagasym import N_FRAC, nested_strat
    print("\n" + "=" * 78)
    print("PART 2  w23a's K=187 learning row, refitted with std=1 (same seeds/holdouts)")
    lc = []
    for rep in range(3):
        ip, ih = next(StratifiedShuffleSplit(1, test_size=HOLD, random_state=1000 + rep)
                      .split(np.zeros(len(y)), y))
        ip, ih = np.sort(ip), np.sort(ih)
        perm = nested_strat(y, ip, np.random.default_rng(230817 + rep))
        for f in N_FRAC:
            rr = np.sort(perm[: int(round(f * len(perm)))])
            sc = Z[rr].std(0)
            sc[sc <= 0] = 1.0
            t = time.time()
            m = LogisticRegression(max_iter=5000, C=1.0).fit(Z[rr] / sc, y[rr])
            auc = roc_auc_score(y[ih], m.decision_function(Z[ih] / sc))
            lc.append(dict(rep=rep, frac=f, n_fit=len(rr), auc=auc))
            print(f"  rep {rep} f={f:<6} n={len(rr):>7,d}  {auc:.6f}  "
                  f"({time.time()-t:.0f}s)", flush=True)
    dl = pd.DataFrame(lc)
    dl.to_csv(os.path.join(HERE, "w23c_curve187.csv"), index=False)
    w = dl.pivot_table(index="rep", columns="frac", values="auc")
    d187 = (w[1.0] - w[0.8]).to_numpy()
    se187 = d187.std(ddof=1) / np.sqrt(len(d187))
    sign = "consistent" if (d187 > 0).all() or (d187 < 0).all() else "SIGN FLIPS"
    print(f"\n  D_187 (std=1) {d187.mean()*1e6:+.2f}e-6  se {se187*1e6:.2f}  "
          f"reps {np.round(d187*1e6,2).tolist()}  [{sign}]")
    print(f"  w23a's std=0 reading was +15.20e-6 se 5.58 [consistent]")
    print("  full-curve deficit vs n=|P| (e-6): " + "  ".join(
        f"f={f}: {(w[f]-w[1.0]).mean()*1e6:+7.2f}" for f in N_FRAC))

    with open(os.path.join(HERE, "w23c_convergence.json"), "w") as f:
        json.dump(dict(arms=rows, d_logloss=d_ll, d_hold_auc=d_auc,
                       tol_recovers=tight, residual_gap=resid, verdict=verdict,
                       D187_std1=dict(mean=float(d187.mean()), se=float(se187),
                                      reps=[float(v) for v in d187],
                                      consistent=bool((d187 > 0).all())),
                       curve187_std1={str(f): float(w[f].mean()) for f in N_FRAC}),
                  f, indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
