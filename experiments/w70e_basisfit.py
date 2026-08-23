"""w70e — CAN THE PUBLIC LEADERBOARD TELL THE TWO CORRECTED-CV BASES APART?

Pre-registration: experiments/w70_prereg.txt, committed d3d680c BEFORE this file existed.
Estimand, gates P1-P4 and the decision rule are in that file, BY REFERENCE and not restated.

⛔ NOT A DECISION. w70 computed the optimism table and is therefore barred from registering a
bar on it (prereg §0). This answers only whether a later run is ALLOWED to cite the LB when it
does decide. A null here is a real result: it forbids that citation.

    .venv/bin/python experiments/w70e_basisfit.py
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE) + "/agent")
sys.path.insert(0, HERE)

import w53a_pricer as PR                                          # noqa: E402

SWEEP = os.path.join(HERE, "w70a_optimism.json")
SEED, NBOOT, NSHUF = 70, 2000, 200
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def refit(cv6):
    """w53a's own fit, on a supplied cv6 column. The design and the row set are PR's."""
    t = PR._t
    X = np.vstack([PR._design(c, r.fam, r["std"], r["corr"], r.era)
                   for c, (_, r) in zip(cv6, t.iterrows())])
    y = t.lb6.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(t) - X.shape[1]
    return beta, resid, float(np.sqrt(float(resid @ resid) / dof))


def main() -> None:
    t = PR._t
    sweep = {r["tag"]: r for r in json.load(open(SWEEP))["rows"]}

    # ---- P1: VALIDITY. Model S must BE w53a. Hard stop. --------------------------------
    beta_s, resid_s, sd_s = refit(t.cv6.values)
    d_beta = float(np.max(np.abs(beta_s - PR.BETA)))
    d_sd = abs(sd_s - PR.RESID_SD)
    print("=" * 96)
    print("w70e  DOES THE LEADERBOARD PREFER A CORRECTED-CV BASIS?   (prereg w70_prereg.txt)")
    print("=" * 96)
    print(f"\nP1 VALIDITY — model S vs w53a itself:  max|dBETA| {d_beta:.3e}   "
          f"|dRESID_SD| {d_sd:.3e}")
    if d_beta > 1e-9 or d_sd > 1e-9:
        fail("model S does NOT reproduce w53a — the harness is not the pricer. HARD STOP.")
        print("\n  ⛔ Nothing below is readable. Registered as a hard stop in prereg §2 P1.")
        sys.exit(1)
    print("   ✅ reproduces w53a exactly.")

    # ---- the honest-basis cv6 column ---------------------------------------------------
    # optimism is in e-6 already once divided by U; cv6 is (cv - MU)*1e6, so subtracting the
    # optimism in e-6 IS the honest basis on this scale.
    opt6 = np.zeros(len(t))
    hit = []
    for i, stem in enumerate(t.stem.values):
        s = sweep.get(stem)
        if s is None:
            continue
        # sanity: this row's cv must actually be the SHIPPED basis, or the shift is meaningless
        if abs((t.cv6.values[i] / 1e6 + PR.MU) - s["shipped"]) > 5e-11:
            fail(f"{stem}: fit-table cv is not the shipped basis; the shift is undefined")
        opt6[i] = s["opt"] / 1e-6
        hit.append(stem)

    # ---- P2: POWER, printed BEFORE the reading -----------------------------------------
    n_corr = len(hit)
    print(f"\nP2 POWER — {n_corr} of {len(t)} fit rows are corrected files in the sweep.")
    if n_corr < 5:
        fail(f"only {n_corr} corrected rows (<5) — prereg §3: the design has no content.")
        print("\n  ⛔ Stopping. D_SD is not reported.")
        sys.exit(1)
    o = opt6[opt6 != 0]
    print(f"   optimism over the corrected rows: n_nonzero {len(o)}  mean {opt6[[t.stem.values.tolist().index(h) for h in hit]].mean():.3f}e-6  "
          f"sd {opt6[[t.stem.values.tolist().index(h) for h in hit]].std(ddof=1):.3f}e-6  max {opt6.max():.3f}e-6")

    beta_h, resid_h, sd_h = refit(t.cv6.values - opt6)
    D_SD = sd_h - sd_s

    rng = np.random.default_rng(SEED)
    boot = np.empty(NBOOT)
    n = len(t)
    for b in range(NBOOT):
        idx = rng.integers(0, n, n)
        rs, rh = resid_s[idx], resid_h[idx]
        boot[b] = (np.sqrt(float(rh @ rh) / len(rh)) - np.sqrt(float(rs @ rs) / len(rs)))
    se = float(boot.std(ddof=1))
    print(f"   paired bootstrap se of D_SD ({NBOOT} resamples, seed {SEED}): {se:.4f}e-6")

    # ---- THE READING -------------------------------------------------------------------
    print(f"\nTHE READING\n   RESID_SD(S) {sd_s:.4f}e-6   RESID_SD(H) {sd_h:.4f}e-6   "
          f"D_SD {D_SD:+.4f}e-6   ({abs(D_SD) / se:.2f} se)")
    if abs(D_SD) < 0.5 and abs(D_SD) < se:
        verdict = "UNINFORMATIVE — the LB cannot adjudicate the basis (the PREDICTED outcome)"
    elif D_SD <= -0.5 and abs(D_SD) >= 2 * se:
        verdict = "EVIDENCE FOR THE HONEST BASIS"
    elif D_SD >= 0.5 and abs(D_SD) >= 2 * se:
        verdict = "EVIDENCE FOR THE SHIPPED BASIS"
    else:
        why = ("|D_SD| >= 0.5 but < 2 se" if abs(D_SD) >= 0.5 else "|D_SD| < 0.5 but >= 1 se")
        verdict = f"UNINFORMATIVE — {why}"
    print(f"   P3 -> **{verdict}**")

    # ---- P4: SHUFFLED NEGATIVE CONTROL -------------------------------------------------
    # ⚠ GUARDS THE NULL BRANCH (prereg §2 P4). A WIDE shuffled spread makes a null STRONGER.
    rng2 = np.random.default_rng(SEED)
    pos = np.flatnonzero(opt6 != 0)
    allpos = np.flatnonzero(np.isin(t.stem.values, hit))
    shuf = np.empty(NSHUF)
    vals = opt6[allpos].copy()
    for k in range(NSHUF):
        v = opt6.copy()
        v[allpos] = rng2.permutation(vals)
        shuf[k] = refit(t.cv6.values - v)[2] - sd_s
    inside = float(np.mean(shuf <= D_SD))
    print(f"\nP4 SHUFFLED CONTROL — {NSHUF} relabellings of the same magnitudes across the "
          f"{len(allpos)} corrected rows")
    print(f"   shuffled D_SD: mean {shuf.mean():+.4f}  sd {shuf.std(ddof=1):.4f}  "
          f"range [{shuf.min():+.4f}, {shuf.max():+.4f}]e-6")
    print(f"   the TRUE D_SD sits at the {inside * 100:.1f}th percentile of that distribution.")
    if 0.05 <= inside <= 0.95:
        print("   ✅ the true relabelling is INDISTINGUISHABLE from an arbitrary one — which is "
              "what\n      P3's null verdict means, made concrete rather than asserted.")
    else:
        print("   ⚠ the true relabelling is OUTSIDE the shuffled bulk. Read P3 again: a null "
              "there\n      alongside this is a contradiction and must be explained, not "
              "reported.")

    out = dict(sd_s=sd_s, sd_h=sd_h, D_SD=D_SD, se=se, n_corr=n_corr, verdict=verdict,
               shuf_mean=float(shuf.mean()), shuf_sd=float(shuf.std(ddof=1)),
               true_pctile=inside, corrected_rows=hit, failures=FAILURES)
    with open(os.path.join(HERE, "w70e_basisfit.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote w70e_basisfit.json   FAILURES {FAILURES}")
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
