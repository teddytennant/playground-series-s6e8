"""w22a: is the h3-vs-ens4 public-LB reversal ONE draw or TEN?

See experiments/w22_prereg.txt, written before any number here was computed.

THE STANDING DEFENCE, which this run tests rather than repeats. Both deadline picks are
h3-side. The h3 side prints one 1e-5 reporting step BELOW its ens4 twin on the public LB in
every pair where both sides are scored, while CV puts h3 above ens4 in every one of them.
w16s §3 closed that as slice noise: a public-sized slice reverses a true ~5e-6 advantage
24% of the time, and the run of reversals "is ONE draw against a fixed slice".

w16s measured the MARGINAL reversal probability, for two pairs, separately. The claim that
the pairs are correlated enough to count as a single observation was ASSERTED, never
measured -- and it is the entire load-bearing element of the defence. It is checkable for
free, because the joint distribution of every pair on the SAME simulated slice is exactly
what w16k's instrument already draws.

Two things established before the code runs, both in the prereg:

  (a) It is TEN pairs, not the eight the journal says. Pairing lb_scores.json with the five
      newest API rows gives ten h3/ens4 pairs with both sides LB-scored and both sides
      stored as OOF, and the h3 side is one step low in all ten.

  (b) Rounding cannot explain the direction. Rounding to 1e-5 is monotone non-decreasing,
      so round(a) < round(b) IMPLIES a < b. Each of the ten observations is hard evidence
      that h3's true public-slice score is strictly below ens4's. Only the magnitude is
      censored, never the sign.

And the point that actually prices the threat, which no previous run has made: public and
private are DISJOINT COMPLEMENTARY halves of one fixed test set. A slice that happens to
favour ens4 mechanically pushes its complement toward h3. So the decision-relevant number
is not P(the public slice reverses); it is P(h3 still wins the PRIVATE complement | the
public slice reversed). Part 4 computes it.

Protocol is w16k/w16s's, unchanged, so Part 2's marginals are directly comparable to the
record: draw N_TEST row indices without replacement from train, first n_pub are the
public-sized slice, the rest are the private-sized complement, every file scored on the
same draw. Seed 1616 and f 0.20 are w16s's. REPS is raised 500 -> 2000 because a joint
tail probability needs many more draws than a marginal mean.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, load_raw  # noqa: E402

from w15i_cvlb import prep, subset_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 296_302
REPS = 2000
SEED = 1616
F = 0.20

# (tag, h3-side oof stem, ens4-side oof stem, h3 public print, ens4 public print)
# Every row: both sides scored on the public LB, both sides stored as OOF, h3 one step low.
PAIRS = [
    ("blend156",        "blend156_h3",         "blend156",           0.97105, 0.97106),
    ("blend158",        "blend158_h3",         "blend158",           0.97105, 0.97106),
    ("blend159",        "blend159_h3",         "blend159",           0.97105, 0.97106),
    ("blend159av",      "blend159av_h3",       "blend159av",         0.97105, 0.97106),
    ("blend160orig",    "blend160orig_h3",     "blend160orig",       0.97105, 0.97106),
    ("blend160origm",   "blend160origm_h3",    "blend160origm",      0.97105, 0.97106),
    ("w14a_repro159av", "w14a_repro159av_h3",  "w14a_repro159av",    0.97105, 0.97106),
    ("corr159",         "w16i_schemeavg",      "w16q_ens4avg",       0.97107, 0.97108),
    ("ad187",           "w20_ad187_h3",        "w20_ad187",          0.97115, 0.97116),
    ("ad187corr",       "w21_ad187corr",       "w21_ad187corr_ens4", 0.97117, 0.97118),
]

# w16s §3's published marginals, used as a reproduction gate on the instrument.
W16S_MARGINAL = {"blend159av": 0.244, "corr159": 0.248}


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    print(f"train rows {n:,}   positives {y.sum():,}")

    stems = sorted({s for _, a, b, _, _ in PAIRS for s in (a, b)})
    pk = {}
    for s in stems:
        v = np.load(os.path.join(SUB, f"oof_{s}.npy")).astype(np.float64)
        assert v.shape == (n,), (s, v.shape)
        pk[s] = prep(v, y)
    print(f"loaded {len(stems)} OOF vectors for {len(PAIRS)} pairs")

    # --- Part 1: CV deltas ------------------------------------------------------------
    full = np.ones(n, dtype=bool)
    cv = {s: subset_auc(pk[s], full) for s in stems}
    print("\n=== Part 1: CV delta (h3 - ens4), full train OOF ===")
    print(f"  {'pair':16s} {'CV h3':>13s} {'CV ens4':>13s} {'d (e-6)':>10s}  LB")
    cvd = {}
    for tag, a, b, pa, pb in PAIRS:
        d = cv[a] - cv[b]
        cvd[tag] = d
        print(f"  {tag:16s} {cv[a]:.9f} {cv[b]:.9f} {d*1e6:+10.3f}  "
              f"{pa:.5f} vs {pb:.5f}  {'h3 LOW' if pa < pb else '??'}")
    npos = sum(v > 0 for v in cvd.values())
    print(f"  CV puts h3 above ens4 in {npos}/{len(PAIRS)} pairs; "
          f"the public LB puts h3 below ens4 in {sum(pa < pb for _,_,_,pa,pb in PAIRS)}/{len(PAIRS)}.")

    # --- Part 2: 2000 paired draws ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    n_priv = N_TEST - n_pub
    print(f"\n=== Part 2: {REPS} draws | public-sized {n_pub:,} rows | "
          f"private-sized {n_priv:,} rows ===")
    tags = [p[0] for p in PAIRS]
    DP = np.zeros((REPS, len(PAIRS)))       # public-sized delta, h3 - ens4
    DV = np.zeros((REPS, len(PAIRS)))       # private-sized complement delta
    for i in range(REPS):
        idx = rng.choice(n, size=N_TEST, replace=False)
        mp = np.zeros(n, dtype=bool)
        mp[idx[:n_pub]] = True
        mv = np.zeros(n, dtype=bool)
        mv[idx[n_pub:]] = True
        sub_p = {s: subset_auc(pk[s], mp) for s in stems}
        sub_v = {s: subset_auc(pk[s], mv) for s in stems}
        for k, (_, a, b, _, _) in enumerate(PAIRS):
            DP[i, k] = sub_p[a] - sub_p[b]
            DV[i, k] = sub_v[a] - sub_v[b]
        if (i + 1) % 250 == 0:
            print(f"  ... {i+1}/{REPS}")

    rev = DP < 0                            # h3 below ens4 on the public-sized slice
    marg = rev.mean(axis=0)
    print("\n=== Part 3 (M3): MARGINAL P(public-sized slice reverses this pair) ===")
    print(f"  {'pair':16s} {'CV d(e-6)':>10s} {'pub mean':>10s} {'pub sd':>9s} {'P(rev)':>8s}  gate")
    for k, tag in enumerate(tags):
        g = ""
        if tag in W16S_MARGINAL:
            se = (marg[k] * (1 - marg[k]) / REPS) ** 0.5
            z = (marg[k] - W16S_MARGINAL[tag]) / se
            g = f"w16s {W16S_MARGINAL[tag]:.3f}  z={z:+.2f} {'PASS' if abs(z) < 3 else 'FAIL'}"
        print(f"  {tag:16s} {cvd[tag]*1e6:+10.3f} {DP[:,k].mean()*1e6:+10.3f} "
              f"{DP[:,k].std(ddof=1)*1e6:9.3f} {marg[k]:8.3f}  {g}")

    # --- Part 4 (M4): the joint ---------------------------------------------------------
    allrev = rev.all(axis=1)
    joint = allrev.mean()
    nrev = rev.sum(axis=1)
    print("\n=== Part 4 (M4): the JOINT — all ten on the SAME slice ===")
    print(f"  P(all {len(PAIRS)} reverse together)   {joint:.4f}   "
          f"({allrev.sum()} of {REPS} draws)")
    print(f"  mean marginal P(reverse)        {marg.mean():.4f}")
    print(f"  independence would give         {np.prod(marg):.3e}")
    print("  distribution of #pairs reversed on one slice:")
    for c in range(len(PAIRS) + 1):
        m = (nrev == c).mean()
        if m > 0:
            print(f"    {c:2d} reversed  {m:7.4f}  {'#' * int(round(m * 120))}")
    C = np.corrcoef(DP.T)
    off = C[np.triu_indices(len(PAIRS), 1)]
    print(f"\n  correlation of the ten slice-level deltas: "
          f"min {off.min():.4f}  median {np.median(off):.4f}  max {off.max():.4f}")
    print(f"  {'':16s} " + " ".join(f"{t[:7]:>7s}" for t in tags))
    for k, tag in enumerate(tags):
        print(f"  {tag:16s} " + " ".join(f"{C[k,j]:7.3f}" for j in range(len(PAIRS))))

    # --- Part 5 (M5): the complementarity that actually prices the threat ---------------
    print("\n=== Part 5 (M5): public and private are DISJOINT COMPLEMENTS ===")
    print("  If a slice favours ens4, its complement is mechanically pushed toward h3.")
    print("  The decision quantity is P(h3 wins PRIVATE | the public slice reversed).")
    print(f"  {'pair':16s} {'corr(pub,priv)':>15s} {'P(h3 priv)':>11s} "
          f"{'P(h3 priv|pub rev)':>19s} {'E[priv|pub rev] e-6':>20s}")
    m5 = {}
    for k, tag in enumerate(tags):
        r = np.corrcoef(DP[:, k], DV[:, k])[0, 1]
        pw = (DV[:, k] > 0).mean()
        sel = rev[:, k]
        pwc = (DV[sel, k] > 0).mean() if sel.sum() else float("nan")
        ec = DV[sel, k].mean() * 1e6 if sel.sum() else float("nan")
        m5[tag] = dict(corr_pub_priv=float(r), p_h3_priv=float(pw),
                       p_h3_priv_given_pubrev=float(pwc), e_priv_given_pubrev=float(ec),
                       n_pubrev=int(sel.sum()))
        print(f"  {tag:16s} {r:+15.4f} {pw:11.3f} {pwc:19.3f} {ec:+20.3f}")

    sel = allrev
    print(f"\n  conditioning on ALL TEN reversing at once ({sel.sum()} draws):")
    if sel.sum() >= 20:
        for k, tag in enumerate(tags):
            print(f"    {tag:16s} P(h3 wins private | all ten reversed) "
                  f"{(DV[sel,k] > 0).mean():.3f}   "
                  f"E[priv d] {DV[sel,k].mean()*1e6:+8.3f}e-6")
    else:
        print(f"    too few draws ({sel.sum()}) to condition on; see the per-pair column above.")

    out = dict(reps=REPS, seed=SEED, f=F, n_pub=n_pub, n_priv=n_priv,
               pairs=[p[0] for p in PAIRS],
               cv_delta={t: float(cvd[t]) for t in tags},
               cv={s: float(cv[s]) for s in stems},
               marginal_reverse={t: float(marg[k]) for k, t in enumerate(tags)},
               pub_mean={t: float(DP[:, k].mean()) for k, t in enumerate(tags)},
               pub_sd={t: float(DP[:, k].std(ddof=1)) for k, t in enumerate(tags)},
               joint_all_reverse=float(joint),
               joint_count=int(allrev.sum()),
               indep_product=float(np.prod(marg)),
               nrev_hist=[float((nrev == c).mean()) for c in range(len(PAIRS) + 1)],
               corr_matrix=C.tolist(), m5=m5)
    with open(os.path.join(HERE, "w22a_jointslice.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.join(HERE, "w22a_DP.npy"), DP)
    np.save(os.path.join(HERE, "w22a_DV.npy"), DV)
    print("\nwrote experiments/w22a_jointslice.json, w22a_DP.npy, w22a_DV.npy")


if __name__ == "__main__":
    main()
