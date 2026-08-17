"""w22b: split w22a's corr(public, private) = -0.205 into its two opposing channels.

See experiments/w22b_prereg.txt, written after w22a printed and before this ran.

w22a settled that the ten h3/ens4 public reversals are ONE latent draw (joint 0.190 against
a 0.280 mean marginal, correlations 0.94-0.99, bimodal reversal histogram). It also found
that conditioning on the public reversal RAISES P(h3 wins private). But that -0.205 is a
mixture of two channels that point opposite ways, and w22a re-draws the pseudo-test block
every rep so it cannot separate them:

  BETWEEN-BLOCK, positive and THREATENING: a block that is simply bad for h3 makes both of
  its halves bad. Under this channel the public reversal is evidence the test rows
  disfavour h3, and it carries into private.

  WITHIN-BLOCK, negative and HARMLESS: for a fixed block the two halves partition it, so an
  h3-poor half forces an h3-rich complement. Pure split noise; it anti-carries.

Which one dominates is what decides whether 10/10 is a threat. Nested design: B blocks,
S splits within each block, so the two channels are identified separately.

One thing worth being explicit about, since it bounds every number below. In reality the
test set is a single FIXED 296,302-row set and the public/private split is also fixed; the
randomness here stands in for our ignorance of both. The between-block channel is therefore
not an artefact -- test rows really are one draw from the same pool as train rows -- and the
variance decomposition in Part 4 is the honest statement of how much of the reversal risk
is "unlucky test set" versus "unlucky split".
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
from w22a_jointslice import F, N_TEST, PAIRS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B = 40          # pseudo-test blocks
S = 50          # public/private splits within each block
SEED = 22022


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    stems = sorted({s for _, a, b, _, _ in PAIRS for s in (a, b)})
    pk = {s: prep(np.load(os.path.join(SUB, f"oof_{s}.npy")).astype(np.float64), y)
          for s in stems}
    tags = [p[0] for p in PAIRS]
    npair = len(PAIRS)
    n_pub = int(round(N_TEST * F))
    print(f"train {n:,} | {B} blocks x {S} splits = {B*S} evals | "
          f"public {n_pub:,} / private {N_TEST-n_pub:,}")

    rng = np.random.default_rng(SEED)
    DP = np.zeros((B, S, npair))
    DV = np.zeros((B, S, npair))
    for b in range(B):
        block = rng.choice(n, size=N_TEST, replace=False)
        for s in range(S):
            perm = rng.permutation(N_TEST)
            mp = np.zeros(n, dtype=bool)
            mp[block[perm[:n_pub]]] = True
            mv = np.zeros(n, dtype=bool)
            mv[block[perm[n_pub:]]] = True
            sp = {t: subset_auc(pk[t], mp) for t in stems}
            sv = {t: subset_auc(pk[t], mv) for t in stems}
            for k, (_, a, bb, _, _) in enumerate(PAIRS):
                DP[b, s, k] = sp[a] - sp[bb]
                DV[b, s, k] = sv[a] - sv[bb]
        print(f"  block {b+1}/{B}  pub mean {DP[b].mean()*1e6:+7.3f}e-6  "
              f"priv mean {DV[b].mean()*1e6:+7.3f}e-6")

    # --- Part 1 (N1): WITHIN-block correlation ------------------------------------------
    print("\n=== Part 1 (N1): WITHIN-block corr(public delta, private delta) ===")
    print("  a fixed block's two halves partition it, so this should sit near -1")
    print(f"  {'pair':16s} {'within corr':>12s} {'between corr':>13s}")
    n1, n2 = {}, {}
    for k, tag in enumerate(tags):
        # within: centre each block, then pool
        wp = DP[:, :, k] - DP[:, :, k].mean(axis=1, keepdims=True)
        wv = DV[:, :, k] - DV[:, :, k].mean(axis=1, keepdims=True)
        cw = float(np.corrcoef(wp.ravel(), wv.ravel())[0, 1])
        # between: block means against each other
        cb = float(np.corrcoef(DP[:, :, k].mean(axis=1), DV[:, :, k].mean(axis=1))[0, 1])
        n1[tag], n2[tag] = cw, cb
        print(f"  {tag:16s} {cw:+12.4f} {cb:+13.4f}")

    # --- Part 3 (N3): the decision number, within block ---------------------------------
    print("\n=== Part 3 (N3): P(h3 wins private | public half reversed) ===")
    print(f"  {'pair':16s} {'pooled':>8s} {'within-blk':>11s} {'per-block min':>14s} "
          f"{'per-block max':>14s} {'blocks<0.5':>11s}")
    n3 = {}
    for k, tag in enumerate(tags):
        rev = DP[:, :, k] < 0
        win = DV[:, :, k] > 0
        pooled = float(win[rev].mean()) if rev.any() else float("nan")
        per = []
        for b in range(B):
            r = rev[b]
            if r.sum() >= 3:
                per.append(float(win[b][r].mean()))
        per = np.array(per) if per else np.array([np.nan])
        nlow = int((per < 0.5).sum())
        n3[tag] = dict(pooled=pooled, per_block_mean=float(np.nanmean(per)),
                       per_block_min=float(np.nanmin(per)), per_block_max=float(np.nanmax(per)),
                       n_blocks_scored=int(len(per)), n_blocks_below_half=nlow,
                       n_rev=int(rev.sum()))
        print(f"  {tag:16s} {pooled:8.3f} {np.nanmean(per):11.3f} "
              f"{np.nanmin(per):14.3f} {np.nanmax(per):14.3f} {nlow:11d}")

    # --- Part 4 (N4): variance decomposition -------------------------------------------
    print("\n=== Part 4 (N4): variance of the PUBLIC delta, between vs within block ===")
    print("  between = 'unlucky test set' (carries into private)")
    print("  within  = 'unlucky split'    (anti-carries into private)")
    print(f"  {'pair':16s} {'sd between':>11s} {'sd within':>10s} {'sd total':>9s} "
          f"{'between share':>14s}")
    n4 = {}
    for k, tag in enumerate(tags):
        bm = DP[:, :, k].mean(axis=1)
        vb = float(bm.var(ddof=1))
        vw = float(DP[:, :, k].var(axis=1, ddof=1).mean())
        # bm carries within-noise/S; remove it for an unbiased between component
        vb_adj = max(vb - vw / S, 0.0)
        share = vb_adj / (vb_adj + vw) if (vb_adj + vw) > 0 else float("nan")
        n4[tag] = dict(var_between=vb_adj, var_within=vw, between_share=float(share))
        print(f"  {tag:16s} {vb_adj**0.5*1e6:11.3f} {vw**0.5*1e6:10.3f} "
              f"{(vb_adj+vw)**0.5*1e6:9.3f} {share:14.3f}")

    shares = np.array([n4[t]["between_share"] for t in tags])
    print(f"\n  between-block share of public-delta variance: "
          f"mean {shares.mean():.3f}  min {shares.min():.3f}  max {shares.max():.3f}")
    verdict = ("S1 — mostly split noise, anti-carries, pick SAFE on this axis"
               if shares.mean() <= 0.25 else
               "S2 — mostly test-set identity, CARRIES into private, pick THREATENED"
               if shares.mean() >= 0.60 else
               "between S1 and S2 — see N3's per-block spread for the decision")
    print(f"  PRE-REGISTERED VERDICT: {verdict}")

    # how often is a block bad enough that h3 genuinely loses private
    print("\n=== Part 5: how bad can a block be? P(h3 loses private) per block ===")
    for k, tag in enumerate(tags):
        if tag not in ("ad187corr", "corr159", "blend159av"):
            continue
        lose = (DV[:, :, k] <= 0).mean(axis=1)
        print(f"  {tag:16s} per-block P(h3 loses private): "
              f"mean {lose.mean():.3f}  max {lose.max():.3f}  "
              f"blocks with P>0.5: {(lose > 0.5).sum()}/{B}")

    out = dict(B=B, S=S, seed=SEED, f=F, n_pub=n_pub, pairs=tags,
               within_corr=n1, between_corr=n2, n3=n3, n4=n4,
               between_share_mean=float(shares.mean()), verdict=verdict)
    with open(os.path.join(HERE, "w22b_blockdecomp.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.join(HERE, "w22b_DP.npy"), DP)
    np.save(os.path.join(HERE, "w22b_DV.npy"), DV)
    print("\nwrote experiments/w22b_blockdecomp.json, w22b_DP.npy, w22b_DV.npy")


if __name__ == "__main__":
    main()
