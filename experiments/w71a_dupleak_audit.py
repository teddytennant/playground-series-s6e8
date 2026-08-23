"""w71a — AUDIT OF THE w70f DUPLICATE OVERRIDE. Its premise fails; the bet is NEGATIVE-EV.

w70f (JOURNAL w70 §8.2) found the 2 train<->test exact feature matches, called their labels
"READ, not inferred", priced the override at **+1.62e-6**, and called it "free, deterministic,
cannot overfit -- public and private move together". §10.5 item 2 then made APPLYING it the
highest-value unspent item on the account.

⛔ THE PREMISE DOES NOT SURVIVE. Three findings, in order of how much they cost:

1. "Both matched train groups are PURE (single label)" -- **both groups have size 1.**
   A group of size 1 is pure by definition. That sentence carries ZERO evidence and reads as
   though it carries the whole argument. It is the load-bearing claim and it is vacuous.

2. Labels are STOCHASTIC given features, measured here, not assumed. Within-train near-duplicate
   pairs (matching on k of 12 features, both labels known) agree at:
        k=8  1179 pairs  0.599     k=10  13 pairs  0.462
        k=9    35 pairs  0.543     k=11   5 pairs  0.400
   A copy mechanism drives agreement toward **1.0** as k rises. It does the opposite. k=8's
   excess over the OOF-implied null (z=+6.86) is model imperfection on shared covariates -- it
   vanishes (z=+0.63,-0.27,-0.58) exactly where copying would show up. Pooled k>=9: 27/53=0.509
   against a null of ~0.50. Fitting agreement = f*1 + (1-f)*0.50 gives a copy fraction
   **f = 0.018, 95% upper bound f <= 0.28.**

3. The collision-count ladder never FLATTENS. Exact-match pair counts by k over train+test:
        k=8 2482 -> k=9 82 -> k=10 20 -> k=11 7 -> k=12 2,  a steady ~3x decay per feature.
   A population of copied rows is a FLOOR: adding features cannot separate copies, so the curve
   would flatten at the copy count. There is no flat. 2 is the tail of a decaying coincidence
   process, not a floor. (Independence would predict ~1e-11 collisions, but the ladder shows
   dependence inflates that by ~2e11x, so the independence argument for "must be copies" -- which
   w70f never actually ran -- is worthless. The ladder is the measurement that matters.)

⛔ AND THE PAYOFF IS SHARPLY ASYMMETRIC, WHICH w70f NEVER COMPUTED. It priced only the branch
where the label transfers. Setting p to a hard 0/1 on a row the model puts at the 33rd percentile
is a large bet: right, it gains a little; wrong, it drags a row the whole width of the
distribution. This script computes both branches and the break-even transfer probability.
"""
import numpy as np, pandas as pd, json, sys
from sklearn.isotonic import IsotonicRegression

TEST = "data/test.csv"
REF  = "submissions/w36_ad199stdcorr_ens4.csv"        # the file w70f itself priced against
OOF  = "submissions/oof_w36_ad199stdcorr_ens4.npy"
IDS  = {735378: 1, 862871: 0}                          # id -> label of the matched TRAIN row

# ⚠ BOTH the submission and the OOF are RANK-SCALED (mean 0.5000), NOT probabilities -- the base
# rate is 0.7094. Reading the file's value as P(y=1) understates the positive count by 30% and
# silently corrupts every AUC delta. Calibrate rank -> probability on TRAIN with isotonic first.


def evidence():
    """The two measurements that decide WHICH branch we are in. Both run from the raw CSVs."""
    tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
    feats = [c for c in tr.columns if c not in ("id", "addicted_label")]
    allx = pd.concat([tr[feats], te[feats]], ignore_index=True)
    n_tr = len(tr); y = tr["addicted_label"].to_numpy()
    oof = np.load(OOF)
    sp = {c: (allx[c].value_counts(normalize=True).values ** 2).sum() for c in feats}
    order = sorted(feats, key=lambda c: -sp[c])
    codes = {c: pd.factorize(allx[c])[0].astype(np.int64) for c in feats}

    key = np.zeros(len(allx), dtype=np.int64); pairs = {}; agree = {}
    for k, c in enumerate(order, 1):
        key = key * (codes[c].max() + 1) + codes[c]
        _, key = np.unique(key, return_inverse=True); key = key.astype(np.int64)
        cnt = np.bincount(key); pairs[k] = float((cnt * (cnt - 1) / 2).sum())
        if k >= 8:                                   # within-TRAIN label agreement
            ktr = key[:n_tr]; o = np.argsort(ktr, kind="stable")
            ks, ys, pz = ktr[o], y[o], oof[o]
            st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]); sz = np.diff(np.r_[st, len(ks)])
            st, sz = st[sz >= 2], sz[sz >= 2]
            np_ = ob = 0; ex = 0.0
            for a, z in zip(st, sz):
                lab, pr = ys[a:a + z], pz[a:a + z]
                for i in range(z):
                    for j in range(i + 1, z):
                        np_ += 1; ob += int(lab[i] == lab[j])
                        ex += pr[i] * pr[j] + (1 - pr[i]) * (1 - pr[j])
            if np_:
                agree[k] = (np_, ob, ex)

    print("  EVIDENCE 1 -- within-TRAIN near-duplicate label agreement (both labels known).")
    print("  A COPY mechanism drives agreement toward 1.0 as k rises. It does the opposite.")
    print(f"    {'k':>3} {'pairs':>7} {'agree':>7} {'null':>7}")
    for k in sorted(agree):
        np_, ob, ex = agree[k]
        print(f"    {k:>3} {np_:>7d} {ob/np_:>7.3f} {ex/np_:>7.3f}")
    tot_p = sum(agree[k][0] for k in agree if k >= 9); tot_o = sum(agree[k][1] for k in agree if k >= 9)
    f_hat = max(0.0, (tot_o / tot_p - 0.5) / 0.5)
    print(f"    pooled k>=9: {tot_o}/{tot_p} = {tot_o/tot_p:.3f}  ->  copy fraction f = {f_hat:.3f}")

    print("\n  EVIDENCE 2 -- the collision ladder never FLATTENS. A copy population is a FLOOR.")
    print(f"    {'k':>3} {'pairs':>10} {'ratio':>8}")
    for k in range(8, 13):
        print(f"    {k:>3} {pairs[k]:>10.4g} {pairs[k]/pairs[k-1]:>8.3f}")
    gm = float(np.exp(np.mean(np.log([pairs[10]/pairs[9], pairs[11]/pairs[10]]))))
    pred = pairs[11] * gm
    print(f"    decay(k=10,11) {gm:.3f} -> predicts {pred:.2f} pairs at k=12; OBSERVED {pairs[12]:.0f}")
    print(f"    Poisson P(>=2 | {pred:.2f}) = {1-np.exp(-pred)*(1+pred):.3f}  "
          f"-- the tail explains it; no copy floor is needed.\n")
    return dict(ladder={k: pairs[k] for k in range(8, 13)},
                agreement={k: dict(pairs=agree[k][0], obs=agree[k][1], null=agree[k][2]) for k in agree},
                f_hat=f_hat, k12_predicted=pred, k12_observed=pairs[12])


def main():
    te  = pd.read_csv(TEST, usecols=["id"])
    sub = pd.read_csv(REF)
    y   = pd.read_csv("data/train.csv", usecols=["addicted_label"])["addicted_label"].to_numpy()
    oof = np.load(OOF)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)

    s   = sub.set_index("id").reindex(te["id"])["addicted_label"].to_numpy(float)   # rank scale
    ids = te["id"].to_numpy()
    pi  = iso.predict(s)                                   # calibrated P(y=1) per test row
    n   = len(s)

    o = np.argsort(s, kind="stable"); ss = s[o]; pis = pi[o]
    cum_neg = np.concatenate([[0.0], np.cumsum(1.0 - pis)])   # expected negatives strictly below rank r
    cum_pos = np.concatenate([[0.0], np.cumsum(pis)])
    P, N = pi.sum(), (1.0 - pi).sum()

    def neg_below(v):  return float(cum_neg[np.searchsorted(ss, v, side="left")])
    def pos_below(v):  return float(cum_pos[np.searchsorted(ss, v, side="left")])

    print("="*92); print("w71a  AUDIT OF THE w70f DUPLICATE OVERRIDE"); print("="*92)
    print()
    EV = evidence()
    print(f"\n  reference {REF}  (rank-scaled; calibrated with isotonic on the OOF)")
    print(f"  base rate {y.mean():.6f}   expected positives {P:,.0f}   negatives {N:,.0f}   rows {n:,}\n")

    tot_t = tot_n = 0.0; rows = []
    for tid, lab in IDS.items():
        i  = int(np.flatnonzero(ids == tid)[0])
        s0 = float(s[i]); q = float(pi[i])                 # q = model's calibrated P(y=1)
        s1 = 1.0 if lab == 1 else 0.0
        # AUC delta given the row's TRUE label, when its score moves s0 -> s1
        d_if_pos = (neg_below(s1) - neg_below(s0)) / (P*N)
        d_if_neg = ((P - pos_below(s1)) - (P - pos_below(s0))) / (P*N)
        transfer = d_if_pos if lab == 1 else d_if_neg      # the branch w70f priced: label is right
        pright   = q if lab == 1 else 1.0 - q
        no_tr    = q*d_if_pos + (1.0 - q)*d_if_neg         # iid redraw: label carries no info
        tot_t += transfer; tot_n += no_tr
        rows.append(dict(id=tid, matched_label=lab, rank=s0, calibrated_p=q,
                         auc_if_transfers=transfer, auc_if_not=no_tr,
                         d_if_pos=d_if_pos, d_if_neg=d_if_neg, p_guess_right=pright))
        print(f"  id {tid}  matched train label {lab}   rank {s0:.4f} @ {(s<s0).mean()*100:5.2f}%   "
              f"calibrated p {q:.4f}")
        print(f"      row really 1 -> {d_if_pos*1e6:+8.3f}e-6     row really 0 -> {d_if_neg*1e6:+8.3f}e-6")
        print(f"      if the label TRANSFERS      : {transfer*1e6:+8.3f}e-6   <- all w70f priced")
        print(f"      if it does NOT (iid redraw) : {no_tr*1e6:+8.3f}e-6   "
              f"(P(guess right) = {pright:.3f})")

    print(f"\n  TOTAL if labels transfer     : {tot_t*1e6:+8.3f}e-6   (w70f headline +1.622e-6)")
    print(f"  TOTAL if they do NOT         : {tot_n*1e6:+8.3f}e-6")
    F_HI = 0.28                                            # 95% upper bound on the copy fraction
    F_PT = 0.018                                           # point estimate
    be = (0.0 - tot_n) / (tot_t - tot_n) if tot_t != tot_n else float("nan")
    print(f"\n  BREAK-EVEN P(labels transfer) = {be*100:.1f}%")
    print(f"  MEASURED copy fraction        = {F_PT*100:.1f}%   (95% upper bound {F_HI*100:.0f}%)")
    print(f"  EV at the measured rate       = {(F_PT*tot_t + (1-F_PT)*tot_n)*1e6:+.3f}e-6")
    print(f"  EV even at the 95% UPPER      = {(F_HI*tot_t + (1-F_HI)*tot_n)*1e6:+.3f}e-6")
    ev_hi = F_HI*tot_t + (1-F_HI)*tot_n
    verdict = "DO NOT APPLY" if ev_hi < 0 else ("APPLY" if F_PT*tot_t + (1-F_PT)*tot_n > 0 else "MARGINAL")
    print(f"\n  VERDICT: {verdict}")
    json.dump(dict(evidence=EV, rows=rows, total_if_transfer=tot_t, total_if_not=tot_n, breakeven=be,
                   copy_fraction=F_PT, copy_fraction_hi=F_HI, ev_at_hi=ev_hi, verdict=verdict),
              open("experiments/w71a_dupleak_audit.json","w"), indent=2)
    print("  wrote experiments/w71a_dupleak_audit.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
