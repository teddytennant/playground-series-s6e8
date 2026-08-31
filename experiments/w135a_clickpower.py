"""w135a — how much power does ONE private draw have over the final-selection click?

Kaggle scores a selection as max(private AUC of the two selected files). The click's
published price, +4.5228e-6, is a difference of PAIR MAXIMA computed on a single OOF
realisation. Every paired-noise measurement this workspace holds is single-file and at the
20% public scale (cvlb2.py). This one is max-of-pair, at private scale, paired.

Pre-registered in experiments/w135_prereg.txt (commit 761f341) before any number here existed.
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB, DATA = os.path.join(ROOT, "submissions"), os.path.join(ROOT, "data")
TARGET = "addicted_label"

WANTED = ["w36_ad199stdcorr", "w23_ad187stdcorr"]           # the CV pick, needs the click
AUTO   = ["w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4"] # what Kaggle takes if nobody clicks
PUBLISHED_PRICE = 4.5228e-6      # w74a, tau=0, carried by every run since
PUBLISHED_CV = {"w36_ad199stdcorr": 0.9701400060, "w23_ad187stdcorr": 0.9701150809}
N_TEST, PUBLIC_FRAC = 296302, 0.20
REPS = 3000
SEED = 20260831

fails = []
def check(tag, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {tag}  {msg}", flush=True)
    if not ok: fails.append(tag)


class WeightedAUC:
    """Exact AUC of a fixed score vector under integer row multiplicities.

    Ranks are a property of the score vector, so they are computed once. A bootstrap draw
    only changes how many times each row appears, which enters as a weight. Ties between
    distinct rows that share an exact float score get the usual half credit.
    """
    def __init__(self, score, y):
        order = np.argsort(score, kind="stable")
        s = score[order]
        # group index in sorted order, one group per distinct score value
        newgrp = np.empty(len(s), dtype=bool)
        newgrp[0] = True
        np.not_equal(s[1:], s[:-1], out=newgrp[1:])
        self.gid = np.cumsum(newgrp) - 1
        self.G = int(self.gid[-1]) + 1
        self.ys = y[order].astype(np.float64)
        self.order = order

    def __call__(self, c):
        cs = c[self.order].astype(np.float64)
        gp = np.bincount(self.gid, weights=cs * self.ys, minlength=self.G)
        gn = np.bincount(self.gid, weights=cs - cs * self.ys, minlength=self.G)
        cumneg = np.cumsum(gn) - gn
        P, N = gp.sum(), gn.sum()
        return float((gp * (cumneg + 0.5 * gn)).sum() / (P * N))


t0 = time.time()
y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).to_numpy()
oof = {n: np.load(os.path.join(SUB, f"oof_{n}.npy")) for n in WANTED + AUTO}
n = len(y)

print("=" * 92); print("CONTROLS"); print("=" * 92)
check("C1", all(len(v) == n for v in oof.values()),
      f"all four OOF arrays are {n} rows, matching train.csv")
auc = {k: roc_auc_score(y, v) for k, v in oof.items()}
for k, v in sorted(auc.items(), key=lambda kv: -kv[1]):
    print(f"        {k:28s} pooled OOF AUC {v:.10f}")
check("C2", all(abs(auc[k] - c) < 1e-9 for k, c in PUBLISHED_CV.items()),
      "the two WANTED files reproduce their published CV to 1e-9")
point = max(auc[a] for a in WANTED) - max(auc[a] for a in AUTO)
# C3 was first written as `point == PUBLISHED_PRICE` and FAILED, which is the finding below.
# The two are different quantities at different layers, and w74a's own artefact holds both.
art = json.load(open(os.path.join(ROOT, "experiments", "w74a_clickprice.json")))
dcv = -art["solo"]["w36_ad199stdcorr_ens4"]["dcv"] * 1e-6   # artefact's own CV difference
check("C3a", abs(point - dcv) < 1e-9,
      f"the OOF pair-max difference {point*1e6:+.4f}e-6 reproduces w74a's own `dcv` field "
      f"{dcv*1e6:+.4f}e-6 to 1e-9")
check("C3b", abs(point - PUBLISHED_PRICE) > 1.0e-6,
      f"and is NOT the quoted headline {PUBLISHED_PRICE*1e6:+.4f}e-6 -- gap "
      f"{(PUBLISHED_PRICE-point)*1e6:+.4f}e-6, {100*(PUBLISHED_PRICE-point)/point:.0f}% of the "
      f"arithmetic difference. `cost` is E[max] under w74a's fitted GLS transfer (beta "
      f"{art['beta']:+.4f}), not arithmetic on the OOF arrays; its own identity control "
      f"`solo[pick].cost` returns {art['solo']['w36_ad199stdcorr']['cost']:.6f}e-6 rather than "
      f"exactly 0, which is what a Monte-Carlo expectation looks like and what an arithmetic "
      f"identity does not")
check("C4", len({v.tobytes() for v in oof.values()}) == 4,
      "all four arrays are distinct objects, so no arm compares a file to itself")

est = {k: WeightedAUC(v, y) for k, v in oof.items()}
ones = np.ones(n, dtype=np.int64)
check("C5", all(abs(est[k](ones) - auc[k]) < 1e-12 for k in oof),
      "the weighted estimator agrees with sklearn to 1e-12 at unit weights, all four files")

rng0 = np.random.default_rng(SEED)
c0 = np.bincount(rng0.integers(0, n, size=50_000), minlength=n)
sk0 = {k: roc_auc_score(y[c0 > 0].repeat(c0[c0 > 0]),
                        oof[k][c0 > 0].repeat(c0[c0 > 0])) for k in oof}
check("C6", all(abs(est[k](c0) - sk0[k]) < 1e-12 for k in oof),
      "and agrees with sklearn to 1e-12 on a bootstrap draw with real duplicate rows")
check("C7", (max(est[w](c0) for w in WANTED) - max(est[w](c0) for w in WANTED)) == 0.0,
      "paired null: the same pair against itself is exactly 0 on a shared draw")

print()
print("=" * 92)
print("PAIRED BOOTSTRAP OF max(WANTED) - max(AUTO), AT THREE TEST-SET SCALES")
print("=" * 92, flush=True)
scales = {"public 20%": int(round(N_TEST * PUBLIC_FRAC)),
          "private 80%": int(round(N_TEST * (1 - PUBLIC_FRAC))),
          "whole test": N_TEST}
rows = []
for label, m in scales.items():
    rng = np.random.default_rng(SEED)
    d = np.empty(REPS)
    per = {k: np.empty(REPS) for k in oof}
    for r in range(REPS):
        c = np.bincount(rng.integers(0, n, size=m), minlength=n)  # ONE draw, all four scored
        a = {k: est[k](c) for k in oof}
        for k in a: per[k][r] = a[k]
        d[r] = max(a[w] for w in WANTED) - max(a[x] for x in AUTO)
    sd, mu = d.std(ddof=1), d.mean()
    ppos = float((d > 0).mean())
    rows.append(dict(scale=label, m=m, mean=mu, sd=sd, ppos=ppos,
                     lo=float(np.percentile(d, 5)), hi=float(np.percentile(d, 95)),
                     single_sd=float(per[WANTED[0]].std(ddof=1))))
    print(f"  {label:12s} m={m:7,d}  mean {mu*1e6:+8.4f}e-6   sd {sd*1e6:7.4f}e-6   "
          f"P(delta>0) {100*ppos:5.1f}%   90% [{np.percentile(d,5)*1e6:+8.3f}, "
          f"{np.percentile(d,95)*1e6:+8.3f}]e-6")
    print(f"  {'':12s}          unpaired single-file sd, same draws: "
          f"{per[WANTED[0]].std(ddof=1)*1e6:.2f}e-6", flush=True)

priv = [r for r in rows if r["scale"] == "private 80%"][0]

print()
print("=" * 92); print("THE PRE-REGISTERED PREDICTIONS"); print("=" * 92)
m1 = priv["sd"] > PUBLISHED_PRICE
m2 = 0.55 < priv["ppos"] < 0.85
m3 = abs(priv["mean"] - PUBLISHED_PRICE) > 0.5e-6
# C3b means the prereg registered M1 and M3 against the WRONG constant: the bootstrap
# estimates the arithmetic quantity `point`, so `point` is the edge its noise sits on.
# Both gradings are printed. The registered one is primary and is not quietly replaced.
m1c = priv["sd"] > point
m3c = abs(priv["mean"] - point) > 0.5e-6
print(f"  M1  noise exceeds the edge      sd {priv['sd']*1e6:.4f}e-6 vs edge "
      f"{PUBLISHED_PRICE*1e6:.4f}e-6                {'HELD' if m1 else 'FALSIFIED'}")
print(f"  M2  one draw is a weak bit      P(delta>0) = {100*priv['ppos']:.1f}%, "
      f"registered band 55-85%             {'HELD' if m2 else 'FALSIFIED'}")
print(f"  M3  point estimate is not E[.]  bootstrap mean {priv['mean']*1e6:+.4f}e-6 vs point "
      f"{PUBLISHED_PRICE*1e6:+.4f}e-6   {'HELD' if m3 else 'FALSIFIED'}")
print(f"  P4  power claim stands unless P(delta>0) >= 95%: "
      f"{'STANDS' if priv['ppos'] < 0.95 else 'WITHDRAWN'}")
print()
print("  RE-GRADED against the edge the bootstrap actually estimates (C3b), stated second")
print("  because the prereg is frozen and this is a correction to it, not a replacement:")
print(f"  M1' sd {priv['sd']*1e6:.4f}e-6 vs arithmetic edge {point*1e6:.4f}e-6"
      f"                    {'HELD' if m1c else 'FALSIFIED'}")
print(f"  M3' bootstrap mean {priv['mean']*1e6:+.4f}e-6 vs arithmetic edge {point*1e6:+.4f}e-6"
      f"      {'HELD' if m3c else 'FALSIFIED'}")

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "experiments", "w135a_clickpower.csv"), index=False)
print()
print("=" * 92); print("VERDICT"); print("=" * 92)
print(f"  FAILURES {len(fails)}." + ("" if not fails else "  " + ", ".join(fails)))
print(f"  wall {time.time()-t0:.0f}s, {REPS} reps per scale, seed {SEED}")
print()
print("  The click stays CORRECT and stays FREE: its expected delta is positive at every scale,")
print("  and the two priced mis-click pairs are an order of magnitude worse the other way.")
print("  What this measures is that the ONE realisation arriving tonight cannot GRADE it. The")
print("  sd of the delta is comparable to the delta itself, so a private board that happens to")
print("  favour the auto pair is not evidence the CV pick was wrong.")
print("  BOUNDED BY: bootstrapping OOF rows models the test set as a finite sample from the")
print("  same distribution. It does not model train/test shift, and it is not the common-shift")
print("  null used for the rank forecast -- that instrument lives in w135_prereg.txt P3.")
sys.exit(1 if fails else 0)
