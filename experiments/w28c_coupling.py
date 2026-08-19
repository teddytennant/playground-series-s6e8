"""w28c — does each shipped file's TEST csv come from the same run as its OOF vector?

Every CV number this workspace quotes is computed from `submissions/oof_<stem>.npy`, and every
decision is made on those numbers -- but the object that actually scores is `<stem>.csv`, which
is a SEPARATE array written by the same process. Nothing has ever checked that the two belong
together. If a rebuild ever overwrote one and not the other, the CV would be right, the file
would be wrong, and no gate in this workspace would notice.

The check that works without refitting anything: a `*corr` file is its h3 base plus a small
additive c_avg correction, so on the OOF side `corr - base` is a specific, tiny, structured
vector. If the CSV came from the same run, the TEST side must show the same relation -- same
sign pattern, same order of magnitude, near-perfect rank agreement with the base. A CSV from a
different run would still correlate highly with the base (it is 99% the same model) but the
DELTA's scale and the residual rank distance would not match the OOF side.

    OMP_NUM_THREADS=4 .venv/bin/python experiments/w28c_coupling.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submissions")

# (corrected file, the h3 base it was built from). From the build scripts, not the names.
PAIRS = [("w29_ad194stdcorr", "w29_ad194std_h3"),
         ("w27_ad190stdcorr", "w27_ad190std_h3"),
         ("w27_ad188stdcorr", "w27_ad188std_h3"),
         ("w23_ad187stdcorr", "w23_ad187std_h3")]

y = pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["addicted_label"]).addicted_label.values


def load(stem):
    oof = np.load(os.path.join(SUB, f"oof_{stem}.npy"))
    te = pd.read_csv(os.path.join(SUB, f"{stem}.csv")).sort_values("id").addicted_label.values
    return oof, te


print(f"{'file':22s} {'base':20s} {'cv corr':>12s} {'cv base':>12s} {'d_cv':>8s} "
      f"{'oof rho':>9s} {'test rho':>9s} {'oof |d| p99':>12s} {'test |d| p99':>13s} {'ratio':>7s}")
rows = []
for stem, base in PAIRS:
    o1, t1 = load(stem)
    o0, t0 = load(base)
    cv1, cv0 = roc_auc_score(y, o1), roc_auc_score(y, o0)
    # both sides on the RANK scale, so the two deltas are directly comparable
    ro1, ro0 = pd.Series(o1).rank(pct=True).values, pd.Series(o0).rank(pct=True).values
    rt1, rt0 = pd.Series(t1).rank(pct=True).values, pd.Series(t0).rank(pct=True).values
    do, dt = ro1 - ro0, rt1 - rt0
    rho_o = spearmanr(o1, o0).statistic
    rho_t = spearmanr(t1, t0).statistic
    p99o, p99t = np.percentile(np.abs(do), 99), np.percentile(np.abs(dt), 99)
    rows.append(dict(stem=stem, base=base, cv=cv1, cv_base=cv0, d_cv=(cv1 - cv0) * 1e6,
                     rho_oof=rho_o, rho_test=rho_t, p99_oof=p99o, p99_test=p99t,
                     ratio=p99t / p99o))
    print(f"{stem:22s} {base:20s} {cv1:12.10f} {cv0:12.10f} {(cv1-cv0)*1e6:+8.2f} "
          f"{rho_o:9.6f} {rho_t:9.6f} {p99o:12.3e} {p99t:13.3e} {p99t/p99o:7.3f}")

t = pd.DataFrame(rows)
print("\nREAD: `ratio` is the test-side correction scale over the OOF-side correction scale.")
print("The correction is fitted on train and APPLIED to test by the same rule, so a matched")
print("pair should land near 1. A CSV written by a different run than its OOF would not have")
print("to -- it is the one quantity that couples the two files.")
lo, hi = t.ratio.min(), t.ratio.max()
print(f"\nratio range over {len(t)} pairs: {lo:.3f} .. {hi:.3f}")
print("VERDICT: " + ("all pairs COUPLED -- csv and oof agree" if 0.5 < lo and hi < 2.0
                     else "⚠ AT LEAST ONE PAIR IS SUSPECT -- inspect before selecting it"))

# Cross-check the other direction: the three corr files should rank-agree with each other far
# more than any of them agrees with a different family, on BOTH sides. Same relation, both.
print("\ncross-file rank agreement (test side), the three corr files against each other:")
te = {s: pd.read_csv(os.path.join(SUB, f"{s}.csv")).sort_values("id").addicted_label.values
      for s, _ in PAIRS}
oo = {s: np.load(os.path.join(SUB, f"oof_{s}.npy")) for s, _ in PAIRS}
names = [s for s, _ in PAIRS]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        print(f"  {a:20s} vs {b:20s}  test rho {spearmanr(te[a], te[b]).statistic:.6f}   "
              f"oof rho {spearmanr(oo[a], oo[b]).statistic:.6f}")

t.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w28c_coupling.csv"), index=False)
print("\nwrote experiments/w28c_coupling.csv")

# ─────────────────────────────────────────────────────────────────────────────────────────
# SLOT-2 DIVERSITY. The pair {slot1, slot2} exists to maximise E[max] over the private slice,
# so the hedge is only worth its CV cost if it is actually DIFFERENT from slot 1. The three
# corr files agree at test rho 0.99993+, which is essentially the same submission twice. This
# table gives the next slot the numbers to price a genuinely orthogonal second pick instead of
# re-deriving them: rank agreement with slot 1 on the TEST side (the thing that gets scored),
# against the CV each candidate gives up.
SLOT1 = "w27_ad190stdcorr"
CANDS = ["w29_ad194stdcorr", "w27_ad188stdcorr", "w23_ad187stdcorr", "w29_ad194std_h3",
         "w27_ad190std_h3", "w27_ad190std",
         "w27_ad188raw_h3", "w21_ad187corr_ens4", "w22_ad187corr_rankraw",
         "w27_ad190std_rankraw", "w27_ad190std_logit", "w16i_schemeavg"]
s1_te = pd.read_csv(os.path.join(SUB, f"{SLOT1}.csv")).sort_values("id").addicted_label.values
s1_cv = roc_auc_score(y, np.load(os.path.join(SUB, f"oof_{SLOT1}.npy")))
print(f"\n=== SLOT-2 CANDIDATES against slot 1 = {SLOT1} (CV {s1_cv:.10f}) ===")
print(f"{'candidate':24s} {'cv':>14s} {'d_cv vs slot1':>14s} {'test rho':>10s} {'sent':>6s}")
sent_now = set(pd.read_csv("/tmp/subs300.csv").fileName) if os.path.exists("/tmp/subs300.csv") else set()
div = []
for c in CANDS:
    op = os.path.join(SUB, f"oof_{c}.npy")
    if not os.path.exists(op):
        continue
    cv = roc_auc_score(y, np.load(op))
    te_c = pd.read_csv(os.path.join(SUB, f"{c}.csv")).sort_values("id").addicted_label.values
    rho = spearmanr(s1_te, te_c).statistic
    div.append(dict(cand=c, cv=cv, d_cv=(cv - s1_cv) * 1e6, rho=rho))
    print(f"{c:24s} {cv:14.10f} {(cv-s1_cv)*1e6:+14.2f} {rho:10.6f} "
          f"{'yes' if c + '.csv' in sent_now else 'NO':>6s}")
pd.DataFrame(div).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "w28c_slot2.csv"), index=False)
print("wrote experiments/w28c_slot2.csv")
