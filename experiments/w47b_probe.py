"""w47b — BREAK the era / CV-region confound that w46 and w47_prereg both called unresolvable.

THE CONFOUND. w46c attributes a -29.82e-6 miss on the five 08-21 files to an `ad>=195`
ERA effect. w47a killed the three obvious alternatives (extrapolation, prospective
optimism, day-clustering) but could not kill the fourth, because it is not identified in
anything this account has sent: every ad>=195 file ever SENT also sits above the top of
the fitted CV range. So "past ad194 an import stops converting" (ERA) and "above
cv 0.970118 the CV->LB relation saturates" (CV-REGION) make identical predictions on
every observation in existence.

    w46 section 2:  "The scope is total ... there is not one file in the queue that the
                     uncorrected predictor prices correctly."
    w47_prereg:     "Nothing on this disk can fully separate them, and no send can
                     either, because there is no low-CV ad>=195 file ... to send."

BOTH OF THOSE ARE WRONG, and this file is the correction. The unsent queue holds
THIRTEEN non-logit ad>=195 files sitting 5.9 to 20.5e-6 BELOW the fitted CV maximum --
inside the support, in the era. They are era files at ordinary CV. They separate it.

    ERA true       -> a probe lands ~ -29.8e-6 under the w30b prediction.
    CV-REGION true -> a probe lands on the w30b prediction, residual ~ 0.

WHY THIS IS FREE. w46a prices the SELECTION hazard of a send as 0 by construction for any
file with P(reach a live tier) <= 0.01. Under the era-corrected predictor every probe is
<= 0.004. That is circular -- it assumes the hypothesis under test -- so every probe below
is screened under BOTH predictors and only those with P <= 0.01 under the *generous* w30b
survive. Eight do.

    .venv/bin/python experiments/w47b_probe.py
"""
from __future__ import annotations

import json, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB                             # noqa: E402
import w46c_predlb as W                            # noqa: E402

FIT_MAX = 0.9701183          # top of the CV range w30b was fitted on (78 files)
SD_FILE = 8.36               # w47a leave-one-out held-out residual sd, e-6
SEP = abs(W.ERA_SHIFT)       # 29.82e-6 separation between the two hypotheses
TIER_CAP = 0.01              # w46a's "cannot reach a live tier" threshold


def wave(s):
    m = re.search(r"ad(\d{3})", s)
    return int(m.group(1)) if m else None


def p_tier(pred, sd):
    return float((1 - norm.cdf((0.971185 - pred) / sd))
                 + (norm.cdf((0.971185 - pred) / sd) - norm.cdf((0.971175 - pred) / sd)))


q = json.load(open(os.path.join(HERE, "w45b_unsent_cv.json")))
rows = []
for r in q:
    s, c = r["stem"], r["cv"]
    w = wave(s)
    if w is None or w < W.ERA_MIN_AD or c >= FIT_MAX or "logit" in s:
        continue                       # logit carries a +147e-6 family term and is vetoed
    if not os.path.exists(os.path.join(SUB, f"{s}.csv")):
        continue
    p30, p46 = W._raw(c, s), W.predict_lb(c, s)
    rows.append(dict(stem=s, cv=c, ad=w, fam=W.family(s), d_supp=(c - FIT_MAX) * 1e6,
                     pred_w30b=p30, pred_w46c=p46,
                     P_w30b=p_tier(p30, W.SD_OLD * 1e-6),
                     P_w46c=p_tier(p46, W.SD_NEW * 1e-6)))
P = pd.DataFrame(rows).sort_values("cv", ascending=False).reset_index(drop=True)

print("=" * 84)
print("CANDIDATE PROBES -- ad>=195 files sitting INSIDE the fitted CV support")
print("=" * 84)
print(f"  {'stem':24s} {'cv':>13s} {'d_supp':>7s} {'fam':>8s} {'w30b':>9s} {'P30b':>7s} "
      f"{'w46c':>9s} {'P46c':>7s}")
for _, x in P.iterrows():
    ok = "FREE" if x.P_w30b <= TIER_CAP else "hazard"
    print(f"  {x.stem:24s} {x.cv:.10f} {x.d_supp:+7.1f} {x.fam:>8s} {x.pred_w30b:.6f} "
          f"{x.P_w30b:7.4f} {x.pred_w46c:.6f} {x.P_w46c:7.4f}  {ok}")
FREE = P[P.P_w30b <= TIER_CAP].reset_index(drop=True)
print(f"\n  {len(FREE)} of {len(P)} are free under the GENEROUS w30b screen "
      f"(max P {FREE.P_w30b.max():.4f}); those are the only ones eligible.")
print(f"  The {len(P)-len(FREE)} rejected are rejected on the hypothesis-under-test's OWN")
print("  worst case, which is the only screen that is not circular here.")

print("\n" + "=" * 84)
print("POWER -- how many probes buy a decision?")
print("=" * 84)
print(f"  per-file held-out sd {SD_FILE:.2f}e-6 (w47a LOO); hypotheses {SEP:.2f}e-6 apart")
print(f"  {'n':>3s} {'sem':>6s} {'z if ERA':>9s} {'z if CV-REGION':>15s} {'P(correct call)':>16s}")
for n in range(1, len(FREE) + 1):
    sem = SD_FILE / np.sqrt(n)
    pc = float(norm.cdf((SEP / 2) / sem))
    print(f"  {n:3d} {sem:6.2f} {-SEP/sem:9.2f} {0.0:15.2f} {pc:16.4f}")
N_PROBE = 5
sem5 = SD_FILE / np.sqrt(N_PROBE)
print(f"\n  {N_PROBE} probes -> sem {sem5:.2f}e-6, the two hypotheses {SEP/sem5:.1f} sems apart,")
print(f"  P(the midpoint rule calls it right) = {norm.cdf((SEP/2)/sem5):.4f}. Five is enough.")

print("\n" + "=" * 84)
print("THE AMENDED 08-22 TEN, and what it costs")
print("=" * 84)
TEN = ["w38_ad202stdcorr", "w40_ad211stdcorr", "w40_ad211std", "w38_ad202std",
       "w36_ad199std_h3", "w36_ad197stdcorr", "w40_ad211std_h3", "w38_ad202std_h3",
       "w40_ad211std_rescale", "w38_ad202std_rescale"]
haz = pd.read_csv(os.path.join(HERE, "w46a_sendhazard.csv")).set_index("stem")
keep, drop = TEN[:5], TEN[5:]
print("  KEEP (w46d order 1-5, they carry essentially all of the drain's value):")
for s in keep:
    print(f"    {s:24s} dE[cost] {haz.loc[s,'d_uniform']:+7.2f}e-6")
print("  DROP (w46d order 6-10):")
for s in drop:
    print(f"    {s:24s} dE[cost] {haz.loc[s,'d_uniform']:+7.2f}e-6")
print(f"\n  w46b's joint: sending NOTHING +21.74e-6, the top ONE +7.56, all TEN +7.09.")
print(f"  So all nine free riders are worth {7.56-7.09:+.2f}e-6 BETWEEN THEM, and the five")
print(f"  dropped here are a fraction of that. Cost of the swap: under +0.3e-6, and the")
print(f"  probes themselves are 0 by construction under both predictors.")

probes = FREE.head(N_PROBE)
print("\n  ADD (the five probes, highest CV first -- highest CV is closest to the support")
print("  edge and therefore the most conservative test of the CV-REGION story):")
for _, x in probes.iterrows():
    print(f"    {x.stem:24s} cv {x.cv:.10f}  {x.d_supp:+6.1f}e-6 inside  ad{x.ad}  "
          f"w30b {x.pred_w30b:.6f} / w46c {x.pred_w46c:.6f}")

print("\n" + "=" * 84)
print("SEND ORDER (latest-first tiebreak: best CV LAST, w46b section 5)")
print("=" * 84)
order = list(probes.stem)[::-1] + keep[::-1]
for i, s in enumerate(order, 1):
    if s in keep:
        cv = haz.loc[s, "cv"]; kind = "DRAIN"; p30 = W._raw(cv, s); p46 = W.predict_lb(cv, s)
    else:
        x = probes[probes.stem == s].iloc[0]
        cv = x.cv; kind = "PROBE"; p30, p46 = x.pred_w30b, x.pred_w46c
    print(f"  {i:2d}  {kind}  {s:24s} cv {cv:.10f}   H0/w30b {p30:.6f}   H1/w46c {p46:.6f}")

print("\n  E4 SLOPE TEST, the reason the probes also improve the OTHER read:")
scored = np.array([(c - FIT_MAX) * 1e6 for _, c, _ in W.ERA_ROWS])
drain = np.array([(haz.loc[s, "cv"] - FIT_MAX) * 1e6 for s in keep])
prob = probes.d_supp.values
for lbl, arr in (("w46d's ten, as registered",
                  np.concatenate([scored, [(haz.loc[s, "cv"] - FIT_MAX) * 1e6 for s in TEN]])),
                 ("this amendment (5 drain + 5 probes)",
                  np.concatenate([scored, drain, prob]))):
    se = SD_FILE / np.sqrt(((arr - arr.mean()) ** 2).sum())
    print(f"    {lbl:36s} n {len(arr):2d}  span {arr.max()-arr.min():5.1f}e-6  "
          f"se(slope) {se:.3f}  t for a true -0.86 slope {-0.86/se:+5.2f}")

json.dump(dict(candidates=P.to_dict("records"), free=list(FREE.stem),
               probes=list(probes.stem), keep=keep, drop=drop, order=order,
               n_probe=N_PROBE, sd_file=SD_FILE, sep=float(SEP),
               p_correct=float(norm.cdf((SEP / 2) / sem5))),
          open(os.path.join(HERE, "w47b_probe.json"), "w"), indent=1)
P.to_csv(os.path.join(HERE, "w47b_candidates.csv"), index=False)
print("\nwrote w47b_probe.json, w47b_candidates.csv")
