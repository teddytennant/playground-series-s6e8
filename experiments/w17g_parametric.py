"""w17g — the same conditioning as w17d, done parametrically, because w17d's GLOBAL branch is
ESS-starved and my own pre-registration said to disbelieve it.

w17d registered P5 as "P(the auto file beats the CV pick privately) stays strictly inside
0.30..0.50 ... A number below 0.30 would be this workspace claiming a resolution the 237,042-row
private slice cannot deliver and should be disbelieved as hard as one above 0.50."  It returned
0.104 / 0.000 / 0.010 on a GLOBAL branch whose ESS is 14.3 out of 6,000 draws. The script's own
guard printed the warning. So P5 is falsified and the numbers that falsified it are not to be
believed either — the honest conclusion is that the exact non-parametric conditioning cannot be
afforded at 6,000 draws, not that the auto-pick is privately dead at P 0.00.

WHY A PARAMETRIC VERSION IS NOW LEGITIMATE, and was not before this run
-----------------------------------------------------------------------
w17d's P2 and P3 both CONFIRMED, and they are exactly the two facts the parametric form needs:
  P2  the coupling of a contrast's private deviation to its public deviation, at fixed
      pseudo-test, is linear with slope -0.2477 (range -0.2487..-0.2464) and |corr| 0.992-0.997.
      AUC is not additive over a partition, so this had to be measured; it was, and it sits on
      the exact-partition -f/(1-f) = -0.2500.
  P3  the variance split w = sigma_t^2/(sigma_t^2+sigma_p^2) is 0.1238, against 0.125 predicted
      from row counts alone.
With a joint that linear and that tight, the conditional expectation is a regression coefficient,
and a regression coefficient estimated on 6,000 draws costs no ESS at all.

    E[d_private | d_public = p, dCV = c]  =  c + gamma * (p - c),
    gamma = (sigma_t^2 + beta*sigma_p^2) / (sigma_t^2 + sigma_p^2)

Every one of sigma_t, sigma_p, beta is already stored per pair in w17d_coupling.json, so this
needs no new draws. Sanity: at beta = -1/4 and w = 1/8, gamma = 1.25w - 0.25 = -0.0938, i.e. a
file whose public score is inflated by 10e-6 relative to its CV is expected to give back about
1e-6 of that privately, and the CV-based contrast is the right centre.

WHAT THIS BUYS AND WHAT IT DOES NOT. The parametric form conditions each contrast on ITS OWN
public difference only. w17d's non-parametric form conditions on all six files' public scores
jointly, which is strictly more information — so where the two disagree by more than the ESS-14
noise, the non-parametric one is using something real and the difference is worth a later slot
with more draws. Neither can move WANTED, which is chosen on CV.

    .venv/bin/python experiments/w17g_parametric.py
"""
from __future__ import annotations

import io
import itertools
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
F = 0.20
WANTED = ("w16i_schemeavg", "blend159av_h3")
AUTO1 = ("w16e_aonly", "w16q_ens4avg", "w16t_cellens4")


def main() -> None:
    j = json.load(open(os.path.join(HERE, "w17d_coupling.json")))
    assert j["gate_w16w"] and j["p2"] and j["p3"], "w17d's gate or P2/P3 did not hold"
    cv, lb = j["cv"], j["lb"]
    print(f"w17d: {j['reps']} draws, gate PASSED, P2 beta {j['coupling_beta_median']:+.4f}, "
          f"P3 w {j['w_median']:.4f}")
    print(f"      GLOBAL ESS {j['ess']['GLOBAL']:.1f}  PERFAM ESS {j['ess']['PERFAM']:.1f}  "
          f"<- why this file exists")

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    live = pd.read_csv(io.StringIO(raw))
    live["stem"] = live["fileName"].str.replace(r"\.csv$", "", regex=True)
    lb_live = live.groupby("stem")["publicScore"].max().to_dict()
    for k, v in lb.items():
        assert abs(lb_live[k] - v) < 1e-12, f"{k}: live LB moved since w17d ran"
    print(f"      live LB re-checked against w17d's for all {len(lb)} files: unchanged "
          f"({len(live)} submissions on the page)")

    def pair(x, z):
        key = f"{x}|{z}" if f"{x}|{z}" in j["varsplit"] else f"{z}|{x}"
        sgn = 1.0 if f"{x}|{z}" in j["varsplit"] else -1.0
        v, c = j["varsplit"][key], j["coupling"][key]
        st, sp, b = v["sigma_t"], v["sigma_p"], c["beta"]
        gamma = (st ** 2 + b * sp ** 2) / (st ** 2 + sp ** 2)
        # conditional sd of the private contrast given the public one
        s_pri = np.hypot(st, abs(b) * sp)
        s_pub = np.hypot(st, sp)
        rho = gamma * s_pub / s_pri
        return sgn, gamma, s_pri * np.sqrt(max(1.0 - rho ** 2, 0.0)), st, sp

    print(f"\n=== the private contrasts, conditioned on the OBSERVED public difference ===")
    print("  E[d_private] = dCV + gamma*(dLB - dCV).  positive = the AUTO file wins privately")
    print(f"{'auto':16s} {'vs CV pick':16s} {'dCV':>7s} {'dLB':>7s} {'gamma':>7s} "
          f"{'E[dpriv]':>9s} {'shift':>7s} {'sd':>6s} {'P(auto)':>8s}")
    rows = []
    for x in AUTO1:
        for z in WANTED:
            _, gamma, sd, st, sp = pair(x, z)
            c, p = cv[x] - cv[z], lb[x] - lb[z]
            e = c + gamma * (p - c)
            pw = float(1.0 - norm.cdf(0.0, loc=e, scale=sd))
            rows.append(dict(auto=x, wanted=z, dcv=c, dlb=p, gamma=gamma, e_cond=e,
                             shift=e - c, sd_cond=sd, p_auto=pw))
            print(f"  {x:14s} {z:16s} {c*1e6:+6.2f} {p*1e6:+6.1f} {gamma:+7.4f} "
                  f"{e*1e6:+8.3f} {(e-c)*1e6:+6.2f} {sd*1e6:5.2f} {pw:8.3f}")
    R = pd.DataFrame(rows)

    print(f"\n=== the click at limit 1, three ways ===")
    print(f"  {'auto file':16s} {'w16w (no cond)':>15s} {'w17d GLOBAL':>12s} "
          f"{'w17d PERFAM':>12s} {'w17g parametric':>16s}")
    out = {"gamma_median": float(R["gamma"].median()), "limit1": {}, "contrasts": R.to_dict("records")}
    for x in AUTO1:
        # against the CV pick, which is the file WANTED would actually be scored on
        r = R[(R["auto"] == x) & (R["wanted"] == "w16i_schemeavg")].iloc[0]
        para = -r["e_cond"]
        d = j["limit1"][x]
        out["limit1"][x] = dict(none=d["NONE"], glob=d["GLOBAL"], perfam=d["PERFAM"],
                                parametric=float(para), p_auto=float(r["p_auto"]))
        print(f"  {x:16s} {d['NONE']*1e6:+14.3f} {d['GLOBAL']*1e6:+11.3f} "
              f"{d['PERFAM']*1e6:+11.3f} {para*1e6:+15.3f}")
    for tag, key in (("no conditioning", "none"), ("w17d GLOBAL (ESS 14)", "glob"),
                     ("w17d PERFAM (ESS 100)", "perfam"), ("w17g parametric", "parametric")):
        v = [out["limit1"][x][key] for x in AUTO1]
        print(f"  range, {tag:24s} {min(v)*1e6:+7.3f}e-6 .. {max(v)*1e6:+7.3f}e-6")

    print(f"\n=== P5 RE-READ against the parametric numbers ===")
    ps = [out["limit1"][x]["p_auto"] for x in AUTO1]
    print(f"  P(auto file beats the CV pick privately): {[round(p, 3) for p in ps]}")
    print(f"  registered band 0.30..0.50; w17d's ESS-14 GLOBAL gave [0.104, 0.000, 0.010]")
    ok5 = all(0.30 <= p <= 0.50 for p in ps)
    print(f"  P5 under the parametric conditioning: {'inside' if ok5 else 'OUTSIDE'} the band")
    print("  Either way P5 stands FALSIFIED — it was registered against w17d's GLOBAL branch and")
    print("  that branch returned what it returned. This is the size of the miss, not a rescue.")

    out["p_auto"] = {x: out["limit1"][x]["p_auto"] for x in AUTO1}
    json.dump(out, open(os.path.join(HERE, "w17g_parametric.json"), "w"), indent=1, sort_keys=True)
    R.to_csv(os.path.join(HERE, "w17g_contrasts.csv"), index=False)
    print("\nwrote experiments/w17g_parametric.json, w17g_contrasts.csv")


if __name__ == "__main__":
    main()
