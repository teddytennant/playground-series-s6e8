"""w46a — price the SELECTION cost of SENDING each unsent file, not its CV.

Pre-registration: experiments/w46_prereg.txt, committed f5a587a BEFORE this file existed.

WHY. JOURNAL w45 section 6 asserted that draining the queue "has non-negative expected
value on the selection axis". That is a two-branch tree and the real one has three. While
NOTHING IS SELECTED, Kaggle auto-picks on best PUBLIC score, so a file we send can DISPLACE
the auto-pick. The queue holds files whose CV is 82-96e-6 below the WANTED file and whose
predicted public score is ABOVE the live top tier -- the logit family, which carries a
+1121e-6 LB-CV gap against h3's +1025e-6. Sending one of those is not free; it is the w13
audit exposure, re-armed.

This does NOT re-price the click. w45a's number is held fixed and is reproduced as a gate.
It prices a different, previously unpriced decision: WHICH ten of the 77 unsent files to send.

MATH. Same LAW-IF scalar map as w18a/w45a, on the file set ENLARGED by one candidate. The
candidate has no observed public score, so the SCENARIO supplies it: conditional on x
rounding to 0.97119 its z = 0.97119 - CV(x), and likewise at 0.97118. Scenario weights come
from w30b's fitted CV->LB predictor with residual sd 7.76e-6.

    delta E[cost of not clicking | we send x]
        = sum_s P(s) * [ cost(auto-pick set under s) - cost(auto-pick set unchanged) ]

both terms inside the SAME posterior, so the information content of the scenario cancels and
what is left is purely the displacement of the auto-pick.

    .venv/bin/python experiments/w46a_sendhazard.py
"""
from __future__ import annotations

import io, json, os, subprocess, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw          # noqa: E402
from w16b_cellweight import fast_auc              # noqa: E402
import stdflag                                    # noqa: E402
from stdflag import family, is_std                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, U = 296_302, 0.20, 1e-6

WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")
TIER1 = ["w36_ad199stdcorr", "w29_ad194stdcorr", "w27_ad190stdcorr", "w21_ad187corr_ens4"]
TIER2 = ["w36_ad199std", "w34_ad195stdcorr", "w27_ad190std", "w21_ad187corr",
         "w22_ad187corr_rankraw"]
BASE = sorted(set(TIER1 + TIER2 + list(WANTED)))

# w45a's four published limit-1 costs, in TIER1 order. Gate G3 must reproduce these.
W45A_PUBLISHED = [0.00, 23.71, 23.88, 39.38]

# --- CV->LB predictor: w46c, the ERA-CORRECTED one -----------------------------------
# ⚠ THIS FILE ORIGINALLY USED w30b DIRECTLY AND ITS FIRST RUN IS PRESERVED AS
# w46a_sendhazard.UNCORRECTED.{csv,json}. w30b over-predicts every ad>=195 file by
# -29.82e-6 (five clean held-out rows, z = -8.6) -- see w46c_predlb.py for the evidence and
# the diagnosis. Under w30b the queue's top file priced at P(clears the tier) = 0.999999;
# corrected it is 0.737. The uncorrected run is kept because the SIZE of that gap is the
# most useful thing this experiment produced.
from w46c_predlb import (predict_lb, scenario_probs, is_corr, new_era,   # noqa: E402
                         ERA_SHIFT, ERA_N, SD_NEW, SD_OLD)


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def emax(mu, sd, rho):
    """E[max(X,Y)] for a bivariate normal (Clark). Exact, no simulation."""
    th = np.sqrt(max(sd[0] ** 2 + sd[1] ** 2 - 2 * rho * sd[0] * sd[1], 1e-300))
    a = (mu[0] - mu[1]) / th
    return mu[0] * norm.cdf(a) + mu[1] * norm.cdf(-a) + th * norm.pdf(a)


def main() -> None:
    # ------------------------------------------------------------------ live board + GATE G1
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree — scores are not deterministic"
    LB = agg["max"].to_dict()
    sent = set(agg.index)
    top = agg["max"].sort_values(ascending=False)
    t1v, t2v = top.iloc[0], sorted(set(top.values))[-2]
    live1, live2 = sorted(top[top == t1v].index), sorted(top[top == t2v].index)
    print(f"GATE G1: auto-slot 1 public {t1v:.5f}: {live1}")
    print(f"         auto-slot 2 public {t2v:.5f}: {live2}")
    if not (live1 == sorted(TIER1) and live2 == sorted(TIER2)):
        print("*** TIERS MOVED SINCE THE w45 PREREG — everything below is void. ***")
        sys.exit(2)
    print("         matches the w45 prereg: PASS")

    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    pi1 = n1 / n
    m, n_pub = N_TEST, int(round(N_TEST * F))

    def oof(k):
        return np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64)

    # ------------------------------------------------------------------ candidate screen
    queue = json.load(open(os.path.join(HERE, "w45b_unsent_cv.json")))
    cand = []
    for r in queue:
        s, c = r["stem"], r["cv"]
        if s in sent:
            continue
        if not os.path.exists(os.path.join(SUB, f"oof_{s}.npy")):
            continue
        p, p19, p18 = scenario_probs(c, s)
        cand.append(dict(stem=s, cv=c, fam=family(s), pred=p, p19=p19, p18=p18))
    priced = [r for r in cand if r["p19"] + r["p18"] > 0.01]
    skipped = [r for r in cand if r["p19"] + r["p18"] <= 0.01]
    n_new = sum(new_era(r["stem"]) for r in cand)
    print(f"\nqueue: {len(cand)} unsent files carry an OOF vector.")
    print(f"  predictor: w46c era-corrected. {n_new} of {len(cand)} are ad>=195 and carry the "
          f"{ERA_SHIFT:+.2f}e-6 shift (n={ERA_N}, predictive sd {SD_NEW:.2f}e-6 vs {SD_OLD:.2f} "
          f"for the {len(cand)-n_new} older-wave files).")
    print(f"  PRICED {len(priced)} with P(reach either live tier) > 0.01")
    print(f"  SKIPPED {len(skipped)} — cannot reach a live tier, delta is 0 by construction.")
    print(f"    (largest skipped P = {max((r['p19']+r['p18'] for r in skipped), default=0):.4f})")

    # ------------------------------------------------------------------ LAW-IF, base rows once
    A0 = np.empty((len(BASE), n1)); B0 = np.empty((len(BASE), n0))
    cv = {}
    for i, k in enumerate(BASE):
        v = oof(k)
        cv[k] = fast_auc(y, v)
        A0[i] = midrank_cdf(np.sort(v[neg]), v[pos])
        B0[i] = 1.0 - midrank_cdf(np.sort(v[pos]), v[neg])

    CORR_CONS = 0.9923594211711959

    def build(names, A, B, zmap):
        """LAW-IF solve on an arbitrary file set. Returns mu, cov, rsd, gap, offI."""
        C1, C0 = np.cov(A), np.cov(B)
        S_t = (1.0 - m / n) * (C1 / (m * pi1) + C0 / (m * (1 - pi1))) / U ** 2
        S_p = (1.0 - n_pub / m) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1))) / U ** 2
        Sx = S_t + S_p
        Mm = S_t @ np.linalg.inv(Sx)
        offI = float(np.abs(Mm - np.eye(len(names)) * np.diag(Mm).mean()).max())
        Kmap = beta * np.eye(len(names)) + (1 - beta) * Mm
        Cpri = (1 - beta) ** 2 * (S_t - Mm @ Sx @ Mm.T)
        Cpri = 0.5 * (Cpri + Cpri.T)
        z = np.array([zmap[k] for k in names])
        D = np.ones((len(names), 1))
        iS = np.linalg.inv(Sx)
        Vg = np.linalg.inv(D.T @ iS @ D)
        gh = Vg @ (D.T @ iS @ z)
        xh = z - D @ gh
        mu = Kmap @ xh
        cov = Cpri + Kmap @ D @ Vg @ D.T @ Kmap.T
        s_pri = abs(beta) * np.sqrt(np.diag(S_p)) / abs(CORR_CONS)
        rsd = s_pri * np.sqrt(max(1 - CORR_CONS ** 2, 0.0))
        return mu, cov, rsd, float(gh[0]), offI

    def coster(names, cvmap, mu, cov, rsd):
        col = {k: i for i, k in enumerate(names)}
        iw = [col[k] for k in WANTED]
        mw = np.array([mu[i] + cvmap[names[i]] / U for i in iw])
        sw = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in iw])
        rw = cov[iw[0], iw[1]] / np.sqrt(cov[iw[0], iw[0]] * cov[iw[1], iw[1]])
        ew = emax(mw, sw, rw)

        def cost(pick):
            i = col[pick]
            return float(ew - (mu[i] + cvmap[pick] / U))
        return cost

    # ------------------------------------------------------------------ GATE G3: reproduce w45a
    z0 = {k: (LB[k] - cv[k]) / U for k in BASE}
    mu0, cov0, rsd0, gap0, offI0 = build(BASE, A0, B0, z0)
    assert offI0 < 1e-6, "GATE G2 failed on the base set — M is not w*I"
    cost0 = coster(BASE, cv, mu0, cov0, rsd0)
    base_l1 = {x: cost0(x) for x in TIER1}
    got = [base_l1[x] for x in TIER1]
    err = max(abs(g - p) for g, p in zip(got, W45A_PUBLISHED))
    print(f"\nGATE G3: base limit-1 costs {np.round(got,2)} vs w45a published "
          f"{W45A_PUBLISHED} — max err {err:.3f}e-6 — "
          f"{'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, "the enlarged solver does not reproduce w45a; nothing below is readable"
    base_uniform = float(np.mean(got))
    base_latest = base_l1["w36_ad199stdcorr"]        # newest member of the tie is the WANTED file
    base_earliest = base_l1["w21_ad187corr_ens4"]    # oldest member of the tie
    print(f"         base cost: uniform {base_uniform:+.2f}  latest-first {base_latest:+.2f}  "
          f"earliest-first {base_earliest:+.2f}  (e-6)")

    # ------------------------------------------------------------------ per-candidate pricing
    out = []
    for j, r in enumerate(priced):
        x = r["stem"]
        v = oof(x)
        cvx = fast_auc(y, v)
        names = BASE + [x]
        A = np.vstack([A0, midrank_cdf(np.sort(v[neg]), v[pos])[None, :]])
        B = np.vstack([B0, (1.0 - midrank_cdf(np.sort(v[pos]), v[neg]))[None, :]])
        cvm = dict(cv); cvm[x] = cvx
        d = {"uniform": 0.0, "latest": 0.0, "earliest": 0.0}
        detail = {}
        for tag, score, p in (("s19", 0.97119, r["p19"]), ("s18", 0.97118, r["p18"])):
            if p < 1e-4:
                continue
            zm = dict(z0); zm[x] = (score - cvm[x]) / U
            mu, cov, rsd, gap, offI = build(names, A, B, zm)
            assert offI < 1e-6, f"GATE G2 failed with {x} added — M is not w*I"
            cost = coster(names, cvm, mu, cov, rsd)
            cb = {t: float(np.mean([cost(k) for k in TIER1])) for t in ("uniform",)}
            cb["latest"] = cost("w36_ad199stdcorr")
            cb["earliest"] = cost("w21_ad187corr_ens4")
            if tag == "s19":
                # x alone holds auto-slot 1 under EVERY tiebreak rule
                new = {t: cost(x) for t in ("uniform", "latest", "earliest")}
            else:
                # x joins the four-way tie: newest member, so latest-first now picks x
                new = {"uniform": float(np.mean([cost(k) for k in TIER1 + [x]])),
                       "latest": cost(x),
                       "earliest": cost("w21_ad187corr_ens4")}
            for t in d:
                d[t] += p * (new[t] - cb[t])
            detail[tag] = dict(p=p, new=new, base_in_posterior=cb, gap=gap)
        out.append(dict(stem=x, cv=cvx, fam=r["fam"], pred=r["pred"], p19=r["p19"],
                        p18=r["p18"], dcv=(cvx - cv["w36_ad199stdcorr"]) / U,
                        d_uniform=d["uniform"], d_latest=d["latest"],
                        d_earliest=d["earliest"], detail=detail))
        print(f"  [{j+1:2d}/{len(priced)}] {x:26s} dCV {out[-1]['dcv']:+7.1f}  "
              f"P19 {r['p19']:.3f} P18 {r['p18']:.3f}  "
              f"dE[cost] uniform {d['uniform']:+8.2f}  latest {d['latest']:+7.2f}  "
              f"earliest {d['earliest']:+6.2f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "detail"} for r in out])
    df = df.sort_values("d_uniform")
    df.to_csv(os.path.join(HERE, "w46a_sendhazard.csv"), index=False)
    pd.set_option("display.width", 220)

    print("\n=== SAFEST TEN by dE[cost] under the uniform tiebreak ===")
    print(df.head(10)[["stem", "fam", "cv", "dcv", "pred", "p19", "p18",
                       "d_uniform", "d_latest"]].to_string(index=False,
                       float_format=lambda v: f"{v:.6f}"))
    print("\n=== MOST HARMFUL TEN ===")
    print(df.tail(10)[["stem", "fam", "cv", "dcv", "pred", "p19", "p18",
                       "d_uniform", "d_latest"]].to_string(index=False,
                       float_format=lambda v: f"{v:.6f}"))

    # ------------------------------------------------------------------ the registered predictions
    print("\n=== the registered predictions (w46_prereg.txt) ===")
    g = df.set_index("stem")["d_uniform"].to_dict()
    logits = ["w36_ad199std_logit", "w38_ad202std_logit", "w40_ad211std_logit"]
    lv = [g.get(k) for k in logits]
    p1 = all(v is not None and v >= 60.0 for v in lv)
    print(f"  P1 the three std_logit files >= +60e-6: {np.round([v if v is not None else np.nan for v in lv],2)}"
          f"  -> {'CONFIRMED' if p1 else '*** FALSIFIED ***'}")
    nharm = int((df.d_uniform > 5.0).sum())
    p2 = nharm >= 5
    print(f"  P2 at least five files price > +5e-6 under uniform: {nharm}"
          f"  -> {'CONFIRMED' if p2 else '*** FALSIFIED ***'}")
    top2 = ["w38_ad202stdcorr", "w40_ad211stdcorr"]
    t2v_ = [g.get(k) for k in top2]
    p3a = all(v is not None and -8.0 < v < 4.0 for v in t2v_)
    p3b = bool((df.d_latest >= -0.01).all())
    p3c = int((df.d_latest > 0.01).sum()) >= 40
    print(f"  P3a w45 s6's top two in (-8,+4): {np.round(t2v_,2)} -> {'yes' if p3a else 'NO'}")
    print(f"  P3b latest-first delta >= 0 for ALL priced files: {p3b}"
          f"   (min {df.d_latest.min():+.3f})")
    print(f"  P3c strictly > 0 for at least 40: {int((df.d_latest>0.01).sum())} -> {'yes' if p3c else 'NO'}")
    print(f"  P3 -> {'CONFIRMED' if (p3a and p3b and p3c) else '*** PARTIAL/FALSIFIED ***'}")

    # P4: the ten w45 s6 named, priced jointly by sequential composition (independent scenarios,
    # each evaluated against the tier as it stands after the previous — approximated by the sum,
    # which is exact to first order and stated as an approximation.)
    S6 = ["w38_ad202stdcorr", "w40_ad211stdcorr", "w36_ad199std_h3", "w40_ad211std_h3",
          "w38_ad202std_h3", "w40_ad211std", "w38_ad202std", "w36_ad197stdcorr",
          "w36_ad199std_h3", "w38_ad202std_hybrid"]
    S6 = list(dict.fromkeys(S6))
    s6d = [g.get(k) for k in S6 if g.get(k) is not None]
    p4 = bool(base_uniform + sum(s6d) < 12.0)
    print(f"  P4 w45 s6's named ten: sum dE[cost] {sum(s6d):+.2f}e-6, "
          f"{base_uniform:+.2f} -> {base_uniform+sum(s6d):+.2f}e-6 (band < +12)"
          f"  -> {'CONFIRMED' if p4 else '*** FALSIFIED ***'}"
          f"   [{len(s6d)}/{len(S6)} of them were priced; the rest could not reach a tier]")
    best10 = list(df.head(10).stem)
    p5 = any(k not in S6 for k in best10)
    print(f"  P5 CV order is not delta order: safest ten contains "
          f"{[k for k in best10 if k not in S6]}  -> {'CONFIRMED' if p5 else '*** FALSIFIED ***'}")

    # ------------------------------------------------------------------ the decision rule
    keep = df[df.d_uniform <= 2.0]
    print(f"\n=== THE SEND LIST for the next UTC day (decision rule fixed in w46_prereg) ===")
    print(f"  eligible (dE[cost] <= +2e-6 under uniform): {len(keep)} of {len(df)} priced "
          f"+ {len(skipped)} unpriced-and-harmless")
    veto = df[df.d_uniform > 2.0]
    print(f"  VETOED ({len(veto)}): " + ", ".join(veto.stem.tolist()))
    order = list(keep.stem) + [r["stem"] for r in sorted(skipped, key=lambda r: -r["cv"])]
    print("  send, in this order:")
    for i, k in enumerate(order[:10], 1):
        dv = g.get(k)
        print(f"    {i:2d}. {k:28s} dE[cost] {('%+.2f' % dv) if dv is not None else '  0.00 (unreachable)'}")

    with open(os.path.join(HERE, "w46a_sendhazard.json"), "w") as f:
        json.dump(dict(tiers=dict(slot1=TIER1, slot2=TIER2), wanted=list(WANTED),
                       base=dict(uniform=base_uniform, latest=base_latest,
                                 earliest=base_earliest, limit1=base_l1),
                       n_priced=len(priced), n_skipped=len(skipped),
                       rows=out, send_order=order[:10],
                       vetoed=veto.stem.tolist(),
                       p1=p1, p2=p2, p3=bool(p3a and p3b and p3c), p4=p4, p5=p5),
                  f, indent=1, default=float)
    print("\nwrote experiments/w46a_sendhazard.csv and .json")


if __name__ == "__main__":
    main()
