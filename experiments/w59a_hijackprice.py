"""w59a — THE HIJACK BREAK-EVEN. The sender's CV bar, measured instead of asserted.

Pre-registration: experiments/w59_prereg.txt, committed a406e6d BEFORE this file existed.

WHY THIS FILE EXISTS.
w58 wired the above-tier gate and set its bar at `min CV over WANTED` (= w23_ad187stdcorr,
0.9701150809, dcv -24.93 vs the pick). The argument, verbatim from `wanted_cv_bar`:

    "A file that can be auto-selected displaces one of our two final entries, so the bar it has
     to clear is the bar the entry it displaces already cleared."

⚠ THAT ARGUMENT DESCRIBES THE WORLD WHERE WE CLICKED. Nothing is selected, so a hijacker does
NOT displace a WANTED file — it displaces A UNIFORM DRAW FROM TIER 1. The bar was set against
the wrong counterfactual, and the right one is `price()` on {X, one tier-1 draw}, which is the
machinery already on disk. w58 refused to give the hijack a point estimate, correctly, because
the hijacker in question (`hboyang_mix`) had no comparable CV — but that refusal was then
carried over onto files that DO have one, and the bar has never been measured.

TWO READINGS.

  UNCONDITIONAL   cost_hijack(X) = mean over d in DRAW(X) of price([X, d]), with X carrying the
                  GLS common gap and no idiosyncratic shift — w58a's treatment of an unsent file.

  CONDITIONAL     the same, but X's (LB - CV) residual is set to E[public | public > tier+STEP/2]
                  — i.e. conditioned on the hijack HAVING HAPPENED, which is the only world in
                  which the question is asked. w57a fits gamma = -0.092: the public-gap to
                  private-gap map is NEGATIVE, so conditioning on a high public score LOWERS the
                  expected private one and the unconditional reading is ANTI-conservative.
                  Registered as P10 with its sign fixed in advance by that mechanism.

DRAW(X) is TIER1 \\ {X}. For a file already IN tier 1, "X hijacks" is the counterfactual that it
landed one reporting step higher, and the tie it leaves behind then holds four files, not five.
The baseline is the true status quo either way: MODEL B over the live 5-file tier.

GATE T / GATE R are w58a's and are not softened: the tier must match what w57a ran on, and the
enlarged NAMES design must reproduce w57a's headline to < 1e-6 on the tier-1 subset.

    .venv/bin/python experiments/w59a_hijackprice.py
"""
from __future__ import annotations

import io, itertools, json, os, subprocess, sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw            # noqa: E402
from w16b_cellweight import fast_auc                # noqa: E402
from w57a_tierprice2 import midrank_cdf, emax       # noqa: E402  — SHARED math, not re-derived

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, U = 296_302, 0.20, 1e-6
STEP = 1e-5                       # the public LB reports to 5 decimals
PRED_SD = 8.77e-6                 # w53a wide branch, the sender's own constant

WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")
PICK = "w36_ad199stdcorr"
CORR_CONS = 0.9923594211711959

W57A_MODEL_B = 7.855203984805849
W57A_UNIFORM_L1 = 17.947859612689353
W58_D = 12.97                     # the dilution break-even, for P4
SENDER_BAR_CV = 0.9701150809      # w26g_send.wanted_cv_bar() as it stands

# The live 08-23 plan, as `w26g_send.py --n 10` dry-plans it this run. P6 is registered against
# it by name.
PLAN_0823 = [
    "w38_ad202std_rescale", "w34_ad195std_h3", "w40_ad211std_rescale", "w36_ad197std",
    "w38_ad202std_hybrid", "w36_ad197std_h3", "w36_ad197stdcorr", "w38_ad202std_h3",
    "w40_ad211std_h3", "w27_ad188stdcorr",
]
P6_NAMED = "w27_ad188stdcorr"
P7_PAIR = ("w38_ad202std_h3", "w40_ad211std_h3")
P8_FILE = "w40_ad211std_h3"


def main() -> None:
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert len(sub) < 500, "hit the page size -- raise it, the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree -- scores are not deterministic"
    LB = agg["max"].to_dict()

    top = agg["max"].sort_values(ascending=False)
    vals = sorted(set(top.values), reverse=True)
    t1v, t2v = vals[0], vals[1]
    TIER1 = sorted(top[top == t1v].index)
    TIER2 = sorted(top[top == t2v].index)

    # ------------------------------------------------------------------------------ GATE T
    prev = json.load(open(os.path.join(HERE, "w57a_tierprice2.json")))["tiers"]
    if (sorted(prev["slot1"]) != TIER1 or sorted(prev["slot2"]) != TIER2
            or abs(prev["slot1_public"] - t1v) > 1e-12):
        print("\n⛔ GATE T: THE TIER HAS MOVED SINCE w57a_tierprice2.json. Re-run w57a first.")
        print(f"   w57a slot1={prev['slot1']} @ {prev['slot1_public']}")
        print(f"   live slot1={TIER1} @ {t1v}")
        sys.exit(2)
    print(f"live board: {len(sub)} scored submissions, {len(agg)} distinct files")
    print(f"  auto-slot 1 public {t1v:.5f}: {len(TIER1)} files -- GATE T PASS (unchanged since w57a)")
    assert PICK in TIER1, "the WANTED pick has fallen OUT of tier 1 -- reprice from scratch"

    # Candidates: every real file with an OOF vector that could stand in for a hijacker. Tier 1
    # and tier 2 give the spread that brackets the break-even; the 08-23 plan is what the answer
    # binds on. No invented dcv -- w58's P6b was the weakest thing in that run for exactly that.
    CAND = list(dict.fromkeys(list(TIER1) + list(TIER2) + PLAN_0823 + list(WANTED)))
    NAMES = sorted(set(CAND))
    missing = [k for k in NAMES if not os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))]
    assert not missing, f"no OOF vector for {missing} -- cannot price them, do not guess"

    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, V[k]) for k in NAMES}

    # ----------------------------------------------------------------------- LAW-IF (w57a's)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.empty((len(NAMES), n1)); B = np.empty((len(NAMES), n0))
    for i, k in enumerate(NAMES):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])
    del V
    C1, C0 = np.cov(A), np.cov(B)
    del A, B
    m, n_pub = N_TEST, int(round(N_TEST * F))
    pi1 = n1 / len(y)
    S_t = (1.0 - m / len(y)) * (C1 / (m * pi1) + C0 / (m * (1 - pi1)))
    S_p = (1.0 - n_pub / m) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1)))
    St, Sp = S_t / U ** 2, S_p / U ** 2
    col = {k: i for i, k in enumerate(NAMES)}
    N = len(NAMES)

    Sxx = St + Sp
    Mm = St @ np.linalg.inv(Sxx)
    Kmap = beta * np.eye(N) + (1 - beta) * Mm
    Cpri = (1 - beta) ** 2 * (St - Mm @ Sxx @ Mm.T)
    Cpri = 0.5 * (Cpri + Cpri.T)

    scored = [k for k in NAMES if k in LB]
    js = [col[k] for k in scored]
    z = np.array([LB[k] - cv[k] for k in scored]) / U
    Ss = Sxx[np.ix_(js, js)]
    D1 = np.ones((len(js), 1))
    iS = np.linalg.inv(Ss)
    Vg = np.linalg.inv(D1.T @ iS @ D1)
    gh = float(np.ravel(Vg @ (D1.T @ iS @ z))[0])
    gamma = beta + (1 - beta) * float(np.diag(Mm).mean())

    xh0 = np.zeros(N)
    for k in NAMES:
        xh0[col[k]] = (LB[k] - cv[k]) / U - gh if k in LB else 0.0
    cov = Cpri + Kmap @ np.ones((N, 1)) @ Vg @ np.ones((1, N)) @ Kmap.T
    rsd = abs(beta) * np.sqrt(np.diag(Sp)) / abs(CORR_CONS) * np.sqrt(max(1 - CORR_CONS ** 2, 0.0))
    print(f"  GLS common gap G = {gh:+.2f}e-6 over {len(scored)} scored files; "
          f"gamma = {gamma:+.6f} (NEGATIVE -> P10's sign)")

    iw = [col[k] for k in WANTED]

    def make_price(mu):
        def emax_set(files):
            ia = [col[f] for f in files]
            ma = np.array([mu[i] + cv[NAMES[i]] / U for i in ia])
            if len(ia) == 1:
                return float(ma[0])
            sa = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in ia])
            ra = cov[ia[0], ia[1]] / np.sqrt(cov[ia[0], ia[0]] * cov[ia[1], ia[1]])
            return float(emax(ma, sa, ra))

        def price(files):
            mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
            sw = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in iw])
            rw = cov[iw[0], iw[1]] / np.sqrt(cov[iw[0], iw[0]] * cov[iw[1], iw[1]])
            return float(emax(mw, sw, rw) - emax_set(files))
        return price

    price0 = make_price(Kmap @ xh0)

    # ------------------------------------------------------------------------------ GATE R
    base = float(np.mean([price0(list(p)) for p in itertools.combinations(TIER1, 2)]))
    uni1 = float(np.mean([price0([x]) for x in TIER1]))
    dR, dU = abs(base - W57A_MODEL_B), abs(uni1 - W57A_UNIFORM_L1)
    print(f"\n  GATE R: MODEL B on tier 1 = {base:.12f} vs w57a {W57A_MODEL_B:.12f} (|d| {dR:.2e})")
    print(f"          uniform limit-1    = {uni1:.12f} vs w57a {W57A_UNIFORM_L1:.12f} (|d| {dU:.2e})")
    if dR > 1e-6 or dU > 1e-6:
        print("⛔ GATE R FAILED: the enlarged design moved w57a's headline. Refusing.")
        sys.exit(3)
    print("          -- PASS")

    # ------------------------------------------------------------- the conditional shift
    # E[public | public > c] under N(m, s). m is the MODEL's own prediction for the file --
    # cv + the common gap -- so the conditioning is internally consistent with the LAW-IF fit
    # rather than borrowed from a different pricer. Reported at both sds.
    c_thresh = t1v + STEP / 2

    def cond_xh(k, s):
        mpred = cv[k] + gh * U
        zc = (c_thresh - mpred) / s
        lam = norm.pdf(zc) / max(norm.sf(zc), 1e-300)
        return (s * lam) / U            # e-6 units, on top of the common gap

    def price_cond(k, s):
        xh = xh0.copy()
        xh[col[k]] = cond_xh(k, s)
        return make_price(Kmap @ xh)

    def draws(x):
        """The tie the hijacker leaves behind: tier 1 minus itself."""
        return [d for d in TIER1 if d != x]

    def cost_hijack(x, price):
        return float(np.mean([price([x, d]) for d in draws(x)]))

    # --------------------------------------------------------------------- THE MEASUREMENT
    print("\n=== HIJACK: X lands ABOVE the tier, auto pair = {X, one draw from the tie} ===")
    print(f"  status quo (no hijack, MODEL B over the live {len(TIER1)}-file tier): {base:+.3f}e-6")
    print(f"  worthless-hijacker limit (w58a's uniform_limit1):                    {uni1:+.3f}e-6")
    print(f"\n  {'candidate':24s} {'dCV':>8s} {'uncond':>9s} {'cond':>9s} {'cond8.77':>9s} "
          f"{'verdict':>7s}  where")
    rows = []
    for x in sorted(CAND, key=lambda k: -cv[k]):
        s_model = float(np.sqrt(Sxx[col[x], col[x]])) * U
        cu = cost_hijack(x, price0)
        cc = cost_hijack(x, price_cond(x, s_model))
        cf = cost_hijack(x, price_cond(x, PRED_SD))
        where = ("tier1" if x in TIER1 else "tier2" if x in TIER2 else "") + \
                (" plan" if x in PLAN_0823 else "")
        rows.append(dict(stem=x, dcv=(cv[x] - cv[PICK]) / U, cv=cv[x], uncond=cu, cond=cc,
                         cond_predsd=cf, s_model_e6=s_model / U,
                         helps_uncond=bool(cu < base), helps_cond=bool(cc < base),
                         where=where.strip() or "-"))
        print(f"  {x:24s} {rows[-1]['dcv']:+8.2f} {cu:+9.3f} {cc:+9.3f} {cf:+9.3f} "
              f"{'HELPS' if cu < base else 'HURTS':>7s}  {where.strip()}")

    # -------------------------------------------------------- GATE I: HIJACK ≡ DILUTION
    # NOT PLANNED. Fell out of the table: w58's dilution deltas and these hijack costs cross
    # `base` between the SAME two files at the SAME fraction. That is not a coincidence, it is
    # an identity. Adding X to a 5-file tier takes MODEL B from a mean over C(5,2)=10 pairs to a
    # mean over C(6,2)=15, and the 5 NEW pairs are exactly {X,d} for d in TIER1 -- which is the
    # average `cost_hijack(X)` already is. So
    #
    #     delta_dilution(X) = (10*base + 5*cost_hijack(X))/15 - base = (cost_hijack(X) - base)/3
    #
    # identically, for every X outside the tier. The two break-evens are THE SAME NUMBER, and an
    # above-tier landing carries exactly 3x the leverage of an in-tier one, in either direction.
    # Asserted numerically so the algebra is a check and not a paragraph.
    print("\n=== GATE I: delta_dilution(X) == (cost_hijack(X) - base)/3 for X outside tier 1 ===")
    worst = 0.0
    for r in rows:
        x = r["stem"]
        if x in TIER1:
            continue
        nb = float(np.mean([price0(list(p))
                            for p in itertools.combinations(sorted(list(TIER1) + [x]), 2)]))
        worst = max(worst, abs((nb - base) - (r["uncond"] - base) / 3.0))
    print(f"  max |delta_dilution - (cost_hijack-base)/3| over {len(rows)-len(TIER1)} files "
          f"= {worst:.3e}e-6")
    if worst > 1e-9:
        print("⛔ GATE I FAILED: the identity does not hold, so D and H are NOT the same number")
        print("   and every claim below that they collapse is wrong. Refusing.")
        sys.exit(5)
    print("  -- PASS. w58's DILUTION break-even D and this run's HIJACK break-even H are the")
    print("     SAME QUANTITY. One bar governs both landings; the hijack just has 3x the lever.")

    sm = float(np.mean([r["s_model_e6"] for r in rows]))
    print(f"\n  model-internal predictive sd for the truncation: mean {sm:.2f}e-6 "
          f"(the sender's PRED_SD is {PRED_SD/U:.2f}e-6; the `cond8.77` column uses that)")

    # ⚠⚠ WITHDRAWN AND REBUILT WITHIN THIS RUN. The first cut of `breakeven` took the FIRST row
    # below `base` and the LAST row at-or-above it, i.e. it interpolated across the WHOLE 40e-6
    # range instead of between the two files that actually straddle the crossing. That is w58's
    # P6b failure in a new costume -- a number produced by extrapolation dressed as a bracket.
    # It read H = 17.79 where the ADJACENT pair reads 12.97. Local brackets only, and every
    # further crossing is reported rather than silently swallowed by a global fit.
    def crossings(key):
        rs = sorted(rows, key=lambda r: -r["dcv"])
        out = []
        for a, b in zip(rs, rs[1:]):
            if (a[key] < base) != (b[key] < base) and a[key] != b[key]:
                t = (base - a[key]) / (b[key] - a[key])
                out.append((-(a["dcv"] + t * (b["dcv"] - a["dcv"])), a, b))
        return out

    def breakeven(key):
        """dcv where cost first crosses `base`, walking DOWN in CV from the pick."""
        cr = crossings(key)
        return cr[0] if cr else (None, None, None)

    H_u, hu_hi, hu_lo = breakeven("uncond")
    H_c, hc_hi, hc_lo = breakeven("cond_predsd")
    H_deg, _, _ = breakeven("cond")
    print("\n=== THE BREAK-EVEN (adjacent brackets only) ===")
    for nm, Hv, hi, lo in (("UNCONDITIONAL", H_u, hu_hi, hu_lo),
                           ("CONDITIONAL@8.77", H_c, hc_hi, hc_lo)):
        if Hv is None:
            print(f"  {nm}: no adjacent straddle in the candidate set -- NOT REPORTED")
            continue
        print(f"  {nm:16s} H = {Hv:6.2f}e-6  -> HELPS iff CV >= {cv[PICK] - Hv*U:.10f}"
              f"   [bracket {hi['stem']} {hi['dcv']:+.2f} .. {lo['stem']} {lo['dcv']:+.2f}]")
    for key, nm in (("uncond", "UNCONDITIONAL"), ("cond_predsd", "CONDITIONAL@8.77")):
        extra = crossings(key)[1:]
        if extra:
            print(f"  ⚠ {nm} has {len(extra)} FURTHER crossing(s) further down the CV ladder: "
                  + ", ".join(f"{h:.2f}e-6 ({a['stem']}..{b['stem']})" for h, a, b in extra)
                  + " -- the cost curve is not globally monotone, see the tier-1 rows.")
    # ⚠ THE `cond` COLUMN IS DEGENERATE AND IS NOT USED AS A BAR. Its truncation sd is the
    # model-internal predictive sd of the (LB-CV) residual, ~557e-6, which is 55x the 10e-6
    # distance to the threshold: the truncated-normal shift then saturates and EVERY file
    # collapses onto the worthless-hijacker limit. Reported so the failure is on the record.
    print(f"  ⚠ the model-sd conditional column saturates at the worthless limit "
          f"({uni1:.2f}e-6) and is NOT used; H would read {H_deg if H_deg is None else round(H_deg,2)}.")

    H_bind = min(h for h in (H_u, H_c) if h is not None) if (H_u or H_c) else None
    print(f"\n  BINDING (the stricter of the two): H = {H_bind:.2f}e-6, "
          f"CV bar {cv[PICK] - H_bind*U:.10f}")
    print(f"  the sender's CURRENT bar: {SENDER_BAR_CV:.10f}  "
          f"(dcv {(SENDER_BAR_CV - cv[PICK])/U:+.2f}e-6)")

    # -------------------------------------------------------------------------- WHAT BINDS
    print("\n=== the live 08-23 ten under the two bars ===")
    bar_new = cv[PICK] - H_bind * U
    flipped = []
    for i, s in enumerate(PLAN_0823, 1):
        if s not in cv:
            print(f"  {i:2d}. {s:24s} (no OOF loaded)")
            continue
        old_ok, new_ok = cv[s] >= SENDER_BAR_CV, cv[s] >= bar_new
        tag = "same" if old_ok == new_ok else "⛔ NEWLY BLOCKED"
        if old_ok and not new_ok:
            flipped.append(s)
        print(f"  {i:2d}. {s:24s} cv {cv[s]:.10f} dcv {(cv[s]-cv[PICK])/U:+7.2f}  "
              f"old {'pass' if old_ok else 'fail'}  new {'pass' if new_ok else 'fail'}  {tag}")

    # ------------------------------------------------------------------------------ P7 / P8
    pair_cost = price0(list(P7_PAIR)) if all(p in cv for p in P7_PAIR) else None
    p8_cost = next((r["uncond"] for r in rows if r["stem"] == P8_FILE), None)
    print("\n=== the lever ===")
    if pair_cost is not None:
        print(f"  P7 both of {P7_PAIR[0]} + {P7_PAIR[1]} above the tier -> {pair_cost:+.3f}e-6 "
              f"(status quo {base:+.3f})")
    if p8_cost is not None:
        print(f"  P8 {P8_FILE} alone above the tier                      -> {p8_cost:+.3f}e-6")

    # ---------------------------------------------------------------- REGISTERED PREDICTIONS
    print("\n=== the registered predictions (w59_prereg.txt) ===")
    dcvs = np.array([r["dcv"] for r in rows]); cus = np.array([r["uncond"] for r in rows])
    sp = float(pd.Series(cus).corr(pd.Series(dcvs), method="spearman"))
    p1 = bool(sp <= -0.95)
    p2 = bool(H_u is not None and 10.0 <= H_u <= 22.0)
    p3 = bool(H_u is not None and H_u < (cv[PICK] - SENDER_BAR_CV) / U)
    # P4 is now decided by GATE I's identity, not by a reading: H IS D. Registered as H > D,
    # so a tie FALSIFIES it, and it is recorded as falsified-by-theorem rather than by noise.
    p4 = bool(H_u is not None and H_u > W58_D + 1e-9)
    pick_cost = next(r["uncond"] for r in rows if r["stem"] == PICK)
    p5 = bool(-4.0 <= pick_cost < 0.0)
    p6 = bool(P6_NAMED in flipped)
    p7 = bool(pair_cost is not None and pair_cost < 4.0)
    p8 = bool(p8_cost is not None and p8_cost < base)
    p10 = bool(H_c is not None and H_u is not None and 0.0 < (H_u - H_c) <= 12.0
               and 4.0 <= H_c <= 20.0)   # H_c is the PRED_SD conditional; the model-sd one is degenerate
    R = [("P1  Spearman(cost,dcv) <= -0.95", p1, f"{sp:+.4f}"),
         ("P2  H_uncond in [10,22]e-6 (pt 16.4)", p2, f"{H_u:.2f}" if H_u else "n/a"),
         ("P3  H < 24.93 -> the sender's bar is TOO LOOSE", p3,
          f"{H_u:.2f} vs {(cv[PICK]-SENDER_BAR_CV)/U:.2f}" if H_u else "n/a"),
         ("P4  H > D = 12.97  (GATE I says H IS D)", p4, f"{H_u:.2f} vs D {W58_D}" if H_u else "n/a"),
         ("P5  cost_hijack(PICK) in [-4,0)", p5, f"{pick_cost:+.3f}"),
         (f"P6  {P6_NAMED} flips to BLOCKED", p6, f"{len(flipped)} flipped: {flipped}"),
         ("P7  best eligible pair < 4.0e-6", p7, f"{pair_cost:+.3f}" if pair_cost else "n/a"),
         ("P8  best eligible single < base", p8, f"{p8_cost:+.3f}" if p8_cost else "n/a"),
         ("P10 0 < H_u - H_c <= 12 and H_c in [4,20]", p10,
          f"H_u {H_u:.2f} H_c {H_c:.2f}" if (H_u and H_c) else "n/a")]
    for nm, ok, val in R:
        print(f"  {'✅ CONFIRMED ' if ok else '❌ FALSIFIED '} {nm:46s} {val}")
    print("  (P9 is checked by re-running w26g_send.py --n 10 dry AFTER the bar is wired.)")

    out = dict(tiers=dict(slot1=TIER1, slot2=TIER2, slot1_public=float(t1v),
                          slot2_public=float(t2v)),
               gate_t="PASS", gate_r=dict(model_b=base, uniform_l1=uni1, w57a=W57A_MODEL_B),
               beta=beta, gamma=gamma, gap=gh, base=base, uniform_limit1=uni1,
               pick=PICK, pick_cv=cv[PICK], threshold_public=c_thresh,
               rows=rows, H_uncond=H_u, H_cond_predsd=H_c, H_cond_modelsd_degenerate=H_deg,
               H_binding=H_bind, identity_gate_max_abs=worst,
               cv_bar_new=(cv[PICK] - H_bind * U) if H_bind else None,
               cv_bar_old=SENDER_BAR_CV,
               plan_0823=PLAN_0823, newly_blocked=flipped,
               pair_cost=pair_cost, p8_cost=p8_cost, spearman=sp,
               predictions=dict(p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7, p8=p8, p10=p10))
    with open(os.path.join(HERE, "w59a_hijackprice.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote experiments/w59a_hijackprice.json")


if __name__ == "__main__":
    main()
