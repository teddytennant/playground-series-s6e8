"""w58a — THE AUTO-SELECTION TIER RULE, PRICED AND MADE BINDING.

w54 wrote the rule: *"a file is SAFE to send iff its predicted public score is < the tier."*
w55 found it was enforced by nothing, and fixed **the NaN branch only** — `w26g_send.py` now
blocks a row with no `pred_lb`. A row WITH a `pred_lb` that sits ABOVE the tier still walks
straight through. Fifth bug of this shape on the send path, and this one has a live victim:

    08-23 slot 1   w48_cal_hboyang_mix.csv   pred_lb 0.971230   tier 0.97118

`w48d_arm217.json` registers that file's whole predictive band at [0.9711976, 0.9712523] —
**entirely above the tier.** With `check_selection` exiting 1, Kaggle auto-selects the best two
by PUBLIC score, so under its own registered point prediction that send makes an aggregator's
raw member vector our FINAL ENTRY #1 — the same file `check_selection.WANTED_INELIGIBLE` bars
from CV-based selection, and whose "CV" is a member OOF AUC that RESEARCH forbids comparing
against a cross-fitted stack CV.

TWO QUESTIONS, ONE INSTRUMENT.

  (A) THE HIJACK. What does it cost if a file lands ABOVE the tier and takes auto-slot 1 alone?
      For `hboyang_mix` this is deliberately NOT given a point estimate: its quality is the
      quantity the 08-23 read exists to measure, so pricing it would assume the answer. What is
      given is the BOUND, which is enough to decide: in the limit where the file is worthless
      the auto pair degenerates to a single uniform draw from the old tier 1, i.e. exactly
      w57a's `uniform_limit1`; in the limit where it is honest it dominates and the cost is
      <= 0. So the exposure spans [<=0, uniform_limit1], and w56 measures the bad end as modal.

  (B) THE DILUTION. What does it cost if a file lands IN the tier as an extra member? This has
      a real answer and it is the send policy for the remaining 27-slot gap. Adding a member
      near the pick's CV dilutes the expensive pick-missing pairs (that is why the 08-22 sends
      came back net favourable); adding a laggard manufactures new ones. The break-even CV
      distance D is measured here, on REAL files, not on a hypothetical dcv: every tier-2 file
      is a counterfactual tier-1 member that missed by one reporting step and already has an
      OOF vector loaded.

GATE R — THE RE-IMPLEMENTATION IS PROVED FAITHFUL, NOT ASSUMED. The LAW-IF block below is
w57a's, evaluated on a LARGER file set (the candidates are appended to NAMES), which changes
the GLS common-gap fit and every covariance entry. So the script recomputes w57a's headline
MODEL-B mean on the tier-1-only subset and requires it to reproduce 7.855203984805849. If the
enlarged design moved that number, every figure here would be on a different scale from the one
it claims to supersede, and the script refuses to run.

GATE T — the live tier is re-derived from the board and compared against what w57a recorded it
ran on. A hard-coded tier is what went stale in w45a; a tier that has MOVED since w57a means
w57a's own numbers are stale too and must not be quoted alongside these.

    .venv/bin/python experiments/w58a_tiergate.py
"""
from __future__ import annotations

import io, itertools, json, os, re, subprocess, sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw            # noqa: E402
from w16b_cellweight import fast_auc                # noqa: E402
from stdflag import family                          # noqa: E402
from w57a_tierprice2 import midrank_cdf, emax       # noqa: E402  — the math is SHARED, not re-derived

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, U = 296_302, 0.20, 1e-6

WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")
PICK = "w36_ad199stdcorr"
CORR_CONS = 0.9923594211711959

# w57a's published headline, which GATE R requires this larger design to reproduce exactly.
W57A_MODEL_B = 7.855203984805849
W57A_UNIFORM_L1 = 17.947859612689353

# The 08-23 plan as `w26g_send.py --n 10` dry-plans it (verified this run). Slot 1 is handled
# separately as the HIJACK case; the other nine are the DILUTION candidates.
PLAN_0823 = [
    "w48_cal_hboyang_mix",      # slot 1 — THE ARM 217 TEST, pred_lb 0.971230, ABOVE the tier
    "w38_ad202std_rescale", "w34_ad195std_h3", "w40_ad211std_rescale", "w36_ad197std",
    "w38_ad202std_hybrid", "w36_ad197std_h3", "w36_ad197stdcorr", "w38_ad202std_h3",
    "w40_ad211std_h3",
]
HIJACK = PLAN_0823[0]


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
    moved = (sorted(prev["slot1"]) != TIER1 or sorted(prev["slot2"]) != TIER2
             or abs(prev["slot1_public"] - t1v) > 1e-12)
    print(f"live board: {len(sub)} scored submissions, {len(agg)} distinct files")
    print(f"  auto-slot 1 public {t1v:.5f}: {len(TIER1)} files")
    print(f"  auto-slot 2 public {t2v:.5f}: {len(TIER2)} files")
    if moved:
        print("\n⛔ GATE T: THE TIER HAS MOVED SINCE w57a_tierprice2.json.")
        print(f"   w57a ran on slot1={prev['slot1']} @ {prev['slot1_public']}")
        print(f"   live         slot1={TIER1} @ {t1v}")
        print("   Re-run w57a_tierprice2.py FIRST. Its MODEL-B mean is GATE R's reference and")
        print("   every figure below is quoted against it.")
        sys.exit(2)
    print(f"  GATE T: tier unchanged since w57a ({len(TIER1)} in slot 1) -- PASS")
    assert PICK in TIER1, "the WANTED pick has fallen OUT of tier 1 -- reprice from scratch"

    # DILUTION candidates: the nine sendable rows of the 08-23 plan, PLUS every tier-2 file.
    # A tier-2 file is a real counterfactual tier-1 member -- it missed by one reporting step.
    CAND = [c for c in PLAN_0823[1:]] + list(TIER2)
    CAND = [c for c in dict.fromkeys(CAND) if c not in TIER1]
    NAMES = sorted(set(TIER1 + TIER2 + list(WANTED) + CAND))
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

    Sxx = St + Sp
    Mm = St @ np.linalg.inv(Sxx)
    Kmap = beta * np.eye(len(NAMES)) + (1 - beta) * Mm
    Cpri = (1 - beta) ** 2 * (St - Mm @ Sxx @ Mm.T)
    Cpri = 0.5 * (Cpri + Cpri.T)
    # ⚠ A candidate is UNSENT: it has no public score, so it contributes no row to the GLS fit
    # of the common gap. Fit `g` on the SCORED files only and apply the same g to everyone.
    scored = [k for k in NAMES if k in LB]
    js = [col[k] for k in scored]
    z = np.array([LB[k] - cv[k] for k in scored]) / U
    Ss = Sxx[np.ix_(js, js)]
    D = np.ones((len(js), 1))
    iS = np.linalg.inv(Ss)
    Vg = np.linalg.inv(D.T @ iS @ D)
    gh = float(np.ravel(Vg @ (D.T @ iS @ z))[0])
    xh = np.zeros(len(NAMES))
    for k in NAMES:
        xh[col[k]] = (LB[k] - cv[k]) / U - gh if k in LB else 0.0
    mu = Kmap @ xh
    cov = Cpri + Kmap @ np.ones((len(NAMES), 1)) @ Vg @ np.ones((1, len(NAMES))) @ Kmap.T
    s_pri = abs(beta) * np.sqrt(np.diag(Sp)) / abs(CORR_CONS)
    rsd = s_pri * np.sqrt(max(1 - CORR_CONS ** 2, 0.0))
    print(f"  GLS common gap G = {gh:+.2f}e-6 over {len(scored)} scored files; "
          f"{len(NAMES) - len(scored)} unscored candidates carry G and no idiosyncratic shift")

    iw = [col[k] for k in WANTED]

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

    def model_b(tier):
        return float(np.mean([price(list(p)) for p in itertools.combinations(sorted(tier), 2)]))

    # ------------------------------------------------------------------------------ GATE R
    base = model_b(TIER1)
    uni1 = float(np.mean([price([x]) for x in TIER1]))
    dR, dU = abs(base - W57A_MODEL_B), abs(uni1 - W57A_UNIFORM_L1)
    print(f"\n  GATE R: MODEL B on tier 1 = {base:.12f}e-6 vs w57a {W57A_MODEL_B:.12f} "
          f"(|d| {dR:.2e})")
    print(f"          uniform limit-1    = {uni1:.12f}e-6 vs w57a {W57A_UNIFORM_L1:.12f} "
          f"(|d| {dU:.2e})")
    if dR > 1e-6 or dU > 1e-6:
        print("⛔ GATE R FAILED: enlarging NAMES moved w57a's headline. Every figure below would")
        print("   be on a different scale from the number it claims to supersede. Refusing.")
        sys.exit(3)
    print("          -- PASS, the enlarged design reproduces w57a exactly")

    # ------------------------------------------------------------- (A) THE HIJACK, BOUNDED
    print(f"\n=== (A) HIJACK: a file lands ABOVE the tier and takes auto-slot 1 alone ===")
    print(f"  auto pair becomes {{X, one uniform draw from the old tier 1}}.")
    print(f"  X worthless  -> the pair degenerates to that single draw -> cost = {uni1:+.2f}e-6")
    print(f"  X honest     -> X dominates E[max]                       -> cost <= 0")
    print(f"  status quo (no hijack, MODEL B over the 5-file tier)     ->        {base:+.2f}e-6")
    print(f"  so the hijack multiplies the standing exposure by up to  {uni1 / base:.2f}x")
    print(f"\n  ⚠ NO POINT ESTIMATE IS GIVEN FOR {HIJACK}. Its quality is exactly what the")
    print( "    08-23 read exists to measure; pricing it here would assume the answer. w48d's")
    print( "    registered band puts its PUBLIC score at [0.9711976, 0.9712523], entirely above")
    print(f"    the {t1v:.5f} tier, and w56 measures INDETERMINATE (-> worthless end) as modal.")

    # -------------------------------------------------------------- (B) THE DILUTION CURVE
    print("\n=== (B) DILUTION: a file lands IN the tier as a 6th member ===")
    print(f"  baseline MODEL B over the live {len(TIER1)}-file tier 1: {base:+.2f}e-6")
    print(f"\n  {'candidate':26s} {'dCV vs pick':>11s} {'new MODEL B':>12s} {'delta':>8s} "
          f"{'verdict':>9s}  where")
    rows = []
    for x in CAND:
        nb = model_b(list(TIER1) + [x])
        d = nb - base
        rows.append(dict(stem=x, dcv=(cv[x] - cv[PICK]) / U, model_b=nb, delta=d,
                         helps=bool(d < 0),
                         where=("tier2" if x in TIER2 else "08-23 plan")))
    rows.sort(key=lambda r: -r["dcv"])
    for r in rows:
        print(f"  {r['stem']:26s} {r['dcv']:+11.2f} {r['model_b']:+12.2f} {r['delta']:+8.2f} "
              f"{'HELPS' if r['helps'] else 'HURTS':>9s}  {r['where']}")

    # break-even D: the dcv at which delta crosses zero, bracketed by the two nearest rows
    hi = max((r for r in rows if r["helps"]), key=lambda r: -r["dcv"], default=None)
    lo = min((r for r in rows if not r["helps"]), key=lambda r: -r["dcv"], default=None)
    Dbe = None
    if hi and lo and hi["dcv"] > lo["dcv"]:
        t = hi["delta"] / (hi["delta"] - lo["delta"])
        Dbe = -(hi["dcv"] + t * (lo["dcv"] - hi["dcv"]))
        print(f"\n  BREAK-EVEN D = {Dbe:.2f}e-6, linearly interpolated between "
              f"{hi['stem']} ({hi['dcv']:+.2f}, {hi['delta']:+.2f}) and "
              f"{lo['stem']} ({lo['dcv']:+.2f}, {lo['delta']:+.2f}).")
        print(f"  A file joining tier 1 HELPS iff its cross-fitted CV is within {Dbe:.1f}e-6 of")
        print(f"  the pick, i.e. CV >= {cv[PICK] - Dbe * U:.10f}. Below that it manufactures new")
        print( "  pick-missing pairs faster than it dilutes the old ones.")
    else:
        print("\n  ⚠ the candidate set does not bracket the break-even; D not reported.")

    # --------------------------------------------------------------------- THE 08-23 PLAN
    print("\n=== the 08-23 plan, classified ===")
    qp = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    qp["stem"] = qp.file.str.replace(r"\.csv$", "", regex=True)
    pl = qp.set_index("stem").pred_lb.to_dict()
    n_above = 0
    for i, s in enumerate(PLAN_0823, 1):
        p = pl.get(s, float("nan"))
        above = p >= t1v
        n_above += bool(above)
        tag = "⛔ ABOVE TIER — HIJACK" if above else "safe (below tier)"
        print(f"  {i:2d}. {s:26s} pred_lb {p:.6f}  {tag}")
    print(f"\n  {n_above} of {len(PLAN_0823)} planned rows sit at or above the "
          f"{t1v:.5f} tier.")

    # ------------------------------------------------- GATE S: the guard is still IN the sender
    # w55a's idiom. A gate that lives in a paragraph is not a gate; a gate that was wired once
    # and then edited out is worse, because the paragraph still says it is there.
    print("\n=== GATE S: the w58 above-tier guard is still wired into w26g_send.py ===")
    src = open(os.path.join(HERE, "w26g_send.py")).read()
    need = {
        "the flag": 'ap.add_argument("--allow-above-tier"',
        "the live tier": "TIER = auto_tier(rows)",
        "the WANTED CV bar": "CVBAR = wanted_cv_bar()",
        "the eligibility test": "_aw = above_tier_reason(r, CVBAR)",
        "the probabilistic test": "_risk >= P_MAX",
        "the block": "above.append((r.file",
    }
    gs_fail = [k for k, v in need.items() if v not in src]
    for k, v in need.items():
        print(f"  {'ok ' if v in src else '⛔ '} {k:24s} {v}")
    # ⚠ AND the constants the sender owns must still match the pricer it deliberately does NOT
    # import. w57 §4: a priced module on the send path is how the sender stops importing, so the
    # sd is duplicated on purpose -- which makes checking it from OUTSIDE the send path mandatory.
    import w53a_pricer as W53A                                              # noqa: E402
    era_sd = float(W53A.pred_sd("w42_ad217stdcorr"))     # ad>=195 -> the WIDE branch
    base_sd = float(W53A.pred_sd("w21_ad187corr_ens4"))  # pre-era   -> the narrow branch
    snd_sd = float(re.search(r"^PRED_SD = ([0-9.e-]+)", src, re.M).group(1))
    ok_sd = abs(snd_sd - era_sd) < 1e-9 and era_sd >= base_sd
    print(f"  {'ok ' if ok_sd else '⛔ '} PRED_SD              sender {snd_sd:.3e} vs w53a wide "
          f"{era_sd:.3e} (narrow {base_sd:.3e})")
    if gs_fail or not ok_sd:
        print("⛔ GATE S FAILED — the guard has been edited out or the pricer was refitted "
              "under it.")
        sys.exit(4)
    print("  -- PASS")

    # ---------------------------------------------------- P7: can the ARM 217 read move WANTED?
    # Registered as the DECISION test. If any branch of w48d/w56 could move WANTED, the read has
    # real selection value and a flat deferral is the wrong instrument.
    import check_selection as CS                                            # noqa: E402
    arm = json.load(open(os.path.join(HERE, "w48d_arm217.json")))
    barred = [pat for pat in CS.WANTED_INELIGIBLE if pat.startswith("w42_ad217")]
    print("\n=== P7: can ANY branch of the 08-23 ARM 217 read move WANTED? ===")
    print(f"  registered rule: HONEST >= {arm['rule']['honest_at_or_above']}, "
          f"INFLATED <= {arm['rule']['inflated_at_or_below']}")
    print(f"  check_selection.WANTED_INELIGIBLE bars ad217: {bool(barred)}")
    print( "  w56's conjunction, as registered: INFLATED -> veto FINAL; INDETERMINATE (modal) ->")
    print( "  change nothing; HONEST -> 'licenses SENDING and pricing ad217 on the LB, never")
    print( "  SELECTING it'. No branch moves WANTED.")
    p7 = bool(barred)
    print(f"  P7 {'CONFIRMED — the read has ZERO expected value for the private score' if p7 else '*** FALSIFIED — the trade is live, do NOT flat-block ***'}")

    out = dict(
        gate_s="PASS", p7_no_branch_moves_wanted=p7,
        sender_pred_sd=snd_sd, w53a_wide_sd=era_sd, w53a_narrow_sd=base_sd,
        tiers=dict(slot1=TIER1, slot2=TIER2, slot1_public=float(t1v), slot2_public=float(t2v)),
        gate_t="PASS", gate_r=dict(model_b=base, w57a=W57A_MODEL_B, uniform_l1=uni1),
        hijack=dict(stem=HIJACK, pred_lb=float(pl.get(HIJACK, float("nan"))),
                    cost_if_worthless=uni1, cost_if_honest="<=0", status_quo=base,
                    multiplier=uni1 / base,
                    note="no point estimate: the file's quality is what the 08-23 read measures"),
        dilution=rows, break_even_e6=Dbe,
        break_even_cv=(cv[PICK] - Dbe * U) if Dbe else None,
        plan_0823={s: float(pl.get(s, float("nan"))) for s in PLAN_0823},
        n_above_tier=n_above, tier_public=float(t1v),
    )
    with open(os.path.join(HERE, "w58a_tiergate.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote experiments/w58a_tiergate.json")


if __name__ == "__main__":
    main()
