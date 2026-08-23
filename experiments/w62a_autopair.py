"""w62a — THE AUTO-SELECTED PAIR IS NOW DETERMINED. Price it.

Pre-registration: experiments/w62_prereg.txt, committed ba1e7c0 BEFORE this file existed.

WHY THIS FILE EXISTS.
The 08-23 ten went out at 12:41 UTC. `w36_ad199stdcorr_ens4` and `w38_ad202stdcorr_ens4` —
w60's build, "the lever" — BOTH scored 0.97119, strictly above the 0.97118 auto-selection tier
they were built to clear, and they are the only two files on the board at that score.

Auto-selection takes the best two by PUBLIC score. With exactly two files above every other
file, **the auto-selected pair is no longer a draw: it is DETERMINED.** w57a's MODEL B — a
uniform draw over the 10 two-subsets of a five-file tier 1, mean +7.855e-6 — prices a quantity
that no longer exists, and w57a says so itself by refusing to run:

    assert PICK in TIER1, "the WANTED pick has fallen OUT of tier 1 -- reprice from scratch"

⚠ THAT ASSERTION IS CORRECT AND IS NOT SOFTENED HERE. `w57a_tierprice2.py` IS NOT EDITED. It is
a registered artefact against a board configuration that no longer exists; re-pointing its R1-R7
at a new board would destroy the record it exists to be. This is its SUCCESSOR, not its patch.

⚠ AND THE PICK IS THE THING THAT MOVED. `w36_ad199stdcorr` (WANTED slot 1, the CV leader) is at
0.97118 and is now in TIER 2. P(pick auto-selected) went 0.400 -> 0.000. The lever bought a
determined pair and gave up the 40% chance of drawing the pick itself. Whether that trade was
net good is the whole question, and nobody here had computed it.

THE MATH IS w57a's, COPIED VERBATIM AND ONLY PARAMETERISED. w57a's own words: "Reusing the math
verbatim is the point: a re-price that also re-derived the estimator could not be compared
against the number it supersedes." Same beta, same LAW-IF conditioning, same GLS common gap,
same Clark E[max], same conservative non-additivity corr, same tau sweep.

GATE V — THE VERBATIM CONTROL. Before it prices anything live, this runs its own estimator on
the board state RECORDED IN `w57a_tierprice2.json` (that run's NAMES, that run's LB, CV
recomputed from the raw OOF vectors) and REFUSES unless it reproduces every `limit1` cost and
every one of the 10 `model_b` pair prices to < 1e-9. A copied estimator that does not reproduce
the number it supersedes is a different estimator wearing its name.

    .venv/bin/python experiments/w62a_autopair.py
"""
from __future__ import annotations

import io, itertools, json, os, subprocess, sys

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw                     # noqa: E402
from w16b_cellweight import fast_auc                         # noqa: E402
from stdflag import family                                   # noqa: E402
from w57a_tierprice2 import midrank_cdf, emax                # noqa: E402 — SHARED, not re-derived

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, U = 296_302, 0.20, 1e-6

WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")
PICK = "w36_ad199stdcorr"
CORR_CONS, CORR_OPT = 0.9923594211711959, 0.9971372069496918

W57A_MODEL_B = 7.855203984805849      # the quantity this supersedes
W60_LEVER = 1.127                     # w60's price for w36_ad199stdcorr_ens4 above the OLD tier
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


# ----------------------------------------------------------------------------------------
# THE ESTIMATOR — w57a's main() body, lines 104-200, with (NAMES, LB, cv) lifted to arguments
# and NOTHING ELSE CHANGED. GATE V is what proves that claim rather than asserting it.
# ----------------------------------------------------------------------------------------
def build(NAMES, LB, cv, V, y, beta, wanted=WANTED):
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    n = len(y)
    A = np.empty((len(NAMES), n1)); B = np.empty((len(NAMES), n0))
    for i, k in enumerate(NAMES):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])
    C1, C0 = np.cov(A), np.cov(B)
    m, n_pub = N_TEST, int(round(N_TEST * F))
    pi1 = n1 / n
    S_t = (1.0 - m / n) * (C1 / (m * pi1) + C0 / (m * (1 - pi1)))
    S_p = (1.0 - n_pub / m) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1)))
    St, Sp = S_t / U ** 2, S_p / U ** 2
    M = St @ np.linalg.inv(St + Sp)
    offI = float(np.abs(M - np.eye(len(NAMES)) * np.diag(M).mean()).max())
    gamma = beta + (1 - beta) * float(np.diag(M).mean())
    col = {k: i for i, k in enumerate(NAMES)}
    iw = [col[k] for k in wanted]

    def uvec(x, z):
        u = np.zeros(len(NAMES)); u[col[x]], u[col[z]] = 1.0, -1.0
        return u

    def solve(tau, corr):
        Stt = St + tau ** 2 * np.eye(len(NAMES))
        Sxx = Stt + Sp
        Mm = Stt @ np.linalg.inv(Sxx)
        Kmap = beta * np.eye(len(NAMES)) + (1 - beta) * Mm
        Cpri = (1 - beta) ** 2 * (Stt - Mm @ Sxx @ Mm.T)
        Cpri = 0.5 * (Cpri + Cpri.T)
        z = np.array([LB[k] - cv[k] for k in NAMES]) / U
        D = np.ones((len(NAMES), 1))
        iS = np.linalg.inv(Sxx)
        Vg = np.linalg.inv(D.T @ iS @ D)
        gh = Vg @ (D.T @ iS @ z)
        xh = z - D @ gh
        mu = Kmap @ xh
        cov = Cpri + Kmap @ D @ Vg @ D.T @ Kmap.T
        s_pri = abs(beta) * np.sqrt(np.diag(Sp)) / abs(corr)
        rsd = s_pri * np.sqrt(max(1 - corr ** 2, 0.0))
        return mu, cov, rsd, float(gh[0])

    def emax_set(mu, cov, rsd, files):
        ia = [col[f] for f in files]
        ma = np.array([mu[i] + cv[NAMES[i]] / U for i in ia])
        if len(ia) == 1:
            return float(ma[0])
        sa = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in ia])
        ra = cov[ia[0], ia[1]] / np.sqrt(cov[ia[0], ia[0]] * cov[ia[1], ia[1]])
        return float(emax(ma, sa, ra))

    def price(mu, cov, rsd, files):
        """cost of NOT clicking = E[max over WANTED] - E[max over the auto picks]."""
        mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
        sw = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in iw])
        rw = cov[iw[0], iw[1]] / np.sqrt(cov[iw[0], iw[0]] * cov[iw[1], iw[1]])
        return float(emax(mw, sw, rw) - emax_set(mu, cov, rsd, files))

    def pjoint(mu, cov, rsd, x):
        if x in wanted:
            return None
        mus = np.array([(cv[x] - cv[k]) / U + float(uvec(x, k) @ mu) for k in wanted])
        cc = np.empty((2, 2))
        for i, ki in enumerate(wanted):
            for j, kj in enumerate(wanted):
                cc[i, j] = uvec(x, ki) @ cov @ uvec(x, kj)
        for i, ki in enumerate(wanted):
            cc[i, i] += rsd[col[x]] ** 2 + rsd[col[ki]] ** 2
        return float(multivariate_normal(mean=-mus, cov=cc, allow_singular=True).cdf([0.0, 0.0]))

    return dict(solve=solve, price=price, pjoint=pjoint, emax_set=emax_set,
                offI=offI, gamma=gamma, col=col)


def _load(names, y):
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    return V, {k: fast_auc(y, V[k]) for k in names}


def main() -> None:
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    # ------------------------------------------------------------------ GATE V
    print("=" * 92)
    print("GATE V — the copied estimator must reproduce w57a_tierprice2.json on w57a's OWN board")
    print("=" * 92)
    ref = json.load(open(os.path.join(HERE, "w57a_tierprice2.json")))
    rNAMES = sorted(ref["cv"])
    rLB = {k: float(v) for k, v in ref["lb"].items()}
    rV, rcv = _load(rNAMES, y)
    # CV is recomputed from the raw OOF, never re-quoted from the table being checked.
    for k in rNAMES:
        if abs(rcv[k] - float(ref["cv"][k])) > 5e-10:
            fail(f"GATE V: CV of {k} does not reproduce from the OOF vector "
                 f"({rcv[k]:.12f} vs {float(ref['cv'][k]):.12f})")
    rE = build(rNAMES, rLB, rcv, rV, y, beta)
    rmu, rcov, rrsd, rgap = rE["solve"](0.0, CORR_CONS)
    if abs(rgap - float(ref["gap"])) > 1e-9:
        fail(f"GATE V: GLS common gap {rgap:.9f} != recorded {float(ref['gap']):.9f}")
    worst = 0.0
    for k, rec in ref["limit1"].items():
        got = rE["price"](rmu, rcov, rrsd, [k])
        worst = max(worst, abs(got - float(rec["cost"])))
        if abs(got - float(rec["cost"])) > 1e-9:
            fail(f"GATE V: limit1[{k}] {got:+.12f} != recorded {float(rec['cost']):+.12f}")
    for key, val in ref["model_b"]["pairs"].items():
        got = rE["price"](rmu, rcov, rrsd, key.split("+"))
        worst = max(worst, abs(got - float(val)))
        if abs(got - float(val)) > 1e-9:
            fail(f"GATE V: model_b[{key}] {got:+.12f} != recorded {float(val):+.12f}")
    mb = float(np.mean([rE["price"](rmu, rcov, rrsd, k.split("+"))
                        for k in ref["model_b"]["pairs"]]))
    print(f"  {len(rNAMES)} files, {len(ref['limit1'])} limit1 costs, "
          f"{len(ref['model_b']['pairs'])} MODEL B pairs")
    print(f"  worst absolute deviation from the recorded artefact: {worst:.3e}   (bar 1e-9)")
    print(f"  MODEL B mean reproduced {mb:.12f}e-6  vs recorded {W57A_MODEL_B:.12f}e-6")
    if FAILURES:
        print("\n⛔ GATE V FAILED — the estimator is not w57a's. REFUSING to price anything.")
        json.dump(dict(gate_v=False, failures=FAILURES),
                  open(os.path.join(HERE, "w62a_autopair.json"), "w"), indent=1)
        sys.exit(1)
    print("  ✅ GATE V PASSED — every recorded price reproduces. P2 CONFIRMED.")

    # ------------------------------------------------------------------ the LIVE board
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
    NAMES = sorted(set(TIER1 + TIER2 + list(WANTED)))
    print("\n" + "=" * 92)
    print(f"LIVE BOARD: {len(sub)} scored submissions, {len(agg)} distinct files")
    print("=" * 92)
    print(f"  auto-slot tier   public {t1v:.5f}: {len(TIER1)} files {TIER1}")
    print(f"  next tier down   public {t2v:.5f}: {len(TIER2)} files {TIER2}")

    # ⚠ THE DETERMINACY TEST. The auto pair is determined IFF tier 1 holds exactly two files:
    # then "best two by true score" is that pair whatever the sub-rounding order inside it.
    # With 1 file the second slot is a draw from tier 2; with >=3 it is a draw within tier 1.
    DETERMINED = len(TIER1) == 2
    if not DETERMINED:
        print(f"\n⛔ TIER 1 HOLDS {len(TIER1)} FILES — the auto pair is NOT determined and this "
              f"file does not\n   price that case. w57a's MODEL B (>=3) or MODEL A (==1) is the "
              f"right instrument.\n   REFUSING.")
        json.dump(dict(gate_v=True, determined=False, n_tier1=len(TIER1), tier1=TIER1),
                  open(os.path.join(HERE, "w62a_autopair.json"), "w"), indent=1)
        sys.exit(2)
    AUTO = list(TIER1)
    print(f"\n  ✅ DETERMINED: exactly two files above every other file. The auto-selected pair "
          f"is\n     {AUTO[0]} + {AUTO[1]}, with no tie to break.")

    V, cv = _load(NAMES, y)
    print(f"\n  {'file':26s} {'CV':>14s} {'public':>9s} {'dCV vs pick':>12s}  fam      where")
    for k in NAMES:
        where = "AUTO" if k in AUTO else ("tier2" if k in TIER2 else "-")
        if k in WANTED:
            where += " WANTED" + ("/PICK" if k == PICK else "")
        print(f"  {k:26s} {cv[k]:.10f} {LB[k]:9.5f} {(cv[k]-cv[PICK])/U:+12.2f}  "
              f"{family(k):<8s} {where}")

    E = build(NAMES, LB, cv, V, y, beta)
    print(f"\n=== LAW-IF ===\n  max |M - w*I| = {E['offI']:.3e}   gamma = {E['gamma']:+.6f}")
    assert E["offI"] < 1e-6, "M is not w*I -- w18a's P6 does not hold on this file set"
    mu, cov, rsd, gap = E["solve"](0.0, CORR_CONS)
    print(f"  free common gap G = {gap:+.2f}e-6 (GLS); residual sd per file "
          f"{rsd.min():.2f}..{rsd.max():.2f}e-6")

    # ------------------------------------------------------------------ THE PRICE
    cost = E["price"](mu, cov, rsd, AUTO)
    print("\n" + "=" * 92)
    print("THE DETERMINED PRICE — cost of NOT clicking, units 1e-6, positive = clicking helps")
    print("=" * 92)
    print(f"  E[max over WANTED]      {E['emax_set'](mu, cov, rsd, list(WANTED)):.4f}")
    print(f"  E[max over the AUTO 2]  {E['emax_set'](mu, cov, rsd, AUTO):.4f}")
    print(f"  cost(auto pair)         {cost:+.4f}e-6")
    print(f"    superseded quantity   {W57A_MODEL_B:+.4f}e-6  (w57a MODEL B, 5-file tier 1)")
    print(f"    improvement           {W57A_MODEL_B - cost:+.4f}e-6")

    # each auto file alone, for the decomposition
    print("\n  each auto file priced ALONE (LIMIT 1), and its P(beats BOTH wanted) on private:")
    solo = {}
    for x in AUTO + [PICK]:
        c = E["price"](mu, cov, rsd, [x])
        pj = E["pjoint"](mu, cov, rsd, x)
        solo[x] = dict(cost=c, p_joint=pj, dcv=(cv[x] - cv[PICK]) / U)
        pjs = f"{pj:8.3f}" if pj is not None else f"{'n/a (self)':>8s}"
        print(f"    {x:26s} dCV {solo[x]['dcv']:+7.2f}  cost {c:+8.3f}  P(joint) {pjs}"
              f"{'   <-- IS the pick' if x == PICK else ''}")

    # ⚠ THE COUNTERFACTUAL THE LEVER ACTUALLY REPLACED. w60 priced {lever, one draw from the OLD
    # five-file tier 1}; that draw was the PICK with probability 1/4. Recompute it HERE, on
    # today's board, so P5 is a like-for-like comparison and not a comparison across boards.
    print("\n  what the pair would have cost with the OTHER slot drawn from the old tier 1:")
    old_t1 = [k for k in ref["tiers"]["slot1"] if k in NAMES]
    alt = {}
    for z in old_t1:
        if z == AUTO[0]:
            continue
        alt[z] = E["price"](mu, cov, rsd, [AUTO[0], z])
        print(f"    {AUTO[0]} + {z:26s} {alt[z]:+8.3f}"
              f"{'   <-- the pick' if z == PICK else ''}")
    alt_mean = float(np.mean(list(alt.values()))) if alt else float("nan")
    print(f"    uniform mean over those {len(alt)} draws                    {alt_mean:+8.3f}e-6")

    # ------------------------------------------------------------------ THE COUNTERFACTUAL LADDER
    # ⚠ SELECTION-MECHANISM COUNTERFACTUALS ONLY. Each row holds the fitted private-side
    # posterior fixed and varies only WHICH FILES SIT IN TIER 1. The single-clear rows keep a
    # file's observed 0.97119 while pretending it did not clear, so they are directional and are
    # NOT point predictions. They exist to answer one question: was clearing TWICE better than
    # clearing once? w60 priced each lever file alone and sent both on the same day.
    A, Bf = AUTO[0], AUTO[1]
    tie = [k for k in ref["tiers"]["slot1"] if k in NAMES]        # the 0.97118 tie, 5 files
    ladder = {}
    ladder["neither_cleared"] = float(np.mean(
        [E["price"](mu, cov, rsd, list(p)) for p in itertools.combinations(tie + [A, Bf], 2)]))
    ladder["only_" + A] = float(np.mean(
        [E["price"](mu, cov, rsd, [A, z]) for z in tie + [Bf]]))
    ladder["only_" + Bf] = float(np.mean(
        [E["price"](mu, cov, rsd, [Bf, z]) for z in tie + [A]]))
    ladder["both_cleared_ACTUAL"] = cost
    ladder["click_pick_plus_" + A] = E["price"](mu, cov, rsd, [PICK, A])
    ladder["click_pick_plus_" + Bf] = E["price"](mu, cov, rsd, [PICK, Bf])
    print("\n  THE COUNTERFACTUAL LADDER (selection mechanism only, posterior held fixed):")
    for k in ["neither_cleared", "only_" + Bf, "both_cleared_ACTUAL", "only_" + A,
              "click_pick_plus_" + A, "click_pick_plus_" + Bf]:
        mark = "   <-- WHAT HAPPENED" if k == "both_cleared_ACTUAL" else (
               "   <-- the best branch available" if k == "only_" + A else "")
        print(f"    {k:44s} {ladder[k]:+8.3f}e-6{mark}")
    second_file_cost = ladder["both_cleared_ACTUAL"] - ladder["only_" + A]
    net_vs_nothing = ladder["neither_cleared"] - ladder["both_cleared_ACTUAL"]
    print(f"\n    the SECOND lever file cost               {second_file_cost:+8.3f}e-6")
    print(f"    net gain vs neither clearing             {net_vs_nothing:+8.3f}e-6"
          f"   (w60 registered +6.730)")

    # ------------------------------------------------------------------ TAU / corr sweep
    print("\n=== TAU SWEEP (w19c's sign-flip test) on the DETERMINED pair ===")
    sweep = {}
    for t in [0.0, 0.5, 1.0, 1.72, 2.5, 3.0, 5.0]:
        mt, ct, rt, _ = E["solve"](t, CORR_CONS)
        sweep[t] = E["price"](mt, ct, rt, AUTO)
        print(f"  tau {t:5.2f}   cost {sweep[t]:+8.3f}e-6")
    flip = any(np.sign(sweep[t]) != np.sign(sweep[0.0]) and abs(sweep[t]) > 0.05 for t in sweep)
    mo, co, ro, _ = E["solve"](0.0, CORR_OPT)
    cost_opt = E["price"](mo, co, ro, AUTO)
    print(f"\n=== corr sensitivity ===\n  conservative {cost:+.3f}   optimistic {cost_opt:+.3f}e-6")

    # ------------------------------------------------------------------ the registered set
    p_pick = 1.0 if PICK in AUTO else 0.0
    print("\n" + "=" * 92)
    print("THE REGISTERED PREDICTIONS (w62_prereg.txt)")
    print("=" * 92)
    p1 = bool(set(AUTO) == {"w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4"} and PICK in TIER2)
    p3 = bool(0.0 < cost <= 6.0)
    p4 = bool(cost < W57A_MODEL_B)
    p5 = bool(cost > W60_LEVER)
    p6 = bool(p_pick == 0.0)
    p7 = bool(not flip)
    checks = [
        ("P1 (OBSERVED, not a forecast) the two lever files are the whole of tier 1", p1,
         f"tier1={AUTO}, pick in tier2={PICK in TIER2}"),
        ("P2 GATE V reproduces w57a to < 1e-9", True, f"worst {worst:.3e}"),
        ("P3 0 < cost <= 6.0e-6 (sign fixed in advance by dCV < 0)", p3, f"{cost:+.4f}e-6"),
        ("P4 STRICT cost < 7.855204e-6 (the send improved the exposure)", p4,
         f"{cost:+.4f} vs {W57A_MODEL_B:+.4f}"),
        ("P5 cost > 1.127e-6 (w60's number priced a draw that could BE the pick)", p5,
         f"{cost:+.4f} vs {W60_LEVER:+.4f}"),
        ("P6 P(pick inside the auto 2) == 0.000, was 0.400", p6, f"{p_pick:.3f}"),
        ("P7 no sign flip over tau 0..5", p7, f"flip={flip}"),
    ]
    for name, ok, detail in checks:
        print(f"  {'CONFIRMED' if ok else '*** FALSIFIED ***':18s} {name}   [{detail}]")
        if not ok:
            fail(name)
    print(f"\nFAILURES: {FAILURES}")

    json.dump(dict(
        gate_v=True, gate_v_worst=worst, determined=True,
        tier1=AUTO, tier1_public=float(t1v), tier2=TIER2, tier2_public=float(t2v),
        auto_pair=AUTO, pick=PICK, pick_in_auto=bool(PICK in AUTO), p_pick=p_pick,
        wanted=list(WANTED), n_scored=int(len(sub)),
        cv={k: cv[k] for k in NAMES}, lb={k: float(LB[k]) for k in NAMES},
        beta=beta, gamma=E["gamma"], gap=gap,
        cost_auto_pair=cost, cost_optimistic=cost_opt,
        supersedes_value=W57A_MODEL_B, improvement=W57A_MODEL_B - cost,
        solo=solo, old_tier1_draws=alt, old_tier1_draw_mean=alt_mean,
        ladder=ladder, second_file_cost=second_file_cost,
        net_vs_nothing=net_vs_nothing, w60_registered_gain=6.730,
        tau_sweep={str(k): v for k, v in sweep.items()}, tau_flip=flip,
        supersedes="w57a_tierprice2 (5-file tier 1, void since the 08-23 sends)",
        p1=p1, p2=True, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7, failures=FAILURES,
    ), open(os.path.join(HERE, "w62a_autopair.json"), "w"), indent=1, default=float)
    print("wrote experiments/w62a_autopair.json")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
