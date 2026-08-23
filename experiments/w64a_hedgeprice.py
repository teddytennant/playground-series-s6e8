"""w64a — IS WANTED's SLOT 2 A HEDGE WORTH 22.5e-6 OF CV? Two instruments, one decision.

Pre-registration: experiments/w64_prereg.txt, committed 0eb2e53 BEFORE this file existed.

THE QUESTION, DEFERRED FOR THREE RUNS (w61 §8.7, w62 §7.7, w63 §10.8).

    check_selection.WANTED = {"w36_ad199stdcorr.csv", "w23_ad187stdcorr.csv"}

Slot 2 is the identical `*stdcorr` construction on the 187 pack. Its stated job (check_selection,
w24 R1) is to INSURE THE MEMBER ADDITIONS: if pack growth is worse on the private slice than CV
says, slot 2 catches it. Every deferral has said the same thing — "slot 2 is a PACK HEDGE whose
value is not its CV; any move needs a registered argument about the HEDGE" — and then not made
the argument. This file makes it, or fails to.

⚠ THE DEFERRAL NAMES THE WRONG CANDIDATE, AND HAS FOR THREE RUNS. All three say
`w40_ad211stdcorr` is "+22.4e-6 above slot 2 on CV and eligible". It is. But `w38_ad202stdcorr`
is +22.5e-6 — HIGHER by 0.116e-6 — and ARM 202 is ARM 211 MINUS the nine `yadoy666` union94
streams and nothing else (w61a GATE M). ARM 211 carries a `WANTED_RETIRED` record whose own text
says "⚠ THIS IS NOT es-CLEARANCE ... w40d's reasoning is still correct about them". ARM 202
carries no encumbrance. So the file argued about for three runs is dominated by its own matched
control on CV and on provenance at once. That is a ledger fact, registered in §1 of the prereg
before anything here was computed.

TWO INSTRUMENTS, BECAUSE THE HEDGE HAS TWO HALVES AND ONE MACHINE SEES ONLY ONE OF THEM.

(A) E[max] on the fitted private posterior. `w63a_setprice.fit` is IMPORTED, not copied — GATE A
    reproduces `w63a_setprice.json`'s determined price to < 1e-9 before anything is read off it.
    A hedge enters this instrument through EXACTLY ONE channel: the correlation between slot 1
    and slot 2. A worse-in-mean, less-correlated second file buys E[max]; that is the whole
    mechanism and (A) is a clean test of it.

(B) The structural half, which (A) is blind to. (A)'s posterior is Gaussian with a single fitted
    common gap; it cannot represent "the entire member-addition family transfers worse than its
    CV". That is what the hedge is actually for, so it is measured directly, as a GLS regression
    of the public-slice residual on PACK SIZE over all 53 scored files that carry an `ad<NNN>`
    tag and an OOF vector on disk. The covariance is the workspace's own `Sxx = St + Sp`, the
    same matrix w59a/w63a use to fit the common gap `gh` — this generalises that fit from
    `D1 = ones` to a design matrix, it does not introduce a second estimator.

    δ = e-6 of transfer lost per member added. THE BREAK-EVEN IS ARITHMETIC AND WAS FIXED IN THE
    PREREG BEFORE δ EXISTED: the move buys 22.507e-6 across 15 members, so it pays iff
    15·δ > −22.507e-6, i.e. δ > −1.5005e-6 per member.

    .venv/bin/python experiments/w64a_hedgeprice.py             # measures, writes the artefact
    .venv/bin/python experiments/w64a_hedgeprice.py --no-write  # what-if, artefact NOT written
"""
from __future__ import annotations

import io, itertools, json, math, os, re, subprocess, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from common import SUB, TARGET, load_raw                      # noqa: E402
from w16b_cellweight import fast_auc                          # noqa: E402
from stdflag import family                                    # noqa: E402
import w63a_setprice as W63A                                  # noqa: E402 — IMPORTED, not copied

COMP = "playground-series-s6e8"
U = 1e-6
LB_STEP = 1e-5                       # the public LB is reported to five decimals

# ---- the decision, as registered -------------------------------------------------------------
SLOT1 = "w36_ad199stdcorr"
INCUMBENT = "w23_ad187stdcorr"
CHALLENGER = "w38_ad202stdcorr"
ALSO = "w40_ad211stdcorr"            # the file the three deferrals named
PACK_OF_INCUMBENT, PACK_OF_CHALLENGER = 187, 202
DELTA_BREAKEVEN = None               # computed from the CV ledger below; the FORMULA is fixed
P2_BAR = 1.0                         # P2: emax gain must exceed +1.0e-6, strictly

# the `*stdcorr` ladder — the SAME construction at every pack size it was ever built at. This is
# the design that makes "how good a hedge is pack N" a readable curve instead of one contrast.
LADDER = ["w23_ad187stdcorr", "w27_ad188stdcorr", "w27_ad190stdcorr", "w29_ad194stdcorr",
          "w34_ad195stdcorr", "w36_ad197stdcorr", "w36_ad199stdcorr", "w38_ad202stdcorr",
          "w40_ad211stdcorr"]

PACK_RE = re.compile(r"ad(\d{3})")
FAILURES = 0
FALSIFIED = []


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def pack_of(stem: str):
    m = PACK_RE.search(stem)
    return int(m.group(1)) if m else None


def load_board():
    """The live scored board. Same read as w63a: one page, asserted not truncated."""
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert len(sub) < 500, "hit the page size -- raise it, the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree -- scores are not deterministic"
    return agg["max"].to_dict()


def load_oof(names, y):
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    return V, {k: fast_auc(y, V[k]) for k in names}


# ==============================================================================================
# GATE A — the imported estimator must reproduce w63a's own artefact on w63a's own design.
# ==============================================================================================
# ⚠ w62's lesson, applied here: a STAMP is not a COMPARISON. This does not read `gate_w: true`
# out of the artefact and believe it. It rebuilds w63a's design from the LIVE board and re-derives
# the determined price, so a board that has moved since w63a ran is caught HERE and not silently
# carried into every number below.
def gate_a(y, beta, LB):
    print("=" * 96)
    print("GATE A — the IMPORTED estimator reproduces w63a_setprice.json on the live board")
    print("=" * 96)
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    top = pd.Series(LB).sort_values(ascending=False)
    vals = sorted(set(top.values), reverse=True)
    TIER1 = sorted(top[top == vals[0]].index)
    TIER2 = sorted(top[top == vals[1]].index)
    if TIER1 != sorted(ref["auto_pair"]):
        print(f"\n⛔ GATE A: live auto-slot-1 {TIER1} is not the pair w63a priced "
              f"{sorted(ref['auto_pair'])}.\n   Re-run w63a first. REFUSING.")
        sys.exit(2)
    if TIER2 != sorted(ref["tiers"]["slot2"]):
        print(f"\n⛔ GATE A: live tier 2 {TIER2} is not w63a's {sorted(ref['tiers']['slot2'])}. "
              f"REFUSING.")
        sys.exit(2)
    CAND = [c for c in W63A.PLAN_0824 if c not in TIER1]
    NAMES = sorted(set(TIER1 + TIER2 + list(W63A.WANTED) + CAND))
    V, cv = load_oof(NAMES, y)
    E = W63A.fit(NAMES, LB, cv, V, y, beta)
    base = E["price"](TIER1)
    dev = abs(base - float(ref["base"]))
    print(f"  w63a design rebuilt: {len(NAMES)} files, determined price {base:.12f}")
    print(f"  recorded in w63a_setprice.json:                       {float(ref['base']):.12f}")
    print(f"  |deviation| = {dev:.3e}   (bar 1e-9)")
    if dev > 1e-9:
        print("\n⛔ GATE A FAILED — the import is not w63a's instrument. REFUSING.")
        sys.exit(1)
    for k in ("gap", "gamma"):
        if abs(E[k] - float(ref[k])) > 1e-9:
            fail(f"GATE A: {k} {E[k]:.12f} != recorded {float(ref[k]):.12f}")
    if FAILURES:
        sys.exit(1)
    print("  ✅ GATE A PASSED — P1 CONFIRMED. base, gap and gamma all reproduce.")
    return TIER1, TIER2, NAMES, base, ref


# ==============================================================================================
# PART A — E[max] over candidate WANTED pairs.
# ==============================================================================================
def part_a(y, beta, LB, base63, NAMES63, TIER1):
    print("\n" + "=" * 96)
    print("PART A — E[max] on the fitted private posterior, over candidate WANTED pairs")
    print("=" * 96)
    NAMES = sorted(set(NAMES63) | set(LADDER))
    missing = [k for k in NAMES if not os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))]
    assert not missing, f"no OOF vector for {missing} -- cannot price them, do not guess"
    V, cv = load_oof(NAMES, y)
    E = W63A.fit(NAMES, LB, cv, V, y, beta)
    emax, cov, col = E["emax_set"], E["cov"], E["col"]

    # GATE R2 — w63a's GATE R, re-run on the enlarged design. If widening the design to the whole
    # ladder moves the headline the design is doing the work, not the data.
    base = E["price"](TIER1)
    dR = abs(base - base63)
    print(f"\n  GATE R2: determined price on the {len(NAMES)}-file ladder design {base:.12f}")
    print(f"           vs the {len(NAMES63)}-file w63a design                  {base63:.12f}")
    print(f"           |d| = {dR:.3e}   (bar 1.0e-6)")
    if dR > 1.0e-6:
        print("⛔ GATE R2 FAILED: widening to the ladder moved the headline. REFUSING.")
        sys.exit(3)
    print("           -- PASS.")

    def rho(a, b):
        return float(cov[col[a], col[b]] / math.sqrt(cov[col[a], col[a]] * cov[col[b], col[b]]))

    solo1 = emax([SLOT1])
    print(f"\n  slot 1 = {SLOT1}  CV {cv[SLOT1]:.10f}  E[private] {solo1:.4f} (units of 1e-6 AUC)")
    print(f"\n  {'slot-2 candidate':22s} {'pack':>5s} {'dCV vs inc':>11s} {'corr w/ s1':>11s} "
          f"{'E[max]':>12s} {'vs incumbent':>13s}")
    inc_emax = emax([SLOT1, INCUMBENT])
    rows = []
    for k in LADDER:
        if k == SLOT1:
            continue
        e = emax([SLOT1, k])
        r = dict(stem=k, pack=pack_of(k), cv=cv[k], dcv=(cv[k] - cv[INCUMBENT]) / U,
                 corr=rho(SLOT1, k), emax=e, gain=(e - inc_emax))
        rows.append(r)
        print(f"  {k:22s} {r['pack']:5d} {r['dcv']:+11.3f} {r['corr']:11.6f} "
              f"{e:12.4f} {r['gain']:+13.4f}")

    gain = next(r["gain"] for r in rows if r["stem"] == CHALLENGER)
    gain_also = next(r["gain"] for r in rows if r["stem"] == ALSO)
    print(f"\n  registered P2: emax({SLOT1}, {CHALLENGER}) - emax({SLOT1}, {INCUMBENT}) "
          f"> +{P2_BAR:.1f}e-6, STRICTLY")
    print(f"                 read {gain:+.4f}e-6   -> "
          f"{'✅ CONFIRMED' if gain > P2_BAR else '🔴 FALSIFIED'}")
    if not gain > P2_BAR:
        FALSIFIED.append(f"P2 emax gain > +{P2_BAR}e-6 strictly (read {gain:+.4f})")

    # THE HEDGE'S ONLY CHANNEL IN THIS INSTRUMENT, read out explicitly rather than implied.
    r_inc, r_cha = rho(SLOT1, INCUMBENT), rho(SLOT1, CHALLENGER)
    print(f"\n  the hedge's whole mechanism here is DECORRELATION:")
    print(f"    corr(slot1, {INCUMBENT})  = {r_inc:.6f}   mean deficit {(cv[INCUMBENT]-cv[SLOT1])/U:+.2f}e-6")
    print(f"    corr(slot1, {CHALLENGER}) = {r_cha:.6f}   mean deficit {(cv[CHALLENGER]-cv[SLOT1])/U:+.2f}e-6")
    print(f"    the incumbent is {'LESS' if r_inc < r_cha else 'MORE'} correlated by "
          f"{abs(r_inc - r_cha):.6f} and pays {(cv[CHALLENGER]-cv[INCUMBENT])/U:.2f}e-6 of mean for it")

    # THE RESOLUTION OF THE INSTRUMENT. A verdict of "prefers X" is worth nothing if the whole
    # ladder fits inside the estimator's own noise floor, so the SPREAD is reported next to the
    # contrast and compared against the floors this workspace already measured.
    spread = max(r["emax"] for r in rows) - min(r["emax"] for r in rows)
    print(f"\n  RESOLUTION: the whole {len(rows)}-file ladder spans {spread:.4f}e-6 of E[max] "
          f"while spanning\n              {max(r['dcv'] for r in rows) - min(r['dcv'] for r in rows):.2f}e-6 of CV. "
          f"The contrast being decided is {gain:+.4f}e-6.")
    print(f"              floors on file: rebuild ~2e-6 (w14a), top-level search optimism "
          f"+0.45e-6/parameter (w36d).")

    # The break-even correlation: holding the challenger's mean, how much decorrelation would the
    # incumbent need to buy back its 22.5e-6 mean deficit? Solved by bisection on the emax
    # function itself with the incumbent's own mean and sd, so it is a property of the estimator.
    from w57a_tierprice2 import emax as _emax
    i1, ii = col[SLOT1], col[INCUMBENT]
    mu1 = solo1
    mui = emax([INCUMBENT])
    s1 = math.sqrt(cov[i1, i1] + E["rsd"][i1] ** 2)
    si = math.sqrt(cov[ii, ii] + E["rsd"][ii] ** 2)
    target = emax([SLOT1, CHALLENGER])
    lo_r, hi_r = -0.999, 0.999999
    for _ in range(200):
        mid = 0.5 * (lo_r + hi_r)
        if float(_emax(np.array([mu1, mui]), np.array([s1, si]), mid)) > target:
            lo_r = mid
        else:
            hi_r = mid
    r_star = 0.5 * (lo_r + hi_r)
    print(f"\n  BREAK-EVEN CORRELATION: with its own mean and sd, the incumbent matches the "
          f"challenger's\n    E[max] at corr(slot1, incumbent) = {r_star:.6f}. It actually reads "
          f"{r_inc:.6f} — it is\n    {'SHORT BY' if r_inc > r_star else 'PAST IT BY'} "
          f"{abs(r_inc - r_star):.6f} of correlation.")

    # HOW MUCH OF THE HEDGE'S CV COST DOES DECORRELATION BUY BACK? The counterfactual holds the
    # incumbent's mean and sd and gives it the CHALLENGER's correlation — i.e. a slot-2 file
    # 22.5e-6 worse in mean that is NOT decorrelated at all. Everything between that and the
    # incumbent's actual E[max] is what the hedge earns.
    flat = float(_emax(np.array([mu1, mui]), np.array([s1, si]), r_cha))
    earned, cost = inc_emax - flat, target - flat
    print(f"    the same file with NO decorrelation (corr forced to the challenger's "
          f"{r_cha:.6f}):\n      E[max] {flat:.4f} vs the incumbent's actual {inc_emax:.4f} vs the "
          f"challenger's {target:.4f}")
    print(f"    => decorrelation earns {earned:+.4f}e-6 of the {cost:+.4f}e-6 the 22.5e-6 mean "
          f"deficit costs\n       — {100.0 * earned / cost:.1f}% of it. THE HEDGE IS PRICED AT "
          f"{100.0 * earned / cost:.0f}% OF WHAT IT COSTS.")

    order = sorted(rows, key=lambda r: r["pack"])
    mono = all(a["emax"] <= b["emax"] for a, b in zip(order, order[1:]))
    print(f"  ladder read: E[max] is {'MONOTONE increasing in pack' if mono else 'NOT monotone in pack'}")
    return dict(names=len(NAMES), base_ladder=base, gate_r2=dR, solo_slot1=solo1,
                incumbent_emax=inc_emax, rows=rows, gain=gain, gain_also=gain_also,
                corr_incumbent=r_inc, corr_challenger=r_cha, corr_breakeven=r_star,
                emax_no_decorr=flat, decorr_earned=earned, decorr_cost=cost,
                decorr_pct=100.0 * earned / cost,
                emax_spread=spread, monotone_in_pack=bool(mono),
                cv={k: cv[k] for k in LADDER}), cv


# ==============================================================================================
# PART B — does PACK GROWTH transfer? The GLS regression the hedge's premise lives or dies on.
# ==============================================================================================
def part_b(y, beta, LB):
    print("\n" + "=" * 96)
    print("PART B — GLS: does the public-slice residual depend on PACK SIZE?")
    print("=" * 96)
    names = sorted(k for k in LB
                   if pack_of(k) and os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))
                   and family(k) != "member")
    V, cv = load_oof(names, y)
    E = W63A.fit(names, LB, cv, V, y, beta)
    Sxx, col = E["Sxx"], E["col"]

    z = np.array([(LB[k] - cv[k]) / U for k in names])
    packs = np.array([pack_of(k) for k in names], float)
    fams = [family(k) for k in names]
    levels = sorted(set(fams))
    ref_fam = "h3" if "h3" in levels else levels[0]
    dummies = [f for f in levels if f != ref_fam]

    X = [np.ones(len(names)), packs - PACK_OF_INCUMBENT]
    cols = ["intercept", "pack-187"]
    for f in dummies:
        X.append(np.array([1.0 if g == f else 0.0 for g in fams]))
        cols.append(f"fam[{f}]")
    X = np.column_stack(X)

    # ⚠ THE ROUNDING TERM. The public LB is reported to 1e-5, so every z carries an independent
    # uniform rounding error of width 1e-5: var = (1e-5)^2/12, i.e. 8.333 in units of 1e-12.
    # `gh`'s own fit omits it — at 100+ files a single common intercept barely notices — but a
    # 53-point regression with a within-design slope does, and omitting it would understate the
    # slope's error bar in exactly the direction that makes P3 easier to confirm.
    R = np.eye(len(names)) * (LB_STEP / U) ** 2 / 12.0
    Om = Sxx + R
    iOm = np.linalg.inv(Om)

    def solve(W):
        XtW = X.T @ W
        Vb = np.linalg.inv(XtW @ X)
        b = Vb @ (XtW @ z)
        return b, Vb

    b_g, Vb_g = solve(iOm)
    # OLS with the SAME rounding-inclusive scale, so the GLS/OLS ratio in P4 is a statement about
    # the CORRELATION STRUCTURE and not about two different variance scales.
    s2 = float(np.mean(np.diag(Om)))
    b_o, Vb_o = solve(np.eye(len(names)) / s2)

    print(f"\n  {len(names)} scored pack-tagged files, packs "
          f"{int(packs.min())}..{int(packs.max())}, families {levels} (reference {ref_fam})")
    print(f"  rounding term added to the diagonal: sd {math.sqrt((LB_STEP/U)**2/12):.3f}e-6")
    print(f"\n  {'term':16s} {'GLS':>10s} {'se':>8s} {'OLS':>10s} {'se':>8s} {'se ratio':>9s}")
    for i, c in enumerate(cols):
        sg, so = math.sqrt(Vb_g[i, i]), math.sqrt(Vb_o[i, i])
        print(f"  {c:16s} {b_g[i]:+10.4f} {sg:8.4f} {b_o[i]:+10.4f} {so:8.4f} {sg/so:9.3f}")

    i_d = cols.index("pack-187")
    delta, sd = float(b_g[i_d]), math.sqrt(Vb_g[i_d, i_d])
    r_alpha = math.sqrt(Vb_g[0, 0] / Vb_o[0, 0])
    r_delta = sd / math.sqrt(Vb_o[i_d, i_d])

    # ---- GOODNESS OF FIT. An se of 0.33e-6 on 53 points is set by the COVARIANCE MODEL, not by
    # the scatter, so it is only trustworthy if the scatter agrees with the model. chi2/dof is
    # the direct test; if the data are more dispersed than Omega says, every interval below is
    # too narrow and must be scaled by sqrt(chi2/dof). Reported and APPLIED, never just reported.
    resid = z - X @ b_g
    chi2 = float(resid @ iOm @ resid)
    dof = len(names) - X.shape[1]
    scale = math.sqrt(max(chi2 / dof, 1.0))       # one-sided: never SHRINK an interval on this
    sd_s = sd * scale
    lo, hi = delta - 1.96 * sd_s, delta + 1.96 * sd_s
    print(f"\n  goodness of fit: chi2 = {chi2:.1f} on {dof} dof, chi2/dof = {chi2/dof:.3f}")
    print(f"  => intervals scaled by sqrt(max(chi2/dof, 1)) = {scale:.3f}  "
          f"(one-sided: a well-fitting model never BUYS a narrower interval here)")

    # ---- THE POWER CONTROL. A null is evidence of absence only if the instrument could have
    # seen the effect. Inject a KNOWN slope at the break-even into z and re-fit: the estimator
    # must recover it, and must reject the null at that injection. Without this, "delta ~ 0"
    # is indistinguishable from "this regression cannot see delta at all".
    power = {}
    for inj in (-1.5005, -0.75, +1.5005):
        z_i = z + inj * (packs - PACK_OF_INCUMBENT)
        b_i = np.linalg.inv(X.T @ iOm @ X) @ (X.T @ iOm @ z_i)
        rec = float(b_i[i_d])
        r_i = z_i - X @ b_i
        sc_i = math.sqrt(max(float(r_i @ iOm @ r_i) / dof, 1.0))
        power[f"{inj:+.4f}"] = dict(recovered=rec, err=rec - (delta + inj),
                                    detected=bool(abs(rec) > 1.96 * sd * sc_i))
    print(f"\n  POWER CONTROL — inject a known slope, re-fit, must recover it:")
    for k, v in power.items():
        print(f"    injected {k}  ->  recovered {v['recovered']:+.4f}  "
              f"(error {v['err']:+.2e})  detected at 2sigma: {v['detected']}")
    if any(abs(v["err"]) > 1e-9 for v in power.values()):
        fail("POWER CONTROL: the estimator does not recover an injected slope exactly")
    if not power[f"{-1.5005:+.4f}"]["detected"]:
        fail("POWER CONTROL: the regression CANNOT detect a break-even-sized slope -- "
             "P3's null is uninformative")

    return dict(names=names, n=len(names), cols=cols, gls=b_g.tolist(),
                gls_se=np.sqrt(np.diag(Vb_g)).tolist(), ols=b_o.tolist(),
                ols_se=np.sqrt(np.diag(Vb_o)).tolist(), delta=delta, delta_se=sd,
                chi2=chi2, dof=dof, scale=scale, delta_se_scaled=sd_s,
                ci=(lo, hi), se_ratio_alpha=r_alpha, se_ratio_delta=r_delta, power=power,
                packs=packs.tolist(), z=z.tolist(), families=fams, ref_fam=ref_fam)


def main() -> None:
    write = "--no-write" not in sys.argv
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    LB = load_board()
    print(f"live board: {len(LB)} distinct scored files\n")

    TIER1, TIER2, NAMES63, base63, ref = gate_a(y, beta, LB)
    A, cv = part_a(y, beta, LB, base63, NAMES63, TIER1)
    B = part_b(y, beta, LB)

    # ---- the break-even, from the CV ledger, by the FORMULA fixed in the prereg -------------
    dcv = (cv[CHALLENGER] - cv[INCUMBENT]) / U
    nmem = PACK_OF_CHALLENGER - PACK_OF_INCUMBENT
    d_break = -dcv / nmem
    print("\n" + "=" * 96)
    print("THE BREAK-EVEN, AND P3")
    print("=" * 96)
    print(f"  the move buys {dcv:+.3f}e-6 of CV across {nmem} members")
    print(f"  => it pays iff  {nmem}*delta > {-dcv:.3f}e-6,  i.e.  delta > {d_break:+.4f}e-6/member")
    print(f"  (registered in w64_prereg §2B as -1.5005; recomputed here from the ledger)")
    print(f"\n  GLS delta = {B['delta']:+.4f} e-6 per member,  se {B['delta_se']:.4f} "
          f"-> {B['delta_se_scaled']:.4f} after the chi2/dof scaling")
    print(f"  95% CI    = [{B['ci'][0]:+.4f}, {B['ci'][1]:+.4f}]   (scaled)")
    p3_sign = B["delta"] > d_break
    p3_excl = B["ci"][0] > d_break
    p3_pt = abs(B["delta"]) < 0.30
    print(f"\n  P3a delta > break-even                  -> {'✅' if p3_sign else '🔴'}")
    print(f"  P3b 95% CI EXCLUDES the break-even      -> {'✅' if p3_excl else '🔴'}")
    print(f"  P3c |delta| < 0.30e-6/member (point)    -> {'✅' if p3_pt else '🔴'}")
    if not (p3_sign and p3_excl):
        FALSIFIED.append(f"P3 delta above break-even with the CI excluding it "
                         f"(delta {B['delta']:+.4f}, CI [{B['ci'][0]:+.4f}, {B['ci'][1]:+.4f}], "
                         f"break-even {d_break:+.4f})")
    if not p3_pt:
        FALSIFIED.append(f"P3c |delta| < 0.30 point estimate (read {abs(B['delta']):.4f})")

    print(f"\n  P4a SE_GLS(alpha)/SE_OLS(alpha) > 2.0   -> "
          f"{B['se_ratio_alpha']:.3f}  {'✅' if B['se_ratio_alpha'] > 2.0 else '🔴'}")
    print(f"  P4b SE_GLS(delta)/SE_OLS(delta) < 1.5   -> "
          f"{B['se_ratio_delta']:.3f}  {'✅' if B['se_ratio_delta'] < 1.5 else '🔴'}")
    if not B["se_ratio_alpha"] > 2.0:
        FALSIFIED.append(f"P4a SE ratio on the intercept > 2.0 (read {B['se_ratio_alpha']:.3f})")
    if not B["se_ratio_delta"] < 1.5:
        FALSIFIED.append(f"P4b SE ratio on the slope < 1.5 (read {B['se_ratio_delta']:.3f})")

    p2 = A["gain"] > P2_BAR
    p3 = p3_sign and p3_excl
    agree = (p2 == p3)
    if not agree:
        FALSIFIED.append("P5 the two halves agree")
    print("\n" + "=" * 96)
    print("THE DECISION — the AND rule fixed in w64_prereg §4, before any number above existed")
    print("=" * 96)

    # ⚠⚠ MY OWN PREREG COMMITTED w63 §3's DEFECT, ONE RUN AFTER RECORDING IT. §3 registers P2 as
    # a NUMBER ("> +1.0e-6, STRICTLY") and §4 condition (i) as a DESCRIPTION ("P2 reads positive
    # — E[max] prefers it"). At the measured +0.1444e-6 those are DIFFERENT ANSWERS, so §4(i)
    # registered neither. w63 §3, verbatim: "A PREREG THAT QUOTES A NUMBER AND A DESCRIPTION THAT
    # ARE DIFFERENT OBJECTS HAS REGISTERED NEITHER." Both readings are printed and the
    # CONSERVATIVE one is taken, because choosing the reading after seeing which way it points is
    # the whole failure the rule exists to stop.
    p2_strict = A["gain"] > P2_BAR                     # §3's number
    p2_loose = A["gain"] > 0.0                         # §4's description
    print(f"  ⚠ §4(i) IS AMBIGUOUS AND THAT IS THIS RUN'S OWN DEFECT — see the note in source.")
    print(f"     as the NUMBER      (gain > +{P2_BAR:.1f}e-6): {A['gain']:+.4f} -> "
          f"{'YES' if p2_strict else 'NO'}")
    print(f"     as the DESCRIPTION (gain > 0):          {A['gain']:+.4f} -> "
          f"{'YES' if p2_loose else 'NO'}")
    print(f"     TAKING THE CONSERVATIVE READING: (i) = {'YES' if p2_strict else 'NO'}")
    print(f"\n  (i)  E[max] prefers {CHALLENGER}                       {'YES' if p2 else 'NO'}")
    print(f"  (ii) delta above break-even, CI excluding it            {'YES' if p3 else 'NO'}")
    move = p2 and p3
    print(f"\n  => WANTED slot 2 {'MOVES to ' + CHALLENGER if move else 'DOES NOT MOVE'}")
    if not agree:
        print("  ⚠ THE TWO HALVES DISAGREE. Each is blind where the other sees; the AND rule "
              "means\n    the hedge survives on whichever half says so. Recorded, not narrated away.")
    # AND THE REASON THE DEFERRAL CAN NOW BE CLOSED RATHER THAN RENEWED.
    print(f"\n  ⚠ THE ITEM IS NOT 'STILL OPEN' — IT IS DECIDED AT A MEASURED SIZE. The E[max]")
    print(f"    instrument spans {A['emax_spread']:.4f}e-6 across a ladder spanning "
          f"{max(r['dcv'] for r in A['rows']) - min(r['dcv'] for r in A['rows']):.1f}e-6 of CV, and the")
    print(f"    contrast three runs deferred is {A['gain']:+.4f}e-6 — inside every noise floor on "
          f"file.\n    The structural half reads delta = {B['delta']:+.4f} +- "
          f"{B['delta_se_scaled']:.4f}, {abs((B['delta']-d_break)/B['delta_se_scaled']):.1f} sigma "
          f"clear of the break-even.\n    The hedge is priced at {A['decorr_pct']:.0f}% of what it costs: "
          f"{abs(A['corr_challenger']-A['corr_incumbent']):.6f} of")
    print(f"    decorrelation earns back {A['decorr_earned']:+.3f}e-6 of the {A['decorr_cost']:+.3f}e-6 "
          f"its {dcv:.1f}e-6 CV deficit costs.\n    STOP SPENDING RUNS ON IT.")

    out = dict(
        prereg="w64_prereg.txt", gate_a=dict(passed=True, base=base63, against="w63a_setprice.json"),
        slot1=SLOT1, incumbent=INCUMBENT, challenger=CHALLENGER, also_considered=ALSO,
        beta=beta, part_a=A, part_b={k: v for k, v in B.items() if k != "names"},
        part_b_names=B["names"], dcv=dcv, n_members=nmem, delta_breakeven=d_break,
        predictions=dict(p1=True, p2=bool(p2), p3a=bool(p3_sign), p3b=bool(p3_excl),
                         p3c=bool(p3_pt), p4a=bool(B["se_ratio_alpha"] > 2.0),
                         p4b=bool(B["se_ratio_delta"] < 1.5), p5=bool(agree)),
        falsified=FALSIFIED, decision_move=bool(move), failures=FAILURES,
        p2_strict=bool(p2_strict), p2_loose=bool(p2_loose),
        prereg_defect="w64_prereg SS3 registers P2 as a NUMBER (>+1.0e-6) and SS4(i) as a "
                      "DESCRIPTION ('reads positive'); at +0.1444e-6 they disagree, so SS4(i) "
                      "registered neither. w63 SS3's own rule, committed one run earlier. Both "
                      "readings recorded; the conservative one taken.",
        caveats=[
            "PART A's posterior is Gaussian with one common gap; it cannot see family-level "
            "structural risk, which is what the hedge is actually for. PART B exists for that.",
            "PART B's design is OBSERVATIONAL: pack size was not randomised, and larger packs "
            "are also later builds. Any time trend in transfer loads onto delta.",
            "The LB rounding term is modelled as independent uniform on every file. Two files "
            "that round the same way are not independent, but the correlated part of their "
            "error is already in Sxx.",
        ])
    print(f"\n  falsified: {FALSIFIED if FALSIFIED else 'none'}")
    if write:
        # ⚠ w63 §8: the WRITE is bound to the GATE, not to the readings. GATE A sys.exits on its
        # own, so reaching here means it passed and a design was fitted; every falsified reading
        # is carried in `falsified` where a reader sees it. The DECISION is separately gated in
        # §4 of the prereg, which is the thing that SHOULD be conditional on the readings.
        with open(os.path.join(HERE, "w64a_hedgeprice.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote experiments/w64a_hedgeprice.json")


if __name__ == "__main__":
    main()
