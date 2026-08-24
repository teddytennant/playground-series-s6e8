"""w63a — THE SUCCESSOR PRICER: a DETERMINED pair, and a price for THE SET BEING SENT.

Pre-registration: experiments/w63_prereg.txt, committed 4737c3b BEFORE this file existed.

WHY THIS FILE EXISTS.
The 08-23 sends moved auto-slot 1 from five files at 0.97118 to TWO at 0.97119. Every instrument
on the send path was derived on the old tier and every one of them now refuses:

    w57a_tierprice2.py   assert PICK in TIER1              -> refuses
    w58a_tiergate.py     GATE T                            -> exits 2
    w59a_hijackprice.py  GATE T                            -> refuses
    w26g_send.py         hijack_cv_bar -> None             -> BLOCKS EVERY ABOVE-TIER FILE

⛔ w62 §7.4: do not soften the assertion, do not edit w57a/w58a/w59a, and do not make the bar
pass by any route other than re-deriving it. THE FIX IS THE PRICER. This is that pricer.

TWO THINGS THIS DOES THAT w59a DOES NOT.

1. IT PRICES A DETERMINED PAIR. w59a's status quo was a uniform draw over the C(5,2) pairs of a
   five-file tie (MODEL B, 7.855e-6) and a hijacker displaced one uniform draw from the tie it
   left behind. Auto-slot 1 now holds exactly two files, so the status quo is ONE pair priced
   exactly (4.523e-6, w62a) and a hijacker displaces a draw from a two-element set.

2. IT PRICES THE SET. w62 §4: w60 applied `cost_hijack(X)` independently to two files and sent
   both on the same day. Realised gain +1.235e-6 against a registered +6.730e-6, because the
   SECOND clearer took the slot the first one would otherwise have drawn from the tie — and that
   tie contained the PICK. ⚠⚠ A PER-FILE HIJACK PRICE IS NOT ADDITIVE AND ITS SIGN CAN INVERT
   WHEN TWO PRICED FILES ARE SENT TOGETHER. P5 below turns that autopsy into a property of the
   instrument: the additive price is optimistic BY CONSTRUCTION, and the gap is exactly the
   crossed-minus-matched E[max] difference.

THE ESTIMATOR IS w59a's, COPIED AND ONLY PARAMETERISED — specifically w59a's UNSCORED-CAPABLE
variant, because a candidate for a FUTURE send has no LB and `w62a.build()` requires one for
every name (it forms `z` over all of NAMES). GATE W is what proves the copy is faithful rather
than asserting it: run on w62a's own eight-file board it must reproduce `cost_auto_pair`, all
three `solo` costs and all six `ladder` entries to < 1e-9.

    .venv/bin/python experiments/w63a_setprice.py            # prices, writes the artefact
    .venv/bin/python experiments/w63a_setprice.py --no-write # what-if, artefact NOT written
"""
from __future__ import annotations

import io, itertools, json, math, os, subprocess, sys

import numpy as np
import pandas as pd
from scipy.stats import norm

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
CORR_CONS = 0.9923594211711959

# the sender's own constants, imported in spirit and asserted below against the module itself so
# a drift in either place is caught here rather than on a send day.
PRED_SD, STEP, P_MAX = 8.77e-6, 1e-5, 0.02

W62A_COST = 4.522768563707359     # the determined price this must reproduce and then extend
W59A_LEVERAGE = 3.0               # "3x", measured at n = 5 and recorded as if it were constant

# ⚠⚠ THE SUPERSEDED BAR IS READ FROM THE SUPERSEDED ARTEFACT, NEVER RE-TYPED. `w63_prereg.txt`
# registered P4 as "H_det < 12.97e-6 ... TIGHTER than w59a's H_binding" and 12.97 IS NOT w59a's
# H_binding. It is `W58_D`, the DILUTION break-even, which coincides with w59a's UNCONDITIONAL
# crossing (12.965); w59a's binding bar is the CONDITIONAL one, 10.590, and `hijack_cv_bar`'s
# own docstring says so in as many words ("The unconditional break-even (12.97e-6) is
# anti-conservative; the conditional one (10.59e-6) is the bar"). A prereg that quotes a NUMBER
# and a DESCRIPTION that are different objects has registered neither. Read, do not re-type.
_W59A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "w59a_hijackprice.json")))
W59A_H_UNCOND = float(_W59A["H_uncond"])        # 12.965...  what "12.97" actually names
W59A_H_BINDING = float(_W59A["H_binding"])      # 10.590...  what P4 said it was comparing to
W59A_BAR_CV = float(_W59A["cv_bar_new"])
W63_PREREG_P4_CONST = 12.97                     # the number as registered, kept for the record

# ---- THE REGISTERED 2026-08-24 TEN (w63_prereg.txt) -------------------------------------------
# The ten highest-CV unsent, non-vetoed, non-member, non-w40d-ineligible files in the queue.
# ⚠ Three of them (`w40_ad211std_rescale`, `w38_ad202std_rescale`, `w27_ad188stdcorr`) were
# BLOCKED on 08-23 by the sender's P_MAX hijack-risk gate at the OLD tier and are unblocked at
# the new one — the tier rising a display step moved their P(above) from 0.082/0.061/0.053 to
# 0.006/0.004/0.003. That is a consequence of the lever clearing that nobody registered.
PLAN_0824 = [
    "w29_ad194std_h3", "w27_ad188stdcorr", "w34_ad195std", "w34_ad196std",
    "w40_ad211std_hybrid", "w34_ad195std_w", "w34_ad195std_wh3", "w34_ad196std_h3",
    "w38_ad202std_rescale", "w40_ad211std_rescale",
]

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


# ------------------------------------------------------------------------------------------
# THE ESTIMATOR — w59a_hijackprice.main() lines 145-205, lifted to a function with (NAMES, LB,
# cv) as arguments and NOTHING ELSE CHANGED. Unscored names are allowed: the GLS common gap is
# fitted over the SCORED subset only and an unscored file's own residual enters as zero, which
# is exactly what w59a does for the files it prices that have never been submitted.
# ------------------------------------------------------------------------------------------
def fit(NAMES, LB, cv, V, y, beta, corr=CORR_CONS):
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    N = len(NAMES)
    A = np.empty((N, n1)); B = np.empty((N, n0))
    for i, k in enumerate(NAMES):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])
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
    Kmap = beta * np.eye(N) + (1 - beta) * Mm
    Cpri = (1 - beta) ** 2 * (St - Mm @ Sxx @ Mm.T)
    Cpri = 0.5 * (Cpri + Cpri.T)

    scored = [k for k in NAMES if k in LB]
    js = [col[k] for k in scored]
    z = np.array([LB[k] - cv[k] for k in scored]) / U
    D1 = np.ones((len(js), 1))
    iS = np.linalg.inv(Sxx[np.ix_(js, js)])
    Vg = np.linalg.inv(D1.T @ iS @ D1)
    gh = float(np.ravel(Vg @ (D1.T @ iS @ z))[0])
    gamma = beta + (1 - beta) * float(np.diag(Mm).mean())

    xh0 = np.zeros(N)
    for k in NAMES:
        xh0[col[k]] = (LB[k] - cv[k]) / U - gh if k in LB else 0.0
    cov = Cpri + Kmap @ np.ones((N, 1)) @ Vg @ np.ones((1, N)) @ Kmap.T
    rsd = abs(beta) * np.sqrt(np.diag(Sp)) / abs(corr) * np.sqrt(max(1 - corr ** 2, 0.0))
    iw = [col[k] for k in WANTED]

    def make(mu):
        def emax_set(files):
            ia = [col[f] for f in files]
            ma = np.array([mu[i] + cv[NAMES[i]] / U for i in ia])
            if len(ia) == 1:
                return float(ma[0])
            sa = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in ia])
            ra = cov[ia[0], ia[1]] / np.sqrt(cov[ia[0], ia[0]] * cov[ia[1], ia[1]])
            return float(emax(ma, sa, ra))

        def price(files):
            """cost of NOT clicking = E[max over WANTED] - E[max over whatever gets selected]."""
            mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
            sw = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in iw])
            rw = cov[iw[0], iw[1]] / np.sqrt(cov[iw[0], iw[0]] * cov[iw[1], iw[1]])
            return float(emax(mw, sw, rw) - emax_set(files))
        return price, emax_set

    price0, emax0 = make(Kmap @ xh0)

    def cond_price(cleared, thresh, s=PRED_SD):
        """w59a's truncated-normal conditioning, applied to EVERY file that cleared.

        A hijack is by definition the branch where the file landed high on the public slice, and
        gamma is NEGATIVE, so conditioning on that landing LOWERS its expected private score.
        w59a conditioned one file at a time because it priced one file at a time; the whole
        point of this run is that a day sends a SET, so every member of the cleared set is
        conditioned on its own landing simultaneously.
        """
        xh = xh0.copy()
        for k in cleared:
            mpred = cv[k] + gh * U
            zc = (thresh - mpred) / s
            lam = norm.pdf(zc) / max(norm.sf(zc), 1e-300)
            xh[col[k]] = (s * lam) / U
        return make(Kmap @ xh)[0]

    return dict(price=price0, emax_set=emax0, cond_price=cond_price, col=col,
                gap=gh, gamma=gamma, Sxx=Sxx, cov=cov, rsd=rsd, names=NAMES)


def _load(names, y):
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    return V, {k: fast_auc(y, V[k]) for k in names}


def gate_w(y, beta):
    """The verbatim control: reproduce w62a's determined board, exactly, before pricing a new one."""
    print("=" * 94)
    print("GATE W — the unscored-capable estimator must reproduce w62a_autopair.json EXACTLY")
    print("=" * 94)
    ref = json.load(open(os.path.join(HERE, "w62a_autopair.json")))
    assert ref["determined"] and ref["gate_v"], "w62a's own artefact does not record a passed run"
    rNAMES = sorted(ref["cv"])
    rLB = {k: float(v) for k, v in ref["lb"].items()}
    rV, rcv = _load(rNAMES, y)
    for k in rNAMES:                       # CV recomputed from the OOF, never re-quoted
        if abs(rcv[k] - float(ref["cv"][k])) > 5e-10:
            fail(f"GATE W: CV of {k} does not reproduce ({rcv[k]:.12f} vs {ref['cv'][k]:.12f})")
    E = fit(rNAMES, rLB, rcv, rV, y, beta)
    p = E["price"]
    A, Bf = ref["tier1"]
    worst, ncheck = 0.0, 0

    def cmp(label, got, want):
        nonlocal worst, ncheck
        ncheck += 1
        worst = max(worst, abs(got - want))
        if abs(got - want) > 1e-9:
            fail(f"GATE W: {label} {got:+.12f} != recorded {want:+.12f}")

    cmp("cost_auto_pair", p([A, Bf]), float(ref["cost_auto_pair"]))
    for k, rec in ref["solo"].items():
        cmp(f"solo[{k}]", p([k]), float(rec["cost"]))
    old_t1 = [k for k in json.load(open(os.path.join(HERE, "w57a_tierprice2.json")))
              ["tiers"]["slot1"] if k in rNAMES]
    lad = dict(ref["ladder"])
    cmp("ladder[neither_cleared]",
        float(np.mean([p(list(q)) for q in itertools.combinations(old_t1 + [A, Bf], 2)])),
        float(lad["neither_cleared"]))
    for x, other in ((A, Bf), (Bf, A)):
        cmp(f"ladder[only_{x}]",
            float(np.mean([p([x, zz]) for zz in old_t1 + [other]])), float(lad["only_" + x]))
    cmp("ladder[both_cleared_ACTUAL]", p([A, Bf]), float(lad["both_cleared_ACTUAL"]))
    for x in (A, Bf):
        cmp(f"ladder[click_pick_plus_{x}]", p([PICK, x]), float(lad["click_pick_plus_" + x]))
    print(f"  {len(rNAMES)} files, {ncheck} recorded prices re-derived")
    print(f"  worst absolute deviation from w62a's artefact: {worst:.3e}   (bar 1e-9)")
    if FAILURES:
        print("\n⛔ GATE W FAILED — this is not w59a's estimator on w62a's board. REFUSING.")
        sys.exit(1)
    print("  ✅ GATE W PASSED — P1 CONFIRMED. The copy is faithful; now price the new board.")
    return worst, ncheck


def main(fill=(), outfile="w63a_setprice.json") -> dict:
    """Price the day. `fill` and `outfile` were added by w79 (prereg experiments/w79_prereg.txt,
    committed 0109708 before the caller existed) and BOTH DEFAULT TO THE ORIGINAL BEHAVIOUR:
    main() with no arguments prices w63a's own design and writes w63a_setprice.json, exactly as
    before. ⚠ That does NOT mean the file it writes today is byte-identical to the one on disk —
    the board has moved since, and w78 measured the move as +0.334e-6 of bar. The stored artefact
    is what w76a/w77a/w77b/w78a pin against, so w79 does not overwrite it: it passes an `outfile`
    instead, and `w79b_fillguard` checks the stored file's md5 is untouched. See the FILL note at
    the design block for what the fillers do and, more to the point, what they do not touch."""
    # ⚠⚠ w79. `FAILURES` is a MODULE global that `fail()` increments and GATE W refuses on. It
    # was written for a file that runs once as a script; the moment main() is callable twice in
    # one process — which w79a does, three arms on one board — the second call inherits the
    # first's count and GATE W refuses AFTER PRINTING "worst deviation 0.000e+00". Reset it here
    # so the counter means what its name says: failures in THIS pricing run.
    global FAILURES
    FAILURES = 0
    write = "--no-write" not in sys.argv
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    # the sender's constants must be the sender's, not a second copy of them
    import w26g_send as SND
    for nm, mine, theirs in (("PRED_SD", PRED_SD, SND.PRED_SD), ("STEP", STEP, SND.STEP),
                             ("P_MAX", P_MAX, SND.P_MAX)):
        assert mine == theirs, f"{nm} drifted: this file {mine}, w26g_send {theirs}"

    worst_w, ncheck_w = gate_w(y, beta)

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
    n_tier = len(TIER1)
    print("\n" + "=" * 94)
    print(f"LIVE BOARD: {len(sub)} scored submissions, {len(agg)} distinct files")
    print("=" * 94)
    print(f"  auto-slot tier  public {t1v:.5f}: {n_tier} files {TIER1}")
    print(f"  next tier down  public {t2v:.5f}: {len(TIER2)} files {TIER2}")

    # ⚠ GATE T2. w62's lesson: a STAMP is not a COMPARISON. This compares the live tier against
    # the tier w62a was DERIVED on, as a SET, every time it runs -- and refuses when they differ
    # instead of stamping itself PASS and letting a later reader trust the stamp.
    rec_tier = sorted(json.load(open(os.path.join(HERE, "w62a_autopair.json")))["tier1"])
    if sorted(TIER1) != rec_tier:
        print(f"\n⛔ GATE T2: the live auto-slot-1 set {sorted(TIER1)} is not the set w62a priced "
              f"{rec_tier}.\n   GATE W above validated the estimator against a board that has "
              f"moved. Re-run w62a first.\n   REFUSING.")
        sys.exit(2)
    if n_tier != 2:
        print(f"\n⛔ TIER 1 HOLDS {n_tier} FILES — not a determined pair. w57a's MODEL B (>=3) or "
              f"MODEL A (==1)\n   is the right instrument, not this one. REFUSING.")
        sys.exit(2)
    A, Bf = TIER1
    print(f"\n  ✅ GATE T2: determined pair {A} + {Bf}, unmoved since w62a.")

    # ------------------------------------------------------------------ the enlarged design
    CAND = [c for c in PLAN_0824 if c not in TIER1]
    BASE_NAMES = sorted(set(TIER1 + TIER2 + list(WANTED) + CAND))
    # ⚠ w79. FILL joins the DESIGN and is WITHHELD FROM THE GLS. `fit` reads its scored set as
    # `[k for k in NAMES if k in LB]`, so a name that is in NAMES but absent from the dict handed
    # to it gets xh0 = 0 and is priced through the coupling -- which is already exactly what this
    # file does with an unsent plan candidate. NO LINE OF `fit` CHANGES; only what it is handed.
    # The fillers exist to put candidates INSIDE the 16.5e-6 bracket hole the break-even is
    # otherwise interpolated across (w63 §5a, w77a, w78). FILL is empty by default.
    FILL = sorted(set(fill) - set(BASE_NAMES))
    NAMES = sorted(set(BASE_NAMES) | set(FILL))
    LB_FIT = {k: v for k, v in LB.items() if k not in set(FILL)}
    missing = [k for k in NAMES if not os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))]
    assert not missing, f"no OOF vector for {missing} -- cannot price them, do not guess"
    V, cv = _load(NAMES, y)
    E = fit(NAMES, LB_FIT, cv, V, y, beta)
    price, gap, gamma = E["price"], E["gap"], E["gamma"]
    base = price([A, Bf])
    print(f"\n=== the enlarged design: {len(NAMES)} files "
          f"({len(TIER1)} tier1 + {len(TIER2)} tier2 + {len(CAND)} candidates + WANTED"
          f"{f' + {len(FILL)} FILL' if FILL else ''}) ===")
    print(f"  GLS common gap G = {gap:+.2f}e-6 over {sum(k in LB_FIT for k in NAMES)} scored; "
          f"gamma = {gamma:+.6f}")
    if FILL:
        print(f"  {len(FILL)} hole-fillers withheld from the GLS: {FILL}")

    # ------------------------------------------------------------------ GATE R
    dR = abs(base - W62A_COST)
    print(f"\n  GATE R: determined price on the enlarged design {base:.12f} vs w62a "
          f"{W62A_COST:.12f}\n          |d| = {dR:.3e}   (bar 1.0e-6)")
    if dR > 1.0e-6:
        print("⛔ GATE R FAILED: the enlarged design moved w62a's headline. Refusing.")
        sys.exit(3)
    print("          -- PASS. P2 CONFIRMED.")

    thresh = t1v + STEP / 2
    worthless = float(np.mean([price([d]) for d in TIER1]))
    print(f"\n  status quo (nothing clears, the DETERMINED pair): {base:+.4f}e-6")
    print(f"  worthless-hijacker limit (X contributes nothing):  {worthless:+.4f}e-6")
    print(f"  the whole usable range is therefore {worthless - base:.4f}e-6 wide — "
          f"it was {'much' if worthless - base < 3 else ''} wider at n = 5")

    # ------------------------------------------------------------------ per-file hijack table
    def only(x, pfun=None):
        """X clears ALONE: auto pair = {X, one uniform draw from the determined pair}."""
        p = pfun or price
        return float(np.mean([p([x, d]) for d in TIER1 if d != x]))

    # pred_lb comes from the queue the SENDER reads, not from a second prediction made here --
    # w47 §"THE CV->LB PREDICTOR" is the one authority for that number and re-deriving it would
    # give the set price a different LB model from the gate that admits the files.
    PRED = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv")).set_index("file")["pred_lb"]
    missing_pred = [c for c in CAND if f"{c}.csv" not in PRED.index]
    assert not missing_pred, f"{missing_pred} have no priced pred_lb in the queue -- do not guess"

    print("\n=== HIJACK, one file at a time (this is the price w60 applied TWICE) ===")
    print(f"  {'candidate':24s} {'dCV':>8s} {'uncond':>9s} {'cond':>9s} {'P(above)':>9s}  verdict")
    rows = []
    for x in sorted(set(CAND + TIER2 + [PICK] + FILL), key=lambda k: -cv[k]):
        cu = only(x)
        cc = only(x, E["cond_price"]([x], thresh))
        pl = float(PRED[f"{x}.csv"]) if x in CAND else float("nan")
        pa = (0.5 * (1 - math.erf(((thresh - pl) / PRED_SD) / math.sqrt(2)))
              if pl == pl else float("nan"))
        rows.append(dict(stem=x, cv=cv[x], dcv=(cv[x] - cv[PICK]) / U, uncond=cu, cond=cc,
                         pred_lb=pl, p_above=pa, where=("tier2" if x in TIER2 else "")
                         + (" plan0824" if x in CAND else "")
                         + (" fill" if x in FILL else ""), helps=bool(cu < base)))
        print(f"  {x:24s} {rows[-1]['dcv']:+8.2f} {cu:+9.3f} {cc:+9.3f} "
              f"{pa if pa == pa else float('nan'):9.4f}  {'HELPS' if cu < base else 'HURTS'}"
              f"  {rows[-1]['where'].strip()}")

    # ------------------------------------------------------------------ GATE J
    # w59a's GATE I generalised. Adding X to a tie of n takes the in-tier price from a mean over
    # C(n,2) pairs to a mean over C(n+1,2), and the n NEW pairs are exactly {X,d} for d in the
    # tie -- which is what `only(X)` already averages. So for every n,
    #
    #     delta_dilution(X) = [C(n,2)*base + n*only(X)]/C(n+1,2) - base
    #                       = n * (only(X) - base) / C(n+1,2)
    #
    # ⚠⚠ THE BREAK-EVEN IS INDEPENDENT OF n (delta = 0 <=> only(X) = base, always), BUT THE
    # LEVERAGE IS NOT. C(n+1,2)/n is 3 at n = 5 and 1.5 at n = 2. w59a's "3x" is not a constant
    # of the problem; it is a reading of a tier size that no longer holds.
    print(f"\n=== GATE J: delta_dilution(X) == n*(only(X)-base)/C(n+1,2) at the LIVE n = {n_tier} ===")
    lev = math.comb(n_tier + 1, 2) / n_tier
    worstJ = 0.0
    for r in rows:
        x = r["stem"]
        if x in TIER1:
            continue
        nb = float(np.mean([price(list(q))
                            for q in itertools.combinations(sorted(TIER1 + [x]), 2)]))
        pred = n_tier * (r["uncond"] - base) / math.comb(n_tier + 1, 2)
        worstJ = max(worstJ, abs((nb - base) - pred))
    print(f"  max |delta_dilution - n*(only-base)/C(n+1,2)| over {len(rows)} files = {worstJ:.3e}")
    if worstJ > 1e-9:
        print("⛔ GATE J FAILED: the identity does not hold and every leverage claim below is wrong.")
        sys.exit(5)
    print(f"  -- PASS. The break-even is UNCHANGED by n (delta=0 <=> only(X)=base for every n),")
    print(f"     so 'one bar governs both landings' SURVIVES. The LEVERAGE does not: "
          f"C(n+1,2)/n = {lev:.1f}x\n     at n = {n_tier}, against the {W59A_LEVERAGE:.0f}x "
          f"w59a measured at n = 5 and the sender still prints.")

    # ------------------------------------------------------------------ the break-even
    # w59a's method verbatim: ADJACENT brackets only. Its first cut interpolated across the whole
    # 40e-6 range and read 17.79 where the adjacent pair read 12.97 -- extrapolation dressed as
    # a bracket. Every further crossing is reported, never swallowed.
    def crossings(key):
        rs = sorted(rows, key=lambda r: -r["dcv"])
        out = []
        for a, b in zip(rs, rs[1:]):
            if (a[key] < base) != (b[key] < base) and a[key] != b[key]:
                t = (base - a[key]) / (b[key] - a[key])
                out.append((-(a["dcv"] + t * (b["dcv"] - a["dcv"])), a, b))
        return out

    print("\n=== THE DETERMINED-PAIR BREAK-EVEN (adjacent brackets only) ===")
    Hs, brackets = {}, {}
    for key, nm in (("uncond", "UNCONDITIONAL"), ("cond", "CONDITIONAL@8.77")):
        cr = crossings(key)
        Hs[key] = cr[0][0] if cr else None
        if not cr:
            print(f"  {nm}: no adjacent straddle in the candidate set -- NOT REPORTED")
            continue
        h, hi, lo = cr[0]
        brackets[key] = dict(hi=hi["stem"], hi_dcv=hi["dcv"], lo=lo["stem"], lo_dcv=lo["dcv"],
                             width=hi["dcv"] - lo["dcv"])
        print(f"  {nm:16s} H = {h:6.2f}e-6  -> HELPS iff CV >= {cv[PICK] - h*U:.10f}"
              f"   [bracket {hi['stem']} {hi['dcv']:+.2f} .. {lo['stem']} {lo['dcv']:+.2f}]")
        for extra in cr[1:]:
            print(f"    ⚠ FURTHER crossing at {extra[0]:.2f}e-6 "
                  f"({extra[1]['stem']}..{extra[2]['stem']}) — the cost curve is not monotone")
    H_bind = min([h for h in Hs.values() if h is not None], default=None)
    bar_new = (cv[PICK] - H_bind * U) if H_bind is not None else None
    if H_bind is None:
        print("  ⛔ no break-even could be bracketed — the artefact will carry cv_bar_new = null "
              "and\n     the sender will keep blocking every above-tier file. That is the correct "
              "failure.")
    else:
        bw = max(b["width"] for b in brackets.values())
        print(f"\n  BINDING (the stricter of the two): H = {H_bind:.2f}e-6, "
              f"CV bar {bar_new:.10f}")
        if FILL:
            print(f"  ✅ THE BRACKET IS {bw:.3f}e-6 WIDE AND THE HOLE IS FILLED. {len(FILL)} "
                  f"candidates were added\n     to the design INSIDE it, so the crossing is "
                  f"MEASURED between adjacent files rather than\n     interpolated across a gap. "
                  f"The GLS is untouched — the fillers are withheld from `LB`\n     and carry "
                  f"xh0 = 0 — and the population came from a rule registered in w79_prereg.txt\n"
                  f"     BEFORE the board was read. That last clause is what separates this from "
                  f"w59a's\n     'choosing the bar by choosing the bracket'.")
        else:
            print(f"  ⚠⚠ THE BRACKET IS {bw:.1f}e-6 WIDE AND NOTHING SITS INSIDE IT. The crossing is "
                  f"linearly\n     interpolated across a hole in the design: every file between "
                  f"dCV {brackets['cond']['hi_dcv']:+.1f}\n     and {brackets['cond']['lo_dcv']:+.1f} "
                  f"was SENT on 08-23 and is now scored, and every unsent file is\n     below the "
                  f"hole. This is the dominant uncertainty in the bar and it is NOT reduced by\n"
                  f"     adding files to the candidate set by hand — w59a: 'persisting a bar derived "
                  f"from\n     EXTRA_CAND would be choosing the bar by choosing the bracket.'")

    # ⚠ LIKE-FOR-LIKE, and NOT what w63_prereg.txt registered. See W59A_H_BINDING above.
    print("\n=== P4, like-for-like against w59a's OWN columns ===")
    print(f"  {'column':18s} {'w59a (n=5 tie)':>15s} {'w63a (n=2 pair)':>16s} {'move':>9s}")
    for key, nm, old in (("uncond", "UNCONDITIONAL", W59A_H_UNCOND),
                         ("cond", "CONDITIONAL@8.77", W59A_H_BINDING)):
        if Hs.get(key) is not None:
            d = Hs[key] - old
            print(f"  {nm:18s} {old:15.2f} {Hs[key]:16.2f} {d:+9.2f}  "
                  f"{'LOOSER' if d > 0 else 'TIGHTER'}")
    print(f"  superseded CV bar {W59A_BAR_CV:.10f} -> "
          f"{bar_new:.10f} ({(bar_new - W59A_BAR_CV)/U:+.2f}e-6)"
          if bar_new is not None else "")

    # ------------------------------------------------------------------ P5: NON-ADDITIVITY
    # The whole reason this file exists. add(X,Y) is w60's arithmetic, reproduced exactly.
    print("\n" + "=" * 94)
    print("P5 — IS THE PER-FILE PRICE ADDITIVE? (w62 §4, as a property of the instrument)")
    print("=" * 94)
    pairs, gaps = [], []
    for x, yq in itertools.combinations(sorted(set(CAND + TIER2), key=lambda k: -cv[k]), 2):
        add = only(x) + only(yq) - base
        joint = price([x, yq])
        pairs.append(dict(x=x, y=yq, add=add, joint=joint, gap=joint - add))
        gaps.append(joint - add)
    gaps = np.array(gaps)
    n_pos = int((gaps > 0).sum())
    print(f"  {len(pairs)} candidate pairs.  joint - add:  min {gaps.min():+.4f}  "
          f"mean {gaps.mean():+.4f}  max {gaps.max():+.4f} e-6")
    print(f"  joint > add for {n_pos}/{len(pairs)} pairs")
    worstp = min(pairs, key=lambda d: d["gap"])
    print(f"  tightest pair: {worstp['x']} + {worstp['y']}  "
          f"add {worstp['add']:+.4f}  joint {worstp['joint']:+.4f}  gap {worstp['gap']:+.4f}")

    # ⚠⚠ POST-HOC, AND LABELLED AS SUCH. The strict form of P5 is FALSIFIED and this partition
    # was written AFTER seeing which pairs broke it. It is a refinement to be registered and
    # tested by a LATER run, not evidence for anything here.
    #
    # The rearrangement argument in w63_prereg.txt assumed spread and forgot DOMINANCE. Write
    # gap = ½[M(X,A)+M(X,B)+M(Y,A)+M(Y,B)] - M(X,Y) - M(A,B). If X dominates both A and B then
    # M(X,·) ~ M(X) throughout and the expression collapses to ½[M(Y,A)+M(Y,B)] - M(A,B), which
    # is NEGATIVE whenever Y is in turn dominated by both — a max is never below an average.
    # So the additive price is optimistic in the regime that actually matters (both candidates
    # weaker than the pair) and pessimistic exactly when one candidate is STRONGER than both
    # members of the pair, i.e. when it is the PICK or something like it.
    helps = {r["stem"]: bool(r["uncond"] < base) for r in rows}
    grp = {}
    for d in pairs:
        k = (helps[d["x"]], helps[d["y"]])
        k = (k[0] and k[1], k[0] or k[1])          # (both help, at least one helps)
        grp.setdefault("both help" if k[0] else ("exactly one helps" if k[1] else "neither helps"),
                       []).append(d["gap"])
    print("\n  POST-HOC partition by whether a member BEATS the status quo pair "
          "(only(X) < base) —\n  written after the falsification, registered for a later run, "
          "evidence for nothing here:")
    for nm in ("both help", "exactly one helps", "neither helps"):
        g = np.array(grp.get(nm, []))
        if g.size:
            print(f"    {nm:20s} n {g.size:4d}   gap>0 {int((g>0).sum()):4d}/{g.size:<4d}  "
                  f"min {g.min():+8.4f}  mean {g.mean():+8.4f}")
    print("\n  ⚠ THE ADDITIVE PRICE IS OPTIMISTIC. add() credits each file with the full benefit"
          "\n    of displacing a tier-1 file, but there is only ONE such displacement to sell: "
          "the\n    second clearer displaces the OTHER tier-1 file and cannot re-earn it. "
          "Algebraically\n    joint - add = ½[M(X,A)+M(X,B)+M(Y,A)+M(Y,B)] - M(X,Y) - M(A,B), "
          "i.e. CROSSED minus\n    MATCHED pairings of the same four files, and E[max] pays "
          "almost nothing for pairing\n    two strong files together. w60 registered +6.730e-6 "
          "and realised +1.235e-6 (w62 §4).")

    # ------------------------------------------------------------------ THE SET PRICE
    # ⚠ THE INSTRUMENT THIS RUN EXISTS TO BUILD. Enumerate the joint landing outcome of the whole
    # day, not one file at a time. Each candidate lands ABOVE the tier, exactly AT it, or below,
    # independently, with probabilities from the sender's own PRED_SD Gaussian. Then:
    #     |above| >= 2 -> the pair is a uniform draw over the 2-subsets of the clearers
    #     |above| == 1 -> {the clearer} + a uniform draw from the tier-1 bucket
    #     |above| == 0 -> a uniform draw over the 2-subsets of the tier-1 bucket, which is the
    #                     determined pair itself when nothing ties either
    # ⚠ APPROXIMATION, STATED: every above-tier landing is pooled into ONE bucket, so two files
    # that both clear are treated as tied rather than ordered. The finer model would take the top
    # two by public score, and since gamma < 0 that selects the WORSE private posterior — so this
    # pooling is mildly ANTI-conservative, not conservative. Recorded, not hidden.
    def resolve(above, at):
        if len(above) >= 2:
            return list(itertools.combinations(sorted(above), 2))
        bucket = sorted(set(TIER1) | set(at))
        if len(above) == 1:
            return [(above[0], d) for d in bucket]
        return list(itertools.combinations(bucket, 2))

    print("\n" + "=" * 94)
    print(f"THE SET PRICE — the whole 2026-08-24 ten priced as ONE day, not ten files")
    print("=" * 94)
    live = [r for r in rows if r["stem"] in CAND and r["p_above"] == r["p_above"]]
    print(f"  {'file':24s} {'cv':>14s} {'pred_lb':>9s} {'P(above)':>9s} {'P(at)':>8s}  "
          f"P_MAX gate")
    P = []
    for r in sorted(live, key=lambda d: -d["p_above"]):
        lo = (thresh - STEP - r["pred_lb"]) / PRED_SD
        p_at = float(norm.cdf((thresh - r["pred_lb"]) / PRED_SD) - norm.cdf(lo))
        P.append((r["stem"], r["p_above"], p_at))
        print(f"  {r['stem']:24s} {r['cv']:.10f} {r['pred_lb']:9.6f} {r['p_above']:9.4f} "
              f"{p_at:8.4f}  {'pass' if r['p_above'] <= P_MAX else '⛔ BLOCKED'}")
    p_none_above = float(np.prod([1 - pa for _, pa, _ in P]))
    print(f"\n  P(no file of the ten clears the tier) = {p_none_above:.6f}   "
          f"P(at least one) = {1-p_none_above:.6f}")

    cache = {}

    def pair_cost(pr, cleared):
        key = (tuple(sorted(cleared)), tuple(sorted(pr)))
        if key not in cache:
            cache[key] = (price(list(pr)) if not cleared
                          else E["cond_price"](cleared, thresh)(list(pr)))
        return cache[key]

    def set_price(members, conditional=True):
        tot = 0.0
        for pattern in itertools.product((0, 1, 2), repeat=len(members)):
            pr = 1.0
            above, at = [], []
            for (stem, pa, pat), s in zip(members, pattern):
                if s == 0:
                    pr *= pa; above.append(stem)
                elif s == 1:
                    pr *= pat; at.append(stem)
                else:
                    pr *= max(1 - pa - pat, 0.0)
            if pr < 1e-12:
                continue
            cl = above if conditional else []
            tot += pr * float(np.mean([pair_cost(q, tuple(cl)) for q in resolve(above, at)]))
        return tot

    sp_cond = set_price(P, True)
    sp_unc = set_price(P, False)
    # the additive counterfactual: w60's arithmetic applied to the whole day
    add_day = base + sum(pa * (only(s) - base) for s, pa, _ in P)
    print(f"\n  status quo, nothing sent                 {base:+.4f}e-6")
    print(f"  SET price of the registered ten (cond)   {sp_cond:+.4f}e-6   "
          f"delta {sp_cond - base:+.4f}")
    print(f"  SET price of the registered ten (uncond) {sp_unc:+.4f}e-6   "
          f"delta {sp_unc - base:+.4f}")
    print(f"  the ADDITIVE price of the same ten       {add_day:+.4f}e-6   "
          f"delta {add_day - base:+.4f}   <-- w60's arithmetic")
    print(f"  additive error on the day                {sp_unc - add_day:+.4f}e-6")

    # ------------------------------------------------------------------ registered predictions
    print("\n" + "=" * 94)
    print("THE REGISTERED PREDICTIONS (w63_prereg.txt)")
    print("=" * 94)
    p1 = bool(worst_w <= 1e-9)
    p2 = bool(dR <= 1.0e-6)
    p3 = bool(worstJ <= 1e-9 and abs(lev - 1.5) < 1e-12)
    # ⚠ P4 IS SCORED ON WHAT IT MEANT, NOT ON THE CONSTANT IT MISQUOTED. The registered text is
    # "H_det < 12.97e-6, STRICTLY -- the determined-pair bar is TIGHTER than w59a's H_binding",
    # and 12.97 is NOT w59a's H_binding (10.590) but its UNCONDITIONAL crossing. Scoring the
    # literal inequality would compare this run's CONDITIONAL bar against w59a's UNCONDITIONAL
    # one and call a LOOSENING a tightening. Both readings are printed; the LIKE-FOR-LIKE one
    # is what P4 is scored on, and it FAILS.
    p4_literal = bool(H_bind is not None and H_bind < W63_PREREG_P4_CONST)
    p4 = bool(H_bind is not None and Hs.get("cond") is not None
              and Hs["cond"] < W59A_H_BINDING and Hs["uncond"] < W59A_H_UNCOND)
    p5 = bool(n_pos == len(pairs) and gaps.mean() > 0)
    p6 = bool((1 - p_none_above) < 0.05)
    checks = [
        (f"P1 GATE W reproduces w62a's {ncheck_w} prices to < 1e-9", p1, f"worst {worst_w:.3e}"),
        ("P2 GATE R: enlarged design within 1.0e-6 of 4.522769", p2, f"|d| {dR:.3e}"),
        (f"P3 GATE J holds and the leverage is 1.5x, not {W59A_LEVERAGE:.0f}x", p3,
         f"worst {worstJ:.3e}, C(n+1,2)/n = {lev:.3f}"),
        ("P4 the bar TIGHTENS, like-for-like on BOTH columns", p4,
         f"cond {Hs.get('cond'):.2f} vs {W59A_H_BINDING:.2f}, "
         f"uncond {Hs.get('uncond'):.2f} vs {W59A_H_UNCOND:.2f}"
         if H_bind is not None else "no bracket"),
        ("P5 STRICT joint > add for EVERY candidate pair", p5,
         f"{n_pos}/{len(pairs)}, mean gap {gaps.mean():+.4f}e-6"),
        ("P6 P(any of the registered ten clears) < 0.05", p6, f"{1-p_none_above:.6f}"),
    ]
    for nm, ok, detail in checks:
        print(f"  {'✅ CONFIRMED' if ok else '❌ *** FALSIFIED ***':22s} {nm:52s} [{detail}]")
        if not ok:
            fail(nm)
    print(f"  (P4 as LITERALLY registered — H_binding < {W63_PREREG_P4_CONST} — reads "
          f"{'TRUE' if p4_literal else 'FALSE'}, and is not the claim: it compares\n"
          f"   this run's CONDITIONAL bar against w59a's UNCONDITIONAL one. Not scored.)")
    print("  (P7 is checked by re-running w26g_send.py --n 10 dry AFTER the artefact is wired.)")

    out = dict(
        supersedes="w59a_hijackprice (five-file tie at 0.97118, void since the 08-23 sends)",
        gate_w=dict(passed=True, worst=worst_w, n_prices=ncheck_w, against="w62a_autopair.json"),
        gate_r=dict(passed=True, base=base, w62a=W62A_COST, dev=dR),
        gate_j=dict(passed=True, worst=worstJ, n_tier=n_tier, leverage=lev,
                    w59a_leverage_at_n5=W59A_LEVERAGE),
        gate_t="PASS",
        tiers=dict(slot1=list(TIER1), slot2=list(TIER2),
                   slot1_public=float(t1v), slot2_public=float(t2v)),
        determined=True, auto_pair=list(TIER1), pick=PICK, pick_cv=cv[PICK],
        # w79: what the design was, and which part of it fed the GLS. A reader has to be able to
        # tell a MEASURED bar from an INTERPOLATED one without re-running anything.
        fill=list(FILL), base_names=list(BASE_NAMES), n_design=len(NAMES),
        gls_scored=sorted(k for k in NAMES if k in LB_FIT), hole_filled=bool(FILL),
        wanted=list(WANTED), beta=beta, gamma=gamma, gap=gap,
        base=base, worthless_limit1=worthless, threshold_public=thresh,
        # ⚠ the key is `H_cond_predsd`, w59a's name for the same quantity, NOT a tidier one.
        # `w59b_barguard.py` reads it to check that H_binding is the stricter of the two, and a
        # successor that renames a field silently retires the test that field exists for.
        rows=rows, H_uncond=Hs.get("uncond"), H_cond_predsd=Hs.get("cond"), H_binding=H_bind,
        cv_bar_new=bar_new, cv_bar_superseded=cv[PICK] - W59A_H_BINDING * U,
        nonadditivity=dict(n_pairs=len(pairs), n_joint_gt_add=n_pos, mean_gap=float(gaps.mean()),
                           min_gap=float(gaps.min()), max_gap=float(gaps.max()),
                           tightest=worstp),
        plan_0824=PLAN_0824, set_price_cond=sp_cond, set_price_uncond=sp_unc,
        set_price_additive=add_day, additive_error=sp_unc - add_day,
        p_any_clears=1 - p_none_above,
        landing=[dict(stem=s, p_above=pa, p_at=pat) for s, pa, pat in P],
        predictions=dict(p1=p1, p2=p2, p3=p3, p4=p4, p4_literal=p4_literal, p5=p5, p6=p6),
        falsified=[nm for nm, ok, _ in checks if not ok], failures=FAILURES,
        bracket=brackets, caveats=[
            (f"the break-even is MEASURED across a "
             f"{max((b['width'] for b in brackets.values()), default=0):.3f}e-6 bracket; "
             f"{len(FILL)} hole-fillers sit in the design and are withheld from the GLS (w79)")
            if FILL else
            ("the break-even is interpolated across a hole in the design "
             f"({max((b['width'] for b in brackets.values()), default=0):.1f}e-6 wide, nothing "
             "inside it)"),
            "every above-tier landing is pooled into one bucket in the set price, so two "
            "clearers are treated as tied rather than ordered; since gamma < 0 the finer model "
            "would select the WORSE private posterior, so this pooling is mildly ANTI-conservative",
        ],
    )
    print(f"\nFAILURES: {FAILURES}")
    if not write:
        print("\n⚠ --no-write: artefact NOT written. The sender keeps reading whatever is on disk.")
        sys.exit(1 if FAILURES else 0)

    # ⚠⚠ THE WRITE IS BOUND TO THE GATES, NOT TO EVERY REGISTERED READING — AND THIS RULE WAS
    # CHANGED AFTER P4 AND P5 CAME BACK FALSIFIED. Said plainly so no later run has to infer it.
    # The first cut refused to write on ANY failure. That is wrong, and it is wrong in a way that
    # would have quietly corrupted every future prereg in this workspace: it makes the artefact
    # — and therefore the send path — hostage to readings that have nothing to do with whether
    # the bar was correctly derived, so a run that wants its pricer to ship acquires an interest
    # in registering only predictions it expects to confirm. w62's prereg says outright of its
    # own P5 "if it is FALSIFIED that is the interesting outcome, not a bug"; a write rule that
    # punishes the interesting outcome is a rule against asking hard questions.
    # ⚠ NEW: COUPLING THE ARTEFACT WRITE TO EVERY REGISTERED READING GIVES A RUN A MOTIVE TO
    # REGISTER ONLY SAFE PREDICTIONS. Bind the write to the GATES; RECORD the readings.
    # GATE W / GATE R / GATE T2 / GATE J all sys.exit before this line, so reaching it means
    # every gate passed; the one remaining bar-relevant condition is that a bar exists at all.
    # Whatever was falsified is written into the artefact as `falsified` so a reader sees it.
    if H_bind is None:
        print("\n⛔ NOT WRITING — no break-even could be bracketed, so there is no bar to ship.")
        sys.exit(1)
    if out["falsified"]:
        print(f"\n⚠ WRITING ANYWAY, with {out['falsified']} recorded in the artefact: every GATE "
              f"passed,\n  so the BAR is sound; what failed are readings about the instrument's "
              f"shape and the\n  direction of the move. See the write-rule note in the source.")
    with open(os.path.join(HERE, outfile), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote experiments/{outfile}")
    return out


if __name__ == "__main__":
    main()
