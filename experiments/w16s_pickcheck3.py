"""w16s: settle h3-vs-ens4 for the DEADLINE PICK on CV-legitimate grounds.

w16q §1 established, paired within member set on all six sets where both mixes are scored,
that `h3` is ABOVE `ens4` on CV 6/6 (mean +4.32e-6) and BELOW it on the public slice 6/6.
It then declined to move the pick, correctly: that second half is an LB reading and the brief
forbids moving a deadline pick on one. It handed slot 8 the CV-side instrument instead --
w16k's 500-rep simulated private slice, seed 1616, f 0.20 -- with `w16q_ens4avg` and
`blend159av` added as candidates. This is that run, plus the two things that follow from
having both transform sides in the candidate set for the first time.

WHAT THIS INSTRUMENT CAN AND CANNOT DO -- written before it was run, because it bounds every
conclusion below and it is easy to overclaim:

  The slices are drawn from TRAIN rows and every file is scored on the same slice each rep.
  The MEAN over reps is therefore a resample of the same OOF vectors that produce the CV --
  it cannot contradict CV in direction, and it is NOT an independent third opinion that could
  break the CV/LB tie. Anyone reading "the simulation prefers h3" as new evidence against the
  public slice is double-counting CV.

  What it genuinely adds is DISPERSION. The private set is ONE draw, not a mean. The
  decision-relevant quantity is P(the CV winner also wins on a single draw of this size).
  A +4.32e-6 CV margin with P(better) 0.55 is a coin flip dressed as a decision; the same
  margin at P 0.97 is not. That number does not exist anywhere in the workspace yet, and it
  is what settles whether the h3/ens4 choice is worth defending at all.

PRE-REGISTERED, fixed before the first number printed:

  R1. WANTED moves to an ens4-side file only if that file's cross-fitted CV EXCEEDS the
      incumbent's AND its E[max] pair beats the current pair. Given w16q §1's 6/6 CV reading
      the honest prediction is that both conditions come out False and WANTED does not move.
      Predicted here so a "no move" cannot be read as inertia.

  R2. The CROSS-AXIS HEDGE pair {w16i_schemeavg, blend159av} is evaluated as a real candidate
      and is the one genuinely new option this slot creates. Both current picks are h3-side,
      so if the transform axis goes the other way on private rows they fail TOGETHER; that
      pair hedges the correction family and the transform axis with the same second file, at
      no extra parameter cost (p23+p0, same as now). It replaces the current pair only if its
      E[max] is >= the current pair's. If it costs, the cost is recorded and the pick stays --
      w16q §2 established the floor for objects on a fixed stored base is EXACTLY ZERO, so a
      small cost here is a real cost and may not be waved away as sub-noise.

  R3. Everything is reported whichever way it comes out, including a result that says the
      current pick is wrong.

Corrected files enter as their CROSS-FITTED OOF -- fold f's rows carry weights fitted without
fold f -- which is conservative, because the shipped test files use full-data weights. Same
seed, same f, same REPS and the same construction of the three w16b arms as w16k, so this
run's numbers for the seven old candidates are directly comparable to w16k's.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, get_folds, load_raw  # noqa: E402

from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import CELL_A, fast_auc, pct, rule_cells  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_TEST = 296_302
REPS = 500
SEED = 1616
F = 0.20

# parameter counts, extending w16k's table with the two ens4-side entries
NPAR = {"blend159av_h3": 0, "blend159av": 0, "blend160origm_h3": 0, "blendtop3": 0,
        "w15f_antistudent_avg": 1, "w16b_cellweight": 7, "w16f_armavg": 10,
        "w16i_schemeavg": 23, "w16q_ens4avg": 23, "w16n_finegrid": 23}
ENS4 = {"blend159av", "w16q_ens4avg"}       # everything else in the set is h3-side
CORRECTED = ("w15f_antistudent_avg", "w16b_cellweight", "w16f_armavg",
             "w16i_schemeavg", "w16q_ens4avg")

# the transform contrast, matched on everything except the h3/ens4 mix
H3_ENS4_PAIRS = (("blend159av_h3", "blend159av"),          # p0, the zero-parameter hedge slot
                 ("w16i_schemeavg", "w16q_ens4avg"))       # p23, the corrected slot

CURRENT = ("w16i_schemeavg", BASE)                          # check_selection.py WANTED
HEDGE = ("w16i_schemeavg", "blend159av")                    # R2's cross-axis candidate

# Part 3: the live auto-selection tiers, read from check_selection.py on 2026-08-16 after
# slot 7's 0.97108 landed. `w16q_ens4avg` alone now holds auto-slot 1 and `blend158_logit`
# has dropped OUT of the top two tiers, so w15i's +9.2/+36.5/+112e-6 exposure ladder is
# quoted from a tier structure that no longer exists. `w16n_finegrid` is in tier 2 and is
# added to the candidate set FOR THIS PRICING ONLY -- pre-registered as INELIGIBLE to move
# WANTED, so that adding a file after seeing Part 1 cannot change the pick.
AUTO_T1 = "w16q_ens4avg"
AUTO_T2 = ("w15f_antistudent_avg", "w16b_cellweight", "w16f_armavg",
           "w16i_schemeavg", "w16n_finegrid")
PRICING_ONLY = ("w16n_finegrid",)


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    j = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    cell = rule_cells(tr)
    assigns = {"glob": np.array(["0"] * n, dtype=object),
               "a_only": np.where(cell == CELL_A, "A", "rest").astype(object),
               "per_cell": cell}
    arm_oof = {}
    for key in ("glob", "a_only", "per_cell"):
        z = br.copy()
        for (_, iva), w in zip(folds, j["fold_weights"][key]):
            for lvl, wv in w.items():
                m = iva[assigns[key][iva] == lvl]
                z[m] = br[m] + float(wv) * c[m]
        arm_oof[key] = z

    vecs = {
        "blend159av_h3": np.load(os.path.join(SUB, "oof_blend159av_h3.npy")),
        "blend159av": np.load(os.path.join(SUB, "oof_blend159av.npy")),
        "blend160origm_h3": np.load(os.path.join(SUB, "oof_blend160origm_h3.npy")),
        "blendtop3": np.load(os.path.join(SUB, "oof_blendtop3.npy")),
        "w15f_antistudent_avg": arm_oof["glob"],
        "w16b_cellweight": arm_oof["per_cell"],
        "w16f_armavg": np.mean([rankdata(arm_oof[k]) for k in
                                ("glob", "a_only", "per_cell")], axis=0),
        "w16i_schemeavg": np.load(os.path.join(SUB, "oof_w16i_schemeavg.npy")),
        "w16q_ens4avg": np.load(os.path.join(SUB, "oof_w16q_ens4avg.npy")),
        "w16n_finegrid": np.load(os.path.join(SUB, "oof_w16n_finegrid.npy")),
    }
    names = list(vecs)
    cv = {nm: fast_auc(y, vecs[nm].astype(np.float64)) for nm in names}
    print("candidate set (cross-fitted OOF for every corrected file):")
    for nm in names:
        side = "ens4" if nm in ENS4 else "h3  "
        if nm in PRICING_ONLY:
            side += " [pricing only, ineligible for WANTED]"
        print(f"  {nm:24s} p{NPAR[nm]:<2d} {side}  CV {cv[nm]:.8f}")

    # --- gate 1: reproduce w16k's seven candidates exactly -------------------------------
    prev = os.path.join(HERE, "w16k_pickcheck2.json")
    if os.path.exists(prev):
        pj = json.load(open(prev))
        print("\ngate: w16k's stored means, reproduced here on the same seed/protocol")
        print("  (any drift means the two runs are not comparable and nothing below stands)")

    pk = {nm: prep(vecs[nm].astype(np.float64), y) for nm in names}
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    A = np.zeros((REPS, len(names)))
    B = np.zeros((REPS, len(names)))     # SAME draws, scored on the PUBLIC-sized complement
    for i in range(REPS):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool)
        m[idx[n_pub:]] = True
        mp = np.zeros(n, dtype=bool)
        mp[idx[:n_pub]] = True
        for k, nm in enumerate(names):
            A[i, k] = subset_auc(pk[nm], m)
            B[i, k] = subset_auc(pk[nm], mp)
    mean = A.mean(axis=0)

    if os.path.exists(prev):
        ok = True
        for nm, mv in zip(pj["names"], pj["mean"]):
            if nm in names:
                d = mean[names.index(nm)] - mv
                flag = "" if abs(d) < 1e-12 else "   ** DRIFT **"
                if abs(d) >= 1e-12:
                    ok = False
                print(f"    {nm:24s} w16k {mv:.8f}  here {mean[names.index(nm)]:.8f}  "
                      f"d {d*1e6:+.4f}e-6{flag}")
        print(f"  gate {'PASSED' if ok else 'FAILED'}")

    # --- how much of this is just CV? ---------------------------------------------------
    print("\nsanity: mean simulated private AUC is a RESAMPLE of CV, not a second opinion")
    for nm in names:
        print(f"  {nm:24s} CV {cv[nm]:.8f}  sim mean {mean[names.index(nm)]:.8f}  "
              f"d {(mean[names.index(nm)] - cv[nm])*1e6:+7.2f}e-6")

    inc = A[:, names.index(BASE)]
    print(f"\nmean simulated private AUC, paired against {BASE}:")
    for k, nm in enumerate(names):
        d = A[:, k] - inc
        print(f"  {nm:24s} mean {mean[k]:.8f}  d {d.mean()*1e6:+6.2f}e-6 "
              f"+/-{d.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P(better) {(d > 0).mean():.3f}")

    # --- THE question: h3 vs ens4, matched, on a single draw ----------------------------
    print("\n=== h3 vs ens4, paired within matched file, on ONE draw of the private size ===")
    print("  P(h3 wins) is the decision number: the private set is one draw, not a mean.")
    for a, b in H3_ENS4_PAIRS:
        d = A[:, names.index(a)] - A[:, names.index(b)]
        se = d.std(ddof=1) / np.sqrt(REPS)
        print(f"  {a:18s} - {b:18s}  CV d {(cv[a]-cv[b])*1e6:+6.2f}e-6   "
              f"sim d {d.mean()*1e6:+6.2f}e-6 +/-{se*1e6:4.2f}   "
              f"draw sd {d.std(ddof=1)*1e6:6.2f}e-6   P(h3 better) {(d > 0).mean():.3f}")

    # --- does the PUBLIC-LB disagreement even need explaining? --------------------------
    # w16q read h3 one 1e-5 reporting step BELOW ens4 on the public slice in 6/6 member sets.
    # Those six are NESTED builds of near-identical objects, so as a reading of the SLICE they
    # are effectively ONE observation, not six. The question this answers: under the CV-side
    # model (h3 genuinely ahead by ~4.2e-6), how often does a PUBLIC-SIZED slice reverse it?
    # If that probability is appreciable, the LB/CV conflict is slice noise and no train/test
    # difference has to be invoked to explain it.
    print("\n=== reconciling the public-LB disagreement, same draws, public-sized slice ===")
    print(f"  private-sized mask {N_TEST - n_pub:,} rows | public-sized mask {n_pub:,} rows")
    for a, b in H3_ENS4_PAIRS:
        dpr = A[:, names.index(a)] - A[:, names.index(b)]
        dpu = B[:, names.index(a)] - B[:, names.index(b)]
        print(f"  {a:18s} - {b:18s}")
        print(f"     private-sized  mean {dpr.mean()*1e6:+6.2f}e-6  draw sd "
              f"{dpr.std(ddof=1)*1e6:5.2f}e-6   P(h3 better) {(dpr > 0).mean():.3f}")
        print(f"     public-sized   mean {dpu.mean()*1e6:+6.2f}e-6  draw sd "
              f"{dpu.std(ddof=1)*1e6:5.2f}e-6   P(h3 better) {(dpu > 0).mean():.3f}"
              f"   -> P(slice REVERSES it) {(dpu <= 0).mean():.3f}")
        # w16q bounded the true LB difference at (-20e-6, 0) from the 5dp rounding steps.
        print(f"     P(public-sized d <= -20e-6, w16q's lower bound) "
              f"{(dpu <= -20e-6).mean():.3f}")

    print("\nhead-to-head, paired, among the corrected files:")
    for a, b in itertools.combinations(CORRECTED, 2):
        d = A[:, names.index(a)] - A[:, names.index(b)]
        print(f"  {a:22s} - {b:22s} {d.mean()*1e6:+6.2f}e-6 "
              f"+/-{d.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P({a[:8]} better) {(d > 0).mean():.3f}")

    # --- E[max] over every pair ----------------------------------------------------------
    print("\nE[max] of every candidate PAIR, best first (this is what Kaggle scores):")
    cur = np.maximum(A[:, names.index(CURRENT[0])], A[:, names.index(CURRENT[1])])
    pairs = []
    elig = [i for i, nm in enumerate(names) if nm not in PRICING_ONLY]
    for i, k in itertools.combinations(elig, 2):
        mx = np.maximum(A[:, i], A[:, k])
        pairs.append((float(mx.mean()), names[i], names[k]))
    pairs.sort(reverse=True)
    for e, a, b in pairs:
        d = np.maximum(A[:, names.index(a)], A[:, names.index(b)]) - cur
        star = ""
        if {a, b} == set(CURRENT):
            star = "  <-- CURRENT PICK (check_selection.py WANTED)"
        elif {a, b} == set(HEDGE):
            star = "  <-- R2 cross-axis hedge"
        sides = {("ens4" if x in ENS4 else "h3") for x in (a, b)}
        mix = "BOTH" if len(sides) == 2 else sides.pop()
        print(f"  E[max] {e:.8f}  vs current {d.mean()*1e6:+6.2f}e-6 "
              f"+/-{d.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P {(d > 0).mean():.3f}  "
              f"p{NPAR[a]}+p{NPAR[b]}  {mix:4s}  {a} + {b}{star}")

    # --- the pre-registered rules, evaluated mechanically ---------------------------------
    print("\n=== PRE-REGISTERED RULES, evaluated ===")
    emax = {frozenset((a, b)): e for e, a, b in pairs}
    e_cur = emax[frozenset(CURRENT)]

    print("R1  move WANTED to an ens4-side file?")
    for h3n, e4n in H3_ENS4_PAIRS:
        cond_cv = cv[e4n] > cv[h3n]
        print(f"    {e4n:18s} CV > {h3n:18s} CV ?  {cv[e4n]:.8f} > {cv[h3n]:.8f}  "
              f"-> {cond_cv}")
    cand = frozenset(("w16q_ens4avg", "blend159av"))
    print(f"    ens4-side pair E[max] {emax[cand]:.8f} > current {e_cur:.8f} ?  "
          f"-> {emax[cand] > e_cur}   (d {(emax[cand]-e_cur)*1e6:+.2f}e-6)")

    print("R2  replace the current pair with the cross-axis hedge?")
    e_h = emax[frozenset(HEDGE)]
    dh = np.maximum(A[:, names.index(HEDGE[0])], A[:, names.index(HEDGE[1])]) - cur
    print(f"    hedge E[max] {e_h:.8f} >= current {e_cur:.8f} ?  -> {e_h >= e_cur}")
    print(f"    price of the transform hedge = {(e_h - e_cur)*1e6:+.3f}e-6 "
          f"+/-{dh.std(ddof=1)/np.sqrt(REPS)*1e6:.3f}   P(hedge better on a draw) "
          f"{(dh > 0).mean():.3f}")
    print("    (w16q §2: the floor for objects on a fixed stored base is ZERO. If this is")
    print("     negative it is a real cost, not noise, and the pick does not move.)")

    # --- Part 3: what NOT clicking actually costs, on the LIVE tier structure -------------
    print("\n=== Part 3: price of the un-made human click, on TODAY's tiers ===")
    print("  w15i's +9.2 / +36.5 / +112e-6 ladder is STALE. It was computed when")
    print("  blend158_logit (CV 0.969961) sat in a top-two tie; slot 7's 0.97108 pushed it")
    print("  out of both tiers, so the catastrophic branch that dominated that ladder is")
    print("  GONE. Repriced here against the same 500 draws.")
    print(f"  live auto-slot 1: {AUTO_T1} alone (public 0.97108)")
    print(f"  live auto-slot 2: 5-way tie {', '.join(AUTO_T2)} (public 0.97107)")
    print("  Kaggle's tiebreak inside a tie is not documented, so every member is priced.")

    e_want = emax[frozenset(CURRENT)]
    a_t1 = A[:, names.index(AUTO_T1)]
    print(f"\n  branch: limit is 2 -> auto gets {{{AUTO_T1}, X}} for X in tier 2")
    rows = []
    for x in AUTO_T2:
        mx = np.maximum(a_t1, A[:, names.index(x)])
        d = mx - cur
        rows.append((float(mx.mean()), x, float(d.mean()), float(d.std(ddof=1) / np.sqrt(REPS)),
                     float((d > 0).mean())))
    for e, x, dm, ds, p in sorted(rows):
        print(f"    E[max] {e:.8f}  vs WANTED {dm*1e6:+6.2f}e-6 +/-{ds*1e6:4.2f}  "
              f"P(auto better) {p:.3f}   {AUTO_T1} + {x}")
    worst = min(rows)
    best = max(rows)
    print(f"    -> cost of not clicking, limit 2: between {-best[2]*1e6:+.2f}e-6 and "
          f"{-worst[2]*1e6:+.2f}e-6 of E[max] (sign: positive = clicking WINS)")

    d1 = a_t1 - cur
    print(f"\n  branch: limit is 1 -> auto gets {AUTO_T1} alone")
    print(f"    E     {a_t1.mean():.8f}  vs WANTED {d1.mean()*1e6:+6.2f}e-6 "
          f"+/-{d1.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P(auto better) {(d1 > 0).mean():.3f}")
    print(f"    -> cost of not clicking, limit 1: {-d1.mean()*1e6:+.2f}e-6")
    print("\n  NOTE the sign, it is the whole point: the auto-pick's tier-1 file")
    print("  `w16q_ens4avg` is the account's best PUBLIC score and one of its WEAKEST")
    print("  corrected files on CV (0.97005152 vs w16i_schemeavg's 0.97005567). Kaggle")
    print("  auto-selecting on public score is exactly the Rogii failure, executed by the")
    print("  platform instead of by us. The click is still worth making; its price is now")
    print("  a few e-6 rather than w15i's worst-branch +112e-6.")

    json.dump(dict(names=names, cv={k: float(v) for k, v in cv.items()},
                   mean=[float(x) for x in mean],
                   h3_ens4=[dict(h3=a, ens4=b,
                                 cv_d=float(cv[a] - cv[b]),
                                 sim_d=float((A[:, names.index(a)] -
                                              A[:, names.index(b)]).mean()),
                                 p_h3=float((A[:, names.index(a)] -
                                             A[:, names.index(b)] > 0).mean()),
                                 pub_sim_d=float((B[:, names.index(a)] -
                                                  B[:, names.index(b)]).mean()),
                                 pub_sd=float((B[:, names.index(a)] -
                                               B[:, names.index(b)]).std(ddof=1)),
                                 pub_p_reverse=float((B[:, names.index(a)] -
                                                      B[:, names.index(b)] <= 0).mean()))
                            for a, b in H3_ENS4_PAIRS],
                   pairs=[dict(e_max=e, a=a, b=b) for e, a, b in pairs],
                   current=list(CURRENT), hedge=list(HEDGE),
                   e_current=float(e_cur), e_hedge=float(e_h),
                   auto=dict(tier1=AUTO_T1, tier2=list(AUTO_T2),
                             limit2=[dict(x=x, e_max=e, d_vs_wanted=dm, se=ds, p=p)
                                     for e, x, dm, ds, p in sorted(rows)],
                             limit1=dict(e=float(a_t1.mean()),
                                         d_vs_wanted=float(d1.mean())))),
              open(os.path.join(HERE, "w16s_pickcheck3.json"), "w"), indent=1)
    print("\nwrote experiments/w16s_pickcheck3.json")


if __name__ == "__main__":
    main()
