"""w69a -- THE 2x2 MEMBER FACTORIAL: lexb (ext_members14) x yadoy (ext_members15).

Registered in experiments/w69_prereg.txt, committed 39843c6 BEFORE this file existed.

WHY. w65a priced the three `zhukovoleksiy` lexb members at -1.058e-6 (deleting them GAINS)
and 3.80e-6 worse than a placebo trio, 4/4 partitions. `ext_members14` sits in every
ad202/ad211/ad216 build and NOTHING has priced its removal from the packs ABOVE 202, where
ARM 211 already reads BELOW ARM 199 on every stored base. w68 section 7 item 4.

A LADDER CANNOT ANSWER THIS AND A 2x2 CAN. The three arms on disk (199, 202, 211) are three
corners of a square; the fourth -- 208 = 211 minus lexb -- has never been built, and without
it the lexb main effect and its INTERACTION with the nine `yadoy666` streams are perfectly
confounded. The interaction IS the candidate mechanism for ARM 211's position.

  A199 = -lexb -yadoy    A202 = +lexb -yadoy    A208 = -lexb +yadoy    A211 = +lexb +yadoy

  E_A_lo = A202 - A199   lexb without the nine   [replicates w65a: -1.058e-6 on h3]
  E_A_hi = A211 - A208   lexb with the nine
  E_B_lo = A208 - A199   the nine without lexb   [ARM 208's OWN matched control]
  E_B_hi = A211 - A202   the nine with lexb      [re-derives w61a IN-PROCESS]
  INTER  = E_A_hi - E_A_lo == E_B_hi - E_B_lo    [the new quantity]

AND WHY IN-PROCESS IS THE POINT, NOT A DETAIL. w61a retired `w40_ad211` from
WANTED_INELIGIBLE on E_B_hi measured BETWEEN TWO SHIPPED FILES against a +-4e-6 bar. w68
then measured the cross-process offset as a PER-FILE CONSTANT, so a between-file difference
carries the DIFFERENCE of two draws -- 5.095e-6, LARGER than that bar. w68 landed one run
after w61 decided, so this is an inference that only becomes available now. Here every
contrast is formed inside ONE process from ONE load on ONE partition, so the shared arm and
most of the partition term cancel exactly (w65 section 0: FORM THE CONTRAST, never compare
derived means -- it took w65's sd from 2.230 to 0.985).

Per-seed results are flushed to disk as they complete. w65a's all-or-nothing artefact cost
three runs of waiting for a number that existed after the first seed.

    .venv/bin/python experiments/w69a_factorial.py --gates-only
    .venv/bin/python experiments/w69a_factorial.py --seeds 42,101,13,7
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import DATA  # noqa: E402
from blend_lab import load_all  # noqa: E402

# All four transforms: `ens4` is the rank-average of all four (stdflag.family_suffix -- a
# bare stem), so the criterion cannot be evaluated on H3's three alone.
TRANS = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")
DROP = ("golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,"
        "lat_ctfixte_r400,om_cat")
# w40f_run.sh, verbatim (the 211 pack). 202 = minus ext_members15; 199 = minus both.
DIRS_211 = ("ext_members3", "ext_members4", "ext_members6", "ext_members7pin",
            "ext_members8", "ext_members10", "ext_members11", "ext_members12",
            "ext_members14", "ext_members15")
DIRS_202 = tuple(d for d in DIRS_211 if d != "ext_members15")
DIRS_199 = tuple(d for d in DIRS_202 if d != "ext_members14")

LEXB = ("lexb_cat_base", "lexb_lgb02", "lexb_xgb_base")            # ext_members14, 3
YADOY = ("y94_lookup", "y94_naji03", "y94_naji05", "y94_pub_rmlp", "y94_pub_tabnet",
         "y94_rmlp_lat3", "y94_rmlp_lat", "y94_tabm_deeper", "y94_tabm_imp")  # 15, 9

ARMS = {"A211": (), "A208": LEXB, "A202": YADOY, "A199": LEXB + YADOY}
SIZES = {"A211": 211, "A208": 208, "A202": 202, "A199": 199}
BASES = ("h3", "ens4", "rescale", "rankraw")        # w61a's criterion, in its order
EXTRA = ("hybrid", "logit")                          # reported, never decided on

PLACEBO_SEED = 69          # prereg section 4 P3, fixed before any number existed
INJECT = 4.0e-6            # prereg P9: the DECISION size, which is the bar itself
BAR = 4e-6

HERE = os.path.dirname(os.path.abspath(__file__))
OUTP = os.path.join(HERE, "w69a_factorial.json")
OUT = {"prereg": "experiments/w69_prereg.txt", "commit": "39843c6",
       "bar": BAR, "criterion_bases": list(BASES), "failures": [], "seeds": {}}


def fail(msg):
    OUT["failures"].append(msg)
    print("  !! " + msg, flush=True)


def flush():
    with open(OUTP, "w") as fh:
        json.dump(OUT, fh, indent=1, default=float)


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def crossfit(Z, y, folds, C=1.0):
    mo = np.zeros(len(y))
    for itr, iva in folds:
        mo[iva] = (LogisticRegression(max_iter=5000, C=C)
                   .fit(Z[itr], y[itr]).decision_function(Z[iva]))
    return mo


def scaled(mats, keys):
    """Exactly w65a/w68a's scaling: per column, from the OOF side, test side dropped."""
    Z = {}
    for k in keys:
        M = mats[k][0]
        s = M.std(0)
        s[s <= 0] = 1.0
        Z[k] = (M / s).astype(M.dtype)
        mats[k] = None
        del M
        gc.collect()
    return Z


def load_pack(dirs, keys):
    names, y, mats, _ = load_all(keys, DROP.split(","),
                                 extra_dirs=tuple(os.path.join(DATA, d) for d in dirs),
                                 dtype="float32")
    return list(names), y, scaled(mats, keys)


# =========================================================================================
# GATES
# =========================================================================================
def gates(names211, Z211, y):
    pos = {n: i for i, n in enumerate(names211)}

    # ---- P1: the pack partitions the way the build scripts say -------------------------
    print("\n== P1 GATE: pack composition ==")
    ok = len(names211) == 211
    miss_l = [n for n in LEXB if n not in pos]
    miss_y = [n for n in YADOY if n not in pos]
    print(f"  |211 load| = {len(names211)}   (expected 211: {ok})")
    print(f"  lexb  present {len(LEXB) - len(miss_l)}/3   missing {miss_l}")
    print(f"  yadoy present {len(YADOY) - len(miss_y)}/9  missing {miss_y}")
    arm_cols = {}
    for a, excl in ARMS.items():
        cols = np.array([i for n, i in sorted(pos.items(), key=lambda kv: kv[1])
                         if n not in excl])
        arm_cols[a] = cols
        good = len(cols) == SIZES[a]
        print(f"  {a}: {len(cols):4d} columns (expected {SIZES[a]}) {'OK' if good else 'MISMATCH'}")
        if not good:
            fail(f"P1: {a} has {len(cols)} columns, expected {SIZES[a]}")
    OUT["P1"] = {"n_load": len(names211), "missing_lexb": miss_l, "missing_yadoy": miss_y,
                 "arm_sizes": {a: int(len(c)) for a, c in arm_cols.items()}}
    if not ok or miss_l or miss_y:
        fail("P1 VOIDS THE RUN: the 211 load does not contain the named members")
    return arm_cols


def gate_p2_p3(names211, Z211, arm_cols):
    """P2: a 9- and a 12-column drop are BITWISE a native load. P3: a placebo 9 is not.

    w68a proved this for a 3-column drop. w68b showed np.std(axis=0) is NOT bitwise
    invariant to the COLUMN COUNT (3.0e-7 in float32), so w68a's result is an empirical
    fact about those matrices, not a theorem, and it does NOT transfer to a wider drop.
    """
    pos = {n: i for i, n in enumerate(names211)}
    p2, p3 = {}, {}
    rng = np.random.default_rng(PLACEBO_SEED)
    pool = [n for n in names211 if n not in LEXB and n not in YADOY]
    placebo = tuple(rng.choice(pool, size=9, replace=False))
    OUT["placebo_9"] = list(placebo)
    print(f"\n  placebo draw (seed {PLACEBO_SEED}): {list(placebo)}")

    for tag, dirs, excl in (("A202", DIRS_202, YADOY), ("A199", DIRS_199, LEXB + YADOY)):
        print(f"\n== P2: subsetted {tag} vs a NATIVE {SIZES[tag]}-member load ==", flush=True)
        nnat, _, Znat = load_pack(dirs, TRANS)
        sub_names = [n for n in names211 if n not in excl]
        same_order = sub_names == nnat
        print(f"  |subset| {len(sub_names)}  |native| {len(nnat)}  "
              f"same SET {set(sub_names) == set(nnat)}  same ORDER {same_order}")
        if not same_order:
            fail(f"P2/{tag}: subset and native name lists differ in set or order")
        keep = np.array([pos[n] for n in nnat])
        d = {}
        for k in TRANS:
            md = float(np.abs(Z211[k][:, keep] - Znat[k]).max())
            d[k] = {"max_abs_diff": md,
                    "bitwise_equal": bool(np.array_equal(Z211[k][:, keep], Znat[k]))}
            print(f"  {k:8s} max|diff| {md:.6e}   bitwise equal: {d[k]['bitwise_equal']}")
            gc.collect()
        p2[tag] = {"same_order": bool(same_order), "transforms": d}
        if not all(v["bitwise_equal"] for v in d.values()):
            fail(f"P2 FALSIFIED for {tag}: subsetting the 211 load is not a native load")

        if tag == "A202":   # P3 negative control, against the same native matrix
            print("\n== P3 negative control: drop nine OTHER columns instead ==")
            pk = np.array([pos[n] for n in names211 if n not in placebo])
            for k in TRANS:
                md = float(np.abs(Z211[k][:, pk] - Znat[k]).max())
                p3[k] = md
                print(f"  {k:8s} max|diff| vs native 202: {md:.6e}")
                gc.collect()
            if not all(v > 0 for v in p3.values()):
                fail("P3 FALSIFIED: the comparison cannot detect a wrong column set -- "
                     "P2 passes VACUOUSLY")
        del Znat, nnat
        gc.collect()

    OUT["P2"], OUT["P3"] = p2, p3
    print(f"\n  P2 -> {'CONFIRMED' if not any('P2' in f for f in OUT['failures']) else 'FALSIFIED'}")


# =========================================================================================
# THE FACTORIAL
# =========================================================================================
def bases_from(oof):
    """The four criterion bases + two extras, from the per-transform OOF meta-vectors."""
    out = {"h3": float(roc_auc_score(Y, np.mean([rk(oof[k]) for k in H3], 0))),
           "ens4": float(roc_auc_score(Y, np.mean([rk(oof[k]) for k in TRANS], 0)))}
    for k in ("rescale", "rankraw", "hybrid", "logit"):
        out[k] = float(roc_auc_score(Y, oof[k]))
    return out


def run_seed(seed, Z211, arm_cols):
    t0 = time.time()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed)
                 .split(np.zeros(len(Y)), Y))
    auc, oof_h3 = {}, {}
    for a in ("A199", "A202", "A208", "A211"):
        oof = {}
        for k in TRANS:
            oof[k] = crossfit(Z211[k][:, arm_cols[a]], Y, folds)
            print(f"    {a} {k:8s} {time.time()-t0:6.0f}s", flush=True)
        auc[a] = bases_from(oof)
        oof_h3[a] = np.mean([rk(oof[k]) for k in H3], 0)
        print(f"  {a}: " + "  ".join(f"{b} {auc[a][b]:.9f}" for b in BASES), flush=True)
        del oof
        gc.collect()

    d = {}
    for b in BASES + EXTRA:
        d[b] = {"E_A_lo": (auc["A202"][b] - auc["A199"][b]) * 1e6,
                "E_A_hi": (auc["A211"][b] - auc["A208"][b]) * 1e6,
                "E_B_lo": (auc["A208"][b] - auc["A199"][b]) * 1e6,
                "E_B_hi": (auc["A211"][b] - auc["A202"][b]) * 1e6}
        d[b]["INTER"] = d[b]["E_A_hi"] - d[b]["E_A_lo"]
        # identity check: the interaction is the same number down either margin
        alt = d[b]["E_B_hi"] - d[b]["E_B_lo"]
        d[b]["INTER_alt"] = alt
        if abs(alt - d[b]["INTER"]) > 1e-9:
            fail(f"seed {seed} base {b}: the two interaction margins disagree")

    # ---- P9 POWER CONTROL: inject at the DECISION size into A208's h3, recover E_B_lo ---
    # ⚠ THE INJECTED DIRECTION HAS TO BE ONE THAT CAN RAISE AN AUC. The first cut added
    # c * N(0,1); random noise is uninformative about y, so the AUC falls MONOTONICALLY in c
    # and the bisection walked to its bracket edge and reported -341,086e-6. Injecting
    # c * (y - ybar) adds TRUE-LABEL signal, so the AUC rises monotonically in c and there is
    # a root to find. ⚠ A BISECTION NEEDS ITS BRACKET VERIFIED, NOT ASSUMED: the bracket is
    # grown until it straddles, and a failure to straddle is a FAILURE, not a silent clamp.
    # ⚠ AND THE DIRECTION MUST BE CONTINUOUS, NOT JUST INFORMATIVE. `y - ybar` is BINARY, so
    # raising c lifts every positive by the same amount and they cross the negatives in
    # BATCHES: the AUC is a step function in c with steps of ~0.08e-6, coarser than the
    # tolerance below, and the bisection cannot land on the target however many iterations it
    # is given. A continuous informative direction (a rank-scaled noisy copy of y) moves the
    # AUC smoothly and lands on +4.000000e-6 exactly. Fixed seed, so every seed and every arm
    # gets the SAME injected direction.
    v = oof_h3["A208"]
    sig = SIGDIR
    a0 = roc_auc_score(Y, v)

    def gain(c):
        return roc_auc_score(Y, v + c * sig) - a0

    hi = 1e-6
    for _ in range(60):                       # GROW until the bracket provably straddles
        if gain(hi) >= INJECT:
            break
        hi *= 2.0
    if gain(hi) < INJECT:
        fail(f"P9 seed {seed}: bracket never reached +{INJECT*1e6:.1f}e-6 of AUC")
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gain(mid) < INJECT:
            lo = mid
        else:
            hi = mid
    got = gain(hi)
    d["h3"]["E_B_lo_injected"] = (roc_auc_score(Y, v + hi * sig)
                                  - roc_auc_score(Y, oof_h3["A199"])) * 1e6
    d["h3"]["inject_recovered"] = got * 1e6
    d["h3"]["inject_coef"] = float(hi)
    if abs(got - INJECT) > 0.10e-6:
        fail(f"P9 seed {seed}: injection landed at {got*1e6:.3f}e-6, not {INJECT*1e6:.1f}e-6")

    # Persist the six h3 OOF vectors so a later run can re-analyse WITHOUT refitting. w65a's
    # artefact held only summaries and every follow-up question cost another 9 CPU-hours.
    np.savez_compressed(os.path.join(HERE, f"w69a_oof_h3_seed{seed}.npz"),
                        **{a: oof_h3[a].astype(np.float32) for a in oof_h3})

    OUT["seeds"][str(seed)] = {"auc": auc, "delta": d, "secs": time.time() - t0}
    flush()
    print(f"  seed {seed} done in {time.time()-t0:.0f}s -> {OUTP}", flush=True)
    for b in BASES:
        print(f"    {b:8s} E_A_lo {d[b]['E_A_lo']:+7.3f}  E_A_hi {d[b]['E_A_hi']:+7.3f}  "
              f"E_B_lo {d[b]['E_B_lo']:+7.3f}  E_B_hi {d[b]['E_B_hi']:+7.3f}  "
              f"INTER {d[b]['INTER']:+7.3f}", flush=True)


def summarise(n_want=None):
    """⚠ A VERDICT OFF A PARTIAL SEED SET IS NOT THE REGISTERED VERDICT.

    w69_prereg registers P5/P6/P7/P8 and R1 over the FULL seed set. Flushing after every seed
    is what makes a killed run salvageable, but the first cut also printed "P5 -> FALSIFIED"
    after seed 42 alone -- off a one-seed mean, against a bar written for a four-seed one.
    That line is exactly the sort of thing a later run quotes as a result. Verdicts are now
    gated on n == n_want and everything below is stamped PROVISIONAL until then.
    """
    seeds = sorted(OUT["seeds"], key=int)
    if not seeds:
        return
    final = (n_want is not None) and (len(seeds) == n_want)
    OUT["provisional"] = not final
    if not final:
        print(f"\n  [PROVISIONAL — {len(seeds)} of {n_want} seeds. The prereg registers every "
              f"reading below over the FULL seed set; no verdict is evaluated yet.]")
    print("\n" + "=" * 86)
    print(f"SUMMARY over {len(seeds)} seed(s): {seeds}   [e-6, paired within-process]")
    print("=" * 86)
    summ = {}
    for b in BASES + EXTRA:
        summ[b] = {}
        for e in ("E_A_lo", "E_A_hi", "E_B_lo", "E_B_hi", "INTER"):
            v = np.array([OUT["seeds"][s]["delta"][b][e] for s in seeds])
            se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            summ[b][e] = {"mean": float(v.mean()),
                          "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                          "se": se, "pos": int((v > 0).sum()), "n": len(v),
                          "t": float(v.mean() / se) if len(v) > 1 and se > 0 else float("nan")}
        print(f"\n  base {b}")
        for e in ("E_A_lo", "E_A_hi", "E_B_lo", "E_B_hi", "INTER"):
            s = summ[b][e]
            print(f"    {e:7s} mean {s['mean']:+8.3f}  sd {s['sd']:6.3f}  se {s['se']:6.3f}"
                  f"  pos {s['pos']}/{s['n']}  t {s['t']:+6.2f}")
    OUT["summary"] = summ

    # ---- the registered readings, evaluated by reference to the prereg ------------------
    print("\n" + "-" * 86)
    n = len(seeds)
    if not final:
        print("  registered readings NOT evaluated — partial seed set.")
        flush()
        return
    p5 = summ["h3"]["E_A_lo"]["mean"]
    OUT["P5"] = {"value": p5, "w65a": -1.058, "within_1.5": abs(p5 + 1.058) <= 1.5}
    print(f"  P5 replication: E_A_lo/h3 {p5:+.3f}e-6 vs w65a's -1.058e-6 -> "
          f"{'CONFIRMED' if OUT['P5']['within_1.5'] else 'FALSIFIED'} (bar +-1.5e-6)")

    if n > 1:
        sd = summ["h3"]["E_B_hi"]["sd"]
        OUT["P6"] = {"sd": sd, "interval": [0.3, 3.0], "in": 0.3 <= sd <= 3.0,
                     "between_file_floor": 5.095}
        print(f"  P6 resolution: sd(E_B_hi/h3) {sd:.3f}e-6 vs the 5.095e-6 BETWEEN-FILE "
              f"floor -> {'CONFIRMED' if OUT['P6']['in'] else 'OUTSIDE [0.3,3.0]'}")
        inter = summ["h3"]["INTER"]["mean"]
        OUT["P7"] = {"value": inter, "interval": [-6.0, 1.0],
                     "in": -6.0 <= inter <= 1.0, "t": summ["h3"]["INTER"]["t"]}
        print(f"  P7 interaction: INTER/h3 {inter:+.3f}e-6 (t {summ['h3']['INTER']['t']:+.2f})"
              f" -> {'CONFIRMED' if OUT['P7']['in'] else 'OUTSIDE [-6,+1]'}")

    worst = max(abs(summ[b]["E_B_lo"]["mean"]) for b in BASES)
    which = max(BASES, key=lambda b: abs(summ[b]["E_B_lo"]["mean"]))
    OUT["P8"] = {"worst_abs": worst, "base": which, "under_bar": worst < BAR * 1e6,
                 "per_base": {b: summ[b]["E_B_lo"]["mean"] for b in BASES},
                 "NOTE": "REPORTED, NOT DECIDED ON -- w69_prereg 2.1/R2 forbids the run "
                         "that built ARM 208 lifting ARM 208's bar."}
    print(f"  P8 ARM 208 matched control E_B_lo: worst |{which}| {worst:.3f}e-6 vs bar 4.0 -> "
          f"{'under' if OUT['P8']['under_bar'] else 'OVER'}  [REPORTED, NOT DECIDED ON]")

    wb = max(abs(summ[b]["E_B_hi"]["mean"]) for b in BASES)
    wbb = max(BASES, key=lambda b: abs(summ[b]["E_B_hi"]["mean"]))
    OUT["R1"] = {"worst_abs": wb, "base": wbb, "exceeds_bar": wb > BAR * 1e6,
                 "per_base": {b: summ[b]["E_B_hi"]["mean"] for b in BASES},
                 "w61a": {"h3": 0.163, "ens4": 0.381, "rescale": 0.946, "rankraw": -2.353},
                 "NOTE": "R1: recorded, key NOT moved this run -- re-imposing a bar is as "
                         "much a decision as lifting one and this run built the arm."}
    print(f"  R1 in-process E_B_hi (w61a's estimand): worst |{wbb}| {wb:.3f}e-6 vs bar 4.0 -> "
          f"{'EXCEEDS' if OUT['R1']['exceeds_bar'] else 'inside'}  [RECORDED, NOT ACTED ON]")
    print("     w61a between-file: h3 +0.163  ens4 +0.381  rescale +0.946  rankraw -2.353")

    rec = [OUT["seeds"][s]["delta"]["h3"]["inject_recovered"] for s in seeds]
    shift = [OUT["seeds"][s]["delta"]["h3"]["E_B_lo_injected"]
             - OUT["seeds"][s]["delta"]["h3"]["E_B_lo"] for s in seeds]
    se_b = summ["h3"]["E_B_lo"]["se"]
    OUT["P9"] = {"recovered_e6": rec, "shift_e6": shift, "se_E_B_lo": se_b,
                 "detected_sigma": float(np.mean(shift) / se_b) if n > 1 and se_b > 0
                 else float("nan")}
    print(f"  P9 power: injected {INJECT*1e6:.1f}e-6, recovered {np.mean(rec):.3f}e-6, "
          f"E_B_lo shifted {np.mean(shift):+.3f}e-6"
          + (f", detected at {OUT['P9']['detected_sigma']:.2f} se -> "
             f"{'CONFIRMED' if OUT['P9']['detected_sigma'] >= 2 else 'UNDERPOWERED'}"
             if n > 1 else " (se needs >1 seed)"))
    flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,101,13,7")
    ap.add_argument("--gates-only", action="store_true")
    args = ap.parse_args()

    global Y, SIGDIR
    t0 = time.time()
    print("== loading the 211 pack ==", flush=True)
    names211, Y, Z211 = load_pack(DIRS_211, TRANS)
    # P9's injected direction: continuous, informative, fixed across seeds and arms.
    SIGDIR = (rk(Y + 0.8 * np.random.default_rng(PLACEBO_SEED).standard_normal(len(Y)))
              - 0.5)
    print(f"  211 pack scaled at {time.time()-t0:.0f}s, {len(names211)} members", flush=True)

    arm_cols = gates(names211, Z211, Y)
    gate_p2_p3(names211, Z211, arm_cols)

    # ---- P4 determinism -----------------------------------------------------------------
    print("\n== P4 determinism control ==")
    f2 = list(StratifiedKFold(5, shuffle=True, random_state=42)
              .split(np.zeros(len(Y)), Y))
    sm = Z211["rescale"][:, arm_cols["A199"][:40]]
    a1 = roc_auc_score(Y, crossfit(sm, Y, f2))
    a2 = roc_auc_score(Y, crossfit(sm, Y, f2))
    OUT["P4"] = {"diff": float(a1 - a2), "exact": a1 == a2}
    print(f"  two crossfits on the same array: diff {a1-a2:.3e}  exact: {a1 == a2}")
    if a1 != a2:
        fail("P4 FALSIFIED: crossfit is not deterministic -- every e-6 number here is void")
    del sm
    gc.collect()
    flush()

    if args.gates_only:
        print(f"\n--gates-only: stopping. FAILURES {len(OUT['failures'])}")
        for f in OUT["failures"]:
            print("  " + f)
        flush()
        return

    want = [int(s) for s in args.seeds.split(",")]
    for seed in want:
        print(f"\n{'='*86}\n== SEED {seed} ==\n{'='*86}", flush=True)
        run_seed(seed, Z211, arm_cols)
        summarise(n_want=len(want))

    print(f"\nFAILURES {len(OUT['failures'])}")
    for f in OUT["failures"]:
        print("  " + f)
    flush()


if __name__ == "__main__":
    main()
