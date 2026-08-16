"""w16t: the per-cell correction on the ENS4 base, and the control it is measured against.

WHERE THIS COMES FROM
---------------------
w16s §8.1 handed slot 9 a specific build: `w16b_cellweight` on the ens4 base. Its reasons,
restated so this script can be read on its own:

  * The account holds exactly ONE ens4-side corrected file (`w16q_ens4avg`, p23) against a
    large h3-side family (`w15f_antistudent_avg` p1, `w16b_cellweight` p7, `w16f_armavg` p10,
    `w16i_schemeavg` p23, `w16n_finegrid` p23). That asymmetry is why w16q had to build one
    from scratch, and it is still there. It is a gap in the OPTION SET, not in the analysis.
  * The per-cell arm is the highest-CV single arm on both bases, and w16q's leave-one-fold-out
    over the five schemes picked it 4/5 on the ens4 base.
  * w16q §5 already showed the `c_avg` weight structure reproduces on the ens4 base (all five
    arms within 0.5e-6 of their h3 counterparts), so the construction is known to transfer.

WHAT THIS RUN ADDS BEYOND RE-RUNNING w16q'S `rule` ARM
------------------------------------------------------
The arm's cross-fitted CV is already in `w16q_ens4base.json` (0.9700513727) — re-deriving it
is a gate, not a finding. The finding this run goes after is the CONTROL.

Both published readings of "does splitting one weight into seven beat splitting it at random"
rest on a SINGLE permuted-membership draw:

    w16b (h3 base)   real +6.195e-6   control +2.370e-6   real - control  +3.825e-6   1 draw
    w16q (ens4 base) real +6.399e-6   control +5.021e-6   real - control  +1.378e-6   1 draw

Those two control numbers differ by 2.65e-6 and nothing in either entry says whether that is
a property of the base or a property of the draw. The difference is load-bearing: the whole
argument for the per-cell arm is "the real segmentation pays back the selection toll a random
one charges", and on the ens4 base — the base this run ships on — that margin currently reads
+1.4e-6 off one draw. This wave has now twice caught a single-observation quantity being
quoted as a property of the pack (w16o's `gap_bagging`, w16q's mix-gap estimator). A control
with no error bar is the same shape.

So: SEVEN control draws per base — the published one, reproduced exactly as a gate, plus six
fresh ones — on both the ens4 and h3 bases, same folds, same grid, same protocol. That turns
"real - control" into a quantity with a standard error on both sides and says whether the
h3/ens4 difference in it survives.

PARTS
-----
0. Gates. Reproduce w16q's three ens4 arm deltas, w16b's h3 per-cell delta, and both published
   control draws. Anything that drifts invalidates everything downstream and is printed.
1. The control ensemble, 7 draws x 2 bases, and `real - control` with a standard error.
2. Nested-pick stability on the ens4 base: leave-one-fold-out argmax over
   {glob, a_only, per_cell}, and the per-fold weight table. This prices the selection this
   script does NOT make — the shipped arm is fixed in advance (S1 below).
3. The full-data fit and the test file. Cell labels recomputed on test.csv by the same
   function; nothing is transferred by row index.
4. A THIRD matched h3/ens4 pair, at p7, on w16s's 500-rep simulated private slice (seed 1616,
   f 0.20). w16s settled h3-vs-ens4 with two pairs, p0 and p23. This adds the p7 pair, which
   w16s could not build because the ens4-side p7 object did not exist. It is a confirmation
   of a settled question on a third object, and per S5 it cannot move anything.

PRE-REGISTERED, fixed here before the script was executed once
--------------------------------------------------------------
S1. SHIP the PER-CELL (7-weight) arm on base `blend159av`, unconditionally, whatever its CV
    and whatever Part 1 says about the control. File `submissions/w16t_cellens4.csv`. There is
    no argmax anywhere in the ship path: w16b chose between its 2-parameter and 7-parameter
    arms on cross-fitted CV, and this run does not, because the object being ported is defined
    as "the thing w16b shipped". If Part 1 comes out saying the per-cell arm does NOT clear
    its control on the ens4 base, the file goes out anyway and the finding is reported. Naming
    the file before the numbers exist is the only defence against the argmax bug that cost
    w16c +1.778e-6 and w16i +1.546e-6.

S2. FALLBACK, and only for this reason: if the file comes out rank-identical to one this
    account has already sent, ship `submissions/w14a_repro159av_h3.csv` instead (CV 0.9700472,
    verified unsent by w16q). Mechanical check, not a judgement.

S3. DEADLINE PICKS. `WANTED` moves only if this file's pooled cross-fitted CV exceeds
    `w16i_schemeavg`'s 0.9700556663, in which case it replaces it in the FIRST slot on CV
    alone. PREDICTED FALSE — w16q already measured this arm at 0.9700513727, which is 4.3e-6
    below, so this is a formality evaluated mechanically so that "no move" cannot be read as
    inertia. The SECOND slot is not touched under any outcome: w16i §4 fixed it to a
    ZERO-PARAMETER file as insurance against the whole `c_avg` family reversing on the private
    rows, and a 7-parameter file built on the same `c_avg` cannot serve that role.

S4. LB PREDICTION 0.97108, alternative 0.97107, written in full with its derivation and its
    own admission of low information content to `experiments/w16t_prereg_lb.txt` before this
    script ran. Not revisited.

S5. Part 4 is a CV-side confirmation of a question w16s CLOSED (h3 wins, P 0.912 / 0.908).
    It cannot move a deadline pick in either direction, and it is reported whichever way it
    comes out. If the p7 pair disagreed with the other two that would be a finding about the
    correction, not a reason to reopen the transform choice.

Nothing here is chosen with reference to the public leaderboard.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402
from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import CELL_A, apply_w, ascend, fast_auc, pct, rule_cells  # noqa: E402
from w16i_schemeavg import permuted  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENS4 = "blend159av"
H3 = "blend159av_h3"
TAG = "w16t_cellens4"
N_TEST = 296_302
FALLBACK = "w14a_repro159av_h3"

# published single-draw numbers this run gates on
W16Q_ARMS = {"glob": 3.362863220468526e-06,
             "a_only": 4.9867209553466905e-06,
             "per_cell": 6.399173514615164e-06}
W16Q_CTRL_RULE = 5.020851442472108e-06
W16Q_RULE_CV = 0.9700513727     # printed to 10 dp in w16q_ens4base.json as 0.9700513726632269
W16B_PER_CELL = 6.195080157267441e-06
W16B_CTRL7 = 2.3704473278707283e-06
W16I_CV = 0.9700556663          # the deadline-pick incumbent, for S3

# fresh control seeds, fixed here
CTRL_SEEDS = (601, 602, 603, 604, 605, 606)

# Part 4
REPS = 500
SEED = 1616
F = 0.20
W16S_PAIRS = {("blend159av_h3", "blend159av"): (4.1900467953939204e-06, 0.912),
              ("w16i_schemeavg", "w16q_ens4avg"): (4.051418625711678e-06, 0.908)}


def arm_xfit(y, br, c, assign, folds):
    """Cross-fitted delta over the base, the shared protocol. Returns (deltas, oof, weights)."""
    z = br.copy()
    d, ws = [], []
    for itr, iva in folds:
        w = ascend(y, br, c, assign, itr)
        zv = apply_w(br, c, assign, w, iva)
        z[iva] = zv
        d.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
        ws.append({str(k): float(v) for k, v in w.items()})
    return np.array(d), z, ws


def main():
    t0 = time.time()
    out = {}
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    cell = rule_cells(tr)
    a_only = np.where(cell == CELL_A, "A", "rest").astype(object)
    glob = np.array(["0"] * n, dtype=object)

    raw = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in (ENS4, H3)}
    br = {k: pct(v) for k, v in raw.items()}
    auc0 = {k: fast_auc(y, v) for k, v in raw.items()}
    print("=" * 78)
    print("w16t  per-cell c_avg correction on the ENS4 base, with a control ENSEMBLE")
    print("=" * 78)
    print(f"n {n:,}   test {N_TEST:,}   folds SKF5 seed42 (frozen)")
    for k in (ENS4, H3):
        print(f"  base {k:16s} pooled OOF AUC {auc0[k]:.10f}")
    print(f"  cell A share (train) {float((cell == CELL_A).mean()):.4f}", flush=True)

    # ------------------------------------------------------------------ PART 0: gates
    print("\n" + "-" * 78)
    print("PART 0  gates -- nothing below stands if these drift")
    print("-" * 78, flush=True)
    arms, arm_oof, arm_w = {}, {}, {}
    for nm, assign in (("glob", glob), ("a_only", a_only), ("per_cell", cell)):
        d, z, ws = arm_xfit(y, br[ENS4], c, assign, folds)
        arms[nm], arm_oof[nm], arm_w[nm] = d, z, ws
        g = W16Q_ARMS[nm]
        print(f"  ens4 {nm:<9} xfit {d.mean()*1e6:+8.4f}e-6   w16q {g*1e6:+8.4f}e-6   "
              f"drift {(d.mean()-g)*1e12:+.3f}e-12   CV {fast_auc(y, z):.10f}", flush=True)

    d_h3, oof_h3, w_h3 = arm_xfit(y, br[H3], c, cell, folds)
    print(f"  h3   per_cell  xfit {d_h3.mean()*1e6:+8.4f}e-6   w16b "
          f"{W16B_PER_CELL*1e6:+8.4f}e-6   drift {(d_h3.mean()-W16B_PER_CELL)*1e12:+.3f}e-12   "
          f"CV {fast_auc(y, oof_h3):.10f}", flush=True)

    # w16q's control draw: one rng(20260816), permuted() called for a_only THEN rule
    rq = np.random.default_rng(20260816)
    _ = permuted(a_only, rq)
    pq = permuted(cell, rq)
    dq, _, _ = arm_xfit(y, br[ENS4], c, pq, folds)
    print(f"  ens4 CTRL(w16q draw)  {dq.mean()*1e6:+8.4f}e-6   w16q "
          f"{W16Q_CTRL_RULE*1e6:+8.4f}e-6   drift {(dq.mean()-W16Q_CTRL_RULE)*1e12:+.3f}e-12",
          flush=True)
    # w16b's control draw: gather form, rng(4242)
    pb = cell[np.random.default_rng(4242).permutation(n)]
    db, _, _ = arm_xfit(y, br[H3], c, pb, folds)
    print(f"  h3   CTRL(w16b draw)  {db.mean()*1e6:+8.4f}e-6   w16b "
          f"{W16B_CTRL7*1e6:+8.4f}e-6   drift {(db.mean()-W16B_CTRL7)*1e12:+.3f}e-12", flush=True)

    gates = {"ens4_glob": float(arms["glob"].mean() - W16Q_ARMS["glob"]),
             "ens4_a_only": float(arms["a_only"].mean() - W16Q_ARMS["a_only"]),
             "ens4_per_cell": float(arms["per_cell"].mean() - W16Q_ARMS["per_cell"]),
             "h3_per_cell": float(d_h3.mean() - W16B_PER_CELL),
             "ens4_ctrl_w16q": float(dq.mean() - W16Q_CTRL_RULE),
             "h3_ctrl_w16b": float(db.mean() - W16B_CTRL7)}
    ok = all(abs(v) < 1e-12 for v in gates.values())
    print(f"  GATE {'PASSED' if ok else 'FAILED'}  (max |drift| "
          f"{max(abs(v) for v in gates.values())*1e12:.3f}e-12)")
    out["gates"] = gates
    out["gate_passed"] = bool(ok)

    # ------------------------------------------------------------------ PART 1: controls
    print("\n" + "-" * 78)
    print("PART 1  control ENSEMBLE: 7 size-matched permuted-membership draws per base")
    print("-" * 78)
    print("  Same 7 level sizes, membership shuffled, everything else identical. This is the")
    print("  toll for splitting one weight into seven AT RANDOM; the real segmentation has to")
    print("  pay it back before it can claim a gain.", flush=True)
    ctrl = {ENS4: [float(dq.mean())], H3: [float(db.mean())]}
    ctrl_src = {ENS4: ["w16q draw (rng 20260816)"], H3: ["w16b draw (rng 4242)"]}
    for s in CTRL_SEEDS:
        for base in (ENS4, H3):
            pa = permuted(cell, np.random.default_rng(s))
            d, _, _ = arm_xfit(y, br[base], c, pa, folds)
            ctrl[base].append(float(d.mean()))
            ctrl_src[base].append(f"rng {s}")
            print(f"  ctrl {base:16s} seed {s}  {d.mean()*1e6:+8.4f}e-6  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    real = {ENS4: arms["per_cell"], H3: d_h3}
    print()
    out["control"] = {}
    for base in (ENS4, H3):
        a = np.array(ctrl[base])
        se = a.std(ddof=1) / np.sqrt(len(a))
        diff = real[base].mean() - a.mean()
        # se of (real - control mean): real is one number per fold, control is 7 draws
        se_r = real[base].std(ddof=1) / np.sqrt(5)
        se_d = float(np.sqrt(se ** 2 + se_r ** 2))
        print(f"  base {base:16s} real {real[base].mean()*1e6:+7.3f}e-6 (se {se_r*1e6:.3f})   "
              f"control mean {a.mean()*1e6:+7.3f}e-6  sd {a.std(ddof=1)*1e6:.3f}  se {se*1e6:.3f}"
              f"  n {len(a)}")
        print(f"       range [{a.min()*1e6:+.3f}, {a.max()*1e6:+.3f}]e-6   "
              f"REAL - CONTROL {diff*1e6:+7.3f}e-6  +/-{se_d*1e6:.3f}  "
              f"t {diff/se_d:+.2f}")
        out["control"][base] = dict(draws=ctrl[base], src=ctrl_src[base],
                                    mean=float(a.mean()), sd=float(a.std(ddof=1)),
                                    se=float(se), real=float(real[base].mean()),
                                    real_se=float(se_r), diff=float(diff), se_diff=se_d)
    dd = out["control"][H3]["diff"] - out["control"][ENS4]["diff"]
    sed = float(np.sqrt(out["control"][H3]["se_diff"] ** 2
                        + out["control"][ENS4]["se_diff"] ** 2))
    print(f"\n  h3 (real-ctrl) MINUS ens4 (real-ctrl) = {dd*1e6:+.3f}e-6 +/-{sed*1e6:.3f}"
          f"   (published single-draw version: +3.825 - +1.378 = +2.447e-6)")
    out["control"]["h3_minus_ens4_margin"] = dict(d=float(dd), se=sed)

    # ------------------------------------------------------------------ PART 2: nested pick
    print("\n" + "-" * 78)
    print("PART 2  nested-pick stability on the ens4 base (the selection this run does NOT make)")
    print("-" * 78)
    names = ["glob", "a_only", "per_cell"]
    picks, held = [], []
    for f in range(5):
        others = [g for g in range(5) if g != f]
        best = max(names, key=lambda nm: arms[nm][others].mean())
        picks.append(best)
        held.append(arms[best][f])
    held = np.array(held)
    naive = max(names, key=lambda nm: arms[nm].mean())
    print(f"  naive argmax arm = {naive}   xfit {arms[naive].mean()*1e6:+.3f}e-6")
    print(f"  nested picks     = {picks}")
    print(f"  nested delta     = {held.mean()*1e6:+.3f}e-6  se "
          f"{held.std(ddof=1)/np.sqrt(5)*1e6:.3f}")
    print(f"  ARM-SELECTION OPTIMISM {(arms[naive].mean()-held.mean())*1e6:+.3f}e-6   "
          f"stability {max(picks.count(p) for p in set(picks))}/5")
    print("  (the shipped arm is per_cell by S1 regardless of any of this)")
    print("\n  per-fold weights, ens4 base:")
    lv = sorted(set(cell))
    print("    fold  " + "  ".join(f"{v.split()[0]:>6s}" for v in lv))
    for i, w in enumerate(arm_w["per_cell"]):
        print(f"    {i}     " + "  ".join(f"{w[v]:6.4f}" for v in lv))
    print("  per-fold weights, h3 base (w16b's published table, recomputed):")
    for i, w in enumerate(w_h3):
        print(f"    {i}     " + "  ".join(f"{w[v]:6.4f}" for v in lv))
    out["nested"] = dict(picks=picks, naive=naive,
                         naive_delta=float(arms[naive].mean()),
                         nested_delta=float(held.mean()),
                         optimism=float(arms[naive].mean() - held.mean()),
                         fold_weights_ens4=arm_w["per_cell"], fold_weights_h3=w_h3)
    out["arms"] = {k: dict(xfit=float(v.mean()), per_fold=[float(x) for x in v],
                           folds_pos=int((v > 0).sum()), cv=float(fast_auc(y, arm_oof[k])))
                   for k, v in arms.items()}
    out["arms"]["h3_per_cell"] = dict(xfit=float(d_h3.mean()),
                                      per_fold=[float(x) for x in d_h3],
                                      folds_pos=int((d_h3 > 0).sum()),
                                      cv=float(fast_auc(y, oof_h3)))

    # ------------------------------------------------------------------ PART 3: the file
    print("\n" + "-" * 78)
    print("PART 3  full-data fit and the test file")
    print("-" * 78, flush=True)
    ship_cv = fast_auc(y, arm_oof["per_cell"])
    print(f"  shipped arm per_cell, pooled cross-fitted CV {ship_cv:.10f}  "
          f"(w16q stored {W16Q_RULE_CV:.10f})", flush=True)
    w_full = ascend(y, br[ENS4], c, cell, np.arange(n))
    print(f"  full-data weights: { {k: round(v, 4) for k, v in w_full.items()} }", flush=True)

    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    assert (te["id"].to_numpy() == ids).all()
    bp = pd.read_csv(os.path.join(SUB, f"{ENS4}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    assert np.isfinite(bp).all() and len(bp) == N_TEST
    btr = pct(bp)
    ct = np.load(os.path.join(HERE, "w15f_c_test_avg.npy")).astype(np.float64)
    assert ct.shape == (N_TEST,)
    cell_te = rule_cells(te)
    assert set(cell_te) == set(w_full), (sorted(set(cell_te)), sorted(w_full))
    shares = {v: float((cell_te == v).mean()) for v in sorted(set(cell_te))}
    print("  test-side cell shares:", {k.split()[0]: round(v, 4) for k, v in shares.items()})

    pred = apply_w(btr, ct, cell_te, w_full, np.arange(N_TEST))
    order = np.lexsort((ids, btr, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub.shape == (N_TEST, 2)
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, f"{TAG}.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, f"oof_{TAG}.npy"), arm_oof["per_cell"] / n)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == f"{TAG}.csv":
            continue
        try:
            o = pd.read_csv(os.path.join(SUB, f))
        except Exception:
            continue
        if list(o.columns) != ["id", TARGET] or len(o) != N_TEST:
            continue
        ov = o.set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        if np.array_equal(rankdata(ov), r):
            dupes.append(f)
    print(f"\n  wrote {path}  rows {len(sub):,}  distinct {sub[TARGET].nunique():,}  "
          f"range [{strict.min():.3e}, {strict.max():.3f}]")
    print(f"  rank-identical to an existing submission file? {dupes or 'no'}")
    for other in ("w16q_ens4avg", "w16b_cellweight", "w16i_schemeavg", ENS4, H3):
        op = os.path.join(SUB, f"{other}.csv")
        if not os.path.exists(op):
            continue
        ov = pd.read_csv(op).set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<18} {np.corrcoef(r, rankdata(ov))[0,1]:.7f}   "
              f"rows differing in rank {int((rankdata(ov) != r).sum()):,}")
    out["file"] = dict(path=path, cv=float(ship_cv), dupes=dupes,
                       full_weights={str(k): float(v) for k, v in w_full.items()},
                       test_shares=shares)

    print(f"\n  S4 pre-registered LB = CV + 0.0010284829 = {ship_cv + 0.0010284829:.10f}"
          f"  ->  {round(ship_cv + 0.0010284829, 5):.5f}")

    # ------------------------------------------------------------------ PART 4: third pair
    print("\n" + "-" * 78)
    print("PART 4  a THIRD matched h3/ens4 pair, at p7, on w16s's simulated private slice")
    print("-" * 78)
    print("  w16s settled this with the p0 and p23 pairs. The p7 pair could not be built then")
    print("  because the ens4-side p7 object did not exist. Per S5 this cannot move anything.")
    vecs = {"blend159av_h3": raw[H3], "blend159av": raw[ENS4],
            "w16i_schemeavg": np.load(os.path.join(SUB, "oof_w16i_schemeavg.npy")),
            "w16q_ens4avg": np.load(os.path.join(SUB, "oof_w16q_ens4avg.npy")),
            "w16b_cellweight": oof_h3, TAG: arm_oof["per_cell"]}
    nm = list(vecs)
    cvs = {k: fast_auc(y, vecs[k].astype(np.float64)) for k in nm}
    pk = {k: prep(vecs[k].astype(np.float64), y) for k in nm}
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    A = np.zeros((REPS, len(nm)))
    B = np.zeros((REPS, len(nm)))
    for i in range(REPS):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool)
        m[idx[n_pub:]] = True
        mp = np.zeros(n, dtype=bool)
        mp[idx[:n_pub]] = True
        for k, name in enumerate(nm):
            A[i, k] = subset_auc(pk[name], m)
            B[i, k] = subset_auc(pk[name], mp)
    pairs = [("blend159av_h3", "blend159av", "p0"),
             ("w16b_cellweight", TAG, "p7"),
             ("w16i_schemeavg", "w16q_ens4avg", "p23")]
    print(f"\n  private-sized mask {N_TEST-n_pub:,} rows | public-sized {n_pub:,} rows | "
          f"{REPS} reps, seed {SEED}")
    out["pairs"] = []
    for a, b, lab in pairs:
        d = A[:, nm.index(a)] - A[:, nm.index(b)]
        du = B[:, nm.index(a)] - B[:, nm.index(b)]
        se = d.std(ddof=1) / np.sqrt(REPS)
        gate = W16S_PAIRS.get((a, b))
        g = ""
        if gate:
            g = (f"   [w16s {gate[0]*1e6:+.2f}e-6 / P {gate[1]:.3f}  "
                 f"drift {(d.mean()-gate[0])*1e12:+.3f}e-12]")
        print(f"  {lab:>3s}  {a:18s} - {b:18s}  CV d {(cvs[a]-cvs[b])*1e6:+6.2f}e-6   "
              f"sim d {d.mean()*1e6:+6.2f}e-6 +/-{se*1e6:4.2f}   draw sd {d.std(ddof=1)*1e6:5.2f}"
              f"   P(h3 better) {(d > 0).mean():.3f}{g}")
        print(f"       public-sized: mean {du.mean()*1e6:+6.2f}e-6  draw sd "
              f"{du.std(ddof=1)*1e6:5.2f}e-6  P(slice REVERSES it) {(du <= 0).mean():.3f}")
        out["pairs"].append(dict(label=lab, h3=a, ens4=b, cv_d=float(cvs[a] - cvs[b]),
                                 sim_d=float(d.mean()), se=float(se),
                                 draw_sd=float(d.std(ddof=1)), p_h3=float((d > 0).mean()),
                                 pub_d=float(du.mean()), pub_sd=float(du.std(ddof=1)),
                                 pub_p_reverse=float((du <= 0).mean())))
    out["cv"] = {k: float(v) for k, v in cvs.items()}

    # ------------------------------------------------------------------ S3, evaluated
    print("\n" + "-" * 78)
    print("PRE-REGISTERED RULES, evaluated mechanically")
    print("-" * 78)
    move = bool(ship_cv > W16I_CV)
    print(f"  S1 shipped arm = per_cell, unconditional                     -> "
          f"{os.path.basename(path)}")
    if dupes:
        print(f"  S2 rank-identical to {dupes}  -> SHIP THE FALLBACK {FALLBACK}.csv INSTEAD")
    else:
        print(f"  S2 rank-identical to a sent file? no  -> fallback {FALLBACK} not needed")
    print(f"  S3 move WANTED slot 1? CV {ship_cv:.10f} > w16i {W16I_CV:.10f}  -> {move}")
    print(f"     (predicted False; second slot untouched under any outcome)")
    print(f"  S5 Part 4 is a confirmation and moved nothing.")
    out["move_wanted"] = move

    json.dump(out, open(os.path.join(HERE, f"{TAG}.json"), "w"), indent=1)
    print(f"\nwrote experiments/{TAG}.json   total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
