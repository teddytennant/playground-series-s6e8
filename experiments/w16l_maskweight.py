"""w16l: the train/test missingness shift as a TRAINING WEIGHT -- the last untried class.

WHERE THIS COMES FROM
---------------------
w16a's item 2, restated by w16c §7 item 2 and w16h §6 item 1: "build a genuinely
transductive member". Three slots have now deferred it and it is the only candidate CLASS
this workspace has not closed. The concrete hook, unchanged since w15c:

  * train and test were masked SEPARATELY. Adversarial train-vs-test is 100% the mask
    (mask-only AUC 0.56472 vs full-frame 0.56280); the observed VALUES are identical
    between the splits to a 382,289-row instrument's precision (values-only adv AUC
    0.49750 against a train-vs-train floor of 0.50059).
  * every one of the twelve columns' NaN rate shifts, up to |z| = 44, while the per-row
    missing COUNT distribution matches (mean 1.2589 train vs 1.2729 test). The budget is
    conserved; its allocation across columns is not.
  * reweighting the OOF POOL to the test mask distribution by the exact 4096-pattern
    density ratio moves estimated AUC +905e-6, which is 88% of the workspace's
    long-standing +1.0e-3 CV->LB gap. Bootstrap over the test rows: +903e-6, sd 27e-6,
    34 sigma.

That last number has only ever been used as a DIAGNOSTIC -- an estimator of test-set AUC
applied to finished files. w15c's own closing note says so, and its "next run" item 2 says
"do not build an importance-weighted RETRAIN off §4 expecting a gain", on the grounds that
the shift is additive per column, correlated -0.97 with CV, and does not move the argmax.
**Every one of those three facts is about the shift as an EVALUATION reweighting.** None of
them is a statement about what happens when the same weights enter the FIT. That is the
different object w16a named, and it has never been built.

WHY IT COULD DO ANYTHING, STATED HONESTLY BEFORE THE RUN
--------------------------------------------------------
The masking is MCAR with respect to the target (w15c §6) and the value distributions are
identical between splits. So P(y | observed values, mask) is the SAME in train and test and
only P(mask) differs -- textbook covariate shift. Under a correctly specified model,
importance weighting is asymptotically worthless and strictly raises variance. The gain, if
there is one, comes entirely from MISSPECIFICATION: a finite-capacity fit allocates its
capacity in proportion to the training measure, and the training measure is the wrong one.
Test carries MORE rows with the generator's two drivers observed (daily 11.07% NaN vs
13.86%, social 16.00% vs 19.38%), and w14d measured those rows as far more rankable
(cell A 0.984 vs both-drivers-missing 0.912).

So the prior is: small, and quite possibly negative, because ESS falls to ~93%. This run
exists to MEASURE it, not to hope.

WHAT IS FITTED
--------------
The object the workspace actually ships: the `h3` rank-ensemble of three cross-fitted
159-member logistic stacks (hybrid / rankraw / rescale), i.e. exactly `blend159av_h3`,
CV 0.97004917, LB 0.97105. `sklearn`'s `LogisticRegression.fit` takes `sample_weight`, so
the ONLY difference between arms is that vector. Same members, same frozen SKF5 seed42
folds, same C, same transforms, same rank-ensemble, same everything else.

  unw   w = 1                       CONTROL. Must reproduce submissions/oof_blend159av_h3.npy.
  imp   w = p_test(mask)/p_train(mask)   the transductive arm. The weights are computed from
                                    the 296,302 UNLABELLED test rows, which is what makes
                                    this transductive; no label enters them.
  anti  w = p_train(mask)/p_test(mask)   the mirror. This is the control that matters.

**Why `anti` and not a permutation control.** For a training weight the informative null is
directional. A permuted-pattern control (w15c ran one and disowned it: it destroys the
smoothness of the weight function, not just its direction, and inflates variance threefold)
answers "would an arbitrary reweighting move the fit?". `anti` answers the question that
decides this: is the SIGN of the mechanism real? It has the identical weight-function
smoothness, the identical marginal spread up to inversion, and the identical ESS penalty --
it differs only in which direction the mass moves. If the mechanism is real the three arms
are ordered imp > unw > anti on every fold; if it is noise they scatter. The imp-anti
contrast carries twice the effect at the same variance and is the highest-powered reading
available, and it is free because `anti` costs one more fit per fold.

TWO ENDPOINTS, BOTH REPORTED, NEITHER ONE ALLOWED TO BE PICKED AFTER THE FACT
----------------------------------------------------------------------------
  plain  cross-fitted OOF AUC on the labelled rows as they are. This is the workspace's
         standard CV and the number every ladder and every deadline decision uses.
  wtd    the same AUC under the mask-importance measure (w15c's `wauc`). This is the
         unbiased estimator of AUC on the TEST distribution, and it is the criterion the
         `imp` arm is by construction trying to maximise.

`imp` should win `wtd` almost mechanically -- that is what it optimises -- so a win there
alone proves nothing. The claim only becomes interesting if `imp` also holds `plain`, or
if the LB reading beats what `plain` predicts off the ladder.

PRE-REGISTERED, FIXED HERE BEFORE THE RUN, NOT REVISITED AFTERWARDS
-------------------------------------------------------------------
1. **SHIP `w16l_maskw_h3` -- the `imp` arm's h3 file -- UNCONDITIONALLY, whatever its CV.**
   Not the best of the three arms, not the best endpoint, not the best transform. Three
   slots in a row have now been burned by an argmax nobody nested (w16c: arm selection
   +1.78e-6; w16i: scheme selection +1.55e-6, and w16i caught the identical bug a third
   time inside its own audit script). The way not to make it a fourth time is to name the
   file before the numbers exist. `unw` is a reproduction of a file already sent and `anti`
   is a control that is not meant to be good, so there is exactly one candidate here.
2. **The deadline pick moves ONLY IF `imp` beats `unw` on the PLAIN cross-fitted CV AND
   `imp` - `anti` > 0 on the PLAIN cross-fitted CV.** Both conditions, plain endpoint only.
   Anything else and `WANTED` stays {w16i_schemeavg.csv, blend159av_h3.csv}.
3. The LB prediction is written into the submission message from the h3 ladder
   (h3 / top cluster -> 0.97105, record 11/11) using the PLAIN CV, before sending.
   `blend159av_h3` is the same object with w = 1 and it returned 0.97105, so this is as
   close to a paired LB reading as this competition allows: two files differing in nothing
   but the training weight vector, one of them already scored.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from blend_lab import HONEST_DROP, load_all, rk  # noqa: E402
from w15c_shift import FEAT, maskcode, wauc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("hybrid", "rankraw", "rescale")
# blend159av: the three xgb_latcat seeds are replaced by their probability mean
# `xgb_latcat_avg3`, so the individual seeds drop out on top of HONEST_DROP. The three
# original-dataset members (`orig_bin`, `orig_binm`, `w15d_origrep_r`) were added to `oof/`
# AFTER blend159av was built -- they are what make blend160orig/blend160origm 160-member
# sets -- so they drop too, or this is a 162-member stack and not the base the whole w16
# correction family sits on. The printed member count MUST read 159.
DROP = tuple(HONEST_DROP) + ("xgb_latcat", "xgb_latcat_s17", "xgb_latcat_s23",
                             "orig_bin", "orig_binm", "w15d_origrep_r")
BASE = "blend159av_h3"
C = 1.0
CLIP_Q = 0.999   # w15c's clip, kept verbatim so the weights are the same object


def make_weights(tr, te):
    """The exact 4096-pattern density ratio, both directions. w15c_shift.main() verbatim."""
    ktr, kte = maskcode(tr), maskcode(te)
    ctr = np.bincount(ktr, minlength=4096).astype(float)
    cte = np.bincount(kte, minlength=4096).astype(float)
    ptr, pte = ctr / ctr.sum(), cte / cte.sum()

    def ratio_to(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, 0.0)
        w = r[ktr]
        hi = np.quantile(w[w > 0], CLIP_Q)
        w = np.clip(w, 0, hi)
        return w / w.mean()

    w_imp = ratio_to(pte, ptr)    # toward the test mask distribution
    w_anti = ratio_to(ptr, pte)   # away from it -- the mirror
    return w_imp, w_anti, ptr, pte, ktr, kte


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)

    w_imp, w_anti, ptr, pte, ktr, kte = make_weights(tr, te)
    print(f"train {n:,} rows, test {len(te):,} rows", flush=True)
    print(f"distinct mask patterns: train {int((np.bincount(ktr, minlength=4096) > 0).sum())} "
          f"test {int((np.bincount(kte, minlength=4096) > 0).sum())} of 4096", flush=True)
    for tag, w in (("imp", w_imp), ("anti", w_anti)):
        print(f"  w[{tag:4s}] sd {w.std():.4f} min {w.min():.4f} max {w.max():.4f} "
              f"ESS {ess(w):,.0f} of {n:,} ({ess(w)/n:.2%})", flush=True)
    print(f"  corr(w_imp, w_anti) = {np.corrcoef(w_imp, w_anti)[0,1]:+.4f}", flush=True)

    arms = {"unw": np.ones(n), "imp": w_imp, "anti": w_anti}

    names, y2, mats, te2 = load_all(KINDS, set(DROP))
    assert np.array_equal(y, y2)
    print(f"{len(names)} members, drop={list(DROP)}", flush=True)
    assert len(names) == 159, (
        f"member set is {len(names)}, not the 159 that produced {BASE}; the control arm "
        "would not reproduce the base and nothing downstream would be comparable")
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    # per-arm, per-transform cross-fitted OOF; test-side full fit only for the shipped arm
    oof = {a: {} for a in arms}
    tst = {}
    fold_auc = {a: {k: [] for k in KINDS} for a in arms}
    for k in KINDS:
        Z, Zt = mats[k]
        for a, w in arms.items():
            mo = np.zeros(n)
            for f, (itr, iva) in enumerate(folds):
                tf = time.time()
                m = LogisticRegression(max_iter=5000, C=C).fit(Z[itr], y[itr],
                                                               sample_weight=w[itr])
                mo[iva] = m.decision_function(Z[iva])
                fold_auc[a][k].append(float(roc_auc_score(y[iva], mo[iva])))
                print(f"    [{k:8s}/{a:4s}] fold {f} AUC {fold_auc[a][k][-1]:.6f} "
                      f"({time.time()-tf:.0f}s)", flush=True)
            oof[a][k] = mo
            print(f"  cross-fitted stack_{k:8s} {a:4s}  plain {roc_auc_score(y, mo):.8f}  "
                  f"wtd {wauc(y, mo, w_imp):.8f}", flush=True)
        # test side: the shipped arm only (pre-registration item 1)
        full = LogisticRegression(max_iter=5000, C=C).fit(Z, y, sample_weight=w_imp)
        tst[k] = full.decision_function(Zt)
        print(f"  full-data fit written for imp/{k} ({time.time()-t0:.0f}s)", flush=True)
        del Z, Zt
        mats[k] = None

    # h3 rank-ensembles
    res = {}
    h3_oof = {}
    for a in arms:
        o = np.mean([rk(oof[a][k]) for k in KINDS], 0)
        h3_oof[a] = o
        res[a] = dict(plain=float(roc_auc_score(y, o)),
                      wtd=float(wauc(y, o, w_imp)))
    t_ens = np.mean([rk(tst[k]) for k in KINDS], 0)

    # the control must reproduce the shipped base
    ref = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    rep = float(np.corrcoef(rankdata(ref), rankdata(h3_oof["unw"]))[0, 1])
    print(f"\ncontrol reproduction of {BASE}: stored CV {roc_auc_score(y, ref):.8f}  "
          f"rebuilt CV {res['unw']['plain']:.8f}  rank corr {rep:.8f}", flush=True)

    print("\n=== h3 rank-ensemble, cross-fitted on the frozen folds ===")
    print(f"{'arm':<6}{'plain CV':>14}{'vs unw':>12}{'wtd CV':>14}{'vs unw':>12}")
    for a in ("unw", "imp", "anti"):
        print(f"{a:<6}{res[a]['plain']:>14.8f}"
              f"{(res[a]['plain']-res['unw']['plain'])*1e6:>+11.3f}e-6"
              f"{res[a]['wtd']:>14.8f}"
              f"{(res[a]['wtd']-res['unw']['wtd'])*1e6:>+11.3f}e-6")

    # paired per-fold deltas on the h3 object -- the honest uncertainty
    print("\n=== paired per-fold deltas on the h3 object (e-6) ===")
    perfold = {}
    for a in ("imp", "anti"):
        for endp in ("plain", "wtd"):
            d = []
            for f, (itr, iva) in enumerate(folds):
                if endp == "plain":
                    va = roc_auc_score(y[iva], h3_oof[a][iva])
                    vb = roc_auc_score(y[iva], h3_oof["unw"][iva])
                else:
                    va = wauc(y[iva], h3_oof[a][iva], w_imp[iva])
                    vb = wauc(y[iva], h3_oof["unw"][iva], w_imp[iva])
                d.append((va - vb) * 1e6)
            d = np.array(d)
            se = d.std(ddof=1) / np.sqrt(len(d))
            perfold[f"{a}_{endp}"] = dict(per_fold=[float(v) for v in d],
                                          mean=float(d.mean()), se=float(se),
                                          t=float(d.mean() / se) if se else 0.0,
                                          folds_pos=int((d > 0).sum()))
            print(f"  {a:4s} - unw  [{endp:5s}]  " + " ".join(f"{v:+7.3f}" for v in d) +
                  f"   mean {d.mean():+7.3f}  se {se:6.3f}  t {d.mean()/se if se else 0:+6.2f}"
                  f"  {int((d>0).sum())}/5")

    # the sign test the anti arm exists for
    gap_plain = (res["imp"]["plain"] - res["anti"]["plain"]) * 1e6
    gap_wtd = (res["imp"]["wtd"] - res["anti"]["wtd"]) * 1e6
    print(f"\nimp - anti  plain {gap_plain:+.3f}e-6   wtd {gap_wtd:+.3f}e-6")
    ordered_plain = res["imp"]["plain"] > res["unw"]["plain"] > res["anti"]["plain"]
    ordered_wtd = res["imp"]["wtd"] > res["unw"]["wtd"] > res["anti"]["wtd"]
    print(f"monotone imp > unw > anti:  plain {ordered_plain}   wtd {ordered_wtd}")

    # pre-registered decision, evaluated mechanically
    move_pick = (res["imp"]["plain"] > res["unw"]["plain"]) and (gap_plain > 0)
    print(f"\nPRE-REGISTERED RULE 2 (move the deadline pick): {move_pick}")

    # write the shipped file -- unconditional, pre-registration item 1
    name = "w16l_maskw_h3"
    sub = pd.DataFrame({"id": te["id"].to_numpy(), TARGET: t_ens})
    sub.to_csv(os.path.join(SUB, f"{name}.csv"), index=False)
    np.save(os.path.join(SUB, f"oof_{name}.npy"), h3_oof["imp"])
    np.save(os.path.join(SUB, f"test_{name}.npy"), t_ens)
    print(f"\nwrote submissions/{name}.csv  rows={len(sub):,}  "
          f"plain CV {res['imp']['plain']:.8f}  wtd CV {res['imp']['wtd']:.8f}", flush=True)

    json.dump(dict(members=len(names), drop=list(DROP), kinds=list(KINDS), C=C,
                   ess={a: ess(w) for a, w in arms.items()},
                   h3=res, per_fold_stack=fold_auc, paired=perfold,
                   imp_minus_anti=dict(plain=gap_plain, wtd=gap_wtd),
                   monotone=dict(plain=bool(ordered_plain), wtd=bool(ordered_wtd)),
                   control_reproduction=dict(stored_cv=float(roc_auc_score(y, ref)),
                                             rebuilt_cv=res["unw"]["plain"],
                                             rank_corr=rep),
                   move_pick=bool(move_pick), shipped=name),
              open(os.path.join(HERE, "w16l_maskweight.json"), "w"), indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
