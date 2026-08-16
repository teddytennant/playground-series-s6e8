"""w16h: sweep the workspace for decisions that PICKED an argmax where they could have AVERAGED.

WHY
---
Two of this wave's three reversals have the same shape. w15i overturned "the CV->LB regression
is dead" by noticing w14a had fitted it pooled over a strong categorical. w16c overturned
w16b's headline +6.195e-6 by noticing the arm had been *selected* on the same cross-fitted folds
it was scored on; averaging the arms instead of picking one deleted the selection step and
reproduced the number honestly (w16f_armavg, CV 0.9700554, LB 0.97107).

Slot 2's parting instruction was to sweep the rest of the workspace for that shape. The three
places named were `blendtop3`, the transform families, and the seed/fold stacks. Seeds and folds
are already averaged (blend159av, the frozen SKF5). The other two are not, and there is a third
nobody named: the MEMBER SET itself.

THE THREE DIMENSIONS TESTED HERE, all on the frozen SKF5 seed42 folds
--------------------------------------------------------------------
D1 MEMBER SET. Six zero-parameter h3 stacks exist on disk, one per member set
   (156/158/159/159av/160orig/160origm). Their CV spans 3.0e-6 and they rank-correlate
   0.99992+. `blend159av_h3` -- the incumbent deadline pick and the base of every corrected
   file this workspace has ever shipped -- is the ARGMAX of that six. Nobody ever asked what
   that argmax costs.

D2 TRANSFORM SUBSET. The 2026-08-11 entry enumerated all 11 subsets of size >= 2 of
   {logit, hybrid, rankraw, rescale} on blend158 and recorded "h3 is the argmax of the whole
   lattice". That is a pick over 11 candidates whose top four sit inside 2e-6. Note the OTHER
   finding of that run -- every subset containing logit is beaten by the same subset without
   it, 7/7 -- is structural and is NOT touched by this: it survives regardless of the argmax.

D3 TOP-K. `blendtop3` averages the three joint-top h3 files. That is halfway to the fix: it
   averages, but only over candidates it first SELECTED by CV. Sweeping k from 1 (pure pick)
   to 6 (pure average) prices the residual selection it kept.

THE INSTRUMENT, identical in all three
--------------------------------------
Leave-one-fold-out nesting, exactly as w16c did for the arms. For each fold f, run the
selection rule on the other four folds only, then read the chosen object on fold f. The
difference between that and the naive fold-mean of the globally-chosen argmax IS the selection
optimism, measured rather than guessed. The averaged object needs no nesting -- it makes no
choice -- so its fold-mean is honest as it stands.

Candidate sets are defined by a rule fixed here BEFORE the run, not by inspection:
D1 is every zero-fitted-parameter `*_h3` OOF on disk, minus `w14a_repro159av_h3` (a
reproduction of another candidate's member set, not a distinct one) and minus
`blend159av_wh3` (two fitted parameters, excluded for the same reason `blendtop3` excluded it).
D2 is the full 15-subset lattice and, separately, the 7-subset logit-free sublattice.
D3 is D1's candidate set.

Nothing in this script is fitted, and nothing in it looks at the public leaderboard.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 296_302

# D1 / D3 candidate set -- the rule, fixed before the run (see docstring).
H3_FAMILY = [
    "blend156_h3",
    "blend158_h3",
    "blend159_h3",
    "blend159av_h3",
    "blend160orig_h3",
    "blend160origm_h3",
]
INCUMBENT = "blend159av_h3"
TRANSFORMS = ["logit", "hybrid", "rankraw", "rescale"]
MEMBERSET_FOR_D2 = "blend159av"


def rk(x):
    return rankdata(x)


def foldmean(y, folds, s):
    return np.array([fast_auc(y[iva], s[iva]) for _, iva in folds])


def nested_pick(y, folds, cands):
    """For each fold: argmax on the OTHER four folds, read on this one. Returns (deltas, picks)."""
    names = list(cands)
    out, picks = [], []
    for itr, iva in folds:
        best, bn = -1.0, None
        for n in names:
            a = fast_auc(y[itr], cands[n][itr])
            if a > best:
                best, bn = a, n
        picks.append(bn)
        out.append(fast_auc(y[iva], cands[bn][iva]))
    return np.array(out), picks


def report(tag, y, folds, cands, avg_vec, naive_name):
    """Print the pick-vs-average comparison for one dimension."""
    naive = foldmean(y, folds, cands[naive_name])
    nest, picks = nested_pick(y, folds, cands)
    avg = foldmean(y, folds, avg_vec)
    print(f"\n=== {tag} ===")
    print(f"  candidates ({len(cands)}): {', '.join(sorted(cands))}")
    print(f"  global argmax = {naive_name}")
    print(f"  nested picks  = {picks}")
    d_opt = naive - nest
    d_avg = avg - nest
    d_inc = avg - naive
    for label, d in (("naive argmax - nested pick  (= SELECTION OPTIMISM)", d_opt),
                     ("average      - nested pick", d_avg),
                     ("average      - naive argmax", d_inc)):
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else float("nan")
        print(f"    {label:<52} {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t(4df) {t:+5.2f}  {int((d > 0).sum())}/5  "
              f"[{' '.join(f'{x*1e6:+6.2f}' for x in d)}]")
    return dict(naive_name=naive_name, picks=picks,
                optimism=float(d_opt.mean()), avg_minus_nested=float(d_avg.mean()),
                avg_minus_naive=float(d_inc.mean()),
                cv_naive=float(fast_auc(y, cands[naive_name])), cv_avg=float(fast_auc(y, avg_vec)))


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    res = {}

    def oof(name):
        return np.load(os.path.join(SUB, f"oof_{name}.npy")).astype(np.float64)

    # ------------------------------------------------------------------ D1 member set
    d1 = {n: rk(oof(n)) for n in H3_FAMILY}
    print("D1 candidate CVs:")
    for n in H3_FAMILY:
        print(f"    {n:<20} {fast_auc(y, d1[n]):.8f}")
    d1_avg = np.mean([d1[n] for n in H3_FAMILY], axis=0)
    d1_argmax = max(H3_FAMILY, key=lambda n: fast_auc(y, d1[n]))
    res["D1_memberset"] = report("D1  MEMBER SET (6 zero-parameter h3 stacks)",
                                 y, folds, d1, d1_avg, d1_argmax)
    print(f"  CV of the 6-way average          {fast_auc(y, d1_avg):.8f}")
    print(f"  CV of blendtop3 (top-3 selected) {fast_auc(y, rk(oof('blendtop3'))):.8f}")
    print(f"  CV of {INCUMBENT} (incumbent)  {fast_auc(y, d1[INCUMBENT]):.8f}")

    # ------------------------------------------------------------------ D2 transform subsets
    tvec = {t: rk(oof(f"{MEMBERSET_FOR_D2}_{t}")) for t in TRANSFORMS}
    for lattice_tag, pool in (("full 15-subset lattice", TRANSFORMS),
                              ("logit-free 7-subset sublattice", TRANSFORMS[1:])):
        subs = {}
        for r in range(1, len(pool) + 1):
            for combo in itertools.combinations(pool, r):
                subs["+".join(combo)] = np.mean([tvec[t] for t in combo], axis=0)
        savg = np.mean([rk(v) for v in subs.values()], axis=0)
        argm = max(subs, key=lambda k: fast_auc(y, subs[k]))
        res[f"D2_{lattice_tag.split()[0]}"] = report(
            f"D2  TRANSFORM SUBSET on {MEMBERSET_FOR_D2} ({lattice_tag})",
            y, folds, subs, savg, argm)

    # ------------------------------------------------------------------ D3 top-k
    print("\n=== D3  TOP-K over the D1 family (blendtop3 is k=3) ===")
    d3 = {}
    for k in range(1, len(H3_FAMILY) + 1):
        per_fold = []
        for itr, iva in folds:
            order = sorted(H3_FAMILY, key=lambda n: -fast_auc(y[itr], d1[n][itr]))
            sel = np.mean([d1[n] for n in order[:k]], axis=0)
            per_fold.append(fast_auc(y[iva], sel[iva]))
        per_fold = np.array(per_fold)
        d3[k] = per_fold
        print(f"  k={k}  nested fold-mean {per_fold.mean():.8f}  "
              f"[{' '.join(f'{x:.6f}' for x in per_fold)}]")
    base = d3[len(H3_FAMILY)]
    print("  paired vs k=6 (the pure average, which selects nothing):")
    for k in range(1, len(H3_FAMILY)):
        d = base - d3[k]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"    k=6 - k={k}  {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se if se > 0 else float('nan'):+5.2f}  {int((d > 0).sum())}/5")
    res["D3_topk"] = {str(k): float(v.mean()) for k, v in d3.items()}

    # ------------------------------------------------------------------ build the D1 average
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    ranks = []
    for n in H3_FAMILY:
        v = pd.read_csv(os.path.join(SUB, f"{n}.csv")).set_index("id") \
            .reindex(ids)[TARGET].to_numpy(np.float64)
        assert np.isfinite(v).all()
        ranks.append(rankdata(v))
    test_avg = np.mean(ranks, axis=0)

    np.save(os.path.join(SUB, "oof_w16h_h3av6.npy"), d1_avg / len(y))
    order = np.lexsort((ids, ranks[H3_FAMILY.index(INCUMBENT)], test_avg))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all() and sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16h_h3av6.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, "test_w16h_h3av6.npy"), strict)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16h_h3av6.csv":
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
    print(f"\n  wrote {path}  CV {fast_auc(y, d1_avg):.8f}")
    print(f"  rank-identical to an existing submission? {dupes or 'no'}")
    for other in (INCUMBENT, "blendtop3"):
        ov = pd.read_csv(os.path.join(SUB, f"{other}.csv")).set_index("id") \
            .reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<16} {np.corrcoef(r, rankdata(ov))[0,1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")
    res["dupes"] = dupes
    json.dump(res, open(os.path.join(HERE, "w16h_pickavg.json"), "w"), indent=1)
    print("  wrote experiments/w16h_pickavg.json")


if __name__ == "__main__":
    main()
