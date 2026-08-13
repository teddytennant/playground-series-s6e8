"""A labelled stand-in for the leaderboard: does the OOF instrument mis-rank the transforms?

WHY THIS EXISTS
---------------
The deadline pick currently rests on one claim: the `logit` stack's cross-fitted CV
UNDERSTATES its test AUC, because an OOF cell is one fold model's output while a test cell
is a 5-fold average, so more OOF cells land on `to_logit`'s clip plateau. The stack is
therefore fitted and scored on the damaged side and predicts on the clean side.

Everything supporting that today is measured against the public leaderboard -- three
`logit - hybrid` LB contrasts, and a bias magnitude (8.9e-5) calibrated on them. That is
structurally the Rogii failure: a correction fitted to public feedback moving the final
pick. The mechanism deserves an instrument that never touches the leaderboard.

THE DESIGN
----------
Hold out 20% of `train` as a pseudo-test set and never fit anything on it. On the
remaining 80%, run the real pipeline exactly:

    pseudo-OOF   = 5-fold cross-validated predictions   (each cell: ONE fold model)
    pseudo-test  = mean of the 5 fold models            (each cell: FIVE models averaged)

That is the identical asymmetry the real members carry, and here BOTH sides are labelled.
So `cv` (pseudo-OOF stack AUC) is the analogue of our CV and `test` (pseudo-test stack AUC)
is the analogue of the LB -- except it is 138k labelled rows we can score ourselves, with
no daily cap and no fixed slice.

The quantity under test is the same one the journal measures:

    displacement = (test - cv)[logit] - (test - cv)[hybrid]

claimed at +9.7e-5 on the real data. Note `test - cv` is not itself meaningful (it absorbs
the split's own level shift); only DIFFERENCES between transforms are, because every
transform is scored on the same rows from the same member matrix.

THE MATCHED CONTROL
-------------------
A displacement could also come from the split, from the member set, or from the stacker.
So the saturating members are dialled in as a dose: the same clean members are stacked
with 0, 1, 2, then all 4 of the clipping members added. If the mechanism is real the
displacement must be ~0 at dose 0 and grow with dose. A flat profile falsifies it, and
that is the outcome this script is built to be able to return.

THE BLENDING QUESTION, WITH GROUND TRUTH
----------------------------------------
Same harness answers today's angle. Weights over the four transform stacks are searched on
the pseudo-OOF -- exactly as `transform_weights.py` searches them on the real OOF -- and
then scored on the pseudo-test. For the first time we can see whether an OOF-fitted blend
weight generalises, and in particular whether the search underweights `logit` because the
criterion it maximises is the damaged one.

    oofsim.py --seed 7                  # train members, then evaluate
    oofsim.py --seed 7 --eval-only      # re-evaluate from the cached member matrices
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CAT, NUM, ROOT, TARGET, load_raw  # noqa: E402
from stack import transform  # noqa: E402

KINDS = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")
CACHE = os.path.join(ROOT, "cache", "oofsim")
os.makedirs(CACHE, exist_ok=True)

# Members are declared in two groups. `CLEAN` are the ones expected to sit at
# sd_test/sd_oof ~= 1.00 with nothing on the clip plateau; `SAT` are chosen to saturate --
# they are the sim's stand-ins for rf/et/naji03/the bolt_lookup family, which are the real
# pack's asymmetric members. The dose experiment adds SAT one at a time in this order.
CLEAN = ("lgb_fast", "lgb_slow", "lgb_stump", "xgb_d6", "hgb", "logreg")
SAT = ("rf", "et", "lookup", "lgb_deep")


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


# --------------------------------------------------------------------------- features


def build_frames(tr):
    """Raw 12 columns only. Two views: NaN-native (boosters) and imputed (sklearn)."""
    X = tr[NUM].astype("float32").copy()
    for c in CAT:
        X[c] = tr[c].astype("category").cat.codes.astype("float32")
        X.loc[X[c] < 0, c] = np.nan
    Xn = X.to_numpy(np.float32)

    med = np.nanmedian(Xn, axis=0)
    Xi = np.where(np.isnan(Xn), med, Xn)
    miss = np.isnan(Xn).astype(np.float32)
    Xi = np.column_stack([Xi, miss])
    # standardise the imputed view so logreg converges and the trees are unaffected
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd <= 0] = 1.0
    Xs = ((Xi - mu) / sd).astype(np.float32)
    return Xn, Xi.astype(np.float32), Xs


# --------------------------------------------------------------------------- members


def fit_member(name, Xn, Xi, Xs, y, itr, iva, ite):
    """Fit one member on `itr`; return (pred on iva, pred on ite). Deterministic."""
    import lightgbm as lgb
    import xgboost as xgb

    def lgbm(**kw):
        p = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                 min_child_samples=100, feature_fraction=1.0, bagging_fraction=1.0,
                 num_threads=16, verbosity=-1, deterministic=True, force_row_wise=True,
                 seed=1)
        p.update(kw)
        n = p.pop("n_rounds")
        d = lgb.Dataset(Xn[itr], label=y[itr])
        m = lgb.train(p, d, num_boost_round=n)
        return m.predict(Xn[iva]), m.predict(Xn[ite])

    if name == "lgb_fast":
        return lgbm(n_rounds=400)
    if name == "lgb_slow":
        return lgbm(n_rounds=1200, learning_rate=0.02, num_leaves=31, min_child_samples=200)
    if name == "lgb_stump":
        return lgbm(n_rounds=1200, num_leaves=8, max_depth=3, min_child_samples=200)
    if name == "lgb_deep":
        # deliberately over-capacity: drives predictions to the unit-interval endpoints,
        # which is the whole point of including it
        return lgbm(n_rounds=600, num_leaves=511, min_child_samples=2, learning_rate=0.08)

    if name == "xgb_d6":
        m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                              min_child_weight=50, tree_method="hist", n_jobs=16,
                              random_state=1, eval_metric="logloss")
        m.fit(Xn[itr], y[itr])
        return m.predict_proba(Xn[iva])[:, 1], m.predict_proba(Xn[ite])[:, 1]

    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=31, random_state=1)
        m.fit(Xn[itr], y[itr])
        return m.predict_proba(Xn[iva])[:, 1], m.predict_proba(Xn[ite])[:, 1]

    if name == "logreg":
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(Xs[itr], y[itr])
        return m.predict_proba(Xs[iva])[:, 1], m.predict_proba(Xs[ite])[:, 1]

    if name == "rf":
        m = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, max_features=0.5,
                                   n_jobs=16, random_state=1)
        m.fit(Xi[itr], y[itr])
        return m.predict_proba(Xi[iva])[:, 1], m.predict_proba(Xi[ite])[:, 1]

    if name == "et":
        m = ExtraTreesClassifier(n_estimators=60, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=16, random_state=1)
        m.fit(Xi[itr], y[itr])
        return m.predict_proba(Xi[iva])[:, 1], m.predict_proba(Xi[ite])[:, 1]

    if name == "lookup":
        # the sim's bolt_lookup stand-in: unsmoothed cell means on the quantisation
        # lattice of the two columns that carry the generator's rule. Small cells return
        # exactly 0.0 or 1.0, which is precisely the behaviour under test.
        key_tr = _lookup_key(Xn[itr])
        df = pd.DataFrame({"k": key_tr, "y": y[itr]})
        g = df.groupby("k")["y"].mean()
        prior = float(y[itr].mean())
        out = []
        for idx in (iva, ite):
            k = _lookup_key(Xn[idx])
            out.append(pd.Series(k).map(g).fillna(prior).to_numpy(np.float64))
        return out[0], out[1]

    raise SystemExit(f"unknown member {name}")


def _lookup_key(X):
    """(daily, social) rounded onto their own lattice -> one integer key."""
    d = np.nan_to_num(X[:, NUM.index("daily_screen_time_hours")], nan=-1.0)
    s = np.nan_to_num(X[:, NUM.index("social_media_hours")], nan=-1.0)
    return (np.round(d * 4).astype(np.int64) * 100_003
            + np.round(s * 4).astype(np.int64))


def train_members(seed, members, frac=1.0):
    """Run the real pipeline's asymmetry on a labelled hold-out.

    Returns O (n_inner, M) pseudo-OOF, T (n_hold, M) pseudo-test, y_inner, y_hold.
    """
    tr, _ = load_raw()
    if frac < 1.0:                      # smoke test only -- never for a reported number
        tr = tr.sample(frac=frac, random_state=0).reset_index(drop=True)
    y_all = tr[TARGET].astype(int).to_numpy()
    Xn, Xi, Xs = build_frames(tr)

    # outer split: the pseudo-test half is never seen by any fit
    inner, hold = next(StratifiedShuffleSplit(1, test_size=0.20, random_state=seed)
                       .split(np.zeros(len(y_all)), y_all))
    inner, hold = np.sort(inner), np.sort(hold)
    y_in, y_ho = y_all[inner], y_all[hold]
    print(f"inner {len(inner):,}  pseudo-test {len(hold):,}  "
          f"base rate {y_in.mean():.5f}/{y_ho.mean():.5f}", flush=True)

    # inner folds: same scheme as the real pipeline (SKF5, shuffle, seed 42)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42)
                 .split(np.zeros(len(inner)), y_in))

    O = np.zeros((len(inner), len(members)))
    T = np.zeros((len(hold), len(members)))
    for j, nm in enumerate(members):
        t0 = time.time()
        tacc = np.zeros(len(hold))
        for itr_l, iva_l in folds:
            pv, pt = fit_member(nm, Xn, Xi, Xs, y_all,
                                inner[itr_l], inner[iva_l], hold)
            O[iva_l, j] = pv
            tacc += pt / len(folds)          # <- the averaging that creates the asymmetry
        T[:, j] = tacc
        print(f"  {nm:10s} oof {roc_auc_score(y_in, O[:, j]):.6f}  "
              f"hold {roc_auc_score(y_ho, T[:, j]):.6f}  ({time.time()-t0:.0f}s)",
              flush=True)
    return O, T, y_in, y_ho, folds


# --------------------------------------------------------------------------- stacking


def stack_one(O, T, y_in, folds, kind, C=1.0):
    """Cross-fit the L2 logit stack exactly as blend_lab.build does. Returns (oof, test)."""
    Z, Zt = transform(O, T, kind)
    Z, Zt = Z.astype("float32"), Zt.astype("float32")
    mo = np.zeros(len(y_in))
    for itr, iva in folds:
        mo[iva] = (LogisticRegression(max_iter=5000, C=C)
                   .fit(Z[itr], y_in[itr]).decision_function(Z[iva]))
    full = LogisticRegression(max_iter=5000, C=C).fit(Z, y_in)
    return mo, full.decision_function(Zt)


def census(O, T, names):
    rows = []
    for j, nm in enumerate(names):
        o, t = O[:, j], T[:, j]
        po = float(((o <= 0) | (o >= 1)).mean())
        pt = float(((t <= 0) | (t >= 1)).mean())
        rows.append(dict(member=nm, oof_pinned=po, test_pinned=pt,
                         asym=po - pt, sd_ratio=float(t.std() / (o.std() + 1e-12))))
    return pd.DataFrame(rows)


def fast_auc(yb, s):
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s), np.float64)
    r[order] = np.arange(1, len(s) + 1)
    npos = int(yb.sum())
    nneg = len(yb) - npos
    return (r[yb].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def paired_boot(y_ho, a, b, reps=200, seed=0):
    """sd of the PAIRED test-AUC difference a-b on the hold-out.

    Both vectors are scored on identical resampled rows, so the shared hold-out noise
    cancels -- the same pairing `cvlb2.py` applies to the public slice. The unpaired sd is
    ~60x larger and is the wrong null for two stacks correlated at 0.999.
    """
    rng = np.random.default_rng(seed)
    n = len(y_ho)
    yb_all = y_ho.astype(bool)
    d = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yb = yb_all[idx]
        if yb.all() or not yb.any():
            continue
        d.append(fast_auc(yb, a[idx]) - fast_auc(yb, b[idx]))
    return float(np.mean(d)), float(np.std(d, ddof=1))


def simplex(k, step):
    n = int(round(1.0 / step))
    out = []
    for cut in itertools.combinations(range(1, n + k - 1), k - 1):
        prev, w = 0, []
        for c in cut:
            w.append(c - prev - 1)
            prev = c
        w.append(n + k - 1 - prev - 1)
        out.append(np.array(w, np.float64) / n)
    return np.array(out)


def evaluate(O, T, y_in, y_ho, folds, members, dose_names, out_json):
    print("\n=== clip census (the mechanism's precondition) ===")
    cen = census(O, T, members)
    print(cen.to_string(index=False, float_format="%.5f"))

    results = []
    for dose in range(len(dose_names) + 1):
        keep = [i for i, nm in enumerate(members)
                if nm in CLEAN or nm in dose_names[:dose]]
        sub = [members[i] for i in keep]
        Os, Ts = O[:, keep], T[:, keep]
        print(f"\n=== dose {dose}: {len(sub)} members "
              f"({', '.join(n for n in sub if n in SAT) or 'none saturating'}) ===",
              flush=True)

        oo, tt = {}, {}
        for k in KINDS:
            mo, mt = stack_one(Os, Ts, y_in, folds, k)
            oo[k], tt[k] = mo, mt
            cv, te = roc_auc_score(y_in, mo), roc_auc_score(y_ho, mt)
            results.append(dict(dose=dose, variant=k, cv=cv, test=te, gap=te - cv))
            print(f"  {k:8s} cv {cv:.6f}  test {te:.6f}  gap {te-cv:+.6f}", flush=True)

        mix = {}
        for nm, ks in (("ens4", KINDS), ("h3", H3)):
            mo = np.mean([rk(oo[k]) for k in ks], 0)
            mt = np.mean([rk(tt[k]) for k in ks], 0)
            mix[nm] = mt
            cv, te = roc_auc_score(y_in, mo), roc_auc_score(y_ho, mt)
            results.append(dict(dose=dose, variant=nm, cv=cv, test=te, gap=te - cv))
            print(f"  {nm:8s} cv {cv:.6f}  test {te:.6f}  gap {te-cv:+.6f}", flush=True)

        # uncertainty on the two contrasts the deadline pick turns on, paired on the
        # hold-out rows so the shared split noise cancels
        for lbl, u, v in (("logit-hybrid", tt["logit"], tt["hybrid"]),
                          ("ens4-h3", mix["ens4"], mix["h3"])):
            m, s = paired_boot(y_ho, u, v)
            results.append(dict(dose=dose, variant=f"boot_{lbl}", cv=np.nan,
                                test=m, gap=s))
            print(f"  boot {lbl:12s} test diff {m:+.6f} +/- {s:.6f}"
                  f"   P(>0) {'high' if m > 2 * s else 'not resolved'}", flush=True)

        # --- the angle: weights searched on the pseudo-OOF, scored on the pseudo-test ---
        R = np.column_stack([rk(oo[k]) for k in KINDS])
        Rt = np.column_stack([rk(tt[k]) for k in KINDS])
        grid = simplex(4, 0.05)
        sc = np.array([roc_auc_score(y_in, R @ w) for w in grid])
        w_oof = grid[int(np.argmax(sc))]
        sct = np.array([roc_auc_score(y_ho, Rt @ w) for w in grid])
        w_true = grid[int(np.argmax(sct))]
        results.append(dict(dose=dose, variant="wsearch", cv=float(sc.max()),
                            test=float(roc_auc_score(y_ho, Rt @ w_oof)),
                            gap=float(roc_auc_score(y_ho, Rt @ w_oof) - sc.max()),
                            w_oof=[round(float(x), 3) for x in w_oof],
                            w_true=[round(float(x), 3) for x in w_true],
                            test_at_w_true=float(sct.max())))
        print(f"  wsearch  w_oof  {np.round(w_oof,3)} -> test "
              f"{roc_auc_score(y_ho, Rt @ w_oof):.6f}")
        print(f"           w_true {np.round(w_true,3)} -> test {sct.max():.6f}  "
              f"(logit weight: OOF picks {w_oof[0]:.2f}, truth wants {w_true[0]:.2f})",
              flush=True)

    df = pd.DataFrame(results)
    print("\n\n=== DISPLACEMENT vs dose (the falsifiable prediction) ===")
    print("displacement = gap(logit) - gap(hybrid); claimed +9.7e-5 on the real data\n")
    piv = df[df.variant.isin(KINDS + ("ens4", "h3"))].pivot(
        index="dose", columns="variant", values="gap")
    piv["logit-hybrid"] = piv["logit"] - piv["hybrid"]
    piv["ens4-h3"] = piv["ens4"] - piv["h3"]
    print(piv.to_string(float_format="%+.6f"))

    print("\n=== does CV rank the transforms the way the truth does? ===")
    for dose in sorted(df.dose.unique()):
        d = df[(df.dose == dose) & df.variant.isin(KINDS + ("ens4", "h3"))]
        by_cv = list(d.sort_values("cv", ascending=False).variant)
        by_te = list(d.sort_values("test", ascending=False).variant)
        rho = np.corrcoef(rankdata(d.cv), rankdata(d.test))[0, 1]
        print(f"  dose {dose}: cv {' > '.join(by_cv)}")
        print(f"          truth {' > '.join(by_te)}   spearman {rho:+.3f}")

    with open(out_json, "w") as f:
        # Stamp the run's scale into the artefact. The 0.05 smoke test once wrote a
        # `results_s7.json` indistinguishable from a real one -- five doses, correct
        # schema, AUCs of 0.947 instead of 0.962 -- which `oofsim_summary.py` would have
        # pooled as a seed. A result file has to say how big the run behind it was.
        json.dump({"meta": {"n_inner": int(len(y_in)), "n_hold": int(len(y_ho)),
                            "n_members": int(O.shape[1])},
                   "census": cen.to_dict("records"),
                   "results": df.to_dict("records")}, f, indent=1)
    print(f"\nwrote {out_json}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="outer hold-out split seed")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0, help="smoke test on a subsample")
    a = ap.parse_args()

    members = list(CLEAN) + list(SAT)
    tag = f"s{a.seed}" + ("" if a.frac >= 1.0 else f"_f{a.frac:g}")
    cO = os.path.join(CACHE, f"O_{tag}.npy")
    cT = os.path.join(CACHE, f"T_{tag}.npy")

    t0 = time.time()
    if a.eval_only and os.path.exists(cO):
        tr, _ = load_raw()
        if a.frac < 1.0:
            tr = tr.sample(frac=a.frac, random_state=0).reset_index(drop=True)
        y_all = tr[TARGET].astype(int).to_numpy()
        inner, hold = next(StratifiedShuffleSplit(1, test_size=0.20, random_state=a.seed)
                           .split(np.zeros(len(y_all)), y_all))
        inner, hold = np.sort(inner), np.sort(hold)
        y_in, y_ho = y_all[inner], y_all[hold]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42)
                     .split(np.zeros(len(inner)), y_in))
        O, T = np.load(cO), np.load(cT)
        print(f"loaded cached members {O.shape} {T.shape}", flush=True)
    else:
        O, T, y_in, y_ho, folds = train_members(a.seed, members, a.frac)
        np.save(cO, O)
        np.save(cT, T)
        print(f"cached -> {cO}", flush=True)

    evaluate(O, T, y_in, y_ho, folds, members, list(SAT),
             os.path.join(CACHE, f"results_{tag}.json"))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
