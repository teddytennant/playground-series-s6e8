"""w93c — RECOMPUTE the CV the selection rests on, from the stored OOF vector, and check the
artefact that would actually be scored.

WHY THIS EXISTS. Every guard around the pick reads the CV out of PROSE. `w84a_pickargmax`
parses it from the submission description, `w57a_tierprice2.json` stores it, `SELECT_THESE.md`
prints it. All three are copies of a number that was computed once, on 2026-08-20, and has
been carried forward ever since. w92's lesson was "only execution is evidence of execution";
this executes the metric.

It also checks the OTHER half of the pipeline, which no guard covers at all: the test-side
CSV. The CV is computed on the OOF vector; the file Kaggle scores is a different array
written by the same build, and nothing on disk has ever asserted that the shipped file is a
well-formed submission for THIS test set.

  G1  the claimed CV recomputes from submissions/oof_<stem>.npy on the frozen folds' labels,
      to within the 3.68e-6 BLAS reproducibility floor blend_lab documents -- and in fact to
      far tighter, since roc_auc_score on a stored vector involves no solver at all.
  G2  the CV ORDERING that selection depends on survives recomputation. This is the claim
      that actually matters: slot 1 must still be the strict argmax over every rival with a
      stored OOF, on recomputed numbers rather than quoted ones.
  G3  the shipped CSV is a valid submission for this test set: exact row count, `id` equal to
      test.csv's ids AS A SET, the sample_submission column names, finite values in (0,1),
      no duplicate ids.
  G4  CONTROL-: a deliberately corrupted OOF vector (10% of rows shuffled) must FAIL G1, so
      G1 is demonstrated to be sensitive rather than merely asserted.

    .venv/bin/python experiments/w93c_pickverify.py       # 0 = ok, 1 = a check failed
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
OUT = os.path.join(HERE, "w93c_pickverify.json")

# The two SELECT_THESE.md names and their CVs as carried in prose. G1 recomputes these.
CLAIMED = {
    "w36_ad199stdcorr": 0.9701400060,      # slot 1, the CV argmax
    "w23_ad187stdcorr": 0.9701150809,      # slot 2, the w64-settled cross-base hedge
}
# The rivals slot 1 must out-CV for the selection to be right. All have a stored OOF.
RIVALS = ["w38_ad202stdcorr", "w40_ad211stdcorr", "w36_ad199stdcorr_ens4",
          "w36_ad199std_h3", "w38_ad202stdcorr_ens4", "w40_ad211std_h3"]
TOL = 3.68e-6                              # blend_lab's documented BLAS reproducibility floor


def cv_of(stem: str, y: np.ndarray):
    p = os.path.join(SUB, f"oof_{stem}.npy")
    if not os.path.exists(p):
        return None, None
    v = np.load(p)
    if v.shape[0] != y.shape[0]:
        return None, f"OOF length {v.shape[0]} != {y.shape[0]} train rows"
    return float(roc_auc_score(y, v)), None


def main() -> int:
    bad: list[str] = []
    tr, _te_unused = load_raw()
    y = tr[TARGET].to_numpy()
    print(f"train {len(y)} rows, base rate {y.mean():.10f}")

    # ---- G1 ------------------------------------------------------------------------------
    recomputed = {}
    for stem, claim in CLAIMED.items():
        got, err = cv_of(stem, y)
        if err or got is None:
            bad.append(f"G1 {stem}: {err or 'no stored OOF vector'}")
            continue
        recomputed[stem] = got
        d = (got - claim) * 1e6
        ok = abs(got - claim) <= TOL
        print(f"{'✅' if ok else '⛔'} G1 {stem:26s} claimed {claim:.10f}  "
              f"recomputed {got:.10f}  delta {d:+.4f}e-6")
        if not ok:
            bad.append(f"G1 {stem} recomputes {d:+.3f}e-6 off its claim, past the {TOL*1e6:.2f}e-6 floor")

    # ---- G2: the ORDERING, on recomputed numbers ------------------------------------------
    tab = []
    for stem in list(CLAIMED) + RIVALS:
        got, err = cv_of(stem, y)
        if got is not None:
            tab.append((stem, got))
    tab.sort(key=lambda t: -t[1])
    print(f"\nrecomputed CV over {len(tab)} files with a stored OOF vector")
    for i, (stem, c) in enumerate(tab, 1):
        mark = "  <== SELECT_THESE slot 1" if stem == "w36_ad199stdcorr" else ""
        print(f"  {i:2d}. {stem:28s} {c:.10f}{mark}")
    if not tab:
        bad.append("G2 no OOF vectors resolved at all")
    elif tab[0][0] != "w36_ad199stdcorr":
        bad.append(f"G2 the recomputed argmax is {tab[0][0]}, NOT the slot-1 pick "
                   f"w36_ad199stdcorr — SELECT_THESE.md must be revisited BY HAND")
    else:
        margin = (tab[0][1] - tab[1][1]) * 1e6
        print(f"✅ G2 slot 1 is the recomputed argmax of {len(tab)}, margin {margin:+.3f}e-6 "
              f"over {tab[1][0]}")

    # ---- G3: the artefact that would actually be scored ------------------------------------
    te = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])
    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"), nrows=1)
    for stem in CLAIMED:
        f = os.path.join(SUB, f"{stem}.csv")
        if not os.path.exists(f):
            bad.append(f"G3 {stem}.csv is not on disk")
            continue
        s = pd.read_csv(f)
        errs = []
        if list(s.columns) != list(ss.columns):
            errs.append(f"columns {list(s.columns)} != {list(ss.columns)}")
        if len(s) != len(te):
            errs.append(f"{len(s)} rows != {len(te)}")
        if s["id"].duplicated().any():
            errs.append(f"{int(s['id'].duplicated().sum())} duplicate ids")
        if set(s["id"]) != set(te["id"]):
            errs.append("id set differs from test.csv")
        v = pd.to_numeric(s[ss.columns[1]], errors="coerce").to_numpy()
        if not np.isfinite(v).all():
            errs.append(f"{int((~np.isfinite(v)).sum())} non-finite values")
        elif v.min() < 0 or v.max() > 1:
            # The metric is AUC, so 0 and 1 are LEGAL endpoints -- an earlier draft of this
            # check demanded the OPEN interval and went red on the one row at exactly 1.0.
            errs.append(f"values outside [0,1]: {v.min():.6g}..{v.max():.6g}")
        # What can actually cost AUC is TIES: a file rounded to k decimals throws away
        # ranking information the CV was computed with. Nothing else on disk checks this.
        nuniq = len(np.unique(v))
        if nuniq < 0.99 * len(v):
            errs.append(f"only {nuniq} distinct values over {len(v)} rows "
                        f"({nuniq/len(v):.3%}) -- ties are discarding ranking information")
        if errs:
            bad.append(f"G3 {stem}.csv: " + "; ".join(errs))
        else:
            print(f"✅ G3 {stem}.csv  {len(s)} rows, ids match test.csv, "
                  f"{ss.columns[1]} in [{v.min():.3g}, {v.max():.6g}], mean {v.mean():.10f}, "
                  f"{nuniq} distinct ({nuniq/len(v):.2%}, no rounding loss)")

    # ---- G4: CONTROL-, a corrupted OOF must fail G1 ----------------------------------------
    stem = "w36_ad199stdcorr"
    v = np.load(os.path.join(SUB, f"oof_{stem}.npy")).copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(v), size=len(v) // 10, replace=False)
    v[idx] = v[rng.permutation(idx)]
    ctl = float(roc_auc_score(y, v))
    if abs(ctl - CLAIMED[stem]) <= TOL:
        bad.append(f"G4 CONTROL- a 10%-shuffled OOF still passes G1 ({ctl:.10f}); "
                   f"G1's tolerance is too loose to detect anything")
    else:
        print(f"✅ G4 CONTROL- 10%-shuffled OOF scores {ctl:.10f}, "
              f"{(ctl - CLAIMED[stem])*1e6:+.1f}e-6 — G1 would have caught it")

    json.dump(dict(recomputed=recomputed, claimed=CLAIMED,
                   ranked=[dict(stem=s, cv=c) for s, c in tab],
                   tol=TOL, control_corrupt_cv=ctl, failures=bad),
              open(OUT, "w"), indent=1)
    print("\nFAILURES: " + (" | ".join(bad) if bad else "0"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
