"""End-to-end reproduction check for the deadline pick (`blend159av_h3`).

The pick is defended at the deadline on its cross-fitted CV. That defence is only worth
something if the file on disk can still be *regenerated* from the frozen folds and the
member library -- otherwise the number in the journal describes an artefact nobody can
rebuild. Nothing in this workspace had ever run that check.

The chain under test is the whole one:

    oof/*.npy + data/ext_members{,2}          (159 members)
      -> stack.transform          x4 transforms
      -> blend_lab.build          5-fold cross-fit + full fit, 24 logistic fits
      -> per-transform CSV/OOF
      -> make_h3 rank-average of hybrid/rankraw/rescale
      -> blend159av_h3

Everything is written under the `w14a_` prefix; no existing artefact is touched. The
comparison is against the stored `submissions/blend159av_h3.csv` and its OOF vector.

Member set: `blend159av` is the 159-member library with the three `xgb_latcat` seeds
replaced by their probability mean (`xgb_latcat_avg3`). `orig_bin`/`orig_binm` postdate it
(they are the 160-member builds) so they are dropped as well -- pinned explicitly, because
the journal's own 2026-08-11 lesson is that anything globbing `oof/` must pin its set.

    w14a_repro.py            # full rebuild, ~750s
    w14a_repro.py --compare  # compare an existing rebuild against the stored files
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from common import DATA, SUB, TARGET, load_raw  # noqa: E402
import blend_lab  # noqa: E402

NAME = "w14a_repro159av"
REF = "blend159av"
KINDS = ("logit", "hybrid", "rankraw", "rescale")
KEEP3 = ("hybrid", "rankraw", "rescale")
DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "xgb_latcat", "xgb_latcat_s17", "xgb_latcat_s23",
        "orig_bin", "orig_binm")

# cross-fitted CVs recorded in logs_blend159av.txt on 2026-08-12
EXPECT = {"logit": 0.969965, "hybrid": 0.970029, "rankraw": 0.970034,
          "rescale": 0.970029, "ens4": 0.970045, "h3": 0.970049}


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def rebuild():
    names, y, mats, te = blend_lab.load_all(KINDS, set(DROP))
    if len(names) != 159:
        raise SystemExit(f"member set is {len(names)}, expected 159 -- drop list is wrong")
    blend_lab.build(names, y, mats, KINDS, te, C=1.0, submit_name=NAME)


def h3_of(base):
    """Rank-average hybrid/rankraw/rescale for `base`, returning (oof, test, ids)."""
    oof, test, ids = [], [], None
    for k in KEEP3:
        o = np.load(os.path.join(SUB, f"oof_{base}_{k}.npy"))
        d = pd.read_csv(os.path.join(SUB, f"{base}_{k}.csv"))
        if ids is None:
            ids = d["id"].to_numpy()
        elif not np.array_equal(ids, d["id"].to_numpy()):
            raise SystemExit(f"{base}_{k}.csv id order differs")
        oof.append(rk(o))
        test.append(rk(d[TARGET].to_numpy()))
    return np.mean(oof, 0), np.mean(test, 0), ids


def compare():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    print("\n=== per-transform stacks: rebuilt vs stored ===")
    print(f"{'transform':10s} {'stored CV':>10s} {'rebuilt CV':>11s} {'delta':>10s} "
          f"{'max|dtest|':>11s} {'spearman':>10s}")
    for k in KINDS + ("",):
        if not k:
            continue
        so = np.load(os.path.join(SUB, f"oof_{REF}_{k}.npy"))
        ro = np.load(os.path.join(SUB, f"oof_{NAME}_{k}.npy"))
        sd = pd.read_csv(os.path.join(SUB, f"{REF}_{k}.csv"))[TARGET].to_numpy()
        rd = pd.read_csv(os.path.join(SUB, f"{NAME}_{k}.csv"))[TARGET].to_numpy()
        a_s, a_r = roc_auc_score(y, so), roc_auc_score(y, ro)
        sp = spearmanr(sd, rd).statistic
        print(f"{k:10s} {a_s:10.6f} {a_r:11.6f} {a_r - a_s:+10.6f} "
              f"{np.abs(sd - rd).max():11.3e} {sp:10.6f}")

    print("\n=== h3 (the deadline pick) ===")
    so_h3 = np.load(os.path.join(SUB, f"oof_{REF}_h3.npy"))
    sd_h3 = pd.read_csv(os.path.join(SUB, f"{REF}_h3.csv"))
    ro_h3, rt_h3, ids = h3_of(NAME)

    if not np.array_equal(sd_h3["id"].to_numpy(), ids):
        raise SystemExit("id order differs between stored and rebuilt h3")
    st = sd_h3[TARGET].to_numpy()

    a_s, a_r = roc_auc_score(y, so_h3), roc_auc_score(y, ro_h3)
    sp = spearmanr(st, rt_h3).statistic
    print(f"stored   blend159av_h3 cross-fitted CV {a_s:.6f}  (journal: {EXPECT['h3']:.6f})")
    print(f"rebuilt  {NAME}_h3     cross-fitted CV {a_r:.6f}")
    print(f"delta {a_r - a_s:+.7f}")
    print(f"OOF   : max|d| {np.abs(so_h3 - ro_h3).max():.3e}  "
          f"identical={np.array_equal(so_h3, ro_h3)}")
    print(f"test  : max|d| {np.abs(st - rt_h3).max():.3e}  "
          f"identical={np.array_equal(st, rt_h3)}  spearman {sp:.8f}")
    disc = int((rankdata(st) != rankdata(rt_h3)).sum())
    print(f"rows whose RANK moved (AUC only sees this): {disc:,} of {len(st):,} "
          f"({100 * disc / len(st):.4f}%)")

    # save the rebuilt h3 so a later run can diff against it without a second 750s pass
    pd.DataFrame({"id": ids, TARGET: rt_h3}).to_csv(
        os.path.join(SUB, f"{NAME}_h3.csv"), index=False)
    np.save(os.path.join(SUB, f"oof_{NAME}_h3.npy"), ro_h3)

    print("\n=== journal check: rebuilt CVs vs logs_blend159av.txt ===")
    ens_o = np.mean([rk(np.load(os.path.join(SUB, f"oof_{NAME}_{k}.npy")))
                     for k in KINDS], 0)
    got = {k: roc_auc_score(y, np.load(os.path.join(SUB, f"oof_{NAME}_{k}.npy")))
           for k in KINDS}
    got["ens4"] = roc_auc_score(y, ens_o)
    got["h3"] = a_r
    bad = 0
    for k, v in EXPECT.items():
        d = got[k] - v
        flag = "ok" if abs(d) < 5e-7 else "MISMATCH"
        bad += flag == "MISMATCH"
        print(f"  {k:8s} journal {v:.6f}  rebuilt {got[k]:.6f}  {d:+.7f}  {flag}")
    print("\nVERDICT: " + ("pipeline reproduces to 6dp" if not bad else
                           f"{bad} transform(s) do NOT reproduce"))
    return bad


if __name__ == "__main__":
    if "--compare" not in sys.argv:
        rebuild()
    sys.exit(1 if compare() else 0)
