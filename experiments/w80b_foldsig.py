"""w80b -- the FOLD-SIGNATURE partition test for data/ext_members17.

Pre-registration: experiments/w80_prereg2.txt, commit 1b79995, written before this file
existed. Every constant below is READ OUT OF THAT TEXT with a regex, never re-typed: if the
prereg is edited the constants move with it, and if a clause it names disappears the reader
asserts rather than guessing.

WHAT THIS IS FOR
----------------
w38c's gate 4 -- recompute per-fold AUC under OUR folds and compare against the author's
printed per-fold numbers -- is unrunnable on this pack: szymonkapiski publishes solo OVERALL
AUC per column and nothing per fold. This is the registered SUBSTITUTE for it.

An OOF column built under partition P carries a per-fold calibration signature: the rows in
P-fold k were predicted by model_k, and model_k has its own bias, so the fold means of the
column differ across P by more than sampling noise. Under a partition that is not P the fold
labels are random with respect to those models and the fold means agree up to noise.

    z      = (c - mean(c)) / sd(c)
    S(c,F) = sum_k n_k * (mean of z over f_k)^2        # ~ chi2(4) when F is random wrt c

⛔ WHAT NO BRANCH OF THIS SCRIPT DOES: adopt anything. Per the prereg, MATCHED discharges
gate 4 BY SUBSTITUTE and makes the pack merely ELIGIBLE for the full vetting a build would
need. It is not a send, not a queue edit, not a HIJACKPRICE change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "data", "ext_members17")
PREREG = os.path.join(ROOT, "experiments", "w80_prereg2.txt")
OUT = os.path.join(ROOT, "experiments", "w80b_foldsig.json")

N_TR, N_COL = 691369, 50

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  ❌ {msg}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def read_prereg() -> dict:
    """Pull the registered design OUT of the prereg text. Never re-type a threshold."""
    txt = open(PREREG).read()
    got = {}
    pats = {
        "n_pos":    r"POS: (\d+) of this workspace's own OOF stems",
        "neg_k":    r"NEG: the SAME \d+ stems evaluated under StratifiedKFold\((\d+), shuffle=True,",
        "neg_seed": r"NEG: the SAME \d+ stems evaluated under StratifiedKFold\(\d+, shuffle=True, random_state=(\d+)\)",
        "q1_auc":   r"the 20 NEG values is >= ([\d.]+)\.",
        "n_col":    r"TEST: the (\d+) pack columns under OUR get_folds\(y\)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        assert m, f"prereg no longer states {k} -- refusing to guess it"
        got[k] = float(m.group(1))
    for k in ("n_pos", "neg_k", "neg_seed", "n_col"):
        got[k] = int(got[k])
    # Q4's send-path list is registered by NAME in the prereg text, so read the names too.
    m = re.search(r"md5 of ([^\n]+),\n\s*([^\n,]+), ([^\s]+) byte-identical", txt)
    assert m, "prereg no longer names the Q4 send-path files -- refusing to guess them"
    got["send_path"] = [os.path.join(ROOT, s.strip()) for s in m.groups()]
    return got


def fold_signature(c: np.ndarray, folds) -> float:
    """S(c,F) = sum_k n_k * (mean of z over f_k)^2, with z the z-score of c. Prereg line 41."""
    sd = c.std()
    assert sd > 0, "constant column has no signature"
    z = (c - c.mean()) / sd
    s = 0.0
    for _tr, va in folds:
        s += va.size * (z[va].mean() ** 2)
    return float(s)


def main() -> int:
    global FAILURES
    FAILURES = 0
    reg = read_prereg()

    print("=" * 78)
    print("w80b -- FOLD-SIGNATURE PARTITION TEST: ext_members17 (szymonkapiski 50-weakest)")
    print("=" * 78)
    print("  registered design read from w80_prereg2.txt, not re-typed:")
    for k, v in reg.items():
        print(f"    {k:10s} {v}")
    print()
    print("  ⚠ THIS PREREG WAS WRITTEN AFTER A FAVOURABLE NUMBER (w80a's +10.14e-6) and says")
    print("    so in its own text. That is why the thresholds live in a commit and this script")
    print("    reads them instead of choosing them.")
    print()

    # ---- Q4 (pre): the send path, before anything is touched ------------------------------
    md5_before = {os.path.basename(p): md5(p) for p in reg["send_path"]}

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[common.TARGET])
    y = tr[common.TARGET].values
    assert y.size == N_TR, f"train.csv is {y.size} rows, expected {N_TR}"

    folds_ours = common.get_folds(y)
    skf_neg = StratifiedKFold(n_splits=reg["neg_k"], shuffle=True, random_state=reg["neg_seed"])
    folds_neg = list(skf_neg.split(np.zeros(y.size), y))

    # sanity: the two partitions must actually differ, or the whole test is vacuous
    lab_o = np.zeros(y.size, dtype=np.int8)
    lab_n = np.zeros(y.size, dtype=np.int8)
    for k, (_t, v) in enumerate(folds_ours):
        lab_o[v] = k
    for k, (_t, v) in enumerate(folds_neg):
        lab_n[v] = k
    agree = float((lab_o == lab_n).mean())
    print(f"  partitions differ: fold-label agreement {agree:.6f} (chance {1/reg['neg_k']:.6f})")
    if abs(agree - 1.0 / reg["neg_k"]) > 0.01:
        fail(f"NEG partition is not independent of OURS (agreement {agree:.6f})")
    else:
        ok("NEG partition is a genuinely foreign, y-stratified 5-fold split")

    # ---- the controls ARE the test --------------------------------------------------------
    stems = sorted(
        f for f in os.listdir(os.path.join(ROOT, "submissions"))
        if f.startswith("oof_") and f.endswith(".npy")
    )[: reg["n_pos"]]
    assert len(stems) == reg["n_pos"], f"only {len(stems)} stems available, need {reg['n_pos']}"
    print(f"\n  CONTROLS -- the {reg['n_pos']} lexicographically first stems in submissions/:")

    pos, neg, per_stem = [], [], []
    for f in stems:
        c = np.load(os.path.join(ROOT, "submissions", f))
        assert c.size == N_TR, f"{f} is {c.size} rows"
        s_pos = fold_signature(c, folds_ours)
        s_neg = fold_signature(c, folds_neg)
        pos.append(s_pos)
        neg.append(s_neg)
        per_stem.append({"stem": f, "S_ours": s_pos, "S_foreign": s_neg})
        print(f"    {f[:44]:44s}  POS {s_pos:12.2f}   NEG {s_neg:8.3f}")

    pos = np.array(pos)
    neg = np.array(neg)

    # ---- the pack -------------------------------------------------------------------------
    oof = np.load(os.path.join(PACK, "oof.npy"), mmap_mode="r")
    assert oof.shape == (N_TR, reg["n_col"]), f"pack oof.npy is {oof.shape}"
    members = pd.read_csv(os.path.join(PACK, "members.csv"))
    print(f"\n  TEST -- the {reg['n_col']} pack columns under OUR get_folds(y):")
    pack, per_col = [], []
    for j in range(reg["n_col"]):
        c = np.asarray(oof[:, j], dtype=np.float64)
        s = fold_signature(c, folds_ours)
        pack.append(s)
        per_col.append({"col": str(members.iloc[j, 0]), "S_ours": s})
    pack = np.array(pack)
    order = np.argsort(pack)
    for j in list(order[:3]) + list(order[-3:]):
        print(f"    {per_col[j]['col']:>8s}  S {pack[j]:12.2f}")

    # ---- Q1 GATING (power) ----------------------------------------------------------------
    lab = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    auc = roc_auc_score(lab, np.r_[pos, neg])
    print("\n" + "-" * 78)
    print(f"  Q1 GATING (POWER): control AUC {auc:.4f}   bound >= {reg['q1_auc']}")
    void = auc < reg["q1_auc"]
    if void:
        fail(f"instrument VOID -- control AUC {auc:.4f} < {reg['q1_auc']}")
    else:
        ok(f"the instrument separates a known-matched from a known-foreign partition")

    # ---- Q2 RECORDED ----------------------------------------------------------------------
    q2 = float(np.median(pos)) > float(np.median(neg))
    print(f"  Q2 RECORDED: median POS {np.median(pos):.3f} > median NEG {np.median(neg):.3f}"
          f"  -> {'CONFIRMED' if q2 else 'FALSIFIED'}")

    # ---- Q3 THE READ ----------------------------------------------------------------------
    t = float(np.median(pack))
    verdict = "VOID"
    if void:
        print("  Q3: NOT EVALUATED -- Q1 failed, the instrument reports no verdict on the pack.")
    elif pos.min() <= neg.max():
        verdict = "VOID"
        fail("control sets overlap (min POS <= max NEG); Q3 is not evaluated")
        print("  Q3: NOT EVALUATED -- the control sets overlap.")
    else:
        print(f"  Q3: t = median S over the {reg['n_col']} pack columns = {t:.3f}")
        print(f"      min(POS) = {pos.min():.3f}   max(NEG) = {neg.max():.3f}")
        if t >= pos.min():
            verdict = "MATCHED"
        elif t <= neg.max():
            verdict = "FOREIGN"
        else:
            verdict = "INDETERMINATE"
        print(f"      VERDICT: {verdict}")

    # ---- Q4 GATING (post) -----------------------------------------------------------------
    md5_after = {os.path.basename(p): md5(p) for p in reg["send_path"]}
    if md5_after == md5_before:
        ok(f"Q4 GATING: send path byte-identical ({len(md5_after)} files)")
    else:
        fail(f"Q4 GATING: the send path MOVED: {md5_before} -> {md5_after}")

    licenses = {
        "VOID": "nothing. The pack is not built with, not queued, not sent.",
        "INDETERMINATE": "nothing. The pack is not built with, not queued, not sent.",
        "FOREIGN": "REFUSE. Close the thread in RESEARCH the way w51 closed ext_members16.",
        "MATCHED": ("gate 4 is discharged BY SUBSTITUTE and the pack becomes ELIGIBLE for the "
                    "full vetting a build would need. NOT an adoption: it rules out the leak, "
                    "it does not establish that +10.14e-6 survives a real pack refit, and it "
                    "says nothing about the es-on-val clause w51 convicted ext_members16 on."),
    }
    print("-" * 78)
    print(f"  WHAT {verdict} LICENSES (fixed in the prereg, not chosen here):")
    print(f"    {licenses[verdict]}")
    print(f"\n  FAILURES {FAILURES}")

    res = {
        "script": "w80b_foldsig.py",
        "prereg": "experiments/w80_prereg2.txt (commit 1b79995)",
        "REGISTERED": True,
        "registered_design": {k: v for k, v in reg.items() if k != "send_path"},
        "partition_agreement": agree,
        "control_auc": float(auc),
        "Q1_void": bool(void),
        "Q2_confirmed": bool(q2),
        "pos": {"median": float(np.median(pos)), "min": float(pos.min()), "max": float(pos.max())},
        "neg": {"median": float(np.median(neg)), "min": float(neg.min()), "max": float(neg.max())},
        "pack": {"median": t, "min": float(pack.min()), "max": float(pack.max())},
        "verdict": verdict,
        "licenses": licenses[verdict],
        "send_path_md5": md5_after,
        "per_stem": per_stem,
        "per_col": per_col,
        "FAILURES": FAILURES,
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sys.exit(main())
