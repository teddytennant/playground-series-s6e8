"""w80e -- the PER-COLUMN PERMUTATION fold-signature test for data/ext_members17.

Pre-registration: experiments/w80_prereg3.txt, commit 5315bc3, written and committed before
this file existed. Every constant below is READ OUT OF THAT TEXT with a regex, never re-typed:
edit the prereg and the constants move with it; delete a clause it names and the reader
asserts rather than guessing.

WHY THIS REPLACES w80b. prereg2 set Q3's MATCHED bar at min(POS) = 46.913, where POS was "the
20 lexicographically first stems" -- which is four blend bases x five transforms, every one a
blend of 150+ members, and not one single model. w80d measured the cost: 20 of 20 of this
workspace's OWN single models, all KNOWN MATCHED, score below that bar. The MATCHED branch was
unreachable for a pack of single models. w80b's INDETERMINATE stands and is not amended; it is
simply uninformative about the pack.

THE FIX IS WITHIN-COLUMN PERMUTATION. The same column supplies its own null under 199 foreign
stratified partitions, so a heavy-tailed or lumpy column cannot manufacture a detection, and
no threshold is read off the observed values. Both the POWER and the FALSE-POSITIVE rate are
measured on this workspace's own single models -- the population the pack belongs to -- and
both are GATING.

⛔ NO BRANCH OF THIS SCRIPT ADOPTS ANYTHING. Per prereg3, MATCHED-EVIDENCE discharges w38c's
gate 4 BY SUBSTITUTE and makes the pack merely ELIGIBLE for the vetting a build would owe. It
is not a send, not a queue edit, not a HIJACKPRICE change, it is not evidence that w80a's
+10.14e-6 survives a refit, and it says nothing about the es-on-val clause w51 convicted
ext_members16 on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "data", "ext_members17")
PREREG = os.path.join(ROOT, "experiments", "w80_prereg3.txt")
OUT = os.path.join(ROOT, "experiments", "w80e_permsig.json")

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
    pats = {
        "M":          r"M = (\d+)\.  Minimum attainable p",
        "alpha":      r"DETECTED\(c\)  <=>  p\(c\) <= ALPHA,  ALPHA = (\d+\.\d+)",
        "seed_base":  r"random_state = (\d+) \+ m",
        "neg_seed":   r"partition StratifiedKFold\(5, shuffle=True, random_state=(\d+)\)",
        "n_splits":   r"S\(c, F_m\) : m = 1\.\.M \}         F_m = StratifiedKFold\((\d+),",
        "g1_fpr":     r"G1 GATING \(VALIDITY\)\. FPR <= (\d+\.\d+)\.",
        "g2_power":   r"G2 GATING \(POWER\)\. POWER >= (\d+\.\d+) on our own known-matched singles",
        "g3_pagg":    r"p_agg <= (\d+\.\d+)  and  D >= \d+   -> MATCHED-EVIDENCE",
        "g3_dmin":    r"p_agg <= \d+\.\d+  and  D >= (\d+)   -> MATCHED-EVIDENCE",
        "g3_noev":    r"p_agg >= (\d+\.\d+)                 -> NO-EVIDENCE",
        "n_col":      r"D = #DETECTED among the (\d+) pack columns",
    }
    got = {}
    for k, p in pats.items():
        m = re.search(p, txt)
        assert m, f"prereg3 no longer states {k} -- refusing to guess it"
        got[k] = float(m.group(1))
    for k in ("M", "seed_base", "neg_seed", "n_splits", "g3_dmin", "n_col"):
        got[k] = int(got[k])
    m = re.search(r"md5 of experiments/([^\n]+),\n\s*experiments/([^\n,]+), experiments/([^\s]+) byte-identical", txt)
    assert m, "prereg3 no longer names the G5 send-path files -- refusing to guess them"
    got["send_path"] = [os.path.join(ROOT, "experiments", s.strip()) for s in m.groups()]
    return got


def fold_labels(y: np.ndarray, seed: int, k: int) -> np.ndarray:
    lab = np.empty(y.size, dtype=np.int16)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_tr, va) in enumerate(skf.split(np.zeros(y.size), y)):
        lab[va] = i
    return lab


def S_of(z: np.ndarray, lab: np.ndarray, k: int) -> float:
    """S = sum_j n_j * (mean z over fold j)^2 = sum_j (sum z over fold j)^2 / n_j."""
    tot = np.bincount(lab, weights=z, minlength=k)
    cnt = np.bincount(lab, minlength=k).astype(np.float64)
    return float(np.sum(tot * tot / cnt))


def zscore(c: np.ndarray) -> np.ndarray:
    sd = c.std()
    assert sd > 0, "constant column has no signature"
    return (c - c.mean()) / sd


def pvals(cols, obs_lab, perm_labs, k, M) -> tuple:
    """Within-column permutation p for each column. Returns (S_obs, p)."""
    S_obs, P = [], []
    for c in cols:
        z = zscore(c)
        s0 = S_of(z, obs_lab, k)
        ge = sum(1 for lab in perm_labs if S_of(z, lab, k) >= s0)
        S_obs.append(s0)
        P.append((1 + ge) / (1 + M))
    return np.array(S_obs), np.array(P)


def main() -> int:
    global FAILURES
    FAILURES = 0
    reg = read_prereg()

    print("=" * 78)
    print("w80e -- PER-COLUMN PERMUTATION FOLD-SIGNATURE TEST: ext_members17")
    print("=" * 78)
    print("  registered design read from w80_prereg3.txt (commit 5315bc3), not re-typed:")
    for k_, v in reg.items():
        if k_ != "send_path":
            print(f"    {k_:10s} {v}")
    print()
    print("  ⚠ prereg3 was written AFTER its author saw every S value in w80b/w80c/w80d and")
    print("    says so in its own first paragraph. What limits the damage is the design: a")
    print("    permutation p-value rather than an eyeballed cut, and BOTH the power and the")
    print("    false-positive rate measured on known-answer populations and GATING.")
    print()

    md5_before = {os.path.basename(p): md5(p) for p in reg["send_path"]}

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[common.TARGET])
    y = tr[common.TARGET].values
    assert y.size == N_TR, f"train.csv is {y.size} rows, expected {N_TR}"
    k, M = reg["n_splits"], reg["M"]

    t0 = time.time()
    lab_ours = np.empty(y.size, dtype=np.int16)
    for i, (_t, v) in enumerate(common.get_folds(y)):
        lab_ours[v] = i
    lab_neg = fold_labels(y, reg["neg_seed"], k)
    perm = [fold_labels(y, reg["seed_base"] + m, k) for m in range(1, M + 1)]
    print(f"  built OURS + NEG + {M} permutation partitions in {time.time()-t0:.1f}s")

    # G5 sanity: the NEG seed must not collide with the permutation block.
    assert not (reg["seed_base"] + 1 <= reg["neg_seed"] <= reg["seed_base"] + M), \
        "NEG seed lies inside the permutation block -- the negative control is not held out"

    # ---- controls: this workspace's own SINGLE models -------------------------------------
    names = sorted(f for f in os.listdir(os.path.join(ROOT, "oof"))
                   if f.startswith("oof_") and f.endswith(".npy"))
    singles, keep = [], []
    for f in names:
        c = np.load(os.path.join(ROOT, "oof", f))
        if c.size == y.size:
            singles.append(c.astype(np.float64))
            keep.append(f[4:-4])
    print(f"\n  CONTROLS -- {len(singles)} of this workspace's own SINGLE models")

    t0 = time.time()
    S_pos, p_pos = pvals(singles, lab_ours, perm, k, M)     # KNOWN MATCHED  -> power
    S_neg, p_neg = pvals(singles, lab_neg, perm, k, M)      # KNOWN MISMATCHED -> FPR
    print(f"  controls done in {time.time()-t0:.1f}s")

    alpha = reg["alpha"]
    det_pos = p_pos <= alpha
    det_neg = p_neg <= alpha
    power = float(det_pos.mean())
    fpr = float(det_neg.mean())

    print(f"\n  {'model':>28s} {'S_ours':>9s} {'p_ours':>8s}  {'S_fgn':>7s} {'p_fgn':>8s}")
    for i, nm in enumerate(keep):
        flag = "DET" if det_pos[i] else "   "
        flag_n = "DET*" if det_neg[i] else "    "
        print(f"  {nm:>28s} {S_pos[i]:9.3f} {p_pos[i]:8.3f} {flag}  "
              f"{S_neg[i]:7.3f} {p_neg[i]:8.3f} {flag_n}")

    # ---- the pack -------------------------------------------------------------------------
    oof = np.load(os.path.join(PACK, "oof.npy"), mmap_mode="r")
    assert oof.shape == (N_TR, reg["n_col"]), f"pack oof.npy is {oof.shape}"
    members = pd.read_csv(os.path.join(PACK, "members.csv"))
    t0 = time.time()
    packcols = [np.asarray(oof[:, j], dtype=np.float64) for j in range(reg["n_col"])]
    S_pk, p_pk = pvals(packcols, lab_ours, perm, k, M)
    print(f"\n  pack done in {time.time()-t0:.1f}s")
    det_pk = p_pk <= alpha
    D = int(det_pk.sum())

    print("\n" + "-" * 78)
    print(f"  G1 GATING (VALIDITY): FPR {fpr:.3f} on {len(singles)} known-MISMATCHED columns"
          f"   bound <= {reg['g1_fpr']}   (nominal alpha {alpha})")
    if fpr > reg["g1_fpr"]:
        fail(f"instrument VOID -- FPR {fpr:.3f} > {reg['g1_fpr']}, the permutation null is "
             f"not calibrated on real columns")
    else:
        ok("the permutation null is calibrated: foreign columns are not detected")

    print(f"  G2 GATING (POWER):    POWER {power:.3f} on {len(singles)} known-MATCHED singles"
          f"   bound >= {reg['g2_power']}")
    if power < reg["g2_power"]:
        fail(f"instrument VOID -- POWER {power:.3f} < {reg['g2_power']}; fewer than half of "
             f"KNOWN-MATCHED single models are detected, so any count on the pack is "
             f"uninterpretable")
    else:
        ok("the instrument detects a matched partition on SINGLE models, the pack's population")

    void = FAILURES > 0
    p_rate = max(fpr, alpha)
    p_agg = float(binom.sf(D - 1, reg["n_col"], p_rate))
    verdict = "VOID"
    if void:
        print("\n  G3: NOT EVALUATED -- a gating control failed. No verdict on the pack.")
    else:
        print(f"\n  G3 THE READ: D = {D} of {reg['n_col']} pack columns detected at p <= {alpha}")
        print(f"      under 'the pack is foreign', D ~ Binomial({reg['n_col']}, {p_rate:g})")
        print(f"      p_agg = P(D >= {D}) = {p_agg:.3e}")
        if p_agg <= reg["g3_pagg"] and D >= reg["g3_dmin"]:
            verdict = "MATCHED-EVIDENCE"
        elif p_agg >= reg["g3_noev"]:
            verdict = "NO-EVIDENCE"
        else:
            verdict = "INDETERMINATE"
        print(f"      VERDICT: {verdict}")

    # ---- G4 RECORDED ----------------------------------------------------------------------
    cols = [str(members.iloc[j, 0]) for j in range(reg["n_col"])]
    detected = [cols[j] for j in range(reg["n_col"]) if det_pk[j]]
    w80c = json.load(open(os.path.join(ROOT, "experiments", "w80c_foldsig_diag.json")))
    ours_11 = {r["col"] for r in w80c["per_col"] if r["class"] == "OURS"}
    subset = set(detected) <= ours_11
    print(f"\n  G4 RECORDED (decides nothing): detected = {detected}")
    print(f"      w80c's OURS set ({len(ours_11)}): {sorted(ours_11)}")
    print(f"      detected is a subset of w80c's OURS: {subset}")

    # ---- G5 GATING ------------------------------------------------------------------------
    md5_after = {os.path.basename(p): md5(p) for p in reg["send_path"]}
    if md5_after == md5_before:
        ok(f"G5 GATING: send path byte-identical ({len(md5_after)} files)")
    else:
        fail(f"G5 GATING: the send path MOVED: {md5_before} -> {md5_after}")

    licenses = {
        "VOID": "NOTHING. The pack is not built with, not queued, not sent.",
        "INDETERMINATE": "NOTHING. The pack is not built with, not queued, not sent.",
        "NO-EVIDENCE": ("NOTHING. Record in RESEARCH and CLOSE THE THREAD the way w51 closed "
                        "ext_members16. Three instruments is enough. ⛔ NO-EVIDENCE is NOT "
                        "proof of foreignness: the test is one-sided by construction."),
        "MATCHED-EVIDENCE": ("gate 4 discharged BY SUBSTITUTE; the pack becomes ELIGIBLE for "
                             "the full vetting a build would owe. NOT an adoption: not a send, "
                             "not a queue edit, not a HIJACKPRICE change; no evidence that "
                             "w80a's +10.14e-6 survives a refit (it is post-hoc, and the "
                             "REGISTERED w80a file says REFUSE -- malformed (P1)); and no "
                             "answer at all to w51's es-on-val clause."),
    }
    print("-" * 78)
    print(f"  WHAT {verdict} LICENSES (fixed in prereg3, not chosen here):")
    print(f"    {licenses[verdict]}")
    print(f"\n  FAILURES {FAILURES}")

    res = {
        "script": "w80e_permsig.py",
        "prereg": "experiments/w80_prereg3.txt (commit 5315bc3)",
        "REGISTERED": True,
        "supersedes": "w80b_foldsig.py (prereg2 control set was blends only; see w80d)",
        "registered_design": {a: b for a, b in reg.items() if a != "send_path"},
        "G1_fpr": fpr, "G2_power": power, "G1_G2_void": bool(void),
        "D": D, "p_agg": p_agg, "binom_rate": p_rate,
        "verdict": verdict, "licenses": licenses[verdict],
        "detected": detected,
        "G4_subset_of_w80c_OURS": bool(subset),
        "controls": [{"model": keep[i], "S_ours": float(S_pos[i]), "p_ours": float(p_pos[i]),
                      "S_foreign": float(S_neg[i]), "p_foreign": float(p_neg[i])}
                     for i in range(len(keep))],
        "pack": [{"col": cols[j], "S_ours": float(S_pk[j]), "p": float(p_pk[j]),
                  "detected": bool(det_pk[j])} for j in range(reg["n_col"])],
        "send_path_md5": md5_after,
        "FAILURES": FAILURES,
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    sys.exit(main())
