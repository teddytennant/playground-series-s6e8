"""w70a — THE SCHEME-SELECTION OPTIMISM IN EVERY `stdcorr` FILE, AND WHAT IT DOES TO THE RANKING.

WHY THIS FILE EXISTS
--------------------
w69 §9 recorded ARM 208's corrected CV as `base 0.9701342350 + nested_delta 3.2315e-6
≈ 0.9701374665` and read a ladder off it. That number is NOT the object the rest of the
workspace ranks on. `w23b_sendqueue.py` ranks on `fast_auc(y, submissions/oof_<name>.npy)` —
the SHIPPED file's own OOF AUC — and for `w69_ad208stdcorr` that is **0.9701391338**, 1.67e-6
higher. Every other rung of w69 §9's ladder (199/202/211) was quoted on the SHIPPED basis, so
the ladder compared one optimism-CORRECTED number against three UNCORRECTED ones.

⚠ THE TWO BASES ARE BOTH LEGITIMATE AND THEY DISAGREE ON THE ORDER. They differ by exactly
`scheme_optimism` = naive_delta - nested_delta: the amount the correction's ARM CHOICE gains
from being made on the same OOF it is then scored on. `w21a` already computes and stores it.
Nobody has ever read it across files.

WHAT THIS IS AND IS NOT
-----------------------
DESCRIPTIVE. Every input is an artefact already on disk; nothing is refitted, no seed is drawn,
no bar is set and NO DECISION IS TAKEN HERE. It answers one question — how large is the
scheme-selection term in the number the sender ranks on, per file — and hands the ranking-basis
question to a later run with the numbers attached. Changing the ranking basis reorders the send
queue and is a decision; this run computed the number and so should not also take it.

    .venv/bin/python experiments/w70a_optimism.py
"""
from __future__ import annotations

import glob, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw                      # noqa: E402
from w16b_cellweight import fast_auc                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
U = 1e-6
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "w21a_*.json"))):
        if os.path.basename(p).startswith("w21a_ckpt_"):
            continue
        d = json.load(open(p))
        tag = d.get("tag")
        if tag is None or "nested_delta" not in d:
            continue
        oof = os.path.join(SUB, f"oof_{tag}.npy")
        if not os.path.exists(oof):
            print(f"  (skip {tag}: no shipped OOF vector)")
            continue
        v = np.load(oof).astype(np.float64)
        if v.shape != y.shape:
            fail(f"{tag}: OOF vector is {v.shape}, expected {y.shape}")
            continue
        shipped = fast_auc(y, v)
        nd, ndv, opt = d["naive_delta"], d["nested_delta"], d["scheme_optimism"]
        picks = d.get("nested_picks", [])

        # GATE 1 — the stored optimism must BE the two deltas' difference, not an independent
        # field that could have drifted from them.
        if abs((nd - ndv) - opt) > 1e-15:
            fail(f"{tag}: scheme_optimism {opt:.6e} != naive-nested {nd - ndv:.6e}")

        # GATE 2 — optimism is zero IF AND ONLY IF every nested fold picked the naive arm. An
        # exact 0.0 in this column is otherwise indistinguishable from a nested loop that never
        # ran, which is precisely the misreading this gate exists to prevent.
        unanimous = bool(picks) and all(k == d["naive_arm"] for k in picks)
        if (opt == 0.0) != unanimous:
            fail(f"{tag}: optimism=={opt!r} but nested_picks={picks} vs naive {d['naive_arm']!r}")

        # GATE 3 — the shipped combo's stored cv must equal the shipped OOF vector's AUC. This
        # is what makes `shipped` and `base_auc + naive/nested_delta` comparable at all.
        stored = d["combos"][d["shipped"]]["cv"]
        if abs(stored - shipped) > 5e-11:
            fail(f"{tag}: stored combo cv {stored:.12f} != shipped OOF AUC {shipped:.12f}")

        rows.append(dict(tag=tag, base=d["base_auc"], shipped=shipped,
                         honest=d["base_auc"] + ndv, naive=nd, nested=ndv, opt=opt,
                         picks="".join(k[0] for k in picks), naive_arm=d["naive_arm"]))

    rows.sort(key=lambda r: -r["shipped"])
    print("=" * 108)
    print("w70a  SCHEME-SELECTION OPTIMISM PER `stdcorr` FILE")
    print("=" * 108)
    print(f"\n  {len(rows)} corrected files with both a shipped OOF vector and a w21a artefact.")
    print(f"\n  {'file':<28} {'shipped':>14} {'honest':>14} {'naive':>8} {'nested':>8} "
          f"{'OPTIM':>8}  {'picks':<6} naive")
    print("  " + "-" * 104)
    for r in rows:
        print(f"  {r['tag']:<28} {r['shipped']:.10f} {r['honest']:.10f} "
              f"{r['naive'] / U:8.3f} {r['nested'] / U:8.3f} {r['opt'] / U:8.3f}  "
              f"{r['picks']:<6} {r['naive_arm']}")

    o = np.array([r["opt"] for r in rows]) / U
    nz = o[o > 0]
    print(f"\n  optimism over {len(o)} files: min {o.min():.3f}  max {o.max():.3f}  "
          f"mean {o.mean():.3f}  zero on {int((o == 0).sum())}, non-zero on {len(nz)}"
          + (f" (mean of the non-zero {nz.mean():.3f}e-6)" if len(nz) else ""))

    # ---- WHAT IT DOES TO THE ORDER -------------------------------------------------------
    by_s = [r["tag"] for r in sorted(rows, key=lambda r: -r["shipped"])]
    by_h = [r["tag"] for r in sorted(rows, key=lambda r: -r["honest"])]
    moved = [(t, by_s.index(t) + 1, by_h.index(t) + 1) for t in by_s if by_s.index(t) != by_h.index(t)]
    print(f"\n  RANK CHANGES, shipped basis -> honest basis: {len(moved)} of {len(rows)} files move.")
    for t, a, b in sorted(moved, key=lambda x: abs(x[2] - x[1]), reverse=True):
        print(f"    {t:<28} #{a:<2} -> #{b:<2}  ({b - a:+d})")

    # ---- THE FOUR-ARM PACK LADDER, BOTH BASES --------------------------------------------
    LAD = [("199", "w36_ad199stdcorr"), ("208", "w69_ad208stdcorr"),
           ("202", "w38_ad202stdcorr"), ("211", "w40_ad211stdcorr")]
    idx = {r["tag"]: r for r in rows}
    if all(t in idx for _, t in LAD):
        print("\n  THE 2x2 MEMBER FACTORIAL'S FOUR PACKS, on both bases (w69 section 9's ladder):")
        print(f"    {'arm':>4} {'shipped':>14} {'rk':>3}   {'honest':>14} {'rk':>3}   {'optim':>8}")
        s = sorted(LAD, key=lambda x: -idx[x[1]]["shipped"])
        h = sorted(LAD, key=lambda x: -idx[x[1]]["honest"])
        for a, t in LAD:
            r = idx[t]
            print(f"    {a:>4} {r['shipped']:.10f} {[x[0] for x in s].index(a) + 1:>3}   "
                  f"{r['honest']:.10f} {[x[0] for x in h].index(a) + 1:>3}   {r['opt'] / U:8.3f}")
        print(f"    shipped order {' > '.join(x[0] for x in s)}"
              f"   |   honest order {' > '.join(x[0] for x in h)}")
    else:
        fail("the four factorial packs are not all present")

    out = dict(rows=rows, n=len(rows), opt_max=float(o.max()), opt_mean=float(o.mean()),
               n_zero=int((o == 0).sum()), rank_changes=len(moved), failures=FAILURES)
    with open(os.path.join(HERE, "w70a_optimism.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote w70a_optimism.json   FAILURES {FAILURES}")


if __name__ == "__main__":
    main()
