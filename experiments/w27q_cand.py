"""w27q -- where a CANDIDATE member would sit in the 188-member pack, for zero fit.

Slot 5's angle is feature engineering measured on CV. The member-level side of that angle is
closed for this pack (RESEARCH: member-level effects translate into the stack at roughly 1/350),
so the question a feature-set variant has to answer is not "is it a better member" but "is it a
DECORRELATED one" -- that is the property w20d and adarsh1077's leave-one-author-out both say a
saturated stack actually pays for.

This reads that straight off stored OOF vectors. No fits, no combiner, minutes of arithmetic on
a box that is already five jobs deep.

Convention is w27l's, deliberately identical so the numbers are comparable to
`experiments/w27l_profile.npz`: rank-transform, then Pearson, then maxcorr against every OTHER
member. LOWER maxcorr = more decorrelated.

⚠ The pack comes from `stack.load_members` with w27h's exact DROP/EXTRA, never a hand-written
directory listing -- w27l's first version enumerated by hand, silently missed `data/oof` (the
74-model public library), and profiled 114 members while calling them 188.

⚠ Candidates are scored against the 188-pack ONLY, and never against each other, so that adding
a second candidate cannot move the first one's number. The pack is the reference frame.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DROP = {"golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "lat_ctraw_r400", "lat_ctfixte_r400"}
EXTRA = ("ext_members", "ext_members2", "ext_members3", "ext_members4", "ext_members6")


def fastrank(v):
    """argsort-of-argsort, w27l's convention. Ties broken arbitrarily rather than averaged;
    on 691k continuous scores that is far below any number reported here."""
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype="float32")
    r[o] = np.arange(len(v), dtype="float32")
    return r / len(v)


def unit_centre(M):
    M -= M.mean(1, keepdims=True)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dir", default="ext_members7",
                    help="directory under data/ holding oof_<name>.npy / test_<name>.npy")
    ap.add_argument("--gate", default="",
                    help="name=auc,... solo AUCs that MUST reproduce; a miss aborts")
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, _ = load_members(y, len(te),
                               extra_dirs=[os.path.join(DATA, d) for d in EXTRA], drop=DROP)
    assert len(names) == 188, f"expected the 188-member pack, got {len(names)}"
    M = np.empty((len(names), len(y)), dtype="float32")
    for i in range(len(names)):
        M[i] = fastrank(O[:, i])
    del O
    print(f"pack: {len(names)} members via stack.load_members (w27h's exact pack)", flush=True)

    cdir = os.path.join(DATA, a.cand_dir)
    cnames = sorted(f[4:-4] for f in os.listdir(cdir) if f.startswith("oof_"))
    if not cnames:
        print(f"no candidates in {cdir}"); return
    C = np.empty((len(cnames), len(y)), dtype="float32")
    solo_c = np.empty(len(cnames))
    for i, n in enumerate(cnames):
        v = np.load(os.path.join(cdir, f"oof_{n}.npy"))
        assert v.shape == (len(y),), f"{n}: {v.shape} != ({len(y)},)"
        assert not np.isnan(v).any(), f"{n}: has NaN"
        solo_c[i] = roc_auc_score(y, v)
        C[i] = fastrank(v)
    print(f"candidates ({len(cnames)}) from data/{a.cand_dir}: {', '.join(cnames)}", flush=True)

    # M6(d) HARNESS GATE -- a mis-indexed export makes every number below meaningless.
    for spec in filter(None, a.gate.split(",")):
        n, want = spec.split("="); want = float(want)
        i = cnames.index(n)
        d = abs(solo_c[i] - want)
        print(f"  GATE {n}: solo {solo_c[i]:.10f} vs registered {want:.10f}  d={d:.2e}", flush=True)
        assert d < 1e-9, f"GATE FAILED for {n} -- export is mis-indexed, all numbers void"

    solo_p = np.array([roc_auc_score(y, M[i]) for i in range(len(names))])
    M = unit_centre(M)
    R = M @ M.T
    np.fill_diagonal(R, -1.0)
    mx_p, who_p = R.max(1), R.argmax(1)
    np.fill_diagonal(R, np.nan)
    del R

    # Candidates vs the PACK only. Never candidate-vs-candidate: see the module docstring.
    Rc = unit_centre(C) @ M.T
    mx_c, who_c = Rc.max(1), Rc.argmax(1)
    med_c = np.median(Rc, axis=1)

    print(f"\n=== pack reference (w27l) ===")
    print(f"  median maxcorr over the 188 : {np.median(mx_p):.5f}   "
          f"10th pct {np.percentile(mx_p, 10):.5f}   min {mx_p.min():.5f}")

    print(f"\n=== candidates ===")
    print(f"{'candidate':>22s} {'solo AUC':>11s} {'maxcorr':>9s} {'closest in pack':>22s} "
          f"{'medcorr':>9s} {'decorr rank':>13s}")
    for i, n in enumerate(cnames):
        # Rank the candidate INTO the pack's maxcorr distribution: how many of the 188 are
        # strictly more decorrelated than it. Reported as k/189, w27l's direction (1 = most).
        rk = int((mx_p < mx_c[i]).sum()) + 1
        print(f"{n:>22s} {solo_c[i]:11.6f} {mx_c[i]:9.5f} {names[who_c[i]]:>22s} "
              f"{med_c[i]:9.5f} {rk:>8d}/{len(names)+1}")

    print(f"\n=== each candidate's five closest pack members ===")
    for i, n in enumerate(cnames):
        o = np.argsort(-Rc[i])[:5]
        print(f"  {n}")
        for j in o:
            print(f"      {names[j]:>24s}  corr {Rc[i, j]:.5f}  solo {solo_p[j]:.6f}")

    np.savez(os.path.join(HERE, "w27q_cand.npz"), cnames=np.array(cnames), solo=solo_c,
             maxcorr=mx_c, medcorr=med_c, closest=np.array([names[j] for j in who_c]),
             pack_names=np.array(names), pack_maxcorr=mx_p)
    print("\n  Read: LOWER maxcorr = more decorrelated. The pack median is the bar; a candidate "
          "above it is a near-copy of something already owned.")


if __name__ == "__main__":
    main()
