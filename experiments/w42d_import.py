"""w42d -- write ext_members16 from the w42b_prereg rule ONLY.

The rule, committed in experiments/w42b_prereg.txt before a single AUC existed:

    maxcorr < 0.99  AND  solo > 0.966319      (both constants lifted from w40d_prereg)

read off experiments/w42c_vet.csv. This script applies the rule; it does not choose. It
asserts the selection is exactly the six members recorded below, so a later rerun against a
changed vet cannot silently import a different pack under the same name.
"""
from __future__ import annotations
import os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA  # noqa: E402
sys.path.insert(0, HERE)

MAXCORR, SOLO = 0.99, 0.966319
OUT = os.path.join(DATA, "ext_members16")
N_TR, N_TE = 691369, 296302
# pinned 2026-08-21, the six the rule returned on first application
PINNED = ["hboyang_mix", "ravi200_l2stack1r", "ravi200_publicm12",
          "ravi200_publicm13", "ern711_multilevel", "ern711_contextual"]

# reuse the vet's own pairing code verbatim -- a second implementation could drift
import importlib.util
spec = importlib.util.spec_from_file_location("w42c", os.path.join(HERE, "w42c_vet.py"))
_src = open(os.path.join(HERE, "w42c_vet.py")).read()
_head = _src.split("# ---------------------------------------------------------------------------- our pack")[0]
ns: dict = {"__file__": os.path.join(HERE, "w42c_vet.py"), "__name__": "w42c_head"}
exec(compile(_head, "w42c_head", "exec"), ns)
collect, pair, norm, OUTDIR = ns["collect"], ns["pair"], ns["norm"], ns["OUTDIR"]

v = pd.read_csv(os.path.join(HERE, "w42c_vet.csv"))
sel = v[(v.maxcorr < MAXCORR) & (v.solo > SOLO)].sort_values("solo", ascending=False)
print(f"rule selects {len(sel)} of {len(v)}:")
print(sel[["member", "ref", "label", "solo", "maxcorr", "nearest"]].to_string(index=False))
assert sorted(sel.member) == sorted(PINNED), \
    f"rule no longer yields the pinned six:\n  got  {sorted(sel.member)}\n  want {sorted(PINNED)}"

os.makedirs(OUT, exist_ok=True)
for _, r in sel.iterrows():
    d = os.path.join(OUTDIR, r.ref.replace("/", "_"))
    o = t = None
    for lab, ol, tl in pair(*collect(d)):
        if lab == str(r.label) or (str(r.label) == "main" and lab == "main"):
            o, t = ol(), tl(); break
    assert o is not None, f"{r.member}: label {r.label!r} not re-found in {r.ref}"
    o = np.asarray(o, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    assert o.shape == (N_TR,) and t.shape == (N_TE,), (r.member, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all() and o.std() > 0 and t.std() > 0, r.member
    # --- SIDE-SCALE VALIDITY, and the repair ------------------------------------------
    # The w42b rule screens on solo AUC and rank correlation. BOTH are rank statistics and
    # both are therefore blind, by construction, to the OOF and test sides of one member
    # arriving on DIFFERENT scales. agent/stack.py's transforms only put the two sides on a
    # common scale "by construction" if they start on one; `logit`/`hybrid`/`rescale` do not
    # re-derive it per side. A member whose test vector is 5x its OOF is not a weak member,
    # it is a BROKEN IMPORT -- the same class of defect as the non-finite and zero-variance
    # asserts already above, and handled here rather than by moving any selection threshold.
    #
    # ENVELOPE: sd_test/sd_oof over all 177 members already on disk spans [0.9923, 1.4145]
    # (gmm_raw is the extreme). [0.95, 1.45] therefore contains the ENTIRE existing pack and
    # cannot reclassify anything we already hold. Measured, not tuned to these candidates.
    ratio = float(t.std() / o.std())
    if not (0.95 <= ratio <= 1.45):
        # Repair, not rejection: map the test side through the OOF empirical quantile
        # function. Monotone per member, so solo AUC, maxcorr and the selection are all
        # unchanged; it is a no-op under `rankraw`, which already ranks each side
        # independently, and it is what makes `logit`/`hybrid` see matched distributions.
        # No guessed constant -- it handles a non-constant mismatch as well as a scale one.
        # np.quantile with 296k quantile points re-partitions per point (O(n*m), it hangs);
        # one sort plus an index is the same empirical quantile function in O(n log n).
        q = (pd.Series(t).rank(method="average").to_numpy() - 0.5) / len(t)
        so = np.sort(o)
        t = so[np.clip((q * len(so)).astype(np.int64), 0, len(so) - 1)]
        print(f"  ! {r.member}: sd_test/sd_oof was {ratio:.4f}, outside the pack envelope "
              f"[0.95, 1.45] -- test side quantile-matched onto the OOF distribution "
              f"(now {t.std() / o.std():.4f})")
    np.save(os.path.join(OUT, f"oof_{r.member}.npy"), o)
    np.save(os.path.join(OUT, f"test_{r.member}.npy"), t)
    print(f"  wrote {r.member:20s} oof[{o.min():.4f},{o.max():.4f}] test[{t.min():.4f},{t.max():.4f}]")

print(f"\n{len(sel)} members -> {OUT}")
