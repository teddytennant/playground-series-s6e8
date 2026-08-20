"""w40e — extract ext_members15 from the union94 matrices, by the w40d_prereg rule ONLY.

The rule (fixed in experiments/w40d_prereg.txt before the members were ranked into it):

    maxcorr < 0.99  AND  solo > 0.966319

read off experiments/w40c_union94_vet.csv. Both constants pre-date this run. This script
applies the rule; it does not choose. It asserts the resulting count matches the nine names
the prereg lists, so an accidental threshold drift cannot pass silently.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA  # noqa: E402

SRC = os.path.join(ROOT, "notebooks", "w40", "out",
                   "yadoy666_94-verified-oof-gpu-accelerated-meta-stack")
OUT = os.path.join(DATA, "ext_members15")
N_TR, N_TE = 691369, 296302
MAXCORR, SOLO = 0.99, 0.966319
PREREG = ["naji05", "naji03", "lookup", "tabm_deeper", "pub_rmlp",
          "tabm_imp", "pub_tabnet", "rmlp_lat3", "rmlp_lat"]

v = pd.read_csv(os.path.join(HERE, "w40c_union94_vet.csv"))
sel = v[(v.maxcorr < MAXCORR) & (v.solo > SOLO)].sort_values("solo", ascending=False)
print(f"rule selects {len(sel)} of {len(v)}:")
print(sel[["member", "solo", "maxcorr", "nearest"]].to_string(index=False))
assert sorted(sel.member) == sorted(PREREG), \
    f"rule no longer yields the pre-registered nine:\n  got  {sorted(sel.member)}\n  want {sorted(PREREG)}"

names = json.load(open(os.path.join(SRC, "union94_member_names.json")))
O = np.load(os.path.join(SRC, "union94_oof_matrix.npy"), mmap_mode="r")
T = np.load(os.path.join(SRC, "union94_test_matrix.npy"), mmap_mode="r")
os.makedirs(OUT, exist_ok=True)

for m in sel.member:
    j = names.index(m)
    o = np.asarray(O[:, j], dtype=np.float64)
    t = np.asarray(T[:, j], dtype=np.float64)
    assert o.shape == (N_TR,) and t.shape == (N_TE,), (m, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all(), m
    assert o.std() > 0 and t.std() > 0, m
    # y94_ prefix: these names (`lookup`, `baseline`, `realmlp`) collide with our own.
    np.save(os.path.join(OUT, f"oof_y94_{m}.npy"), o)
    np.save(os.path.join(OUT, f"test_y94_{m}.npy"), t)
    print(f"  wrote y94_{m}  oof[{o.min():.4f},{o.max():.4f}] test[{t.min():.4f},{t.max():.4f}]")

print(f"\n{len(sel)} members -> {OUT}")
