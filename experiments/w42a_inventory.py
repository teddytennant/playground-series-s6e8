"""w42a -- inventory every CANDIDATE kernel-output dir for importable member vectors.

The w40 scan classified 218 refs and left 53 CANDIDATEs; only yadoy666's union94 was
mined (-> ext_members15). This walks all 53 dirs and reports every artefact that could
carry a full-length OOF (691369) or test (296302) vector, with its column names where
it is tabular. It DECIDES NOTHING and writes no member -- the w34b gates come later.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notebooks", "w40", "out")
N_TR, N_TE = 691369, 296302

led = pd.read_csv(os.path.join(HERE, "w40", "ledger2.csv"))
cands = led[led.verdict.astype(str).str.contains("CANDIDATE", na=False)].ref.tolist()
SKIP_AUTHORS = ("yadoy666/94-verified",)   # already mined into ext_members15
rows = []

def probe(path, ref):
    rel = os.path.relpath(path, OUT)
    ext = os.path.splitext(path)[1].lower()
    sz = os.path.getsize(path)
    try:
        if ext == ".npy":
            a = np.load(path, mmap_mode="r")
            shp, cols = a.shape, ""
        elif ext == ".npz":
            z = np.load(path, mmap_mode="r")
            for k in z.files:
                b = z[k]
                if b.ndim >= 1 and b.shape[0] in (N_TR, N_TE):
                    rows.append(dict(ref=ref, file=rel + f"[{k}]", kind="npz",
                                     rows=b.shape[0], shape=str(b.shape), cols="", mb=sz/1e6))
            return
        elif ext == ".parquet":
            d = pd.read_parquet(path)
            shp, cols = d.shape, "|".join(map(str, d.columns[:14]))
        elif ext in (".csv", ".gz"):
            d = pd.read_csv(path, nrows=5)
            n = sum(1 for _ in open(path, "rb", buffering=1 << 20)) - 1 if ext == ".csv" else -1
            shp, cols = (n, d.shape[1]), "|".join(map(str, d.columns[:14]))
        else:
            return
    except Exception as e:
        rows.append(dict(ref=ref, file=rel, kind="ERR", rows=-1, shape=str(e)[:60], cols="", mb=sz/1e6))
        return
    if shp[0] in (N_TR, N_TE):
        rows.append(dict(ref=ref, file=rel, kind=ext.lstrip("."), rows=shp[0],
                         shape=str(shp), cols=cols, mb=sz/1e6))

for ref in cands:
    if any(ref.startswith(s) for s in SKIP_AUTHORS):
        continue
    d = os.path.join(OUT, ref.replace("/", "_"))
    if not os.path.isdir(d):
        continue
    for root, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".log"):
                continue
            probe(os.path.join(root, f), ref)

r = pd.DataFrame(rows)
r.to_csv(os.path.join(HERE, "w42a_inventory.csv"), index=False)
print(f"{len(r)} full-length artefacts across {r.ref.nunique() if len(r) else 0} refs\n")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 62)
for ref, g in r.groupby("ref"):
    tr = (g.rows == N_TR).sum(); te = (g.rows == N_TE).sum()
    print(f"=== {ref}   oof-len:{tr}  test-len:{te}")
    print(g[["file", "kind", "rows", "shape", "cols"]].to_string(index=False))
    print()
