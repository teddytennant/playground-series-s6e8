"""w71b — GUARD: the train<->test duplicate override stays UNAPPLIED.

w70 §8.2 priced the override at +1.62e-6 and §10.5 made applying it the highest-value unspent
item on the account. w71a retracted it: that +1.62e-6 is ONLY the branch where the matched train
label transfers. The other branch is -5.90e-6, break-even needs P(transfer) >= 78.4%, and the
measured copy fraction is 1.8% (95% upper bound 28%). EV is negative across the whole range.

This guard fails if a future run applies it anyway -- the failure mode is silent, because an
overridden file has an UNCHANGED CV (the override touches TEST ids only) and so passes every
existing gate: w23b recomputes the md5 from the file, w48e re-verifies against that fresh md5,
and nothing in the chain ever looks at the two ids. ⚠ THE WHOLE SEND-PATH APPARATUS IS BLIND TO
THIS CHANGE BY CONSTRUCTION. That is exactly why it needs its own guard.

  1. no submission CSV carries a hard 0.0/1.0 at id 735378 or 862871
  2. w70f's two apply paths both refuse (non-zero exit)
  3. no .pre_dupleak backup exists (w70f --inplace writes one before rewriting)
"""
import glob, os, subprocess, sys
import pandas as pd

IDS = [735378, 862871]
PY = sys.executable

def main() -> int:
    print("=" * 92); print("w71b  DUPLICATE-OVERRIDE GUARD"); print("=" * 92)
    fail = []

    hits = []
    for f in sorted(glob.glob("submissions/*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "id" not in d.columns or d.shape[1] < 2:
            continue
        v = d.loc[d["id"].isin(IDS)].iloc[:, 1].tolist()
        if any(x in (0.0, 1.0) for x in v):
            hits.append((f, v))
    print(f"\n  1. submission files with a hard 0/1 at {IDS}: {len(hits)}")
    for f, v in hits[:10]:
        print(f"       ⛔ {f}  {v}")
    if hits:
        fail.append(f"{len(hits)} submission file(s) carry the retracted override")

    for args in (["--inplace", "w36_ad199stdcorr_ens4"], ["--apply", "/tmp/_a.csv", "/tmp/_b.csv"]):
        r = subprocess.run([PY, "experiments/w70f_dupleak.py", *args],
                           capture_output=True, text=True)
        ok = r.returncode != 0
        print(f"  2. w70f {args[0]:<9} refuses: {'yes' if ok else 'NO'}  (rc={r.returncode})")
        if not ok:
            fail.append(f"w70f {args[0]} no longer refuses")

    bak = glob.glob("submissions/*.pre_dupleak")
    print(f"  3. .pre_dupleak backups (proof of an apply): {len(bak)}")
    if bak:
        fail.append(f"{len(bak)} .pre_dupleak backup(s) exist — the override was applied")

    print(f"\n  FAILURES {len(fail)}")
    for m in fail:
        print(f"    ⛔ {m}")
    if not fail:
        print("  ✅ the override is unapplied and both apply paths are shut.")
    return 1 if fail else 0

if __name__ == "__main__":
    sys.exit(main())
