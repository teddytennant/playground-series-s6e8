"""w89c — apply w29's registered member rule to an external pack's members.csv, BEFORE downloading it.

w29g's correction, in full: `maxcorr` is not a screen, because noise decorrelates exactly like
signal does (Spearman(maxcorr, solo AUC) = +0.87 across the ten profiled members). The rule
that replaced it:

    a candidate member is worth a pack refit only if it is BOTH decorrelated AND within
    ~0.005 of the pack median solo AUC

and the SECOND test is the binding one. This file runs only the second test, which is the one
you can run from a 7 KB CSV -- so a pack can be refused without downloading its arrays. It
cannot ADMIT anything: passing here means "worth the download", never "worth importing", and
w51's es-on-val clause still stands between any pack and a build.

    .venv/bin/python experiments/w89c_screen.py data/w89_nhtquyn/members.csv
    .venv/bin/python experiments/w89c_screen.py --selftest
"""
import os, sys
import numpy as np
import pandas as pd

# w29g. Pack median solo AUC at the time the rule was registered, and its tolerance.
PACK_MEDIAN = 0.966
TOL = 0.005
FLOOR = PACK_MEDIAN - TOL

AUC_COLS = ("solo_oof_auc", "oof_auc", "auc", "solo_auc", "cv")


def read_members(path):
    df = pd.read_csv(path)
    col = next((c for c in AUC_COLS if c in df.columns), None)
    if col is None:
        raise SystemExit(f"{path}: no solo-AUC column among {AUC_COLS}; columns are {list(df.columns)}")
    a = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(a).all():
        raise SystemExit(f"{path}: {int((~np.isfinite(a)).sum())} non-finite AUCs -- refusing to screen a "
                         f"pack whose own published numbers are malformed")
    return df, a


def screen(a):
    passing = int((a >= FLOOR).sum())
    best = float(a.max())
    return passing, best, (FLOOR - best) / TOL


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    rc = 0
    for path in args:
        df, a = read_members(path)
        passing, best, shortfall = screen(a)
        print(f"\n{path}")
        print(f"  members {len(a)}   solo AUC {a.min():.6f} .. {best:.6f}   median {a.median():.6f}")
        print(f"  w29 floor = {PACK_MEDIAN} - {TOL} = {FLOOR:.4f}")
        print(f"  members clearing the floor: {passing} of {len(a)}")
        if "family" in df.columns:
            g = df.assign(_a=a).groupby("family")["_a"].agg(["size", "max"]).sort_values("max", ascending=False)
            print("  best by family: " + ", ".join(f"{k} {v['max']:.4f} (n={int(v['size'])})"
                                                   for k, v in g.iterrows()))
        if passing:
            print(f"  ⚠ {passing} member(s) clear the floor — WORTH THE DOWNLOAD, and nothing more. "
                  f"w51's es-on-val clause is untouched by this screen.")
            rc = 1
        else:
            print(f"  ⛔ REFUSED on w29's binding test. The pack's BEST member is {FLOOR - best:.4f} "
                  f"below the floor — {shortfall:.1f}x the whole tolerance. Do not download it.")
    return rc


def selftest():
    """The screen is worthless if it cannot admit. Both directions, and the boundary."""
    fails = 0

    def chk(tag, vals, want_pass):
        nonlocal fails
        p, _, _ = screen(pd.Series(vals, dtype=float))
        ok = (p > 0) == want_pass
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {tag:54s} passing={p} (want {'>0' if want_pass else '0'})")

    chk("C1 all weak (nhtquyn's real range)", [0.853, 0.90, 0.92996], False)
    chk("C2 one member at the pack median", [0.853, 0.966], True)
    chk("C3 exactly ON the floor -- inclusive", [FLOOR], True)
    chk("C4 one tick below the floor", [FLOOR - 1e-9], False)
    chk("C5 a member ABOVE the pack median", [0.9700], True)
    # C6 the screen must not be satisfiable by the pack being large.
    chk("C6 10,000 members, all weak", [0.92] * 10000, False)
    # C7 a malformed pack must RAISE, not screen as refused -- a NaN AUC is not a low AUC.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,solo_oof_auc\na,0.9\nb,\n")
        tmp = f.name
    try:
        read_members(tmp)
        print("  FAIL a NaN AUC screened instead of raising")
        fails += 1
    except SystemExit:
        print("  ok   C7 a NaN AUC raises -- malformed is not the same as weak")
    finally:
        os.unlink(tmp)
    print(f"\nselftest failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
