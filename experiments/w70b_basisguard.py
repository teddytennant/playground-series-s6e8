"""w70b — THE CORRECTED-CV BASIS GUARD. Fast; run it with the standing suite.

WHAT IT PROTECTS, IN ONE SENTENCE
---------------------------------
Every `stdcorr` file has TWO defensible CVs — the SHIPPED one (`fast_auc` of the shipped OOF
vector, which is what `w23b_sendqueue`, `w26d_queueprice`, `w53a.FIT_CV_MAX` and every ladder in
the journal read) and the HONEST one (`base_auc + nested_delta`, which removes the gain the
correction's ARM CHOICE gets from being made on the same OOF it is scored on). They differ by
`scheme_optimism`, which w70a measured at 0.000–3.523e-6 across the 20 corrected files on disk —
**larger than the ±2.0e-6 bar w65's R1 decides on and comparable to the ±4e-6 one.**

w69 §9 mixed them: it quoted ARM 208 on the honest basis and 199/202/211 on the shipped one, and
read a ladder off the result. The four-pack order is 199 > 208 > 202 > 211 on one basis and
199 > 211 > 208 > 202 on the other. ⚠ **A NUMBER THAT IS RIGHT ON EITHER BASIS IS WRONG IN A
LADDER THAT MIXES THEM**, and nothing in the workspace could see it, because both numbers are
real, both are stored, and neither is labelled.

THE THREE THINGS THIS ASSERTS
  A. w70a's sweep COVERS every corrected file on disk. Freshness by SET MEMBERSHIP, never by
     mtime — w68 §6: a freshness rule on a derived artefact destroys a later stage's stamp. A
     new *corr build makes this fail until the sweep is re-run, which is the point.
  B. Each covered file's recorded `shipped` still equals the `combos[shipped]["cv"]` in its own
     w21a artefact, and `opt == naive - nested`. Cheap identities; no AUC is recomputed here
     (w70a does that, over the actual OOF vectors).
  C. THE RANKER IS ON ONE BASIS. Every corrected file's `cv` in `w26d_queueprice.csv` must match
     the SAME basis as every other — not each one whichever it happens to be. Switching bases
     wholesale is a legitimate decision; doing it to some files and not others is the defect.

⛔ THIS GUARD DOES NOT PREFER A BASIS. It is deliberately satisfied by either, so a later run can
switch the ranker without fighting it — as long as it switches ALL of them and re-derives the
registered send days, whose contents are a CV ordering.

    .venv/bin/python experiments/w70b_basisguard.py
"""
from __future__ import annotations

import glob, json, os, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
U, TOL = 1e-6, 5e-11
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def main() -> None:
    sweep_path = os.path.join(HERE, "w70a_optimism.json")
    if not os.path.exists(sweep_path):
        fail("w70a_optimism.json is absent — run w70a_optimism.py")
        sys.exit(1)
    sweep = json.load(open(sweep_path))
    rows = {r["tag"]: r for r in sweep["rows"]}
    print(f"  w70a sweep covers {len(rows)} corrected files; it reports "
          f"{sweep['failures']} failures of its own.")
    if sweep["failures"]:
        fail(f"the sweep itself reports {sweep['failures']} failures — its table is not usable")

    # ---- A. COVERAGE ----------------------------------------------------------------------
    # ⚠ tag -> PATH, not a filename template. `w21a_ad187corr.json` carries tag
    # `w21_ad187corr`, so `w21a_{tag}.json` does not resolve for every artefact.
    on_disk, path_of = set(), {}
    for p in sorted(glob.glob(os.path.join(HERE, "w21a_*.json"))):
        if os.path.basename(p).startswith("w21a_ckpt_"):
            continue
        d = json.load(open(p))
        tag = d.get("tag")
        if tag and "nested_delta" in d and os.path.exists(
                os.path.join(os.path.dirname(HERE), "submissions", f"oof_{tag}.npy")):
            on_disk.add(tag)
            path_of[tag] = p
    missing = sorted(on_disk - set(rows))
    if missing:
        fail(f"{len(missing)} corrected file(s) on disk are NOT in the sweep — re-run "
             f"w70a_optimism.py: {missing}")
    stale = sorted(set(rows) - on_disk)
    if stale:
        fail(f"the sweep names {len(stale)} file(s) that no longer have both an artefact and an "
             f"OOF vector: {stale}")
    print(f"  A. coverage: {len(on_disk)} corrected files on disk, "
          f"{len(missing)} uncovered, {len(stale)} stale.")

    # ---- B. THE STORED IDENTITIES ---------------------------------------------------------
    for tag in sorted(set(rows) & on_disk):
        r = rows[tag]
        d = json.load(open(path_of[tag]))
        stored = d["combos"][d["shipped"]]["cv"]
        if abs(stored - r["shipped"]) > TOL:
            fail(f"{tag}: sweep shipped {r['shipped']:.12f} != artefact {stored:.12f}")
        if abs((r["naive"] - r["nested"]) - r["opt"]) > 1e-15:
            fail(f"{tag}: opt {r['opt']:.6e} != naive-nested {r['naive'] - r['nested']:.6e}")
        if abs((d["base_auc"] + r["nested"]) - r["honest"]) > 1e-15:
            fail(f"{tag}: honest {r['honest']:.12f} != base+nested")
    print(f"  B. identities: shipped/opt/honest reproduce from the w21a artefacts.")

    # ---- C. ONE BASIS IN THE RANKER -------------------------------------------------------
    qp = os.path.join(HERE, "w26d_queueprice.csv")
    if not os.path.exists(qp):
        fail("w26d_queueprice.csv is absent — the ranker's basis cannot be checked")
    else:
        q = pd.read_csv(qp)
        q["stem"] = q.file.str.replace(".csv", "", regex=False)
        bases, unknown = {}, []
        for r in q.itertuples():
            s = rows.get(r.stem)
            if s is None or pd.isna(r.cv):
                continue
            if abs(float(r.cv) - s["shipped"]) <= TOL:
                bases.setdefault("shipped", []).append(r.stem)
            elif abs(float(r.cv) - s["honest"]) <= TOL:
                bases.setdefault("honest", []).append(r.stem)
            else:
                unknown.append((r.stem, float(r.cv)))
        # ⚠ VACUITY CHECK — "one basis" over an empty set is trivially true and would let this
        # guard pass while checking nothing. A corrected file must actually be in the ranker.
        n = sum(len(v) for v in bases.values())
        if n == 0:
            fail("no corrected file in w26d_queueprice.csv could be matched to either basis — "
                 "this check is vacuous, not passing")
        if unknown:
            fail(f"{len(unknown)} corrected file(s) carry a cv on NEITHER basis: {unknown[:5]}")
        if len(bases) > 1:
            fail("THE RANKER MIXES BASES — " + "; ".join(
                f"{k}: {len(v)} ({', '.join(sorted(v)[:3])}...)" for k, v in bases.items()))
        elif bases:
            k = next(iter(bases))
            print(f"  C. w26d ranks {n} corrected file(s) and all of them are on the "
                  f"**{k.upper()}** basis. Consistent.")

    print(f"\n  FAILURES {FAILURES}")
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
