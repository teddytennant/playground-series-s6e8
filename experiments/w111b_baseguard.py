"""w111b — STANDING GUARD: THE PICK'S PROVENANCE CHAIN STILL RESOLVES, AND ITS FILE STILL
REPRODUCES BYTE FOR BYTE.

WHAT WAS NOT COVERED. `w93c_pickverify` recomputes the pick's CV from the STORED OOF vector and
checks the shipped CSV is a well-formed submission for this test set. Its own docstring names
what it cannot do: "the CV is computed on the OOF vector; the file Kaggle scores is a DIFFERENT
ARRAY written by the same build, and nothing on disk has ever asserted" that the two came from
one build. w111 closed that by REBUILDING both picks from their bases -- 8 days after the
originals, byte-identical CSV and OOF, all 75 log lines equal but the four carrying the tag.
This guard keeps that closed for the remaining days at ~3 s instead of ~20 min.

THE CHAIN, ONE LINK PER CONTROL
  C1  the pick's OWN w21a artefact names a base, and that base's OOF and CSV are on disk.
      ⚠ THE BASE MUST COME FROM THE PICK'S ARTEFACT, NEVER FROM A SIBLING'S RUN SCRIPT.
      w111 read it off `w60a_run.sh` -- which builds the ens4 TWIN -- and spent three minutes
      reproducing the wrong file. The two bases differ (`w36_ad199std_h3` vs `w36_ad199std`),
      so this is a live confusion, not a hypothetical one; C1 asserts they still differ.
  C2  the metric EXECUTES: base_auc recomputes from the stored base OOF on the real labels to
      < 1e-9. w92's rule -- only execution is evidence of execution.
  C3  BYTE IDENTITY, re-checked against two independently written files: the shipped
      submissions/<pick> and the w111 reproduction kept in experiments/w111_repro_out/ must
      still have the digests the record holds, and must still equal each other.
  C4  NEGATIVE CONTROLS, three ways, each on a scratch copy: a base name that does not exist
      must fail C1; a base_auc moved by 1e-8 must fail C2; a one-byte edit of the reproduction
      must fail C3.

⚠ HONEST LIMIT. C3 is a PIN. It makes the evidence tamper-evident from 2026-08-28 onward; it
cannot testify about anything that happened before the record was written. The thing that
testifies about the original build is the reproduction itself, and that is a one-off in the
journal, not a property this file can re-derive cheaply.

    .venv/bin/python experiments/w111b_baseguard.py      # 0 = ok, 1 = a control failed
"""
from __future__ import annotations

import hashlib, json, os, shutil, sys, tempfile

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submissions")
REPRO = os.path.join(HERE, "w111_repro_out")
REC = os.path.join(HERE, "w111a_reproduction.json")
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, load_raw  # noqa: E402

AUC_TOL = 1e-9


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c1(pick: str, d: dict, bad: list) -> None:
    art_p = os.path.join(HERE, f"w21a_{pick}.json")
    if not os.path.exists(art_p):
        bad.append(f"C1 {pick}: its own w21a artefact is gone -- the base is unknowable")
        return
    art = json.load(open(art_p))
    if art.get("base") != d["base"]:
        bad.append(f"C1 {pick}: artefact base {art.get('base')!r} != record {d['base']!r}")
        return
    for p in (os.path.join(SUB, f"oof_{d['base']}.npy"), os.path.join(SUB, f"{d['base']}.csv")):
        if not os.path.exists(p):
            bad.append(f"C1 {pick}: base file missing: {os.path.basename(p)}")


def c2(pick: str, d: dict, y, bad: list) -> float | None:
    p = os.path.join(SUB, f"oof_{d['base']}.npy")
    if not os.path.exists(p):
        return None                      # C1 already reported it
    v = np.load(p).astype(np.float64)
    if v.shape[0] != y.shape[0]:
        bad.append(f"C2 {pick}: base OOF is {v.shape[0]} rows, labels are {y.shape[0]}")
        return None
    got = float(roc_auc_score(y, v))
    if abs(got - d["base_auc"]) > AUC_TOL:
        bad.append(f"C2 {pick}: base AUC recomputes {got:.12f} vs recorded "
                   f"{d['base_auc']:.12f}, off by {(got-d['base_auc'])*1e6:+.4f}e-6")
    return got


def c3(pick: str, d: dict, bad: list) -> None:
    checks = [(os.path.join(SUB, f"{pick}.csv"), d["pick_csv_md5"]),
              (os.path.join(SUB, f"oof_{pick}.npy"), d["pick_oof_md5"]),
              (os.path.join(REPRO, f"{d['repro_tag']}.csv"), d["repro_csv_md5"]),
              (os.path.join(REPRO, f"oof_{d['repro_tag']}.npy"), d["repro_oof_md5"])]
    for p, want in checks:
        if not os.path.exists(p):
            bad.append(f"C3 {pick}: {os.path.basename(p)} is gone")
            continue
        got = md5(p)
        if got != want:
            bad.append(f"C3 {pick}: {os.path.basename(p)} digest {got[:12]} != {want[:12]}")
    if d["pick_csv_md5"] != d["repro_csv_md5"] or d["pick_oof_md5"] != d["repro_oof_md5"]:
        bad.append(f"C3 {pick}: the record itself no longer claims byte identity")


def main() -> int:
    bad: list[str] = []
    if not os.path.exists(REC):
        print(f"⛔ {os.path.basename(REC)} is missing -- run w111a_record.py")
        return 1
    rec = json.load(open(REC))["picks"]
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy()
    print(f"w111b — the pick's provenance chain, {len(y):,} labelled rows\n")

    for pick, d in rec.items():
        c1(pick, d, bad)
        got = c2(pick, d, y, bad)
        c3(pick, d, bad)
        print(f"  {pick:24s} base {d['base']:18s} recorded {d['base_auc']:.10f}"
              + (f"  recomputed {got:.10f}" if got is not None else "  recompute SKIPPED")
              + f"  csv {d['pick_csv_md5'][:10]} == repro {d['repro_csv_md5'][:10]}")

    bases = {d["base"] for d in rec.values()}
    if len(bases) < len(rec):
        bad.append("C1 the two picks now record the SAME base -- the h3/ens4 twins are the "
                   "confusion this guard exists for; they must not collapse")
    else:
        print(f"\n✅ C1 the {len(rec)} picks name {len(bases)} DISTINCT bases: {sorted(bases)}")

    # ---- C4: three negative controls, each on a scratch copy ------------------------------
    ctrl = []
    d0 = dict(next(iter(rec.values())))
    p0 = next(iter(rec))

    b = []
    c1(p0, {**d0, "base": "w111_no_such_base"}, b)
    ctrl.append(("a base name the artefact does not carry", bool(b)))

    b = []
    c2(p0, {**d0, "base_auc": d0["base_auc"] + 1e-8}, y, b)
    ctrl.append(("a base_auc moved by 1e-8", bool(b)))

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(REPRO, f"{d0['repro_tag']}.csv")
        dst = os.path.join(td, os.path.basename(src))
        shutil.copyfile(src, dst)
        with open(dst, "r+b") as fh:          # flip one byte of the last row
            fh.seek(-2, os.SEEK_END)
            ch = fh.read(1)
            fh.seek(-2, os.SEEK_END)
            fh.write(b"9" if ch != b"9" else b"8")
        ctrl.append(("a one-byte edit of the reproduction", md5(dst) != d0["repro_csv_md5"]))

    print("\nC4 negative controls (each must FIRE):")
    for name, fired in ctrl:
        print(f"  {'✅' if fired else '⛔'} {name}")
        if not fired:
            bad.append(f"C4 control did not fire: {name}")

    print("\nFAILURES: " + (" | ".join(bad) if bad else "0"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
