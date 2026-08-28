"""w111a — write the provenance record the guard reads. Run once; the guard re-checks it.

Every number and every digest in `w111a_reproduction.json` is READ FROM A FILE here, never
typed. That is w60b's rule: a check whose expected values come from the same keystrokes as the
thing checked is not a check. The record holds, for each of the two picks:

  base            the base named by the pick's OWN w21a artefact (NOT by a sibling's run
                  script -- w111 read the base off w60a_run.sh, which builds the ens4 twin,
                  and reproduced the wrong file for three minutes because of it)
  base_auc        as the artefact records it
  pick_md5        digest of the shipped CSV and OOF on disk
  repro_md5       digest of the 2026-08-28 reproduction, kept in experiments/w111_repro_out/

    .venv/bin/python experiments/w111a_record.py     # rewrites the record from disk
"""
from __future__ import annotations

import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submissions")
REPRO = os.path.join(HERE, "w111_repro_out")
OUT = os.path.join(HERE, "w111a_reproduction.json")

# pick tag -> the scratch tag its 2026-08-28 reproduction was written under
PAIRS = {"w36_ad199stdcorr": "w111repro_h3", "w36_ad199stdcorr_ens4": "w111repro"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rec = {}
    for pick, tag in PAIRS.items():
        art = json.load(open(os.path.join(HERE, f"w21a_{pick}.json")))
        rec[pick] = dict(
            base=art["base"],
            base_auc=art["base_auc"],
            pick_csv_md5=md5(os.path.join(SUB, f"{pick}.csv")),
            pick_oof_md5=md5(os.path.join(SUB, f"oof_{pick}.npy")),
            repro_tag=tag,
            repro_csv_md5=md5(os.path.join(REPRO, f"{tag}.csv")),
            repro_oof_md5=md5(os.path.join(REPRO, f"oof_{tag}.npy")),
        )
        # every scalar the shipped artefact records must also be what the reproduction recorded
        rep = json.load(open(os.path.join(HERE, f"w21a_{tag}.json")))
        for k in ("nested_delta", "naive_delta", "scheme_optimism", "naive_arm", "shipped"):
            if art[k] != rep[k]:
                print(f"⛔ {pick}: {k} differs between the shipped artefact and its reproduction")
                return 1
        rec[pick]["artefact_fields_match_repro"] = True
    json.dump(dict(written="2026-08-28", picks=rec,
                   note="digests read from disk; see JOURNAL w111 for the run that made them"),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    for p, d in rec.items():
        print(f"  {p:24s} base {d['base']:20s} csv {d['pick_csv_md5'][:12]}  "
              f"repro {d['repro_csv_md5'][:12]}  identical {d['pick_csv_md5'] == d['repro_csv_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
