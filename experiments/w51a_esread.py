"""w51a -- the es-on-val provenance read that w42b's clause has demanded since ARM 217.

Reads the FIVE non-`hboyang_mix` members of ext_members16 out of their source authors' own
logs and notebooks, applies the two gates registered in `experiments/w51_prereg.txt` BEFORE
any log was opened, and returns the verdict that decides whether `w50_ad216stdcorr`
(CV 0.9701500880, the highest CV ever built here, +10.1e-6 above the WANTED file) may be
argued as a deadline pick.

This script does not re-derive the verdicts -- it RE-EXTRACTS the evidence lines from the
files on disk so a later run can see them without re-reading four notebooks, and it
re-runs the one measurement that is not a quotation: whether ravi20076's published fold
assignment is our fold assignment.

    .venv/bin/python experiments/w51a_esread.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import get_folds, load_raw, TARGET                       # noqa: E402

NB = os.path.join(ROOT, "notebooks")
COLLATION = os.path.join(NB, "w40/out/ravi20076_playgrounds6e8-datacollation-v1",
                         "OOF_Preds_PublicV1_1.parquet")


def logtext(path: str) -> str:
    """Kaggle .log files are JSON stream records, not plain text."""
    try:
        return "\n".join(r["data"].rstrip("\n") for r in json.load(open(path)))
    except Exception:
        return ""


# ---------------------------------------------------------------- the five, with evidence
# `source` is the file the quoted line comes from; `quote` is a substring that must still be
# present in it, so this script FAILS LOUDLY if a later run's re-pull changes the evidence.
MEMBERS = [
    dict(member="ern711_contextual",
         ref="ern711/contextualized-deep-univariate-spline-transformer",
         source="w40/out/ern711_contextualized-deep-univariate-spline-transformer/"
                "contextualized-deep-univariate-spline-transformer.log",
         quote="Fold 0: early stopping.",
         gate_a="FAIL", gate_b="PASS", level="L1 (spline transformer NN)",
         why="Per-epoch `FINAL` AUC is computed on the held-out fold; `Fold 0: early "
             "stopping.` then `FOLD 0 BEST | E09 | ... FINAL 0.967637` and `OOF after "
             "fold 0: 0.967637`. The published OOF IS the argmax over ~16 noisy per-epoch "
             "readings taken on the very rows it publishes."),
    dict(member="ern711_multilevel",
         ref="ern711/multi-level-deep-univariate-spline-transformer",
         source="w40/out/ern711_multi-level-deep-univariate-spline-transformer/"
                "multi-level-deep-univariate-spline-transformer.log",
         quote="hypernetwork early stopping",
         gate_a="FAIL", gate_b="PASS", level="L1 (spline transformer + hypernetwork)",
         why="The SAME artefact applied TWICE and nested: `Fold 0: base early stopping.` "
             "-> `BASE BEST | E09`, then `Fold 0: hypernetwork early stopping.` -> "
             "`FOLD 0 HYPER BEST | Base E09 0.967685 | Hyper E01 0.967868 | Gain "
             "+0.000183`, and `OOF after fold 0: 0.967868` takes the second argmax. The "
             "author prints the selection gain itself as the headline number."),
    dict(member="ravi200_publicm12",
         ref="mhamza0810/s6e8-single-model-fe-cv-0-96947 (via ravi20076 datacollation, "
             "column PUBLICM12)",
         source="w51_hamza/s6e8-single-model-fe-cv-0-96947.ipynb",
         quote="early_stopping_rounds=200",
         gate_a="FAIL", gate_b="PASS", level="L1 (XGBoost)",
         why="`n_estimators: 20000` with `early_stopping_rounds=200` and "
             "`eval_set=[(x_val, y_val)]`, where `x_val` is the held-out fold whose "
             "`val_preds` are written straight into the OOF. 20,000 candidate stopping "
             "points selected on the scored fold -- the strongest form of the artefact "
             "in this set."),
    dict(member="ravi200_publicm13",
         ref="tamerlanomralinov/s6e8-lookup-transformer-insights-lb-0-97041 (via ravi20076 "
             "datacollation, column PUBLICM13)",
         source="w34/out/tamerlanomralinov_s6e8-lookup-transformer-insights-lb-0-97041/"
                "s6e8-lookup-transformer-insights-lb-0-97041.log",
         quote="valAUC=",
         gate_a="FAIL", gate_b="PASS", level="L1 (lookup transformer NN)",
         why="`fold0 ep 11 valAUC=0.96784 best=0.96784` ... `fold 0 done AUC=0.96784` -- "
             "the reported fold AUC is the RUNNING BEST on the validation fold. ALSO "
             "`folds 11`: this author used an 11-fold partition, so its OOF is not even "
             "produced on our partition and each value comes from a model trained on "
             "~10/11 of the data."),
    dict(member="ravi200_l2stack1r",
         ref="ravi20076/playgrounds6e8-public-l2stack-v1",
         source="w34/ravi20076_playgrounds6e8-public-l2stack-v1/"
                "playgrounds6e8-public-l2stack-v1.ipynb",
         quote="Ridge",
         gate_a="PASS", gate_b="FAIL", level="L2 (Ridge stack over 17 columns)",
         why="GATE A PASSES on its own terms -- a Ridge with `max_iter=100_000`, no "
             "`early_stopping`, no `eval_set`, no callback anywhere, and a log that "
             "prints only per-fold OOF logloss. GATE B FAILS: it is a LEVEL-2 stack, and "
             "the 17 columns it stacks are the same public pool whose es status the four "
             "rows above just failed. A clean stacker over contaminated columns publishes "
             "contaminated predictions."),
]

print("=" * 96)
print("w51a  es-on-val PROVENANCE READ -- the five non-hboyang members of ext_members16")
print("      gates registered in experiments/w51_prereg.txt BEFORE any log was opened")
print("=" * 96)

rows = []
for m in MEMBERS:
    p = os.path.join(NB, m["source"])
    body = logtext(p) if p.endswith(".log") else open(p).read()
    found = m["quote"] in body
    assert found, (f"⛔ evidence line {m['quote']!r} is NO LONGER in {m['source']}. The "
                   f"source changed under this verdict -- re-read it, do not trust the row.")
    ok = m["gate_a"] == "PASS" and m["gate_b"] == "PASS"
    rows.append(dict(member=m["member"], gate_a=m["gate_a"], gate_b=m["gate_b"],
                     level=m["level"], promotes=ok, ref=m["ref"], source=m["source"],
                     evidence_line_present=found, why=m["why"]))
    print(f"\n  {m['member']:22s}  GATE A {m['gate_a']}   GATE B {m['gate_b']}   {m['level']}")
    print(f"    ref      {m['ref']}")
    print(f"    evidence {m['source']}   (quote verified present)")
    for chunk in [m["why"][i:i + 88] for i in range(0, len(m["why"]), 88)]:
        print(f"      {chunk}")

# ----------------------------------------------------------- the one live measurement
tr, _ = load_raw()
y = tr[TARGET].to_numpy()
ours = np.empty(len(y), int)
for k, (_, va) in enumerate(get_folds(y)):
    ours[va] = k
theirs = pd.read_parquet(COLLATION, columns=["fold_nb"]).fold_nb.to_numpy().astype(int)
ct = pd.crosstab(ours, theirs).to_numpy()
identical = bool((ct.max(axis=1) == ct.sum(axis=1)).all())
relabelled = bool(identical and not np.array_equal(ct, np.diag(ct.diagonal())))

print("\n" + "-" * 96)
print("  LIVE MEASUREMENT -- is ravi20076's published fold assignment OUR fold assignment?")
print(f"    our fold sizes    {np.bincount(ours).tolist()}")
print(f"    their fold sizes  {np.bincount(theirs).tolist()}")
print(f"    same partition    {identical}      labels permuted   {relabelled}")
if identical and not relabelled:
    print("    => BYTE-IDENTICAL, SAME LABELS. StratifiedKFold(5, shuffle=True, "
          "random_state=42),\n       the community default, which is also ours and also "
          "hboyang's (result.json).")
    print("    ⚠ THIS IS WHY THE CLAUSE MATTERS HERE MORE THAN ANYWHERE. When an author "
          "early-stops\n      on their held-out fold k, the inflation lands on EXACTLY the "
          "rows our combiner scores\n      as fold k. A different partition would dilute the "
          "artefact; an identical one does not.")

n_fail = sum(1 for r in rows if not r["promotes"])
branch = "R1 DISCHARGED" if n_fail == 0 else "R2 CLAUSE STANDS"
print("\n" + "=" * 96)
print(f"  {n_fail} of {len(rows)} members fail at least one gate  ->  {branch}")
print("  ARM 216 is NOT WANTED-eligible. The seven-file ad216 VETO in w48e_order.py STANDS.")
print("  `w36_ad199stdcorr` (CV 0.9701400060) remains the WANTED deadline pick.")
print("=" * 96)

out = dict(members=rows, n_fail=n_fail, branch=branch,
           fold_partition_identical_to_ours=identical,
           fold_labels_permuted=relabelled,
           ad216_stdcorr_cv=0.9701500880, wanted_cv=0.9701400060,
           ad216_promoted=False)
dst = os.path.join(HERE, "w51a_esread.json")
json.dump(out, open(dst, "w"), indent=2)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "w51a_esread.csv"), index=False)
print(f"\nwrote {dst} and w51a_esread.csv")
