"""w90a — price an external OOF VECTOR on our own labels and our own frozen folds.

`w89c_screen.py` refuses a pack from its published `members.csv` without downloading the
arrays. That only works for a pack that publishes one. The two artefacts w90's spelling fix
surfaced do not:

    stephentarter/ps-s06e08-nn-tabular-predictions   one NN member, oof + test, no members.csv
    atakanaldemir/s6e8-v13-diversity-anchor-...      one 244-member STACK, oof + submission

So the solo AUC has to be computed here, on OUR labels, rather than read off theirs. That is
strictly better evidence than a published number: it cannot be mis-stated, and where the
author DID publish one (the anchor's audit json says 0.9701665486177532) reproducing it is an
independent check that the artefact is aligned to `train.csv` in original row order as claimed.

⛔ What this file can conclude. It runs w29g's binding test -- the same FLOOR, imported from
`w89c_screen` so there is one owner of the constant -- and w29g's test can only REFUSE. Passing
it means "worth the download", never "worth importing": w51's es-on-val clause stands between
any external pack and a build, and this file cannot see it. The per-fold spread it prints is a
read of the field, not a step toward one.

⚠ THE ARRAYS ARE NOT KEPT. Both are refused or unimportable and together they are 39 MB of
nothing; the NUMBERS are the artefact and they are in `logs/w90/w90a_extread.log`. To re-run:

    cd experiments/w90_probe
    kaggle datasets download stephentarter/ps-s06e08-nn-tabular-predictions
    kaggle datasets download atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124 \
        -f v13_diversity_anchor_oof.csv

Without them the file reports "not on disk; skipped" per source and still exits 0.

    .venv/bin/python experiments/w90a_extread.py
"""
import os, sys, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import get_folds, DATA          # noqa: E402
from w89c_screen import FLOOR, PACK_MEDIAN, TOL  # noqa: E402

PROBE = os.path.join(HERE, "w90_probe")
PICK_OOF = os.path.join(ROOT, "submissions", "oof_w36_ad199stdcorr.npy")

# (label, how to get a DataFrame with an id column and one score column, claimed auc or None)
SOURCES = [
    ("stephentarter/ps-s06e08-nn-tabular-predictions",
     ("zip", os.path.join(PROBE, "ps-s06e08-nn-tabular-predictions.zip"), "predictions/nn_oof_probs.csv"),
     None, "prob_1"),
    ("atakanaldemir/s6e8-v13-diversity-anchor-lb-0-97124",
     ("csv", os.path.join(PROBE, "v13_diversity_anchor_oof.csv"), None),
     0.9701665486177532, None),
]


def load(spec):
    kind, path, inner = spec
    if kind == "zip":
        with zipfile.ZipFile(path) as z:      # no `unzip` on this machine, by long standing
            with z.open(inner) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(path)


def align(df, ids, want=None):
    """Return the score column in OUR train.csv row order, or raise saying why it cannot be."""
    idcol = next((c for c in df.columns if c.lower() == "id"), None)
    if idcol is None:
        raise SystemExit(f"no id column in {list(df.columns)}")
    scorecols = [c for c in df.columns if c != idcol]
    if want is not None:
        if want not in scorecols:
            raise SystemExit(f"asked for '{want}', artefact has {scorecols}")
        scorecols = [want]
    if len(scorecols) != 1:
        raise SystemExit(f"expected exactly one score column, got {scorecols}")
    if len(df) != len(ids):
        raise SystemExit(f"{len(df)} rows vs our {len(ids)}")
    same_order = bool((df[idcol].values == ids).all())
    s = df.set_index(idcol)[scorecols[0]].reindex(ids)
    if s.isna().any():
        raise SystemExit(f"{int(s.isna().sum())} of our ids are missing from the artefact")
    return s.values.astype(np.float64), same_order, scorecols[0]


def main():
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=["id", "addicted_label"])
    ids, yy = y["id"].values, y["addicted_label"].values
    folds = np.empty(len(yy), dtype=np.int64)
    for k, (_, va) in enumerate(get_folds(yy)):
        folds[va] = k
    print(f"our frozen partition: n={len(yy)}  fold sizes={np.bincount(folds).tolist()}")

    pick = np.load(PICK_OOF).ravel() if os.path.exists(PICK_OOF) else None
    if pick is not None:
        print(f"our PICK w36_ad199stdcorr OOF AUC = {roc_auc_score(yy, pick):.10f}\n")

    rc = 0
    for label, spec, claimed, want in SOURCES:
        print(f"\n{label}")
        try:
            df = load(spec)
        except (FileNotFoundError, KeyError) as e:
            print(f"  ⚠ not on disk ({e}); skipped")
            continue
        # A published `target` column is a far stronger alignment check than matching ids:
        # it says the artefact's rows carry the labels WE have, in the order we read them.
        tcol = next((c for c in df.columns if c.lower() in ("target", "label", "y", "addicted_label")), None)
        if tcol is not None:
            t, _, _ = align(df[[next(c for c in df.columns if c.lower() == "id"), tcol]], ids)
            agree = float((t.astype(np.int64) == yy).mean())
            print(f"  artefact ships a '{tcol}' column: agreement with OUR labels = {agree:.10f}"
                  f"  {'✅' if agree == 1.0 else '⛔ NOT our labels/order'}")
            if agree != 1.0:
                rc = 1
        v, same_order, col = align(df, ids, want)
        auc = roc_auc_score(yy, v)
        per = [roc_auc_score(yy[folds == k], v[folds == k]) for k in range(folds.max() + 1)]
        print(f"  column '{col}'   rows {len(v)}   ids in our train.csv order: {same_order}")
        print(f"  solo/stack OOF AUC on OUR labels = {auc:.10f}")
        print("  per fold (OUR frozen partition): " + " ".join(f"{a:.6f}" for a in per))
        print(f"  fold spread max-min = {(max(per) - min(per)) * 1e6:+.1f}e-6")
        if claimed is not None:
            d = (auc - claimed) * 1e6
            verdict = "REPRODUCES" if abs(d) < 0.05 else "DOES NOT REPRODUCE"
            print(f"  published AUC {claimed:.10f}   ours - theirs = {d:+.3f}e-6   -> {verdict}")
            if abs(d) >= 0.05:
                rc = 1
        if pick is not None:
            print(f"  Pearson r with our pick's OOF = {np.corrcoef(v, pick)[0, 1]:.6f}")
        # w29g's binding test, on a single vector: the pack median IS this member.
        clears = auc >= FLOOR
        print(f"  w29 floor {PACK_MEDIAN} - {TOL} = {FLOOR:.4f}  ->  "
              f"{'CLEARS (worth the download, NOT an import)' if clears else 'REFUSED'}"
              f"   margin {(auc - FLOOR) / TOL:+.2f} x TOL")
    print("\n⛔ Clearing w29g admits nothing. w51's es-on-val clause is the next gate and this"
          "\n   file cannot see it. Six days out, that chain is not runnable.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
