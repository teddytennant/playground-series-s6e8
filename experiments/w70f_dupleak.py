"""w70f — THE TRAIN↔TEST DUPLICATE LEAK: exactly two rows, worth +1.62e-6, and free.

⚠⚠ WHY THIS EXISTS AT ALL, WHICH IS THE POINT WORTH MORE THAN THE 1.62e-6.
JOURNAL line 19481 (w?? , 2026-08-2x) recorded, as a CLOSURE:

    "Train has no exact feature duplicates (691,369 rows -> 691,369 distinct on NUM+CAT), so
     duplicate-group leakage tests are unavailable here. Checked this run, recorded so no one
     re-checks."

**That measurement is correct and it closes the WRONG QUESTION.** Train-internal duplicates are
about LEAKAGE WITHIN TRAIN — whether CV folds must be grouped. TRAIN↔TEST overlap is a different
question entirely: whether any TEST row's label is already known. The note answered the first,
and its final clause ("so no one re-checks") closed the second, which nobody had asked.
⛔ **A NULL RECORDED AGAINST THE WRONG QUESTION CLOSES THE RIGHT ONE**, and it stayed closed for
every run since. It was reopened only because a public notebook hard-codes two test ids.

WHAT IS ACTUALLY TRUE (measured here, not quoted):
  * train has 691,369 distinct feature vectors over the 12 features — the old note is right.
  * EXACTLY 2 of the 296,302 test rows match a train row on the full feature vector:
        id 735378 -> label 1     id 862871 -> label 0
    Both matched train groups are pure (single label), so the labels are not inferred, they are
    READ. These are precisely the two ids `amanatar/s6e8-elite-rank-average-ensemble-0-97123`
    hard-codes as "duplicate magic" — independently rediscovered here from the data.

WHAT IT IS WORTH, computed from our own OOF rather than assumed (see `value()`):
  our stacks put both rows near the 33rd percentile; the positive belongs at the top and the
  negative at the bottom.
        positive @33.2%  beats 90.7% of negatives  -> +0.441e-6 on the full test set
        negative @33.8%  above  10.2% of positives -> +1.181e-6
        TOTAL                                       **+1.62e-6**
  Each row is in the public slice with probability 0.20 and the conditional public gains are
  +2.20e-6 and +5.91e-6, so the EXPECTED public gain is 0.20 x 8.11 = +1.62e-6 as well.

✅ **THIS IS NOT PUBLIC-LB CHASING AND THAT DISTINCTION IS THE WHOLE REASON IT IS SAFE.** The
brief's Rogii warning is about hedges tuned to the public slice that die on the private one.
This is two ground-truth labels: it moves public and private by the SAME +1.62e-6 in
expectation, it is deterministic, it cannot overfit, and it is orthogonal to every model.
For scale: it is larger than the 1.05e-6 lexb price w65 spent four seeds measuring, and ~5% of
the 33e-6 gap to the leader.

⛔ THIS SCRIPT DOES NOT APPLY THE OVERRIDE TO ANY REGISTERED FILE. The 08-24 and 08-25 tens are
registered and md5-pinned; rewriting a planned file behind its plan's back is exactly what this
workspace's machinery exists to prevent. A later run should REGISTER the change, then apply it.

🎯 **HOW TO APPLY IT, AND WHY `--inplace` IS THE RIGHT MODE RATHER THAN A RENAME.** The override
touches TEST ids only, so a file's cross-fitted CV — computed by `w23b_sendqueue` as
`fast_auc(y, submissions/oof_<name>.npy)` over TRAIN rows — is **completely unchanged**. Only the
md5 moves, and `w23b` recomputes that from the file. So the whole chain stays self-consistent iff
the override is applied **BEFORE step 1**:

    w70f --inplace <stem>...        # rewrites submissions/<stem>.csv, backs up to .pre_dupleak
    w23b_sendqueue.py               # re-globs, recomputes md5 AND cv -> both correct
    w48e_order.py --day <DAY> --write ; w26g_send.py --n 10 ; --go

⛔ **DO NOT reach for `--apply IN OUT` for this.** A renamed copy has no `oof_<newname>.npy`, so
`w23b` records `cv = nan`, drops it from the ranked queue and `w48e` cannot defend it — the file
becomes unsendable. `--apply` exists for scratch verification, not for the send path.

    .venv/bin/python experiments/w70f_dupleak.py                      # find, verify, value
    .venv/bin/python experiments/w70f_dupleak.py --apply A.csv B.csv  # scratch copy, never in place
    .venv/bin/python experiments/w70f_dupleak.py --inplace stem [stem...]   # the send-path mode
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from common import SUB, TARGET, load_raw                              # noqa: E402

N_TEST, BASE_RATE, PUB_FRAC = 296_302, 0.7094243450313797, 0.20
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def _keys(df, feat):
    """Row-wise feature tuples, NaN-safe. ⚠ A plain str-join raises on NaN and a `.astype(str)`
    join maps NaN to 'nan', which COLLIDES with a literal 'nan' category. Sentinel, then tuple."""
    v = df[feat].astype(object).where(pd.notna(df[feat]), "<NA>").values
    return pd.Series(list(map(tuple, v)), index=df.index)


def find():
    tr, te = load_raw()
    feat = [c for c in tr.columns if c not in ("id", TARGET)]
    ktr, kte = _keys(tr, feat), _keys(te, feat)

    # GATE 1 — reproduce the OLD note's measurement. If train-internal duplicates HAVE appeared,
    # the two questions are entangled again and the reading below is not clean.
    if ktr.nunique() != len(tr):
        fail(f"train now has {len(tr) - ktr.nunique()} internal duplicate feature vectors — "
             f"the train↔test reading below is no longer a clean separate question")
    print(f"  train: {ktr.nunique()} distinct feature vectors of {len(tr)} rows "
          f"({len(feat)} features) — no internal duplicates, as the old note said.")

    # ⚠ PLAIN DICTS, NOT A GROUPBY INDEX. The keys are TUPLES, and `frame.loc[a_tuple]` is a
    # MULTI-AXIS indexer to pandas — it raises `Too many indexers` rather than looking up a row.
    tot, pos = {}, {}
    for k, yy in zip(ktr.values, tr[TARGET].astype(int).values):
        tot[k] = tot.get(k, 0) + 1
        pos[k] = pos.get(k, 0) + int(yy)
    hits = {}
    for i, k in kte.items():
        if k in tot:
            hits[int(te.at[i, "id"])] = dict(label=pos[k] / tot[k], n_train=tot[k])

    # GATE 2 — every matched group must be PURE. An impure group gives a probability, not a
    # label, and hard-coding 0/1 from it would be worse than leaving the model's own estimate.
    impure = {i: h for i, h in hits.items() if h["label"] not in (0.0, 1.0)}
    if impure:
        fail(f"matched train groups are NOT label-pure: {impure} — do NOT hard-code these")
    print(f"  test↔train exact matches: {len(hits)} of {N_TEST} rows, "
          f"{len(hits) - len(impure)} label-pure.")
    for i, h in sorted(hits.items()):
        print(f"     id {i} -> label {int(h['label'])}   ({h['n_train']} train row(s))")
    return {i: int(h["label"]) for i, h in hits.items() if not impure}


def value(overrides, oof_name="w36_ad199stdcorr_ens4"):
    """What the override is worth, in e-6 of test AUC, from OUR OWN OOF — never assumed."""
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    o = np.load(os.path.join(SUB, f"oof_{oof_name}.npy")).astype(np.float64)
    sub = pd.read_csv(os.path.join(SUB, f"{oof_name}.csv")).set_index("id")["addicted_label"]
    pct = sub.rank(pct=True)
    n1 = int(round(BASE_RATE * N_TEST))
    n0 = N_TEST - n1
    tot = 0.0
    print(f"\n  what it is worth, priced off {oof_name}'s own OOF and test ranks:")
    for i, lab in sorted(overrides.items()):
        thr = float(np.quantile(o, float(pct[i])))
        if lab == 1:                      # to the TOP: now beats every negative
            g = (1.0 - float((o[y == 0] < thr).mean())) / n1
            note = f"beats {float((o[y == 0] < thr).mean()) * 100:.1f}% of negatives"
        else:                             # to the BOTTOM: now beaten by every positive
            g = float((o[y == 1] < thr).mean()) / n0
            note = f"above {float((o[y == 1] < thr).mean()) * 100:.1f}% of positives"
        tot += g
        print(f"     id {i}  true {lab}  our p {sub[i]:.6f} @ {pct[i] * 100:5.2f}%  "
              f"({note})  -> +{g * 1e6:.3f}e-6")
    print(f"     TOTAL on the full test set: **+{tot * 1e6:.3f}e-6**")
    print(f"     each row is in the public slice w.p. {PUB_FRAC}, so the EXPECTED public gain "
          f"is the same +{tot * 1e6:.3f}e-6 —\n     it is ground truth, not a slice-specific "
          f"hedge, so public and private move together.")
    return tot * 1e6


def apply_to(src, dst, overrides):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("⛔ refusing to overwrite in place — pass a distinct output path")
    if os.path.exists(dst):
        raise SystemExit(f"⛔ {dst} exists; refusing to overwrite")
    d = pd.read_csv(src)
    assert len(d) == N_TEST and list(d.columns) == ["id", "addicted_label"], (
        f"{src} is not a submission: {d.shape} {list(d.columns)}")
    before = {i: float(d.loc[d.id == i, "addicted_label"].iloc[0]) for i in overrides}
    for i, lab in overrides.items():
        n = int((d.id == i).sum())
        assert n == 1, f"id {i} appears {n} times in {src}"
        d.loc[d.id == i, "addicted_label"] = float(lab)
    assert d.addicted_label.notna().all() and len(d) == N_TEST
    d.to_csv(dst, index=False)
    print(f"  wrote {dst}")
    for i in overrides:
        print(f"     id {i}: {before[i]:.6f} -> {float(overrides[i]):.1f}")


def inplace(stems, overrides):
    """Rewrite submissions/<stem>.csv in place, keeping the OOF pairing (and so the CV) intact.

    ⚠ A backup is written FIRST and the run aborts if one already exists — a second application
    would be a no-op on the values but would overwrite the only copy of the original."""
    for stem in stems:
        src = os.path.join(SUB, f"{stem}.csv")
        bak = src + ".pre_dupleak"
        if not os.path.exists(src):
            raise SystemExit(f"⛔ {src} does not exist")
        if os.path.exists(bak):
            raise SystemExit(f"⛔ {bak} already exists — {stem} looks already overridden; "
                             f"refusing to clobber the original")
        if not os.path.exists(os.path.join(SUB, f"oof_{stem}.npy")):
            raise SystemExit(f"⛔ {stem} has no oof_{stem}.npy — w23b would rank it cv=nan and "
                             f"w48e could not defend it. Refusing.")
        d = pd.read_csv(src)
        assert len(d) == N_TEST and list(d.columns) == ["id", "addicted_label"], stem
        os.replace(src, bak)
        before = {i: float(d.loc[d.id == i, "addicted_label"].iloc[0]) for i in overrides}
        for i, lab in overrides.items():
            assert int((d.id == i).sum()) == 1, f"id {i} not unique in {stem}"
            d.loc[d.id == i, "addicted_label"] = float(lab)
        assert d.addicted_label.notna().all() and len(d) == N_TEST
        d.to_csv(src, index=False)
        print(f"  {stem}: " + ", ".join(
            f"id {i} {before[i]:.6f}->{float(overrides[i]):.1f}" for i in sorted(overrides))
            + f"   (original at {os.path.basename(bak)})")
    print("\n  ⚠ NOW RE-RUN `w23b_sendqueue.py` so the queue's md5 matches. The CV is unchanged "
          "by\n    construction — the override touches TEST ids and the CV is an OOF/TRAIN "
          "quantity.")


def main() -> None:
    print("=" * 92)
    print("w70f  TRAIN↔TEST DUPLICATE LEAK")
    print("=" * 92 + "\n")
    ov = find()
    if FAILURES:
        print(f"\n  FAILURES {FAILURES} — not pricing.")
        sys.exit(1)
    if not ov:
        print("\n  no usable overrides.")
        return
    gain = value(ov)
    with open(os.path.join(HERE, "w70f_dupleak.json"), "w") as f:
        json.dump(dict(overrides={str(k): v for k, v in ov.items()},
                       gain_e6_full_test=gain, failures=FAILURES), f, indent=1)
    print(f"\n  wrote w70f_dupleak.json   FAILURES {FAILURES}")
    print("  ⛔ NOT APPLIED to any registered file. Use --apply <in> <out> after registering it.")


if __name__ == "__main__":
    if "--inplace" in sys.argv:
        k = sys.argv.index("--inplace")
        o = find()
        assert not FAILURES, "gates failed; refusing to apply"
        stems = sys.argv[k + 1:]
        if not stems:
            raise SystemExit("⛔ --inplace needs at least one stem")
        inplace(stems, o)
    elif "--apply" in sys.argv:
        k = sys.argv.index("--apply")
        o = find()
        assert not FAILURES, "gates failed; refusing to apply"
        apply_to(sys.argv[k + 1], sys.argv[k + 2], o)
    else:
        main()
