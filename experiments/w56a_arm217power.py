"""w56 (2026-08-22) — WHAT THE 08-23 ARM 217 READ CAN AND CANNOT LICENSE.

`w48d_arm217.json` pre-registers, for the 08-23 slot-1 send of `hboyang_mix`'s raw
test vector:

    lb >= 0.97116 -> HONEST   (ARM 217's +41.7e-6 of CV is real)
    lb <= 0.97080 -> INFLATED (ARM 217's CV is an imported artefact; drop the member)
    in between    -> report it and change nothing

w48d itself discarded a first design as "Unfalsifiable". This run opened by suspecting
the surviving design of the same fault in the other direction — that its rejection
region is unreachable under the es-on-val mechanism it tests. **That suspicion is
REFUTED below and recorded as refuted** (part 2): at the extreme alternative the read
lands well inside the INFLATED region. The design is sound. Send it as registered.

What the run DID find is upstream of the thresholds and is new. Nobody had read this
notebook's source — w51 audited the other five members of `ext_members16` and left
`hboyang_mix` explicitly untested, reserved for this read. Read it (part 1) and the
vector is an AGGREGATOR over seven third-party OOF libraries, 138 of 149 streams, none
of them es-clearable, including the lookup-transformer family w51 convicted BY SOURCE
READ on our exact fold partition. So `w40d_prereg`'s rule already binds on ARM 217
whatever the read says, and part 3 registers the conjunction BEFORE the score exists.

This script changes NO threshold. `w48d_arm217.json` is read, never retyped.

⚠ `notebooks/` is gitignored, so the sources this script asserts against are NOT in the
repo. On a fresh checkout, re-pull them first — the assertions are the point, and a
re-pull that changes the source SHOULD fail this script loudly:

    kaggle kernels pull hboyang/s6e8-150-member-fusion \
        -p notebooks/w56/hboyang_fusion -m
    kaggle kernels output hboyang/s6e8-150-member-fusion \
        -p notebooks/w40/out/hboyang_s6e8-150-member-fusion

Deterministic. The one RNG is seeded and only builds the degradation carrier in 2b.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "w56", "hboyang_fusion", "s6e8-150-member-fusion.ipynb")
META = os.path.join(ROOT, "notebooks", "w56", "hboyang_fusion", "kernel-metadata.json")
LOG = os.path.join(ROOT, "notebooks", "w40", "out",
                   "hboyang_s6e8-150-member-fusion", "s6e8-150-member-fusion.log")
PREREG = os.path.join(ROOT, "experiments", "w48d_arm217.json")
OUT = os.path.join(ROOT, "experiments", "w56a_arm217power.json")

# The two anchors, from experiments/w21a_w{40,42}_ad2{11,17}stdcorr.json.
ARM211_BASE_AUC = 0.9701331846110894
ARM217_BASE_AUC = 0.9701748950350747
DISPUTED_GAIN = ARM217_BASE_AUC - ARM211_BASE_AUC          # +41.71e-6

report: dict = {}


def _src(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def _assert_in(hay: str, needle: str, where: str) -> None:
    """w51a discipline: every quoted line is re-asserted against its source."""
    if needle not in hay:
        raise SystemExit(f"⛔ SOURCE CHANGED: {where!r} no longer contains {needle!r}")


# ---------------------------------------------------------------------------
# 1. POOL AUDIT — what the vector under test is actually made of
# ---------------------------------------------------------------------------
def pool_audit() -> dict:
    print("=" * 78)
    print("1. POOL AUDIT — read from the notebook source, not from a summary")
    print("=" * 78)

    nb = _src(NB)
    meta = json.loads(_src(META))

    # The seven mounted datasets ARE the pool. Every one is a third-party OOF library.
    sources = meta["dataset_sources"]
    for s in sources:
        _assert_in(json.dumps(meta), s, "kernel-metadata.json")
    print(f"dataset_sources ({len(sources)}):")
    for s in sources:
        print(f"    {s}")

    # The member registry, lifted out of the notebook JSON.
    cell = json.loads(nb)
    src = "".join("".join(c["source"]) for c in cell["cells"] if c["cell_type"] == "code")
    _assert_in(src, "MEMBER_NAMES = [", "notebook")
    body = "[" + src.split("MEMBER_NAMES = [", 1)[1].split("\n]", 1)[0] + "]"
    import ast as _ast
    names = [str(x) for x in _ast.literal_eval(body)]
    assert len(names) == 149, f"expected 149 members, parsed {len(names)}"

    # These claims are load-bearing for part 3; re-assert each one.
    for q in [
        "# naji*   -> najiama blends",
        "# bolt_*  -> boltuzamaki library",
        "# sz_*    -> szymonkapiski library",
        "# golem_* -> dariushafshar golem library",
        "_blend_oof_predictions.csv",
    ]:
        _assert_in(src, q, "notebook")

    prefixes = {
        "naji (najiama BLENDS, level-2)": lambda n: n.startswith("naji"),
        "candidate_naji (blends OF blends)": lambda n: n.startswith("candidate_naji"),
        "bolt_ (boltuzamaki library)": lambda n: n.startswith("bolt_"),
        "sz_ (szymonkapiski 47-model library)": lambda n: n.startswith("sz_"),
        "fm_ (raykkretzschmar fm-lattice)": lambda n: n.startswith("fm_"),
        "golem_ (dariushafshar golem library)": lambda n: n.startswith("golem_"),
        "a_ (adarsh1077 library)": lambda n: n.startswith("a_"),
        "fresh_/local_ (hboyang's own)": lambda n: n.startswith(("fresh_", "local_")),
    }
    counts = {label: len([n for n in names if fn(n)]) for label, fn in prefixes.items()}
    assert sum(counts.values()) == len(names), \
        f"prefix buckets cover {sum(counts.values())} of {len(names)} members"

    print(f"\nMEMBER_NAMES parsed: {len(names)} (result.json says member_count 149)")
    for label, k in counts.items():
        print(f"    {k:4d}  {label}")

    # Everything except hboyang's own eleven is somebody else's OOF vector. The six
    # `candidate_naji*` are hboyang's blends OF third-party members, so they carry the
    # same streams and count as third-party too.
    third_party = len(names) - counts["fresh_/local_ (hboyang's own)"]
    print(f"\n  third-party streams: {third_party} / {len(names)} "
          f"= {100.0 * third_party / len(names):.1f}%")

    # Cross-reference against this workspace's own es-on-val convictions (w51).
    # w51 convicted, by reading the source logs:
    #   ern711_contextual, ern711_multilevel, ravi200_publicm12 (mhamza0810),
    #   ravi200_publicm13 (tamerlanomralinov lookup transformer),
    #   ravi200_l2stack1r  (clean stacker over contaminated columns -> Gate B)
    # The pool below carries the same families by name.
    families = {
        "lookup-transformer family (w51 Gate A FAIL on tamerlanomralinov)":
            [n for n in names if "lookup" in n],
        "sz_pub_* — szymonkapiski's copies of PUBLIC models":
            [n for n in names if n.startswith("sz_pub")],
        "ravi20076 (w51 Gate B FAIL as a level-2 over contaminated columns)":
            [n for n in names if "ravi" in n],
        "tabm / realmlp / resnet / ft-transformer NN members":
            [n for n in names if any(k in n for k in
                                     ("tabm", "rmlp", "realmlp", "resnet", "fttransformer",
                                      "tabnet", "deepfm", "gandalf", "tabr", "dcnv2"))],
    }
    print("\n  overlap with families this workspace has already convicted (w51):")
    for label, hit in families.items():
        print(f"    {len(hit):4d}  {label}")
        if hit:
            print(f"          e.g. {', '.join(hit[:6])}")

    # The log's own words: full-data fits produce the SUBMISSION; VALIDATE runs the CV.
    log = _src(LOG)
    for q in ["pool built: 149 members", "fitting dual on full data",
              "fitting regime on full data", "base meta-stack pooled OOF = 0.970119"]:
        _assert_in(log, q, "s6e8-150-member-fusion.log")
    print("\n  log re-asserted: 'pool built: 149 members', 'fitting dual on full data',")
    print("                   'fitting regime on full data'.")

    print("\n  ⇒ `hboyang_mix` is an AGGREGATOR over seven third-party OOF libraries.")
    print("    NONE of those streams is es-cleared, and this workspace cannot clear them:")
    print("    RESEARCH (w40) — \"an AGGREGATOR's streams cannot be es-cleared\".")

    return {"n_members": len(names), "counts": counts,
            "third_party": third_party, "dataset_sources": sources,
            "families": {k: len(v) for k, v in families.items()}}


# ---------------------------------------------------------------------------
# 2. THE delta AXIS — put both hypotheses on the units the prereg reads
# ---------------------------------------------------------------------------
def rank01(v: np.ndarray) -> np.ndarray:
    return (rankdata(v) - 0.5) / v.size


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def xfit_cv(cols: list[np.ndarray], y: np.ndarray, folds) -> float:
    """Cross-fitted logistic stack over `cols`; pooled OOF AUC on the frozen folds."""
    X = np.column_stack([logit(rank01(c)) for c in cols])
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m.fit(X[tr], y[tr])
        oof[va] = m.decision_function(X[va])
    return roc_auc_score(y, oof)


def delta_axis(pre: dict) -> dict:
    """delta = how much `oof_hboyang_mix` overstates the honest quality of the fusion.

    The SAME delta drives both sides of the prereg, and that is the whole reason the
    read is informative:
      - on CV, delta is why ARM 217 gained;
      - on LB, the test vector is NOT inflated by es-on-val (RESEARCH w40: it
        "inflates a member's OOF and not its test vector"), so the read comes in
        delta below the point prediction.
    So map every hypothesis onto delta, then read off which branch it fires.
    """
    print()
    print("=" * 78)
    print("2. THE delta AXIS — which branch does each hypothesis fire?")
    print("=" * 78)

    pred = float(pre["hboyang_pred"])
    hi = float(pre["rule"]["honest_at_or_above"])
    lo = float(pre["rule"]["inflated_at_or_below"])
    band = [float(x) for x in pre["hboyang_band"]]
    assert abs(pre["actual6"] - DISPUTED_GAIN * 1e6) < 0.01, "prereg gain moved"

    d_leave_honest = pred - hi          # delta at which the read stops saying HONEST
    d_enter_infl = pred - lo            # delta at which the read starts saying INFLATED

    # H_FULL: the entire standalone-AUC outlier is the artefact. w49b/w49c corrected
    # w48d's raw +885.9e-6 to +708.3e-6 after transforming every control the same way,
    # and RESEARCH says "Quote 708e-6".
    d_full = 708.3e-6

    print(f"  point prediction         : {pred:.6f}  (band {band[0]:.6f}..{band[1]:.6f},"
          f" width {(band[1] - band[0]) * 1e6:.0f}e-6)")
    print(f"  HONEST   at or above     : {hi:.5f}   -> fires for delta <= "
          f"{d_leave_honest * 1e6:.0f}e-6")
    print(f"  INFLATED at or below     : {lo:.5f}   -> fires for delta >= "
          f"{d_enter_infl * 1e6:.0f}e-6")
    print(f"  in between               : 'report it and change nothing'")

    print(f"\n  H_HONEST  delta = {0.0:8.1f}e-6 -> lb {pred:.5f} -> HONEST")
    print(f"  H_FULL    delta = {d_full * 1e6:8.1f}e-6 -> lb {pred - d_full:.5f} -> "
          f"{'INFLATED' if pred - d_full <= lo else 'INDETERMINATE'}"
          f"   (w49c's corrected standalone gap; RESEARCH says quote 708e-6)")

    dead_lo = d_leave_honest / d_full
    dead_hi = d_enter_infl / d_full
    print(f"\n  ⇒ the test IS powered against the extreme alternative: at H_FULL the read")
    print(f"    lands {(lo - (pred - d_full)) * 1e6:.0f}e-6 clear inside the INFLATED region.")
    print(f"  ⇒ but the INDETERMINATE zone is delta in "
          f"[{d_leave_honest * 1e6:.0f}, {d_enter_infl * 1e6:.0f}]e-6, i.e. "
          f"{100 * dead_lo:.0f}%..{100 * dead_hi:.0f}% of H_FULL —")
    print(f"    {100 * (dead_hi - dead_lo):.0f} percentage points of the hypothesis space")
    print(f"    maps to 'change nothing'. PARTIAL contamination is the modal outcome.")

    return {"pred": pred, "honest_at": hi, "inflated_at": lo, "band": band,
            "d_leave_honest": d_leave_honest, "d_enter_inflated": d_enter_infl,
            "d_full": d_full, "lb_at_full": pred - d_full,
            "dead_zone_frac": [dead_lo, dead_hi]}


def marginal_value(y, folds) -> dict:
    """What is `hboyang_mix` actually worth to a stack, measured not quoted.

    ⚠ INSTRUMENT LIMITATION, recorded so it is not re-spent. The degradation family
    below blends the column's ranks toward an independent uniform. That lowers its
    standalone AUC very fast while leaving its rank-correlation with the honest signal
    high, so the stacker keeps extracting value long after the AUC has collapsed. The
    curve is therefore NOT usable as a delta -> stack-CV transfer, and the earlier
    attempt in this run to read a delta off its local slope is WITHDRAWN. What the
    sweep does establish, and what is kept, is the level at lambda = 0.
    """
    print()
    print("=" * 78)
    print("2b. WHAT THE COLUMN IS WORTH TO A STACK — measured on the frozen folds")
    print("=" * 78)

    base = np.load(os.path.join(ROOT, "submissions", "oof_w40_ad211std_h3.npy"))
    h = np.load(os.path.join(ROOT, "data", "ext_members16", "oof_hboyang_mix.npy"))
    assert base.shape == y.shape and h.shape == y.shape

    auc_base = roc_auc_score(y, base)
    auc_h = roc_auc_score(y, h)
    print(f"  standalone AUC  ARM 211 h3 base : {auc_base:.10f}")
    print(f"  standalone AUC  oof_hboyang_mix : {auc_h:.10f}   "
          f"(result.json pooled_oof_auc_mix 0.9701815536)")

    cv0 = xfit_cv([base], y, folds)
    cv1 = xfit_cv([base, h], y, folds)
    gain = cv1 - cv0
    print(f"\n  cross-fitted 1-col control (ARM 211 base alone) : {cv0:.10f}")
    print(f"  cross-fitted 2-col (base + hboyang_mix)         : {cv1:.10f}")
    print(f"  marginal value of the column                    : {gain * 1e6:+.2f}e-6")
    print(f"  what the ARM 211 -> ARM 217 pack refit delivered "
          f"for SIX members       : {DISPUTED_GAIN * 1e6:+.2f}e-6")
    print(f"\n  ⇒ optimally combined against the pack's own output the column is worth")
    print(f"    {gain / DISPUTED_GAIN:.1f}x what the 217-column refit extracted from all six.")
    print(f"    The pack's own stacker UNDER-uses it. That is a fact about the combiner,")
    print(f"    not evidence either way on contamination.")

    # The sweep is retained for the record, with its limitation stated above.
    rng = np.random.default_rng(0)
    carrier = rng.random(len(y))
    rh = rank01(h)
    rows = []
    print(f"\n  degradation sweep (⚠ NOT a delta transfer — see the docstring):")
    print(f"  {'lambda':>7} {'AUC(col)':>13} {'dAUC(col)':>13} {'2-col CV':>14} {'gain':>12}")
    for lam in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0):
        z = (1.0 - lam) * rh + lam * carrier
        a = roc_auc_score(y, z)
        cv = xfit_cv([base, z], y, folds)
        rows.append({"lam": lam, "auc_col": a, "d_auc_col": a - auc_h, "cv": cv,
                     "gain": cv - cv0})
        print(f"  {lam:7.2f} {a:13.8f} {(a - auc_h) * 1e6:12.1f}e-6 {cv:14.10f} "
              f"{(cv - cv0) * 1e6:+11.2f}e-6")
    print(f"\n  Read the level, not the slope: at lambda 0.02 the column has lost "
          f"{-rows[1]['d_auc_col'] * 1e6:.0f}e-6 of AUC")
    print(f"  and still carries {rows[1]['gain'] * 1e6:+.1f}e-6 — an honest member "
          f"{-rows[1]['d_auc_col'] * 1e6:.0f}e-6 weaker would not.")

    return {"auc_base": auc_base, "auc_h": auc_h, "cv0": cv0, "cv1": cv1,
            "gain": gain, "gain_vs_pack": gain / DISPUTED_GAIN, "sweep": rows}


# ---------------------------------------------------------------------------
# 3. WHAT THE READ CAN LICENSE — the conjunction rule, registered BEFORE the read
# ---------------------------------------------------------------------------
def conjunction(pool: dict, axis: dict) -> dict:
    print()
    print("=" * 78)
    print("3. WHAT A HONEST READ CAN AND CANNOT LICENSE (registered before the read)")
    print("=" * 78)

    tp = pool["third_party"]
    n = pool["n_members"]
    print(f"  Part 1 established, from the notebook source: {tp}/{n} of the streams in")
    print(f"  `hboyang_mix` are third-party, drawn from {len(pool['dataset_sources'])} "
          f"public OOF libraries,")
    def _fam(sub: str) -> int:
        for k, v in pool["families"].items():
            if sub in k:
                return v
        return 0

    print(f"  including {_fam('lookup-transformer')} lookup-transformer-family members "
          f"and {_fam('sz_pub_')} `sz_pub_*` copies of public models.")
    print(f"  NONE of them is es-cleared, and this workspace CANNOT clear them:")
    print(f"  RESEARCH (w40) — \"an AGGREGATOR's streams cannot be es-cleared\".")
    print(f"  w51 convicted the lookup-transformer family by source read, on OUR EXACT")
    print(f"  fold partition (StratifiedKFold(5, shuffle=True, random_state=42)).")

    print(f"\n  So the w40d rule ALREADY binds on ARM 217, independent of the 08-23 read:")
    print(f"    \"an arm containing un-es-cleared streams may be built, priced, queued")
    print(f"     and sent, but is NOT ELIGIBLE for check_selection.WANTED on CV alone.\"")

    print(f"\n  THE CONJUNCTION, registered now, before the score exists:")
    print(f"    INFLATED (lb <= {axis['inflated_at']:.5f}) -> as registered: drop")
    print(f"        `hboyang_mix`, the ad217 veto is FINAL. Decisive; the branch is")
    print(f"        reachable (H_FULL lands at {axis['lb_at_full']:.5f}).")
    print(f"    in between -> as registered: report it, change nothing. The veto stands.")
    print(f"        This is the MODAL outcome — the indeterminate zone spans "
          f"{100 * axis['dead_zone_frac'][0]:.0f}%..{100 * axis['dead_zone_frac'][1]:.0f}%")
    print(f"        of the contamination hypothesis.")
    print(f"    HONEST (lb >= {axis['honest_at']:.5f}) -> the member is not GROSSLY")
    print(f"        contaminated, and w45 §3 / w44's import line are reopened on the")
    print(f"        record exactly as w48d says. ⚠ IT DOES NOT CLEAR w40d. ARM 217 stays")
    print(f"        WANTED-INELIGIBLE on CV, because HONEST rules out gross contamination")
    print(f"        and nothing narrower — delta <= {axis['d_leave_honest'] * 1e6:.0f}e-6 is")
    print(f"        still enough contamination to be worth more than the whole disputed")
    print(f"        +{DISPUTED_GAIN * 1e6:.1f}e-6 in a stack that under-uses the column by 2x.")

    print(f"\n  ⇒ The only thing on disk inside gold's range is the ad217 family, and NO")
    print(f"    branch of this read puts it there on CV. A HONEST read is a licence to")
    print(f"    SEND and price ad217 on the LB, not a licence to SELECT it.")
    print(f"    Selecting on a public read is the Rogii failure. Do not.")

    return {"binds": "w40d", "third_party": tp, "n": n}
def main() -> None:
    print("w56a — ARM 217 / hboyang_mix: what the 08-23 registered read can resolve")
    print("This script changes NO threshold. w48d_arm217.json is read, not retyped.\n")

    pre = json.load(open(PREREG))
    report["pool"] = pool_audit()
    report["axis"] = delta_axis(pre)

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    y = tr[common.TARGET].to_numpy(np.int8)
    folds = common.get_folds(y)
    del tr

    report["value"] = marginal_value(y, folds)
    report["conjunction"] = conjunction(report["pool"], report["axis"])

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  The 08-23 read is WELL POSED and its INFLATED branch is reachable — my")
    print("  opening hypothesis this run (that it was unfalsifiable) is REFUTED by the")
    print("  numbers above and is recorded as refuted. Send it as registered.")
    print("  What is NEW: the read's HONEST branch does not clear w40d, because the")
    print("  vector is an aggregator over 138 un-es-clearable third-party streams.")
    print("  Registered before the score exists, so it cannot be argued backwards.")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open(OUT, "w") as fh:
        json.dump(_clean(report), fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
