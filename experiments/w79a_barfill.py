"""w79a — ADOPT THE HOLE-FILLED HIJACK BAR, OR REFUSE TO.

Pre-registration: experiments/w79_prereg.txt, committed 0109708 BEFORE this file existed.

WHAT IS BEING DECIDED
  w63a's hijack break-even is interpolated across a 16.49e-6 hole in its own design and the
  live CV bar 0.9701288617 comes off that straight line. w77a found 13 scored files sitting
  inside the hole; w77d filled it and read 5.255e-6 -> 0.9701347511, 5.9e-6 STRICTER; w78
  showed the one contested modelling choice (withhold a realised score from the GLS, or not)
  is worth 0.36e-6 against the hole's 5.56e-6. What was still owed: a filler rule fixed before
  the board is read, a re-derivation through w63a's own main(), and the send impact.

WHAT THIS FILE DOES NOT DO
  * It does not touch `w63a_setprice.json`. md5 is taken before and after and compared. Every
    GATE A and pin in w76a/w77a/w77b/w77c/w77d/w78a/w78b reads that file and must keep working.
  * It does not change one line of `w63a.fit`. The fillers are withheld from the dict `fit` is
    handed, which is the treatment w63a already gives an unsent candidate (w77d's observation).
  * It does not submit, requeue, replan, or refit a model.

THE THREE ARMS, all through `w63a.main()`, same board, same estimator, same day:
  CONTROL  fill=()     -> w79a_control.json   the live-board bar with the hole still open
  PRIMARY  fill=FILL   -> w79a_barfill.json   scored files inside the bracket
  ROBUST   fill=FILLA  -> w79a_fillall.json   every OOF file inside the bracket, scored or not

⚠ CONTROL is the comparison, NOT the artefact on disk. w63a_setprice.json was derived on
yesterday's board; comparing the filled bar against it would confound the hole with the board's
overnight turnover, which w78 measured at +0.334e-6. Everything below compares like with like.

    .venv/bin/python experiments/w79a_barfill.py
"""
from __future__ import annotations

import contextlib, hashlib, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import SUB, TARGET, load_raw                       # noqa: E402
from w16b_cellweight import fast_auc                           # noqa: E402
import w63a_setprice as W63A                                   # noqa: E402 — the instrument
import w76a_addtest as W76A                                    # noqa: E402 — the board read

U = 1e-6

# ---- THE REGISTERED BARS (w79_prereg.txt). Read, never re-derived from what comes back. -------
P2_BAR = 1e-9        # GATE D: the GLS common gap must not move
P3_BAR = 1.0e-6      # w63a's own GATE R tolerance, applied to base and worthless
P4_BAR = 5.0         # e-6, the binding bracket width once filled (w63a's is 16.4916)
P5_REF = 0.9701288617   # the live bar the move must be STRICTER than
P6_REF, P6_BAR = 0.9701347511, 3.0e-6   # w77d's rewound-board proposal, RECORDED not gating
P7_BAR = 2.0e-6      # FILL vs FILLA — if the bar depends on this, it is not identified
P1_MIN = 10          # RECORDED not gating

GATING = ("P2", "P3", "P4", "P5", "P7")     # P9 is checked outside this file — see §P9 below


def md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def run_arm(label: str, fill, outfile: str) -> dict | None:
    """One `w63a.main()` call, stdout to its own log so three arms do not interleave."""
    log = os.path.join(HERE, f"w79a_{label}.log")
    print(f"  arm {label:8s} fill={len(fill):3d} -> {outfile}  (log w79a_{label}.log) ", end="",
          flush=True)
    with open(log, "w") as fh, contextlib.redirect_stdout(fh):
        try:
            res = W63A.main(fill=fill, outfile=outfile)
        except SystemExit as e:
            print(f"\nSystemExit {e.code}")
            res = None
    if res is None:
        print("⛔ REFUSED — read the log")
    else:
        print(f"ok   H_bind {res['H_binding']:.4f}e-6   bar {res['cv_bar_new']:.10f}")
    return res


def main() -> None:
    before = md5(os.path.join(HERE, "w63a_setprice.json"))
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    # ⚠ READ, never re-typed (w63 §9). These two numbers ARE the hole.
    LO = float(ref["bracket"]["uncond"]["lo_dcv"])
    HI = float(ref["bracket"]["uncond"]["hi_dcv"])
    PICK = ref["pick"]

    print("=" * 96)
    print("w79a — the hole-filled hijack bar. Prereg experiments/w79_prereg.txt (0109708).")
    print("=" * 96)
    print(f"  w63a's bracket: {ref['bracket']['uncond']['hi']} {HI:+.4f} .. "
          f"{ref['bracket']['uncond']['lo']} {LO:+.4f}   width "
          f"{float(ref['bracket']['uncond']['width']):.4f}e-6, EMPTY")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    lb_all, t1, t2, t1v, t2v = W76A.board()
    print(f"\n  LIVE board: {len(lb_all)} distinct scored files; "
          f"slot1 {t1v:.5f} {t1}; slot2 {t2v:.5f} {len(t2)} files")

    # BASE is w63a's own design, rebuilt from the LIVE board by w63a's own expression. If the
    # live tiers have moved off the artefact's, GATE T2 inside main() will refuse anyway; this
    # records the fact here so the log says WHY rather than leaving it to a stack trace.
    BASE = sorted(set(t1) | set(t2) | set(W63A.WANTED) | set(W63A.PLAN_0824))
    ref_base = sorted(set(ref["tiers"]["slot1"]) | set(ref["tiers"]["slot2"])
                      | set(ref["wanted"]) | set(ref["plan_0824"]))
    print(f"  BASE design {len(BASE)} files; matches the stored artefact's: {BASE == ref_base}")

    # ---------------------------------------------------------------- the registered rule
    stems = sorted(f[4:-4] for f in os.listdir(SUB)
                   if f.startswith("oof_") and f.endswith(".npy"))
    print(f"\n  {len(stems)} OOF vectors on disk. Computing CV for all of them "
          f"(never re-quoted from a table) ...", flush=True)
    cv = {k: fast_auc(y, np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64))
          for k in stems}
    cvp = cv[PICK]
    dcv = {k: (v - cvp) / U for k, v in cv.items()}

    inside = sorted((k for k in stems if k not in set(BASE) and LO < dcv[k] < HI),
                    key=lambda k: -dcv[k])
    FILL = [k for k in inside if k in lb_all]
    FILLA = list(inside)
    print(f"\n=== THE REGISTERED FILLER RULE, applied to the live board ===")
    print(f"  {'stem':32s} {'dCV':>9s}  scored  in FILL")
    for k in inside:
        print(f"  {k:32s} {dcv[k]:+9.4f}  {'yes' if k in lb_all else ' no':>6s}  "
              f"{'*' if k in FILL else ''}")
    print(f"\n  FILL  (scored, inside the hole, not in BASE): {len(FILL)}")
    print(f"  FILLA (every OOF inside the hole, not in BASE): {len(FILLA)}")

    # ---------------------------------------------------------------- the three arms
    print(f"\n=== THREE ARMS THROUGH w63a.main(), same board, same estimator ===")
    C = run_arm("control", (), "w79a_control.json")
    F = run_arm("barfill", FILL, "w79a_barfill.json")
    A = run_arm("fillall", FILLA, "w79a_fillall.json")
    if C is None or F is None:
        print("\n⛔ an arm refused. Nothing is adopted and no verdict is claimed.")
        json.dump(dict(adopted=False, reason="an arm refused inside w63a.main()",
                       fill=FILL, filla=FILLA),
                  open(os.path.join(HERE, "w79a_verdicts.json"), "w"), indent=1)
        sys.exit(1)

    # ---------------------------------------------------------------- the registered verdicts
    stats = dict(
        n_fill=len(FILL), n_filla=len(FILLA),
        gap_control=float(C["gap"]), gap_fill=float(F["gap"]),
        d_gap=abs(float(F["gap"]) - float(C["gap"])),
        base_control=float(C["base"]), base_fill=float(F["base"]),
        d_base=abs(float(F["base"]) - float(C["base"])),
        worthless_control=float(C["worthless_limit1"]),
        worthless_fill=float(F["worthless_limit1"]),
        d_worthless=abs(float(F["worthless_limit1"]) - float(C["worthless_limit1"])),
        width_fill=max(b["width"] for b in F["bracket"].values()),
        width_control=max(b["width"] for b in C["bracket"].values()),
        bar_control=float(C["cv_bar_new"]), bar_fill=float(F["cv_bar_new"]),
        bar_fillall=(float(A["cv_bar_new"]) if A else None),
        bar_stored=float(ref["cv_bar_new"]),
        H_control=float(C["H_binding"]), H_fill=float(F["H_binding"]),
        d_fill_filla=(abs(float(A["cv_bar_new"]) - float(F["cv_bar_new"])) if A else None),
        d_w77d=abs(float(F["cv_bar_new"]) - P6_REF),
    )
    P = {
        "P1": (stats["n_fill"] >= P1_MIN,
               f"|FILL| = {stats['n_fill']} (bar >= {P1_MIN})"),
        "P2": (stats["d_gap"] < P2_BAR,
               f"|d gap| = {stats['d_gap']:.3e} (bar {P2_BAR:.0e}) — the GLS is untouched"),
        "P3": (stats["d_base"] < P3_BAR and stats["d_worthless"] < P3_BAR,
               f"|d base| = {stats['d_base']:.3e}, |d worthless| = {stats['d_worthless']:.3e} "
               f"(bar {P3_BAR:.1e})"),
        "P4": (stats["width_fill"] < P4_BAR,
               f"binding bracket {stats['width_fill']:.4f}e-6 (bar {P4_BAR}), was "
               f"{stats['width_control']:.4f}e-6 unfilled"),
        "P5": (stats["bar_fill"] > P5_REF,
               f"bar {stats['bar_fill']:.10f} vs live {P5_REF:.10f} "
               f"({(stats['bar_fill'] - P5_REF)/U:+.3f}e-6)"),
        "P6": (stats["d_w77d"] < P6_BAR,
               f"|bar - w77d {P6_REF:.10f}| = {stats['d_w77d']/U:.3f}e-6 (bar {P6_BAR/U:.1f}e-6)"),
        "P7": (stats["d_fill_filla"] is not None and stats["d_fill_filla"] < P7_BAR,
               f"|bar(FILLA) - bar(FILL)| = "
               f"{(stats['d_fill_filla'] or float('nan'))/U:.3f}e-6 (bar {P7_BAR/U:.1f}e-6)"),
    }
    print("\n" + "=" * 96)
    print("THE REGISTERED PREDICTIONS")
    print("=" * 96)
    for k in sorted(P):
        ok, why = P[k]
        tag = "GATING" if k in GATING else "recorded"
        print(f"  {k}  {'✅ CONFIRMED' if ok else '⛔ FALSIFIED'}  [{tag:8s}]  {why}")

    # ---------------------------------------------------------------- P8, the send impact
    print("\n" + "=" * 96)
    print("P8 — THE SEND IMPACT [recorded, NOT gating: blocking a weak row is the POINT]")
    print("=" * 96)
    # ⚠ UNSENT is determined from the LIVE board, not from the queue's own `sent` column — that
    # column was written before the 08-24 ten went out and this run does not refresh the queue.
    # The gate the bar sits behind is `w26g_send.above_tier_reason`, which only fires on a row
    # predicted to land AT OR ABOVE the tier; `cv < bar` is that function's own clause, quoted.
    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    thresh = float(C["threshold_public"])
    newly, unsent = [], 0
    for _, r in q.iterrows():
        stem = str(r["file"]).replace(".csv", "")
        if stem not in cv or stem in lb_all:
            continue
        unsent += 1
        if cv[stem] >= stats["bar_control"] and cv[stem] < stats["bar_fill"]:
            newly.append(dict(stem=stem, cv=float(cv[stem]), priority=float(r["priority"]),
                              pred_lb=float(r["pred_lb"]),
                              at_or_above=bool(float(r["pred_lb"]) >= thresh)))
    nonveto = [d for d in newly if d["priority"] >= 0]
    reachable = [d for d in nonveto if d["at_or_above"]]
    print(f"  {unsent} unsent queue rows have an OOF vector; threshold_public {thresh:.6f}")
    print(f"  {len(newly)} of them are admitted by the control bar and blocked by the filled one:")
    for d in sorted(newly, key=lambda d: -d["cv"]):
        print(f"    {d['stem']:32s} CV {d['cv']:.10f}  pred_lb {d['pred_lb']:.6f}  "
              f"priority {d['priority']:+.0f}  "
              f"{'⚠ NON-VETOED' if d['priority'] >= 0 else 'already vetoed'}"
              f"{'  AND predicted AT/ABOVE tier' if d['at_or_above'] else ''}")
    print(f"  NON-VETOED newly blocked: {len(nonveto)};  of those, predicted at/above the tier "
          f"so the gate would actually fire: {len(reachable)}")
    above = sorted(((cv[st] - stats["bar_fill"]) / U, st)
                   for st in (str(f).replace(".csv", "") for f in q["file"])
                   if st in cv and st not in lb_all and cv[st] >= stats["bar_fill"])
    if above:
        print(f"  nearest unsent queue row still ABOVE the filled bar: {above[0][1]} by "
              f"{above[0][0]:+.3f}e-6")

    # ---------------------------------------------------------------- the adoption rule
    gating_ok = all(P[k][0] for k in GATING)
    print("\n" + "=" * 96)
    print(f"REGISTERED ADOPTION RULE: P2 and P3 and P4 and P5 and P7 and P9.")
    print(f"  P2..P7 gating set: {'ALL HOLD' if gating_ok else 'NOT SATISFIED — ' + ', '.join(k for k in GATING if not P[k][0])}")
    print(f"  P9 (the 08-25 ten is unchanged) is checked OUTSIDE this file, by pointing")
    print(f"     w26g_send.HIJACKPRICE at w79a_barfill.json and dry-running the sender. This")
    print(f"     file therefore records `adoption_cleared_here`, NOT `adopted`.")
    print("=" * 96)

    after = md5(os.path.join(HERE, "w63a_setprice.json"))
    print(f"\n  w63a_setprice.json md5 {before} -> {after}   "
          f"{'✅ UNTOUCHED' if before == after else '⛔ OVERWRITTEN'}")
    assert before == after, "w63a_setprice.json was overwritten — every downstream pin is void"

    json.dump(dict(prereg="experiments/w79_prereg.txt @ 0109708",
                   adoption_cleared_here=bool(gating_ok), gating=list(GATING),
                   verdicts={k: bool(v[0]) for k, v in P.items()},
                   why={k: v[1] for k, v in P.items()},
                   stats=stats, bars=dict(P2=P2_BAR, P3=P3_BAR, P4=P4_BAR, P5_ref=P5_REF,
                                          P6_ref=P6_REF, P6=P6_BAR, P7=P7_BAR, P1_min=P1_MIN),
                   rule=dict(lo_dcv=LO, hi_dcv=HI, pick=PICK, n_oof=len(stems),
                             base=BASE, base_matches_artefact=bool(BASE == ref_base)),
                   fill=FILL, filla=FILLA, inside_dcv={k: dcv[k] for k in inside},
                   send_impact=dict(newly_blocked=newly, n_nonvetoed=len(nonveto),
                                    n_unsent_with_oof=unsent, n_gate_reachable=len(reachable),
                                    nearest_above=(above[0][1] if above else None),
                                    nearest_above_margin_e6=(above[0][0] if above else None)),
                   w63a_md5_before=before, w63a_md5_after=after),
              open(os.path.join(HERE, "w79a_verdicts.json"), "w"), indent=1, default=float)
    print("\nwrote w79a_verdicts.json / w79a_{control,barfill,fillall}.json")


if __name__ == "__main__":
    main()
