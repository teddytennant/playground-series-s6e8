"""w79b — the standing guard for the adopted hole-filled hijack bar (w79).

Seven controls. File reads and one signature read; NO refit, NO network; runs in well under a
second. Pre-registration: experiments/w79_prereg.txt, committed 0109708.

C1  THE SENDER READS THE FILLED ARTEFACT, at the adopted bar, and the artefact says of itself
    that it is filled. Read through `w26g_send.HIJACKPRICE`, never by re-typing the path
    (w63 §9). This overlaps w77b C5 on purpose: w77b guards it from the w77 side and this from
    the w79 side, and a run that edits one and not the other gets caught by the other.

C2  ⚠⚠ THE ONE THAT MATTERS. `w63a_setprice.json` IS BYTE-UNTOUCHED. It is not the live bar any
    more, and that is exactly why it is fragile: nothing on the send path reads it, so a later
    run has no daily reason to notice if it moves — and w76a GATE A, w77a GATE A, w77b C1/C4,
    w77c, w77d GATE D and w78a GATE A ALL pin against it. Losing it silently voids six runs of
    work. md5, not a field comparison, because the failure mode is "somebody re-ran the pricer".

C3  THE FILLERS STILL SATISFY THE REGISTERED RULE, recomputed from the artefacts rather than
    trusted: every filler's dCV strictly inside w63a's own stored uncond bracket, none of them
    in the base design, and the count still 13.

C4  THE GLS WAS NOT ENLARGED. The whole construction is "the fillers enter the DESIGN and are
    withheld from the FIT" (w77d's observation, w78's 80x attenuation argument). If a later run
    lets them into `LB` the bar becomes the ARM-L pathology w77c documented. The filled arm's
    `gls_scored` must be identical to the control arm's and their `gap` must agree at 1e-9.

C5  THE BAR IS STILL IDENTIFIED. (Shares `stats.d_fill_filla` with C7 — see the selftest note.) FILL (scored files in the hole) and FILLA (every OOF file in
    the hole) must still price within the registered 2.0e-6. This is the control that stops the
    bar from being chosen by choosing the population, which is w59a's objection to the whole
    manoeuvre and the reason w79 registered a second arm at all.

C6  `w63a_setprice.main`'s DEFAULTS ARE STILL THE OLD BEHAVIOUR: `fill=()` and
    `outfile="w63a_setprice.json"`. w79 made that function callable with an enlarged design; if
    a later run flips a default, every existing caller silently starts pricing a different one.

C7  EVERY RECORDED VERDICT IS RECOMPUTED FROM w79a's OWN `stats` TABLE against the bars in
    `w79_prereg.txt`, and any disagreement fires. w78b C1's move: after three runs where a
    verdict on disk needed pinning by hand, verdict-checking is mechanical, not editorial.

    .venv/bin/python experiments/w79b_fillguard.py
    .venv/bin/python experiments/w79b_fillguard.py --selftest
"""
from __future__ import annotations

import hashlib, inspect, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE) + "/agent")
sys.path.insert(0, HERE)

ADOPTED_BAR = 0.9701349052
N_FILL = 13
W63A_MD5 = "4fd6820e59217fbe0dfa44bffb820057"
# the registered bars, re-typed here ONLY so a mutation of the artefact cannot also mutate the
# thing it is checked against. They are the same numbers as w79a_barfill.BARS, by construction.
BARS = dict(P2=1e-9, P3=1.0e-6, P4=5.0, P5_ref=0.9701288617, P6_ref=0.9701347511,
            P6=3.0e-6, P7=2.0e-6, P1_min=10)

FAILURES = []


def chk(tag: str, ok: bool, msg: str) -> None:
    if not ok:
        FAILURES.append(tag)
        print(f"  *** {tag} FAILED: {msg}")
    else:
        print(f"  {tag} ok")


def run(mut=None, md5_override=None) -> int:
    global FAILURES
    FAILURES = []
    mut = mut or {}

    def load(name):
        d = json.load(open(os.path.join(HERE, name)))
        for path, val in mut.get(name, []):
            t = d
            for k in path[:-1]:
                t = t[k]
            t[path[-1]] = val
        return d

    w63a = load("w63a_setprice.json")
    V = load("w79a_verdicts.json")
    ctrl = load("w79a_control.json")
    fill = load("w79a_fillall.json") and load("w79a_barfill.json")

    import w26g_send as SEND                                    # noqa: E402
    import w63a_setprice as W63A                                # noqa: E402
    sender_file = os.path.basename(SEND.HIJACKPRICE)
    read = load(sender_file) if sender_file == "w79a_barfill.json" else json.load(
        open(SEND.HIJACKPRICE))
    chk("C1", sender_file == "w79a_barfill.json"
        and abs(float(read["cv_bar_new"]) - ADOPTED_BAR) < 1e-9
        and read.get("hole_filled") is True and len(read.get("fill", [])) == N_FILL,
        f"the sender reads {sender_file} at {float(read.get('cv_bar_new', float('nan'))):.10f} "
        f"with hole_filled={read.get('hole_filled')} and {len(read.get('fill', []))} fillers; "
        f"expected w79a_barfill.json / {ADOPTED_BAR:.10f} / True / {N_FILL}.")

    got = md5_override or hashlib.md5(
        open(os.path.join(HERE, "w63a_setprice.json"), "rb").read()).hexdigest()
    chk("C2", got == W63A_MD5,
        f"w63a_setprice.json md5 is {got}, not {W63A_MD5}. Something re-ran the pricer onto the "
        f"superseded path. w76a/w77a/w77b/w77c/w77d/w78a all pin against that file and every one "
        f"of them is now testing a different artefact than the one it was written for. Recover "
        f"it with `git checkout -- experiments/w63a_setprice.json` before running anything else.")

    lo, hi = float(w63a["bracket"]["uncond"]["lo_dcv"]), float(w63a["bracket"]["uncond"]["hi_dcv"])
    dcv, base = V["inside_dcv"], set(V["rule"]["base"])
    # ⚠ a filler with NO recorded dCV is a rule violation, not a KeyError. The rule is a
    # predicate on dCV; a name the artefact cannot place inside the bracket was not selected by
    # it, whatever else is true of it.
    bad = [k for k in V["fill"]
           if k not in dcv or not (lo < float(dcv[k]) < hi) or k in base]
    chk("C3", not bad and len(V["fill"]) == N_FILL and V["rule"]["base_matches_artefact"],
        f"the filler population no longer satisfies the registered rule: {bad or 'count'} "
        f"({len(V['fill'])} fillers, bracket ({lo:+.4f}, {hi:+.4f}), base matches artefact "
        f"{V['rule']['base_matches_artefact']}).")

    chk("C4", sorted(fill["gls_scored"]) == sorted(ctrl["gls_scored"])
        and abs(float(fill["gap"]) - float(ctrl["gap"])) < BARS["P2"],
        f"the fillers reached the GLS: filled arm fits {len(fill['gls_scored'])} scored files "
        f"against the control's {len(ctrl['gls_scored'])}, gap {float(fill['gap']):+.6f} vs "
        f"{float(ctrl['gap']):+.6f}. That is the ARM-L pathology w77c documented, not this bar.")

    d57 = V["stats"]["d_fill_filla"]
    chk("C5", d57 is not None and float(d57) < BARS["P7"],
        f"FILL and FILLA now price {float(d57 or float('nan'))/1e-6:.3f}e-6 apart against the "
        f"registered {BARS['P7']/1e-6:.1f}e-6. The bar would be a choice of population, which is "
        f"w59a's objection, and adoption would no longer be defensible.")

    sig = inspect.signature(W63A.main).parameters
    chk("C6", sig["fill"].default == () and sig["outfile"].default == "w63a_setprice.json",
        f"w63a_setprice.main defaults moved to fill={sig['fill'].default!r}, "
        f"outfile={sig['outfile'].default!r}. Every existing caller now prices a design it did "
        f"not ask for, or writes over a path it did not name.")

    # ---- C7: recompute all seven verdicts from the artefact's own stats -----------------------
    s = V["stats"]
    want = {
        "P1": s["n_fill"] >= BARS["P1_min"],
        "P2": s["d_gap"] < BARS["P2"],
        "P3": s["d_base"] < BARS["P3"] and s["d_worthless"] < BARS["P3"],
        "P4": s["width_fill"] < BARS["P4"],
        "P5": s["bar_fill"] > BARS["P5_ref"],
        "P6": s["d_w77d"] < BARS["P6"],
        "P7": s["d_fill_filla"] is not None and s["d_fill_filla"] < BARS["P7"],
    }
    dis = [k for k, v in want.items() if bool(v) != bool(V["verdicts"][k])]
    chk("C7", not dis and all(want[k] for k in ("P2", "P3", "P4", "P5", "P7")),
        f"recorded verdicts disagree with the artefact's own stats for {dis}, or a GATING "
        f"prediction is no longer satisfied by the numbers on disk. The adoption rule in "
        f"w79_prereg.txt is P2 and P3 and P4 and P5 and P7 and P9; if one of those stops "
        f"holding, the bar the sender is using is not the bar that was adopted.")
    return len(FAILURES)


def main() -> None:
    if "--selftest" in sys.argv:
        print("SELFTEST — each planted defect must fire its OWN control and no other\n")
        # ⚠ THE EXPECTED SET, NOT A SINGLE TAG — and the one case where it has two members is
        # DECLARED, not discovered after the fact. C5 and C7 both read `stats.d_fill_filla`:
        # C5 asks whether FILL and FILLA still agree, C7 asks whether the RECORDED P7 verdict
        # matches that same number. Breaking the number breaks both, correctly. They are not
        # redundant — C5 survives a later run dropping P7 from `verdicts`, and C7 survives a
        # later run editing a verdict without touching the stat.
        cases = [
            (["C1"], {"w79a_barfill.json": [(["hole_filled"], False)]}, None),
            (["C2"], None, "0" * 32),
            (["C3"], {"w79a_verdicts.json": [(["fill"], ["w36_ad199stdcorr"])]}, None),
            (["C4"], {"w79a_barfill.json": [(["gap"], 999.0)]}, None),
            (["C5", "C7"], {"w79a_verdicts.json": [(["stats", "d_fill_filla"], 9e-6)]}, None),
            (["C7"], {"w79a_verdicts.json": [(["verdicts", "P4"], False)]}, None),
        ]
        bad = 0
        for want, mut, md5o in cases:
            run(mut, md5o)
            ok = sorted(FAILURES) == sorted(want)
            print(f"  planted {'+'.join(want):7s} -> fired {FAILURES or 'nothing'}  "
                  f"{'✅' if ok else '⛔ WRONG CONTROL'}\n")
            bad += (not ok)
        print("  ⚠ C6 has no planted case here: it reads a live function SIGNATURE, and the only "
              "honest\n    way to plant it is to edit w63a_setprice.py, which a selftest may not "
              "do. It is a\n    one-line read and it runs on every live invocation above.")
        print(f"\nselftest {'PASSED' if not bad else 'FAILED'} — six plantable defects, each "
              f"firing its declared control set.")
        sys.exit(1 if bad else 0)

    print("w79b — the hole-filled hijack bar guard")
    n = run()
    print(f"\nFAILURES {n}")
    if n:
        print("⛔ The adopted bar, the construction behind it, or the superseded artefact it was "
              "measured\n   against has moved. Read JOURNAL w79 and experiments/w79_prereg.txt.")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
