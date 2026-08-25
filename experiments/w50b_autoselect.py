"""w50b -- does the 08-23 calibration send hand auto-slot 1 to a raw member, and does
w49 Sec.3's "send it FIRST" actually prevent that?

NO PRE-REGISTRATION, and that is deliberate rather than an omission: this is a
DETERMINISTIC ENUMERATION of Kaggle's auto-selection rule over the live public scores and
a registered predicted band. There is no fitted quantity that could be steered, so a
prereg would be theatre. The one judgement call -- how to price an object whose CV is the
very thing under suspicion -- is stated as an explicit two-sided bound, not chosen.

THE CLAIM UNDER TEST (w49 journal Sec.3, carried into RESEARCH):

  "A cal file whose prediction is ABOVE the account best is sent FIRST, not last. Nothing
   is selected, so Kaggle auto-selects on best public score, and w46b Sec.5's latest-first
   tiebreak means the last file sent wins a public tie. Sent first, it loses every tie to
   the stacks behind it."

Send order is a TIEBREAK-ONLY lever. It can only bind when the cal file's public score
EQUALS a stack's. w48d's registered band for that file is [0.97120, 0.97125] and the
account best is 0.97118, so across the ENTIRE registered band the cal file is STRICTLY
ABOVE every stack and no tiebreak is ever consulted. This script measures what fraction
of the registered band the mitigation actually covers.

    .venv/bin/python experiments/w50b_autoselect.py
"""
from __future__ import annotations

import io, json, os, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMP = "playground-series-s6e8"
GRID = 1e-5                       # the public board's reporting step

# --- registered constants, quoted from their sources, not re-derived --------------------
CAL_BAND = (0.97120, 0.97125)     # w48d_arm217.json, the registered 08-23 prediction
CAL_HONEST = 0.97116              # >= this => ARM 217's gain is real
CAL_INFLATED = 0.97080            # <= this => imported artefact, drop hboyang_mix
CAL_OOF = 0.9701815536            # w49a, verified to <5e-10 against w48d
WANTED = ["w36_ad199stdcorr", "w23_ad187stdcorr"]
PICK_CV = 0.9701400060            # w36_ad199stdcorr, the WANTED first choice
# w45a: the live tie is exactly level on public, so the CV deficit passes through nearly
# intact -- cost ~= 1.09 * |dCV| (gamma = -0.0918). Quoted, not re-fitted.
GAMMA_PASSTHROUGH = 1.09


def live_submissions() -> pd.DataFrame:
    """page_size 200 -- RESEARCH's w17 pagination warning: the default 50 silently truncates."""
    out = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "500"],
        capture_output=True, text=True, timeout=300).stdout
    df = pd.read_csv(io.StringIO(out))
    df = df[df.publicScore.notna()].copy()
    df["stem"] = df.fileName.str.replace(".csv", "", regex=False)
    df["date"] = pd.to_datetime(df.date)
    return df.sort_values("date")


def auto_select(df: pd.DataFrame, limit: int, tiebreak: str) -> list[str]:
    """Kaggle's documented default: best public score, up to `limit`. The tiebreak among
    equal public scores is UNDOCUMENTED -- w45a brackets it, so we enumerate it too."""
    asc = tiebreak == "earliest"
    d = df.sort_values(["publicScore", "date"], ascending=[False, asc])
    return list(d.stem.head(limit))


def main() -> None:
    df = live_submissions()
    print("=" * 80)
    print("w50b  AUTO-SELECTION UNDER THE 08-23 CALIBRATION SEND")
    print("=" * 80)
    print(f"\n{len(df)} scored submissions on the live board "
          f"(page_size 200; the default 50 would have truncated this).")

    best = df.publicScore.max()
    tier1 = df[df.publicScore == best]
    print(f"\nlive best public {best:.5f}, held by {len(tier1)} files:")
    for _, r in tier1.sort_values("date").iterrows():
        mark = "  <- WANTED #1" if r.stem in WANTED else ""
        print(f"   {r.date:%Y-%m-%d %H:%M:%S}  {r.stem}{mark}")
    newest = tier1.sort_values("date").iloc[-1].stem
    print(f"\n  newest of the tie: {newest}"
          f"{'  == WANTED #1, so latest-first gives us the pick FREE' if newest in WANTED else ''}")

    print("\n" + "-" * 80)
    print("BASELINE -- auto-selection as the board stands TODAY (nothing selected)")
    print("-" * 80)
    for limit in (1, 2):
        for tb in ("latest", "earliest"):
            sel = auto_select(df, limit, tb)
            hit = sum(s in WANTED for s in sel)
            print(f"  limit {limit}  {tb:>8}-first -> {sel}   WANTED captured: {hit}/{len(WANTED)}")

    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("THE MITIGATION AUDIT -- what fraction of the registered band does send order cover?")
    print("=" * 80)
    lo, hi = CAL_BAND
    steps = [round(lo + k * GRID, 5) for k in range(int(round((hi - lo) / GRID)) + 1)]
    print(f"\nw48d's registered band [{lo:.5f}, {hi:.5f}] is {len(steps)} reporting steps: {steps}")
    print(f"account best public {best:.5f}\n")
    binds = [s for s in steps if s == best]
    print(f"  steps where the cal file TIES a stack (send order can bind):  {len(binds)}/{len(steps)}")
    print(f"  steps where it is STRICTLY ABOVE (order irrelevant):          "
          f"{sum(s > best for s in steps)}/{len(steps)}")
    print(f"  steps where it is STRICTLY BELOW (cannot be selected):        "
          f"{sum(s < best for s in steps)}/{len(steps)}")
    print("\n  >> w49 Sec.3's 'send it FIRST' mitigation covers "
          f"{len(binds)}/{len(steps)} of the registered band.")
    if not binds:
        print("  >> IT IS A NO-OP ACROSS THE WHOLE REGISTERED BAND. Sending first neither")
        print("     helps nor hurts; the file wins auto-slot 1 outright on every step of the")
        print("     band w48d actually predicted. The protection recorded in RESEARCH does")
        print("     not exist. (Sending first is still harmless, and it still buys w37c's R5")
        print("     guarantee that the experiment happens if the day is cut short.)")

    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BRANCH ENUMERATION -- what actually gets auto-selected on 08-23")
    print("=" * 80)
    rows = []
    for lb in [CAL_INFLATED, 0.97100, best, 0.97120, 0.97123, 0.97125]:
        for order in ("first", "last"):
            # sent first => earliest timestamp of the day; last => latest. Only the ordering
            # of the cal row against the day's stacks matters here.
            ts = pd.Timestamp("2026-08-23 00:07:00") if order == "first" \
                else pd.Timestamp("2026-08-23 00:07:40")
            sim = pd.concat([df[["stem", "publicScore", "date"]],
                             pd.DataFrame([{"stem": "w48_cal_hboyang_mix",
                                            "publicScore": lb, "date": ts}])],
                            ignore_index=True)
            for limit in (1, 2):
                for tb in ("latest", "earliest"):
                    sel = auto_select(sim, limit, tb)
                    rows.append(dict(lb=lb, order=order, limit=limit, tiebreak=tb,
                                     cal_selected="w48_cal_hboyang_mix" in sel,
                                     wanted_captured=sum(s in WANTED for s in sel),
                                     selection=";".join(sel)))
    res = pd.DataFrame(rows)
    for lb in sorted(res.lb.unique()):
        sub = res[res.lb == lb]
        tag = ("INFLATED" if lb <= CAL_INFLATED else
               "HONEST" if lb >= CAL_HONEST else "in-between")
        print(f"\n  cal file scores {lb:.5f}  [{tag}]")
        same = sub.groupby(["limit", "tiebreak"]).selection.nunique().max()
        for (limit, tb), g in sub.groupby(["limit", "tiebreak"]):
            sels = g.set_index("order").selection.to_dict()
            note = "" if sels["first"] == sels["last"] else "   <- SEND ORDER MATTERS HERE"
            print(f"    limit {limit} {tb:>8}-first: {sels['first']}"
                  f"   WANTED {g.wanted_captured.iloc[0]}/2{note}")
        print(f"    send order changes the selection in this branch: "
              f"{'YES' if same > 1 else 'NO'}")

    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IS BEING AUTO-SELECTED ON A HIGH PUBLIC SCORE ACTUALLY BAD HERE?")
    print("=" * 80)
    print(f"""
The veto on this member rests on its OOF vector: {CAL_OOF:.10f}, +708e-6 clear of the best
of the other 176 members on a like-for-like ranked comparison (w49c). That is a TRAIN-SIDE
object, and it is the reason ARM 217's CROSS-FITTED CV cannot be trusted.

Its TEST vector is a different object. Kaggle holds the test labels; nothing an author does
in a public notebook can contaminate a public-LB score of their own test predictions. So a
public read of {CAL_BAND[0]:.5f}-{CAL_BAND[1]:.5f} is an HONEST out-of-sample measurement of
exactly the vector that would be auto-selected -- on a 20% slice, with the usual slice noise,
but not contaminated in the way the OOF is.

  => The two failure modes do not overlap. The OOF can be worthless while the test vector is
     genuinely strong, and auto-selection reads only the latter.

  => At limit 2 (Kaggle's documented default) the HONEST branch selects
     {{cal file, {WANTED[0]}}} -- the best public object AND our CV pick. That is a
     BETTER pair than today's, not a worse one, and it still captures WANTED #1.

  => The exposure is confined to limit 1, where the cal file would be selected ALONE and
     we would hold no stack at all. w45a already prices limit 1 as the adverse branch of an
     undocumented rule.
""")
    print(f"  The residual limit-1 question is public->private shrinkage, and it is already")
    print(f"  measured: beta = -0.2477 (w17d), so a +{(0.97123-best)*1e5:.0f}e-5 public lead is")
    print(f"  expected to carry ~{(0.97123-best)*(1-0.2477)*1e5:.2f}e-5 onto private -- still")
    print(f"  positive. Auto-selecting a genuinely-higher-scoring file is not the Rogii")
    print(f"  failure; Rogii was selecting a file that scored higher on public and WORSE on CV")
    print(f"  than an available alternative, which is the {WANTED[0]} case at "
          f"{GAMMA_PASSTHROUGH:.2f} x |dCV|, not this one.")

    out = os.path.join(HERE, "w50b_autoselect.csv")
    res.to_csv(out, index=False)
    js = dict(n_scored=int(len(df)), best_public=float(best),
              tier1=list(tier1.sort_values("date").stem),
              newest_of_tier1=newest, newest_is_wanted=bool(newest in WANTED),
              band=list(CAL_BAND), band_steps=steps,
              steps_where_send_order_binds=len(binds),
              mitigation_is_noop=bool(not binds))
    with open(os.path.join(HERE, "w50b_autoselect.json"), "w") as fh:
        json.dump(js, fh, indent=2)
    print(f"\nwrote {out} and w50b_autoselect.json")


if __name__ == "__main__":
    main()
