"""w93b — the CV-to-LB gap over EVERY file this account has sent, read live, read-only.

WHAT THIS ANSWERS. The brief's core discipline is "write the CV score alongside the LB score
for every experiment; the CV-to-LB gap is itself the most useful number you will collect."
The workspace has that number in three places and none of them is the whole history today:

  experiments/w25a_cvlb_full.csv  91 rows, LAST WRITTEN 2026-08-22 -- and it MUST NOT be
                                  refreshed casually: `w57c_muguard`/`w75b_muguard` pin the
                                  LIVE pricer's centring against this file's `cv>=0.97` mean,
                                  so rewriting it re-centres the pricer as a side effect
                                  (w92 §2). This module therefore READS NOTHING FROM IT and
                                  WRITES NOTHING TO IT.
  w82a_pricecal                   calibration of the pricer's own published predictions.
  w75a_erarefresh                 the era term, ad>=195 files only.

None of those is "every experiment, today". This is. It fetches the live submission list,
parses the CV each file claimed AT SEND TIME out of its own immutable description, and
reports the gap distribution, the correlation, and where the two SELECTION candidates sit on
both axes. Its output file is its own; nothing else reads it.

⛔ READ-ONLY BY CONSTRUCTION. It writes exactly one artefact, `w93b_cvlbaudit.json`, and no
   other module consumes it. Do not make it write `w25a_cvlb_full.csv`.
⚠ CV here is the number in the DESCRIPTION, which is what past-you claimed when sending. That
   is the honest source for an audit: a later run cannot edit it. G3 cross-checks it against
   the on-disk ledger where both carry a file, which is w84a's G5 idiom.

CONTROLS
  G1  the fetch is paginated and strictly under its own cap (w86a's lesson: the 50-row
      default silently cost w82a 18 rows and two days).
  G2  every row that scored carries a public score in the plausible band for this board.
  G3  description CV agrees with the on-disk ledger where both carry a file.
  G4  CONTROL-: the reported Pearson r is computed on COMPLETE CASES and the dropped count is
      stated. w92 §5 found `w15j_cvlb` printing `pearson=+nan` beside an n that was wrong for
      both statistics it labelled; this asserts n_used == len(complete) and prints both.

    .venv/bin/python experiments/w93b_cvlbaudit.py        # 0 = ok, 1 = a control failed
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w84a_pickargmax import PAGE, SUBS_ARGV, CV_RE, SLOT1, SLOT2, fetch_raw, parse  # noqa: E402
from w91a_subdate import parse_sub_dates                                            # noqa: E402
import kaggle_list                                                                  # noqa: E402

OUT = os.path.join(HERE, "w93b_cvlbaudit.json")
LEDGER = os.path.join(HERE, "w57a_tierprice2.json")
# Not every sent file is an ATTEMPT. The `w37_cal_*` family are deliberate single-member
# MEASUREMENT probes -- their own descriptions say "this is a MEASUREMENT, not an attempt on
# the board" -- and they land 2-9e-3 below the blend band by design. Classifying them by
# SCORE would be the stdflag mistake (a number is not provenance), so they are classified by
# what their description says they are.
# ⚠⚠ THERE ARE **TWO** DECLARATION TEMPLATES, AND UNTIL w112 ONLY ONE WAS WIRED IN (2026-08-29).
# The principle above is right and was already written down; the implementation carried a single
# literal. `w26g_send.py` emits the `w37`-era probe wording AND, since w55, a `tail-fill` wording
# that declares exactly the same thing in different words -- and declares something STRONGER,
# because it names a `w55a` bound below the auto-selection tier. On 2026-08-29 the day's ten were
# the first send to put four tail-fill files 3.4-9.5e-3 below the band, and G2 called them
# ATTEMPTS. 🎯 A CLASSIFIER THAT READS A DECLARATION MUST KNOW EVERY WORDING THAT DECLARES IT --
# the bug was one missing string, in a check whose docstring had the policy exactly right.
PROBE_MARKS = ("not an attempt on the board",      # w37-era single-member probes
               "a MEASUREMENT, not a candidate")   # w55 tail-fill; carries a certified bound
AUC_BAND = (0.5, 1.0)               # any valid probability AUC
BLEND_BAND = (0.9700, 0.9720)       # where every real attempt on this board has landed
# ⛔ NOT AN EXEMPTION. A declared measurement is excused the blend band and then held to the
# property that actually matters: it must have landed BELOW the auto-selection tier, because a
# measurement ABOVE the tier is a file Kaggle could auto-select while nothing is selected. The
# tier is read live from `w55a_unpriced.json` -- the same artefact the sender certifies against.
UNPRICED = os.path.join(HERE, "w55a_unpriced.json")


def main() -> int:
    bad: list[str] = []
    df = parse(fetch_raw(SUBS_ARGV))
    df["date"] = parse_sub_dates(df["date"])

    # ⚠ w140: this used to read `len(df) >= PAGE -> truncated`. The fetch now paginates
    # (w84a -> kaggle_list), so the row count legitimately EXCEEDS the server's page cap and
    # the old test inverted: at 201 rows against a 200-row cap it called a COMPLETE read
    # truncated. Completeness is enforced at the source -- kaggle_list follows
    # next_page_token and raises rather than returning a partial list -- so what is worth
    # asserting here is that the read is not bounded by a page size at all.
    if len(df) <= kaggle_list.PAGE_CAP:
        print(f"⚠ G1 {len(df)} submissions, at or under the server page cap "
              f"{kaggle_list.PAGE_CAP} -- pagination is untested this run, not passing")
    else:
        print(f"✅ G1 {len(df)} submissions fetched, past the server page cap "
              f"{kaggle_list.PAGE_CAP}; the read is paginated, not capped")

    df["cv"] = df["description"].astype(str).str.extract(CV_RE)[0].astype(float)
    df["lb"] = pd.to_numeric(df["publicScore"], errors="coerce")

    desc = df["description"].astype(str)
    df["probe"] = False
    for mark in PROBE_MARKS:
        df["probe"] |= desc.str.contains(mark, regex=False)
    scored = df.dropna(subset=["lb"])
    off_auc = scored[(scored.lb <= AUC_BAND[0]) | (scored.lb > AUC_BAND[1])]
    attempts = scored[~scored.probe]
    off_band = attempts[(attempts.lb < BLEND_BAND[0]) | (attempts.lb > BLEND_BAND[1])]
    if len(off_auc):
        bad.append(f"G2 {len(off_auc)} scored file(s) are not a valid AUC: "
                   f"{off_auc[['fileName', 'lb']].head().to_dict('records')}")
    elif len(off_band):
        bad.append(f"G2 {len(off_band)} ATTEMPT(s) outside the blend band {BLEND_BAND}: "
                   f"{off_band[['fileName', 'lb']].head().to_dict('records')}")
    else:
        print(f"✅ G2 {len(scored)} scored: {len(attempts)} attempts all inside "
              f"{BLEND_BAND}, {int(scored.probe.sum())} declared measurement probes "
              f"({scored[scored.probe].lb.min():.5f}..{scored[scored.probe].lb.max():.5f}) "
              f"excluded by their OWN description; {len(df) - len(scored)} unscored")

    # ---- G2b: what the exemption is replaced by, not what it lets through -----------------
    probes = scored[scored.probe]
    if not os.path.exists(UNPRICED):
        bad.append(f"G2b {os.path.basename(UNPRICED)} is missing -- the tier is unknown, so the "
                   f"{len(probes)} declared measurement(s) cannot be held to anything")
    elif len(probes):
        tier = json.load(open(UNPRICED))["tier"]
        over = probes[probes.lb >= tier]
        if len(over):
            bad.append(f"G2b {len(over)} DECLARED MEASUREMENT(s) scored at or above the "
                       f"{tier} auto-selection tier -- they are auto-selectable while nothing "
                       f"is selected: {over[['fileName', 'lb']].to_dict('records')}")
        else:
            print(f"✅ G2b all {len(probes)} declared measurement(s) landed below the {tier} "
                  f"auto-selection tier; closest is {probes.lb.max():.5f}, margin "
                  f"{(tier - probes.lb.max()) * 1e6:+.1f}e-6")

    # ---- G3: the description CV vs the on-disk ledger, where both carry a file ------------
    n_agree = n_dis = 0
    if os.path.exists(LEDGER):
        lcv = {str(k).replace(".csv", ""): float(v)
               for k, v in json.load(open(LEDGER))["cv"].items()}
        for _, r in df.dropna(subset=["cv"]).iterrows():
            k = str(r["fileName"]).replace(".csv", "")
            if k in lcv:
                if abs(lcv[k] - r["cv"]) < 5e-9:
                    n_agree += 1
                else:
                    n_dis += 1
    if n_dis:
        bad.append(f"G3 {n_dis} file(s) where the sent description and the ledger disagree on CV")
    elif n_agree < 10:
        # w92 §5: a green check that compared nothing is worse than a red one.
        bad.append(f"G3 VACUOUS -- only {n_agree} overlapping files; the ledger key format "
                   f"has drifted and this check is comparing almost nothing")
    else:
        print(f"✅ G3 description CV == ledger CV on {n_agree} overlapping files, 0 disagreements")

    # ---- the audit itself: complete cases only, drops stated (G4) -------------------------
    both = df[~df.probe].dropna(subset=["cv", "lb"])
    dropped_cv = int(len(attempts) - len(both))
    per_file = (both.groupby("fileName", as_index=False)
                    .agg(cv=("cv", "max"), lb=("lb", "max"), n=("ref", "size"),
                         first=("date", "min")))
    per_file["gap"] = per_file["lb"] - per_file["cv"]

    r_p = float(per_file["cv"].corr(per_file["lb"], method="pearson"))
    r_s = float(per_file["cv"].corr(per_file["lb"], method="spearman"))
    n_used = int(per_file[["cv", "lb"]].dropna().shape[0])
    if n_used != len(per_file):
        bad.append(f"G4 complete-case count {n_used} != frame length {len(per_file)}")
    elif not np.isfinite(r_p):
        bad.append("G4 pearson is not finite on complete cases")
    else:
        print(f"✅ G4 statistics on {n_used} complete cases "
              f"({dropped_cv} scored rows carried no parseable CV and were dropped)")

    g = per_file["gap"]
    print(f"\nCV -> LB, {len(per_file)} distinct files, {int(per_file['n'].sum())} sends")
    print(f"  gap = LB - CV     mean {g.mean()*1e6:+9.1f}e-6   sd {g.std()*1e6:8.1f}e-6")
    print(f"                    min  {g.min()*1e6:+9.1f}e-6   max {g.max()*1e6:+9.1f}e-6")
    print(f"  pearson(cv,lb)  {r_p:+.4f}      spearman  {r_s:+.4f}")
    print(f"  CV  span {per_file.cv.min():.10f} .. {per_file.cv.max():.10f} "
          f"({(per_file.cv.max()-per_file.cv.min())*1e6:.1f}e-6)")
    print(f"  LB  span {per_file.lb.min():.5f} .. {per_file.lb.max():.5f} "
          f"({(per_file.lb.max()-per_file.lb.min())*1e6:.1f}e-6)")

    # LB reporting granularity -- how many distinct public values the whole history occupies
    vals = sorted(per_file.lb.unique())
    print(f"  the public board reports {len(vals)} DISTINCT values over this history: "
          f"{', '.join(f'{v:.5f}' for v in vals[-6:])} (top 6)")

    print("\ntop 8 by CV                                   CV            LB      gap(e-6)")
    for _, r in per_file.sort_values("cv", ascending=False).head(8).iterrows():
        print(f"  {r.fileName:36s} {r.cv:.10f}  {r.lb:.5f}  {r.gap*1e6:+8.1f}")
    print("top 8 by LB                                   CV            LB      gap(e-6)")
    for _, r in per_file.sort_values(["lb", "cv"], ascending=False).head(8).iterrows():
        print(f"  {r.fileName:36s} {r.cv:.10f}  {r.lb:.5f}  {r.gap*1e6:+8.1f}")

    print("\nTHE TWO SELECTION CANDIDATES ON BOTH AXES")
    out_slots = {}
    for tag, fn in (("slot1", SLOT1), ("slot2", SLOT2)):
        row = per_file[per_file.fileName == fn]
        if row.empty:
            bad.append(f"{tag} {fn} is not in the sent history")
            continue
        r = row.iloc[0]
        cv_rank = int((per_file.cv > r.cv).sum()) + 1
        lb_rank = int((per_file.lb > r.lb).sum()) + 1
        tie = int((per_file.lb == r.lb).sum())
        print(f"  {tag} {fn:32s} CV {r.cv:.10f} rank {cv_rank:3d}/{len(per_file)}   "
              f"LB {r.lb:.5f} rank {lb_rank:3d} (tied with {tie-1})")
        out_slots[tag] = dict(file=fn, cv=float(r.cv), lb=float(r.lb),
                              cv_rank=cv_rank, lb_rank=lb_rank, lb_tie=tie)

    # What the public board would auto-select, and what CV prefers. The whole brief in 4 lines.
    auto = per_file.sort_values(["lb", "cv"], ascending=False).head(2).fileName.tolist()
    cvpick = per_file.sort_values("cv", ascending=False).head(2).fileName.tolist()
    print(f"\n  public-argmax pair (what auto-selection takes) : {auto}")
    print(f"  CV-argmax pair                                 : {cvpick}")
    print(f"  SELECT_THESE.md names                          : [{SLOT1!r}, {SLOT2!r}]")

    json.dump(dict(n_files=len(per_file), n_sends=int(per_file["n"].sum()),
                   gap_mean=float(g.mean()), gap_sd=float(g.std()),
                   pearson=r_p, spearman=r_s, n_used=n_used, dropped_no_cv=dropped_cv,
                   slots=out_slots, auto_pair=auto, cv_pair=cvpick,
                   note="read-only audit; writes nothing that any other module reads"),
              open(OUT, "w"), indent=1)

    print("\nFAILURES: " + (" | ".join(bad) if bad else "0"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
