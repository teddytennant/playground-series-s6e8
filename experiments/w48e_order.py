"""w48e -- write the REGISTERED 08-22 ten into the artefact the sender actually reads.

⛔⛔ THE DEFECT THIS FIXES, found by dry-running the whole chain rather than reading it.

`w26g_send.py` does not read JOURNAL.md, RESEARCH.md, w47b_prereg.txt or w47b_probe.json. It
reads ONE file, `experiments/w26d_queueprice.csv`, and sends its priority-1 band in send_rank
order. That file was last written on 2026-08-20 14:30 by `w37d_order.py`, which encoded the
08-21 day. Nothing has written the 08-22 plan into it. So at 02:00 UTC on 08-21 the sender's
dry run planned this ten:

     1 w37_cal_ravi_realmlp1c   2 w37_cal_dkv_xgb        3 w36_ad199std_h3
     4 w36_ad199std_logit  ⛔VETOED                      5 w29_ad194stdcorr_ens4  ⛔VETOED
     6 w34_ad196std_logit       7 w29_ad194stdcorr_rescale ⛔VETOED
     8 w34_ad196std             9 w34_ad195std          10 w34_ad195std_rescale

THREE VETOED FILES, and not one of the five probes that w47b registered to settle ERA vs
CV-REGION. A run that opened the window and typed the documented command would have breached
the veto and destroyed the day's experiment in the same keystroke.

TWO ROOT CAUSES, both fixed here rather than described:

  1. THE VETO EXISTED ONLY IN PROSE. It is quoted in RESEARCH.md and re-asserted in four
     journal entries, and it appears in NO executable file. `grep -rn veto experiments/*.py`
     returns w46a's internal variable and nothing else. A rule that only a human re-reads is
     not a rule the tooling can honour. `VETO` below is a hard assert.
  2. THE PRICES WERE w30b's. The carried CSV predates w46c, so every ad>=195 row in it was
     30e-6 high. Repriced here through `w26d_queueprice.predict(..., stem=...)`.

⚠ AND ONE MORE FILE JOINS THE VETO, on this run's evidence. `w42_ad217std_logit` did not
exist when the veto was written. Under the corrected pricer it is now the HIGHEST predicted
LB in the entire 84-file queue at 0.97131 with P(beat account best) = 1.00, on a cross-fitted
CV 44e-6 BELOW the leader -- it collects `fam[logit]` +147.14e-6 on top of an ad217 base whose
CV w48d shows is inflated by an imported member (`hboyang_mix`, standalone OOF AUC 0.9701816,
+886e-6 clear of the best of the other 176). That is precisely the shape the veto exists to
stop: a file that wins an auto-selection tier on public score while being a bad pick on CV.
All six ARM 217 files are vetoed while nothing is selected -- w42b_prereg made them
WANTED-ineligible before they were built, and a file that may not be chosen deliberately must
not be reachable accidentally either.

    .venv/bin/python experiments/w48e_order.py            # plan only, writes nothing
    .venv/bin/python experiments/w48e_order.py --write    # write the queue CSV
"""
from __future__ import annotations

import hashlib, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET                                  # noqa: E402
import stdflag                                                  # noqa: E402
from stdflag import family, is_std                              # noqa: E402
import w26d_queueprice as QP                                    # noqa: E402  (import is side-effect free)
import w46c_predlb as W46                                       # noqa: E402

N_TEST = 296302
DST = os.path.join(HERE, "w26d_queueprice.csv")

# ---------------------------------------------------------------------------- THE VETO, IN CODE
# Binding while `check_selection` reports nothing selected. Each entry carries the reason, so a
# later run can retire one on evidence instead of on forgetting.
VETO = {
    "w29_ad194stdcorr_ens4":  "w39 §4 / w46a: reaches a live auto-selection tier on public "
                              "score while sitting well below the CV leader.",
    "w29_ad194stdcorr_rescale": "same, `rescale` carries +34.6e-6 of family term.",
    "w36_ad199std_logit":     "`logit` carries fam +147.14e-6 -- the largest family term by "
                              "4.2x -- on a CV 82.9e-6 below the leader.",
    "w38_ad202std_logit":     "same construction on the 202 pack.",
    "w40_ad211std_logit":     "same construction on the 211 pack.",
    # added w48, 2026-08-21
    "w42_ad217std_logit":     "w48e: highest predicted LB in the queue (0.97131, P 1.00) on a "
                              "CV 44e-6 below the leader. Logit term on a contaminated base.",
    "w42_ad217stdcorr":       "w48d: ARM 217. CV inflated by hboyang_mix; WANTED-ineligible "
                              "by w42b_prereg. Not reachable by auto-selection either.",
    "w42_ad217std":           "w48d: ARM 217, as above.",
    "w42_ad217std_h3":        "w48d: ARM 217, as above.",
    "w42_ad217std_hybrid":    "w48d: ARM 217, as above.",
    "w42_ad217std_rankraw":   "w48d: ARM 217, as above.",
    "w42_ad217std_rescale":   "w48d: ARM 217, as above.",
    # added w51, 2026-08-21. ARM 216 = ARM 217 minus `hboyang_mix`, built by w50a_run.sh and
    # landed on disk 03:35 UTC. `blend_lab --build` emits the WHOLE transform family as a
    # side-effect, so seven ad216 files now sit in submissions/ where w23b_sendqueue globs
    # them, and NOTHING was stopping the queue writer from promoting them.
    #
    # ⛔ w50_prereg Sec.5, written before the build: "NOT WANTED-eligible, whatever d_five
    # turns out to be. Five of the six imported members still have unread es-on-val status
    # and dropping the sixth does not discharge that clause." d_five came back +12.73e-6
    # (R3 IN-BETWEEN), so the clause stands unchanged.
    #
    # ⚠ AND THE ad216 FAMILY IS EXACTLY THE SHAPE THIS VETO EXISTS FOR. `w50_ad216stdcorr`
    # is CV 0.9701500880 -- the HIGHEST CV ever built in this workspace, +10.1e-6 above the
    # WANTED file -- so anything that ranks the queue on CV puts it near the top, and
    # `w50_ad216std_logit` collects fam[logit] +147.14e-6 on top of that same base, so
    # anything that ranks on predicted LB puts it FIRST. Both orderings reach it. That is
    # precisely how `w42_ad217std_logit` got planned into a send list.
    "w50_ad216stdcorr":       "w51: ARM 216. WANTED-ineligible by w50_prereg Sec.5. Highest "
                              "CV on disk (0.9701500880) -- unreachable by accident until "
                              "the es-on-val clause is discharged on evidence.",
    "w50_ad216std":           "w51: ARM 216, as above.",
    "w50_ad216std_h3":        "w51: ARM 216, as above.",
    "w50_ad216std_hybrid":    "w51: ARM 216, as above.",
    "w50_ad216std_rankraw":   "w51: ARM 216, as above.",
    "w50_ad216std_rescale":   "w51: ARM 216, as above.",
    "w50_ad216std_logit":     "w51: ARM 216 AND the logit family term on the highest base CV "
                              "on disk. The single most dangerous file in submissions/.",
}

# ------------------------------------------------------------------- THE REGISTERED DAYS
# ⚠ w48 found the sender executing a THIRTY-SIX HOUR STALE list. A single hardcoded ORDER is
# how that happened: nothing tied the list to the day it was written for. ORDERS is keyed by
# the UTC send date and the writer refuses to run for a day that is not registered, so a list
# can never be executed on the wrong day and a missing list fails loudly instead of silently
# re-sending yesterday's.
REG = json.load(open(os.path.join(HERE, "w47b_probe.json")))

# ---- 08-23: the ARM 217 test, plus the nine highest-CV files left in the queue ----------
# Slot 1 is `w48_cal_hboyang_mix`, registered in w48d and built by w49a. Slots 2-10 are the
# nine best remaining by CV, which INCLUDES all five files w47b dropped from the 08-22 ten
# ("they go back in the queue for 08-23 or later, not away" -- honoured here).
#
# ⚠ WHY THE CALIBRATION FILE GOES FIRST AND NOT LAST. Its registered prediction is 0.97123,
# ABOVE this account's 0.97118 best, and NOTHING IS SELECTED, so Kaggle would auto-select on
# best public score. Under w46b §5's latest-first tiebreak the LAST file sent wins a public
# tie -- so sending a raw single-member vector last would hand it every tie against our own
# stacks. Sent FIRST, it loses every tie to the nine stacks that follow it. Sending it first
# also guarantees the experiment happens if the day is cut short (w37c R5's rule).
ORDER_0823 = [
    "w48_cal_hboyang_mix",
    "w38_ad202std_rescale", "w34_ad195std_h3", "w40_ad211std_rescale", "w36_ad197std",
    "w38_ad202std_hybrid", "w36_ad197std_h3", "w36_ad197stdcorr", "w38_ad202std_h3",
    "w40_ad211std_h3",
]
WHY_0823 = {
    "w48_cal_hboyang_mix":  "THE ARM 217 TEST, registered in w48d_arm217.json before this file "
                            "existed. Raw test vector of imported member `hboyang_mix`, whose "
                            "standalone OOF AUC on our frozen folds is 0.9701816 -- +886e-6 "
                            "clear of the best of the other 176 members and ABOVE our whole "
                            "217-member stack. Predicted LB 0.97123, band [0.97120, 0.97125]. "
                            ">=0.97116 HONEST; <=0.97080 INFLATED, drop the member. Priced off "
                            "the 34 NEAREST anchors, not the w36f line, because w37e's R2 "
                            "killed the linear form at 6.3σ. NOT a deadline candidate.",
    "w38_ad202std_rescale": "DRAIN. One of the five w47b dropped from the 08-22 ten; back in "
                            "the queue as promised.",
    "w34_ad195std_h3":      "DRAIN. ad195, h3.",
    "w40_ad211std_rescale": "DRAIN. One of w47b's five dropped files.",
    "w36_ad197std":         "DRAIN. ad197, ens4.",
    "w38_ad202std_hybrid":  "DRAIN. ad202, hybrid.",
    "w36_ad197std_h3":      "DRAIN. ad197, h3.",
    "w36_ad197stdcorr":     "DRAIN. One of w47b's five dropped files; corrected h3 on ad197.",
    "w38_ad202std_h3":      "DRAIN. One of w47b's five dropped files.",
    "w40_ad211std_h3":      "DRAIN, best CV of the nine and therefore LAST -- latest-first "
                            "tiebreak (w46b §5). One of w47b's five dropped files.",
}

WHY_0822 = {
    "w40_ad211std_rankraw": "PROBE 1/5. ad211, INSIDE the fitted CV support. ERA predicts it "
                            "lands ~-29.8e-6 under w30b; CV-REGION predicts ~0. Screened free "
                            "under the GENEROUS uncorrected w30b so the test is not circular.",
    "w38_ad202std_rankraw": "PROBE 2/5. Same test on the 202 pack, different transform.",
    "w34_ad196std_hybrid":  "PROBE 3/5. ad196 -- the earliest era pack that has an in-support "
                            "file, so the probe set is not carried by one wave.",
    "w34_ad195std_hybrid":  "PROBE 4/5. ad195, the first pack of the era.",
    "w36_ad197std_hybrid":  "PROBE 5/5 and the cleanest single contrast in the design (w47 §6): "
                            "ad197/hybrid/std INSIDE support, against w36_ad199std_hybrid "
                            "(ad199/hybrid/std, ABOVE support) which already scored 0.97114, "
                            "residual -32.5e-6. If that one pair splits, the answer is "
                            "CV-REGION and no averaging is needed to see it.",
    "w36_ad199std_h3":      "DRAIN. Highest-CV unsent non-vetoed file (0.9701354276).",
    "w38_ad202std":         "DRAIN. 202 pack, ens4 family.",
    "w40_ad211std":         "DRAIN. 211 pack, ens4 family.",
    "w40_ad211stdcorr":     "DRAIN. Corrected h3 on the 211 pack.",
    "w38_ad202stdcorr":     "DRAIN, best CV of the ten and therefore LAST -- latest-first "
                            "tiebreak (w46b §5): if it ties on public score with an earlier "
                            "file, the later timestamp wins the tier, free.",
}

ORDERS = {"2026-08-22": list(REG["order"]), "2026-08-23": ORDER_0823}
WHYS   = {"2026-08-22": WHY_0822,           "2026-08-23": WHY_0823}

# calibration files have no entry in w23b_sendqueue (no stack CV to rank on) and must be
# injected, exactly as w37c_prereg.py did for the w37 batch. stem -> (builder json, note).
CAL_ROWS = {"w48_cal_hboyang_mix": "w49a_calhboyang.json"}

DAY = None
for i, a in enumerate(sys.argv):
    if a == "--day":
        DAY = sys.argv[i + 1]
if DAY is None:
    from datetime import datetime, timezone
    DAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("=" * 90)
print(f"w48e  THE REGISTERED TEN FOR {DAY} (UTC) -> the artefact w26g_send.py actually reads")
print("=" * 90)

if DAY not in ORDERS:
    print(f"\n  ⛔ NO LIST IS REGISTERED FOR {DAY}. Registered days: "
          f"{', '.join(sorted(ORDERS))}.")
    print("     Refusing to write. Add the day to ORDERS/WHYS -- do NOT re-run another day's "
          "list,\n     which is exactly the w48 defect this keying exists to prevent.")
    sys.exit(2)
ORDER, WHY = ORDERS[DAY], WHYS[DAY]

bad = sorted(set(ORDER) & set(VETO))
assert not bad, f"⛔ the registered order contains VETOED files: {bad}"
assert len(ORDER) == 10 and len(set(ORDER)) == 10, "the registered order is not ten distinct files"
print(f"\n  veto check: {len(VETO)} files vetoed, none of them in the registered ten. OK")

# ------------------------------------------------------------------ rebuild the queue, priced w46c
q = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
q["stem"] = q.file.str.replace(".csv", "", regex=False)
stdflag.require_corr_registered(q.stem)
# carry the rows w23b cannot regenerate (the w37 calibration files are registered by
# w37c_prereg.py and have no stored OOF vector, so w23b drops them from the ranked queue).
def _live_sent():
    """Filenames Kaggle has actually scored. Asked live -- a cached list is exactly how the
    queue came to describe already-sent files as candidates. Falls back to the cache only if
    the API call fails, and says so when it does."""
    import subprocess
    snip = r"""
import json, os
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.competitions.types.competition_enums import SubmissionGroup
tok = json.load(open(os.path.expanduser(
    os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle") + "/credentials.json")))["access_token"]
with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    r = ApiListSubmissionsRequest(); r.competition_name = "playground-series-s6e8"
    r.group = SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL; r.page_size = 500
    n = sorted({s.file_name for s in
                c.competitions.competition_api_client.list_submissions(r).submissions})
assert len(n) < 500, "page saturated"
print(json.dumps(n))
"""
    cache = os.path.join(HERE, "w48e_sent.json")
    try:
        out = subprocess.run(["/home/nixos/.local/share/uv/tools/kaggle/bin/python", "-c", snip],
                             capture_output=True, text=True, timeout=180)
        names = json.loads(out.stdout)
        json.dump(names, open(cache, "w"), indent=0)
        return {f[:-4] if f.endswith(".csv") else f for f in names}
    except Exception as e:
        if os.path.exists(cache):
            print(f"  ⚠ live submission list unavailable ({e!r}); falling back to the cache, "
                  f"which may be stale.")
            return {f[:-4] if f.endswith(".csv") else f
                    for f in json.load(open(cache))}
        print(f"  ⚠⚠ no live list and no cache ({e!r}) -- the sent filter is OFF this run.")
        return set()


LIVE_SENT = _live_sent()
if os.path.exists(DST):
    prev = pd.read_csv(DST)
    extra = prev[~prev.file.isin(q.file)][["file", "cv", "rows", "md5"]].copy()
    extra["stem"] = extra.file.str.replace(".csv", "", regex=False)
    # ⚠ drop anything the carry would resurrect that has since been SENT. The previous CSV is
    # a day and a half old; ten of its rows went out at 00:07 UTC today. w26g dedupes by
    # filename anyway, but a queue that lists sent files as candidates is how the WANTED pin
    # came to read "unsent" for a file that had already landed.
    if LIVE_SENT:
        n0 = len(extra); extra = extra[~extra.stem.isin(LIVE_SENT)]
        if n0 - len(extra):
            print(f"  dropped {n0-len(extra)} carried row(s) that are already scored on Kaggle")
    extra["sent"] = False
    if len(extra):
        print(f"  carried {len(extra)} row(s) w23b cannot regenerate: {', '.join(extra.stem)}")
        q = pd.concat([q, extra], ignore_index=True)
# inject registered calibration files (no stack CV, so w23b never lists them)
for stem, jf in CAL_ROWS.items():
    if stem in q.stem.values or stem in LIVE_SENT:
        continue
    jp = os.path.join(HERE, jf)
    if not os.path.exists(jp):
        print(f"  ⚠ {stem}: {jf} missing -- not injected. Run its builder first.")
        continue
    m = json.load(open(jp))
    q = pd.concat([q, pd.DataFrame([dict(file=f"{stem}.csv", sent=False, cv=m["oof_auc"],
                                         rows=m["rows"], md5=m["md5"], stem=stem)])],
                  ignore_index=True)
    print(f"  injected calibration row {stem} (standalone OOF AUC {m['oof_auc']:.10f})")

q = q.reset_index(drop=True)
q["fam"] = q.stem.map(family)
q["std"] = q.stem.map(is_std)
q["corr"] = q.stem.map(QP.is_corr)
# ⚠ rows with no stored OOF vector have no CV and therefore no price. They are NOT dropped --
# w37_cal_ravi_realmlp1c and w37_cal_dkv_xgb are legitimate unsent calibration files whose
# whole point is that they are not ranked on CV. They keep priority 0 and sort last.
_has = q.cv.notna()
q["pred_lb"] = np.nan
q.loc[_has, "pred_lb"] = [QP.predict(r.cv, r.fam, r.std, r.corr, r.stem)
                          for r in q[_has].itertuples()]
from scipy.stats import norm as _norm                            # noqa: E402
q["p_beat"] = np.nan
q.loc[_has, "p_beat"] = 1.0 - _norm.cdf(
    (QP.BEST_LB + QP.STEP / 2 - q.loc[_has, "pred_lb"])
    / np.array([QP.resid_sd(s) for s in q.loc[_has, "stem"]]))
# ⚠ A CALIBRATION FILE MUST NOT CARRY A STACK PRICE. QP.predict is the cv->LB line fitted on
# CROSS-FITTED STACKS; applied to a raw member vector it produced `pred 0.97129, P(beat) 1.00`
# for w48_cal_hboyang_mix -- i.e. it announced that a single imported member is CERTAIN to beat
# the account best. w37e's R2 killed the linear form off-range at 6.3σ precisely so this would
# not be done. Overwrite with the builder's own registered prediction, and blank p_beat: these
# files are measurements and must never sort or be promoted on a probability of beating anyone.
for stem, jf in CAL_ROWS.items():
    m = q.stem == stem
    if m.any() and os.path.exists(os.path.join(HERE, jf)):
        q.loc[m, "pred_lb"] = json.load(open(os.path.join(HERE, jf)))["pred_lb"]
        q.loc[m, "p_beat"] = np.nan
        q.loc[m, "fam"] = "member"

q["vetoed"] = q.stem.isin(VETO)
print(f"  {int((~_has).sum())} row(s) carry no CV and are left unranked: "
      f"{', '.join(q[~_has].stem)}")

print(f"\n  queue {len(q)} unsent files, priced under w46c "
      f"({QP.RESID:.2f}e-6 pre-era / {QP.RESID_ERA:.2f}e-6 for ad>={W46.ERA_MIN_AD})")
v = q[q.vetoed].sort_values("pred_lb", ascending=False)
print(f"\n  ⛔ VETOED and therefore NOT sendable, highest predicted LB first:")
for r in v.itertuples():
    print(f"     {r.stem:26s} cv {r.cv:.10f}  pred {r.pred_lb:.5f}  P(beat) {r.p_beat:.2e}")
print(f"     -- the top of this list is ABOVE the top of the plan below. That is the point.")

# ------------------------------------------------------------------ verify the ten, end to end
print("\n  --- verifying the ten against the artefacts on disk ---")
yv = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[TARGET])[TARGET].values
idx = q.set_index("stem")
ok = True
for i, stem in enumerate(ORDER, 1):
    f = os.path.join(SUB, f"{stem}.csv")
    prob = []
    if stem not in idx.index:
        prob.append("NOT IN QUEUE")
    if not os.path.exists(f):
        prob.append("no CSV")
    else:
        d = pd.read_csv(f)
        if len(d) != N_TEST:
            prob.append(f"rows {len(d)}")
        if d.isna().any().any():
            prob.append("NaN")
        m = hashlib.md5(open(f, "rb").read()).hexdigest()
        if stem in idx.index and isinstance(idx.loc[stem, "md5"], str) and idx.loc[stem, "md5"] != m:
            prob.append("md5 drift")
    o = os.path.join(SUB, f"oof_{stem}.npy")
    if not os.path.exists(o):
        prob.append("no OOF")
    elif stem in idx.index:
        cv = float(roc_auc_score(yv, np.load(o).ravel()))
        if abs(cv - idx.loc[stem, "cv"]) > 5e-10:
            prob.append(f"CV {cv:.10f} != {idx.loc[stem,'cv']:.10f}")
    r = idx.loc[stem] if stem in idx.index else None
    print(f"  {i:2d}. {stem:24s} cv {r.cv:.10f}  pred {r.pred_lb:.5f}  "
          f"{'OK' if not prob else '⛔ ' + '; '.join(prob)}")
    ok &= not prob
assert ok, "⛔ at least one registered file failed verification -- nothing written"
print("\n  all ten: 296,302 rows, no NaN, md5 matches the queue, CV reproduces from the OOF "
      "vector. OK")

# ------------------------------------------------------------------ write
for c in ("send_rank", "msg", "why"):
    q[c] = np.nan
q["why"] = q["why"].astype(object)
q["priority"] = 0
for i, stem in enumerate(ORDER, 1):
    m = q.stem == stem
    q.loc[m, "priority"] = 1
    q.loc[m, "send_rank"] = i
    q.loc[m, "why"] = WHY[stem]
# a vetoed file must never be reachable: push it behind everything the pricer would offer.
q.loc[q.vetoed, "priority"] = -1
q = q.sort_values(["priority", "send_rank", "pred_lb"], ascending=[False, True, False])

print(f"\n  {'#':>3} {'file':28s} {'cv':>13s} {'pred':>8s} {'P':>9s}  kind")
for r in q[q.priority == 1].itertuples():
    kind = WHY[r.stem].split(".")[0].split(",")[0].strip()[:34]
    pb = "     --  " if not np.isfinite(r.p_beat) else f"{r.p_beat:9.2e}"
    print(f"  {int(r.send_rank):3d} {r.file:28s} {r.cv:.10f} {r.pred_lb:8.5f} {pb}  {kind}")

# ⚠ STAMP THE DAY INTO THE ARTEFACT (w53). w49 keyed the WRITER to the day so a list can never
# be written for the wrong one. It did not key the READER. `w26g_send.py` reads this CSV with no
# idea which day it was written for, and w53 dry-ran the consequence: with the 08-22 CSV still on
# disk, all ten of its priority-1 rows are already sent, w26g skips them, falls through to the
# priority-0 tail and plans a DIFFERENT ten -- one that omits `w48_cal_hboyang_mix` (slot 1 of the
# registered 08-23 list, and the only live path to gold on this account) and includes two `logit`
# files, which is the precise auto-selection exposure the VETO above exists to close. The veto is
# NOT in this file's columns -- it is applied as priority -1 and the column is dropped -- so a
# stale CSV carries no veto at all. One constant column closes it; w26g refuses to --go unless
# this equals today's UTC date.
q["plan_day"] = DAY

if "--write" in sys.argv:
    q.drop(columns=["vetoed"]).to_csv(DST, index=False)
    print(f"\n  wrote {os.path.basename(DST)} -- {len(q[q.priority==1])} ranked, "
          f"{int((q.priority == -1).sum())} vetoed and pushed to the back.")
else:
    print("\n  PLAN ONLY. Nothing written. Re-run with --write.")
