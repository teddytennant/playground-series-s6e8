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

# ---- 08-24: the drain, ten files, ZERO experiments ------------------------------------------
# ⚠ THE TEN ARE IMPORTED FROM THE PRICER, NEVER RE-TYPED. `w63a_setprice.PLAN_0824` is the set
# w63a priced as a DAY — its P6 ("P(any of the ten clears the tier) < 0.05", read 0.012199) is a
# statement about THIS list, and a second hand-typed copy here would let the registered list and
# the priced list drift apart without either file noticing. w62 §1 is what that costs.
from w63a_setprice import PLAN_0824 as ORDER_0824                          # noqa: E402

# ⚠ THREE OF THE TEN WERE BLOCKED YESTERDAY AND ARE UNBLOCKED TODAY, AND NOTHING ABOUT THEM
# CHANGED. `w40_ad211std_rescale`, `w38_ad202std_rescale` and `w27_ad188stdcorr` failed the
# sender's P_MAX = 0.02 hijack-risk gate on 08-23 at P(above) = 0.082 / 0.061 / 0.053. The two
# lever files then cleared and took auto-slot 1 from 0.97118 to 0.97119 — a full display step —
# and the same three files now read 0.006 / 0.004 / 0.003 against the higher threshold. The
# board moving UP made previously-unsendable files sendable. That is a consequence of w60's
# lever that nobody registered, and it is why they lead the list on CV.
_WHY_DRAIN = ("DRAIN. Highest-CV unsent non-vetoed file left in the queue; sent in ascending "
              "CV order so the best of the ten goes LAST and wins any public tie (w46b §5).")
_WHY_UNBLOCKED = ("DRAIN, and NEWLY SENDABLE: blocked on 08-23 by the P_MAX hijack-risk gate at "
                  "P(above the 0.97118 tier) = {p:.3f}; the tier is now 0.97119 and the same "
                  "file reads {q:.3f}. Nothing about the file changed — the board did.")
WHY_0824 = {s: _WHY_DRAIN for s in ORDER_0824}
for _s, _p, _q in (("w40_ad211std_rescale", 0.0823, 0.0057),
                   ("w38_ad202std_rescale", 0.0611, 0.0036),
                   ("w27_ad188stdcorr", 0.0530, 0.0029)):
    WHY_0824[_s] = _WHY_UNBLOCKED.format(p=_p, q=_q)
WHY_0824["w40_ad211std_rescale"] += (" Best CV of the ten and therefore LAST.")

ORDERS = {"2026-08-22": list(REG["order"]), "2026-08-23": ORDER_0823,
          "2026-08-24": list(ORDER_0824)}
WHYS   = {"2026-08-22": WHY_0822,           "2026-08-23": WHY_0823,
          "2026-08-24": WHY_0824}

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
# whole point is that they are not ranked on CV.
# ⚠⚠ THE ORIGINAL COMMENT ENDED "They keep priority 0 and sort last." THAT WAS THE ARGUMENT,
# AND IT IS THE SAME ONE w54 REFUTED FOR THE VETO: sorting last is safe only while the queue
# outlasts the calendar, and w54 measured that it does not (63 sendable against 90 remaining
# slots). "Sorts last" is reached on ~2026-08-29, and an unpriced row is one the tier rule
# -- "a filler is SAFE iff its predicted public score is < 0.97118" -- cannot be evaluated on
# at all, sent with a submission message reading "predicted LB nan ... P(beat) nan".
# w55a bounds each such row on two instruments (Spearman-to-nearest-scored + the w37c prereg,
# each calibrated on this account's own landed history) and registers a point estimate below.
_has = q.cv.notna()
q["pred_lb"] = np.nan
# ⚠⚠ w62: PRICE THROUGH `QP._price`, NOT `QP.predict`. w60 moved the `member` label out of this
# file and into `stdflag.family()` so a reprice could not lose it — which means `q["fam"]` above
# now reads "member" for `w48_cal_hboyang_mix` BEFORE this line instead of after it, and
# `w53a_pricer` rightly REFUSES an unfitted family. That made this script raise KeyError, and
# because w60 skipped running it the break stayed invisible until the queue needed re-stamping.
# `QP._price` is w26d's own branch — the builder's registered instrument for a member row, the
# cv→LB line for everything else — so both writers of this CSV now price by ONE rule.
# ⛔ Do NOT "fix" this by moving the CAL_ROWS override earlier or by mapping member→h3: the
# override below sets the SAME number from the SAME json, and h3 is exactly the silent collapse
# w53a refuses. The label must reach the pricer, and the pricer must branch on it.
q.loc[_has, "pred_lb"] = [QP._price(r) for r in q[_has].itertuples()]
from scipy.stats import norm as _norm                            # noqa: E402
q["p_beat"] = np.nan
q.loc[_has, "p_beat"] = 1.0 - _norm.cdf(
    (QP.BEST_LB + QP.STEP / 2 - q.loc[_has, "pred_lb"])
    / np.array([QP.resid_sd(s) for s in q.loc[_has, "stem"]]))
# a MEASUREMENT never sorts, or is promoted, on a probability of beating anyone. w26d blanks
# this by FAMILY; the CAL_ROWS override below blanks it only for the stems it enumerates, so a
# member row outside CAL_ROWS (`stdflag.MEMBER_FILES` is the wider set) would keep a p_beat.
q.loc[q.fam == "member", "p_beat"] = np.nan
# ⚠ A CALIBRATION FILE MUST NOT CARRY A STACK PRICE. QP.predict is the cv->LB line fitted on
# CROSS-FITTED STACKS; applied to a raw member vector it produced `pred 0.97129, P(beat) 1.00`
# for w48_cal_hboyang_mix -- i.e. it announced that a single imported member is CERTAIN to beat
# the account best. w37e's R2 killed the linear form off-range at 6.3σ precisely so this would
# not be done. Overwrite with the builder's own registered prediction, and blank p_beat: these
# files are measurements and must never sort or be promoted on a probability of beating anyone.
# ⚠⚠ AND ITS MESSAGE (w55). The comment above says a calibration send must not be described as
# a queue-drain -- "a future run reading the submission history would misread its ~0.958 public
# score as a huge regression" is `w26g_send.py`'s own words for why the `msg` override exists.
# THE OVERRIDE WAS NEVER WIRED. `msg` is reset to NaN at write time below and nothing ever set
# it, so w48_cal_hboyang_mix -- slot 1 of the 08-23 list, THE ARM 217 TEST, the file whose
# public score decides whether the ad217 veto is re-argued -- was going to ship: "queue-drain
# ... P(beats the 0.97118 account best) nan ... this is -66.5e-6 below the best sent CV", where
# that last figure compares a raw member's OOF against a cross-fitted stack CV, the one
# comparison RESEARCH.md says must never be made. `WHY[stem]` is already the registered
# rationale, thresholds and all; it just was not connected to anything. Connect it.
_REGMSG = {}
for stem, jf in CAL_ROWS.items():
    m = q.stem == stem
    if m.any() and os.path.exists(os.path.join(HERE, jf)):
        q.loc[m, "pred_lb"] = json.load(open(os.path.join(HERE, jf)))["pred_lb"]
        q.loc[m, "p_beat"] = np.nan
        q.loc[m, "fam"] = "member"
        if stem in WHY:
            _REGMSG[f"{stem}.csv"] = (
                f"w26g slot-1 calibration {stem} — {WHY[stem]} Family `member`: a MEASUREMENT, "
                f"not a candidate. Its 0.9701816 is a raw member's standalone OOF and is NOT a "
                f"cross-fitted stack CV — do not compare the two, and do not read this file's "
                f"public score as a leaderboard attempt.")

# ⚠ THE UNPRICEABLE ROWS (w55). Any row still carrying NaN pred_lb after the CAL_ROWS pass is
# invisible to the tier rule. `w55a_unpriced.json` certifies each one below the auto-selection
# tier on the tighter of two instruments and supplies a point estimate; we adopt it, and mark
# the row `member` for the same reason the calibration rows are marked: it is a MEASUREMENT,
# not a candidate, and must never enter a max(CV) or sort on a probability of beating anyone.
# A row NOT in the registry keeps its NaN on purpose -- `w26g_send.py` blocks it (w55), which
# is the fail-safe. Certifying a new one is a w55a run, not an edit here.
_UP = os.path.join(HERE, "w55a_unpriced.json")
if os.path.exists(_UP):
    _up = json.load(open(_UP))
    _n = 0
    for _f, _r in _up["rows"].items():
        _m = (q.file == _f) & q.pred_lb.isna()
        if not _m.any():
            continue
        if not _r.get("safe"):
            print(f"  ⛔ {_f} is in the w55a registry but NOT certified below the tier; "
                  f"leaving it unpriced so the sender blocks it")
            continue
        q.loc[_m, "pred_lb"] = _r["reg_lb"]
        q.loc[_m, "p_beat"] = np.nan
        q.loc[_m, "fam"] = "member"
        # ...and its MESSAGE, here, at the registration site. Without this the row falls through
        # to `w26g`'s queue-drain template, which formats `cv` and `p_beat` -- both NaN on a
        # member row -- and ships a description reading "CV nan ... P(beat) nan ... nan e-6
        # below the best sent CV". This account's submission descriptions ARE its memory across
        # runs (the brief says so outright), so a `nan` description is a real loss, and the
        # evidence for the certification belongs in the same place the certification is applied.
        _REGMSG[_f] = (
            f"w55 tail-fill {_f[:-4]} — a MEASUREMENT, not a candidate. Family `member`: this "
            f"is a raw member / superseded object with NO cross-fitted stack CV, so it is "
            f"deliberately unranked and must never enter a max(CV). Sent only to fill a slot "
            f"that would otherwise go unused. w55a certifies its public score BELOW the "
            f"{_up['tier']:.5f} auto-selection tier — instrument `{_r['instrument']}`, point "
            f"estimate {_r['reg_lb']:.6f}, bound {_r['bound']:.6f}, margin "
            f"{_r['margin_e6']:+.1f}e-6 — so it cannot be auto-selected while nothing is "
            f"selected, and therefore costs nothing. NOT a deadline pick.")
        _n += 1
    print(f"  priced {_n} previously-unpriceable row(s) from w55a_unpriced.json "
          f"(tier {_up['tier']:.5f}; every one certified below it)")
else:
    print("  ⚠ w55a_unpriced.json absent — unpriceable rows stay NaN and w26g will block them")

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
# ⚠ AFTER the reset, never before -- this loop wipes `msg`, which is why the two earlier
# assignment sites had to hand their text to `_REGMSG` rather than write it directly (w55).
q["msg"] = q["msg"].astype(object)
for _f, _t in _REGMSG.items():
    q.loc[q.file == _f, "msg"] = _t
if _REGMSG:
    print(f"  registered a submission message for {len(_REGMSG)} member row(s): "
          f"{', '.join(sorted(f[:-4] for f in _REGMSG))}")
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
