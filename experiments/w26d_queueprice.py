"""w26d — price the ENTIRE unsent send queue under the w25f model, before spending a day on it.

The brief says an unused slot is pure waste and to send all ten every day, because submissions
here never evict each other and the public LB shows best-of-all. That is still true. What it
does NOT say is that the queue is inexhaustible, and this slot noticed that it very nearly is
exhausted: `w23b_sendqueue` reports the best CV among ALREADY-SENT files as 0.9701150809 and
the best CV among the 48 unsent ones as 0.9700483189 -- 67e-6 BELOW it. Every remaining
candidate is materially worse on CV than something already on the board.

w25f now lets that be priced rather than guessed. It fits, over the 60 scored files with
CV >= 0.97,  LB = const + 1.909*CV_e6 + family + standardised,  with residual sd 8.41e-6
against an independently simulated slice-noise floor of 8.21e-6 -- i.e. the fit is complete
and the residual is pure slice draw. So for each unsent file we can state a predicted LB and,
with the residual as the only remaining uncertainty, P(this file beats the account best).

That probability is the number that decides how to spend the 08-19 Kaggle day. It cannot make
sending harmful -- the brief's economics are sound and a send still cannot hurt public rank --
but it distinguishes "drain the queue" from "build something above 0.9701150809 first", and
only one of those is worth a run's compute.

    .venv/bin/python experiments/w26d_queueprice.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


# The classifiers now live in `experiments/stdflag.py` and are IMPORTED, not copied.
#
# ⚠⚠ SECOND BUG IN THIS FLAG, FOUND 2026-08-19 (w28). w27 slot 3 replaced a hard-coded
# whitelist with the substring test `"std" in stem`, which was right on every row w25f fitted
# and WRONG on the six `w27_ad188raw*` files -- built by `w27k_ctrawctl.sh`, which passes
# `--standardize`; the "raw" is the CT-raw MEMBER, not a raw combiner. Each was handed the
# +27.43e-6 the `standardised` coefficient withholds, and `w27_ad188raw` consequently priced
# at P(beat the account best) = 0.548, the best number on disk. Corrected: 8.4e-4, and the
# whole ten-file day drops from P = 0.647 to 0.083. A filename is not provenance; `stdflag`
# derives the flag from the WAVE NUMBER, because `--standardize` entered the chain at w23 and
# every build since passes it. That module gates itself against w25f's own 60 rows.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
sys.path.insert(0, HERE)
from common import SUB  # noqa: E402 — w60, for the un-regenerable-row carry below
import stdflag  # noqa: E402
from stdflag import STD_FILES, family, is_std  # noqa: E402,F401

# w25f fits on CENTRED CV: cv_e6 = (cv - mu) * 1e6, mu the mean over its own 60 rows. mu is
# not in the JSON, so it is recomputed here from the same table under the same CV >= 0.97
# filter. Feeding the model raw CV instead gave a predicted LB of 2.82 on the first run --
# absurd enough to be caught on sight, which is exactly why the gate below now exists: the
# next parameterisation error will not be six orders of magnitude out and will not be obvious.
_t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
_fit = _t[_t.cv >= 0.97].copy()
MU = _fit.cv.mean()

# ⚠ THE MODEL IS w26e's REFIT, NOT w25f (changed w28, 2026-08-19). w26e found that the two
# `*corr` files -- one of which is `WANTED` slot 1 -- were labelled `ens4` by the suffix rule
# when they are corrected **h3** mixes, refitted with only those two labels changed, and got a
# LOWER residual sd (8.349e-6 vs 8.408e-6) with no extra parameters. That refit then sat unused
# for two days while the pricer kept the wrong labels. It matters most exactly where the
# decision is: the queue's top two files are BOTH `*corr`, so under the suffix rule each was
# collecting the +13.9e-6 `ens4` term on top of an h3 base.
# ⚠ THE MODEL IS w30b, NOT w26e (changed w30, 2026-08-20). The 08-20 ten-file drain was the
# first genuinely HELD-OUT test this predictor has ever had -- ten files priced before upload
# under a fit containing none of them (w30a). It came back unbiased overall (mean residual
# +1.67e-6, z +0.60) but with the two `*stdcorr` files +19.3e-6 high, z +3.08. w30b refits with
# a c_avg-CORRECTION indicator, which w26e had no term for, and gets +12.61e-6 (se 4.07,
# t +3.10) with the residual sd falling 8.23 -> 7.76e-6 -- BELOW the 8.70e-6 slice+grid noise
# floor, i.e. the CV->LB relation is now fully accounted for. Family coefficients barely move,
# so this is not the h3 term in disguise: the six corrected files span h3, ens4 AND rankraw.
# ⚠ CAVEAT, the reason to read the corr term as ~+10e-6 rather than +12.6: it survives a
# quadratic-CV guard (+9.92, t +2.12) and the top-CV band (+15.71, t +3.37) but NOT the drop of
# today's two near-twins (+7.57, t +1.59). Positive in all four cuts, significant in three.
# ⚠ THE MODEL IS w46c, NOT w30b (changed w48, 2026-08-21). w30b is fitted on 78 files that are
# ALL ad<=194. Five ad>=195 files first scored on 08-21 came in a mean -29.82e-6 BELOW its
# prediction (z -8.0 at w47a's leave-one-out sd of 8.36e-6). This file kept the raw w30b copy
# inline for two runs after that was known, and was therefore wrong by -30e-6 on 38 of the 77
# queue files -- including the one it printed into the permanent Kaggle submission log at 00:07
# UTC on 08-21 as "predicted LB 0.971212 with P(beats the account best) 1.00e+00". That file
# scored 0.97118. A probability published as 1.00 for an event that did not happen.
#
# The era term has now survived SIX independent attacks: w47a's E1 (extrapolation above
# support), E2 (prospective optimism), E3 (understated precision) and G (day clustering), plus
# w48c's S1/S2 (an over-steep global slope -- refitting with the six sub-floor files moves the
# slope only 1.856 -> 1.810 and the shift only -29.8 -> -27.8e-6) and S3 (curvature inside
# support, which makes it BIGGER at -34.1e-6). It is not a model artefact.
#
# The correction lives in ONE place, w46c_predlb.py. Do not re-inline it here or anywhere.
# ⚠ THE MODEL IS w53a's M2, NOT w46c (changed w53, 2026-08-22). w46c's flat era LEVEL dummy
# was refuted by the ten-file experiment w47b designed to test it: all three registered rules
# fired against it (w52 §2), and w52b's leave-one-day-out comparison put the era x cv
# INTERACTION ahead of the dummy on the full 12-day LODO (8.69 vs 9.59e-6) and on the era slice
# (8.77 vs 12.82e-6). The era files do not sit a fixed distance below the line; they convert CV
# to LB at 79% of the base slope. w52 §8.4 registered this rewire as the next run's job and
# warned that the GATE below asserts against w30b and would hard-fail inside `w48e_order.py` on
# the send path if the model were swapped without moving it. It has been moved, not deleted.
#
# The correction lives in ONE place, w53a_pricer.py. Do not re-inline it here or anywhere.
import w53a_pricer as PR  # noqa: E402
import w46c_predlb as W46  # noqa: E402  — SUPERSEDED; kept only for ERA_MIN_AD / new_era

M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C = M["coefs"]             # w30b's coefficients, still used by the M0 comparison printed below
RESID = PR.RESID_SD        # 7.72e-6 -- M2's in-sample residual sd, dof-corrected
RESID_ERA = PR.SD_ERA      # 8.77e-6 -- M2's HELD-OUT era-slice RMSE under leave-one-day-out
BEST_LB = 0.97118          # account best, w21_ad187corr_ens4
STEP = 1e-5                # the LB reports to 5 decimals


def is_corr(stem):
    """Carries the c_avg / scheme-average correction. Every such build in this workspace has
    `corr` in its stem and nothing else does; stdflag.CORR_MAP enforces registration."""
    return "corr" in stem


def predict(cv, fam, standardised, corrected=False, stem=None):
    """Predicted public LB, in absolute AUC. `standardised` is the combiner-scaling flag,
    `corrected` the c_avg-correction flag (w30b).

    ⚠ PASS `stem`. Without it this prices the file as PRE-ERA, which is wrong by the whole era
    slope adjustment for any ad>=195 build. The argument is optional only so the pre-w48 callers
    (w39b, w39c, w39d) keep working unchanged; every file those three price is ad<=194, where
    era and non-era agree exactly. Any NEW caller must pass it."""
    era = stem is not None and PR.new_era(stem)
    return PR.predict_flags(cv, fam, standardised, corrected, era)


def resid_sd(stem=None):
    """The predictive sd for `stem`, in absolute AUC. Wider in the new era because M2's era
    slope is estimated from 15 points and the era slice is where it is asked to extrapolate;
    RESID_ERA is w52c's HELD-OUT leave-one-day-out RMSE on that slice, not an in-sample number."""
    return (RESID_ERA if (stem is not None and PR.new_era(stem)) else RESID) * 1e-6


# GATE. MOVED to M2 (w53), not deleted -- w52 §8's standing warning is that deleting the
# self-check is the tempting shortcut when the model underneath it changes. It has caught two
# real parameterisation slips already (raw-vs-centred CV, and ddof), and both were silent.
#
# The gate now re-predicts the 93 rows M2 was fitted on THROUGH `predict()` -- the public entry
# point this file's callers use, including `w48e_order.py` on the send path -- and requires
# w53a's own dof-corrected residual sd back. Note the era flag is passed via `stem`, so a
# regression in the era routing (the exact thing this rewire changes) fails the gate loudly.
_g = pd.read_csv(os.path.join(HERE, "w52b_cvlb93.csv"))
_g6 = (_g.lb.values - np.array([predict(r.cv, r.fam, bool(r["std"]), bool(r["corr"]), r.stem)
                                for _, r in _g.iterrows()])) * 1e6
_dof = len(_g) - len(PR.NAMES)
_rsd = float(np.sqrt((_g6 ** 2).sum() / _dof))
print(f"GATE: M2 refitted-sample residual sd {_rsd:.3f}e-6 (dof {_dof}) vs w53a's "
      f"{RESID:.3f}e-6 -- {'PASS' if abs(_rsd - RESID) < 0.01 else 'FAIL'}")
assert abs(_rsd - RESID) < 0.01, "predict() does not reproduce w53a's M2; nothing below is readable"
# Second gate, unchanged in spirit from w53a's: the WANTED deadline pick is the one row whose
# true LB this workspace cares about. If a refit moves it, the pricer changed under the decision.
_wp = predict(PR.WANTED_CV, family(PR.WANTED_STEM), is_std(PR.WANTED_STEM),
              is_corr(PR.WANTED_STEM), PR.WANTED_STEM)
print(f"GATE: {PR.WANTED_STEM} pred {_wp:.6f} vs actual {PR.WANTED_LB:.5f} -- "
      f"{'PASS' if abs(_wp - PR.WANTED_LB) < 5e-6 else 'FAIL'}")
assert abs(_wp - PR.WANTED_LB) < 5e-6, "M2 no longer reproduces the WANTED pick's actual LB"


q = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
q = q[~q.sent].dropna(subset=["cv"]).copy()
q["stem"] = q.file.str.replace(".csv", "", regex=False)
stdflag.require_corr_registered(q.stem)   # a new *corr file must be classified BY HAND
q["fam"] = q.stem.map(family)
q["std"] = q.stem.map(is_std)          # see is_std: a whitelist here was a BUG
q["corr"] = q.stem.map(is_corr)
# ⚠ MEMBER ROWS ARE NOT PRICED BY THIS MODEL (w60). `family()` now derives `fam == "member"`
# itself (stdflag.MEMBER_FILES) instead of waiting for `w48e_order.py` to write the label into
# the CSV — and `w53a_pricer` rightly REFUSES a family that is not in its fitted design, because
# an unknown family collapses onto the h3 reference level and produces a price that looks real.
# A raw member vector has no transform and no cross-fitted stack CV, so there is nothing here to
# price it with. Its predicted LB comes from the instrument its own builder registered, and its
# `p_beat` is blanked: a MEASUREMENT must never sort, or be promoted, on a probability of
# beating anyone (w48e's words, now enforced where the label lives).
MEMBER_PRED = {"w48_cal_hboyang_mix": ("w49a_calhboyang.json", "pred_lb")}


def _price(r):
    if r.fam == "member":
        src = MEMBER_PRED.get(r.stem)
        if src and os.path.exists(os.path.join(HERE, src[0])):
            return float(json.load(open(os.path.join(HERE, src[0])))[src[1]])
        # No registered instrument -> NaN, and w55's rule then applies: a row the tier test
        # cannot be evaluated on is NOT covered by it, and `w26g_send` blocks on the NaN.
        return float("nan")
    return predict(r.cv, r.fam, r.std, r.corr, r.stem)


q["pred_lb"] = [_price(r) for r in q.itertuples()]
q["sd"] = [resid_sd(r.stem) for r in q.itertuples()]
q["era"] = [W46.new_era(r.stem) for r in q.itertuples()]

# P(beat the account best). The file must print STRICTLY above 0.97118, and the LB rounds to
# 1e-5, so the target on the underlying scale is BEST_LB + STEP/2.
q["p_beat"] = 1.0 - norm.cdf((BEST_LB + STEP / 2 - q.pred_lb) / q.sd)
q.loc[q.fam == "member", "p_beat"] = np.nan   # a measurement never sorts on P(beat) -- w48e

# PRIORITY. The queue is ranked on predicted PUBLIC LB, which is the right order for chasing
# public rank and the WRONG order for the one thing a submission is actually needed for here:
# Kaggle's final-selection dialog lists SUBMITTED entries only, so a deadline pick that is
# never sent cannot be ticked. `w27_ad188stdcorr` sat unsent for four slots behind three
# journal entries saying "send it first" because the sender reads this file and this file did
# not rank it first. Anything in `check_selection.WANTED` now goes to the head of the queue.
from check_selection import WANTED  # noqa: E402
q["priority"] = q.file.isin(WANTED).astype(int)
q = q.sort_values(["priority", "pred_lb"], ascending=[False, False])
_pin = q[q.priority == 1].file.tolist()
print(f"PINNED to the head of the queue (check_selection.WANTED, unsent): {_pin or 'none'}")

print(f"{len(q)} unsent files with a stored OOF vector, priced under w53a M2 "
      f"(resid sd {RESID:.2f}e-6 pre-era / {RESID_ERA:.2f}e-6 for ad>={W46.ERA_MIN_AD}, "
      f"centred at mu={MU:.10f})\n")
print("TOP 12 BY PREDICTED PUBLIC LB")
print(f"{'file':32s} {'fam':8s} {'std':5s} {'cv':>14s} {'pred LB':>9s} {'P(>best)':>9s}")
for r in q.head(12).itertuples():
    print(f"{r.file:32s} {r.fam:8s} {str(r.std):5s} {r.cv:.10f} {r.pred_lb:9.5f} "
          f"{r.p_beat:9.2e}")

pbest = q.p_beat.max()
# P(at least one of the ten best beats it), treating the residuals as independent draws.
# They are NOT independent -- all ten are read off ONE fixed public slice and share member
# sets -- so this OVERSTATES the chance, which is the safe direction for the conclusion.
ten = q.head(10)
p_any = 1.0 - np.prod(1.0 - ten.p_beat.values)

print(f"\nbest single file          P(beat {BEST_LB}) = {pbest:.3e}")
print(f"best TEN sent together    P(at least one)  = {p_any:.3e}   "
      f"(independence assumed -> an OVERSTATEMENT)")
print(f"\npredicted LB of the best unsent file: {q.pred_lb.iloc[0]:.5f}   "
      f"= {(q.pred_lb.iloc[0]-BEST_LB)*1e6:+.1f}e-6 vs the account best "
      f"({(q.pred_lb.iloc[0]-BEST_LB)/STEP:+.1f} reporting steps)")

# What CV would a NEW file need for an even-money shot at the record?
#
# ⚠ This used to omit the `standardised` term, so every bar it printed was the bar for an
# UNSTANDARDISED file — and every build this workspace has made since w23 is standardised. The
# quoted "ens4 bar is 0.9701108460, 4.2e-6 BELOW the current CV leader" is that omission: the
# real bar for the standardised ens4 files we actually build is 0.9701252, +10.1e-6 ABOVE the
# leader. Both are printed now so the assumption can never be dropped again.
need = (BEST_LB + STEP / 2) * 1e6
# The CV leader is read LIVE off the queue, never quoted -- it was hard-coded at 0.9701150809
# (the w23 leader) for three waves after two better files existed, so every "vs leader" figure
# printed here between w27 and w29 was stale by 3.0e-6.
LEADER = float(max(q.cv.max(), _fit.cv.max()))
print(f"\nCV needed for an even-money shot at {BEST_LB:.5f}, BY FAMILY AND BY SCALING")
print(f"  (leader {LEADER:.10f})")
print(f"  {'family':>7s}  {'unstd':>16s}  {'std':>16s}  {'std+corr':>16s}   {'gap':>10s}")
for fam in ("h3", "ens4", "hybrid", "rankraw", "rescale", "logit"):
    base = (need - C["const"] - C.get(f"fam[{fam}]", 0.0)) / C["cv_e6"] * 1e-6 + MU
    std = base - C["std"] / C["cv_e6"] * 1e-6
    stdc = std - C["corr"] / C["cv_e6"] * 1e-6
    print(f"  {fam:>7s}  {base:16.10f}  {std:16.10f}  {stdc:16.10f}   "
          f"{(stdc - LEADER)*1e6:+8.1f}e-6")
print("  Every file this workspace builds is standardised, and the ones worth building are")
print("  ALSO corrected -- read the std+corr column. `gap` is what a NEW build must add to")
print("  the CV leader to be an even-money shot at the board; negative means already there.")

# ⚠⚠ GUARDS ADDED w39 (2026-08-20). READ BEFORE RUNNING THIS FILE.
#
# `w26d_queueprice.csv` is not just this script's output any more. `w37d_order.py` adds
# `send_rank` / `msg` / `why` to it, and `w26g_send.py` reads those to decide WHAT goes out in
# WHICH slot with WHICH message. It also carries rows this script cannot regenerate: the
# `w37_cal_*` calibration files are registered by `w37c_prereg.py`, not by `w23b_sendqueue.csv`.
#
# Both ways of losing that were demonstrated live in w39, silently, with no error:
#   1. `import w26d_queueprice` (done only to reuse `predict()`) ran the write at import time
#      and dropped all three columns.
#   2. Re-running the script as intended rebuilt `priority` from `w23b_sendqueue.csv` alone,
#      cutting the 13-file pre-registered send order down to 1 row and dropping all 7 messages.
#
# So: the write happens only under `__main__` (import is now side-effect free), the carried
# columns are merged back, and the write ABORTS if it would drop a row or a set `send_rank`
# that the file on disk already has. To deliberately re-price from scratch, pass --force, and
# re-run `w37d_order.py` immediately afterwards to rebuild the order.
#
# ⚠ EXTENDED w60, 2026-08-22. The guard above was doing its job and STILL blocked real work:
# w60 built two new files, re-ran `w23b_sendqueue.py`, and the reprice refused because the four
# OOF-less `w37_cal_*` / member rows are not regenerable from `w23b_sendqueue.csv`. The only
# documented way past it was `--force`, whose own instruction ("re-run w37d_order.py") points at
# a script that hard-codes the 08-21 order — every file in it long since sent. So the choices on
# offer were: lose the four rows, or leave two new files out of the queue the sender reads.
#
# That is a false choice, and the fix is to make the REBUILD LOSSLESS rather than to force past
# a guard that is correctly telling you the rebuild is lossy. A row is carried verbatim iff the
# script provably cannot regenerate it — no `oof_<stem>.npy` on disk, which is the exact reason
# `w23b_sendqueue.py` drops it — and it is carried UNPRICED, because it never had a CV to price
# from. ⛔ Any row lost for ANY OTHER reason still trips the refusal: the guard caught a real
# w39 bug and must keep its teeth. Carrying is reported by name, never silent.
_CARRY = ["send_rank", "msg", "why"]
_dst = os.path.join(HERE, "w26d_queueprice.csv")


def _unpriceable(stems):
    """Stems with no stored OOF vector — exactly what w23b_sendqueue.py drops, and the only
    rows this script is entitled to carry rather than rebuild."""
    return {s for s in stems
            if not os.path.exists(os.path.join(SUB, f"oof_{str(s).replace('.csv','')}.npy"))}


def _write(force=False):
    global q
    lost_rows, lost_rank, carried = [], [], []
    if os.path.exists(_dst):
        prev = pd.read_csv(_dst)
        keep = [c for c in _CARRY if c in prev.columns]
        if keep:
            q = q.merge(prev[["file"] + keep], on="file", how="left")
        lost_rows = sorted(set(prev.file) - set(q.file))
        carry_ok = sorted(_unpriceable(lost_rows))
        if carry_ok:
            add = prev[prev.file.isin(carry_ok)].copy()
            for c in q.columns:
                if c not in add.columns:
                    add[c] = float("nan")
            q = pd.concat([q, add[q.columns]], ignore_index=True)
            carried = carry_ok
            lost_rows = sorted(set(lost_rows) - set(carry_ok))
        if "send_rank" in prev.columns:
            ranked = prev[prev.send_rank.notna()]
            lost_rank = sorted(set(ranked.file) & set(lost_rows))
    if carried:
        print(f"\n  carried {len(carried)} un-regenerable row(s) forward UNPRICED (no OOF vector "
              f"on disk, so this script never priced them and cannot):")
        for f in carried:
            print(f"      {f}")
    if (lost_rows or lost_rank) and not force:
        print(f"\n*** REFUSING TO WRITE {os.path.basename(_dst)} ***")
        print(f"    {len(lost_rows)} row(s) on disk are not in the rebuilt queue and would be")
        print(f"    dropped, {len(lost_rank)} of them carrying a set send_rank:")
        for f in lost_rows:
            print(f"      {'[RANKED] ' if f in lost_rank else '          '}{f}")
        print("    Nothing was written. Re-run with --force ONLY if you will then re-run")
        print("    experiments/w37d_order.py to rebuild the send order it just destroyed.")
        sys.exit(3)
    q.to_csv(_dst, index=False)
    json.dump(dict(n=len(q), resid_sd=RESID, best_lb=BEST_LB, p_best_single=float(pbest),
                   p_any_of_ten=float(p_any), pred_lb_best=float(q.pred_lb.iloc[0])),
              open(os.path.join(HERE, "w26d_queueprice.json"), "w"), indent=1)
    print("\nwrote experiments/w26d_queueprice.{csv,json}")


if __name__ == "__main__":
    _write(force="--force" in sys.argv)
else:
    print("\nimported, not run: w26d_queueprice.csv left untouched")
