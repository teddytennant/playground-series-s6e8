"""w55a — AN UNPRICEABLE ROW IS AN UNVETOABLE ROW. Find them, bound them, and make the
bound bind in code.

THE GAP. w54 established the one number that makes a spare slot free or a liability:

    auto-selection tier = 2nd-best PUBLIC score on the account = 0.97118
    a filler is SAFE iff its predicted public score is < the tier

and wrote it into RESEARCH.md and into `w26g_send.py`'s own unfilled-slots message. It is
enforced by **nothing**. That is w54's own lesson one level up — *a rule that lives in the
ORDER of a list is not a rule*, and neither is a rule that lives in a printed paragraph.

It matters because the queue contains rows the pricer **cannot price**. Four sendable rows
carry `cv = NaN` and `pred_lb = NaN`:

    w15f_antistudent_cv  w16d_membercell  w37_cal_dkv_xgb  w37_cal_ravi_realmlp1c

`w26g_send.py`'s plan loop has no test for that. They sort to the back on a NaN key, survive
every existing skip (md5, check_file, veto), and on ~2026-08-29 they go out with a submission
message that reads, literally, "CV nan, family ens4, w26d predicted LB nan with P(beats the
0.97118 account best) nan ... this is nan e-6 below the best sent CV". The journal is explicit
that past submission descriptions are this account's main memory across runs; four of them
saying `nan` is a provenance loss, and the tier rule cannot be evaluated on them at all.

WHAT THIS SCRIPT DOES, AND WHY IT IS A MEASUREMENT AND NOT AN ASSERTION. It would be easy to
wave at these four -- they are old, low-CV objects and "obviously" score nowhere near the tier.
This account has been burned by obviously. So the bound is computed from the account's own
scored history:

  1. Read every scored submission (111) and every unpriceable queue row from `submissions/`.
  2. Rank each prediction vector on a FIXED deterministic stride of the test rows (no RNG --
     `Math.random`-style nondeterminism would make the bound unreproducible).
  3. Build the empirical ENVELOPE from the 111*110/2 scored pairs: for a grid of Spearman
     thresholds s, the largest |public_i - public_j| ever observed on this account among pairs
     with spearman >= s. This converts "how similar is this file to one I have a score for"
     into a defensible upper bound on how far its public score can be from that anchor.
  4. For each unpriceable row: nearest scored anchor by Spearman, bound = anchor + envelope,
     verdict SAFE iff bound < the live tier.

     ⚠ That instrument does NOT cover every row, and the script says so instead of guessing.
     `w37_cal_dkv_xgb`'s nearest scored neighbour is only 0.98905 Spearman away, which falls
     off the grid entirely and collapses to the global max |delta| of 8330e-6 -- a vacuous
     bound. A raw imported member has a better instrument, and it was pre-registered a week
     ago: `w37c_prereg.csv` carries a `pred_lb` for all SEVEN calibration files, and FIVE of
     them have since landed. Their residuals (+1560, +737, +166, +44, -4 e-6) calibrate it
     out of sample. So instrument B, for any row appearing in that prereg, is
     `pred_lb + max residual over the landed five`. Each row is certified on the TIGHTER of
     whichever instruments are VALID for it; a row no instrument covers is not certified.
  5. Emit `w55a_unpriced.json` -- the registry `w48e_order.py` reads to price these rows
     honestly, so they stop being NaN at all.

  6. REGRESSION TEST, in the shape w54a established: assert the NaN guard string is still
     present in `w26g_send.py`. Prose in a journal does not survive; a test that re-reads the
     sender does. Exit 1 if the guard is gone.

    .venv/bin/python experiments/w55a_unpriced.py
"""
from __future__ import annotations

import io, json, os, subprocess, sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submissions")
QUEUE = os.path.join(HERE, "w26d_queueprice.csv")
SENDER = os.path.join(HERE, "w26g_send.py")
DST = os.path.join(HERE, "w55a_unpriced.json")

# w54: the submissions API silently truncates to 50 rows. Always pass a page size and guard.
PAGE = 500
# Deterministic subsample of the 296,302 test rows. A stride, not an RNG draw: the bound has to
# be reproducible by any later run without carrying a seed around.
STRIDE = 5
# The guard string w26g_send.py must still contain for this test to pass.
GUARD = "w55: unpriceable"


def _api_rows():
    env = dict(os.environ)
    env.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    out = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e8", "-v",
         "--page-size", str(PAGE)],
        capture_output=True, text=True, env=env, timeout=180)
    if out.returncode != 0:
        raise SystemExit(f"submissions API failed: {out.stderr.strip()[:300]}")
    rows = pd.read_csv(io.StringIO(out.stdout))
    if len(rows) >= PAGE:
        raise SystemExit(f"submissions API returned {len(rows)} >= page size {PAGE}; raise PAGE")
    return rows


def _scored(rows):
    s = rows[rows.status == "SubmissionStatus.COMPLETE"].copy()
    s["publicScore"] = s.publicScore.astype(float)
    # one row per file: its best public score (a file resent scores identically anyway)
    s = s.sort_values("publicScore", ascending=False).drop_duplicates("fileName")
    return s[["fileName", "publicScore"]].reset_index(drop=True)


def _ranks(path):
    v = pd.read_csv(path, usecols=["addicted_label"]).addicted_label.to_numpy()[::STRIDE]
    return rankdata(v).astype(np.float32)


def main():
    rows = _api_rows()
    sc = _scored(rows)
    tier = float(np.sort(sc.publicScore.to_numpy())[::-1][1])   # 2nd best = auto-selection tier
    best = float(sc.publicScore.max())
    print(f"scored files {len(sc)}   account best {best:.5f}   auto-selection tier {tier:.5f}")

    q = pd.read_csv(QUEUE)
    send = q[(q.priority >= 0)]
    bad = send[send.pred_lb.isna() | send.cv.isna()]
    print(f"\nqueue: {len(q)} unsent, {len(send)} sendable, "
          f"{len(bad)} UNPRICEABLE (cv or pred_lb is NaN)")
    if not len(bad):
        print("  nothing unpriceable — the registry is empty and the guard is a no-op.")

    missing = [f for f in sc.fileName if not os.path.exists(os.path.join(SUB, f))]
    if missing:
        raise SystemExit(f"{len(missing)} scored files absent from submissions/: {missing[:4]}")

    print(f"\nranking {len(sc)} scored + {len(bad)} unpriceable vectors "
          f"(every {STRIDE}th test row) ...")
    R = np.vstack([_ranks(os.path.join(SUB, f)) for f in sc.fileName])
    B = np.vstack([_ranks(os.path.join(SUB, f)) for f in bad.file]) if len(bad) else \
        np.zeros((0, R.shape[1]), np.float32)
    print(f"  rank matrix {R.shape}, subsampled from 296302 rows")

    # ---- the envelope: what the account's own history says similarity buys you ----------
    C = np.corrcoef(R)                                  # Spearman, since R holds ranks
    P = sc.publicScore.to_numpy()
    D = np.abs(P[:, None] - P[None, :])
    iu = np.triu_indices(len(sc), 1)
    cs, ds = C[iu], D[iu]
    grid = [0.99, 0.999, 0.9999, 0.99995, 0.99999]
    env = {}
    print("\nEMPIRICAL ENVELOPE over the account's own scored pairs "
          f"({len(cs)} pairs of {len(sc)} files):")
    print("   spearman >=    pairs    max |delta public|")
    for g in grid:
        sel = cs >= g
        env[g] = float(ds[sel].max()) if sel.any() else float("nan")
        print(f"   {g:<12.5f}  {int(sel.sum()):5d}    {env[g]*1e6:9.1f}e-6"
              + ("" if sel.any() else "   (no pairs — bound unusable at this threshold)"))

    def envelope(s):
        """Largest |delta public| this account has ever shown at similarity >= s.
        Falls back to the loosest populated threshold, never to an optimistic one."""
        for g in sorted(grid, reverse=True):
            if s >= g and np.isfinite(env[g]):
                return env[g], g
        return float(ds.max()), 0.0

    # ---- INSTRUMENT B: the calibration preregs, validated on their landed rows ----------
    # ⚠ TWO FILES NOW. `w85b_prereg.csv` registers the 25 w85 slot fillers in the SAME schema
    # and for the same reason -- a pred_lb written down before the score exists. The overshoot
    # `b_max` stays a single number measured over every LANDED prereg row on the account, so a
    # w85 row is bounded by residuals that were earned, not by one asserted here. While only
    # the w37 rows have landed, b_max is exactly what it was before this file changed.
    _pres = [pd.read_csv(os.path.join(HERE, f)) for f in
             ("w37c_prereg.csv", "w85b_prereg.csv")
             if os.path.exists(os.path.join(HERE, f))]
    pre = pd.concat(_pres, ignore_index=True)
    assert pre.file.is_unique, "a file is registered in more than one prereg"
    act = dict(zip(sc.fileName, sc.publicScore))
    pre["actual"] = pre.file.map(act)
    landed = pre.dropna(subset=["actual"])
    b_max = float((landed.actual - landed.pred_lb).max())
    print(f"\nINSTRUMENT B — {len(_pres)} prereg(s), {len(landed)} of {len(pre)} landed. "
          f"residuals (actual - pred), e-6:")
    for r in landed.itertuples():
        print(f"   {r.file:28s} pred {r.pred_lb:.6f}  actual {r.actual:.5f}  "
              f"{(r.actual - r.pred_lb)*1e6:+8.1f}")
    print(f"   worst overshoot {b_max*1e6:+.1f}e-6  ->  bound = registered pred_lb + that")
    PRE = {r.file: float(r.pred_lb) for r in pre.itertuples()}

    # ---- bound each unpriceable row on every instrument that is VALID for it ------------
    reg, verdicts = {}, []
    for i, r in enumerate(bad.itertuples()):
        c = (np.corrcoef(np.vstack([B[i], R]))[0, 1:])
        j = int(np.argmax(c))
        s_, anchor, apub = float(c[j]), sc.fileName[j], float(sc.publicScore[j])
        e, g = envelope(s_)
        cands = {"spearman": {"bound": apub + e, "point": apub, "anchor": anchor,
                              "spearman": s_, "envelope": e, "bucket": g,
                              "valid": g > 0.0}}
        if r.file in PRE:
            cands["prereg"] = {"bound": PRE[r.file] + b_max, "point": PRE[r.file],
                               "anchor": "prereg", "overshoot": b_max, "valid": True}
        ok = {k: v for k, v in cands.items() if v["valid"]}
        print(f"\n  {r.file}")
        for k, v in cands.items():
            tag = "" if v["valid"] else "   ⛔ INVALID — envelope collapsed to the global max"
            print(f"    {k:9s} point {v['point']:.6f}  bound {v['bound']:.6f}{tag}")
            if k == "spearman":
                print(f"              nearest {v['anchor']} (public {v['point']:.5f}), "
                      f"spearman {v['spearman']:.7f}, bucket >= {v['bucket']:g}")
        if not ok:
            print("    ⛔ NO VALID INSTRUMENT — this row cannot be certified below the tier.")
            verdicts.append(False)
            reg[r.file] = {"safe": False, "why": "no valid instrument"}
            continue
        use = min(ok, key=lambda k: ok[k]["bound"])
        bound, point = ok[use]["bound"], ok[use]["point"]
        safe = bound < tier
        verdicts.append(safe)
        print(f"    certified on `{use}`  bound {bound:.6f}  tier {tier:.5f}  ->  "
              f"{'SAFE' if safe else '⛔ COULD BE AUTO-SELECTED'}"
              f"   margin {(tier - bound)*1e6:+.1f}e-6")
        reg[r.file] = {"instrument": use, "bound": bound, "reg_lb": point,
                       "margin_e6": (tier - bound) * 1e6, "safe": bool(safe),
                       "all": {k: {kk: vv for kk, vv in v.items()} for k, v in cands.items()}}

    payload = {"tier": tier, "account_best": best, "stride": STRIDE,
               "n_scored": int(len(sc)), "envelope": {str(k): v for k, v in env.items()},
               "rows": reg}
    with open(DST, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nwrote {os.path.relpath(DST, ROOT)}")

    # ---- the regression test (w54a's shape) ---------------------------------------------
    src = open(SENDER).read()
    ok = GUARD in src
    print(f"\nguard check: `{GUARD}` in w26g_send.py -> {'present ✅' if ok else 'MISSING ⛔'}")
    if not ok:
        print("  The sender no longer refuses unpriceable rows. Four NaN-priced files can\n"
              "  reach the API with a `nan` submission message and no tier check. Restore it.")
        return 1
    if not all(verdicts):
        print("  ⛔ an unpriceable row is NOT bounded below the tier — do not send it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
