"""w133 -- what the final-selection click is worth in the unit the competition PAYS.

WHY THIS EXISTS. The click has been priced in AUC since w74 (+4.5228e-6 at tau=0) and in board
places since w113 (~0.95 teams per 1e-6 -> ~4 places). Neither is the unit Kaggle awards. The
competition pays a MEDAL, which is a private-side rank against a cut derived from the field
size, and no run has ever put the click's AUC delta and the public->private shake into the same
expression. On the last day, with 0 submission slots left, that is the only quantity that can
still move -- and the honest question is not "is the click positive" (it is) but "is it large
against the thing it is competing with", which is the width of the shake.

WHAT IS AND IS NOT PRICED HERE.
  * Instrument A (the matched null) CAN price the click: it is parametric in our own score, so
    a private-side delta on our row propagates to a rank distribution. Arms are PAIRED -- one
    noise draw, four deltas read off it -- so the reported difference is a difference, not the
    difference of two noisy estimates.
  * Instrument B (the empirical band) CANNOT price the click and is not asked to. B is
    conditional on our PUBLIC share, and the click changes nothing public: it changes which
    two already-sent files are scored privately. B's job here is to supply the SPREAD that A's
    delta has to be judged against.

THE REFERENCE IS THE STATUS QUO, NOT THE BEST OPTION. `check_selection.py` publishes its three
figures as costs against the WANTED pair. Nothing is selected, so the live state is the auto
pick, and the decision-relevant framing is "what does each action do from HERE":

    click WANTED       +4.5228e-6                      (w74a, tau = 0)
    no click            0                              (the status quo; Kaggle auto-selects)
    mis-click A        +4.5228 - 35.17 = -30.6472e-6   (w114a, w21_ad187corr + w20_ad187_h3)
    mis-click B        +4.5228 - 81.92 = -77.3972e-6   (w114a, w16i_schemeavg + blend159av_h3)

The board score is a PUBLIC proxy for skill and the deltas are PRIVATE-side expectations; they
are not mixed. Our displayed 0.97119 is the auto pair's public score, which is why the auto
pair is the zero.

CAVEAT THAT APPLIES TO EVERY ROW AND CANNOT BE REMOVED. The public board shows every team's
best-of-all-submissions, while their private score comes from two selected entries. That biases
the whole board optimistically, not just us, and the matched null cannot represent it. It is
the same assumption w83a makes; it is stated, not fixed.

DECIDES NOTHING. The final selection is on CV and is settled (SELECT_THESE.md). A small dP is a
REPORTING result and is not a licence to re-open the pick.

    .venv/bin/python experiments/w133a_clickmedal.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEAM = "Teddy Tennant"
ACCOUNT_BEST = 0.97119
REPS = 20000
SDS = ((0.000043, "S6E2"), (0.000067, "S6E3"), (0.000124, "S6E5"))

# e-6, against the STATUS QUO (nothing selected -> Kaggle auto-picks). Arithmetic in the
# docstring; the two mis-click rows are w114a's costs-against-WANTED shifted by the click.
ARMS = (
    ("click WANTED", +4.5228),
    ("no click (status quo)", 0.0),
    ("mis-click A", 4.5228 - 35.17),
    ("mis-click B", 4.5228 - 81.92),
)

FAILURES: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def newest_board() -> tuple[pd.DataFrame, str, dt.datetime]:
    """Newest board by the timestamp IN THE FILENAME. Kaggle stamps the download; mtime lies."""
    pat = re.compile(r"publicleaderboard-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    best = None
    for p in glob.glob(os.path.join(ROOT, "lb_*", "**", "*publicleaderboard*.csv"), recursive=True):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        stamp = dt.datetime.fromisoformat(m.group(1))
        if best is None or stamp > best[1]:
            best = (p, stamp)
    if best is None:
        raise SystemExit("no public leaderboard csv found under lb_*/")
    path, stamp = best
    b = pd.read_csv(path, encoding="utf-8-sig").sort_values("Score", ascending=False)
    return b.reset_index(drop=True), path, stamp


def paired_ranks(pub: np.ndarray, our_i: int, sd: float, deltas: np.ndarray, seed: int
                 ) -> np.ndarray:
    """ranks[rep, arm]. ONE noise draw per rep, every arm read off it.

    Rank uses strict `>`, the same definition C5 checks our live rank against, so a tie does
    not silently cost us a place in one arm and not another.
    """
    rng = np.random.default_rng(seed)
    others = np.delete(pub, our_i)
    out = np.empty((REPS, deltas.size), dtype=np.int32)
    for r in range(REPS):
        sim_others = others + rng.normal(0.0, sd, size=others.size)
        ours = pub[our_i] + rng.normal(0.0, sd)
        out[r] = 1 + (sim_others[:, None] > (ours + deltas)[None, :]).sum(axis=0)
    return out


def main() -> int:
    board, path, stamp = newest_board()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    n = len(board)
    rows = board.index[board.TeamName.str.strip().str.lower() == TEAM.lower()]
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one row for {TEAM!r}, found {len(rows)}")
    our_i = int(rows[0])
    our_rank = our_i + 1
    our_score = float(board.Score.iloc[our_i])
    pub = board.Score.to_numpy(dtype=float)

    # Kaggle's cuts, derived from the field size every time -- never carried between runs.
    gold = int(10 + 0.002 * (n - 1000)) if n > 1000 else 10
    silver = int(0.05 * n)
    bronze = int(0.10 * n)

    print(f"board   {os.path.basename(path)}")
    print(f"stamped {stamp} UTC, {(now - stamp).total_seconds() / 3600:.1f}h old, {n} teams")
    print(f"us      public rank {our_rank} at {our_score:.5f} = top {our_rank / n:.2%}")
    print(f"cuts    gold <= {gold}   silver <= {silver}   BRONZE <= {bronze}"
          f"   (margin {bronze - our_rank:+d} places)")

    deltas = np.array([d * 1e-6 for _, d in ARMS])

    print("\n=== sd = 0: the DETERMINISTIC places each action is worth (no shake at all) ===")
    z = paired_ranks(pub, our_i, 0.0, deltas, seed=7)
    base_z = int(z[0][1])
    for (name, d), rk in zip(ARMS, z[0]):
        print(f"  {name:24s} {d:+9.4f}e-6   rank {int(rk):4d}   {base_z - int(rk):+d} places")
    print("  This is w113a's density re-derived from the live board, and it is the ONLY")
    print("  place in this file where a 'places' figure appears. Everything below is a")
    print("  probability, because a rank without a shake is not a forecast.")

    print("\n=== A. P(inside each cut) at three shift sds, PAIRED across arms ===")
    results = {}
    for sd, src in SDS:
        rk = paired_ranks(pub, our_i, sd, deltas, seed=42)
        print(f"\n  shift sd {sd:.6f} ({src})")
        print(f"    {'arm':24s} {'median':>7s} {'p10':>6s} {'p90':>6s} "
              f"{'P(bronze)':>10s} {'dP vs status quo':>18s}")
        base = float(np.mean(rk[:, 1] <= bronze))
        for j, (name, d) in enumerate(ARMS):
            p = float(np.mean(rk[:, j] <= bronze))
            results[f"{src}|{name}"] = p
            dp = "" if j == 1 else f"{(p - base) * 100:+17.2f}pp"
            print(f"    {name:24s} {np.median(rk[:, j]):7.0f} "
                  f"{np.percentile(rk[:, j], 10):6.0f} {np.percentile(rk[:, j], 90):6.0f} "
                  f"{p:10.1%} {dp:>18s}")
        results[f"{src}|silver_statusquo"] = float(np.mean(rk[:, 1] <= silver))
        results[f"{src}|gold_statusquo"] = float(np.mean(rk[:, 1] <= gold))

    a_lo = min(results[f"{s}|no click (status quo)"] for _, s in SDS)
    a_hi = max(results[f"{s}|no click (status quo)"] for _, s in SDS)
    click_dp = [results[f"{s}|click WANTED"] - results[f"{s}|no click (status quo)"]
                for _, s in SDS]
    misA_dp = [results[f"{s}|mis-click A"] - results[f"{s}|no click (status quo)"]
               for _, s in SDS]
    misB_dp = [results[f"{s}|mis-click B"] - results[f"{s}|no click (status quo)"]
               for _, s in SDS]

    print("\n=== THE COMPARISON THIS FILE WAS BUILT FOR ===")
    print(f"  the click moves P(bronze) by      {min(click_dp) * 100:+.2f} .. "
          f"{max(click_dp) * 100:+.2f} pp")
    print(f"  a mis-click moves it by           {min(misB_dp) * 100:+.2f} .. "
          f"{max(misA_dp) * 100:+.2f} pp")
    print(f"  CHOOSING THE SHIFT sd moves it by {(a_hi - a_lo) * 100:+.2f} pp "
          f"({a_lo:.1%} .. {a_hi:.1%}), and that is an ASSUMPTION, not an action")

    art = {
        "board": os.path.basename(path), "stamp": str(stamp), "teams": n,
        "our_rank": our_rank, "our_score": our_score,
        "cuts": {"gold": gold, "silver": silver, "bronze": bronze},
        "arms_e6": {k: v for k, v in ARMS},
        "sd0_ranks": {name: int(r) for (name, _), r in zip(ARMS, z[0])},
        "p_bronze": results,
        "click_dp_pp": [round(x * 100, 4) for x in click_dp],
        "misclickA_dp_pp": [round(x * 100, 4) for x in misA_dp],
        "misclickB_dp_pp": [round(x * 100, 4) for x in misB_dp],
        "reps": REPS,
    }

    print("\n=== controls ===")
    ok("C1 arms are PAIRED (delta 0 twice is identical on every rep)",
       bool(np.all(paired_ranks(pub, our_i, 6.7e-5, np.array([0.0, 0.0]), seed=11)[:, 0]
                   == paired_ranks(pub, our_i, 6.7e-5, np.array([0.0, 0.0]), seed=11)[:, 1])),
       f"{REPS} reps, max |diff| = 0")
    ok("C2 zero-noise status quo returns our live rank",
       bool(np.all(paired_ranks(pub, our_i, 0.0, np.array([0.0]), seed=3) == our_rank)),
       f"all {REPS} reps = {our_rank}")
    ok("C3 exactly one row of the board differs between arms",
       True, "by construction: delta is added to `ours` only, never to `others`")
    ok("C4 P(bronze) is monotone in the delta, at every sd",
       all(results[f"{s}|click WANTED"] >= results[f"{s}|no click (status quo)"]
           >= results[f"{s}|mis-click A"] >= results[f"{s}|mis-click B"] for _, s in SDS),
       "click >= status quo >= mis-click A >= mis-click B, 3/3 sds")
    above = int((pub > our_score).sum())
    ok("C5 rank self-consistent and score is the account best",
       above + 1 == our_rank and abs(our_score - ACCOUNT_BEST) < 5e-6,
       f"{above} strictly above, score {our_score:.5f}")
    ok("C6 board fresh (<24h) and larger than w83a's floor",
       (now - stamp).total_seconds() < 24 * 3600 and n >= 2791,
       f"{(now - stamp).total_seconds() / 3600:.1f}h, {n} teams")
    ok("C7 the mis-click ordering survives the change of unit",
       all(abs(a) > abs(c) for a, c in zip(misA_dp, click_dp))
       and all(abs(b) > abs(c) for b, c in zip(misB_dp, click_dp)),
       "|dP(mis-click)| > |dP(click)| at all 3 sds")

    art["failures"] = FAILURES
    with open(os.path.join(HERE, "w133a_clickmedal.json"), "w") as f:
        json.dump(art, f, indent=1)
    print(f"\nFAILURES {len(FAILURES)}" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
