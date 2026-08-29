"""w113 — where the MEDAL CUTS actually fall on this board, and which way we are drifting.

WHY THIS EXISTS. `LEADERBOARD.md` has tracked our public position as a PERCENTILE ("top
5.75%") since w92, and `w83a_reproject.py` reports P(top 5%) / P(top 10%). Neither prints the
number Kaggle actually awards a medal on, which is a RANK derived from the field size:

    gold   <= 10 + 0.2% * (N - 1000)        silver <= 5% * N        bronze <= 10% * N

At N = 3,241 that is 14 / 162 / 324. A percentile and a medal cut agree only for silver and
bronze and never for gold, and the field here grew by 265 teams in three days, so the cut
MOVES underneath a quoted rank. Derive it from the board every time; do not carry it.

⚠ THE DENSITY FIGURE IS THE EASY THING TO GET WRONG, AND THIS RUN GOT IT WRONG FIRST.
`(score > 0.97110) & (score <= 0.97120)` is a **1e-4** window, not the 1e-5 it reads like, so
dividing its count by 10 overstates the local density by 10x — and the overstatement flatters
exactly the number a run most wants to be big (the rank value of a small AUC gain). The real
figure near us is ~1 team per 1e-6, so the +4.5228e-6 selection click is worth about FIVE
places, not fifty. Windows below are stated as half-widths and divided by 2w, so the arithmetic
cannot silently drop a decade again.

⛔ ADOPTS NOTHING. Reports. The final selection is on CV and is settled (SELECT_THESE.md).

    .venv/bin/python experiments/w113a_medalcut.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_SCORE = 0.97119                      # account best public; C2 re-derives it from the board
CLICK_E6 = 4.5228                        # w74a expected private AUC of the selection click

# public rank, field size, and the stamp on the board it was read from. Appended to, never edited.
DRIFT = [
    ("2026-08-25", 140, 2881),
    ("2026-08-26", 171, 2976),
    ("2026-08-29", 249, 3241),
]


def newest_board():
    """The board whose FILENAME stamp is latest. Kaggle stamps the download; mtime does not."""
    best = None
    for p in glob.glob(os.path.join(ROOT, "lb_*", "**", "*publicleaderboard*.csv"), recursive=True):
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", os.path.basename(p))
        if m and (best is None or m.group(1) > best[0]):
            best = (m.group(1), p)
    if best is None:
        raise SystemExit("no downloaded board found under lb_*/")
    return best


def cuts(n):
    return {"gold": int(10 + 0.002 * (n - 1000)), "silver": int(0.05 * n), "bronze": int(0.10 * n)}


stamp, path = newest_board()
lb = pd.read_csv(path)
s = np.sort(lb.Score.to_numpy())[::-1]
N = len(s)
rank = int((s > OUR_SCORE).sum()) + 1
c = cuts(N)

print("=" * 90)
print(f"w113  MEDAL CUTS ON THE LIVE BOARD — {os.path.basename(path)}")
print("=" * 90)
print(f"\n  {N} teams, stamped {stamp}Z.  us: public {OUR_SCORE:.5f}  rank {rank} "
      f"= top {rank / N * 100:.2f}%")
print(f"\n  {'medal':8s} {'rank cut':>9s} {'public score needed':>20s} {'places away':>12s}")
for k in ("gold", "silver", "bronze"):
    r = c[k]
    need = s[r - 1]
    away = rank - r
    where = f"{away:+d} (INSIDE)" if away <= 0 else f"{away:+d}"
    print(f"  {k:8s} {r:9d} {need:20.5f} {where:>12s}")

print("\n  --- local density: how many teams a small AUC gain actually passes ---")
dens = []
for w in (2e-5, 4e-5, 1e-4):
    cnt = int(((s > OUR_SCORE - w) & (s <= OUR_SCORE + w)).sum())
    d = cnt / (2 * w) * 1e-6
    dens.append(d)
    print(f"  half-width {w:.0e}: {cnt:4d} teams over a {2*w:.0e} span -> {d:5.2f} teams per 1e-6")
d_mid = float(np.median(dens))
print(f"  median {d_mid:.2f} teams per 1e-6  ->  the {CLICK_E6:.4f}e-6 click is worth about "
      f"{CLICK_E6 * d_mid:.0f} places")
print("  ⚠ the click's value is NOT this number. It is not repeating the Rogii failure;")
print("    the rank equivalent is here only so nobody reads '4.5e-6' as 'negligible'.")

print("\n  --- drift (public; the field grows and the cut moves with it) ---")
print(f"  {'date':12s} {'rank':>6s} {'field':>7s} {'pct':>7s} {'bronze cut':>11s} {'margin':>8s}")
for d, r, n in DRIFT:
    b = cuts(n)["bronze"]
    print(f"  {d:12s} {r:6d} {n:7d} {r / n * 100:6.2f}% {b:11d} {b - r:+8d}")

fail = []
age_h = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timestamp(stamp)).total_seconds() / 3600
if age_h >= 24:
    fail.append(f"C1 board is {stamp}Z, {age_h:.1f}h old — re-download before quoting a rank")
if N < DRIFT[-1][2]:
    fail.append(f"C2 board has {N} teams, fewer than the {DRIFT[-1][2]} last recorded — truncated?")
if abs(float(s[rank - 1]) - OUR_SCORE) > 1e-9:
    fail.append(f"C3 rank {rank} does not sit at {OUR_SCORE}; the board has {s[rank-1]} there")
if max(dens) / min(dens) > 4:
    fail.append(f"C4 density varies {max(dens)/min(dens):.1f}x across windows — no single figure")

print("\n  --- controls ---")
for f in fail:
    print(f"  [FAIL] {f}")
if not fail:
    print(f"  [PASS] C1 board fresh ({age_h:.1f}h)   C2 field not shrinking   C3 rank sits at our score   "
          "C4 density stable across windows")
sys.exit(1 if fail else 0)
