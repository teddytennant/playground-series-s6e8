"""Structural screen: which public dataset is actually the source of S6E8?

`RESEARCH.md` names `jayjoshi37/smartphone-usage-and-addiction-prediction` (7,500 rows) as
the original and records both routes into it as closed. This script does not re-test those
routes. It tests the prior question nobody asked: **is that file even the source?**

The screen is a set of structural signatures that a tabular generator preserves. The
load-bearing one is the accounting identity the competition frame satisfies exactly:

    daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours

Zero violations in 421,427 complete competition rows. A generative model trained on a
source that violates a constraint in the majority of its rows does not invent it.

    .venv/bin/python experiments/w15d_screen_source.py <dir-of-candidate-datasets>
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA  # noqa: E402

# canonical column names in the competition frame -> tolerant aliases
WANT = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
        "work_study_hours", "sleep_hours", "notifications_per_day",
        "app_opens_per_day", "weekend_screen_time"]


def norm(c):
    return c.strip().lower().replace(" ", "_").replace("-", "_")


def signature(df, name):
    df = df.rename(columns={c: norm(c) for c in df.columns})
    have = [c for c in WANT if c in df.columns]
    out = dict(name=name, rows=len(df), cols=len(df.columns), matched=len(have))
    if len(have) < 4:
        return out, None
    need = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours"]
    if all(c in df.columns for c in need):
        d = pd.to_numeric(df["daily_screen_time_hours"], errors="coerce")
        s = sum(pd.to_numeric(df[c], errors="coerce") for c in need[1:])
        m = d.notna() & s.notna()
        if m.sum():
            slack = (d[m] - s[m])
            out["n_complete"] = int(m.sum())
            out["viol_frac"] = float((slack < -1e-9).mean())
            out["slack_min"] = float(slack.min())
            out["slack_mean"] = float(slack.mean())
    for c in have:
        v = pd.to_numeric(df[c], errors="coerce")
        out[f"{c}__min"] = float(v.min())
        out[f"{c}__max"] = float(v.max())
        out[f"{c}__mean"] = float(v.mean())
    return out, df


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/w15d_cand"
    rows = []

    comp = pd.read_csv(os.path.join(DATA, "train.csv"))
    r, _ = signature(comp, "*** COMPETITION train ***")
    rows.append(r)
    orig = pd.read_csv(os.path.join(DATA, "orig",
                                    "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"))
    r, _ = signature(orig, "*** RESEARCH.md's original (jayjoshi37) ***")
    rows.append(r)

    for f in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f}: {e}")
            continue
        if len(df) < 50:
            continue
        r, _ = signature(df, os.path.relpath(f, root))
        rows.append(r)

    df = pd.DataFrame(rows)
    cols = ["name", "rows", "cols", "matched", "n_complete", "viol_frac", "slack_min", "slack_mean"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print("\n=== the accounting identity  daily >= social + gaming + work_study ===")
    print(df[cols].to_string(index=False))

    print("\n=== marginal ranges of the five budget columns (min .. max, mean) ===")
    key = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
           "work_study_hours", "weekend_screen_time"]
    for _, r in df.iterrows():
        if r.get("matched", 0) < 4:
            continue
        bits = []
        for c in key:
            if f"{c}__min" in r and pd.notna(r.get(f"{c}__min")):
                bits.append(f"{c.split('_')[0][:5]} {r[f'{c}__min']:5.2f}-{r[f'{c}__max']:6.2f}"
                            f"({r[f'{c}__mean']:5.2f})")
        print(f"{r['name'][:60]:62s} " + "  ".join(bits))

    out = os.path.join("experiments", "w15d_screen_source.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
