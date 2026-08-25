"""w91 (2026-08-25) -- ONE OWNER for parsing the Kaggle submissions `date` column.

WHY THIS EXISTS
---------------
`kaggle competitions submissions -v` prints the timestamp with microseconds
(`2026-08-25 12:40:32.297000`) EXCEPT when the microsecond field happens to be zero, and
then it prints `2026-08-25 12:40:07` with no fractional part at all. That is roughly a
1-in-1000 event per submission and this account crossed it on 2026-08-25: exactly 1 of
141 rows has no fractional seconds.

`pd.to_datetime(series)` with no `format=` infers the format from the FIRST element and
then requires every other element to match it, so one such row turns the whole call into

    ValueError: time data "2026-08-25 12:40:07" doesn't match format "%Y-%m-%d %H:%M:%S.%f"

That reads like a Kaggle API change. It is not; it is one row's missing milliseconds.
`format="ISO8601"` accepts both spellings and is what every caller must use.

⚠ THIS IS NOT COSMETIC. `w50b_autoselect.py` sorts on this column to resolve the
TIEBREAK among files holding the same public score, and the tiebreak is what decides
which two files Kaggle auto-selects for private scoring. The account cannot click a
selection (no browser, w74), so auto-selection IS the final selection. The parser that
orders that tie is on the outcome path.

⛔ DO NOT re-introduce a bare `pd.to_datetime(df.date)` on a submissions frame anywhere.
   `w91b_dateguard.py` scans for it and exits 1. Call this instead.
⛔ DO NOT "fix" a caller by dropping the offending row. The row is a real submission and
   it is one of the ten sent on 2026-08-25.
⛔ DO NOT relax this to `errors="coerce"`. A NaT would sort to one end of the tiebreak
   silently, which is the same failure with no traceback.

The `[:10]` string slice used by `w26g_send.py` (day counting) and `w54a_vetoexpiry.py`
is a DIFFERENT and equally correct idiom -- it never parses the time at all. It is left
alone deliberately; this module is only for callers that need the time.
"""
from __future__ import annotations

import pandas as pd

#: The one format string. Both spellings the CLI emits are ISO 8601.
FORMAT = "ISO8601"


def parse_sub_dates(s) -> pd.Series:
    """Parse the submissions `date` column. Raises rather than coercing.

    >>> parse_sub_dates(pd.Series(["2026-08-25 12:40:32.297000",
    ...                            "2026-08-25 12:40:07"])).is_monotonic_decreasing
    True
    """
    out = pd.to_datetime(pd.Series(s).astype(str), format=FORMAT)
    n_bad = int(out.isna().sum())
    if n_bad:
        bad = pd.Series(s).astype(str)[out.isna()].tolist()[:5]
        raise ValueError(f"{n_bad} submission timestamp(s) did not parse: {bad}")
    return out
