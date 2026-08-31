"""w91b — the submissions timestamp parser is ONE OWNER, and the bug it fixes is real.

WHAT BROKE (2026-08-25, found by running the instrument, not by reading it).
`w50b_autoselect.py` — the module that enumerates which files Kaggle will AUTO-SELECT for
private scoring, which is this account's only selection mechanism because it cannot click
(w74) — died with

    ValueError: time data "2026-08-25 12:40:07" doesn't match format "%Y-%m-%d %H:%M:%S.%f"

One of the ten files sent on 2026-08-25 landed on a whole second, so the CLI printed its
timestamp with NO fractional part. `pd.to_datetime` with no `format=` infers the format
from the FIRST row and then demands every other row match it. 1 row in 141 is enough.

⚠ IT WAS NOT ONE MODULE. Seven call sites parsed that column with a bare `pd.to_datetime`,
including `w28a_cvlb_refresh`, `w25a_cvlb_full` and `w75a_erarefresh` — the three that
refresh the CV->LB calibration and the era slope. All seven were broken from 12:40Z on, and
none of the 33 standing checks touches any of them, so the suite stayed green through it.
This is w90's lesson arriving from the other side: there the fix enumerated the cases in
front of it; here the SUITE enumerated the modules in front of it.

SIX CHECKS, and the two that matter are the CONTROLS.
  G1  `w91a_subdate` exists, exports `parse_sub_dates`, pins FORMAT = "ISO8601".
  G2  source scan: no module that reads the submissions frame parses `date` with a bare
      `pd.to_datetime`. Scoped by "does this module fetch or read the submissions list",
      so `w47a_extrap` — which parses a date-ONLY `day` column out of a local CSV — is out
      of scope by the rule, not by a name exemption.
  G3  CONTROL-: a planted module with the bare call IS flagged by the G2 scanner.
      Written against a find we already have, per w89's lesson: the scanner is checked
      against the exact seven lines it was built to catch, from git.
  G3b every module importing the helper imports it BEFORE its first use — the w91
      patch itself got this wrong once, and neither a syntax check nor G2 can see it.
  G4  LIVE: `parse_sub_dates` parses every row of the live column, no NaT.
  G5  CONTROL-: the bare `pd.to_datetime` still RAISES on that same live column.
      Exercised, not asserted. The day G5 stops raising, the CLI changed its output and
      this whole file can be retired — report that, do not silently pass.

⛔ DO NOT satisfy G2 by adding the offending module to an exemption list. The rule is
   "does it read the submissions frame", and a module that does must use the helper.
⛔ DO NOT satisfy G5 by narrowing the live column to the rows that agree. The mixed
   column IS the regression fixture and the board is append-only, so it cannot regress.

    .venv/bin/python experiments/w91b_dateguard.py      # 0 = intact, 1 = broken
"""
from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_list   # noqa: E402  paginated submission reads
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
COMP = "playground-series-s6e8"

from w91a_subdate import FORMAT, parse_sub_dates  # noqa: E402

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  ⛔ {msg}")


# The seven lines that were actually broken, recovered from git rather than retyped.
# G3 replays them through the scanner; a scanner that misses any of these is broken.
KNOWN_BAD = [
    ('w15j_tiebreak.py', 'd["date"] = pd.to_datetime(d["date"])'),
    ('w15i_cvlb.py', 's["date"] = pd.to_datetime(s["date"])'),
    ('w15j_cvlb.py', 'sub["date"] = pd.to_datetime(sub["date"])'),
    ('w28a_cvlb_refresh.py', 'sub["date"] = pd.to_datetime(sub["date"])'),
    ('w25a_cvlb_full.py', 'sub["date"] = pd.to_datetime(sub["date"])'),
    ('w75a_erarefresh.py', 'sub["date"] = pd.to_datetime(sub["date"])'),
    ('w50b_autoselect.py', 'df["date"] = pd.to_datetime(df.date)'),
]

# A module is IN SCOPE iff it obtains the submissions list. Both routes count: the CLI
# call, and reading a stored copy of its output.
SCOPE_MARKERS = ('"submissions"', "'submissions'", "competitions submissions", "subs.csv")


def in_scope(src: str) -> bool:
    return any(m in src for m in SCOPE_MARKERS)


def bare_date_parses(src: str) -> list[tuple[int, str]]:
    """AST, not grep: find pd.to_datetime(...) calls with no format= whose argument
    mentions `date`. A regex over the line would miss a wrapped call and would trip on
    the word `pd.to_datetime` inside a docstring — this file is full of both."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    hits = []
    lines = src.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name != "to_datetime":
            continue
        if any(k.arg == "format" for k in node.keywords):
            continue
        arg_src = ast.unparse(node.args[0]) if node.args else ""
        if "date" not in arg_src.lower():
            continue
        hits.append((node.lineno, lines[node.lineno - 1].strip()))
    return hits


def scan(files) -> list[tuple[str, int, str]]:
    out = []
    for p in files:
        src = open(p).read()
        if not in_scope(src):
            continue
        for ln, text in bare_date_parses(src):
            out.append((os.path.relpath(p, ROOT), ln, text))
    return out


print("=" * 92)
print("w91b — is the submissions timestamp parsed by one owner, and is the bug still real?")
print("=" * 92)

# --- G1 ------------------------------------------------------------------------------
print("\nG1  the helper")
if FORMAT != "ISO8601":
    fail(f'w91a.FORMAT is {FORMAT!r}, not "ISO8601" — the mixed column will not parse')
else:
    print('  ✅ w91a_subdate.FORMAT == "ISO8601"')
probe = pd.Series(["2026-08-25 12:40:32.297000", "2026-08-25 12:40:07"])
try:
    got = parse_sub_dates(probe)
    assert len(got) == 2 and got.notna().all()
    print("  ✅ parse_sub_dates handles both spellings in one column")
except Exception as e:  # pragma: no cover - this is the thing under test
    fail(f"parse_sub_dates failed on the mixed probe: {type(e).__name__}: {e}")

# --- G2 ------------------------------------------------------------------------------
print("\nG2  source scan over the tree")
PY = [os.path.join(d, f)
      for d in (os.path.join(ROOT, "experiments"), os.path.join(ROOT, "agent"))
      for f in sorted(os.listdir(d)) if f.endswith(".py")]
scoped = [p for p in PY if in_scope(open(p).read())]
hits = scan(PY)
print(f"  {len(PY)} modules scanned, {len(scoped)} of them read the submissions frame")
if hits:
    for rel, ln, text in hits:
        fail(f"{rel}:{ln} parses a date column with a bare pd.to_datetime — {text}")
    print("     Use `from w91a_subdate import parse_sub_dates`. Do not add an exemption.")
else:
    print("  ✅ 0 bare date parses among the in-scope modules")

# --- G3 CONTROL- ---------------------------------------------------------------------
print("\nG3  CONTROL- the scanner catches every line it was built to catch")
missed = []
with tempfile.TemporaryDirectory() as td:
    for i, (orig, line) in enumerate(KNOWN_BAD):
        p = os.path.join(td, f"planted_{i}.py")
        with open(p, "w") as fh:
            fh.write('import pandas as pd\nout = ["submissions"]\n'
                     'd = sub = df = pd.DataFrame({"date": []})\n' + line + "\n")
        if not scan([p]):
            missed.append(orig)
if missed:
    fail(f"the scanner MISSES {len(missed)}/{len(KNOWN_BAD)} known-bad lines: {missed}")
else:
    print(f"  ✅ all {len(KNOWN_BAD)} historical bad lines are flagged when replanted")
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "clean.py")
    with open(p, "w") as fh:
        fh.write('import pandas as pd\nout = ["submissions"]\n'
                 'd = pd.to_datetime(pd.Series([]), format="ISO8601")\n')
    if scan([p]):
        fail("the scanner flags a CORRECT call — it is stuck failing")
    else:
        print("  ✅ CONTROL+ a correct format=-carrying call is not flagged")

# --- G4 / G5 live --------------------------------------------------------------------
# --- G3b -------------------------------------------------------------------------------
# Found the hard way: the w91 patch inserted its import after the LAST top-level import,
# and `w25a_cvlb_full.py` keeps a `from scipy.stats import ...` forty lines BELOW its
# first use of the column. The import landed at line 80 and the call was at line 34 —
# a NameError that no syntax check and no source scan can see.
print("\nG3b every importer of parse_sub_dates imports it BEFORE it uses it")
bad_order, n_ordered = [], 0
for p_ in PY:
    src_ = open(p_).read()
    if "parse_sub_dates" not in src_:
        continue
    try:
        tree_ = ast.parse(src_)
    except SyntaxError:
        continue
    if any(isinstance(n, ast.FunctionDef) and n.name == "parse_sub_dates"
           for n in ast.walk(tree_)):
        continue                      # w91a itself defines it; it has nothing to import
    imp_ln, use_lns = None, []
    for node in ast.walk(tree_):
        if isinstance(node, ast.ImportFrom) and node.module == "w91a_subdate":
            if any(a.name == "parse_sub_dates" for a in node.names):
                imp_ln = node.lineno
        elif isinstance(node, ast.Name) and node.id == "parse_sub_dates" \
                and isinstance(node.ctx, ast.Load):
            use_lns.append(node.lineno)
    rel_ = os.path.relpath(p_, ROOT)
    # w92: the trigger is a LOAD of the name, not a mention of it. `w92a_smokerun` names
    # `parse_sub_dates` only in prose and in the string it matches module names against, and
    # a module with no load cannot NameError on it. Flagging it was a false positive, and a
    # guard that cries wolf on documentation is one somebody eventually silences.
    if not use_lns:
        continue
    if imp_ln is None:
        bad_order.append((rel_, "uses parse_sub_dates without importing it"))
    elif min(use_lns) < imp_ln:
        bad_order.append((rel_, f"import at line {imp_ln}, first use at line {min(use_lns)}"))
    n_ordered += 1
if bad_order:
    for rel_, why in bad_order:
        fail(f"{rel_}: {why}")
else:
    print(f"  ✅ {n_ordered} module(s) load parse_sub_dates, all import it first")
with tempfile.TemporaryDirectory() as td:
    p_ = os.path.join(td, "late_import.py")
    with open(p_, "w") as fh:
        fh.write("import pandas as pd\n"
                 "x = parse_sub_dates(pd.Series([]))\n"
                 "from w91a_subdate import parse_sub_dates\n")
    tree_ = ast.parse(open(p_).read())
    imp_ln = next(n.lineno for n in ast.walk(tree_)
                  if isinstance(n, ast.ImportFrom) and n.module == "w91a_subdate")
    use_ln = min(n.lineno for n in ast.walk(tree_)
                 if isinstance(n, ast.Name) and n.id == "parse_sub_dates"
                 and isinstance(n.ctx, ast.Load))
    if use_ln < imp_ln:
        print(f"  ✅ CONTROL- a planted late import is detected (use {use_ln} < import {imp_ln})")
    else:
        fail("the G3b ordering test does not detect a planted late import")


print("\nG4  LIVE the real column parses")
# w140: paginated -- the old capped read silently stopped at 200 rows.
live = pd.DataFrame(kaggle_list.submissions(COMP))
assert len(live) >= 141, f"page-size truncation or auth failure: {len(live)} rows"
col = live["date"].astype(str)
with_frac = int(col.str.contains(r"\.\d").sum())
without = len(col) - with_frac
print(f"  {len(col)} rows: {with_frac} with fractional seconds, {without} without")
try:
    t = parse_sub_dates(col)
    print(f"  ✅ all {len(t)} parsed, {t.min()} .. {t.max()}")
except Exception as e:
    fail(f"parse_sub_dates failed on the LIVE column: {type(e).__name__}: {e}")

print("\nG5  CONTROL- the bare call still raises on that same column")
if without == 0:
    fail("the live column no longer mixes spellings — G5 cannot be exercised. "
         "Check whether the CLI changed its output before trusting a green G2.")
else:
    try:
        pd.to_datetime(col)
        fail("a bare pd.to_datetime NO LONGER raises on the mixed live column. "
             "pandas or the CLI changed; re-read w91a before retiring anything.")
    except ValueError as e:
        print(f"  ✅ bare pd.to_datetime raises, as it must: {str(e).splitlines()[0][:88]}")

print(f"\nFAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
