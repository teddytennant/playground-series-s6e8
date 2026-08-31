"""w132a -- ROW 3 (CatBoost), 16th handing. WHAT THE CONTROL'S OWN UNCERTAINTY IS.

Row 3's +10.04e-6/member is not only a price. It is the CONTROL quoted verbatim in rows 1, 7
and 9 -- "against the same-process base104 CatBoost control at +10.04e-6/member that
reproduced w123 to +0.0000e-6" -- and each of those cells then declares its own arms
"sign-flipping / none distinguishable from zero" against it.

That sentence certifies REPRODUCIBILITY. It says nothing about DISCRIMINATING POWER, and the
control's own separation from zero, at the per-member scope the arms are judged at, has never
been printed anywhere: not in a cell, not in a journal entry, not in any of the five run
artefacts. This file prints it, and checks that publishing it moves no verdict.

Reads only artefacts already on disk. Fits nothing, so it is deterministic and fast.

    .venv/bin/python experiments/w132a_row3.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The three split lines, frozen as literals out of w124a_run.log (w115: a control anchored to
# a file that a later run may rewrite is not a control). Verified against the log by A0.
W124_SPLITS = {
    "base":       (0.969801, 0.969994, 0.970096),
    "+xgb_only":  (0.969866, 0.970083, 0.970184),
    "+xgb_dedup": (0.969867, 0.970085, 0.970182),
    "+lgb_only":  (0.969827, 0.970026, 0.970134),
    "+cat_only":  (0.969864, 0.970078, 0.970190),
}
W124_N = {"+xgb_only": 12, "+xgb_dedup": 11, "+lgb_only": 8, "+cat_only": 8}

FAILURES = []


def fail(tag, msg):
    FAILURES.append(f"{tag}: {msg}")
    print(f"  !! FAIL {tag}: {msg}")


def sd(xs):
    n = len(xs)
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def load(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


print("=" * 78)
print("w132a  ROW 3 -- THE CONTROL'S OWN UNCERTAINTY, AT THE SCOPE IT IS QUOTED IN")
print("=" * 78)

# ---------------------------------------------------------------- A0  the log still says this
print("\nA0  the frozen split lines still match w124a_run.log")
logp = os.path.join(HERE, "w124a_run.log")
txt = open(logp).read()
for k, vals in W124_SPLITS.items():
    for v in vals:
        tok = f"{k}={v:.6f}" if k != "base" else f"base={v:.6f}"
        if tok not in txt:
            fail("A0", f"{tok!r} not in w124a_run.log")
print(f"    all {sum(len(v) for v in W124_SPLITS.values())} literals present in w124a_run.log")

# ---------------------------------------------------- A  reproduce the published group numbers
print("\nA   REPRODUCE THE PUBLISHED GROUP NUMBERS from the split lines")
base = W124_SPLITS["base"]
arms = {}
for k in ("+xgb_only", "+xgb_dedup", "+lgb_only", "+cat_only"):
    d = [a - b for a, b in zip(W124_SPLITS[k], base)]
    n = W124_N[k]
    arms[k] = {
        "n": n,
        "per_split_e6": [x * 1e6 for x in d],
        "delta_group_e6": sum(d) / 3 * 1e6,
        "sd_group_e6": sd(d) * 1e6,
        "per_member_e6": sum(d) / 3 / n * 1e6,
        "sd_member_e6": sd(d) / n * 1e6,
    }
    a = arms[k]
    a["t"] = a["delta_group_e6"] / a["sd_group_e6"]
    a["t_member"] = a["per_member_e6"] / a["sd_member_e6"]

print(f"    {'arm':<12} {'n':>3}  {'delta_grp':>10} {'sd_grp':>8}   {'/member':>9} {'sd/member':>9}   {'t':>6}")
for k, a in arms.items():
    print(f"    {k:<12} {a['n']:>3}  {a['delta_group_e6']:>+9.2f}e-6 {a['sd_group_e6']:>7.2f}   "
          f"{a['per_member_e6']:>+8.2f}e-6 {a['sd_member_e6']:>8.2f}   {a['t']:>6.2f}")

# The log prints its split lines at 6 dp, so everything re-derived from them is quantised:
# +/-0.5e-6 per AUC, +/-1.0e-6 per paired difference, ~0.6e-6 on a 3-split mean and ~1.0e-6 on
# a 3-split sd. Tolerances below are that budget, not a fudge -- w123a's full-precision
# +cat_only sd is 15.6883e-6 against 15.82e-6 here, inside it.
pub = {"+xgb_only": (81, 13, 6.74), "+xgb_dedup": (81, 13, 7.38),
       "+lgb_only": (32, 6, 4.04), "+cat_only": (80, 16, 10.04)}
TOL_D, TOL_SD = 1.0, 1.5
for k, (pd_, psd, ppm) in pub.items():
    a = arms[k]
    tol_pm = 1.0 / a["n"] + 0.01
    if abs(a["delta_group_e6"] - pd_) > TOL_D:
        fail("A", f"{k} group delta {a['delta_group_e6']:.2f} vs published {pd_} (tol {TOL_D})")
    if abs(a["sd_group_e6"] - psd) > TOL_SD:
        fail("A", f"{k} group sd {a['sd_group_e6']:.2f} vs published {psd} (tol {TOL_SD})")
    if abs(a["per_member_e6"] - ppm) > tol_pm:
        fail("A", f"{k} per-member {a['per_member_e6']:.2f} vs published {ppm} (tol {tol_pm:.3f})")
cf = r3_pre = None
print(f"    all four arms reproduce the published delta/sd/price inside the 6-dp quantisation")
print(f"    budget ({TOL_D}e-6 on the delta, {TOL_SD}e-6 on the sd, 1/n e-6 on the price)")

# --------------------------------------------------------------------- R1  the stored sd scope
print("\nR1  IS THE STORED `sd` AT GROUP SCOPE? (w123a_row3.json, w127a_row7.json)")
r3 = load("w123a_row3.json")
rows = []
for name, rec in r3["paired"].items():
    n = rec.get("n")
    if n is None and rec["per_member"]:
        n = round(rec["delta"] / rec["per_member"])
    rows.append((f"w123a {name}", n, rec["delta"], rec["sd"], rec["per_member"]))
r7 = load("w127a_row7.json")
for name, rec in r7["arms"].items():
    rows.append((f"w127a {name.strip()}", rec["n"], rec["delta"], rec["sd"], rec["per_member"]))

print(f"    {'artefact / arm':<42} {'n':>3} {'sd/delta':>9} {'sd/per_mem':>11} {'ratio/n':>8}")
for nm, n, d, s, pm in rows:
    r_group = s / d if d else float("nan")
    r_mem = s / pm if pm else float("nan")
    ratio = r_mem / r_group if r_group else float("nan")
    print(f"    {nm:<42} {n:>3} {r_group:>9.5f} {r_mem:>11.5f} {ratio:>8.4f}")
    if n and abs(ratio - n) > 1e-6:
        fail("R1", f"{nm}: sd/per_member is not n x sd/delta (ratio {ratio:.4f}, n={n})")
print("    ⟹ every stored `sd` is at GROUP scope; `per_member` beside it is at MEMBER scope.")
print("       The two keys sit in one dict with no scope key. Ratio is exactly n, per arm.")

# -------------------------------------------------------- R2  the number nobody has published
print("\nR2  THE CONTROL'S PER-MEMBER UNCERTAINTY -- never published anywhere")
cc = r3["paired"]["+cat_only"]
n_cat = 8
sd_mem = cc["sd"] / n_cat
t_cat = cc["per_member"] / sd_mem
print(f"    group delta        {cc['delta']*1e6:+9.4f}e-6   over n={n_cat} CatBoosts")
print(f"    group sd           {cc['sd']*1e6:>9.4f}e-6   <- the ONLY uncertainty ever printed")
print(f"    per-member price   {cc['per_member']*1e6:+9.4f}e-6")
print(f"    PER-MEMBER sd      {sd_mem*1e6:>9.4f}e-6   <- NEW")
print(f"    t                  {t_cat:>9.4f}")
if abs(sd_mem * 1e6 - 1.96) > 0.02:
    fail("R2", f"per-member sd {sd_mem*1e6:.4f}e-6 outside the registered 1.96 +/- 0.02")
if abs(t_cat - 5.12) > 0.05:
    fail("R2", f"t {t_cat:.4f} outside the registered 5.12 +/- 0.05")
if abs(t_cat) < 3.0:
    fail("R2", f"FALSIFIER FIRED: control |t| {abs(t_cat):.2f} < 3.0")
print(f"    ⟹ the control is {abs(t_cat):.2f} sd from zero at the scope it is quoted in.")

# ------------------------------------------------------------------- R3  t is scale-invariant
print("\nR3  t IS SCALE-INVARIANT, so no published verdict can move")
worst = 0.0
checked = 0
for nm, n, d, s, pm in rows:
    if not (s and d and pm and n):
        continue
    tg, tm = d / s, pm / (s / n)
    worst = max(worst, abs(tg - tm))
    checked += 1
for k, a in arms.items():
    worst = max(worst, abs(a["t"] - a["t_member"]))
    checked += 1
print(f"    max |t_group - t_member| over {checked} arms: {worst:.3e}")
if worst > 1e-9:
    fail("R3", f"FALSIFIER FIRED: t is not scale-invariant (max gap {worst:.3e}) -- VALUE finding")
print("    ⟹ REPORTING finding, not a value finding. NOTHING re-opens.")

# --------------------------------------------- R4  one t-scale, classified by the artefacts
print("\nR4  ONE t-SCALE. The classifier is the artefacts\' OWN `sign` field, not my judgement.")
scale = [("row 3  CONTROL 8 CatBoosts", n_cat, cc["per_member"] * 1e6, sd_mem * 1e6, "consistent")]
r1 = load("w131a_row1.json")
ac = r1["arm_c"]
scale.append(("row 1  origmodel enrolment", 1, ac["mean_e6"], ac["sd_e6"],
              "SIGN FLIPS" if ac["sign_flipping"] else "consistent"))
for name, rec in r7["arms"].items():
    n = rec["n"]
    scale.append((f"row 7  {name.strip()[:28]}", n, rec["per_member"] * 1e6,
                  rec["sd"] / n * 1e6, rec["sign"]))
r9 = load("w128a_row9.json")
for name, rec in r9["arms"].items():
    scale.append((f"row 9  {name.strip()[:28]}", 1, rec["delta"] * 1e6,
                  rec["sd"] * 1e6, rec["sign"]))

print(f"    {'arm':<40} {'n':>2} {'/member':>10} {'sd/member':>10} {'|t|':>5}  artefact says")
ts = {}
for nm, n, pm, sdm, sign in scale:
    t = abs(pm / sdm) if sdm else float("nan")
    ts[nm] = (t, sign)
    print(f"    {nm:<40} {n:>2} {pm:>+9.3f}e-6 {sdm:>9.3f} {t:>5.2f}  {sign}")

flips = {nm: t for nm, (t, sg) in ts.items() if sg == "SIGN FLIPS"}
cons = {nm: t for nm, (t, sg) in ts.items() if sg != "SIGN FLIPS"}
if t_cat < 3.0:
    fail("R4", "control |t| below 3")
bad = [nm for nm, t in flips.items() if t >= 2.0]
if bad:
    fail("R4", f"FALSIFIER FIRED: a SIGN FLIPS arm has |t| >= 2: {bad}")
if max(flips.values()) >= t_cat:
    fail("R4", f"FALSIFIER FIRED: a SIGN FLIPS arm |t| ({max(flips.values()):.2f}) >= control ({t_cat:.2f})")
print(f"    control |t| = {t_cat:.2f};  every SIGN FLIPS arm |t| <= {max(flips.values()):.2f}")
print("    ⟹ R4 holds: the yardstick separates from zero, and nothing it certifies as null does.")

# ------------------------------------------- R6  what `consistent` is actually worth, unregistered
print("\nR6  BUT `consistent` IS NOT A SIGNIFICANCE TEST -- and it is the only one this")
print("    column has ever used. Sign-consistency over 3 splits, under a symmetric null:")
print(f"    P = 2 * (1/2)^3 = {2 * 0.5 ** 3:.3f}, i.e. a {2*0.5**3:.0%} false-positive rate by construction.")
weak = sorted(((t, nm) for nm, t in cons.items() if t < 3.0))
print(f"    {len(cons)} arm(s) are labelled `consistent`; {len(weak)} of them have |t| < 3:")
for t, nm in weak:
    print(f"       |t| = {t:>4.2f}   {nm}")
perm = [(t, nm) for nm, t in cons.items() if "PERMUTED" in nm]
if not perm:
    fail("R6", "row 9's permuted null is not in the scale -- the demonstration is vacuous")
else:
    pt, pnm = perm[0]
    print(f"    🎯 THE LIVE ONE: {pnm} is labelled `consistent` at |t| = {pt:.2f}.")
    print("       It is a PERMUTED CONTROL. By construction it is a null, and the label passes it.")
    if pt >= 3.0:
        fail("R6", f"the permuted null has |t| {pt:.2f} >= 3 -- the demonstration does not hold")
print("    ⟹ `consistent` separates NOTHING at the 3-sigma level. Every closure in this table")
print("       rests on sign-consistency; not one of them has ever carried a t.")

# ------------------------------------------------------------------ R5  where the sd is stored
print("\nR5  AUDIT -- every site that stores or prints an uncertainty beside a per-member price")
sites = []

# (a) the machine-readable artefacts, which is where a later run actually reads from
for nm in ("w123a_row3.json", "w124a_row4.json", "w127a_row7.json",
           "w128a_row9.json", "w131a_row1.json"):
    flat = json.dumps(load(nm))
    has_bare_sd = '"sd"' in flat
    has_pm = '"per_member"' in flat
    scoped = any(k in flat for k in ('"sd_group"', '"sd_member"', '"sd_per_member"', '"sd_scope"'))
    if has_bare_sd and has_pm and not scoped:
        sites.append((nm, "json", '"sd" + "per_member"', False,
                      "group-scope sd in the same dict as a member-scope price, no scope key"))
    elif has_bare_sd or has_pm:
        sites.append((nm, "json", '"sd"/"per_member"', True,
                      "scoped, or only one of the two present"))

# (b) the prose: RESEARCH.md lines carrying an uncertainty AND a per-member magnitude
res = open(os.path.join(ROOT, "RESEARCH.md")).read().splitlines()
unc = re.compile(r"(±|\+/-)")
for i, ln in enumerate(res, 1):
    if not (unc.search(ln) and ("/member" in ln or "per member" in ln)):
        continue
    for m in unc.finditer(ln):
        w = ln[max(0, m.start() - 100): m.start() + 100]
        labelled = any(t in w.lower() for t in ("group delta", "group scope", "per member", "per-member"))
        sites.append(("RESEARCH.md", str(i), m.group(0), labelled, w.strip()[:96]))

unl = [x for x in sites if not x[3]]
print(f"    {len(sites)} site(s) examined; {len(unl)} carry an unscoped uncertainty")
for src, ln, tok, lab, ctx in sites:
    print(f"    [{'ok  ' if lab else 'BARE'}] {src:<18} {ln:<5} {tok:<24} {ctx}")
if not unl:
    fail("R5", "FALSIFIER FIRED: zero unscoped sites -- #61 has no defect to register")
print("    ⟹ the ANGLE INDEX cell and the w124 family table DO name their scope (`on the group")
print("       delta`, a `paired delta` column header). The JSON artefacts every later run reads")
print("       do NOT, and that is where the ratio is off by n.")

# ------------------------------------------------------------------------------------- verdict
out = {
    "run": "w132", "row": 3, "handings": 16,
    "w124_arms": arms,
    "control": {"n": n_cat, "delta_group": cc["delta"], "sd_group": cc["sd"],
                "per_member": cc["per_member"], "sd_per_member": sd_mem, "t": t_cat},
    "t_scale": {nm: {"t": t, "sign": sg} for nm, (t, sg) in ts.items()},
    "sign_consistency_fpr": 0.25,
    "scale_invariance_max_gap": worst,
    "audit_sites": len(sites), "audit_bare": len(unl),
    "failures": FAILURES,
}
with open(os.path.join(HERE, "w132a_row3.json"), "w") as fh:
    json.dump(out, fh, indent=2)

print("\n" + "=" * 78)
print(f"FAILURES: {len(FAILURES)}")
for f_ in FAILURES:
    print("  " + f_)
print("=" * 78)
sys.exit(1 if FAILURES else 0)
