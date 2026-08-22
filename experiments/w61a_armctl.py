"""w61a -- THE ARM 211 FOUR-BASE MATCHED-CONTROL TEST, registered by w60_prereg.txt and
deliberately left unrun by the run that registered it.

w40d_prereg.txt, committed 2026-08-20 BEFORE ARM 211 existed, bars ARM 211 from
`check_selection.WANTED` whatever its CV: the nine `yadoy666` union94 streams come from an
AGGREGATOR and their es-on-val status is UNKNOWN -- neither log-read nor absent-by-mechanism.
w60 keyed that rule into `check_selection.WANTED_INELIGIBLE` (it had been registered and never
keyed for two days) and registered, but did NOT run, the one test that may retire it:

    RETIRE `w40_ad211` from WANTED_INELIGIBLE iff the ad211-minus-ad202 matched-control delta
    is within +-4e-6 on ALL FOUR transform bases held on disk (h3, ens4, rescale, rankraw),
    measured on the frozen folds, AND that reading is quoted in the dict value in the same
    commit.

The logic of the test: ARM 211 is ARM 202 plus exactly those nine streams. If an es-on-val
member's OOF is inflated, the meta-combiner UP-weights it and the stack's cross-fitted CV goes
UP -- inflation is a gain, not a loss. So a null group delta bounds the inflation the rule
exists to refuse. It is not es-clearance and this file does not claim it is.

⚠ THIS IS AN AUDIT, NOT A BLIND TEST. Every CV read here has been on disk since 08-21/08-22
and is quoted in w23b_sendqueue.csv. What is pre-fixed is the CRITERION, not the numbers. See
w61_prereg.txt's honesty clause. The two things this adds over re-quoting the ledger are the
re-derivation from the raw OOF vectors (the chain w48a found stale before) and the enforcement.

    .venv/bin/python experiments/w61a_armctl.py
"""
from __future__ import annotations

import json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, DATA, TARGET                  # noqa: E402

TOL_CV = 5e-10          # w48a's float64 AUC reproduction tolerance
BAR = 4e-6              # w60_prereg's rebuild floor. NOT settable from here.

# The four bases the criterion names, and the file stems that carry them. `ens4` is the BARE
# stem (stdflag.family_suffix: a bare stem is the rank-average of all four transforms).
CRITERION_BASES = ["h3", "ens4", "rescale", "rankraw"]
# Bases held on disk but NOT in the criterion. Reported, never decided on (w61_prereg P5).
EXTRA_BASES = ["hybrid", "logit"]
# The corrected-h3 pair w60 already published at -0.116e-6. A control on MY recomputation,
# not part of the criterion (w61_prereg P4).
CORR_PAIR = ("w40_ad211stdcorr", "w38_ad202stdcorr")

def stems(base):
    if base == "ens4":
        return "w40_ad211std", "w38_ad202std"
    return f"w40_ad211std_{base}", f"w38_ad202std_{base}"

OUT = {"bar": BAR, "criterion_bases": CRITERION_BASES, "failures": []}
def fail(msg):
    OUT["failures"].append(msg)
    print("  ⛔ " + msg)

# ------------------------------------------------------------------ GATE M: MATCHEDNESS
# A delta is only a control if the two arms differ by exactly the nine streams. The criterion
# assumes this and does not check it; a failure here REFUSES regardless of the deltas.
print("GATE M -- matchedness of the ARM 211 / ARM 202 pair")
def shvars(path):
    txt = open(path).read()
    d = {}
    for k in ("D", "X", "NAME"):
        m = re.search(rf"^{k}=(.*)$", txt, re.M)
        if m:
            d[k] = m.group(1).strip()
    d["standardize"] = "--standardize" in txt
    d["make_h3"] = "make_h3.py" in txt
    return d

a = shvars(os.path.join(HERE, "w40f_run.sh"))   # ARM 211
b = shvars(os.path.join(HERE, "w38d_run.sh"))   # ARM 202
if a["D"] != b["D"]:
    fail(f"--drop lists differ: {a['D']!r} vs {b['D']!r}")
if a["standardize"] != b["standardize"] or not a["standardize"]:
    fail("--standardize differs or is absent")
if not (a["make_h3"] and b["make_h3"]):
    fail("one arm does not run make_h3.py")
xa = a["X"].split(","); xb = b["X"].split(",")
added = [d for d in xa if d not in xb]
removed = [d for d in xb if d not in xa]
if removed:
    fail(f"ARM 211 DROPS dirs the control has: {removed} -- not a matched control")
if added != ["ext_members15"]:
    fail(f"ARM 211 adds {added}, expected exactly ['ext_members15']")
else:
    print(f"  ✅ ARM 211 = ARM 202 + {added[0]}, nothing else changed "
          f"(--drop identical, --standardize both, make_h3 both)")
OUT["added_dirs"] = added

emdir = os.path.join(DATA, "ext_members15")
oofs = sorted(f for f in os.listdir(emdir) if f.startswith("oof_") and f.endswith(".npy"))
tests = sorted(f for f in os.listdir(emdir) if f.startswith("test_") and f.endswith(".npy"))
paired = sorted(f[4:] for f in oofs) == sorted(f[5:] for f in tests)
if len(oofs) != 9 or not paired:
    fail(f"ext_members15 holds {len(oofs)} oof / {len(tests)} test vectors, paired={paired}; "
         "the rule is written about NINE streams")
else:
    print(f"  ✅ ext_members15 = 9 paired streams: {[f[4:-4] for f in oofs]}")
OUT["streams"] = [f[4:-4] for f in oofs]

# ------------------------------------------------------------------ recompute CV from raw OOF
print("\nRecomputing CV from submissions/oof_<stem>.npy on the frozen folds")
y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values

# Every ledger that carries a CV, so a stale table cannot pass silently (w48a chain B).
LEDGERS = {}
def _load(path, stemcol, cvcol, strip_csv=False):
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if stemcol not in df.columns or cvcol not in df.columns:
        return
    s = df[stemcol].astype(str)
    if strip_csv:
        s = s.str.replace(r"\.csv$", "", regex=True)
    LEDGERS[os.path.basename(path)] = dict(zip(s, df[cvcol].astype(float)))
_load(os.path.join(HERE, "w23b_sendqueue.csv"), "file", "cv", strip_csv=True)
_load(os.path.join(HERE, "w26d_queueprice.csv"), "stem", "cv")
_load(os.path.join(HERE, "w25a_cvlb_full.csv"), "stem", "cv")
_load(os.path.join(HERE, "w46c_cvlb_live.csv"), "stem", "cv")
_load(os.path.join(HERE, "w48a_cv_recomputed.csv"), "stem", "true_cv")

CV = {}
def cv_of(stem):
    if stem in CV:
        return CV[stem]
    p = os.path.join(SUB, f"oof_{stem}.npy")
    if not os.path.exists(p):
        fail(f"no OOF vector for {stem} -- the criterion says 'measured on the frozen folds'")
        CV[stem] = None
        return None
    v = np.load(p)
    if v.shape[0] != y.shape[0]:
        fail(f"{stem}: OOF length {v.shape[0]} != y {y.shape[0]}")
        CV[stem] = None
        return None
    c = float(roc_auc_score(y, v))
    CV[stem] = c
    # w61_prereg P2: every ledger that names this stem must reproduce it.
    for name, d in LEDGERS.items():
        if stem in d and abs(d[stem] - c) > TOL_CV:
            fail(f"{stem}: {name} says {d[stem]:.12f}, raw OOF says {c:.12f} "
                 f"(err {(d[stem]-c)*1e6:+.4f}e-6) -- a ledger that does not reproduce "
                 "is not a measurement")
    return c

rows = []
for base in CRITERION_BASES + EXTRA_BASES:
    s211, s202 = stems(base)
    c211, c202 = cv_of(s211), cv_of(s202)
    if c211 is None or c202 is None:
        continue
    rows.append(dict(base=base, in_criterion=base in CRITERION_BASES,
                     stem_211=s211, cv_211=c211, stem_202=s202, cv_202=c202,
                     delta_e6=(c211 - c202) * 1e6))
c211, c202 = cv_of(CORR_PAIR[0]), cv_of(CORR_PAIR[1])
corr_delta = None if (c211 is None or c202 is None) else (c211 - c202) * 1e6
rows.append(dict(base="corr_h3(control, NOT in criterion)", in_criterion=False,
                 stem_211=CORR_PAIR[0], cv_211=c211, stem_202=CORR_PAIR[1], cv_202=c202,
                 delta_e6=corr_delta))
OUT["rows"] = rows

print(f"\n{'base':<38} {'ARM 211 CV':>14} {'ARM 202 CV':>14} {'delta e-6':>10}  in-crit")
for r in rows:
    print(f"{r['base']:<38} {r['cv_211']:>14.10f} {r['cv_202']:>14.10f} "
          f"{r['delta_e6']:>+10.3f}  {'YES' if r['in_criterion'] else 'no'}")

# ------------------------------------------------------------------ the registered predictions
crit = [r for r in rows if r["in_criterion"]]
if len(crit) != 4:
    fail(f"only {len(crit)} of the four criterion bases could be measured -- REFUSE")
worst = max((abs(r["delta_e6"]) for r in crit), default=float("inf"))
p1 = len(crit) == 4 and worst <= BAR * 1e6
p3 = any(r["delta_e6"] < 0 for r in crit) and any(r["delta_e6"] > 0 for r in crit)
p4 = corr_delta is not None and abs(corr_delta - (-0.116)) <= 0.01
extra_over = [r["base"] for r in rows if (not r["in_criterion"]) and r["base"] != CORR_PAIR[0]
              and r["delta_e6"] is not None and abs(r["delta_e6"]) > BAR * 1e6
              and r["base"].startswith(tuple(EXTRA_BASES))]

OUT.update(worst_criterion_delta_e6=worst, P1=bool(p1), P2=not OUT["failures"],
           P3=bool(p3), P4=bool(p4), corr_delta_e6=corr_delta,
           extra_bases_over_bar=extra_over)

print("\nREGISTERED PREDICTIONS")
print(f"  P1  all four criterion deltas <= {BAR*1e6:.1f}e-6 (worst {worst:+.3f}e-6): "
      f"{'✅ PASS' if p1 else '❌ FAIL'}   <-- THE DECISION TEST")
print(f"  P2  every ledger reproduces the raw OOF to {TOL_CV:.0e}: "
      f"{'✅ PASS' if not OUT['failures'] else '❌ FAIL'}  "
      f"({sum(len(d) for d in LEDGERS.values())} ledger rows across {len(LEDGERS)} files)")
print(f"  P3  sign is NOT uniform across the four: {'✅ PASS' if p3 else '❌ FAIL — '}"
      f"{'' if p3 else 'a uniform sign is the signature of a real group contribution'}")
print(f"  P4  corr-h3 control reproduces w60's -0.116e-6 (got {corr_delta:+.3f}e-6): "
      f"{'✅ PASS' if p4 else '❌ FAIL'}")
print(f"  P5  non-criterion bases over the bar: {extra_over if extra_over else 'none'} "
      "(REPORTED, NOT DECIDED ON)")

# ------------------------------------------------------------------ the verdict
verdict = "RETIRE" if (p1 and not OUT["failures"]) else "KEEP"
OUT["verdict"] = verdict
OUT["quote"] = (f"w61a four-base matched control, {'PASS' if verdict == 'RETIRE' else 'FAIL'}: "
                + ", ".join(f"{r['base']} {r['delta_e6']:+.2f}" for r in crit)
                + f" e-6, worst {worst:+.2f}e-6 vs the +-4e-6 bar.")
print(f"\nVERDICT: {verdict}")
print("  " + OUT["quote"])
if verdict == "KEEP":
    print("  The rule STANDS. Do not edit the criterion to make it pass.")

with open(os.path.join(HERE, "w61a_armctl.json"), "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print(f"\nFAILURES: {len(OUT['failures'])}  -> experiments/w61a_armctl.json")
raise SystemExit(1 if OUT["failures"] else 0)
