"""w48a -- PROVENANCE AUDIT of everything the selection and the pricer rest on.

The angle for this slot is consolidation, and consolidation here means one thing: every
number that decides the deadline pick is re-derived from the raw artefact rather than
re-quoted from a table. Four independent chains are checked, and each one can fail on its
own:

  A. CV REPRODUCES.  For every stem with a stored OOF vector, recompute roc_auc(y, oof) on
     the frozen SKF5 seed-42 folds and compare to every CV the workspace has recorded for
     it. A stale CV anywhere in this chain silently moves w30b's coefficients, the -29.82e-6
     era estimate, w47a's four attacks, and the WANTED pick.
  B. THE REGISTRIES AGREE.  w25a_cvlb_full.csv, w46c_cvlb_live.csv, w23b_sendqueue.csv and
     w45b_unsent_cv.json all carry CVs. They have never been cross-checked against one
     another.
  C. THE LB COLUMN IS LIVE.  Every lb in the fit table is compared to what Kaggle reports
     right now. A mis-joined or stale LB is exactly what would manufacture an era shift.
  D. THE FIT SAMPLE IS THE WHOLE SAMPLE.  Any scored submission missing from
     w46c_cvlb_live.csv means the predictor was fitted on a FILTERED sample, which is a
     selection effect masquerading as a model.

Then the question the angle actually asks -- is the strongest submission the one selected --
is re-derived, twice: once on raw CV (the rule as written) and once on ERA-DEFLATED CV,
because if w46c's era finding is true then raw CV is not comparable across the ad194/ad195
boundary and the selection rule has been comparing incomparable numbers since 08-21.

    .venv/bin/python experiments/w48a_provenance.py
"""
from __future__ import annotations

import json, os, subprocess, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET                       # noqa: E402
import w46c_predlb as W                              # noqa: E402

COMP = "playground-series-s6e8"
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
TOL_CV = 5e-10          # float64 AUC reproduction tolerance
OUT = {}

# ---------------------------------------------------------------- live submissions
SNIPPET = r"""
import json, os
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.competitions.types.competition_enums import SubmissionGroup
tok = json.load(open(os.path.expanduser(
    os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle") + "/credentials.json")))["access_token"]
out = {}
for name, g in (("selected", SubmissionGroup.SUBMISSION_GROUP_SELECTED),
                ("successful", SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL)):
    with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
        r = ApiListSubmissionsRequest()
        r.competition_name = %r
        r.group = g
        r.page_size = 500          # w17's pagination trap: never accept a default page size
        resp = c.competitions.competition_api_client.list_submissions(r)
        out[name] = [(s.ref, s.file_name, s.public_score, str(s.date)) for s in resp.submissions]
print(json.dumps(out))
""" % COMP

p = subprocess.run([KAGGLE_PY, "-c", SNIPPET], capture_output=True, text=True)
if p.returncode != 0:
    sys.exit("API call failed:\n" + p.stderr.strip())
api = json.loads(p.stdout)
live = {}
for ref, fn, sc, dt in api["successful"]:
    if sc in (None, ""):
        continue
    stem = fn[:-4] if fn.endswith(".csv") else fn
    r = live.setdefault(stem, dict(n=0, scores=[], dates=[]))
    r["n"] += 1; r["scores"].append(float(sc)); r["dates"].append(dt)
for r in live.values():
    r["lb"] = max(r["scores"])
    r["day"] = min(r["dates"])[:10]

print("=" * 90)
print("w48a  PROVENANCE AUDIT")
print("=" * 90)
print(f"\nlive API: {len(api['successful'])} scored submissions, {len(live)} distinct stems, "
      f"{len(api['selected'])} SELECTED")
if len(api["successful"]) >= 500:
    sys.exit("  ⛔ page_size 500 saturated -- the list is truncated, refusing to audit on it.")
OUT["n_scored"] = len(api["successful"]); OUT["n_stems"] = len(live)
OUT["n_selected"] = len(api["selected"])

# ---------------------------------------------------------------- registries
REG = {}


def add(name, stem, cv):
    if cv is None or not np.isfinite(cv):
        return
    REG.setdefault(stem, {})[name] = float(cv)


t25 = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv"))
for _, r in t25.iterrows():
    add("w25a", r.stem, r.cv)
t46 = pd.read_csv(os.path.join(HERE, "w46c_cvlb_live.csv"))
for _, r in t46.iterrows():
    add("w46c", r.stem, r.cv)
t23 = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
for _, r in t23.iterrows():
    add("w23b", str(r.file).replace(".csv", ""), r.cv)
for r in json.load(open(os.path.join(HERE, "w45b_unsent_cv.json"))):
    add("w45b", r["stem"], r["cv"])

# ---------------------------------------------------------------- A + B
y = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[TARGET])[TARGET].values
rows = []
for stem, srcs in sorted(REG.items()):
    f = os.path.join(SUB, f"oof_{stem}.npy")
    true_cv = np.nan
    if os.path.exists(f):
        v = np.load(f).ravel()
        if len(v) == len(y) and not np.isnan(v).any():
            true_cv = float(roc_auc_score(y, v))
    vals = list(srcs.values())
    rows.append(dict(stem=stem, true_cv=true_cv, n_src=len(srcs),
                     reg_spread=(max(vals) - min(vals)) * 1e6,
                     reg_cv=vals[0], have_oof=os.path.exists(f),
                     **{f"cv_{k}": v for k, v in srcs.items()}))
A = pd.DataFrame(rows)
A["err6"] = (A.reg_cv - A.true_cv) * 1e6

chk = A[A.true_cv.notna()]
bad_a = chk[chk.err6.abs() > TOL_CV * 1e6]
bad_b = A[A.reg_spread > TOL_CV * 1e6]
print(f"\nA. CV REPRODUCES FROM THE OOF VECTOR")
print(f"   {len(chk)} of {len(A)} registered stems have a stored OOF vector to check against.")
print(f"   mismatches beyond {TOL_CV:.0e}: {len(bad_a)}")
for _, r in bad_a.iterrows():
    print(f"     ⛔ {r.stem:28s} registered {r.reg_cv:.10f}  recomputed {r.true_cv:.10f}  "
          f"{r.err6:+.3f}e-6")
print(f"\nB. THE FOUR REGISTRIES AGREE WITH EACH OTHER")
print(f"   {(A.n_src > 1).sum()} stems appear in more than one registry; "
      f"disagreements: {len(bad_b)}")
for _, r in bad_b.iterrows():
    print(f"     ⛔ {r.stem:28s} spread {r.reg_spread:+.3f}e-6  "
          f"{ {k: v for k, v in REG[r.stem].items()} }")
OUT["A_checked"], OUT["A_bad"] = len(chk), len(bad_a)
OUT["B_multi"], OUT["B_bad"] = int((A.n_src > 1).sum()), len(bad_b)

# ---------------------------------------------------------------- C + D
print(f"\nC. THE lb COLUMN MATCHES KAGGLE RIGHT NOW")
badc = []
for _, r in t46.iterrows():
    if r.stem in live and abs(live[r.stem]["lb"] - r.lb) > 1e-9:
        badc.append((r.stem, r.lb, live[r.stem]["lb"]))
miss = [r.stem for _, r in t46.iterrows() if r.stem not in live]
print(f"   {len(t46)} rows in the predictor's fit+heldout table; "
      f"lb mismatches vs live: {len(badc)}; rows with no live send at all: {len(miss)}")
for s, a, b in badc:
    print(f"     ⛔ {s:28s} table {a:.5f}  live {b:.5f}")
for s in miss:
    print(f"     ⛔ {s:28s} is in the fit table but was never scored on Kaggle")

print(f"\nD. THE FIT SAMPLE IS THE WHOLE SAMPLE")
untabled = sorted(set(live) - set(t46.stem))
print(f"   scored stems absent from w46c_cvlb_live.csv: {len(untabled)}")
for s in untabled:
    known = REG.get(s, {})
    cv = f"{list(known.values())[0]:.10f}" if known else "     no CV on disk    "
    print(f"     - {s:30s} lb {live[s]['lb']:.5f}  day {live[s]['day']}  cv {cv}")
OUT["C_bad"], OUT["C_missing"] = len(badc), len(miss)
OUT["D_untabled"] = untabled

# ---------------------------------------------------------------- the selection question
SLOPE = W.C["cv_e6"]                       # LB e-6 per CV e-6
DEFLATE = abs(W.ERA_SHIFT) / SLOPE         # CV e-6 that the era shift says was not real
FIT_MAX = 0.9701183

sent = []
for stem, r in live.items():
    cv = A.set_index("stem").true_cv.get(stem, np.nan)
    if not np.isfinite(cv):
        k = REG.get(stem, {})
        cv = list(k.values())[0] if k else np.nan
    if np.isfinite(cv):
        sent.append(dict(stem=stem, cv=cv, lb=r["lb"], era=W.new_era(stem),
                         above=cv > FIT_MAX))
S = pd.DataFrame(sent)
S["cv_era"] = S.cv - np.where(S.era, DEFLATE * 1e-6, 0.0)
S["cv_reg"] = np.where(S.above, FIT_MAX, S.cv)      # CV-REGION, total-saturation form

print("\n" + "=" * 90)
print("IS THE STRONGEST SUBMISSION THE ONE SELECTED?")
print("=" * 90)
print(f"\n  w30b slope {SLOPE:.4f} LB-e-6 per CV-e-6, so the {W.ERA_SHIFT:+.2f}e-6 era shift")
print(f"  is {DEFLATE:.2f}e-6 of CV that the ad>=195 files got for free.")
print(f"\n  {len(S)} scored stems have a CV on disk. Top 8 under each ranking:\n")
for lbl, col in (("RAW CV  (the rule as written)", "cv"),
                 ("ERA-DEFLATED CV  (w46c true)", "cv_era"),
                 ("CV-REGION-CAPPED  (the alternative)", "cv_reg")):
    print(f"  --- {lbl} ---")
    for i, (_, r) in enumerate(S.sort_values(col, ascending=False).head(8).iterrows(), 1):
        tag = "ERA" if r.era else "   "
        star = "  <== current WANTED" if r.stem == "w36_ad199stdcorr" else ""
        print(f"    {i}. {r.stem:26s} {r[col]:.10f} {tag}  lb {r.lb:.5f}{star}")
    print()

w = S.set_index("stem")
OUT["deflate_cv_e6"] = DEFLATE
OUT["argmax"] = {c: S.loc[S[c].idxmax(), "stem"] for c in ("cv", "cv_era", "cv_reg")}
for c, lbl in (("cv", "raw"), ("cv_era", "era-deflated"), ("cv_reg", "region-capped")):
    top = S.loc[S[c].idxmax()]
    d = (w.loc["w36_ad199stdcorr", c] - top[c]) * 1e6
    print(f"  {lbl:14s} argmax = {top.stem:26s}  WANTED is {d:+7.2f}e-6 vs it")

json.dump(OUT, open(os.path.join(HERE, "w48a_provenance.json"), "w"), indent=1, default=str)
A.to_csv(os.path.join(HERE, "w48a_cv_recomputed.csv"), index=False)
S.sort_values("cv", ascending=False).to_csv(os.path.join(HERE, "w48a_sent_ranked.csv"), index=False)
print("\nwrote w48a_provenance.json, w48a_cv_recomputed.csv, w48a_sent_ranked.csv")
