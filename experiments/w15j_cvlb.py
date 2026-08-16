"""w15j — consolidated CV<->LB map over the full live submission history.

Merges experiments/audit_results.csv (cross-fitted CV on the frozen SKF5 seed42 folds)
with the live Kaggle submission list, so the LB column is current rather than stale.
Answers two closing questions:
  1. What does the public slice actually resolve?  (transform-family rule, n=38)
  2. Which CV-good file has this account never sent?
"""
import subprocess, sys, io
import pandas as pd, numpy as np

COMP = "playground-series-s6e8"

raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
                     capture_output=True, text=True).stdout
sub = pd.read_csv(io.StringIO(raw))
sub["date"] = pd.to_datetime(sub["date"])
sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
assert (sub["status"] == "SubmissionStatus.COMPLETE").all(), "a submission is not COMPLETE"
print(f"live submissions: {len(sub)}  all COMPLETE")

# best (and only, scores are deterministic) public score per file stem
lb = sub.groupby("stem")["publicScore"].agg(["max", "min", "count"])
inconsistent = lb[(lb["max"] != lb["min"])]
print(f"stems submitted more than once with differing scores: {len(inconsistent)}")

audit = pd.read_csv("experiments/audit_results.csv")
audit = audit.rename(columns={"name": "stem", "lb": "lb_stale"})
m = audit.merge(lb["max"].rename("lb_live"), left_on="stem", right_index=True, how="left")

def family(s):
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit", "w2", "w"):
        if s.endswith("_" + f) or s.endswith(f) and f in ("w", "w2"):
            return f
    return "ens4"          # bare blendNNN = rank-average of all four transforms
m["fam"] = m["stem"].map(family)
m.loc[m["stem"] == "blendtop3", "fam"] = "h3"
m["sent"] = m["lb_live"].notna()

print("\n=== 1. What the public slice resolves: LB by transform family (sent files only) ===")
s = m[m["sent"]]
g = s.groupby("fam").agg(n=("lb_live", "size"), lb_min=("lb_live", "min"),
                         lb_max=("lb_live", "max"), lb_sd=("lb_live", "std"),
                         cv_min=("cv", "min"), cv_max=("cv", "max"))
g["cv_span_e6"] = (g["cv_max"] - g["cv_min"]) * 1e6
print(g.sort_values("lb_max", ascending=False).to_string(
    float_format=lambda v: f"{v:.6g}"))

print("\n=== 2. Never-sent files, by CV ===")
ns = m[~m["sent"]].sort_values("cv", ascending=False)
print(ns[["stem", "cv", "fam"]].head(15).to_string(index=False,
      float_format=lambda v: f"{v:.7f}"))

print("\n=== 3. CV->LB, sent files, correlation at current resolution ===")
print(f"n={len(s)}  pearson(cv,lb)={np.corrcoef(s['cv'], s['lb_live'])[0,1]:+.4f}  "
      f"spearman={s['cv'].corr(s['lb_live'], method='spearman'):+.4f}")
print(f"distinct LB values among sent files: {sorted(s['lb_live'].unique())}")
m.to_csv("experiments/w15j_cvlb.csv", index=False)
print("\nwrote experiments/w15j_cvlb.csv")
