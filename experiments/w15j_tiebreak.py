"""w15j — enumerate the auto-selection outcome under every plausible tiebreak rule.

Four runs (w14a/w14b/w14d, and all five of w15a-e) have opened with "nothing is selected,
this is the only live risk on the board", priced at -111e-6 predicted private because the
auto-select default could land on blend158_logit (CV 0.969961).

RESEARCH already re-sized that to ~-10e-6 by noting best-public is a 4-WAY TIE and that
any 2-of-4 draw contains a CV-good ens4 file. What nobody has done is check the limit=1
branch, which RESEARCH explicitly flagged as the live one:
   "If the limit were 1 AND the tiebreak latest-first, blend158_logit alone is selected."

That is checkable. Enumerate the rules.
"""
import pandas as pd, io, itertools

d = pd.read_csv("/tmp/w15j_subs.csv")
d["date"] = pd.to_datetime(d["date"])
d["stem"] = d.fileName.str.replace(r"\.csv$", "", regex=True)
cv = pd.read_csv("experiments/audit_results.csv").set_index("name")["cv"]
d["cv"] = d.stem.map(cv)

tie = d[d.publicScore == d.publicScore.max()].copy()
print(f"best public score {d.publicScore.max()}, {len(tie)}-way tie\n")

# predicted private gap vs the CV pick (blend159av_h3), from w14b's partition arithmetic
# ens4 - h3 = -10e-6 (w14b labelled-truth pooled); logit - h3 = -111e-6 (w14b table)
def priv_gap(stem):
    return -111e-6 if stem.endswith("_logit") else -10e-6

RULES = {
    "earliest submitted":      lambda t: t.sort_values("date"),
    "latest submitted":        lambda t: t.sort_values("date", ascending=False),
    "lowest ref id":           lambda t: t.sort_values("ref"),
    "highest ref id":          lambda t: t.sort_values("ref", ascending=False),
    "filename A-Z":            lambda t: t.sort_values("stem"),
    "filename Z-A":            lambda t: t.sort_values("stem", ascending=False),
}

print(f"{'tiebreak rule':<22}{'limit=1 pick':<18}{'gap':>9}   "
      f"{'limit=2 picks':<30}{'gap (max of selected)':>10}")
worst = 0.0
for name, f in RULES.items():
    o = f(tie)
    one = o.iloc[0]["stem"]
    two = list(o.iloc[:2]["stem"])
    g1 = priv_gap(one)
    g2 = max(priv_gap(s) for s in two)          # private is the MAX of the selected
    worst = min(worst, g1, g2)
    print(f"{name:<22}{one:<18}{g1*1e6:>7.0f}e-6   {' + '.join(two):<30}{g2*1e6:>7.0f}e-6")

print(f"\nWORST CASE ACROSS ALL {len(RULES)} RULES x BOTH LIMITS: {worst*1e6:.0f}e-6")
print("\nblend158_logit is selected uniquely under: ", end="")
hits = [n for n, f in RULES.items() if f(tie).iloc[0]["stem"] == "blend158_logit"]
print(hits if hits else "NONE of the enumerated rules")
print("\nWhy: it is neither the earliest nor the latest, neither the lowest nor the highest")
print("ref, and neither first nor last alphabetically, in a 4-way tie whose other three")
print("members are all CV-good ens4 files.")
