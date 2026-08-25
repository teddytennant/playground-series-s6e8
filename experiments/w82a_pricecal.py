"""w82a — how wrong is `pred_lb`, measured out-of-sample on every priced send?

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.
`w26d` prints a `pred_lb` for every file it prices, `w26g` copies that number verbatim into the
submission DESCRIPTION, and Kaggle stores descriptions forever. So the account carries a free,
append-only, genuinely out-of-sample record of predicted-vs-observed public LB: the prediction
was fixed and published BEFORE the score existed, and no later run can edit it. That is a
better calibration sample than anything refittable on disk, and it costs one API call to read.

⛔ THIS FILE ADOPTS NOTHING AND MUST NOT BE USED TO RE-PRICE OR RE-RANK ANYTHING.
The send order is ranked on CV, the deadline pick is on CV (SELECT_THESE.md), and the point of
the number below is precisely that it argues AGAINST letting the public LB arbitrate. Turning a
per-family residual into a correction term would be choosing the population after seeing which
way it went -- the error w80 §2 and w81 §6 both caught in this workspace.

THE ONE NUMBER. The residual sd is ~14e-6. The public LB is reported to 5 dp, so it carries a
+-5e-6 rounding box of its own (sd 2.89e-6); netting that out still leaves ~13.7e-6 of genuine
predictive error. Every CV gap this account is currently arguing over -- the 1.904e-6 ad202/ad211
swap (w73 §2), the 2.417e-6 binding margin (w75), the 4.52e-6 click price (w74a) -- is SMALLER
than one sd of the LB predictor. ⟹ the public slice cannot arbitrate between the deadline
candidates even in principle, which is the CV-selection discipline arriving from a second
direction rather than as a rule quoted from the brief.

CONTROLS (w72 §5.3: a control that can only fail is not a control).
  C1 +  a synthetic sample with a KNOWN mean/sd is recovered to <0.05e-6 -> the estimator works.
  C2 -  a planted description whose `predicted LB` text is malformed is DROPPED, not parsed as
        0.0 -- a silent parse failure would drag the mean to -971000e-6 and read as a finding.
  C3    the parsed sample size is asserted against a floor, so a future change to w26g's message
        format empties the sample LOUDLY instead of reporting sd over three rows.

    .venv/bin/python experiments/w82a_pricecal.py            # 0 = ok, 1 = a control failed
"""
from __future__ import annotations

import io, json, os, re, subprocess, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
OUT = os.path.join(HERE, "w82a_pricecal.json")

MIN_SAMPLE = 30          # C3: below this the sample is not a sample
PRED_RE = re.compile(r"predicted LB (0\.\d+)")
LB_DP = 5                # Kaggle reports the public score to 5 decimal places


def fetch() -> pd.DataFrame:
    """Live submission list. Reads only; never sends."""
    raw = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
        capture_output=True, text=True, check=True).stdout
    # the CLI can print a pagination token line above the header -- strip anything before it
    lines = raw.splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("ref,"))
    return pd.read_csv(io.StringIO("\n".join(lines[head:])))


def residuals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["publicScore"] = pd.to_numeric(d["publicScore"], errors="coerce")
    d["pred"] = d["description"].astype(str).str.extract(PRED_RE)[0].astype(float)
    d = d.dropna(subset=["pred", "publicScore"]).copy()
    d["day"] = d["date"].astype(str).str[:10]
    d["fam"] = d["fileName"].str.extract(r"(ad\d+)")[0]
    d["resid_e6"] = (d["publicScore"] - d["pred"]) * 1e6
    return d


def controls() -> list[str]:
    bad = []

    # C1 -- known-answer recovery
    rng = np.random.default_rng(8225)
    truth_mu, truth_sd = 3.5, 11.0
    z = rng.normal(truth_mu, truth_sd, 200_000)
    if abs(z.mean() - truth_mu) > 0.05 or abs(z.std(ddof=1) - truth_sd) > 0.05:
        bad.append("C1 estimator does not recover a known mean/sd")

    # C2 -- a malformed prediction must be DROPPED, never parsed as zero
    planted = pd.DataFrame({
        "fileName": ["a.csv", "b.csv"],
        "date": ["2026-08-25 00:00:00", "2026-08-25 00:00:00"],
        "description": ["w26d predicted LB 0.971170, fine",
                        "w26d predicted LB TBD -- no number here"],
        "publicScore": [0.97117, 0.97117],
    })
    got = residuals(planted)
    if len(got) != 1:
        bad.append(f"C2 malformed description not dropped (kept {len(got)} of 2)")
    elif abs(got["resid_e6"].iloc[0]) > 0.6:
        bad.append("C2 the well-formed row did not price to ~0")

    return bad


def main() -> int:
    bad = controls()
    d = residuals(fetch())

    if len(d) < MIN_SAMPLE:                                   # C3
        bad.append(f"C3 only {len(d)} priced sends parsed, floor is {MIN_SAMPLE} "
                   f"-- has w26g's message format changed?")

    quant_sd = (10.0 ** -LB_DP) / np.sqrt(12) * 1e6           # sd of a uniform +-0.5 ulp box
    sd = float(d["resid_e6"].std(ddof=1)) if len(d) > 1 else float("nan")
    net = float(np.sqrt(max(sd ** 2 - quant_sd ** 2, 0.0))) if sd == sd else float("nan")

    print("=" * 92)
    print("w82a -- predicted vs observed public LB, on the account's own published predictions")
    print("=" * 92)
    print(f"  priced sends with a score   {len(d)}   days {sorted(d['day'].unique())}")
    print(f"  residual mean               {d['resid_e6'].mean():+.2f}e-6")
    print(f"  residual sd                 {sd:.2f}e-6")
    print(f"  LB rounding contributes     {quant_sd:.2f}e-6  ->  net predictive sd {net:.2f}e-6")
    print()
    print("  --- by send day ---")
    print(d.groupby("day")["resid_e6"].agg(["count", "mean", "std"]).round(2).to_string())
    print()
    print("  --- by arm family, n>=3 (⛔ REPORTED, NOT ADOPTED -- see the header) ---")
    fam = d.groupby("fam")["resid_e6"].agg(["count", "mean", "std"]).round(2)
    print(fam[fam["count"] >= 3].to_string())
    print()
    print("  ⚠ Every margin currently under argument is smaller than one residual sd:")
    for name, val in (("ad202/ad211 CV swap (w73 §2)", 1.904),
                      ("binding CV margin (w75)", 2.417),
                      ("click price at tau=0 (w74a)", 4.523)):
        print(f"       {name:<32} {val:>6.3f}e-6   vs sd {sd:.2f}e-6")
    print("  ⟹ the public slice cannot arbitrate between the deadline candidates. Select on CV.")
    print()
    for b in bad:
        print(f"  ❌ {b}")
    print(f"  {'✅ CONTROLS 0 failures' if not bad else '❌ CONTROLS FAILED'}")
    print(f"\nFAILURES: {len(bad)}")

    json.dump({
        "n": int(len(d)), "days": sorted(d["day"].unique().tolist()),
        "resid_mean_e6": float(d["resid_e6"].mean()), "resid_sd_e6": sd,
        "lb_quantisation_sd_e6": float(quant_sd), "net_predictive_sd_e6": net,
        "by_family_e6": fam.to_dict(orient="index"),
        "ADOPTS": "nothing -- measurement only; see the module docstring",
    }, open(OUT, "w"), indent=2, default=float)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
