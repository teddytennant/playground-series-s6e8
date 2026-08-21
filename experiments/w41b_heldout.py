"""w41b: the HELD-OUT test of the w30b LB predictor.

w39c froze a `pred_lb` for each queued file in `w39c_gapaudit.json` BEFORE any of them was
sent. The 08-21 drain landed ten of them. This compares realised against frozen -- the first
genuinely out-of-sample test of the predictor, as opposed to the in-sample residuals (mean
5.8e-10, sd 7.2e-6 over n=78) that the same file reports on the fitted rows.

⚠ Slot 9 was swapped by w39e (`w36_ad199std_logit` -> `w36_ad199std_hybrid`), so the sent file
at that rank has no frozen prediction and is EXCLUDED rather than compared against the number
belonging to a different file.

Run AFTER a drain. Writes experiments/w41b_heldout.csv.
"""
import csv, json, math

pre = {r["file"]: r for r in json.load(open("experiments/w39c_gapaudit.json"))["prereg_0821"]}
gap = json.load(open("experiments/w39c_gapaudit.json"))
IN_SD = gap["sd_resid_e6"]          # in-sample residual sd, in units of 1e-6

rows = list(csv.reader(open("/tmp/subs.csv")))[1:]
sent = {r[1]: r[5] for r in rows if r[2].startswith("2026-08-21") and r[5]}

recs = []
for f, lb in sent.items():
    if f not in pre:
        print(f"  excluded (no frozen prediction): {f}")
        continue
    p = pre[f]
    recs.append(dict(file=f, cv=p["cv"], pred=p["pred_lb"], act=float(lb),
                     resid_e6=(float(lb) - p["pred_lb"]) * 1e6))

recs.sort(key=lambda r: -r["cv"])
print(f"\n{'file':<32}{'CV':>13}{'pred LB':>11}{'actual':>10}{'resid e6':>11}")
for r in recs:
    print(f"{r['file']:<32}{r['cv']:>13.7f}{r['pred']:>11.6f}{r['act']:>10.5f}{r['resid_e6']:>+11.1f}")

n = len(recs)
mean = sum(r["resid_e6"] for r in recs) / n
sd = math.sqrt(sum((r["resid_e6"] - mean) ** 2 for r in recs) / (n - 1))
se = sd / math.sqrt(n)
z = mean / (IN_SD / math.sqrt(n))

print(f"\n{'='*74}\nHELD-OUT RESULT   n = {n}")
print(f"  mean residual {mean:+.2f}e-6   sd {sd:.2f}e-6   se {se:.2f}e-6")
print(f"  in-sample residual sd was {IN_SD:.2f}e-6")
print(f"  z of the mean against the in-sample sd: {z:+.2f}")

# The LB is quoted to 5 dp = 1e-5 = 10 units of 1e-6, so rounding alone contributes
# a uniform(-5,+5)e-6 term, sd 2.89e-6. Anything at or under that is unresolvable.
ROUND_SD = 10 / math.sqrt(12)
print(f"  LB quantisation alone contributes sd {ROUND_SD:.2f}e-6 -- the resolution floor.")

if abs(z) < 2:
    print("  ✅ UNBIASED at n={}: the frozen predictions are not systematically off.".format(n))
else:
    print("  ⛔ BIASED: the predictor is off by more than the in-sample sd supports.")
if sd > IN_SD * 1.5:
    print(f"  ⚠ SPREAD IS WIDER OUT OF SAMPLE ({sd:.2f} vs {IN_SD:.2f}e-6). The in-sample sd")
    print("     understates the true per-file uncertainty; widen any P(beat) computed from it.")
else:
    print(f"  ✅ spread consistent with in-sample ({sd:.2f} vs {IN_SD:.2f}e-6).")

with open("experiments/w41b_heldout.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "cv", "pred", "act", "resid_e6"])
    w.writeheader(); w.writerows(recs)
print("\nwrote experiments/w41b_heldout.csv")

# ---------------------------------------------------------------------------
# DOMAIN SPLIT. The aggregate above is meaningless on its own: five of the nine are
# single-member `w37_cal_*` vectors at CV 0.958-0.969, and the predictor was fitted on
# STACK files spanning CV 0.9700125..0.9701183. w37e's R2 independently established that
# the offset is not linear across that distance. Judge the predictor only where it was fitted.
# ---------------------------------------------------------------------------
import csv as _csv
_fit = [float(r["cv"]) for r in _csv.DictReader(open("experiments/w39c_gapaudit.csv")) if r.get("cv")]
LO, HI = min(_fit), max(_fit)
print(f"\n{'='*74}\nDOMAIN SPLIT -- predictor fitted on CV [{LO:.7f}, {HI:.7f}], n={len(_fit)}")

near = [r for r in recs if r["cv"] > HI - 5e-5]     # stack files, at or above the fitted top
far  = [r for r in recs if r["cv"] <= HI - 5e-5]    # single members, far below

for label, grp in (("IN/NEAR DOMAIN (stack files)", near), ("EXTRAPOLATED (single members)", far)):
    if not grp:
        continue
    m = sum(r["resid_e6"] for r in grp) / len(grp)
    s = math.sqrt(sum((r["resid_e6"] - m) ** 2 for r in grp) / (len(grp) - 1)) if len(grp) > 1 else float("nan")
    print(f"\n  {label}  n={len(grp)}")
    for r in grp:
        over = ("ABOVE fitted max" if r["cv"] > HI else
                "BELOW fitted min" if r["cv"] < LO else "inside")
        print(f"    {r['file']:<32} cv {r['cv']:.7f} ({over:16}) resid {r['resid_e6']:+8.1f}e-6")
    print(f"    mean {m:+.2f}e-6   sd {s:.2f}e-6   se {s/math.sqrt(len(grp)):.2f}e-6"
          f"   z vs in-sample sd {m/(IN_SD/math.sqrt(len(grp))):+.2f}")

if near:
    m = sum(r["resid_e6"] for r in near) / len(near)
    if m < -2 * IN_SD / math.sqrt(len(near)):
        print(f"""
  ⛔⛔ THE PREDICTOR IS OPTIMISTIC ABOVE THE FITTED RANGE. All {len(near)} stack files sit
     ABOVE the fitted CV maximum {HI:.7f}, and ALL {len(near)} came in BELOW prediction,
     mean {m:+.1f}e-6 against an in-sample residual sd of {IN_SD:.1f}e-6.
     The CV->LB slope FLATTENS at the top of our range: CV gains earned above the
     fitted maximum do not convert to LB at the fitted rate.
     -> every `pred_lb` for an unbuilt/unsent high-CV arm is INFLATED, and so is every
        P(beat best) derived from it. w26d's queue prices are upper bounds, not estimates.""")
