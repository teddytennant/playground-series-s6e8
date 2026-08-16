# w15d — durable facts for RESEARCH.md

## Correction + expansion to "The original dataset — CLOSED" section

### Positive identification of the source (was an assumption, now evidence)

The competition's linked original is **`algozee/smartphone-addiction-prediction-data`**, and
it is **deleted** (so is the account) — forum topic 731719, "Original Dataset not available",
2026-08-01 00:02 UTC. It is byte-identical to the copy we hold:

- the surviving notebook `lukhilaksh/smartphone-addiction-prediction-89-beats` reads
  `/kaggle/input/datasets/algozee/smartphone-addiction-prediction-data/Smartphone_Usage_And_Addiction_Analysis_7500_Rows (1).csv`
  — the `" (1)"` is a browser-download artefact, i.e. algozee re-uploaded someone else's copy;
- `danishzulfiqar5050/smartphone-addiction-prediction` ships a file under that exact name at
  **md5 `d831a326bc6f0ab76056a12279cb0047`**, identical to
  `data/orig/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`.

Independently confirmed the same day by w15a via MILANFX's `s6e08originaldata` mirror.
**No other public Kaggle dataset carries the 12-column schema** — 15 candidates plus all four
of jayjoshi37's plausible siblings screened by `experiments/w15d_screen_source.py`; every hit
is a byte-copy. There is no better original to find. Do not search again.

### The generator did TWO things, not one — and the second one breaks the transfer

RESEARCH already records the accounting identity as a generator artefact. The symmetric half
was missing and it is the decision-relevant one:

| constraint | ORIGINAL | COMPETITION |
|---|---|---|
| `daily >= social + gaming + work_study` | violated in **60.7%**, min slack −12.36 | **0 / 603,714** complete rows (421,427 train + 182,287 test), min slack exactly 0.00 |
| `weekend − daily ∈ [0.50, 3.00]` | **100.0%**, min exactly 0.50, max exactly 3.00 | **52.0%** train / 52.1% test, range [−7.91, +11.49] |

Each frame satisfies a hard constraint the other lacks. The original was constructed as
`weekend = daily + U(0.5, 3)` with independent components; the competition destroys that and
enforces a budget instead. Downstream:

| | orig | comp |
|---|---|---|
| corr(daily, social / gaming / work) | +0.010 / +0.001 / +0.003 | **+0.596 / +0.429 / +0.526** |
| mean social / gaming / work | 3.27 / 2.01 / 3.24 | 2.47 / 1.46 / 2.37 |
| `r = (soc+gam+work)/daily` median / q95 / q99 / max | 1.140 / 2.713 / 3.543 / 5.026 | 0.885 / 0.949 / 0.989 / **1.0000** |

`daily` itself is untouched (threshold 8.0 sits at pct 55.41 orig vs 52.21 comp; 6.0 at 33.87
vs 30.87). Only the three components were rescaled. `r` piling up against a hard 1.0 with q99
at 0.989 is a **repair**, not a learned soft constraint.

**So the 7,500 rows are a sample of a different joint distribution, not a small sample of
ours.** That is the mechanism behind concat's monotone harm (−58e-6 at 1×); it was never a
dilution problem.

### ⚠ The two frames have DIFFERENT LABEL FUNCTIONS on 86% of the rows

Addiction rate by `social_media_hours`:

| social | ≤1 | 1–2 | 2–3 | 3–3.5 | 3.5–4 | 4–4.5 | 4.5–5 | 5–6 |
|---|---|---|---|---|---|---|---|---|
| **original** | 0.503 | 0.551 | 0.537 | 0.545 | 0.546 | **1.000** | 1.000 | 1.000 |
| **competition** | 0.263 | 0.518 | 0.816 | 0.945 | 0.979 | 0.990 | 0.999 | 1.000 |

The original is **flat noise below social = 4.0** then a hard step to exactly 1.0. The
competition has **no step at 4.0** — a monotone ramp across the whole range, steepest exactly
where the original is flat. Restricted to `daily ≤ 6` the competition still runs 0.074 → 1.000
in `social` alone. **86.07% of competition rows have `social ≤ 4`.**

This sharpens RESEARCH's "the generator smeared a crisp two-threshold rule into a ramp": the
smear is not a fuzzing of the same rule, it **moved the signal into a region where the source
has none**. That is a hard ceiling on anything fitted to the original.

### Route 1 (separate estimator) is now closed with the right instrument, and so is its best-possible version

**The strongest original-trained member this workspace can build** (`experiments/w15d_origrepair.py`):
quantile-map the original's `r` onto the competition's `r` and rescale its three components,
then the unchanged `orig_member.py` recipe (10 MCAR-masked copies at competition rates, 8 bags,
500 rounds). No competition label is used — the repair reads only the unlabelled `r` marginal.

| mode | transfer AUC | spearman to pack h3 |
|---|---|---|
| `none` (= `orig_binm`) | 0.885258 | +0.854 |
| **`r`** | **0.921475** | +0.890 |
| `r+wd` (also repair weekend) | 0.915082 | +0.864 |
| `shuffle` control (random comp `r`) | 0.920083 | +0.893 |

**+36e-3, ten times the +36e-4 that missingness masking bought** — but the shuffled control
recovers 0.9201 of the 0.9215, so **+35e-3 is the marginal shift and only +1.4e-3 is the rank
matching**. Repairing the weekend construction on top is −6.4e-3 (the competition's
`weekend − daily` is wide *because* the generator smeared it; forcing that onto the original
injects noise). Saved as `oof/oof_w15d_origrep_r.npy` / `test_w15d_origrep_r.npy`.

### The one-parameter instrument — use this whenever "member X is worth zero" needs checking

w14a measured the 159-member design at condition number ~1e18, which made every recorded
member-worth-zero suspect. The fix is to ask with **one** parameter instead of 159:
`z(w) = (1−w)·rank(pack) + w·rank(member)`, 181-point grid, in-sample **and** cross-fitted on
the frozen folds. `experiments/w15d_twoway.py` (4 packs × 4 members) and `w15d_twoway2.py`.

```
pack blend159av_h3 / blendtop3 / blend158_h3 / blend159av_rankraw
  orig_binm    in-sample w*=0.000  gain +0.00e-6   cross-fitted 0.000   +0.00e-6   0/5
  orig_bin     in-sample w*=0.000  gain +0.00e-6   cross-fitted 0.000   +0.00e-6   0/5
  PERM control in-sample w*=0.000  gain +0.00e-6   cross-fitted 0.000   +0.00e-6   0/5
  NOISE control in-sample w*=0.002 gain +0.24e-6   cross-fitted 0.001   -1.01e-6   0/5
  origrep_r    in-sample w*=0.002  gain +0.09e-6   cross-fitted 0.002   -0.08e-6   3/5
```

**Read the in-sample row, not the cross-fitted one.** An in-sample search cannot penalise a
useful member — it is free to overfit *toward* it — so w\* = 0.000 exactly is the strong result.
The uniform-noise control takes w\* = 0.002, so **`orig_binm` scores strictly below pure
noise**, and the repaired member's +0.09e-6 is a twentieth of the pipeline's own 2e-6
reproducibility floor. Second direction, same verdict: a 2-parameter cross-fitted logistic
`y ~ logit(pack) + rank(orig_binm)` gives **−24e-6** against a −2.9e-6 permuted control.

**Conclusion: the zero is a property of the member, not of the stacker's geometry.** w14a's
condition number is real and does not exculpate anything.

### Measured null — do not repeat

`orig_binm` is **not** failing by extrapolating off-support. 10.32% of competition rows fall
outside the original's `(daily, social)` box; `orig_binm` scores 0.8806 inside vs 0.8759
outside — flat (`experiments/w15d_support.py`). Support is not the problem, the label function
is (§ above).

### Add to the closed list

- **The original dataset, all three remaining routes.** Concat was closed 2026-08-11; w15d
  closes the separate estimator (w\*=0.000, below a pure-noise control, with the singular
  stacker removed from the argument) and column semantics (the label functions differ on 86%
  of the frame). The strongest version of the idea — geometry-repaired refit, +36e-3 transfer —
  is a measured null in the pack. **The Playground "find the original" edge is genuinely absent
  in S6E8, and the reason is structural, not procedural.**
- **Searching for a better original.** No other public dataset has the schema; the official one
  is deleted and byte-identical to ours.

### Small operational notes

- `kaggle datasets list -s <term>` + `kaggle datasets download -d <ref> --unzip` is enough to
  screen a candidate source in one pass; `experiments/w15d_screen_source.py <dir>` takes a
  directory of them and prints the identity/marginal signature table.
- Kaggle competition metadata is reachable without the JS shell:
  `GET https://www.kaggle.com/api/v1/competitions/list?search=<terms>` with
  `Authorization: Bearer <access_token>` from `/home/nixos/.kaggle/credentials.json` (OAuth,
  **not** `kaggle.json`; the venv has no `certifi`, so pass an unverified SSL context). Returns
  `maxDailySubmissions=10`, `evaluationMetric='Roc Auc Score'`, `userRank`, `teamCount`. It
  does **not** return the overview prose. `GET /api/v1/datasets/view/<owner>/<slug>` returns a
  dataset's description and `currentVersionNumber`.
- Board at 2026-08-15 ~11:50 UTC: `teamCount` **1878**, `user_rank` **21**.
- Best public notebook is `najiama/ensemble-of-ensembles-lb-0-97101` at 0.97101, below our
  0.97106 (confirmed independently by w15a).
