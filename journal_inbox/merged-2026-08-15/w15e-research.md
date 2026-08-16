### ⚠ A small additive rank correction costs −1.1e-5 before it earns anything — measured 2026-08-15 (w15e)

The workspace has repeatedly considered "add a small correction vector to the blend's ranks".
It has never priced the **toll**. Adding a heavy-tailed correction to a strict ranking degrades
it wherever the correction is uninformative, and that cost is systematic, not noise.

Measured with the matched control this workspace requires: the same correction values with
their **rows permuted** (same sd, same tails, same weight, zero information), added to
`oof_blend159av_h3` percentile ranks on 400 draws of exactly 296,302 labelled rows.
Correction sd 0.0217, weight 0.10 — a perturbation reproducing rank-corr 0.9999719 vs base.

| | AUC change |
|---|---|
| **null mean** | **−1.105e-05** |
| null sd | 4.70e-06 |
| q05 / q50 / q95 | −1.92e-05 / −1.10e-05 / −3.52e-06 |
| min / max | −2.35e-05 / +1.49e-06 (**1 of 400 draws positive**) |

**Operational rule: a rank correction of this size needs ~+3e-5 of gross signal to show +2e-5
net.** Price any such proposal against the toll before building it. The toll scales with the
correction's magnitude, so re-run `experiments/w15e_nullband.py` with the candidate's own
values rather than reusing −1.1e-5 as a constant.

**Corollary — the public LB cannot adjudicate corrections of this class.** LB scores quantise
at 1e-5 and the null band is ±0.5e-5 at full test size (wider by 1/sqrt(f) on a public slice
that is a fraction f of the 296,302 rows), while the toll shifts the centre by −1.1e-5. One
paired read therefore resolves nothing finer than ~±1e-5. Do not spend slots sweeping a
correction weight against LB response.

### The public submission space is one ranking, and it is behind us — measured 2026-08-15 (w15e)

`experiments/w15e_extcorr.py` rank-correlates every downloadable public submission against our
pack on the 296,302 test rows. **Re-run it before any future "blend with a public file" idea.**

- Every public file at LB ≥ 0.9709 sits at Spearman **0.9980–0.9988** against `blend159av_h3`,
  and **0.9995–1.0000 against each other**.
- Several are the same file re-published. md5-identical pairs found: `najiama/ensemble-of-
  ensembles-lb-0.97101` ≡ `anthonytherrien/…-vault/submission.csv`; `raykkretzschmar/mix-the-
  meta-models` ≡ `najiama/s6e8-psa/Rayk_submission.csv`; `krasnov/top-1-0.97099` ≡ the vault's
  `submission (1).csv`. `anthonytherrien/…-nn-residual-network` rank-correlates **0.99999993**
  with najiama's ensemble — it is that file plus noise, not a neural network.
- **The best public file is 0.97101; we are at 0.97106.** The public space is behind us and
  redundant with us. The leaders' ~18e-5 is not in the public notebooks.
- The only decorrelated public files are weak and fall under the closed accuracy-floor rule:
  `najiama/s6e8-psa/Naji_KNN_submission.csv` (0.947 vs our blend) and
  `ravi20076/playgrounds6e8-public-l2stack-v1` (0.959).

⚠ **Correction to a figure the 2026-08-15 missions quote.** Our pack is **not** internally
correlated at ~0.9999. Over all 14,028 member pairs the test-space Spearman is median
**0.98142**, q99 0.99816, **min 0.78058**; per-member maxcorr median 0.99783, **min 0.92746**
(`orig_binm`). The 0.9999 figure describes our *blends*, not our *members*.

### Transductive signal is the one direction orthogonal to our pack — 2026-08-15 (w15e)

`raykkretzschmar/s6e8-transductive-anti-student-signals` (dataset, 2026-08-14) ships test-space
model signals only — no labels, no submission: `test_teacher`, `test_student`,
`global_control`/`global_reconstructed`, `specialist_*`, `retrieval_signal`, and a 240,000-row
`reference_contrast`. Construction is in `raykkretzschmar/mix-the-meta-models-then-learn-what-
they-miss` (pulled to `notebooks/w15e_rayk_mixmeta/`): a raw-feature LightGBM teacher, a
smoother LightGBM **student** regressing the teacher's percentile ranks *while carrying the
unlabeled test rows with their teacher predictions at weight 0.245*, and the teacher-minus-
student rank residual signed-squared as the correction, added at a published weight of 0.10.

**Measured against all 168 of our members: max |ρ| 0.0772, median +0.017.** Every public
*submission* sits at 0.94–0.999 against us; this sits at 0.077. Two coherence checks: the
teacher rank-correlates 0.9908 with our best blend (a competent model, not noise), and the
members most aligned with the correction are exactly our smoothest ones — `golem_c` (spline
GAM), `logreg`, `knn`, `realmlp`, `bolt_fttransformer`, `bolt_tabr_retrieval`, all at ≈ −0.07,
which is the sign and the set a "what a smooth model misses" contrast should produce.

**Why our pack cannot produce it: all 168 members are inductive** — fitted on train rows,
applied to test rows blind. A signal defined by reconstruction failure *on the test
distribution* is orthogonal to that class by construction. This is NOT the closed pseudo-
labeling item (−0.0034): no labels or pseudo-labels of the target are involved, only the
teacher's own outputs.

Author's evidence (not ours, and not reproducible here): nested regeneration over five outer ×
four inner folds, +0.000018 / +0.000019 / +0.000036 / +0.000025 on four independent OOF
anchors, **60/60 anchor-by-fold positive**, with the leaderboard explicitly not used to choose.

**It has no OOF and cannot be scored on our frozen folds — that is a property of the object.**
Our one LB read (ref 55526742, 0.97104 vs base 0.97105) is uninformative for the reason in the
toll section above. The open route is to **rebuild the teacher/student inductively on our own
frozen folds** and price it on 691,369 labelled rows; budget a full run with the box to itself
(~30 LightGBM fits on 550k rows) and hold the weight at 0.10 rather than searching it.

### Verifying a transcribed public recipe against its author's own output — technique (w15e)

When a public correction ships as test-space vectors, the failure mode you can actually test is
mis-transcription. Apply your version to the author's **own base file** and require it to move
that file *towards* the author's published output, against a row-permuted control of the same
values. `experiments/w15e_verify_recipe.py`: corr to his output went 0.9999423 → **0.9999685**
(45.5% of the gap closed), while the shuffled control moved *away* to 0.9999142 ± 1.2e-7 —
**z = +447**. Cheap, and it converts "I think I read the notebook right" into a measurement.

### New public OOF, 2026-08-15 — checked and not worth importing

Re-ran the REST dataset enumeration (four search terms). Five datasets are new since the
2026-08-11 sweep. Three ship real OOF: `mohankrishnathalla/s6e8-{xgb,cat-mlp,lgb-dart}-oof`,
the `_v3` tuner outputs. Gated on our frozen folds (`experiments/w15e_newoof.py`):

| member | solo OOF | maxcorr | nearest |
|---|---|---|---|
| mkt_xgb_v3 | 0.965882 | 0.998518 | `mkt_xgb` |
| mkt_cat_v3 | 0.965032 | 0.997288 | `mkt_cat` |
| mkt_lgb_v3 | 0.966155 | 0.998332 | `mkt_lgb` |

Honest OOF, all under the 0.9720 credibility ceiling, but each is a re-tune of the same
author's member we already hold. Not imported. The other two new datasets are
`najiama/s6e8-psa` (three submission CSVs, no OOF) and
`anthonytherrien/predicting-smartphone-addiction-vault` (other people's submissions
re-uploaded — see the md5 collisions above).
