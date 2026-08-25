# S6E8 V13 Diversity Anchor — Public LB 0.97124

This dataset contains the aligned test prediction and its exact fold-safe OOF
counterpart used in the research behind **S6E8 Regime-Calibrated Rank Fusion
| LB 0.97127**.

The anchor was built from fold-safe OOF stacking and deliberately weak/diverse
members. It is published separately so the notebook can remain short,
transparent, and exactly reproducible.

## Files

### `v13_diversity_anchor_lb97124.csv`

- Competition: `playground-series-s6e8`
- Rows: original `test.csv` / `sample_submission.csv` order
- Columns: `id`, `addicted_label`
- Verified public leaderboard score: **0.97124**

### `v13_diversity_anchor_oof.csv`

- Rows: original `train.csv` order
- Columns: `id`, `prediction`
- Honest OOF scheme: `StratifiedKFold(5, shuffle=True, random_state=42)`
- OOF ROC-AUC: **0.9701665486177532**
- Improvement versus the pre-V13 anchor: **5/5 folds**
- Target labels are intentionally not duplicated in this artifact

### `v13_diversity_anchor_oof_audit.json`

Machine-readable alignment, fold metrics, integrity statistics, and SHA-256.

## Method

The V13 extension combines 194 frozen validated members with Szymon
Kapiński's 50 deliberately weak/diverse public members. Each meta-model
validation row is predicted by a model that did not train on that row. The
244-member logistic meta-model uses rank-Gaussian member features,
`C=0.03`, and `max_iter=5000`. The final diversity correction uses the same
`0.70` weight as the published V13 test endpoint.

The OOF AUC is reported for transparency, not as an unseen holdout estimate:
the final correction weight was selected on the complete OOF comparison grid.

## Credits

The diversity extension was inspired by Szymon Kapiński's public work and the
CC0 weak-50 prediction library:

- https://www.kaggle.com/code/szymonkapiski/s6e8-the-score-i-would-not-pick
- https://www.kaggle.com/datasets/szymonkapiski/s6e8-50-weakest-oof-models

Additional public OOF libraries used during the research campaign came from
Naji, boltuzamaki, Ray Kretzschmar, Darius Hafshar, Adarsh and other S6E8
community contributors. Thank you for sharing reproducible artifacts.

License: CC0-1.0.
