# ⏳ ONE MANUAL ACTION IS OUTSTANDING — final-submission selection

**Deadline 2026-08-31 23:59. Needs a browser. It cannot be done from the Kaggle API** (the
write path was probed and falsified 2026-08-13; only the *read* is exposed, which is what
`experiments/check_selection.py` uses).

Right now `check_selection.py` exits **1: nothing is selected**, so Kaggle will auto-pick the
two final entries by **public** score. That is selecting on the public slice — the exact
mistake that cost this account ~2.5 points of RMSE on `rogii-wellbore-geology-prediction`.

## What to click

<https://www.kaggle.com/competitions/playground-series-s6e8/submissions> → **Use for Final Score**
on exactly these two, and on nothing else:

| # | submission ref | file | CV | public |
|---|---|---|---|---|
| 1 | **55656399** | `w36_ad199stdcorr.csv` | 0.9701400060 | 0.97118 |
| 2 | **55588167** | `w23_ad187stdcorr.csv` | 0.9701150809 | 0.97116 |

Both are already submitted; this only flags them.

## What happens if nobody clicks

Kaggle takes the top two by public score, which today is **determined** — exactly two files
sit at 0.97119 and everything else is at 0.97118 or below:

    w36_ad199stdcorr_ens4.csv   (ref 55714895)   CV 0.9701365875
    w38_ad202stdcorr_ens4.csv   (ref 55714897)   CV 0.9701349001

Neither is the CV-preferred pair. Priced on the live board:

| | cost of NOT clicking |
|---|---|
| tau = 0 | **+4.52e-6** |
| tau = 2.10e-6 (95% upper, refit 2026-08-24) | **+3.07e-6** |
| tau = 5.0e-6 (far above anything measured) | −0.50e-6 |

`experiments/w74a_clickprice.py`, two verbatim controls at 0.000e+00. Honest reading: **the
click is worth about 3–4.5e-6 of expected private AUC, and it is not a large number.** At the
current board density (rank 92 of 2,774, leader 490e-6 ahead) that is roughly a place or two,
not a medal. It is worth doing because it costs one minute and removes a known failure mode,
not because it wins anything.

⚠ Do not "improve" on this list by picking the higher public scores. That is the failure.
Selection here is on CV, and the CV ordering is stable — `w36_ad199stdcorr` is argmax on
**both** defensible CV bases (w73), so this pick does not depend on an unsettled convention.

## How to verify it took

    .venv/bin/python experiments/check_selection.py > /tmp/cs.txt 2>&1; echo $?   # 0 = selected

⛔ Read the status through a redirect or `$(...)`, never through a pipe — `$?` after a pipeline
is `tail`'s status and has already produced one fabricated finding in this workspace.
