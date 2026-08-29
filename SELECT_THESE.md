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

## ⚠ AND DO NOT CLICK A FILE NAMED ANYWHERE ELSE (w114, 2026-08-29)

The two rows in the table above are the whole instruction. Clicking a *different* pair is a
far larger error than not clicking at all — priced on w74a's own estimator
(`experiments/w114a_misclick.py`, GATE R reproduces w74a's headline to 0.000e+00):

| what you click | cost vs the table above |
|---|---|
| nothing at all (Kaggle auto-selects on public) | **+4.52e-6** |
| `w21_ad187corr` + `w20_ad187_h3` | **+35.17e-6** — 7.8x worse |
| `w16i_schemeavg` + `blend159av_h3` | **+81.92e-6** — 18.1x worse |

Those two wrong pairs are not hypothetical: until this run `check_selection.py` printed the
first inside a green-tick "✅ RESOLVED. WANTED HAS MOVED" box and annotated the second as
"current WANTED". Both were true in August's first week and have been false since 08-19/08-21.
That narration now lives behind `check_selection.py --history`, and standing check #48
(`w114b_selectguard`) fails if any printed line names a selectable file that is not WANTED.

## How to verify it took

    .venv/bin/python experiments/check_selection.py > /tmp/cs.txt 2>&1; echo $?   # 0 = selected

⛔ Read the status through a redirect or `$(...)`, never through a pipe — `$?` after a pipeline
is `tail`'s status and has already produced one fabricated finding in this workspace.

---

## Re-verified 2026-08-24 (w75) — the pick is unchanged and now stands on a wider check

- `w74b_clickstaleguard` **rc=0**: tier membership did not move on the 08-24 sends, so the
  **+4.52e-6** price above is live and exactly two files still sit at 0.97119.
- The slot-1 argmax was re-run over the **whole sent population** — all 122 of the 131 stems
  that have a stored OOF vector, with **CV recomputed from that vector**, not parsed from a
  submission description. `w36_ad199stdcorr` is the argmax. The 9 stems without an OOF all
  score ≤ 0.97107 public and cannot contend for a pick decided on CV.
- The CV→LB **era term** was refreshed from n=5 to n=27 (**−29.82 → −23.77e-6**). The pick does
  not move, and its binding margin is **era-INVARIANT**: the nearest era-deflated challenger,
  `w38_ad202stdcorr`, is in the same era, so the deflation cancels. Slot 1 would only lose the
  argmax at era = −40.35e-6, which is 7.4 se from the refreshed estimate.
- Slot 2 remains `w23_ad187stdcorr` on w64's settled reasoning: the whole slot-2 decision is
  worth **0.14e-6**, and if it is ever re-opened the candidate is `w38_ad202stdcorr`, not
  `w40_ad211stdcorr`.
- Across the 122 sent files, **Spearman(CV, public LB) = +0.820**. Selecting on CV here is
  supported by the data, not just by the Rogii precedent.
