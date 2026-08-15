### Correction: the "irreducible coin flip" band is a fact about the ORIGINAL, not about us

`RESEARCH.md`'s generating-rule table says the `(social ≤ 4, 6 < daily ≤ 8)` cell is "an
irreducible coin flip (flat in both drivers, splits Mild 558 / Moderate 467)". That is true of
the **1,025 original rows**. Measured on our own OOF (`experiments/w14d_bandmap.py`,
2026-08-14), the same cell holds **102,202 competition rows** and `blend159av_h3` ranks them at
**within-cell AUC 0.939442**. It is *more* rankable than cell D (`social ≤ 4, daily ≤ 6`,
0.923973) and than the both-drivers-missing population (0.912365). The generator's smear did not
only blur the thresholds, it gave the band an internal ordering the pack has largely found.
Do not plan a run around "the band is a coin flip".

### Where the OOF AUC deficit actually lives (exact, not estimated)

Every positive/negative pair belongs to exactly one (cell of pos, cell of neg) bucket, so
`AUC = Σ_ij U(pos_i, neg_j)/(Npos·Nneg)` with `U` the Mann-Whitney statistic (ties 0.5). The
identity reproduces `roc_auc_score` to 0.00e+00 and splits the deficit `1 − 0.970049 = 0.029951`
exactly. On the generator's seven cells:

- **within-cell 23.3%, cross-cell 76.7%.**
- Biggest single bucket is **D×D at 13.5%**; the band's own within bucket is **4.9%**, fifth
  overall. D also supplies the four largest cross-cell terms. **D — the cell the real rule says
  is unanimously negative (orig rate 0.0000) and where the generator smeared it to 0.3253 — is
  where the loss is, not the band.**
- Exact oracle ceilings: perfect within the band **+0.001461**; perfect within every cell
  +0.006964; perfect across all cell pairs +0.022987. Use these to price any regional idea
  before building it.

`experiments/w14d_bandmap.py` prints all of this in ~3 min from saved vectors, no refitting.
`W14D_BLEND` env var picks the blend.

### CLOSED (2026-08-14): error analysis / targeted correction on the generator's rule cells

Two instruments, both with matched controls, both null:

1. **Cross-fitted per-cell isotonic** (the only thing that can touch the 76.7% cross-cell mass,
   since monotone maps cannot reorder within a cell): real cells **−118e-6**, size-matched
   permuted-cell control **−124e-6**, so **real − ctrl = +6e-6** with both arms negative. The
   loss is the five per-fold isotonic maps not being mutually monotone, not the segmentation.
   Reproduces `iso_regime.py`'s −85e-6 on a different partition and adds the control it lacked.
2. **Cell-local LightGBM** (`experiments/w14d_cellboost.py`) — 40-column frame + the stack score
   as a feature, fitted on one cell's rows only, frozen folds restricted to the cell, against
   the same frame row-permuted within the cell. **real − ctrl negative at 9 of 9 checkpoints**
   across BAND / D / G, monotonically worse with capacity. This is `resid_boost2.py --mode
   feature` with the "it never got to specialise on the region" objection removed.

### Operational gotcha: `pgrep -f <script>.py` matches your own waiter

`until ! pgrep -f w14d_cellboost.py; do sleep 10; done` never exits — the waiter's own command
line contains the pattern, so `pgrep -f` matches it. Three background waits hung on this after
the job had already finished. Use `pgrep -f "python.*<script>"`, or check the log's last line.
