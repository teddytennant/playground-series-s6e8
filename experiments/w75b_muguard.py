"""w75b — the CV->LB predictor's CENTRING must come from the same board as its COEFFICIENTS.

WHY (w75, 2026-08-24). `w46c_predlb.py:59` re-derives MU from `w25a_cvlb_full.csv` at import,
while its coefficients are frozen in `w30b_corrterm.json`. Those two agree TODAY only because
w30b happened to be fitted on the board the CSV currently holds. That CSV has already been
regenerated once behind a script that depended on it: `w28a_cvlb_refresh.py` re-derived the
same constant from it, asserted it matched `w26e_famfix.json`'s stored mu -- true when it ran
-- and today the board has grown 60 -> 78 rows above the cv>=0.97 floor, the centring has moved
+4.91e-6, and w28a's own assert would FIRE. w28a cannot reproduce its own output.

⚠ A CONSTANT RE-DERIVED FROM A MUTABLE FILE IS NOT FROZEN, however loudly the file says it is.
The next regeneration silently de-centres the MANDATED predictor, and nothing in w46c would
say so: predictions would just shift by slope * d(mu) on every file at once, which looks
exactly like a real era step.

This guard is one file read. It does not refit anything and it does not touch w46c.

    .venv/bin/python experiments/w75b_muguard.py        # rc 0 = centring and coefficients agree
    .venv/bin/python experiments/w75b_muguard.py --selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def checks(w25, w30b, w26e, w28csv, w28json, era_shift, era_n):
    """Pure. Returns the list of complaints, so the self-test can feed it planted inputs."""
    bad = []
    live_mu = float(w25[w25.cv >= 0.97].cv.mean())
    live_n = int((w25.cv >= 0.97).sum())

    # C1 -- the live centring is the one w30b's coefficients were fitted with.
    if abs(live_mu - w30b["mu"]) > 1e-12:
        bad.append(f"C1 w46c's MU re-derived from w25a_cvlb_full.csv is {live_mu:.13f} but "
                   f"w30b's coefficients were fitted at {w30b['mu']:.13f} "
                   f"({(live_mu - w30b['mu'])*1e6:+.2f}e-6) — the predictor is DE-CENTRED")
    # C2 -- and the board itself is the board w30b was fitted on.
    if live_n != w30b["n"]:
        bad.append(f"C2 w25a_cvlb_full.csv holds {live_n} rows above the cv>=0.97 floor; "
                   f"w30b was fitted on {w30b['n']} — the fit board has been regenerated")

    # C3 -- w28a's frozen artefact must still reproduce UNDER ITS MODEL'S OWN mu. This is the
    # repaired path; it is what makes the C1/C2 failure diagnosable rather than merely loud.
    C, mu26 = w26e["coefs_new"], w26e["mu"]
    p = [(C["const"] + C["cv_e6"] * (r.cv - mu26) * 1e6 + C.get(f"fam[{r.fam}]", 0.0)
          + (C["standardised"] if r.std else 0.0)) * 1e-6 for r in w28csv.itertuples()]
    r6 = (w28csv.lb.values - np.array(p)) * 1e6
    f = (w28csv.in_w25f.values & (w28csv.cv.values >= 0.97))
    rsd = float(np.sqrt((r6[f] ** 2).sum() / (f.sum() - len(C))))
    n = (~w28csv.in_w25f.values) & (w28csv.cv.values >= 0.97)
    if abs(rsd - w28json["gate_resid_sd"]) > 1e-6:
        bad.append(f"C3a w28a gate_resid_sd {rsd:.6f} != stored {w28json['gate_resid_sd']:.6f}")
    if abs(r6[n].mean() - w28json["oos_mean"]) > 1e-6:
        bad.append(f"C3b w28a oos_mean {r6[n].mean():+.6f} != stored {w28json['oos_mean']:+.6f}")
    if float(np.abs(np.array(p) - w28csv.pred.values).max()) > 1e-12:
        bad.append("C3c w28a's stored per-row predictions do not reproduce under the frozen mu")

    # C4 -- the era term is still the n=5 one. w75a refreshed it to -23.77e-6 on n=27 and
    # DELIBERATELY did not re-parameterise (it feeds only send ordering, priced at zero, and
    # the WANTED margin is era-invariant). If a later run does change it, this fires so that
    # run must re-read every stored predicted-LB rather than inherit it silently.
    if era_n != 5 or abs(era_shift - (-29.8164)) > 0.01:
        bad.append(f"C4 w46c.ERA_SHIFT is {era_shift:+.4f}e-6 on n={era_n}, not the recorded "
                   f"-29.8164 on n=5 — every stored predicted LB downstream is now stale")
    return bad


def load():
    w25 = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
    w30b = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
    w26e = json.load(open(os.path.join(HERE, "w26e_famfix.json")))
    w28csv = pd.read_csv(os.path.join(HERE, "w28a_cvlb_full.csv"))
    w28json = json.load(open(os.path.join(HERE, "w28a_cvlb_refresh.json")))
    sys.path.insert(0, HERE)
    import w46c_predlb as W46
    return w25, w30b, w26e, w28csv, w28json, float(W46.ERA_SHIFT), int(W46.ERA_N)


if __name__ == "__main__":
    args = load()
    bad = checks(*args)

    if "--selftest" in sys.argv:
        # BOTH controls, from the start (w74 section 4).
        # (a) PASSES on the real, recorded-good state:
        assert not bad, f"self-test: the live state should be clean, got {bad}"
        # (b) FAILS on each planted defect, one at a time:
        w25, w30b, w26e, w28csv, w28json, es, en = args
        grown = pd.concat([w25, w25.iloc[[0]].assign(cv=0.9702)], ignore_index=True)
        b1 = checks(grown, w30b, w26e, w28csv, w28json, es, en)
        assert any(c.startswith("C1") for c in b1) and any(c.startswith("C2") for c in b1), b1
        b2 = checks(w25, w30b, {**w26e, "mu": w26e["mu"] + 1e-6}, w28csv, w28json, es, en)
        assert sum(c.startswith("C3") for c in b2) == 3, b2
        b3 = checks(w25, w30b, w26e, w28csv, w28json, -23.77, 27)
        assert any(c.startswith("C4") for c in b3), b3
        print(f"self-test OK — clean on the live state; fires on all three planted defects "
              f"({len(b1)} + {len(b2)} + {len(b3)} complaints)")
        sys.exit(0)

    for c in bad:
        print("  ⛔", c)
    print(f"w75b_muguard: {len(bad)} complaint(s)")
    sys.exit(1 if bad else 0)
