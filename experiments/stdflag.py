"""The combiner-standardisation flag, derived from BUILD PROVENANCE rather than from the
filename spelling. Pure functions, no import-time side effects — import this, do not copy it.

⚠ WHY THIS MODULE EXISTS (w28, 2026-08-19). The `standardised` term in the w25f CV→LB model
carries **-27.43e-6**, the largest single coefficient after CV itself and 3.3 LB reporting
steps. Whether a file gets it is therefore the difference between "worth a slot" and
"hopeless", and the flag has now been wrong TWICE, in opposite directions:

1. w25f/w26d classified with a hard-coded whitelist of seven `w23_*` names. Every build made
   after w23 scored as UNstandardised and collected +27.43e-6 it had not earned. Fixed in
   w27 slot 3 by replacing the whitelist with the substring test `"std" in stem`.

2. That substring test is ALSO wrong, in the same direction, and w28 caught it: the six
   `w27_ad188raw*` files were built by `experiments/w27k_ctrawctl.sh`, which passes
   `--standardize` explicitly — the "raw" in the name refers to the CT-RAW MEMBER
   (`lat_ctraw_r400` in place of `lat_ctfix_r400`), not to a raw combiner. They contain no
   substring "std", so they scored as unstandardised and were handed +27.43e-6 each. Under
   that error `w27_ad188raw` priced at **P(beat the account best) = 0.548**, the best number
   on disk and enough to reorder a whole send day. Corrected it is 5.9e-4.

The lesson is that a filename is not provenance. What actually determines the flag here is
WHEN the file was built: `blend_lab --standardize` entered the chain at w23 and **every**
build this workspace has made since passes it. So the flag is a property of the wave number,
which every submission stem carries in its `wNN_` prefix, and non-`w` stems (`blend*`,
`stack*`) all predate w23 by construction.

The gate at the bottom of this file re-derives the flag for the 60 files w25f was fitted on
and asserts it reproduces w25f's whitelist EXACTLY, so the model's own rows cannot drift.
"""
from __future__ import annotations

import re

# The seven files w25f actually fitted the `standardised` coefficient on. Frozen: the model
# JSON is fitted against exactly this labelling and re-labelling these rows would invalidate
# every stored coefficient.
STD_FILES = {"w23_ad187stdcorr", "w23_ad187std_h3", "w23_ad187std", "w23_ad187std_logit",
             "w23_ad187std_h3_hybrid", "w23_ad187std_h3_rankraw", "w23_ad187std_h3_rescale"}

FIRST_STD_WAVE = 23          # `--standardize` entered the build chain at w23 and never left.

_WAVE = re.compile(r"^w(\d+)[a-z]?_")


def wave(stem: str):
    """The wave number a submission stem was built in, or None for pre-wave names."""
    m = _WAVE.match(stem)
    return int(m.group(1)) if m else None


def is_std(stem: str) -> bool:
    """Was this file's meta-combiner fitted on a unit-sd design matrix?

    Provenance rule: a build is standardised iff it was made at or after w23. `blend*` and
    `stack*` stems carry no wave and all predate w23.
    """
    w = wave(stem)
    return w is not None and w >= FIRST_STD_WAVE


def family_suffix(stem: str) -> str:
    """The suffix rule EXACTLY as w25f fitted it. Use this only to reproduce w25f's own rows."""
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit"):
        if stem.endswith("_" + f):
            return f
    if stem.endswith("_w") or stem.endswith("_w2") or stem in ("blend156w", "blend156w2"):
        return "w"
    return "ens4"          # a bare stem is the rank-average of all four transforms


# ⚠ `corr` IS NOT A TRANSFORM, so no suffix rule can classify a `*corr` file (w26e). Every one
# of them is the 5-arm `c_avg` correction applied to an **h3** mix, and the suffix rule drops
# them into `ens4` -- putting the h3 member of a matched h3/ens4 pair into the ens4 group next
# to its own counterpart, and adding a spurious +13.9e-6 to its predicted LB. w26e found this
# and built the map; w28 found that the map was never extended to the two `w27_*` corr files
# built since, which are the top TWO entries of the send queue.
#
# EVERY stem containing "corr" must appear here. `require_corr_registered()` enforces it.
CORR_MAP = {
    "w21_ad187corr": "h3",              # w21 §2, the h3-side file of w21's matched pair
    "w21_ad187corr_ens4": "ens4",       # w21 §9, its genuine ens4 counterpart
    "w22_ad187corr_rankraw": "rankraw",  # a genuine single-transform build
    "w23_ad187stdcorr": "h3",           # the standardised 187 twin of w21_ad187corr
    "w27_ad188stdcorr": "h3",           # w27 slot 6, same construction on the 188 pack
    "w27_ad190stdcorr": "h3",           # w27 slot 7, same construction on the 190 pack
    "w29_ad194stdcorr": "h3",           # w29 slot 10, same construction on the 194 pack
    # w30 slot 1: the same 5-arm correction on the NON-h3 bases of the 194 pack. These are the
    # first corrected files whose base is genuinely not h3, so their family is NOT "h3" -- the
    # correction is applied to the transform named in the suffix and that is the family it
    # belongs to. Registered in the same commit as the build, per the standing rule.
    "w29_ad194stdcorr_ens4": "ens4",
    "w29_ad194stdcorr_rankraw": "rankraw",
    "w29_ad194stdcorr_rescale": "rescale",
    # w34 slot 5: the same 5-arm correction on the h3 base of the om_ftt packs. Same
    # construction as w29_ad194stdcorr, one member wider.
    "w34_ad195stdcorr": "h3",           # 194 + om_ftt
    "w34_ad196stdcorr": "h3",           # 194 + om_ftt + om_cat
    # w36 slot 6: same 5-arm correction on the h3 base of the four-clean-member pack. BOTH
    # names are registered because which one exists is decided at run time by the rule
    # pre-registered in w36b_prereg.txt -- 199 if the ravi pair clears its sign gate, else
    # 197. Registered BEFORE the build so the standing "same commit as the build" rule cannot
    # be missed by a run that only sees one of the two names on disk.
    "w36_ad199stdcorr": "h3",           # 195 + ravi_xgb1c + ravi_lgbm1c + ram_hgb + ram_lgb
    "w36_ad197stdcorr": "h3",           # 195 + ram_hgb + ram_lgb
    # w48 slot 8, 2026-08-21. THESE THREE WERE MISSING AND THE GAP WAS NOT COSMETIC: two of
    # them, w40_ad211stdcorr and w38_ad202stdcorr, are slots 9 and 10 of the 08-22 send list
    # registered in w47b_prereg.txt, and require_corr_registered() is an ASSERT reached at
    # import time by w26d_queueprice -- so w26d, w39b, w39c and w39d all died on it and the
    # registered ten could not have been priced. Caught by re-running the whole chain end to
    # end rather than by reading it. Classified the same way as every entry above, from the
    # build script rather than the suffix: all three runners set W21A_BASE="${NAME}_h3", so
    # the corrected base is h3 and the family is h3.
    "w38_ad202stdcorr": "h3",           # w38d_run.sh, base w38_ad202std_h3
    "w40_ad211stdcorr": "h3",           # w40f_run.sh, base w40_ad211std_h3
    "w42_ad217stdcorr": "h3",           # w42e_run.sh, base w42_ad217std_h3
                                        # ⛔ ARM 217 -- WANTED-ineligible, w42b_prereg + w48d
    # w51: ARM 216 = ARM 217 minus `hboyang_mix` (w50a_run.sh). W21A_BASE was
    # `w50_ad216std_h3`, so the corrected base is h3 and the family is h3.
    # ⛔ ARM 216 INHERITS w42b's WANTED-INELIGIBILITY -- w50_prereg Sec.5 says so explicitly:
    # five of the six ext_members16 authors still have unread es-on-val status and dropping
    # the sixth does not discharge that clause.
    "w50_ad216stdcorr": "h3",           # w50a_run.sh, base w50_ad216std_h3
}


def family(stem: str) -> str:
    """The CORRECTED transform family: the suffix rule with the `*corr` map applied.

    Pair this with w26e's refitted coefficients (`w26e_famfix.json` -> `coefs_new`, residual
    sd 8.349e-6), NOT with w25f's (`coefs_old`, 8.408e-6) -- w25f was fitted with the two
    known-wrong labels in place, so mixing its coefficients with these labels is neither model.
    """
    return CORR_MAP.get(stem, family_suffix(stem))


def require_corr_registered(stems) -> None:
    """Raise if any `*corr` stem is missing from CORR_MAP. A new one must be classified BY HAND."""
    missing = sorted(s for s in stems if "corr" in s and s not in CORR_MAP)
    assert not missing, ("unregistered *corr files -- classify each by reading its build "
                         f"script, do NOT let the suffix rule guess: {missing}")


def gate(stems) -> None:
    """Assert the provenance rule reproduces w25f's whitelist on the rows it was fitted on."""
    derived = {s for s in stems if is_std(s)}
    known = {s for s in stems if s in STD_FILES}
    extra = derived - known
    missing = known - derived
    assert not missing, f"provenance rule LOSES w25f standardised rows: {sorted(missing)}"
    # Files built after w25f was fitted are legitimately new positives; only pre-w25f ones
    # would be a contradiction, and every stem w25f saw is either in STD_FILES or pre-w23.
    late = {s for s in extra if (wave(s) or 0) < FIRST_STD_WAVE}
    assert not late, f"provenance rule INVENTS standardised rows: {sorted(late)}"


if __name__ == "__main__":
    import os
    import pandas as pd
    t = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "w25a_cvlb_full.csv")).dropna(subset=["cv"])
    fit = t[t.cv >= 0.97].stem.tolist()
    gate(fit)
    print(f"GATE PASS: provenance rule reproduces w25f's whitelist on all {len(fit)} fitted rows")
    old = lambda s: s in STD_FILES or "std" in s            # the w27-slot-3 substring rule
    for s in sorted(t.stem):
        if old(s) != is_std(s):
            print(f"  std DIFFERS from the substring rule: {s:28s} substring={old(s)} provenance={is_std(s)}")
        if family_suffix(s) != family(s):
            print(f"  fam CORRECTED by CORR_MAP:          {s:28s} suffix={family_suffix(s)} -> {family(s)}")
    require_corr_registered(t.stem)
    print("all *corr stems in w25a's table are registered in CORR_MAP")
