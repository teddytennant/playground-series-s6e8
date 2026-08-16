"""Regenerate the cross-fitted OOF vector for the shipped `w16e_aonly` file.

`w16e_aonly.py` printed its cross-fitted CV but never stored the OOF vector, so
`submissions/oof_w16e_aonly.npy` is missing and w16w cannot price the file. This
rebuilds it with the identical protocol -- same frozen SKF5 seed42 folds, same
coordinate ascent, same 0..0.02 grid -- and asserts that the AUC of the rebuilt
vector reproduces w16e_aonly.json's stored `cv` exactly. If it does not, the
vector is not written.

No model choices here. This is bookkeeping w16e should have done.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from w16b_cellweight import CELL_A, apply_w, ascend, fast_auc, pct, rule_cells  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    br = pct(np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64))
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    a_only = np.where(rule_cells(tr) == CELL_A, "A", "rest").astype(object)

    z = br.copy()
    for itr, iva in folds:
        z[iva] = apply_w(br, c, a_only, ascend(y, br, c, a_only, itr), iva)

    # ⚠ TWO CV CONVENTIONS EXIST IN THIS WORKSPACE AND THEY DIFFER.
    #   (a) base_auc + mean(per-fold dAUC)  -- what w16b and w16e print and store
    #   (b) pooled AUC of the cross-fitted OOF vector -- what w16q and w16t print
    # They are not the same number: (a) averages five per-fold AUCs, (b) scores one
    # pooled ranking. Both are legitimate; quoting one as the other is not.
    j = json.load(open(os.path.join(HERE, "w16e_aonly.json")))
    base_auc = fast_auc(y, br)
    conv_a = base_auc + j["xfit"]
    conv_b = fast_auc(y, z)
    print(f"(a) base + mean per-fold dAUC   {conv_a:.10f}   w16e_aonly.json cv "
          f"{j['cv']:.10f}   drift {(conv_a - j['cv'])*1e12:+.3f}e-12")
    print(f"(b) pooled cross-fitted OOF AUC {conv_b:.10f}")
    print(f"    convention (b) - (a) = {(conv_b - conv_a)*1e6:+.3f}e-6")
    if abs(conv_a - j["cv"]) >= 1e-15:
        print("*** convention (a) does not reproduce the stored cv — not writing ***")
        return
    path = os.path.join(SUB, "oof_w16e_aonly.npy")
    np.save(path, z)
    j["cv_pooled"] = float(conv_b)
    j["cv_convention"] = "cv = base_auc + mean(per-fold dAUC); cv_pooled = AUC of the OOF vector"
    json.dump(j, open(os.path.join(HERE, "w16e_aonly.json"), "w"), indent=1)
    print(f"wrote {path} and added cv_pooled to w16e_aonly.json")


if __name__ == "__main__":
    main()
