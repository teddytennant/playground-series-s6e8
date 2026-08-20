"""w36c -- the CHEAP half of the vetting gate for the 10 refs w36 re-pulled with
`kaggle kernels output` (the endpoint w34 §3.1 found had never been tried on rows RESEARCH
had filed as "submission only" from `kaggle datasets download`).

Cheap on purpose: solo AUC + per-fold AUC under our frozen SKF5 against the author's own
printed per-fold numbers. It does NOT compute maxcorr, because the 195-member pack costs
~2 GB of rank matrices and the w36b build owns that memory until it finishes. maxcorr is
deferred to whichever run promotes a member out of here.

Gate 4 (per-fold reproduction) is the load-bearing one: it is a foreign-partition test that
needs nothing but the member and our folds, and a foreign partition is worse than es-on-val.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_D = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_D, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402

SRC = os.path.join(ROOT_D, "notebooks", "w36", "out")
N_TR, N_TE = 691369, 296302

# author's printed per-fold AUCs, transcribed from the kernel log BY HAND, in fold order
CAND = {
  "mkt_realmlp": dict(
      d="mohankrishnathalla_s6e8-realmlp-oof-saver", oof="oof_realmlp.npy", test="test_realmlp.npy",
      folds=[0.95847, 0.95875, 0.95749, 0.95849, 0.95801], overall=0.95813,
      es="",  # `Trainer.fit stopped: max_epochs=512 reached` on ALL FIVE folds -- ES never fired
  ),
  "mkt_mlp": dict(
      d="mohankrishnathalla_s6e8-tabm-oof-saver", oof="oof_mlp.npy", test="test_mlp.npy",
      folds=[0.94078, 0.94080, 0.94201, 0.94265, 0.94183], overall=0.94142,
      es="`Early stop at epoch 85` on every fold, and a per-fold checkpoint is saved",
  ),
  "omid_tabm": dict(
      d="omidbaghchehsaraei_tabm-for-predicting-smartphone-addiction", oof="oof.csv",
      oofcol="oof_pred", test=None,
      folds=[0.96673, 0.96778, 0.96772, 0.96840, 0.96732], overall=0.96751,
      es="`New best epoch!` then `Restoring best model` per fold -- explicit best-epoch selection",
  ),
}


def main():
    y = load_raw()[0][TARGET].to_numpy(np.int8)
    assert len(y) == N_TR
    folds = list(get_folds(y))
    rows = []
    for nm, c in sorted(CAND.items()):
        d = os.path.join(SRC, c["d"])
        p = os.path.join(d, c["oof"])
        if not os.path.exists(p):
            print(f"{nm}: MISSING {p}"); continue
        if c["oof"].endswith(".npy"):
            o = np.load(p).astype(np.float64)
        else:
            df = pd.read_csv(p)
            assert (df["id"].to_numpy() == np.arange(N_TR)).all(), f"{nm}: id order is not 0..N-1"
            # gate 2: the file carries the label, so check it against ours rather than trusting it
            if TARGET in df.columns:
                mm = int((df[TARGET].to_numpy(np.int8) != y).sum())
                print(f"  {nm}: gate2 label mismatches = {mm}")
                assert mm == 0, f"{nm}: {mm} label mismatches -> not our row indexing"
            o = df[c["oofcol"]].to_numpy(np.float64)
        assert len(o) == N_TR, f"{nm}: {len(o)} rows, expected {N_TR}"
        ours = [roc_auc_score(y[va], o[va]) for _, va in folds]
        overall = roc_auc_score(y, o)
        diff = max(abs(a - b) for a, b in zip(ours, c["folds"]))
        rows.append(dict(member=nm, solo_auc=overall, log_overall=c["overall"],
                         g4_maxdiff=diff, g4_ok=bool(diff < 1e-4), es_on_val=c["es"]))
        print(f"{nm:14s} solo {overall:.6f} (log {c['overall']:.5f})  g4 {diff:.2e}"
              f"  {'OUR FOLDS' if diff < 1e-4 else 'FOREIGN PARTITION'}"
              f"  {'clean' if not c['es'] else 'ES-ON-VAL'}")
        print(f"    ours [{' '.join(f'{x:.5f}' for x in ours)}]")
        print(f"    log  [{' '.join(f'{x:.5f}' for x in c['folds'])}]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "w36c_lightvet.csv"), index=False)
    print(f"\npack median solo (w35c, unchanged pack) 0.966319 -- w29's acceptance gate wants a")
    print(f"member within 0.005 of it AND decorrelated.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
