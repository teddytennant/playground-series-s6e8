"""w26f — sweep the L2 penalty on the STANDARDISED (converged) 187-member combiner.

Pre-registered in full at experiments/w26_prereg.txt ADDENDUM D1 BEFORE this ran once. The
grid, the hypotheses H-D1/H-D2/H-D3, the selection geometry and the ship rule are fixed there
and are not renegotiated here.

Short version. w23a tested "an L2 penalty helps at 187 members" and falsified it flat -- every
lam from 1e-8 to 1e-5 read -1.41 … +3.38e-6 and every one sign-flipped across reps. But that
sweep ran on the UNSTANDARDISED design, which w23 §1 showed in the same wave is (i) not at its
optimum, because tol=1e-4 is a max-gradient rule and the hybrid columns have sd 1.82 … 27.59,
and (ii) anisotropically shrunk, the widest member penalised ~230x less than the narrowest.
The isotropic converged design has never been swept. That is what this does.

SELECTION IS ON HELD-OUT ROWS, NOT ON CROSS-FITTED CV (prereg §D3). The combiner is cross-
fitted on the same frozen folds that produced its member OOF columns, and whether that leak
rewards a better-converged fit is exactly what w25d is testing; picking a hyperparameter on
the suspect instrument would be the rogii-wellbore failure with a different axis label. This
reuses w25d's splits exactly, so w25d's std arm IS the C=1.0 row and costs nothing to check.

Two arms, run in this order, because only the first one selects:

    --kind K   THE SELECTOR. 5 reps x 80/20 StratifiedShuffleSplit(random_state SEED0+r),
               fit on the pool, scored on rows the fit never saw, paired within rep.
    --cv K     THE COMPANION READING, required by prereg §D3 ("cross-fitted CV is REPORTED
               alongside for every cell but is not the selector"). Cross-fits the same C grid
               on the frozen folds. The gap between the two columns, read over a whole C
               curve rather than w25d's single point, is a second free reading on the leak.

Checkpoints after every cell and resumes from disk: background jobs do not survive the end of
a run's session here and `setsid` is not installed.

    .venv/bin/python experiments/w26f_csweep.py --kind hybrid     # selector, per transform
    .venv/bin/python experiments/w26f_csweep.py --cv   hybrid     # companion, per transform
    .venv/bin/python experiments/w26f_csweep.py --report

⚠ ARTEFACTS ARE KEYED BY seed0. `--seed0 2000` is the FRESH-SPLIT re-run prereg §D5 demands
before any winner is believed, and it writes to its own CSV and its own holdout directory.
The first version of this file keyed cells by (kind, C, rep) alone, which would have made the
fresh-split run find all 35 cells "already on disk" and re-print the seed-1000 numbers as if
they were the confirmation -- the same class of defect as w24 (a re-cut silently consuming
the record it is supposed to be checked against). Do not un-key these paths.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402
from blend_lab import HONEST_DROP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = 0.20
KINDS = ("hybrid", "rankraw", "rescale")
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)   # prereg §D2 H-D3, FIXED, not to be widened
REPS = 5
SEED0 = 1000                 # w25d's splits exactly, so its std arm is the C=1.0 row
CV_GATE = 0.9701182          # w26d: CV needed for an even-money shot at the record, family h3
CVDIR = os.path.join(HERE, "w26f_cvoof")

# The cross-fitted CV of the C=1.0 standardised build, read off logs_w23f_stdbuild4.txt --
# the run that actually produced submissions/w23_ad187std_{hybrid,rankraw,rescale}.csv. The
# --cv arm's C=1.0 cells reproduce these or something is wrong with the fit path, and that is
# a free gate rather than a new experiment. Recorded to 6 dp because that is all the log has.
# NOTE ON CONVENTION: blend_lab.build(std=True) takes the column scale from ALL training rows
# and then cross-fits; this file takes it from the fold-train rows only, which is the clean
# version. sd over 553k vs 691k rows differs by far less than the 1e-6 these are compared at,
# so a mismatch here is a real defect and not the convention gap.
W23F_STD_CV = {"hybrid": 0.970098, "rankraw": 0.970092, "rescale": 0.970094}


def csv_path(seed0):
    return os.path.join(HERE, f"w26f_csweep_s{seed0}.csv")


def holddir(seed0):
    return os.path.join(HERE, f"w26f_hold_s{seed0}")


def splits_for(y, reps, seed0=SEED0):
    out = []
    for r in range(reps):
        ip, ih = next(StratifiedShuffleSplit(1, test_size=HOLD, random_state=seed0 + r)
                      .split(np.zeros(len(y)), y))
        out.append((np.sort(ip), np.sort(ih)))
    return out


def _read(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)


def load_done(seed0):
    """Deduped on (kind, C, rep). A cell whose .npy went missing is recomputed and appends a
    second row; keeping the last one means the CSV and the .npy on disk always agree."""
    df = _read(csv_path(seed0), ["kind", "C", "rep", "hold_auc", "n_iter", "secs", "seed0"])
    if not df.empty:
        df = df.drop_duplicates(subset=["kind", "C", "rep"], keep="last").reset_index(drop=True)
    return df


def load_cv():
    df = _read(os.path.join(HERE, "w26f_cv.csv"),
               ["kind", "C", "cv_auc", "n_iter_max", "secs"])
    if not df.empty:
        df = df.drop_duplicates(subset=["kind", "C"], keep="last").reset_index(drop=True)
    return df


def cell_path(kind, C, rep, seed0):
    return os.path.join(holddir(seed0), f"{kind}__C{C:g}__rep{rep}.npy")


def cv_cell_path(kind, C):
    return os.path.join(CVDIR, f"{kind}__C{C:g}.npy")


def _atomic_npy(path, arr):
    np.save(path + ".tmp.npy", np.asarray(arr, dtype="float32"))
    os.replace(path + ".tmp.npy", path)


def _atomic_csv(path, df):
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def append_row(row, d, kind, seed0):
    """One CSV row + ONE .npy per cell, both written atomically so a kill mid-write cannot
    leave a truncated artefact that the resume path then trusts.

    Deliberately one file per cell rather than one growing .npz per transform. A single npz
    would be recompressed from scratch on every one of the 35 cells, and by the last cell that
    is ~19 MB of float32 re-deflated to save 0.5 MB of new data -- tens of seconds of pure
    waste per cell on a sweep whose whole point is that standardised fits are cheap.
    """
    os.makedirs(holddir(seed0), exist_ok=True)
    _atomic_npy(cell_path(kind, row["C"], row["rep"], seed0), d)
    _atomic_csv(csv_path(seed0),
                pd.concat([load_done(seed0), pd.DataFrame([row])], ignore_index=True))


def load_Z(kind, note=""):
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    Z, _t = transform(O, T, kind)
    Z = Z.astype("float64")
    del O, T, _t, te
    gc.collect()
    print(f"{len(names)} members, {kind} Z {Z.shape}, {time.time()-t0:.0f}s {note}", flush=True)
    return y, Z


def run_kind(kind, seed0):
    """THE SELECTOR ARM. Held-out rows, paired within rep. prereg §D3."""
    t0 = time.time()
    df = load_done(seed0)
    # A cell counts as done only when BOTH artefacts are present. A cell killed between the
    # .npy write and the CSV write is recomputed; the cost is one 15s fit.
    done = {(r.C, r.rep) for r in df[df.kind == kind].itertuples()
            if os.path.exists(cell_path(kind, r.C, r.rep, seed0))}
    todo = [(c, r) for c in GRID for r in range(REPS) if (c, r) not in done]
    if not todo:
        print(f"{kind}: all {len(GRID)*REPS} cells already on disk (seed0={seed0})")
        return
    print(f"{kind}: {len(todo)} of {len(GRID)*REPS} cells to run, seed0={seed0}", flush=True)

    y, Z = load_Z(kind, f"seed0={seed0}")
    for rep, (ip, ih) in enumerate(splits_for(y, REPS, seed0)):
        if all((c, rep) in done for c in GRID):
            continue
        Ztr, Zho, ytr, yho = Z[ip], Z[ih], y[ip], y[ih]
        s = Ztr.std(0)
        s[s <= 0] = 1.0
        Ztr /= s            # standardised design: the penalty is isotropic and lbfgs converges
        Zho /= s
        for C in GRID:
            if (C, rep) in done:
                continue
            t = time.time()
            m = LogisticRegression(max_iter=5000, C=C, tol=1e-4).fit(Ztr, ytr)
            el = time.time() - t
            d = m.decision_function(Zho)
            auc = roc_auc_score(yho, d)
            ni = int(np.ravel(m.n_iter_)[0])
            append_row(dict(kind=kind, C=C, rep=rep, hold_auc=auc, n_iter=ni, secs=el,
                            seed0=seed0), d, kind, seed0)
            print(f"  C {C:<6g} rep{rep}  auc {auc:.10f}  iters {ni:5d}  {el:6.1f}s",
                  flush=True)
            del m
        del Ztr, Zho
        gc.collect()
    print(f"{kind} done, {time.time()-t0:.0f}s")


def run_cv(kind):
    """THE COMPANION ARM — prereg §D3, "REPORTED alongside for every cell but is not the
    selector". Cross-fits each C on the frozen folds (StratifiedKFold(5, seed 42), the same
    folds that produced the member OOF columns), standardising on the fold-train rows.

    This is NOT allowed to pick C. Its job is (a) the ship gate's second half, since a cell
    ships only if it ALSO clears CV_GATE, and (b) a C-curve reading of the same holdout-vs-CV
    gap that w25d measures at one point.
    """
    t0 = time.time()
    df = load_cv()
    done = {r.C for r in df[df.kind == kind].itertuples()
            if os.path.exists(cv_cell_path(kind, r.C))}
    todo = [c for c in GRID if c not in done]
    if not todo:
        print(f"{kind}: all {len(GRID)} cross-fitted cells already on disk")
        return
    print(f"{kind}: {len(todo)} of {len(GRID)} cross-fitted cells to run", flush=True)

    y, Z = load_Z(kind, "cross-fitted arm")
    folds = get_folds(y)
    os.makedirs(CVDIR, exist_ok=True)
    for C in todo:
        t = time.time()
        mo = np.zeros(len(y))
        nis = []
        for itr, iva in folds:
            A = Z[itr]
            s = A.std(0)
            s[s <= 0] = 1.0
            A /= s
            m = LogisticRegression(max_iter=5000, C=C, tol=1e-4).fit(A, y[itr])
            B = Z[iva]
            B /= s
            mo[iva] = m.decision_function(B)
            nis.append(int(np.ravel(m.n_iter_)[0]))
            del A, B, m
            gc.collect()
        auc = roc_auc_score(y, mo)
        el = time.time() - t
        _atomic_npy(cv_cell_path(kind, C), mo)
        _atomic_csv(os.path.join(HERE, "w26f_cv.csv"),
                    pd.concat([load_cv(), pd.DataFrame([dict(
                        kind=kind, C=C, cv_auc=auc, n_iter_max=max(nis), secs=el)])],
                        ignore_index=True))
        tag = ""
        if C == 1.0 and kind in W23F_STD_CV:
            tag = (f"   [w23f build read {W23F_STD_CV[kind]:.6f}, "
                   f"delta {(auc - W23F_STD_CV[kind])*1e6:+.2f}e-6]")
        print(f"  C {C:<6g}  cvAUC {auc:.10f}  iters<={max(nis):5d}  {el:6.1f}s{tag}",
              flush=True)
    print(f"{kind} cv done, {time.time()-t0:.0f}s")


def report(seed0=SEED0):
    df = load_done(seed0)
    cv = load_cv()
    if df.empty and cv.empty:
        print("nothing on disk yet")
        return
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    sp = splits_for(y, REPS, seed0)

    per = {}
    if not df.empty:
        print(f"=== SELECTOR: held-out AUC by C, paired against C=1.0 (e-6), seed0={seed0} ===")
        for kind in KINDS:
            k = df[df.kind == kind]
            if k.empty:
                continue
            p = k.pivot(index="rep", columns="C", values="hold_auc")
            if 1.0 not in p.columns:
                print(f"  {kind}: no C=1.0 baseline yet, skipped")
                continue
            d6 = p.sub(p[1.0], axis=0) * 1e6
            per[kind] = {float(c): float(d6[c].mean()) for c in d6.columns}
            print(f"\n  {kind}")
            for c in sorted(d6.columns):
                v = d6[c].dropna()
                if not len(v):
                    continue
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                print(f"    C {c:<6g}  D {v.mean():+7.3f}e-6  se {se:5.3f}  "
                      f"{int((v > 0).sum())}/{len(v)} positive")

    percv = {}
    if not cv.empty:
        print("\n=== COMPANION (NOT THE SELECTOR): cross-fitted CV on the frozen folds ===")
        for kind in KINDS:
            k = cv[cv.kind == kind].sort_values("C")
            if k.empty:
                continue
            base = k[k.C == 1.0].cv_auc
            print(f"\n  {kind}")
            for r in k.itertuples():
                d6 = (r.cv_auc - base.iloc[0]) * 1e6 if len(base) else float("nan")
                print(f"    C {r.C:<6g}  cvAUC {r.cv_auc:.10f}  D {d6:+7.3f}e-6")
                percv.setdefault(kind, {})[float(r.C)] = float(r.cv_auc)
            if len(base) and kind in W23F_STD_CV:
                print(f"    reproduction gate: C=1.0 vs logs_w23f_stdbuild4 "
                      f"{W23F_STD_CV[kind]:.6f} -> "
                      f"{(base.iloc[0]-W23F_STD_CV[kind])*1e6:+.2f}e-6")

    have = [k for k in KINDS if not df[df.kind == k].empty] if not df.empty else []
    if len(have) == len(KINDS):
        print("\n=== the h3 MIX (rank-average of the three stacks) on held-out rows ===")
        rows = []
        for C in GRID:
            ds = []
            for rep, (_ip, ih) in enumerate(sp):
                paths = [cell_path(kk, C, rep, seed0) for kk in KINDS]
                if not all(os.path.exists(q) for q in paths):
                    ds = None
                    break
                e = np.mean([rankdata(np.load(q)) for q in paths], 0)
                ds.append(roc_auc_score(y[ih], e))
            if ds is not None:
                rows.append(dict(C=C, mix_auc=float(np.mean(ds)),
                                 per_rep=[float(v) for v in ds]))
        if rows:
            mx = pd.DataFrame(rows)
            base = mx[mx.C == 1.0].mix_auc
            if len(base):
                mx["D6"] = (mx.mix_auc - base.iloc[0]) * 1e6
            # the mix's cross-fitted twin, same rank-average over the three cross-fitted OOFs
            mixcv = []
            for C in GRID:
                paths = [cv_cell_path(kk, C) for kk in KINDS]
                mixcv.append(float(roc_auc_score(
                    y, np.mean([rankdata(np.load(q)) for q in paths], 0)))
                    if all(os.path.exists(q) for q in paths) else float("nan"))
            mx["mix_cv"] = mixcv
            mx["cv_clears_gate"] = mx.mix_cv >= CV_GATE
            print(mx[["C", "mix_auc", "D6", "mix_cv", "cv_clears_gate"]].to_string(index=False))
            best = mx.loc[mx.mix_auc.idxmax()]
            print(f"\nbest cell C={best.C:g} at {best.D6:+.3f}e-6 vs C=1.0")
            print("PREREG §D2 H-D1 registered -2 to +5e-6 and EXPECTED FAILURE; anything "
                  "above +8e-6\nis to be disbelieved and re-run on fresh splits "
                  "(--seed0 2000) before it is built on.")
            print(f"PREREG §D3 SHIP RULE: a cell ships only if it wins HERE *and* its "
                  f"cross-fitted CV\nclears {CV_GATE:.7f}. Both columns are above; neither "
                  f"alone is sufficient, and the\ncross-fitted one NEVER selects.")
            mx.to_csv(os.path.join(HERE, f"w26f_mix_s{seed0}.csv"), index=False)
    json.dump(dict(grid=list(GRID), reps=REPS, seed0=seed0, per_transform=per,
                   per_transform_cv=percv, cv_gate=CV_GATE,
                   w23f_repro_target=W23F_STD_CV),
              open(os.path.join(HERE, f"w26f_csweep_s{seed0}.json"), "w"), indent=1)
    print(f"\nwrote experiments/w26f_csweep_s{seed0}.json"
          + (f", w26f_mix_s{seed0}.csv" if len(have) == len(KINDS) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=KINDS, help="selector arm, one transform")
    ap.add_argument("--cv", default=None, choices=KINDS,
                    help="companion cross-fitted arm, one transform. NOT a selector.")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed0", type=int, default=SEED0,
                    help="1000 = w25d's splits (default). 2000 = the FRESH splits prereg "
                         "§D5 requires before believing any winner. Artefacts are keyed by "
                         "this, so the two never share a cell.")
    a = ap.parse_args()
    if a.report:
        report(a.seed0)
    elif a.cv:
        run_cv(a.cv)
    elif a.kind:
        run_kind(a.kind, a.seed0)
    else:
        ap.error("one of --kind, --cv, --report")


if __name__ == "__main__":
    main()
