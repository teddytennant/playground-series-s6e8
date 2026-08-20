"""w38b -- vet every OOF artefact the w38a kernel-endpoint sweep turned up.

w38a ran `kaggle kernels output` over 244 prioritised refs. The pool ledger this workspace
has been carrying was built almost entirely on `kaggle datasets download`, which w34 3.1
showed is a DIFFERENT endpoint returning DIFFERENT files, so most of the pool has never had
the kernel endpoint run over it at all.

This is the cheap half plus maxcorr. Gates, in the order they kill things:

  g1 shape      -- OOF must be exactly N_TR rows, test exactly N_TE.
  g2 label      -- if the file carries the target, it must match ours EXACTLY (row indexing).
  g3 solo AUC   -- must be credible (<= 0.9720) and within w29's strength band of the pack
                   median; recorded either way, since strength is a soft gate.
  g4 dup        -- max |corr| on the logit scale against EVERY member in the pack. This is
                   what kills most public OOF: it is a member we already hold under another
                   name. Asserts the pack size, per RESEARCH's ext_members/ext_members2 note.

Early-stopping-on-validation and foreign-partition are NOT decided here -- they need the
notebook source, and only the survivors of g1-g4 are worth pulling source for. That is w38c.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_D = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_D, "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, to_logit  # noqa: E402

SRC = os.path.join(ROOT_D, "notebooks", "w38", "out")
N_TR, N_TE = 691369, 296302
CREDIBLE_MAX = 0.9720          # import_ext2's gate 3, unchanged
PACK_MEDIAN_SOLO = 0.966319    # w35c
SKIP_COLS = {"id", "fold", "fold_id", TARGET, "target", "label", "y"}


def _vecs(path):
    """Return {colname: float vector} for every numeric column of the right length."""
    out = {}
    if path.endswith(".npy"):
        try:
            a = np.load(path, allow_pickle=False)
        except Exception:
            return out
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a.ravel()
        if a.ndim == 1 and a.shape[0] in (N_TR, N_TE):
            out[""] = a.astype(np.float64)
        return out
    if path.endswith((".csv", ".parquet")):
        try:
            df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        except Exception:
            return out
        if len(df) not in (N_TR, N_TE):
            return out
        for c in df.columns:
            if c.lower() in SKIP_COLS or not np.issubdtype(df[c].dtype, np.number):
                continue
            v = df[c].to_numpy(np.float64)
            if np.isfinite(v).all() and v.std() > 0:
                out[c] = v
        # keep the label around for gate 2
        for c in df.columns:
            if c.lower() in (TARGET, "target", "label", "y") and len(df) == N_TR:
                out["__label__"] = df[c].to_numpy(np.float64)
    return out


def _stem(fn):
    """Normalise a filename to a stem that an oof/test pair should share."""
    s = re.sub(r"\.(npy|csv|parquet)$", "", fn)
    for pat in (r"^oof[_-]?", r"[_-]?oof$", r"^test[_-]?", r"[_-]?test$",
                r"^pred[_-]?", r"[_-]?preds?$", r"[_-]?predictions?$", r"^sub(mission)?[_-]?"):
        s = re.sub(pat, "", s)
    return s.strip("_-").lower()


def collect(d):
    """Pair up (oof, test) candidates inside one kernel-output directory."""
    # nested: stephentarter ships its OOF under predictions/, so a top-level listdir misses it
    files = []
    for root, _dirs, fns in os.walk(d):
        for fn in sorted(fns):
            files.append(os.path.relpath(os.path.join(root, fn), d))
    files = sorted(files)
    oof_side, test_side, label = {}, {}, None
    for rel in files:
        fn = os.path.basename(rel)
        p = os.path.join(d, rel)
        if not os.path.isfile(p) or not fn.endswith((".npy", ".csv", ".parquet")):
            continue
        if fn.startswith("sample_submission"):
            continue
        vs = _vecs(p)
        if "__label__" in vs:
            label = vs.pop("__label__")
        for col, v in vs.items():
            key = (_stem(fn) + ("_" + col.lower() if col else "")).strip("_")
            key = re.sub(r"^(oof|test|pred)_", "", key)
            (oof_side if len(v) == N_TR else test_side)[key] = v
    pairs = {}
    for k, o in oof_side.items():
        t = test_side.get(k)
        if t is None and len(test_side) == 1 and len(oof_side) == 1:
            t = next(iter(test_side.values()))   # the single-model dir: submission.csv IS its test
        if t is not None:
            pairs[k] = (o, t)
        else:
            pairs[k] = (o, None)
    return pairs, label


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    assert len(y) == N_TR

    names, O, _ = load_members(
        y, N_TE,
        extra_dirs=tuple(os.path.join(DATA, x) for x in sorted(os.listdir(DATA))
                         if x.startswith("ext_members") and not x.endswith("es")),
        drop=set(DEFAULT_DROP))
    print(f"pack for the dup gate: {len(names)} members", flush=True)
    assert len(names) >= 180, f"pack is only {len(names)} members -- the dup gate would be blind"
    Z = to_logit(O)
    Z = (Z - Z.mean(0)) / Z.std(0)
    del O

    rows = []
    for sub in sorted(os.listdir(SRC)):
        d = os.path.join(SRC, sub)
        if not os.path.isdir(d):
            continue
        try:
            pairs, label = collect(d)
        except Exception as e:                     # noqa: BLE001
            print(f"{sub}: collect failed {e}")
            continue
        if not pairs:
            continue
        g2 = "n/a" if label is None else ("OK" if int((label.astype(np.int8) != y).sum()) == 0
                                          else "LABEL MISMATCH")
        for k, (o, t) in sorted(pairs.items()):
            auc = roc_auc_score(y, o)
            z = to_logit(o.reshape(-1, 1)).ravel()
            z = (z - z.mean()) / z.std()
            j = int(np.abs(Z.T @ z / len(z)).argmax())
            mc = float(abs(Z[:, j] @ z / len(z)))
            v = "ok"
            if g2 == "LABEL MISMATCH":
                v = "LABEL MISMATCH"
            elif t is None:
                v = "NO TEST VECTOR"
            elif auc > CREDIBLE_MAX:
                v = "NOT CREDIBLE"
            elif mc > 0.995:
                v = "DUP"
            elif abs(auc - PACK_MEDIAN_SOLO) > 0.005:
                v = "WEAK" if auc < PACK_MEDIAN_SOLO else "ok"
            rows.append(dict(kernel=sub, stream=k, solo_auc=round(auc, 6),
                             maxcorr=round(mc, 6), nearest=names[j],
                             has_test=t is not None, g2_label=g2, verdict=v))
            print(f"  {v:14s} {sub[:44]:44s} {k[:22]:22s} solo {auc:.6f} "
                  f"maxcorr {mc:.5f} -> {names[j]}", flush=True)
    df = pd.DataFrame(rows).sort_values(["verdict", "maxcorr"])
    df.to_csv(os.path.join(HERE, "w38b_vet.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))
    print("\nSURVIVORS (verdict ok):")
    print(df[df.verdict == "ok"].to_string(index=False))


if __name__ == "__main__":
    main()
