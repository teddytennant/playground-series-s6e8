"""w42c -- pair, then gate, every importable stream in the 52 unmined CANDIDATE dirs.

Applies experiments/w42b_prereg.txt, which was committed before this file existed:

    ext_members16 = every candidate with  maxcorr < 0.99  AND  solo > 0.966319

Both constants are lifted unchanged from w40d_prereg.txt. This script does not choose and
does not tune; it pairs by fixed conventions, measures, and reports. Unpairable streams are
listed as UNPAIRABLE rather than resolved by a judgement call.

PAIRING CONVENTIONS (fixed, and applied identically to every dir)
  .npy   oof_X.npy / X_oof.npy  <->  test_X.npy / X_test.npy       label X
  .npz   key oof_X              <->  key test_X                    label X
  table  one eligible column    ->   label from the FILE stem
         many eligible columns  ->   label from each COLUMN name
  labels are normalised: oof_/test_/sub_/submission_/preds_ affixes and a leading
  mean_ / cv_mean_ are stripped, so `oof_prediction` meets `cv_mean_prediction`.
  LAST RESORT, and only when a dir produced NO matched pair at all: if exactly one
  oof-length and one test-length vector exist, pair them. A dir with leftovers on both
  sides after a successful match is NOT resolved this way -- that would be a guess.
"""
from __future__ import annotations
import os, re, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402

OUTDIR = os.path.join(ROOT, "notebooks", "w40", "out")
N_TR, N_TE = 691369, 296302
MAXCORR, SOLO = 0.99, 0.966319          # w40d constants, unchanged
SKIP = ("yadoy666/94-verified",)         # already mined -> ext_members15

DROP_TR = {"id", "target", "y", "addicted_label", "fold_nb", "fold", "kfold"}
DROP_TE = {"id", "fold_nb", "fold", "kfold"}
DROP_RE = re.compile(r"(^fold_?\d+$|_std$|uncertainty)")

AFFIX = r"(oof|test|subs?|submissions?|preds?|predictions?)"

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\.(npy|npz|csv|gz|parquet)$", "", s)
    s = re.sub(r"^(cv_)?mean_", "", s)
    for _ in range(4):
        s = re.sub(r"^" + AFFIX + r"[_-]", "", s)
        s = re.sub(r"[_-]" + AFFIX + r"$", "", s)
    if re.fullmatch(AFFIX, s):          # a bare affix carries no identity
        s = ""
    return s.strip("_-")

def eligible(cols, n):
    drop = DROP_TR if n == N_TR else DROP_TE
    return [c for c in cols if str(c).lower() not in drop and not DROP_RE.search(str(c).lower())]

def side_of(name: str):
    n = name.lower()
    if "oof" in n: return "oof"
    if "test" in n or "sub" in n: return "test"
    return None

def collect(d):
    """-> (oof_items, test_items) as [(label, loader)]; loader() returns a 1-D float array."""
    oof, test = [], []
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            p, low = os.path.join(root, f), f.lower()
            if low.endswith(".log") or low.startswith("sample_submission"):
                continue
            try:
                if low.endswith(".npy"):
                    a = np.load(p, mmap_mode="r")
                    if a.ndim != 1 or a.shape[0] not in (N_TR, N_TE):   # 2-D matrices skipped
                        continue
                    s = side_of(f) or ("oof" if a.shape[0] == N_TR else "test")
                    (oof if a.shape[0] == N_TR else test).append(
                        (norm(f), (lambda q=p: np.asarray(np.load(q), dtype=np.float64))))
                elif low.endswith(".npz"):
                    z = np.load(p)
                    for k in z.files:
                        b = z[k]
                        if b.ndim != 1 or b.shape[0] not in (N_TR, N_TE): continue
                        if side_of(k) is None: continue
                        (oof if b.shape[0] == N_TR else test).append(
                            (norm(k), (lambda q=p, kk=k: np.asarray(np.load(q)[kk], dtype=np.float64))))
                elif low.endswith((".csv", ".parquet", ".csv.gz")):
                    df = pd.read_parquet(p) if low.endswith(".parquet") else pd.read_csv(p)
                    if len(df) not in (N_TR, N_TE): continue
                    n = len(df)
                    if "id" in df.columns:                       # align to canonical row order
                        if not df["id"].is_monotonic_increasing:
                            df = df.sort_values("id")
                    cols = [c for c in eligible(df.columns, n)
                            if pd.api.types.is_numeric_dtype(df[c])]
                    if not cols: continue
                    for c in cols:
                        v = df[c].to_numpy(dtype=np.float64)
                        # a single-column table is registered under BOTH the file stem and
                        # the column name, so a pair whose columns agree exactly is not
                        # split apart by disagreeing file names.
                        labs = [norm(str(c))] if len(cols) > 1 else [norm(f), norm(str(c))]
                        for lab in dict.fromkeys(labs):
                            (oof if n == N_TR else test).append((lab, (lambda vv=v: vv)))
            except Exception:
                continue
    return oof, test

def pair(oof, test):
    """Match on normalised label; fall back to the single-vector rule only if nothing matched."""
    tmap = {}
    for lab, ld in test: tmap.setdefault(lab, ld)
    used, seen, pairs = set(), set(), []
    for lab, ld in oof:
        key = id(ld.__defaults__[0])          # one vector, possibly two labels
        if lab in tmap and lab not in used and key not in seen:
            pairs.append((lab or "main", ld, tmap[lab])); used.add(lab); seen.add(key)
    if not pairs and len(oof) == 1 and len(test) == 1:
        pairs.append((oof[0][0] or "main", oof[0][1], test[0][1]))
    return pairs

# ---------------------------------------------------------------------------- our pack
tr, _ = load_raw()
y = tr[TARGET].values
folds = get_folds(y)
print(f"y {y.shape} base rate {y.mean():.6f}; folds {[len(v) for _, v in folds]}")

ourpaths = {}
for d in sorted(os.listdir(DATA)):
    p = os.path.join(DATA, d)
    if os.path.isdir(p) and (d.startswith("ext_members") or d == "oof"):
        for f in os.listdir(p):
            if f.startswith("oof_") and f.endswith(".npy"):
                ourpaths.setdefault(f[4:-4], os.path.join(p, f))
for f in os.listdir(os.path.join(ROOT, "oof")):
    if f.startswith("oof_") and f.endswith(".npy"):
        ourpaths.setdefault(f[4:-4], os.path.join(ROOT, "oof", f))

OK, ranks = [], []
for k, p in ourpaths.items():
    try:
        v = np.load(p, mmap_mode="r")
        if v.shape == (N_TR,):
            OK.append(k)
            ranks.append(pd.Series(np.asarray(v, dtype=np.float64)).rank().to_numpy(np.float32))
    except Exception:
        pass
OMr = np.array(ranks, dtype=np.float32) if ranks else np.zeros((0, N_TR), np.float32)
del ranks
OMz = (OMr - OMr.mean(1, keepdims=True))
OMz /= np.linalg.norm(OMz, axis=1, keepdims=True)
print(f"{len(OK)} of our own OOF vectors held, for the duplicate screen\n")

# ---------------------------------------------------------------------------- candidates
led = pd.read_csv(os.path.join(HERE, "w40", "ledger2.csv"))
cands = [r for r in led[led.verdict.astype(str).str.contains("CANDIDATE", na=False)].ref
         if not any(r.startswith(s) for s in SKIP)]

rows, unpair = [], []
for ref in cands:
    d = os.path.join(OUTDIR, ref.replace("/", "_"))
    if not os.path.isdir(d): continue
    oof, test = collect(d)
    ps = pair(oof, test)
    if not ps:
        if oof or test:
            unpair.append(dict(ref=ref, n_oof=len(oof), n_test=len(test),
                               oof_labels="|".join(sorted({l for l, _ in oof})),
                               test_labels="|".join(sorted({l for l, _ in test}))))
        continue
    auth, slug = ref.split("/")
    tag = re.sub(r"[^a-z0-9]+", "", auth.lower())[:7]
    # two kernels by one author can normalise to the same label ("main"); the slug
    # disambiguates them so no member silently overwrites another on import.
    sslug = re.sub(r"[^a-z0-9]+", "", re.sub(r"^s6e8-|^playgrounds6e8-", "", slug).lower())[:10]
    for lab, ol, tl in ps:
        try:
            o, t = ol(), tl()
        except Exception as e:
            print(f"  !! {ref} {lab}: {str(e)[:60]}"); continue
        if not (np.isfinite(o).all() and np.isfinite(t).all() and o.std() > 0 and t.std() > 0):
            print(f"  -- {ref} {lab}: non-finite or constant"); continue
        solo = roc_auc_score(y, o)
        per = [roc_auc_score(y[va], o[va]) for _, va in folds]
        r = pd.Series(o).rank().to_numpy(np.float32)
        r = r - r.mean(); r /= np.linalg.norm(r)
        c = OMz @ r
        k = int(np.argmax(c))
        lb = re.sub(r"[^a-z0-9]+", "", lab.lower())[:16]
        name = f"{tag}_{lb}" if lb and lb != "main" else f"{tag}_{sslug}"
        rows.append(dict(member=name, ref=ref, label=lab, solo=solo,
                         fold_sd=float(np.std(per)), fold_min=min(per), fold_max=max(per),
                         maxcorr=float(c[k]), nearest=OK[k]))
        print(f"  {name:26s} solo {solo:.6f}  foldsd {np.std(per):.2e}  "
              f"maxcorr {c[k]:.5f} vs {OK[k]:<24s} [{ref.split('/')[1][:34]}]")

df = pd.DataFrame(rows).sort_values("solo", ascending=False)
df.to_csv(os.path.join(HERE, "w42c_vet.csv"), index=False)
pd.DataFrame(unpair).to_csv(os.path.join(HERE, "w42c_unpairable.csv"), index=False)

print(f"\n{len(df)} paired streams from {df.ref.nunique()} refs; "
      f"{len(unpair)} refs UNPAIRABLE")
sel = df[(df.maxcorr < MAXCORR) & (df.solo > SOLO)]
print(f"\nTHE w42b RULE (maxcorr < {MAXCORR}, solo > {SOLO}) SELECTS {len(sel)} of {len(df)}:")
print(sel[["member", "ref", "solo", "maxcorr", "nearest"]].to_string(index=False))
assert df.member.is_unique, f"member names collide: {df.member[df.member.duplicated()].tolist()}"

if len(sel) > 1:                       # information only -- the rule above is unchanged
    print("\nwithin-selection rank correlation (a near-duplicate PAIR is kept, per w40d's "
          "naji03/naji05 precedent, but it is recorded):")
    V = []
    for _, r in sel.iterrows():
        d0 = os.path.join(OUTDIR, r.ref.replace("/", "_"))
        o, t = collect(d0)
        for lab, ol, _tl in pair(o, t):
            if lab == r.label or (not r.label and lab == "main"):
                V.append(pd.Series(ol()).rank().to_numpy(np.float32)); break
    if len(V) == len(sel):
        M = np.array(V); M = M - M.mean(1, keepdims=True)
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        print(pd.DataFrame(M @ M.T, index=sel.member.values,
                           columns=sel.member.values).round(4).to_string())

print(f"\ntop 20 by solo regardless of the rule:")
print(df.head(20)[["member", "solo", "maxcorr", "nearest"]].to_string(index=False))
