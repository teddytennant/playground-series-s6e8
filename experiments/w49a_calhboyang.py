"""w49a -- build `w48_cal_hboyang_mix.csv`, the ONE clean test of ARM 217.

Assigned by w48 §8 item 5. w48d pre-registered the test but the artefact was never built.

WHAT THIS TESTS. ARM 217 beat its own pre-registration by z +12.5 (+41.7e-6 over six added
members against a fitted history of +0.90e-6/member). w48d found why: one of the six, the
imported public vector `hboyang_mix`, has a STANDALONE out-of-fold AUC of 0.9701816 on our
frozen folds -- +886e-6 clear of the best of the other 176 members on disk, and ABOVE our
entire 217-member cross-fitted stack (0.9701749). A single public vector that alone out-AUCs
a 217-member stack is not a base model; the natural explanation is that its OOF is not
honestly out-of-fold with respect to our split.

That is circumstantial, and w48 said so: the OOF-vs-test agreement diagnostic did NOT convict
(hboyang's gap +0.0094 is ordinary; `ravi200_publicm12` at +0.0488 is the outlier on that
axis). So the case rests on outlier magnitude alone -- which is exactly why it needs a direct
reading rather than another diagnostic.

THE READING. Submit hboyang_mix's RAW test vector, as a w37-style calibration send, and let
Kaggle score exactly these predictions. If the OOF is honest, the LB should sit near the OOF
plus the usual gap. If the OOF is inflated by leakage against our split, the LB will fall far
short of it, because leakage inflates the OOF and does nothing for the test set.

⚠ THE PREDICTION IS *LOCAL*, NOT FROM THE w36f LINE. w37e's R2 killed the linear form: the
CLEAN anchor `mkt_realmlp` missed by +1.56e-3 = 6.3σ, so the offset is NOT linear in AUC over
a wide range. w48d therefore priced this off the 34 NEAREST of 88 anchors instead of a global
fit, which is the right response to a known non-linearity. That choice is carried here
unchanged, not re-derived.

Registered in `w48d_arm217.json` BEFORE this file existed:
    predicted LB 0.97123, band [0.97120, 0.97125]
    lb >= 0.97116 -> HONEST   -- ARM 217's gain is real, w44's import line closed too early,
                                 and w45 §3 ("no seventh arm") needs revisiting ON THE RECORD.
    lb <= 0.97080 -> INFLATED -- ARM 217's CV is an imported artefact; drop `hboyang_mix`.
    in between    -> report it and change nothing.

⚠ NOT A DEADLINE CANDIDATE UNDER ANY OUTCOME. This is a raw member vector submitted to
measure a bias. `check_selection.WANTED` is untouched by it. Final selection is on CV.
"""
from __future__ import annotations

import hashlib, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, DATA, TARGET, ID                        # noqa: E402

EXT = os.path.join(DATA, "ext_members16")
MEMBER = "hboyang_mix"
STEM = "w48_cal_hboyang_mix"
N_TEST = 296302
REG = json.load(open(os.path.join(HERE, "w48d_arm217.json")))

print("=" * 90)
print(f"w49a  build {STEM}.csv -- the registered ARM 217 test (w48d)")
print("=" * 90)

# ---------------------------------------------------------------- verify the OOF side first
y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
oof = np.load(os.path.join(EXT, f"oof_{MEMBER}.npy")).ravel()
assert len(oof) == len(y), f"oof {len(oof)} != train {len(y)}"
auc = float(roc_auc_score(y, oof))
print(f"\n  OOF vector      : {len(oof):,} rows, AUC {auc:.10f}")
print(f"  w48d registered : {REG['outlier']['auc']:.10f}")
assert abs(auc - REG["outlier"]["auc"]) < 5e-10, "⛔ OOF AUC does not reproduce w48d's reading"
print(f"  reproduces to <5e-10. OK -- this is the same vector w48d found.")

# ---------------------------------------------------------------- build the submission
test_ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=[ID])[ID].values
p = np.load(os.path.join(EXT, f"test_{MEMBER}.npy")).ravel()
assert len(p) == N_TEST == len(test_ids), f"test {len(p)} vs {N_TEST} vs {len(test_ids)}"
assert np.isfinite(p).all(), "⛔ non-finite values in the test vector"

sub = pd.DataFrame({ID: test_ids, TARGET: p})
dst = os.path.join(SUB, f"{STEM}.csv")
sub.to_csv(dst, index=False)
md5 = hashlib.md5(open(dst, "rb").read()).hexdigest()

print(f"\n  test vector     : {len(p):,} rows, "
      f"min {p.min():.6f} max {p.max():.6f} mean {p.mean():.6f}, "
      f"{len(np.unique(p)):,} distinct")
print(f"  wrote           : submissions/{STEM}.csv  md5 {md5}")

# the OOF vector is ALSO saved under the stem so w48e's end-to-end verifier can reproduce a
# "CV" for it. For a raw member that CV *is* the standalone OOF AUC -- which is the whole
# quantity under test, so it must travel with the file rather than be re-derived at send time.
np.save(os.path.join(SUB, f"oof_{STEM}.npy"), oof)
print(f"  wrote           : submissions/oof_{STEM}.npy (standalone OOF, AUC {auc:.10f})")

out = dict(stem=STEM, member=MEMBER, oof_auc=auc, md5=md5, rows=int(len(p)),
           n_distinct=int(len(np.unique(p))),
           pred_lb=REG["hboyang_pred"], band=REG["hboyang_band"], rule=REG["rule"])
json.dump(out, open(os.path.join(HERE, "w49a_calhboyang.json"), "w"), indent=1)

print(f"""
  THE REGISTERED READING (w48d, unchanged -- restated so the send message can quote it)
    predicted LB  {REG['hboyang_pred']:.5f}   band [{REG['hboyang_band'][0]:.5f}, {REG['hboyang_band'][1]:.5f}]
    >= {REG['rule']['honest_at_or_above']:.5f}  -> HONEST   (revisit w44 §6 / w45 §3 on the record)
    <= {REG['rule']['inflated_at_or_below']:.5f}  -> INFLATED (drop {MEMBER}; ARM 217 CV is an artefact)
    otherwise    -> report, change nothing
""")
