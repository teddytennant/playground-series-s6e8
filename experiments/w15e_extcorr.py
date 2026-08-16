"""w15e: the test-space correlation map of PUBLIC submissions against our pack.

The mission's premise: our 160 members are all grown from the same soil, maxcorr ~0.97
on OOF, and a member built by someone else is the cheapest genuine decorrelation
available. A public submission CSV has no OOF, so it cannot be weighted on CV -- but its
TEST vector can still be compared, and a correlation meaningfully below what our own
members show each other means it carries a direction we do not have.

What this script measures, all on the 296,302 test rows and all in SPEARMAN
(rank) correlation, because AUC is a pure ranking functional and Pearson on
probabilities would mostly report the tail-saturation shape rather than the ordering:

  1. the pack's INTERNAL test-space baseline -- every pair among a sample of our own
     members, and every pair among the four transform variants of our best blend. This
     is the reference number the mission quotes as "~0.9999"; it is measured here
     rather than assumed.
  2. every public submission vs our best blend, vs each other, and vs the pack members.
  3. a leave-one-in check: is a public file closer to our blend than our own worst
     member is?

Nothing is submitted or written into submissions/ by this script.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, OOF, SUB  # noqa: E402

W15E = os.path.join(DATA, "w15e")
N_TEST = 296_302

# Our reference points. blend159av_h3 / blend160origm_h3 are the deadline picks
# (CV 0.970049); blend158_logit is the current public-LB leader among our entries.
OURS_BLENDS = [
    "blend159av_h3", "blend160origm_h3", "blend156_h3", "blend159av_logit",
    "blend158_logit", "blend159av_rankraw", "blend159av_hybrid", "blend159av_rescale",
]

# Public submissions: (label, path, published LB score or None)
PUBLIC = [
    ("naji_eoe_97101", "kout/najiama_ensemble-of-ensembles-lb-0-97101/submission.csv", 0.97101),
    ("krasnov_top1_97099", "kout/daniilkrasnovvv_s6e8-top-1-public-0-97099/submission.csv", 0.97099),
    ("naji_97097", "kout/najiama_s6e8-addiction-lb-0-97097/submission.csv", 0.97097),
    ("rotor_rankblend", "kout/nikita7364777_s6e8-rank-blend-65-35/submission.csv", None),
    ("amanatar_97092", "kout/amanatar_s6e8-elite-rank-average-ensemble-0-97092/submission.csv", 0.97092),
    ("rambe_97092", "kout/rauffauzanrambe_s6e8-technology-addiction-lb-0-97092-prime/submission.csv", 0.97092),
    ("rustam_97092", "kout/rustambhadouriya_student-health-rank-average-0-97092/submissions_h.csv", 0.97092),
    ("ravi_l2stack", "kout/ravi20076_playgrounds6e8-public-l2stack-v1/submission.csv", None),
    ("rayk_mixmeta", "kout/raykkretzschmar_mix-the-meta-models-then-learn-what-they-miss/submission.csv", None),
    ("rayk_bandmodel", "kout/raykkretzschmar_mix-the-meta-models-then-learn-what-they-miss/submission_band_model.csv", None),
    ("krasnov_memes2", "kout/daniilkrasnovvv_s6e8-current-best-open-solution-memes-2/submission.csv", None),
    ("anthony_nnres", "kout/anthonytherrien_predicting-smartphone-addict-nn-residual-network/submission.csv", None),
    ("anthony_vault", "anthonytherrien_predicting-smartphone-addiction-vault/submission.csv", None),
    ("naji_psa_knn", "najiama_s6e8-psa/Naji_KNN_submission.csv", None),
    ("naji_psa_lgbm", "najiama_s6e8-psa/Naji_LGBM_submission.csv", None),
    ("naji_psa_rayk", "najiama_s6e8-psa/Rayk_submission.csv", None),
    ("mkt_xgb_v3", "mohankrishnathalla_s6e8-xgb-oof/submission_xgb_v3.csv", None),
    ("mkt_cat_v3", "mohankrishnathalla_s6e8-cat-mlp-oof/submission_cat_v3.csv", None),
    ("mkt_lgb_v3", "mohankrishnathalla_s6e8-lgb-dart-oof/submission_lgb_v3.csv", None),
]


def zrank(v):
    r = rankdata(v)
    return (r - r.mean()) / r.std()


def load_sub(path, ids_ref):
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != "id"][0]
    assert len(df) == N_TEST, (path, len(df))
    df = df.set_index("id").reindex(ids_ref)
    v = df[col].to_numpy(np.float64)
    assert np.isfinite(v).all(), path
    return v


def main():
    ref = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = ref["id"].to_numpy()

    # ---------- 1. our own reference vectors ----------
    ours, onames = [], []
    for b in OURS_BLENDS:
        p = os.path.join(SUB, f"{b}.csv")
        if os.path.exists(p):
            ours.append(zrank(load_sub(p, ids)))
            onames.append(b)

    # a sample of individual pack members, from all three source libraries + ours
    mem, mnames = [], []
    srcs = [(os.path.join(DATA, "oof", "oof"), "lib"),
            (os.path.join(DATA, "ext_members"), "ext"),
            (os.path.join(DATA, "ext_members2"), "ext2"),
            (OOF, "own")]
    for d, tag in srcs:
        if not os.path.isdir(d):
            continue
        fs = sorted(glob.glob(os.path.join(d, "test_*.npy")))
        for f in fs:
            nm = os.path.basename(f)[5:-4]
            v = np.load(f)
            if v.shape != (N_TEST,):
                continue
            mem.append(zrank(v.astype(np.float64)))
            mnames.append(f"{tag}:{nm}")
    print(f"loaded {len(ours)} of our blends, {len(mem)} pack members\n")

    M = np.column_stack(mem)
    n = N_TEST

    # ---------- pack internal baseline ----------
    G = (M.T @ M) / n
    iu = np.triu_indices(len(mnames), 1)
    pv = G[iu]
    np.fill_diagonal(G, -np.inf)
    mx = G.max(1)
    print("=== PACK INTERNAL test-space Spearman, all "
          f"{len(pv):,} member pairs ===")
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"  q{q:<5} {np.quantile(pv, q):.5f}")
    print(f"  min {pv.min():.5f}   max {pv.max():.5f}")
    print(f"\n  per-member MAXCORR vs the rest: min {mx.min():.5f} "
          f"q10 {np.quantile(mx, .10):.5f} median {np.median(mx):.5f}")
    lo = np.argsort(mx)[:8]
    print("  the 8 most decorrelated members we own:")
    for j in lo:
        print(f"    {mnames[j]:<42s} maxcorr {mx[j]:.5f}")

    B = np.column_stack(ours)
    Gb = (B.T @ B) / n
    iub = np.triu_indices(len(onames), 1)
    print(f"\n=== our BLENDS vs each other: min {Gb[iub].min():.6f} "
          f"median {np.median(Gb[iub]):.6f} max {Gb[iub].max():.6f} ===")
    print(pd.DataFrame(Gb, index=onames, columns=onames).to_string(float_format="%.5f"))

    # ---------- 2. public submissions ----------
    pub, pnames, plb = [], [], []
    for lab, rel, lb in PUBLIC:
        p = os.path.join(W15E, rel)
        if not os.path.exists(p):
            print(f"[missing] {lab} {rel}")
            continue
        pub.append(zrank(load_sub(p, ids)))
        pnames.append(lab)
        plb.append(lb)
    P = np.column_stack(pub)

    best = ours[onames.index("blend159av_h3")] if "blend159av_h3" in onames else ours[0]
    rows = []
    for j, lab in enumerate(pnames):
        c_blend = float(P[:, j] @ best / n)
        cm = (M.T @ P[:, j]) / n
        cb = (B.T @ P[:, j]) / n
        rows.append(dict(public=lab, lb=plb[j], vs_best_blend=c_blend,
                         vs_any_blend_max=cb.max(),
                         maxcorr_member=cm.max(),
                         nearest=mnames[int(np.argmax(cm))],
                         medcorr_member=float(np.median(cm))))
    df = pd.DataFrame(rows).sort_values("vs_best_blend")
    print("\n=== PUBLIC SUBMISSIONS vs OUR PACK (test-space Spearman) ===")
    print(df.to_string(index=False, float_format="%.5f"))

    # public vs public
    Gp = (P.T @ P) / n
    print("\n=== public vs public ===")
    print(pd.DataFrame(Gp, index=pnames, columns=pnames).to_string(float_format="%.4f"))

    out = os.path.join(ROOT, "experiments", "w15e_extcorr.csv")
    df.to_csv(out, index=False)
    np.savez(os.path.join(ROOT, "experiments", "w15e_extcorr.npz"),
             pub_names=np.array(pnames), Gp=Gp,
             pack_pair_q=np.array([np.quantile(pv, q) for q in
                                   (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]),
             pack_maxcorr=mx, mem_names=np.array(mnames))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
