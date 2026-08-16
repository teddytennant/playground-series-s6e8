"""w15c — the train/test missingness shift, and whether it explains the CV->LB gap.

Observed (experiments/w15c_twins.py and below): per-column NaN rates differ between train
and test at up to |z| = 44, while the number-missing-per-row distribution is nearly
identical (mean 1.2589 train vs 1.2729 test). So the missingness BUDGET matches across the
split and its ALLOCATION across columns does not.

That matters because the two drivers the generator's rule uses are observed MORE often in
test (daily 11.07% NaN vs 13.86%; social 16.00% vs 19.38%). Rows with an observed driver
are far more rankable (w14d: cell A 0.984 vs both-drivers-missing 0.912). If test is a
systematically easier mix than train, unweighted OOF AUC UNDERSTATES test AUC -- which is
the sign of the workspace's long-standing ~+1.0e-3 CV->LB gap.

Test: reweight the OOF pool to the test missingness distribution by the exact mask-pattern
density ratio, and see how much of the observed CV->LB distance that closes.
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import roc_auc_score

FEAT = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
        'work_study_hours','sleep_hours','notifications_per_day',
        'app_opens_per_day','weekend_screen_time','gender','stress_level',
        'academic_work_impact']

def maskcode(df):
    """Missingness pattern as an integer 0..4095 -- 12 columns, bit i = column i is NaN."""
    m = df[FEAT].isna().to_numpy()
    return m @ (1 << np.arange(len(FEAT)))

def wauc(y, s, w):
    """Weighted ROC AUC = P(s_pos > s_neg) + 0.5 P(=) under the weight measure.
    Vectorised: sort by score, accumulate negative weight strictly below each tie group."""
    o = np.argsort(s, kind='mergesort')
    s, wpos, wneg = s[o], (w * y)[o], (w * (1 - y))[o]
    cneg = np.concatenate([[0.0], np.cumsum(wneg)])
    # tie groups: start index of each run of equal scores
    newgrp = np.concatenate([[True], s[1:] != s[:-1]])
    gid = np.cumsum(newgrp) - 1
    ng = gid[-1] + 1
    start = np.flatnonzero(newgrp)
    stop = np.concatenate([start[1:], [len(s)]])
    gpos = np.bincount(gid, weights=wpos, minlength=ng)
    below = cneg[start]
    tie_neg = cneg[stop] - cneg[start]
    return float((gpos * (below + 0.5 * tie_neg)).sum() / (wpos.sum() * wneg.sum()))

def main():
    tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
    y = tr['addicted_label'].to_numpy()
    ktr, kte = maskcode(tr), maskcode(te)

    ctr = np.bincount(ktr, minlength=4096).astype(float)
    cte = np.bincount(kte, minlength=4096).astype(float)
    ptr, pte = ctr / ctr.sum(), cte / cte.sum()
    print(f'distinct missingness patterns: train {int((ctr>0).sum()):,} '
          f'test {int((cte>0).sum()):,} of 4096')
    print(f'train rows whose pattern never occurs in test: '
          f'{(cte[ktr]==0).mean():.4%}')
    print(f'test  rows whose pattern never occurs in train: '
          f'{(ctr[kte]==0).mean():.4%}')

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(ptr > 0, pte / ptr, 0.0)
    w = ratio[ktr]
    hi = np.quantile(w[w > 0], 0.999)
    w = np.clip(w, 0, hi); w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f'\nweights: sd {w.std():.4f} min {w.min():.4f} max {w.max():.4f}  '
          f'ESS {ess:,.0f} of {len(w):,} ({ess/len(w):.2%})')

    order = np.argsort(-ctr)[:12]
    print('\ntop-12 missingness patterns by train frequency '
          '(pattern printed as which columns are NaN):')
    print(f'{"cols missing":<34}{"train":>9}{"test":>9}{"test/train":>11}')
    for k in order:
        nm = ','.join(FEAT[i][:9] for i in range(12) if k >> i & 1) or '(none)'
        r = pte[k] / ptr[k] if ptr[k] else np.nan
        print(f'{nm[:33]:<34}{ptr[k]:>9.5f}{pte[k]:>9.5f}{r:>11.4f}')

    # per-column NaN rate, train vs test, with z
    print(f'\n{"column":<26}{"tr_na":>8}{"te_na":>8}{"diff":>9}{"z":>8}')
    for i, c in enumerate(FEAT):
        a = tr[c].isna().mean(); b = te[c].isna().mean()
        p = (tr[c].isna().sum() + te[c].isna().sum()) / (len(tr) + len(te))
        se = np.sqrt(p * (1 - p) * (1 / len(tr) + 1 / len(te)))
        print(f'{c:<26}{a:>8.4f}{b:>8.4f}{b-a:>9.4f}{(b-a)/se:>8.1f}')

    lb = json.load(open('experiments/lb_scores.json'))
    rows = []
    for f, s in sorted(lb.items()):
        pth = f'submissions/oof_{f[:-4]}.npy'
        if not os.path.exists(pth): continue
        p = np.load(pth)
        if len(p) != len(y): continue
        rows.append((f[:-4], roc_auc_score(y, p), wauc(y, p, w), s))
    df = pd.DataFrame(rows, columns=['file', 'cv', 'cv_w', 'lb'])
    df['gap'] = df['lb'] - df['cv']; df['gap_w'] = df['lb'] - df['cv_w']
    df['shift'] = df['cv_w'] - df['cv']
    print(f'\n=== {len(df)} files with both an OOF vector and an LB score ===')
    print(df.sort_values('lb', ascending=False).to_string(
        index=False, float_format=lambda v: f'{v:.6f}'))
    print(f'\nmean CV->LB gap unweighted : {df["gap"].mean():+.6f} (sd {df["gap"].std():.6f})')
    print(f'mean CV->LB gap reweighted : {df["gap_w"].mean():+.6f} (sd {df["gap_w"].std():.6f})')
    print(f'mean reweighting shift     : {df["shift"].mean():+.6f} (sd {df["shift"].std():.6f})')
    print(f'\nspearman CV  vs LB: {df["cv"].corr(df["lb"], method="spearman"):+.4f}')
    print(f'spearman CVw vs LB: {df["cv_w"].corr(df["lb"], method="spearman"):+.4f}')
    df.to_csv('experiments/w15c_shift_results.csv', index=False)

if __name__ == '__main__':
    main()
