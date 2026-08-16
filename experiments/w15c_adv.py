"""w15c — decompose the train/test shift: is it only the missingness mask, or do the
observed VALUES shift as well?

The public notebook `georgymamarin/...` reports an adversarial train-vs-test AUC of 0.57
and concludes "missingness carries nothing about the target". That conflates two things.
w15c_shift.py shows the mask allocation shifts at up to |z| = 44, which alone would
produce an adversarial AUC well above 0.5. The open question is whether anything shifts
BESIDES the mask -- because a value shift would be a different and larger story.

Three arms:
  mask   : the 12 missingness indicators only
  values : COMPLETE ROWS ONLY (no NaN anywhere), so the mask is constant and cannot leak
  both   : the raw frame as the models actually see it

If `values` lands at 0.500 the shift is entirely a masking artefact.
Per-column marginal tests on complete rows are reported alongside, because an AUC of
0.51 is hard to interpret and a KS statistic is not.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

NUM = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
       'work_study_hours','sleep_hours','notifications_per_day',
       'app_opens_per_day','weekend_screen_time']
CAT = ['gender','stress_level','academic_work_impact']
FEAT = NUM + CAT
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63,
              min_child_samples=200, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, num_threads=3, verbose=-1, seed=42)

def adv(X, z, rounds=400, label=''):
    """5-fold adversarial AUC of X predicting z (0=train row, 1=test row)."""
    oof = np.zeros(len(z))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tri, vai in skf.split(X, z):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tri], z[tri]), num_boost_round=rounds)
        oof[vai] = m.predict(X.iloc[vai])
    a = roc_auc_score(z, oof)
    print(f'  {label:<34} n={len(z):>8,}  cols={X.shape[1]:>2}  adv AUC = {a:.5f}',
          flush=True)
    return a

def main():
    tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
    for d in (tr, te):
        for c in CAT: d[c] = d[c].astype('category')
    # align categories
    for c in CAT:
        cats = sorted(set(tr[c].cat.categories) | set(te[c].cat.categories))
        tr[c] = tr[c].cat.set_categories(cats); te[c] = te[c].cat.set_categories(cats)

    z = np.r_[np.zeros(len(tr)), np.ones(len(te))].astype(int)
    full = pd.concat([tr[FEAT], te[FEAT]], ignore_index=True)

    print('=== adversarial validation, train vs test ===')
    mask = pd.concat([tr[FEAT].isna(), te[FEAT].isna()], ignore_index=True).astype(np.int8)
    a_mask = adv(mask, z, label='mask only (12 NaN indicators)')
    a_both = adv(full, z, label='full frame (values + mask)')

    # complete rows only -- mask is constant, so only values can separate
    cr = ~full.isna().any(axis=1)
    Xc, zc = full[cr].reset_index(drop=True), z[cr.to_numpy()]
    print(f'  [complete rows: {int((zc==0).sum()):,} train / {int((zc==1).sum()):,} test]')
    a_val = adv(Xc, zc, label='values only (complete rows, no mask)')

    # negative control: split TRAIN complete rows in half at random and re-run.
    # anything the instrument reports above this is not the instrument.
    trc = full[cr & (z == 0)].reset_index(drop=True)
    rng = np.random.default_rng(7)
    zc0 = (rng.random(len(trc)) < 0.5).astype(int)
    a_ctrl = adv(trc, zc0, label='CONTROL: train-vs-train random half')

    print(f'\n  mask only  {a_mask:.5f}\n  values     {a_val:.5f}\n'
          f'  control    {a_ctrl:.5f}   <- instrument floor\n  full       {a_both:.5f}')
    print(f'  values - control = {a_val - a_ctrl:+.5f}')

    print('\n=== per-column marginals on COMPLETE ROWS only (train vs test) ===')
    A = full[cr & (z == 0)]; B = full[cr & (z == 1)]
    print(f'{"column":<26}{"stat":>10}{"p":>12}{"tr mean":>11}{"te mean":>11}')
    for c in NUM:
        s, p = stats.ks_2samp(A[c].to_numpy(), B[c].to_numpy())
        print(f'{c:<26}{s:>10.5f}{p:>12.3g}{A[c].mean():>11.4f}{B[c].mean():>11.4f}')
    for c in CAT:
        ct = pd.crosstab(np.r_[np.zeros(len(A)), np.ones(len(B))],
                         pd.concat([A[c], B[c]], ignore_index=True))
        chi2, p, _, _ = stats.chi2_contingency(ct)
        print(f'{c:<26}{chi2:>10.3f}{p:>12.3g}')

if __name__ == '__main__':
    main()
