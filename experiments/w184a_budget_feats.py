"""w184: what is the components-budget feature worth on our frozen folds?

The 2nd place writeup (Xin Feng, forum topic 738856) names three features:
    fake_social = daily - work - game     (and the two rotations)
built as RAW columns, not as imputations. On this data the budget ratio
r = (social+gaming+work)/daily has max exactly 1.0000 and zero violations
over 421,427 complete rows, so each fake_* is a proven upper bound on the
component it names -- and 7-19% of each component is missing.

Measures 5-fold AUC with and without the three columns, on the frozen folds
(StratifiedKFold(5, shuffle=True, random_state=42)), same params both arms.
"""
import numpy as np, pandas as pd, lightgbm as lgb, time, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DAILY, SOC, GAM, WRK = ('daily_screen_time_hours', 'social_media_hours',
                        'gaming_hours', 'work_study_hours')
CATS = ['gender', 'stress_level', 'academic_work_impact']

def load(add_budget):
    df = pd.read_csv('data/train.csv')
    y = df.pop('addicted_label').values
    df = df.drop(columns=['id'])
    if add_budget:
        df['fake_social'] = df[DAILY] - df[WRK] - df[GAM]
        df['fake_game']   = df[DAILY] - df[SOC] - df[WRK]
        df['fake_work']   = df[DAILY] - df[SOC] - df[GAM]
        df['budget_r']    = (df[SOC] + df[GAM] + df[WRK]) / df[DAILY]
    for c in CATS:
        df[c] = df[c].astype('category')
    return df, y

PARAMS = dict(objective='binary', metric='auc', learning_rate=0.06,
              num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1, seed=42, force_row_wise=True)
ROUNDS = 700

def run(add_budget, tag):
    X, y = load(add_budget)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    t0 = time.time()
    for k, (tri, vai) in enumerate(skf.split(X, y)):
        ds = lgb.Dataset(X.iloc[tri], y[tri])
        dv = lgb.Dataset(X.iloc[vai], y[vai], reference=ds)
        m = lgb.train(PARAMS, ds, ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        print(f'  {tag} fold{k} auc={roc_auc_score(y[vai], oof[vai]):.6f} '
              f'iter={m.best_iteration}', flush=True)
    auc = roc_auc_score(y, oof)
    print(f'{tag:10s} OOF AUC = {auc:.6f}   ({time.time()-t0:.0f}s)', flush=True)
    np.save(f'experiments/w184a_oof_{tag}.npy', oof)
    return auc

if __name__ == '__main__':
    base = run(False, 'base')
    bud  = run(True,  'budget')
    print(f'\ndelta = {(bud-base)*1e6:+.1f}e-6   base {base:.6f} -> budget {bud:.6f}')
