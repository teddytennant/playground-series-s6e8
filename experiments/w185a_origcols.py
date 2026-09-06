"""w185: the original dataset as COLUMNS -- the route ANGLE INDEX row 1 never priced.

Row 1 closed "the original dataset" on two routes:
    as ROW    concat extra training rows      -58.0e-6 at 1x, -3,340e-6 at 50x (w131a)
    as MEMBER a first-stage model fit on the -1.02e-6/member, t=0.97 (w131a arm C)
              7,500 originals, enrolled

The 14th-place writeup (topic 739004) names a THIRD route we never ran, and it is the
one they actually adopted:

    "Five configs used the public Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
     as a proxy for the generator's source: nearest-neighbour features, prior means,
     class-conditional CDF differences, and a first-stage prediction trained on it."

The first three are COLUMNS derived from the original, not rows and not a member.
2nd place (topic 738856) lists "Original Data as COL" as a standing config axis too.

Every transform here is fit on the 7,500 original rows ONLY. The original carries its
own labels and is disjoint from the competition frame, so the map is constant across
folds -- there is no in-fold refit and no leakage channel. That is also why it is cheap.

Arms, all on the frozen folds (StratifiedKFold(5, shuffle=True, random_state=42)) and
the exact LightGBM params of w184a, so `base` is comparable to w184a's base run:

    cdf    class-conditional CDF differences, 10 numeric columns
    prior  original mean label by decile bin (10 cols) + by category (3 cols)
    knn    k=25 neighbour label mean and mean distance over scaled numerics
    all    the three groups together
"""
import numpy as np, pandas as pd, lightgbm as lgb, time, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ORIG = 'data/orig/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
NUM = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
       'work_study_hours', 'sleep_hours', 'notifications_per_day',
       'app_opens_per_day', 'weekend_screen_time']
CATS = ['gender', 'stress_level', 'academic_work_impact']

PARAMS = dict(objective='binary', metric='auc', learning_rate=0.06,
              num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1, seed=42, force_row_wise=True)
ROUNDS = 700


def orig_frame():
    o = pd.read_csv(ORIG)
    return o, o['addicted_label'].values


def add_cdf(df, o, oy):
    """F1(x) - F0(x) from the original's class-conditional empirical CDFs."""
    for c in NUM:
        a = np.sort(o.loc[oy == 1, c].dropna().values)
        b = np.sort(o.loc[oy == 0, c].dropna().values)
        x = df[c].values
        f1 = np.searchsorted(a, x, 'right') / max(len(a), 1)
        f0 = np.searchsorted(b, x, 'right') / max(len(b), 1)
        d = f1 - f0
        df[f'cdfd_{c}'] = np.where(np.isnan(x), np.nan, d)
    return df


def add_prior(df, o, oy):
    """Original mean label, by decile of each numeric column and by category."""
    for c in NUM:
        edges = np.unique(np.nanquantile(o[c].values, np.linspace(0, 1, 11)))
        ob = np.clip(np.searchsorted(edges, o[c].values, 'right') - 1, 0, len(edges) - 2)
        m = pd.Series(oy).groupby(ob).mean()
        x = df[c].values
        xb = np.clip(np.searchsorted(edges, x, 'right') - 1, 0, len(edges) - 2)
        v = m.reindex(xb).values
        df[f'oprior_{c}'] = np.where(np.isnan(x), np.nan, v)
    for c in CATS:
        m = pd.Series(oy).groupby(o[c].values).mean()
        df[f'oprior_{c}'] = df[c].map(m).astype(float)
    return df


def add_knn(df, o, oy, k=25):
    """Label mean and mean distance over the k nearest original rows."""
    mu = o[NUM].mean(); sd = o[NUM].std().replace(0, 1.0)
    ref = ((o[NUM] - mu) / sd).fillna(0.0).values
    qry = ((df[NUM] - mu) / sd).fillna(0.0).values
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(ref)
    dist, idx = nn.kneighbors(qry)
    df['oknn_mean'] = oy[idx].mean(axis=1)
    df['oknn_dist'] = dist.mean(axis=1)
    df['oknn_obs'] = df[NUM].notna().sum(axis=1).values
    return df


def load(arm):
    df = pd.read_csv('data/train.csv')
    y = df.pop('addicted_label').values
    df = df.drop(columns=['id'])
    if arm != 'base':
        o, oy = orig_frame()
        if arm in ('cdf', 'all'):
            df = add_cdf(df, o, oy)
        if arm in ('prior', 'all'):
            df = add_prior(df, o, oy)
        if arm in ('knn', 'all'):
            df = add_knn(df, o, oy)
    for c in CATS:
        df[c] = df[c].astype('category')
    return df, y


def run(arm):
    X, y = load(arm)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    t0 = time.time()
    for k, (tri, vai) in enumerate(skf.split(X, y)):
        ds = lgb.Dataset(X.iloc[tri], y[tri])
        dv = lgb.Dataset(X.iloc[vai], y[vai], reference=ds)
        m = lgb.train(PARAMS, ds, ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        print(f'  {arm} fold{k} auc={roc_auc_score(y[vai], oof[vai]):.6f} '
              f'iter={m.best_iteration}', flush=True)
    auc = roc_auc_score(y, oof)
    print(f'{arm:6s} ncol={X.shape[1]:3d}  OOF AUC = {auc:.6f}   '
          f'({time.time()-t0:.0f}s)', flush=True)
    np.save(f'experiments/w185a_oof_{arm}.npy', oof)
    return auc


if __name__ == '__main__':
    arms = sys.argv[1:] or ['base', 'cdf', 'prior', 'knn', 'all']
    res = {a: run(a) for a in arms}
    if 'base' in res:
        print()
        for a, v in res.items():
            if a != 'base':
                print(f'{a:6s} delta = {(v-res["base"])*1e6:+8.1f}e-6   '
                      f'{res["base"]:.6f} -> {v:.6f}')
