"""w185: test-side pair for the original-as-COLUMN route, to price it on PRIVATE.

The board closed 2026-08-31, but Kaggle still ACCEPTS and SCORES late submissions
(100/day) and returns publicScore AND privateScore. Verified this run: resending
`w36_ad199stdcorr_ens4.csv` came back 0.97119 / 0.97093, matching the record exactly.
So a paired pair of files measures a manoeuvre on the real test set, not on CV.

This fits the same two arms as w185a -- `base` and `all` -- on the frozen folds and
averages the 5 fold models over test. Identical params, identical folds, identical
seed; the ONLY difference between the two files is the 24 original-derived columns.
The pair is a MEASUREMENT, not a leaderboard candidate: a single LightGBM lands far
below this workspace's 0.97119 best and must never enter a max(CV) selection.

⚠ The private score is ground truth for a CLOSED competition. It is usable here to
learn what a manoeuvre is worth. It would be selection-on-private, i.e. the Rogii
failure, if this board were live. Do not carry the habit forward.
"""
import numpy as np, pandas as pd, lightgbm as lgb, time, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'experiments')
from w185a_origcols import (PARAMS, ROUNDS, CATS, NUM, orig_frame,
                            add_cdf, add_prior, add_knn)


def build(path, arm, drop):
    df = pd.read_csv(path)
    keep = df['id'].values
    df = df.drop(columns=drop)
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
    return df, keep


def run(arm):
    X, _ = build('data/train.csv', arm, ['id', 'addicted_label'])
    y = pd.read_csv('data/train.csv', usecols=['addicted_label'])['addicted_label'].values
    Xt, tid = build('data/test.csv', arm, ['id'])
    Xt = Xt[X.columns]
    for c in CATS:
        Xt[c] = Xt[c].cat.set_categories(X[c].cat.categories)

    oof = np.zeros(len(y)); pt = np.zeros(len(Xt))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    t0 = time.time()
    for k, (tri, vai) in enumerate(skf.split(X, y)):
        ds = lgb.Dataset(X.iloc[tri], y[tri])
        dv = lgb.Dataset(X.iloc[vai], y[vai], reference=ds)
        m = lgb.train(PARAMS, ds, ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        pt += m.predict(Xt, num_iteration=m.best_iteration) / 5
        print(f'  {arm} fold{k} auc={roc_auc_score(y[vai], oof[vai]):.6f}', flush=True)
    auc = roc_auc_score(y, oof)
    out = f'submissions/w185_origcol_{arm}.csv'
    pd.DataFrame({'id': tid, 'addicted_label': pt}).to_csv(out, index=False)
    print(f'{arm:5s} ncol={X.shape[1]:3d} OOF={auc:.6f} -> {out} ({time.time()-t0:.0f}s)',
          flush=True)
    return auc


if __name__ == '__main__':
    arms = sys.argv[1:] or ['base', 'cdf']
    r = {a: run(a) for a in arms}
    if len(r) == 2 and 'base' in r:
        b = r['base']; a = [v for k, v in r.items() if k != 'base'][0]
        print(f'\nCV delta = {(a-b)*1e6:+.1f}e-6   {b:.6f} -> {a:.6f}')
