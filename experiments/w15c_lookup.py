"""w15c — does an in-fold row-identity lookup add anything on top of the stack?

Every lookup here is built INSIDE the fold: for held-out fold k the key->label map is
computed from the other four folds only, never from fold k. That is the whole point of
the test; a lookup built across folds measures a fantasy.

Instrument: conditional AUC of the lookup feature GIVEN the model score. We bin the
stack's OOF probability into fine quantile bins and, within each bin, compute the
Mann-Whitney statistic of the lookup value against the label. Pooled over bins this is
the AUC of the lookup feature among rows the model already scores identically -- so it
is exactly the signal the lookup adds that the model does not already have. 0.5 = dead.

Control: the same lookup values randomly permuted WITHIN each bin, which fixes the
marginal distribution and the bin structure and destroys only the row correspondence.
"""
import numpy as np, pandas as pd, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED, NF = 42, 5
FEAT = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
        'work_study_hours','sleep_hours','notifications_per_day',
        'app_opens_per_day','weekend_screen_time','gender','stress_level',
        'academic_work_impact']
BLEND = 'submissions/oof_blend159av_h3.npy'

# key subsets to try, coarse -> fine
SUBSETS = {
 'rule2         (daily,social)': ['daily_screen_time_hours','social_media_hours'],
 'rule2+cats                  ': ['daily_screen_time_hours','social_media_hours',
                                  'gender','stress_level','academic_work_impact'],
 'screen4                     ': ['daily_screen_time_hours','social_media_hours',
                                  'gaming_hours','work_study_hours'],
 'screen4+weekend             ': ['daily_screen_time_hours','social_media_hours',
                                  'gaming_hours','work_study_hours','weekend_screen_time'],
 'behav5 (notif,opens,sleep,+)': ['notifications_per_day','app_opens_per_day',
                                  'sleep_hours','daily_screen_time_hours','age'],
 'all-but-4                   ': ['daily_screen_time_hours','social_media_hours',
                                  'gaming_hours','work_study_hours','sleep_hours',
                                  'notifications_per_day','app_opens_per_day','age'],
 'all12                       ': FEAT,
 'namask12 (missingness only) ': None,   # special-cased: the NaN pattern as the key
}

def keyframe(df, cols):
    parts = []
    for c in cols:
        s = df[c]
        if s.dtype.kind == 'O':
            parts.append(s.fillna('@').astype(str))
        else:
            parts.append(s.map(lambda v: '@' if pd.isna(v) else format(v, '.4f')))
    return parts[0].str.cat(parts[1:], sep='|')

def cond_auc(feat, y, p, nbins=200, rng=None):
    """Pooled within-bin Mann-Whitney AUC of `feat` vs `y`, conditioning on `p`.
    Rows with no lookup value (feat is NaN) are dropped. Returns (auc, n_used)."""
    ok = ~np.isnan(feat)
    f, yy, pp = feat[ok], y[ok], p[ok]
    if len(f) < 1000: return np.nan, len(f)
    bins = pd.qcut(pp, nbins, labels=False, duplicates='drop')
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        yb, fb = yy[m], f[m]
        npos, nneg = yb.sum(), (1 - yb).sum()
        if npos == 0 or nneg == 0: continue
        if rng is not None: fb = rng.permutation(fb)
        r = pd.Series(fb).rank().to_numpy()
        u = r[yb == 1].sum() - npos * (npos + 1) / 2.0
        num += u; den += npos * nneg
    return (num / den if den else np.nan), int(ok.sum())

def main():
    tr = pd.read_csv('data/train.csv')
    y = tr['addicted_label'].to_numpy()
    p = np.load(BLEND)
    print(f'stack OOF AUC {roc_auc_score(y, p):.6f}  n={len(y):,}', flush=True)

    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    print(f'\n{"key subset":<30}{"cov%":>8}{"medN":>6}{"condAUC":>10}{"ctrl":>10}'
          f'{"real-ctrl":>11}')
    print('-' * 75)
    for name, cols in SUBSETS.items():
        if cols is None:
            k = tr[FEAT].isna().astype(int).astype(str).agg(''.join, axis=1)
        else:
            k = keyframe(tr, cols)
        kv = k.to_numpy()

        rate = np.full(len(y), np.nan)
        cnt = np.zeros(len(y))
        for tri, vai in folds:                       # <-- lookup built INSIDE the fold
            g = pd.DataFrame({'k': kv[tri], 'y': y[tri]}).groupby('k')['y'].agg(['sum','size'])
            s = pd.Series(kv[vai])
            m = s.map(g['size']).to_numpy()
            sm = s.map(g['sum']).to_numpy()
            with np.errstate(invalid='ignore'):
                rate[vai] = sm / m
            cnt[vai] = np.nan_to_num(m)

        cov = np.isfinite(rate)
        med = int(np.median(cnt[cov])) if cov.any() else 0
        a, n = cond_auc(rate, y, p)
        rng = np.random.default_rng(0)
        c, _ = cond_auc(rate, y, p, rng=rng)
        d = a - c if np.isfinite(a) and np.isfinite(c) else np.nan
        print(f'{name:<30}{cov.mean():>7.2%}{med:>6}{a:>10.5f}{c:>10.5f}{d:>11.5f}')

    # ---- id structure, conditional on features ----
    print('\n=== id structure, conditional on the model score ===')
    idv = tr['id'].to_numpy().astype(float)
    a, n = cond_auc(idv, y, p)
    rng = np.random.default_rng(1)
    c, _ = cond_auc(idv, y, p, rng=rng)
    print(f'conditional AUC of raw id vs label, given stack score: {a:.6f} '
          f'(ctrl {c:.6f}, real-ctrl {a-c:+.6f})')
    resid = y - p
    print(f'corr(id, y - p_stack) = {np.corrcoef(idv, resid)[0,1]:+.6f}')
    # id modulo small numbers -- interleaved generation would show here
    for m in (2, 3, 5, 7, 10, 16, 100):
        g = pd.DataFrame({'m': (tr['id'] % m).to_numpy(), 'r': resid}).groupby('m')['r']
        mu, se = g.mean(), g.std() / np.sqrt(g.size())
        z = (mu / se).abs().max()
        print(f'  id % {m:>3}: max |z| of residual mean across classes = {z:.2f}')

if __name__ == '__main__':
    main()
