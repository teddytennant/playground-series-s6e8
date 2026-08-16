"""w15c — error bars on the lookup null, and a corrected id-structure test.

Fixes two things in w15c_lookup.py:
  * the control was a single permutation, so `real - ctrl` at 0.6% coverage was quoted
    against an unknown noise scale. Here the control is run NCTRL times and the null sd
    is reported, so `real - ctrl` can be read in sigma.
  * the `id % m` test measured each class's residual mean against ZERO. The h3 stack
    score is a rank-average, not a calibrated probability, so `y - p` has a large
    nonzero mean for every class and the |z| of ~400 was that offset, not structure.
    Corrected: one-way ANOVA of the residual ACROSS classes, plus a matched control that
    permutes the class labels.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED, NF, NCTRL = 42, 5, 24
FEAT = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
        'work_study_hours','sleep_hours','notifications_per_day',
        'app_opens_per_day','weekend_screen_time','gender','stress_level',
        'academic_work_impact']
BLEND = 'submissions/oof_blend159av_h3.npy'

SUBSETS = {
 'rule2 (daily,social)': ['daily_screen_time_hours','social_media_hours'],
 'rule2+cats':           ['daily_screen_time_hours','social_media_hours',
                          'gender','stress_level','academic_work_impact'],
 'screen4':              ['daily_screen_time_hours','social_media_hours',
                          'gaming_hours','work_study_hours'],
 'screen4+weekend':      ['daily_screen_time_hours','social_media_hours','gaming_hours',
                          'work_study_hours','weekend_screen_time'],
 'behav5':               ['notifications_per_day','app_opens_per_day','sleep_hours',
                          'daily_screen_time_hours','age'],
 'namask12':             None,
}

def keyframe(df, cols):
    parts = []
    for c in cols:
        s = df[c]
        parts.append(s.fillna('@').astype(str) if s.dtype.kind == 'O'
                     else s.map(lambda v: '@' if pd.isna(v) else format(v, '.4f')))
    return parts[0].str.cat(parts[1:], sep='|')

def cond_auc_core(f, yy, bins, rng=None):
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        yb, fb = yy[m], f[m]
        npos = yb.sum(); nneg = len(yb) - npos
        if npos == 0 or nneg == 0: continue
        if rng is not None: fb = rng.permutation(fb)
        r = pd.Series(fb).rank().to_numpy()
        num += r[yb == 1].sum() - npos * (npos + 1) / 2.0
        den += npos * nneg
    return num / den if den else np.nan

def main():
    tr = pd.read_csv('data/train.csv')
    y = tr['addicted_label'].to_numpy()
    p = np.load(BLEND)
    print(f'stack OOF AUC {roc_auc_score(y, p):.6f}   mean(y-p) = {(y-p).mean():+.4f} '
          f'<- nonzero: h3 is a rank-average, not a calibrated probability\n', flush=True)

    folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))

    print(f'{"key subset":<22}{"cov%":>8}{"medN":>6}{"real":>9}{"ctrl mu":>9}'
          f'{"ctrl sd":>9}{"z":>8}')
    print('-' * 71)
    for name, cols in SUBSETS.items():
        k = (tr[FEAT].isna().astype(int).astype(str).agg(''.join, axis=1) if cols is None
             else keyframe(tr, cols)).to_numpy()
        rate = np.full(len(y), np.nan); cnt = np.zeros(len(y))
        for tri, vai in folds:
            g = pd.DataFrame({'k': k[tri], 'y': y[tri]}).groupby('k')['y'].agg(['sum','size'])
            s = pd.Series(k[vai])
            m = s.map(g['size']).to_numpy(); sm = s.map(g['sum']).to_numpy()
            with np.errstate(invalid='ignore'): rate[vai] = sm / m
            cnt[vai] = np.nan_to_num(m)
        ok = np.isfinite(rate)
        f, yy, pp = rate[ok], y[ok], p[ok]
        bins = pd.qcut(pp, 200, labels=False, duplicates='drop')
        real = cond_auc_core(f, yy, bins)
        ctrl = np.array([cond_auc_core(f, yy, bins, np.random.default_rng(s))
                         for s in range(NCTRL)])
        z = (real - ctrl.mean()) / ctrl.std(ddof=1)
        print(f'{name:<22}{ok.mean():>7.2%}{int(np.median(cnt[ok])):>6}'
              f'{real:>9.5f}{ctrl.mean():>9.5f}{ctrl.std(ddof=1):>9.5f}{z:>8.2f}')

    # ---------- corrected id structure ----------
    print('\n=== id structure (corrected: variation ACROSS classes, not vs zero) ===')
    resid = y - p
    idv = tr['id'].to_numpy()
    print(f'{"m":>5}{"F(real)":>11}{"F ctrl mu":>11}{"F ctrl sd":>11}{"z":>8}')
    for m in (2, 3, 4, 5, 7, 10, 16, 64, 100, 1000):
        cls = idv % m
        def fstat(r):
            df = pd.DataFrame({'c': cls, 'r': r})
            g = df.groupby('c')['r']
            mu, n = g.mean().to_numpy(), g.size().to_numpy()
            gm = r.mean()
            between = (n * (mu - gm) ** 2).sum() / (len(mu) - 1)
            within = ((r - df['c'].map(g.mean()).to_numpy()) ** 2).sum() / (len(r) - len(mu))
            return between / within
        real = fstat(resid)
        ctrl = np.array([fstat(np.random.default_rng(1000 + s).permutation(resid))
                         for s in range(NCTRL)])
        print(f'{m:>5}{real:>11.4f}{ctrl.mean():>11.4f}{ctrl.std(ddof=1):>11.4f}'
              f'{(real-ctrl.mean())/ctrl.std(ddof=1):>8.2f}')

    print(f'\ncorr(id, residual) = {np.corrcoef(idv.astype(float), resid)[0,1]:+.6f}')
    # monotone drift: AUC of id restricted within fine model-score bins already done above
    q = pd.qcut(idv, 20, labels=False)
    d = pd.DataFrame({'q': q, 'y': y, 'p': p}).groupby('q').agg(
        base=('y', 'mean'), auc=('y', 'size'))
    print('label base rate across 20 contiguous id blocks: '
          f'min {d["base"].min():.5f} max {d["base"].max():.5f} sd {d["base"].std():.5f} '
          f'(binomial sd {np.sqrt(y.mean()*(1-y.mean())/(len(y)/20)):.5f})')

if __name__ == '__main__':
    main()
