"""w15c — controls for the missingness-shift result.

w15c_shift.py: reweighting the OOF pool to the test missingness distribution raises the
estimated AUC by +0.000905 and closes 88% of the +0.001025 CV->LB gap. Before that can be
believed it needs the control this workspace requires after erroran.py's chi2 near-miss:
does ANY reweighting of the same magnitude do this, or only the test-directed one?

Three arms, all with the identical weight MARGINAL distribution:
  real     : w(pattern) = P_test(pattern) / P_train(pattern)
  permuted : the same 4096 ratio values, randomly reassigned to patterns  (24 seeds)
  inverse  : P_train / P_test, i.e. reweighting AWAY from test by the same amount

If the effect is a real directional property of the test mask mix, `real` is large and
positive, `permuted` sits at zero, and `inverse` is large and negative.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from w15c_shift import maskcode, wauc, FEAT

BLENDS = ['blend159av_h3', 'blend158_h3', 'blend156', 'blend158_logit', 'stack_pub74_logit']

def norm(w):
    hi = np.quantile(w[w > 0], 0.999)
    w = np.clip(w, 0, hi)
    return w / w.mean()

def main():
    tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
    y = tr['addicted_label'].to_numpy()
    ktr, kte = maskcode(tr), maskcode(te)
    ptr = np.bincount(ktr, minlength=4096) / len(ktr)
    pte = np.bincount(kte, minlength=4096) / len(kte)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(ptr > 0, pte / ptr, 0.0)
        inv = np.where((pte > 0) & (ptr > 0), ptr / pte, 0.0)

    w_real = norm(ratio[ktr])
    w_inv = norm(inv[ktr])

    print(f'{"blend":<20}{"plain CV":>11}{"real":>11}{"inverse":>11}'
          f'{"perm mu":>11}{"perm sd":>10}{"z(real)":>10}')
    print('-' * 84)
    for b in BLENDS:
        p = np.load(f'submissions/oof_{b}.npy')
        base = roc_auc_score(y, p)
        a_real = wauc(y, p, w_real) - base
        a_inv = wauc(y, p, w_inv) - base
        perm = []
        for s in range(24):
            rng = np.random.default_rng(s)
            idx = np.flatnonzero(ptr > 0)
            r2 = np.zeros(4096)
            r2[idx] = rng.permutation(ratio[idx])     # same values, wrong patterns
            perm.append(wauc(y, p, norm(r2[ktr])) - base)
        perm = np.array(perm)
        z = (a_real - perm.mean()) / perm.std(ddof=1)
        print(f'{b:<20}{base:>11.6f}{a_real:>+11.6f}{a_inv:>+11.6f}'
              f'{perm.mean():>+11.6f}{perm.std(ddof=1):>10.6f}{z:>10.1f}')

    # which columns' shift carries the effect: reweight one column's NaN rate at a time
    print('\n=== per-column attribution (reweight ONE column\'s NaN rate to test, '
          'others left alone) ===')
    p = np.load('submissions/oof_blend159av_h3.npy')
    base = roc_auc_score(y, p)
    m = tr[FEAT].isna().to_numpy()
    print(f'{"column":<26}{"tr_na":>8}{"te_na":>8}{"dAUC":>11}')
    rows = []
    for i, c in enumerate(FEAT):
        a = m[:, i].mean(); b = te[c].isna().mean()
        w = np.where(m[:, i], b / a, (1 - b) / (1 - a))
        d = wauc(y, p, w / w.mean()) - base
        rows.append((c, a, b, d))
        print(f'{c:<26}{a:>8.4f}{b:>8.4f}{d:>+11.6f}')
    print(f'{"SUM of single-column":<26}{"":>8}{"":>8}'
          f'{sum(r[3] for r in rows):>+11.6f}')
    print(f'{"joint (all at once)":<26}{"":>8}{"":>8}'
          f'{wauc(y, p, w_real) - base:>+11.6f}')

if __name__ == '__main__':
    main()
