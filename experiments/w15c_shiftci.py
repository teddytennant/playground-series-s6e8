"""w15c — the right uncertainty on the +0.000903 reweighting effect, and the matched control.

w15c_shiftctrl.py ran a permutation control that reassigned the 4096 pattern-ratios to
random patterns: it gave sd 0.003029, so real - perm was only z = 0.9. That control is the
WRONG null and it should not be read as weakening the result. It answers "would an
arbitrary reweighting move the AUC?" (yes, by +-0.003), whereas the weights here are not
chosen, they are COMPUTED from measured train/test NaN rates. Permuting patterns destroys
the smoothness of the weight function, not just its direction, and inflates the variance.

Two correct instruments instead:

  (a) BOOTSTRAP. The only sampling uncertainty in +0.000903 is the estimate of the test
      mask distribution from 296,302 test rows. Resample test rows, recompute the weights,
      recompute the weighted AUC. That is the honest error bar on the plug-in estimate.

  (b) COLUMN-SHUFFLE control. Keep the same twelve per-column NaN-rate deltas and assign
      them to the WRONG columns. This preserves the magnitude and the smoothness of the
      weight function and destroys only which column moved. It answers the one genuinely
      open question: is it a coincidence that the columns whose NaN rate FELL in test are
      the two the generator's rule actually uses?
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from w15c_shift import maskcode, wauc, FEAT

NBOOT, NPERM = 200, 300

def norm(w):
    w = np.clip(w, 0, np.quantile(w[w > 0], 0.999))
    return w / w.mean()

def main():
    tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
    y = tr['addicted_label'].to_numpy()
    ktr, kte = maskcode(tr), maskcode(te)
    ptr = np.bincount(ktr, minlength=4096) / len(ktr)
    p = np.load('submissions/oof_blend159av_h3.npy')
    base = roc_auc_score(y, p)

    def eff(pte):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(ptr > 0, pte / ptr, 0.0)
        return wauc(y, p, norm(r[ktr])) - base

    pte0 = np.bincount(kte, minlength=4096) / len(kte)
    point = eff(pte0)
    print(f'point estimate (joint mask reweighting): {point:+.6f}')

    rng = np.random.default_rng(0)
    boot = np.array([eff(rng.multinomial(len(kte), pte0) / len(kte))
                     for _ in range(NBOOT)])
    print(f'(a) bootstrap over {len(kte):,} test rows, {NBOOT} reps: '
          f'mean {boot.mean():+.6f}  sd {boot.std(ddof=1):.6f}  '
          f'95% CI [{np.quantile(boot,0.025):+.6f}, {np.quantile(boot,0.975):+.6f}]')
    print(f'    -> the effect is {point/boot.std(ddof=1):.0f} sigma from zero as an estimate.')

    # (b) column-shuffle control: same 12 deltas, wrong columns
    m = tr[FEAT].isna().to_numpy()
    a = m.mean(0)
    b = np.array([te[c].isna().mean() for c in FEAT])

    def indep_eff(target):
        """weights from per-column independent reweighting a -> target"""
        w = np.ones(len(m))
        for i in range(len(FEAT)):
            w *= np.where(m[:, i], target[i] / a[i], (1 - target[i]) / (1 - a[i]))
        return wauc(y, p, w / w.mean()) - base

    real_ind = indep_eff(b)
    print(f'\n(b) independent per-column reweighting, real assignment: {real_ind:+.6f}')
    d = b - a
    rng = np.random.default_rng(1)
    perm = np.array([indep_eff(a + d[rng.permutation(len(FEAT))]) for _ in range(NPERM)])
    z = (real_ind - perm.mean()) / perm.std(ddof=1)
    print(f'    column-shuffled control ({NPERM} reps): mean {perm.mean():+.6f} '
          f'sd {perm.std(ddof=1):.6f}   z(real) = {z:+.2f}')
    print(f'    P(shuffle >= real) = {(perm >= real_ind).mean():.4f}')
    print('\n    Reads as: how unusual is it that the NaN-rate drops landed on the columns')
    print('    that matter most? Not whether the +0.0009 is real -- (a) settles that.')

if __name__ == '__main__':
    main()
