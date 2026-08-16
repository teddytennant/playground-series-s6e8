"""w15c — exact and near-exact row-identity collisions between train and test.

Mission: measure test<->train feature-tuple twins, their empirical label rate,
and whether a twin lookup beats the model on the covered subset.
Nothing here is fold-aware yet; this is the census only.
"""
import numpy as np, pandas as pd, sys

FEAT = ['age','daily_screen_time_hours','social_media_hours','gaming_hours',
        'work_study_hours','sleep_hours','notifications_per_day',
        'app_opens_per_day','weekend_screen_time','gender','stress_level',
        'academic_work_impact']

def keyframe(df, cols):
    """Stable string key over cols; NaN -> '@'. Floats via repr of the raw csv text
    would be safest, but values are on a fixed lattice so a 4dp format is exact."""
    parts = []
    for c in cols:
        s = df[c]
        if s.dtype.kind == 'O':
            parts.append(s.fillna('@').astype(str))
        else:
            parts.append(s.map(lambda v: '@' if pd.isna(v) else format(v, '.4f')))
    return parts[0].str.cat(parts[1:], sep='|')

def main():
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    y = tr['addicted_label'].values
    print(f'train {tr.shape} test {te.shape}', flush=True)

    ktr = keyframe(tr, FEAT)
    kte = keyframe(te, FEAT)

    # --- 1. train-internal duplicates ---
    vc = ktr.value_counts()
    print('\n=== TRAIN-INTERNAL exact duplicate groups (all 12 features) ===')
    print(f'distinct keys: {len(vc):,} over {len(ktr):,} rows')
    print(f'rows in a group of size>1: {int(vc[vc>1].sum()):,} '
          f'({vc[vc>1].sum()/len(ktr):.6%})')
    print(f'largest group: {vc.iloc[0]}')
    print('group-size histogram (top):')
    print(vc.value_counts().sort_index().head(10))

    # --- 2. test -> train exact twins ---
    grp = pd.DataFrame({'k': ktr, 'y': y}).groupby('k')['y'].agg(['size','sum'])
    hit = kte.map(grp['size']).fillna(0).astype(int)
    print('\n=== TEST -> TRAIN exact twins (all 12 features) ===')
    print(f'test rows with >=1 exact train twin: {int((hit>0).sum()):,} '
          f'/ {len(kte):,} = {(hit>0).mean():.6%}')
    print(f'total twin rows available: {int(hit.sum()):,}')
    if (hit > 0).any():
        print('twin-count histogram:')
        print(hit[hit > 0].value_counts().sort_index().head(10))
        cov = hit > 0
        rate = (kte[cov].map(grp['sum']) / kte[cov].map(grp['size']))
        print(f'empirical label rate of twins on covered rows: {rate.mean():.6f}')
        print(f'  vs train base rate {y.mean():.6f}')

    # --- 3. test -> test internal duplicates ---
    vte = kte.value_counts()
    print('\n=== TEST-INTERNAL exact duplicates ===')
    print(f'distinct keys: {len(vte):,} over {len(kte):,} rows; '
          f'rows in group>1: {int(vte[vte>1].sum()):,}')

    # --- 4. how deep does collision go? drop columns one at a time ---
    print('\n=== collision coverage as columns are dropped (test->train) ===')
    print(f'{"cols used":>10} {"dropped":<26} {"test rows w/ twin":>18} {"coverage":>10}')
    for drop in [[], ['age'], ['weekend_screen_time'],
                 ['weekend_screen_time','daily_screen_time_hours'],
                 ['age','gender','stress_level','academic_work_impact'],
                 ['weekend_screen_time','age','gaming_hours','work_study_hours'],
                 ['weekend_screen_time','age','gaming_hours','work_study_hours',
                  'sleep_hours','notifications_per_day','app_opens_per_day']]:
        cols = [c for c in FEAT if c not in drop]
        a = keyframe(tr, cols); b = keyframe(te, cols)
        s = a.value_counts()
        h = b.map(s).fillna(0).astype(int)
        print(f'{len(cols):>10} {",".join(drop)[:26]:<26} {int((h>0).sum()):>18,} '
              f'{(h>0).mean():>9.4%}')

if __name__ == '__main__':
    main()
