"""w185: score every submission file that was built but never sent.

The board closed 2026-08-31 and the final rank (319/3531) is locked, but Kaggle still
accepts late submissions at 100/day and returns publicScore AND privateScore. Verified
this run by resending w36_ad199stdcorr_ens4.csv: 0.97119 / 0.97093, matching the record.

32 files sit in submissions/ that appear in no row of the 201-row history. Seven are
`w42_ad217*` and seven are `w50_ad216*` -- LATER stack generations than `ad211`, which
holds this workspace's best private score (0.97094). Nobody ever found out what they
score. This sends them and reads the scores back.

⚠ MEASUREMENT ONLY. Nothing here can change the final standing, and the private column
must not be used to pick anything -- see the oracle section at the top of RESEARCH.md.
"""
import subprocess, time, json, sys, csv

COMP = 'playground-series-s6e8'
PAUSE = 4.0


def sent_names():
    out = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', COMP,
                          '--page-size', '200', '-v'],
                         capture_output=True, text=True).stdout
    rows = list(csv.DictReader(out.splitlines()))
    return rows


def main():
    rows = sent_names()
    have = {r['fileName'] for r in rows}
    import glob, os
    disk = sorted(os.path.basename(p) for p in glob.glob('submissions/*.csv'))
    todo = [f for f in disk if f not in have]
    print(f'{len(rows)} in history, {len(disk)} on disk, {len(todo)} never sent',
          flush=True)
    if '--dry' in sys.argv:
        for f in todo:
            print('  would send', f)
        return
    for i, f in enumerate(todo, 1):
        msg = f'w185 late-score {f[:-4]} — MEASUREMENT of an unsent file, not a pick'
        r = subprocess.run(['kaggle', 'competitions', 'submit', '-c', COMP,
                            '-f', f'submissions/{f}', '-m', msg],
                           capture_output=True, text=True)
        ok = 'Successfully submitted' in (r.stdout + r.stderr)
        print(f'[{i:2d}/{len(todo)}] {"ok " if ok else "FAIL"} {f}', flush=True)
        if not ok:
            print('   ', (r.stdout + r.stderr).strip()[-300:], flush=True)
        time.sleep(PAUSE)
    print('\nwaiting 60s for scoring...', flush=True)
    time.sleep(60)
    rows = sent_names()
    by = {}
    for r in rows:
        if r['description'].startswith('w185 late-score'):
            by[r['fileName']] = (r['publicScore'], r['privateScore'], r['status'])
    print(f'\n{"file":42s} {"public":>9s} {"private":>9s}')
    got = [(v[1], k, v) for k, v in by.items() if v[1]]
    for _, k, v in sorted(got, reverse=True):
        print(f'{k:42s} {v[0]:>9s} {v[1]:>9s}')
    print(f'\n{len(by)} scored back of {len(todo)} sent')
    json.dump(by, open('experiments/w185c_unsent_scores.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
