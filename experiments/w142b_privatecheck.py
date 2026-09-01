"""w142b — is the private result published yet? Read it off the submission list, not the board.

The grader (w135b) refuses to grade because the downloaded board still looks public. But it
also reported "at least one privateScore". This asks how many, and whether the SELECTED group
is now non-empty -- which is what P1 turns on.
"""
import sys, json, collections
sys.path.insert(0, "experiments")
import kaggle_list

for group in ("all", "successful", "selected"):
    try:
        rows = kaggle_list.submissions(group=group)
    except SystemExit as e:
        print(f"{group:12s} READ FAILED: {e}")
        continue
    withpriv = [r for r in rows if str(r["privateScore"]).strip() not in ("", "None")]
    print(f"{group:12s} rows {len(rows):4d}   with privateScore {len(withpriv):4d}")
    if group == "selected":
        for r in rows:
            print(f"    SELECTED  {r['fileName']:44s} pub {r['publicScore']} priv {r['privateScore']}")

rows = kaggle_list.submissions(group="all")
json.dump(rows, open("experiments/w142b_allsubs.json", "w"), indent=1)
print(f"\nwrote experiments/w142b_allsubs.json  ({len(rows)} rows)")

# which rows carry a private score, and what are they?
withpriv = [r for r in rows if str(r["privateScore"]).strip() not in ("", "None")]
print(f"\nrows carrying privateScore: {len(withpriv)}")
for r in sorted(withpriv, key=lambda r: -float(r["privateScore"]))[:20]:
    print(f"  priv {r['privateScore']}  pub {r['publicScore']}  {r['fileName'][:46]:46s} {r['date']}")

st = collections.Counter(str(r["status"]) for r in rows)
print("\nstatus counts:", dict(st))
