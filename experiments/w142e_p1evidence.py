"""w142e — P1 came back UNGRADED on the route the grader took. Is there a second route?

w135b compares the private max of the WANTED pair against the private max of the AUTO pair and
finds both are 0.97093 at the board's 5 d.p., so it cannot say which was selected. That is a
correct refusal, and no arithmetic on those four numbers will rescue it.

But P1 makes a claim the four numbers are not the only evidence for: that Kaggle AUTO-SELECTED
ON PUBLIC SCORE. That claim is testable, because auto-selection on public has a consequence --
it must leave behind any file whose private score is high but whose public score is not.
"""
import json
import pandas as pd

rows = json.load(open("experiments/w142b_allsubs.json"))
df = pd.DataFrame(rows)
df["stem"] = df["fileName"].str.replace(r"\.csv$", "", regex=True)
df["pub"] = pd.to_numeric(df["publicScore"], errors="coerce")
df["priv"] = pd.to_numeric(df["privateScore"], errors="coerce")
BOARD_PRIV, BOARD_PUB = 0.97093, 0.97119

print("=" * 92)
print("w142e — a second route to P1")
print("=" * 92)

best_priv = df["priv"].max()
top_priv = df[df["priv"] >= best_priv - 1e-12]
print(f"\n  our best PRIVATE score over all {len(df)} sends: {best_priv:.5f}")
for r in top_priv.itertuples():
    print(f"      {r.stem:30s} priv {r.priv:.5f}  pub {r.pub:.5f}")
print(f"  the private BOARD shows us at:                   {BOARD_PRIV:.5f}")

print("\n  E1  the board is scoring a SELECTION, not the best of all our files")
print(f"      board {BOARD_PRIV:.5f} < best available {best_priv:.5f} "
      f"({(best_priv - BOARD_PRIV)*1e6:+.0f}e-6). If the private board simply reported our best")
print("      private file, these would be equal. They are not, so a selection was applied.")

top_pub = df[df["pub"] >= df["pub"].max() - 1e-12]
print(f"\n  E2  the selection maximises PUBLIC, and that is what left {top_priv.iloc[0]['stem']} behind")
print(f"      highest public score we ever sent: {df['pub'].max():.5f}, held by "
      f"{len(top_pub['stem'].unique())} distinct file(s):")
for s in sorted(top_pub["stem"].unique()):
    p = df.loc[df["stem"] == s, "priv"].max()
    print(f"      {s:30s} pub {df.loc[df['stem']==s,'pub'].max():.5f}  priv {p:.5f}")
print(f"      max private over that public tier = "
      f"{top_pub.groupby('stem')['priv'].max().max():.5f}, which is the board's {BOARD_PRIV:.5f}")
print(f"      and the best-private file sits at pub {top_priv.iloc[0]['pub']:.5f}, BELOW the "
      f"{df['pub'].max():.5f} tier, so a public-argmax rule cannot pick it.")

print("\n  E3  the API reports zero SELECTED submissions")
print("      kaggle_list.submissions(group='selected') -> 0 rows, consistent with 'nobody")
print("      clicked'. ⚠ WEAK ON ITS OWN: no control exists for whether this endpoint ever")
print("      reports an AUTO-selection, and the competition is closed so one cannot be made.")

print("\n" + "-" * 92)
print("  VERDICT. E1+E2 confirm the mechanism P1 named: a selection was applied and it was")
print("  made on PUBLIC score, discarding our best private file. They do NOT distinguish")
print("  WANTED from AUTO -- nothing can, both are 0.97093 at 5 d.p. P1 stays UNGRADED on")
print("  the click itself, and the click is shown to have been worth 0 either way.")
