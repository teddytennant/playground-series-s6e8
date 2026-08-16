# Manual wave — Kaggle day UTC 2026-08-16

This wave is being driven by hand because **the automated nightly run failed**.
`kaggle-playground.timer` fired at 20:10 EDT on 2026-08-15 and every one of its 10
slots died immediately with:

```
Failed to authenticate: OAuth session expired and could not be refreshed
```

That is the headless `claude -p` login expiring under systemd, not anything wrong
with Kaggle or the workspace. The timer has already fired, so it will not retry
until 20:10 EDT on 2026-08-16 and the day's whole quota is available to this wave.

**Never trust a submission count quoted in this file or in a slot prompt.** The
first version of this file said zero submissions had been made on the current day;
it was already 1, because an interrupted slot-1 attempt had submitted without
writing a journal entry. Count the live API listing yourself, every time.

## What that means for you

- The Kaggle submission day rolls at **00:00 UTC = 20:00 EDT**. The day that matters
  is **UTC 2026-08-16**, which began at 20:00 EDT on Aug 15.
- Daily cap is **10** (confirmed twice, see `RESEARCH.md`).
- Ten slots are being run **sequentially**, one agent per slot. You are one of them.
  The slot before you has already finished and written to `JOURNAL.md`. Read it.
- Submissions dated `2026-08-15` in the API listing belong to the **previous** day
  and do not count against your quota. Count only rows dated `2026-08-16`.

## Verify the count yourself before you submit

```
kaggle competitions submissions -c playground-series-s6e8 -v | head -20
```

Count the rows whose date is `2026-08-16`. That number plus one must be <= 10.
The CLI also prints `N submissions remaining today` after a successful submit —
that line is authoritative, and if it disagrees with your count, believe the CLI
and write the discrepancy into `JOURNAL.md`.

## Standing notes for this wave

- **Submit.** The brief's "submission economics" section overrides the base
  playbook's "skip a slot rather than spend it on noise" rule for this competition.
  Public LB is best-of-all-submissions, nothing evicts anything, so an unused slot
  is pure waste. Every slot should end in a real submission unless the cap is
  genuinely reached or the API is down.
- **Do not resubmit an identical file.** Scores are deterministic. Vary something
  real. Check `submissions/` and the API history before sending — this account has
  already sent a lot of files.
- **Final selection is still on CV.** Chase the public slice for information, never
  for the deadline pick.
- The angle you are handed comes from a rotation list written when this competition
  was cold. Several of those angles (build the CV harness, tune a first LightGBM,
  find the original dataset) are long since closed out — the journal will tell you.
  If your angle is exhausted, say so explicitly in `JOURNAL.md`, then spend the slot
  on the highest-value open question you can find instead. Do not simply repeat the
  previous slot's idea.
- If a shell command fails because of sandbox or network restrictions, re-run it
  with the sandbox disabled. Kaggle API access and the workspace `.venv` are both
  expected to work.
- **Every number you write must be a number you measured.** On 2026-08-16 a slot
  shipped a Kaggle submission message quoting per-fold deltas it had constructed to
  sum to the correct mean rather than reading them off the run. Kaggle submission
  descriptions are immutable, so that is permanent and public. If you have not
  computed a figure, do not write it — in a submission message, in `JOURNAL.md`, in
  `RESEARCH.md`, or in a report. Write "not measured" instead. A submission message
  with three real numbers is worth more than one with ten, most of them invented.
- Git author is Teddy Tennant <teddytennant@icloud.com>. No AI attribution anywhere:
  not in commits, not in code comments, not in anything posted to Kaggle.
