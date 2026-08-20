#!/usr/bin/env bash
# w40a -- extend the w38a kernel-output sweep to every ref the ledger has NO row for.
#
# w38a scanned a prioritised 244 of the 461 listed refs. Re-listing the competition today
# (three sort orders, three pages each) returns 461 unique refs, of which 218 have no
# ledger row. 21 of those were never in `all_meta` at all; the rest were deprioritised out
# of w38a's target list, or dispositioned only through the DATASET endpoint -- and w34 3.1
# established those are different endpoints, so a dataset disposition does not mean the
# kernel output has been read. Targets are ordered by votes, so a disk abort still leaves
# the most promising refs scanned.
#
# Same artefact-based classification as w38a: a non-submission file >=1 MB is a CANDIDATE.
# Nothing here decides anything -- candidates go to the w34b gates in a vet step.
set -u
cd "$(dirname "$0")/.."
OUT=notebooks/w40/out
LEDGER=experiments/w40/ledger2.csv
LOG=experiments/w40a_poolscan2.log
MIN_FREE_GB=12
mkdir -p "$OUT"
[ -f "$LEDGER" ] || echo "ref,n_files,verdict,files" > "$LEDGER"
n=0
while read -r ref; do
  [ -z "$ref" ] && continue
  grep -qF "$ref," "$LEDGER" && continue
  free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "DISK ABORT at ${free}G free, $n refs scanned this pass" | tee -a "$LOG"; exit 2
  fi
  d="$OUT/$(echo "$ref" | tr '/' '_')"
  if [ ! -d "$d" ]; then
    mkdir -p "$d"
    timeout 240 kaggle kernels output "$ref" -p "$d" >/dev/null 2>&1
  fi
  find "$d" -mindepth 1 -maxdepth 2 -type d \( -name "AutogluonModels" -o -name "AutoML_*" \
       -o -name catboost_info -o -name cache -o -name TrainingSummary -o -name AutoViz_Plots \) \
       -exec rm -rf {} + 2>/dev/null
  files=$(ls "$d" 2>/dev/null | tr '\n' ' ')
  nf=$(ls "$d" 2>/dev/null | wc -l)
  cand=$(find "$d" -maxdepth 3 -type f -size +1M 2>/dev/null \
         | grep -viE '/(submission|sample_submission)[^/]*\.csv$' | tr '\n' ' ')
  if [ -n "$cand" ]; then v=CANDIDATE; else if [ "$nf" -eq 0 ]; then v=EMPTY; else v=nothing; fi; fi
  echo "$ref,$nf,$v,\"$files\"" >> "$LEDGER"
  echo "$v  $ref  [$files]" | tee -a "$LOG"
  n=$((n+1))
done < experiments/w40/scan_targets2.txt
echo "SCAN DONE, $n refs this pass" | tee -a "$LOG"
