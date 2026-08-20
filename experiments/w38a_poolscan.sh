#!/usr/bin/env bash
# w38a -- batch `kaggle kernels output` over every prioritised kernel ref that no earlier
# sweep has dispositioned. w34 3.1 established that `kernels output` and `datasets download`
# are DIFFERENT endpoints; the pool ledger was built almost entirely on the dataset endpoint,
# so the kernel endpoint has never been run over the bulk of the 461-ref pool.
#
# Classification is by ARTEFACT, not by title: any file that is not submission-shaped and is
# large enough to hold 691,369 rows is a candidate. Writes one ledger row per ref.
set -u
OUT=notebooks/w38/out
LEDGER=experiments/w38/ledger.csv
mkdir -p "$OUT"
[ -f "$LEDGER" ] || echo "ref,n_files,verdict,files" > "$LEDGER"
while read -r ref; do
  [ -z "$ref" ] && continue
  grep -qF "$ref," "$LEDGER" && continue
  d="$OUT/$(echo "$ref" | tr '/' '_')"
  if [ ! -d "$d" ]; then
    mkdir -p "$d"
    timeout 240 kaggle kernels output "$ref" -p "$d" >/dev/null 2>&1
  fi
  # AutoML kernels dump multi-GB model trees; prune them the moment they land or the box
  # fills up (17 GB over 173 refs before this guard existed).
  find "$d" -mindepth 1 -maxdepth 2 -type d \( -name "AutogluonModels" -o -name "AutoML_*" \
       -o -name catboost_info -o -name cache -o -name TrainingSummary -o -name AutoViz_Plots \) \
       -exec rm -rf {} + 2>/dev/null
  files=$(ls "$d" 2>/dev/null | tr '\n' ' ')
  n=$(ls "$d" 2>/dev/null | wc -l)
  # candidate = a non-log, non-submission file at least 1 MB
  cand=$(find "$d" -maxdepth 3 -type f -size +1M 2>/dev/null \
         | grep -viE '/(submission|sample_submission)[^/]*\.csv$' | tr '\n' ' ')
  if [ -n "$cand" ]; then v=CANDIDATE; else if [ "$n" -eq 0 ]; then v=EMPTY; else v=nothing; fi; fi
  echo "$ref,$n,$v,\"$files\"" >> "$LEDGER"
  echo "$v  $ref  [$files]"
done < experiments/w38/scan_targets.txt
echo "SCAN DONE"
