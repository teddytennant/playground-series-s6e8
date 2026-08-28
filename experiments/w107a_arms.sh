set -u
cd /home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
PY=.venv/bin/python
run(){ W107A_DOC="$1" timeout 300 $PY experiments/w107a_lineref.py > /tmp/arm.out 2>&1; echo $?; }
say(){ printf '%-58s %s\n' "$1" "$2"; }

# ---- C1-  live document is clean
rc=$(run RESEARCH.md); [ "$rc" = 0 ] && say "C1- live document green" PASS || say "C1- live document green" "FAIL(rc=$rc)"

# ---- C1+  a freshly planted internal ref is caught
cp RESEARCH.md /tmp/d1.md
printf '\nA planted claim, see line 4242 above for the number.\n' >> /tmp/d1.md
rc=$(run /tmp/d1.md)
grep -q 'cites line 4242 of the document itself' /tmp/arm.out && [ "$rc" = 1 ] \
  && say "C1+ planted internal ref caught" PASS || say "C1+ planted internal ref caught" "FAIL(rc=$rc)"

# ---- C2  historical true positive: restore the w26-era stale pointer this run removed
cp RESEARCH.md /tmp/d2.md
$PY - <<'PY'
import io
s=io.open('/tmp/d2.md',encoding='utf-8').read()
s=s.replace("three model classes** — grep `THE CT SKEW IS A PROPERTY OF THE MATRIX`: −19.26e-6 (xgb),\n   −82.68e-6 (cat), on top of the",
            "three model classes** — see line ~7137: −19.26e-6 (xgb), −82.68e-6 (cat), on top of the")
io.open('/tmp/d2.md','w',encoding='utf-8').write(s)
PY
rc=$(run /tmp/d2.md)
n=$(grep -c '^FAIL ' /tmp/arm.out)
grep -q 'cites line 7137' /tmp/arm.out && [ "$n" = 1 ] \
  && say "C2 historical stale ref rediscovered, alone" PASS || say "C2 historical stale ref rediscovered, alone" "FAIL(n=$n rc=$rc)"

# ---- C3  the fence-stripper is load-bearing, and not vacuous
cp RESEARCH.md /tmp/d3.md
printf '\n```\nquoted transcript mentioning line 4242\n```\n' >> /tmp/d3.md
rc=$(run /tmp/d3.md)
[ "$rc" = 0 ] && say "C3a citation inside a fence is invisible" PASS || say "C3a citation inside a fence is invisible" "FAIL(rc=$rc)"
sed 's/^        if ln.lstrip().startswith("```"):/        if False:/' experiments/w107a_lineref.py > /tmp/nofence.py
W107A_DOC=/tmp/d3.md timeout 300 $PY /tmp/nofence.py > /tmp/arm2.out 2>&1; rc2=$?
[ "$rc2" = 1 ] && grep -q 'cites line 4242' /tmp/arm2.out \
  && say "C3b ...and visible once the stripper is off" PASS || say "C3b ...and visible once the stripper is off" "FAIL(rc=$rc2)"

# ---- C4  external resolution really can fail (range + missing file)
cp RESEARCH.md /tmp/d4.md
printf '\nSee `agent/features.py` line 99999 for the block.\n' >> /tmp/d4.md
rc=$(run /tmp/d4.md)
grep -q 'has only .* lines, cited 99999' /tmp/arm.out && [ "$rc" = 1 ] \
  && say "C4a external out-of-range caught" PASS || say "C4a external out-of-range caught" "FAIL(rc=$rc)"
cp RESEARCH.md /tmp/d4b.md
printf '\nSee `experiments/no_such_file_xyz.py` line 12 for the block.\n' >> /tmp/d4b.md
rc=$(run /tmp/d4b.md)
grep -q 'named file(s) not found on disk' /tmp/arm.out && [ "$rc" = 1 ] \
  && say "C4b external missing file caught" PASS || say "C4b external missing file caught" "FAIL(rc=$rc)"

# ---- C5  dates are not citations
cp RESEARCH.md /tmp/d5.md
printf '\nA note dated line 2026-08-11 and another at line 08-27, neither a pointer.\n' >> /tmp/d5.md
rc=$(run /tmp/d5.md)
[ "$rc" = 0 ] && say "C5 date-shaped text is not a citation" PASS || say "C5 date-shaped text is not a citation" "FAIL(rc=$rc)"

# ---- C0  non-vacuity: an instrument reporting absence must prove it can still see
printf 'a document with no citations at all\n' > /tmp/d0.md
rc=$(run /tmp/d0.md)
grep -q 'C0: scanner found NO line citations' /tmp/arm.out && [ "$rc" = 1 ] \
  && say "C0a empty document FAILS, does not go green" PASS || say "C0a empty document FAILS, does not go green" "FAIL(rc=$rc)"
printf 'Only an internal one: see line 4242.\n' > /tmp/d0b.md
rc=$(run /tmp/d0b.md)
grep -q 'C0: scanner resolved NO external citations' /tmp/arm.out && [ "$rc" = 1 ] \
  && say "C0b no-external-exercised FAILS" PASS || say "C0b no-external-exercised FAILS" "FAIL(rc=$rc)"
