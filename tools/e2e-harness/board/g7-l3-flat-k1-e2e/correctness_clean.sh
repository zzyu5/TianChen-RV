#!/usr/bin/env bash
# Clean re-compare of saved A/B correctness files (strip llama-cli load-spinner + warning
# that pollute the raw byte-diff). Compares ONLY the generated completion text.
# Usage: FORMAT=q4_0 correctness_clean.sh
set -u
FMT="${FORMAT:?}"
clean(){ # strip warning line + "Loading model..." spinner runs; keep completion text
  sed -e 's/--no-conversation is not supported[^L]*//g' \
      -e 's/Loading model\.\.\.//g' \
      -e 's/[|/\\-]//g' "$1" | tr -s ' \t\n' ' ' | sed 's/^ *//; s/ *$//'
}
PASS=0; FAIL=0
for IDX in 1 2 3 4; do
  A=/tmp/flat_${FMT}_A_$IDX.txt; B=/tmp/flat_${FMT}_B_$IDX.txt
  [ -s "$A" ] && [ -s "$B" ] || { echo "prompt[$IDX]: MISSING"; FAIL=$((FAIL+1)); continue; }
  ca=$(clean "$A"); cb=$(clean "$B")
  echo "prompt[$IDX] A: $(echo "$ca"|cut -c1-110)"
  echo "prompt[$IDX] B: $(echo "$cb"|cut -c1-110)"
  if [ "$ca" = "$cb" ]; then echo "  VERDICT: COMPLETION-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else echo "  VERDICT: DIVERGE (both coherent? inspect)"; FAIL=$((FAIL+1)); fi
done
echo "== CLEAN SUMMARY: identical=$PASS diverge=$FAIL =="
[ "$FAIL" = 0 ] && echo "CORRECTNESS(clean): GREEN" || echo "CORRECTNESS(clean): inspect diverge (fp-order near-tie vs garbage)"
