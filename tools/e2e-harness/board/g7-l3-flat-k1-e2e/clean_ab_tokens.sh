#!/usr/bin/env bash
# Normalization stripping spinner/backspace/box-draw AND the inline perf-stats line
# ([ Prompt: X ts Generation: Y ts ]) whose numbers ARE the throughput we measure.
# Compares ONLY the deterministic completion tokens (A ON vs B OFF).
DIR="$1"
clean(){ LC_ALL=C tr -cd '\11\12\40-\176' < "$1" | tr -d '|/\\\b' | sed -e 's/-//g' -e 's/\[ Prompt:[^]]*\]//g' | tr -s ' \t\n' ' ' | sed 's/^ *//;s/ *$//'; }
P=0;F=0
for IDX in 1 2 3 4; do
  A="$DIR/A_$IDX.txt"; B="$DIR/B_$IDX.txt"
  ca=$(clean "$A"); cb=$(clean "$B")
  # show just the completion portion (after last "> ")
  comp="${ca##*> }"
  if [ "$ca" = "$cb" ]; then echo "prompt[$IDX]: COMPLETION-IDENTICAL  ::  ${comp:0:80}"; P=$((P+1))
  else echo "prompt[$IDX]: DIVERGE"; diff <(echo "$ca") <(echo "$cb"); F=$((F+1)); fi
done
echo "== CLEAN SUMMARY: token_identical=$P diverge=$F =="
