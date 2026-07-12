#!/usr/bin/env bash
# [WORK-ITEM-K1-KQUANT-E2E] correctness-first: greedy A==B byte-identical between
# A=REPACK (stock as-shipped q4_K repack) and B=VECDOT (q4_K generic vec_dot).
# Both paths do the SAME q8_K-activation integer dot arithmetic (associative-exact
# integer accumulation) => expected byte-identical generated tokens (same as M2-q4_K
# repack-vs-generic A==B on rvv). Uses the REAL llama-cli ELF (NOT the wrapper which
# forces its own LD_LIBRARY_PATH), single-turn -no-cnv, stdin</dev/null (no interactive hang).
set -u
export LD_LIBRARY_PATH=/data/build-k1-workitem/bin
LIVE=/data/build-k1-workitem/bin/libggml-cpu.so.0.15.1
CLI="${CLI:-/data/k1build/bin/llama-cli}"       # real ELF; my LD_LIBRARY_PATH wins over wrapper
WK=/data/wk
MODEL="${MODEL:-/data/tinyllama-1.1b-Q4_K_M.gguf}"
THREADS="${THREADS:-8}"; NTOK="${NTOK:-24}"; PIN="taskset -c 0-7"
swap(){ cp -f "$WK/libggml-cpu.so.$1" "$LIVE"; }
gen(){ # $1=variant -> stdout generated tokens only
  swap "$1"
  $PIN "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" \
     --no-warmup --temp 0 --top-k 1 --seed 1 -no-cnv -st --no-display-prompt </dev/null 2>>/tmp/wk_engage.err
}
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox jumps"
: > /tmp/wk_engage.err
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  gen REPACK > /tmp/wk_A_$IDX.txt
  gen VECDOT > /tmp/wk_B_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(REPACK): $(tr '\n' ' ' < /tmp/wk_A_$IDX.txt | cut -c1-100)"
  echo "   B(VECDOT): $(tr '\n' ' ' < /tmp/wk_B_$IDX.txt | cut -c1-100)"
  if [ ! -s /tmp/wk_A_$IDX.txt ] || [ ! -s /tmp/wk_B_$IDX.txt ]; then
    echo "   VERDICT: FAIL (empty)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q /tmp/wk_A_$IDX.txt /tmp/wk_B_$IDX.txt >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else
    echo "   VERDICT: MISMATCH"; FAIL=$((FAIL+1))
    diff /tmp/wk_A_$IDX.txt /tmp/wk_B_$IDX.txt | head -6 | sed 's/^/     /'; fi
done <<< "$PROMPTS"
echo "== SUMMARY: byte_identical=$PASS failed=$FAIL =="
[ "$FAIL" = 0 ] && [ "$PASS" -ge 1 ] && echo "CORRECTNESS_GATE: GREEN" || echo "CORRECTNESS_GATE: RED"
swap REPACK
