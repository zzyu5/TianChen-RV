#!/usr/bin/env bash
# [G5 M2 q4_K] v2 RIGOROUS engage-proof + correctness gate. Uses llama-completion
# (pure completion; --no-display-prompt => stdout == ONLY generated tokens) so the
# A(ON=emitted vl=8) vs B(OFF=stock generic) comparison is a RAW byte-diff of real
# generated tokens (MIRAGE-proof; no CLI chrome). Kernel-engage asserted via banners.
# HARD GATE: every prompt must be byte-identical A==B AND kernel must have engaged.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_q4k
MODEL="${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"; NTOK="${NTOK:-24}"
PIN="taskset -c $CORES"
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q4k$1" "$LIVE"; }
echo "== MODEL=$MODEL  sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)  cores=$CORES threads=$THREADS ntok=$NTOK =="

: > /tmp/g5_q4k_engage_all.err
gen(){ # $1=variant $2=idx  -> stdout raw generated tokens; stderr banners+stats to file
  swap "$1"
  printf '%s' "$PROMPT" | timeout "${GEN_TIMEOUT:-300}" $PIN "$BIN/llama-completion" \
    -m "$MODEL" --no-display-prompt --no-warmup --temp 0 --top-k 1 --seed 1 \
    -t "$THREADS" -n "$NTOK" 2>>/tmp/g5_q4k_engage_all.err
}
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox
In the year 2050, humanity"
PASS=0; FAIL=0; IDX=0; ENGAGED=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  gen ON  "$IDX" > /tmp/g5_q4k_A_$IDX.txt
  gen OFF "$IDX" > /tmp/g5_q4k_B_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(ON emit): $(tr '\n' ' ' < /tmp/g5_q4k_A_$IDX.txt | cut -c1-120)"
  echo "   B(OFF stk): $(tr '\n' ' ' < /tmp/g5_q4k_B_$IDX.txt | cut -c1-120)"
  if [ ! -s /tmp/g5_q4k_A_$IDX.txt ] || [ ! -s /tmp/g5_q4k_B_$IDX.txt ]; then
    echo "   VERDICT: FAIL (empty generation)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q /tmp/g5_q4k_A_$IDX.txt /tmp/g5_q4k_B_$IDX.txt >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else
    echo "   VERDICT: MISMATCH (A!=B)"; FAIL=$((FAIL+1))
    diff /tmp/g5_q4k_A_$IDX.txt /tmp/g5_q4k_B_$IDX.txt | head -8 | sed 's/^/     /'; fi
done <<< "$PROMPTS"
ENGAGED=$(grep -c "TCRV G5-M2 EMITTED" /tmp/g5_q4k_engage_all.err || true)
echo "== ENGAGE: total emitted-kernel banner fires (ON runs) = $ENGAGED (expect >0) =="
echo "== LOGITS SANITY =="
if grep -aiE 'nan|-nan|[^a-z]inf[^a-z]' /tmp/g5_q4k_engage_all.err >/dev/null 2>&1; then
  echo "   FAIL: NaN/Inf in stderr"; FAIL=$((FAIL+1)); else echo "   OK: no NaN/Inf"; fi
echo "== CORRECTNESS SUMMARY: byte_identical=$PASS failed=$FAIL engaged_banners=$ENGAGED =="
if [ "$FAIL" = 0 ] && [ "$PASS" -ge 1 ] && [ "$ENGAGED" -ge 1 ]; then echo "CORRECTNESS_GATE: GREEN"; else echo "CORRECTNESS_GATE: RED"; fi
swap OFF
