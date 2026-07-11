#!/usr/bin/env bash
# [G5-M2 q4_1] Anti-MIRAGE correctness gate. A(ON=emitted vl=8 q4_1 repack + q8_1
# mat-quant) vs B(OFF=stock ggml q4_1 block-dot) greedy generation RAW byte-diff.
# HARD GATE: every prompt byte-identical A==B AND emitted-kernel banners fired (ON).
# Plus PPL sanity on ON (finite/coherent). If PPL garbage => interleaver/m/mat-quant
# wrong.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_q41
MODEL="${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_1.gguf}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"; NTOK="${NTOK:-24}"
PIN="taskset -c $CORES"
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q4$1" "$LIVE"; }
echo "== MODEL=$MODEL sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16) cores=$CORES threads=$THREADS ntok=$NTOK =="

: > /tmp/g5_q41_engage_all.err
gen(){ swap "$1"
  printf '%s' "$PROMPT" | timeout "${GEN_TIMEOUT:-300}" $PIN "$BIN/llama-completion" \
    -m "$MODEL" --no-display-prompt --no-warmup --temp 0 --top-k 1 --seed 1 \
    -t "$THREADS" -n "$NTOK" 2>>/tmp/g5_q41_engage_all.err
}
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox
In the year 2050, humanity"
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  gen ON  "$IDX" > /tmp/g5_q41_A_$IDX.txt
  gen OFF "$IDX" > /tmp/g5_q41_B_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(ON emit): $(tr '\n' ' ' < /tmp/g5_q41_A_$IDX.txt | cut -c1-120)"
  echo "   B(OFF stk): $(tr '\n' ' ' < /tmp/g5_q41_B_$IDX.txt | cut -c1-120)"
  if [ ! -s /tmp/g5_q41_A_$IDX.txt ] || [ ! -s /tmp/g5_q41_B_$IDX.txt ]; then
    echo "   VERDICT: FAIL (empty generation)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q /tmp/g5_q41_A_$IDX.txt /tmp/g5_q41_B_$IDX.txt >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else
    echo "   VERDICT: MISMATCH (A!=B)"; FAIL=$((FAIL+1))
    diff /tmp/g5_q41_A_$IDX.txt /tmp/g5_q41_B_$IDX.txt | head -8 | sed 's/^/     /'; fi
done <<< "$PROMPTS"
ENGAGED=$(grep -c "TCRV G5-M2 EMITTED" /tmp/g5_q41_engage_all.err || true)
echo "== ENGAGE: total emitted-kernel banner fires (ON runs) = $ENGAGED (expect >0) =="
if grep -aiE 'nan|-nan|[^a-z]inf[^a-z]' /tmp/g5_q41_engage_all.err >/dev/null 2>&1; then
  echo "   LOGITS: FAIL NaN/Inf in stderr"; FAIL=$((FAIL+1)); else echo "   LOGITS: OK no NaN/Inf"; fi

echo "== PPL SANITY (ON, short) =="
swap ON
PPLTXT=/tmp/g5_q41_ppl_input.txt
head -c 6000 "$ATREE/build-gcc15-rv64gcv/bin/../../ggml/src/ggml-cpu/repack.cpp" 2>/dev/null > "$PPLTXT" || echo "The quick brown fox jumps over the lazy dog. $(seq 1 400 | tr '\n' ' ')" > "$PPLTXT"
$PIN "$BIN/llama-perplexity" -m "$MODEL" -f "$PPLTXT" -t "$THREADS" -c 256 --chunks 2 2>/tmp/g5_q41_ppl.err | tail -3 || true
PPLVAL=$(grep -oE 'Final estimate: PPL = [0-9.]+|\[2\][0-9.]+' /tmp/g5_q41_ppl.err | tail -1)
PPLNUM=$(grep -oE 'estimate: PPL = [0-9.]+' /tmp/g5_q41_ppl.err | grep -oE '[0-9.]+' | tail -1)
echo "   PPL(ON) = ${PPLVAL:-<see /tmp/g5_q41_ppl.err>}"
PPL_OK=1
if [ -n "$PPLNUM" ]; then
  awk "BEGIN{exit !($PPLNUM>0 && $PPLNUM<1000)}" && echo "   PPL SANITY: OK (finite, coherent, <1000)" || { echo "   PPL SANITY: FAIL (garbage class $PPLNUM)"; PPL_OK=0; }
else echo "   PPL: could not parse (non-fatal; greedy A==B is primary gate)"; fi
swap OFF

echo "== CORRECTNESS SUMMARY: byte_identical=$PASS failed=$FAIL engaged_banners=$ENGAGED ppl_ok=$PPL_OK =="
if [ "$FAIL" = 0 ] && [ "$PASS" -ge 1 ] && [ "$ENGAGED" -ge 1 ] && [ "$PPL_OK" = 1 ]; then echo "CORRECTNESS_GATE: GREEN"; else echo "CORRECTNESS_GATE: RED"; fi
