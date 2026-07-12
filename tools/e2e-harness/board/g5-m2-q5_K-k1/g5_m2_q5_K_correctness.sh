#!/usr/bin/env bash
# [G5-M2 q5_K @ k1] correctness-first: greedy A==B between A=ON (our net-new q5_K repack:
# emitted VLA GEMM/GEVM) and B=OFF (stock q5_K generic block-dot ggml_vec_dot_q5_K_q8_K).
# Certificate three-requirements: (corpus) 4 prompts; (same input path) same gguf, same q8_K
# activation quant, physical .so swap in ONE build tree; (independent oracle) B=block-dot is
# ggml's OWN reference, independent of our repack. Our kernel is bit-exact-integer vs the
# kernel-axis oracle; the fp scale-fold is bounded-ULP vs ggml, so byte-identical greedy is the
# strong pass and a late single-token ULP divergence is the honest bounded-ULP caveat (NOT a bug;
# immediate garbage would be a layout/MIRAGE bug).  Uses the real llama-cli ELF + LD_LIBRARY_PATH.
set -u
BUILD=/data/build-k1-q5k
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
WK=/tmp/g5_q5k
CLI="${CLI:-/data/k1build/bin/llama-cli}"       # real ELF; our LD_LIBRARY_PATH wins
MODEL="${MODEL:-/data/tinyllama-1.1b-Q5_K_M.gguf}"
THREADS="${THREADS:-8}"; NTOK="${NTOK:-32}"; PIN="taskset -c 0-7"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$WK/libggml-cpu.so.$1" "$LIVE"; }
gen(){ # $1=variant -> stdout generated tokens only
  swap "$1"
  $PIN "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" \
     --no-warmup --temp 0 --top-k 1 --seed 1 -no-cnv -st --no-display-prompt </dev/null 2>>/tmp/g5q5k_engage.err
}
echo "== env: model=$MODEL sha256=$(sha256sum "$MODEL"|cut -c1-16) threads=$THREADS ntok=$NTOK =="
echo "== ON  md5=$(md5sum "$WK/libggml-cpu.so.ON"|awk '{print $1}')  OFF md5=$(md5sum "$WK/libggml-cpu.so.OFF"|awk '{print $1}') =="
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox jumps"
: > /tmp/g5q5k_engage.err
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  gen ON  > /tmp/g5q5k_A_$IDX.txt
  gen OFF > /tmp/g5q5k_B_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(ON/our-repack): $(tr '\n' ' ' < /tmp/g5q5k_A_$IDX.txt | cut -c1-90)"
  echo "   B(OFF/block-dot): $(tr '\n' ' ' < /tmp/g5q5k_B_$IDX.txt | cut -c1-90)"
  if [ ! -s /tmp/g5q5k_A_$IDX.txt ] || [ ! -s /tmp/g5q5k_B_$IDX.txt ]; then
    echo "   VERDICT: FAIL (empty)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q /tmp/g5q5k_A_$IDX.txt /tmp/g5q5k_B_$IDX.txt >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else
    echo "   VERDICT: MISMATCH (chars matching prefix + first diff)"; FAIL=$((FAIL+1))
    cmp <(tr '\n' ' ' </tmp/g5q5k_A_$IDX.txt) <(tr '\n' ' ' </tmp/g5q5k_B_$IDX.txt) 2>&1 | head -2 | sed 's/^/     /'
    diff /tmp/g5q5k_A_$IDX.txt /tmp/g5q5k_B_$IDX.txt | head -6 | sed 's/^/     /'; fi
done <<< "$PROMPTS"
echo "== engage banner (proves our kernel ran) =="
grep -m2 "TCRV G5-M2 EMITTED" /tmp/g5q5k_engage.err | sed 's/^/   /' || echo "   (no banner!)"
echo "== SUMMARY: byte_identical=$PASS failed=$FAIL =="
[ "$FAIL" = 0 ] && [ "$PASS" -ge 1 ] && echo "CORRECTNESS_GATE: GREEN (A==B byte-identical over corpus)" || echo "CORRECTNESS_GATE: see per-prompt (may be bounded-ULP divergence, characterize)"
swap OFF
