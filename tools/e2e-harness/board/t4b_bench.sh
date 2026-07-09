#!/bin/bash
# t4b_bench.sh -- run one llama-bench prefill sweep for variant A (tcrv q4_K-gate-ON,
# repack-GEMM) or Bstock (upstream-native, ggml block-dot). JSON output for raw samples.
# usage: t4b_bench.sh {A|Bstock} <pp_csv> <reps> <tag>
set -u
. /opt/tcrv-toolchains/env.sh 2>/dev/null
MODE="$1"; PP="$2"; REPS="$3"; TAG="$4"
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
TCRV=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
UP=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
if [ "$MODE" = "A" ]; then
  cp -f "$TCRV/libggml-cpu.so.0.15.1.A-q4kON" "$TCRV/libggml-cpu.so.0.15.1"
  BIN="$TCRV/llama-bench"; LD="$TCRV"
else
  BIN="$UP/llama-bench"; LD="$UP"
fi
export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
OUT=/tmp/t4b_${TAG}.json
ERR=/tmp/t4b_${TAG}.err
taskset -c 8-15 "$BIN" -m "$MODEL" -p "$PP" -n 0 -t 8 -r "$REPS" -o json > "$OUT" 2> "$ERR"
echo "BENCH_EXIT=$? mode=$MODE pp=$PP reps=$REPS tag=$TAG"
echo "--- q4_K path banner (A should show REPACK GEMM; Bstock none) ---"
grep -oE "TCRV Q4_K [A-Z- ]+" "$ERR" 2>/dev/null | sort | uniq -c
echo "--- json avg_ts ---"
grep -oE '"(n_prompt|avg_ts|avg_ns)"[^,]*' "$OUT" 2>/dev/null | paste - - - | head
