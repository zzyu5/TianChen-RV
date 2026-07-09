#!/bin/bash
# t4b_ppl.sh -- perplexity (pure-GEMM prefill correctness check) for variant A or Bstock.
# usage: t4b_ppl.sh {A|Bstock} <tag> [ctx]
set -u
. /opt/tcrv-toolchains/env.sh 2>/dev/null
MODE="$1"; TAG="$2"; CTX="${3:-256}"
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
CORPUS=/tmp/t4b_corpus.txt
TCRV=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
UP=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
if [ "$MODE" = "A" ]; then
  cp -f "$TCRV/libggml-cpu.so.0.15.1.A-q4kON" "$TCRV/libggml-cpu.so.0.15.1"
  BIN="$TCRV/llama-perplexity"; LD="$TCRV"
else
  BIN="$UP/llama-perplexity"; LD="$UP"
fi
export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
OUT=/tmp/ppl_${TAG}.out; ERR=/tmp/ppl_${TAG}.err
taskset -c 8-15 "$BIN" -m "$MODEL" -f "$CORPUS" -c "$CTX" -t 8 > "$OUT" 2> "$ERR"
echo "PPL_EXIT=$? mode=$MODE tag=$TAG ctx=$CTX"
echo "=== ppl result ($TAG) ==="
grep -iE "Final estimate|PPL|perplexity|nan" "$OUT" "$ERR" 2>/dev/null | tail -6
echo "=== q4_K route ($TAG) ==="
grep -oE "TCRV Q4_K [A-Z- ]+" "$ERR" 2>/dev/null | sort | uniq -c
