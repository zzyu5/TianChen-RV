#!/bin/bash
# t4b_seal.sh -- greedy coherent-generation seal for variant A (tcrv q4_K repack-GEMM)
# or Bstock (upstream ggml block-dot). Deterministic greedy (--temp 0), fixed prompt.
# usage: t4b_seal.sh {A|Bstock} <tag> [npredict]
set -u
. /opt/tcrv-toolchains/env.sh 2>/dev/null
MODE="$1"; TAG="$2"; NP="${3:-48}"
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
TCRV=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
UP=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
if [ "$MODE" = "A" ]; then
  cp -f "$TCRV/libggml-cpu.so.0.15.1.A-q4kON" "$TCRV/libggml-cpu.so.0.15.1"
  BIN="$TCRV/llama-completion"; LD="$TCRV"
else
  BIN="$UP/llama-completion"; LD="$UP"
fi
export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
OUT=/tmp/seal_${TAG}.out; ERR=/tmp/seal_${TAG}.err
PROMPT="Question: What is the capital of France? Answer:"
taskset -c 8-15 "$BIN" -m "$MODEL" -p "$PROMPT" -n "$NP" --temp 0 --seed 42 -t 8 \
  --no-warmup > "$OUT" 2> "$ERR"
echo "SEAL_EXIT=$? mode=$MODE tag=$TAG"
echo "=== generated ($TAG) ==="
cat "$OUT"
echo ""
echo "=== q4_K route evidence ($TAG) ==="
grep -oE "TCRV Q4_K [A-Z- ]+" "$ERR" 2>/dev/null | sort | uniq -c
