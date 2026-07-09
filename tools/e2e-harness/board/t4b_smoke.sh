#!/bin/bash
# t4b_smoke.sh -- greedy coherent-generation smoke test for one libggml-cpu.so variant.
# Arg1 = tag (e.g. B-q4kOFF / A-q4kON). Uses the .so already in place in bin/.
. /opt/tcrv-toolchains/env.sh 2>/dev/null
cd /home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"
TAG="${1:-X}"
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
OUT=/tmp/cli_$TAG.out
ERR=/tmp/cli_$TAG.err
: > "$OUT"; : > "$ERR"
taskset -c 8-15 ./llama-cli -m "$MODEL" \
  -p "The capital of France is" -n 24 --temp 0 --seed 42 -t 8 -no-cnv \
  > "$OUT" 2> "$ERR"
echo "CLI_EXIT=$?" >> "$OUT"
echo "DONE_$TAG" >> "$OUT"
