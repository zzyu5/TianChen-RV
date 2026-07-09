#!/bin/bash
# t4b_ppl_so.sh -- run perplexity with a SPECIFIC saved libggml-cpu .so copy (tcrv tree).
# usage: t4b_ppl_so.sh <so_suffix e.g. Agen|A-q4kON|B-q4kOFF> <tag> [ctx]
. /opt/tcrv-toolchains/env.sh 2>/dev/null
SO="$1"; TAG="$2"; CTX="${3:-128}"
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
BIN=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
cp -f "$BIN/libggml-cpu.so.0.15.1.$SO" "$BIN/libggml-cpu.so.0.15.1"
export LD_LIBRARY_PATH="$BIN:$LD_LIBRARY_PATH"
OUT=/tmp/pplso_${TAG}.out; ERR=/tmp/pplso_${TAG}.err
: > "$OUT"; : > "$ERR"
taskset -c 8-15 "$BIN/llama-perplexity" -m "$MODEL" -f /tmp/t4b_corpus.txt -c "$CTX" -t 8 > "$OUT" 2> "$ERR"
echo "PPLSO_EXIT=$? so=$SO tag=$TAG" >> "$OUT"
grep -iE "Final estimate|PPL =|nan" "$ERR" >> "$OUT" 2>/dev/null
echo "PPLSO_DONE_$TAG" >> "$OUT"
