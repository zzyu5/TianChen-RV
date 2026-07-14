#!/usr/bin/env bash
# [decisive-kquant-gcc-vs-vlen] Correctness gate: greedy byte-identical
# A(NUMER=q4kON.clangrepack emitted repack) == B(DENOM=q4kOFF.gcc block-dot)
# + emitted-kernel engage banner + NaN sanity. Confirms the clang-recompiled
# emitted kernel is numerically correct vs the block-dot reference.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BIN=$ATREE/build-gcc15-rv64gcv/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/dkgv
NUMER="${NUMER:-q4kON.clangrepack}"; DENOM="${DENOM:-q4kOFF.gcc}"
MODEL="${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf}"
CORES=8-15; THREADS=8; NTOK="${NTOK:-24}"; PIN="taskset -c $CORES"
COMP="$BIN/llama-completion"
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
cp -f "$LIVE" "$SCR/LIVE.beforeCORR.bak"
swap(){ cp -f "$SCR/libggml-cpu.so.$1" "$LIVE"; }
echo "== CORRECTNESS A=$NUMER B=$DENOM model=$(basename $MODEL) ntok=$NTOK =="
: > "$SCR/corr.err"
gen(){ swap "$1"; printf '%s' "$PROMPT" | timeout 400 $PIN "$COMP" -m "$MODEL" --no-display-prompt --no-warmup --temp 0 --top-k 1 --seed 1 -t "$THREADS" -n "$NTOK" 2>>"$SCR/corr.err"; }
PROMPTS="The capital of France is
Q: What is 2 + 2? A:"
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue; IDX=$((IDX+1))
  gen "$NUMER" > "$SCR/corrA_$IDX.txt"; gen "$DENOM" > "$SCR/corrB_$IDX.txt"
  echo "-- [$IDX] '$PROMPT'"
  echo "   A(repack): $(tr '\n' ' ' < "$SCR/corrA_$IDX.txt" | cut -c1-90)"
  echo "   B(blkdot): $(tr '\n' ' ' < "$SCR/corrB_$IDX.txt" | cut -c1-90)"
  if [ ! -s "$SCR/corrA_$IDX.txt" ] || [ ! -s "$SCR/corrB_$IDX.txt" ]; then echo "   FAIL(empty)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q "$SCR/corrA_$IDX.txt" "$SCR/corrB_$IDX.txt" >/dev/null; then echo "   A==B byte-identical"; PASS=$((PASS+1)); else echo "   A!=B"; FAIL=$((FAIL+1)); diff "$SCR/corrA_$IDX.txt" "$SCR/corrB_$IDX.txt"|head -6|sed 's/^/     /'; fi
done <<< "$PROMPTS"
ENG=$(grep -c "TCRV G5-M2 EMITTED" "$SCR/corr.err" || true)
NAN=$(grep -aiwE 'nan|-nan|inf' "$SCR/corr.err" | wc -l || true)
echo "== engage=$ENG(>0) nan=$NAN(0) pass=$PASS fail=$FAIL =="
cp -f "$SCR/LIVE.beforeCORR.bak" "$LIVE"
{ [ "$FAIL" = 0 ] && [ "$PASS" -ge 1 ] && [ "$ENG" -ge 1 ]; } && echo "CORRECTNESS_GATE: GREEN" || echo "CORRECTNESS_GATE: RED"
