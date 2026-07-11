#!/usr/bin/env bash
# [G5 M2 q4_K] Single-instance (flock) clean 8B prefill+short-decode A/B measurement.
# Writes RAW llama-bench json to per-run files (no python-in-pipe => no contention
# corruption). Interleaved ours(q4kON)/stock(q4kOFF), 3 passes. Parse offline.
exec 9>/tmp/g5_q4k/.measure_8b.lock
if ! flock -n 9; then echo "ALREADY-RUNNING (flock held) -- abort duplicate"; exit 99; fi
set -u
ATREE=/home/ubuntu/tcrv-llamacpp; BIN=$ATREE/build-gcc15-rv64gcv/bin
LIVE=$BIN/libggml-cpu.so.0.15.1; SCR=/tmp/g5_q4k
M=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
OUT=/tmp/g5_q4k/m8b; mkdir -p "$OUT"; rm -f "$OUT"/*.json
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q4k$1" "$LIVE"; }
PP="${PP:-128}"; TG="${TG:-16}"; REPS="${REPS:-6}"; PASSES="${PASSES:-3}"
echo "8B measure: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES cores=8-15 gov=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_governor 2>/dev/null)"
swap ON; taskset -c 8-15 "$BIN/llama-bench" -m "$M" -p 16 -n 4 -t 8 -r 1 >/dev/null 2>&1  # warmup
for p in $(seq 1 "$PASSES"); do
  swap ON;  taskset -c 8-15 "$BIN/llama-bench" -m "$M" -p "$PP" -n "$TG" -t 8 -r "$REPS" -o json > "$OUT/ours_p$p.json"  2>/dev/null
  swap OFF; taskset -c 8-15 "$BIN/llama-bench" -m "$M" -p "$PP" -n "$TG" -t 8 -r "$REPS" -o json > "$OUT/stock_p$p.json" 2>/dev/null
  echo "  pass $p done: ours=$(wc -c <"$OUT/ours_p$p.json")B stock=$(wc -c <"$OUT/stock_p$p.json")B"
done
swap OFF
echo "8B-MEASURE-DONE"
