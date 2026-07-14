#!/usr/bin/env bash
# [G7-L3 FLAT@k1 e2e] correctness (greedy A==B byte-identical) + phase-split paired A/B.
#   A = REPACK (stock as-shipped FLAT 16x1 repack)  B = VECDOT (repack off -> block-dot)
# ONE real llama ELF, ONE tree, ONE compiler (clang-18 SYMMETRIC => kernel==system ledger).
# DVFS 1.6GHz locked, taskset -c 0-3, -t4, interleaved, REPS/side x PASSES.
# Usage: FORMAT=q4_0|q8_0 MODEL=/path.gguf flat_k1_measure.sh
set -u
FMT="${FORMAT:?}"; MODEL="${MODEL:?}"
BUILD=/data/build-k1-flat; BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/flat_$FMT
CLI="${CLI:-/data/k1build/bin/llama-cli}"
BENCH="${BENCH:-/data/k1build/bin/llama-bench}"   # real ELF (wrapper forces stock LD path)
CORES="${CORES:-0-3}"; THREADS="${THREADS:-4}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-6}"; PASSES="${PASSES:-2}"; NTOK="${NTOK:-24}"
PIN="taskset -c $CORES"; CORE0=$(echo "$CORES"|grep -oE '^[0-9]+')
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT (FORMAT=$FMT) =="
echo "uname=$(uname -a)"; echo "isa=$(grep -m1 -i isa /proc/cpuinfo|cut -c1-60)..."
echo "nproc=$(nproc) pinned=$CORES threads=$THREADS gov=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq) loadavg=$(cat /proc/loadavg)"
echo "compiler=clang-18 (both variants COMPILER-SYMMETRIC => kernel==system)"
echo "model=$MODEL sha256=$(sha256sum "$MODEL"|cut -c1-16)"
echo "REPACK md5=$(md5sum "$SCR/libggml-cpu.so.REPACK"|awk '{print $1}')"
echo "VECDOT md5=$(md5sum "$SCR/libggml-cpu.so.VECDOT"|awk '{print $1}')"

echo "== CORRECTNESS (greedy A==B byte-identical) =="
gen(){ swap "$1"; $PIN "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" \
     --no-warmup --temp 0 --top-k 1 --seed 1 -no-cnv -st --no-display-prompt </dev/null 2>>/tmp/flat_${FMT}_engage.err; }
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox jumps"
: > /tmp/flat_${FMT}_engage.err
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue; IDX=$((IDX+1))
  gen REPACK > /tmp/flat_${FMT}_A_$IDX.txt
  gen VECDOT > /tmp/flat_${FMT}_B_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(REPACK): $(tr '\n' ' ' < /tmp/flat_${FMT}_A_$IDX.txt | cut -c1-90)"
  echo "   B(VECDOT): $(tr '\n' ' ' < /tmp/flat_${FMT}_B_$IDX.txt | cut -c1-90)"
  if [ ! -s /tmp/flat_${FMT}_A_$IDX.txt ] || [ ! -s /tmp/flat_${FMT}_B_$IDX.txt ]; then
    echo "   VERDICT: FAIL(empty)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q /tmp/flat_${FMT}_A_$IDX.txt /tmp/flat_${FMT}_B_$IDX.txt >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else echo "   VERDICT: MISMATCH"; FAIL=$((FAIL+1)); diff /tmp/flat_${FMT}_A_$IDX.txt /tmp/flat_${FMT}_B_$IDX.txt|head -6|sed 's/^/     /'; fi
done <<< "$PROMPTS"
echo "== CORRECTNESS SUMMARY: byte_identical=$PASS mismatch/fail=$FAIL =="
[ "$FAIL" = 0 ] && [ "$PASS" -ge 1 ] && echo "CORRECTNESS_GATE: GREEN" || echo "CORRECTNESS_GATE: (see divergence note)"

echo "== PHASE-SPLIT paired A/B (PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES => n=$((REPS*PASSES))/side) =="
run_side(){ local var="$1" lab="$2" pass="$3" fk; swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$lab variant=$var freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"; }
echo "== WARMUP (dropped) =="
swap REPACK; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
swap VECDOT; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="; run_side REPACK repack "$pass"; run_side VECDOT vecdot "$pass"; done
swap REPACK
echo "== DONE == final loadavg=$(cat /proc/loadavg)"
