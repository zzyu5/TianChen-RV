#!/usr/bin/env bash
# [G7-L3 q4_1@k1] correctness (greedy A==B) + phase-split paired A/B e2e.
#   A = ON  (our net-new VLEN256 vl=16 emitted repack)
#   B = OFF (stock block-dot, ggml_vec_dot_q4_1_q8_1 -- q4_1 has NO stock repack)
# ONE real llama ELF, ONE tree, ONE compiler (clang-18 SYMMETRIC => kernel==system).
# DVFS performance, taskset -c 0-3, -t4, interleaved, REPS/side x PASSES => n=12/side.
set -u
MODEL="${MODEL:-/data/g7q41/tinyllama-q4_1.gguf}"
BUILD=/data/build-k1-q41; BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/data/g7q41/scratch
CLI=/data/k1build/bin/llama-cli
BENCH=/data/k1build/bin/llama-bench
# NOTE: a FOREIGN q5_K llama-cli is hung 90min at 100% on core 1 (another line;
# not killed). Cores 4-7 are free -> pin there (homogeneous X60; ratio is
# cluster-independent). Documented deviation from nominal pin 0-3.
CORES="4-7"; THREADS=4; CORE0=4
PP=128; TG=32; REPS=6; PASSES=2; NTOK=24
PIN="taskset -c $CORES"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT (q4_1@k1) =="
echo "uname=$(uname -a)"; echo "isa=$(grep -m1 -i isa /proc/cpuinfo|cut -c1-60)..."
echo "nproc=$(nproc) pinned=$CORES threads=$THREADS gov=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor)"
echo "compiler=clang-18 (both variants COMPILER-SYMMETRIC => kernel==system)"
echo "model=$MODEL sha256=$(sha256sum "$MODEL"|cut -c1-16)"
echo "OFF md5=$(md5sum "$SCR/libggml-cpu.so.OFF"|awk '{print $1}')  ON md5=$(md5sum "$SCR/libggml-cpu.so.ON"|awk '{print $1}')"

echo "== LOAD-GATE (verify NO CPU-bound competitor on pinned cores $CORES) =="
# loadavg is inflated by the foreign core-1 hog; gate on the pinned cores instead.
BUSY=$(for c in 4 5 6 7; do ps -eLo psr,pcpu --no-headers | awk -v c=$c '$1==c && $2>20{print}'; done | wc -l)
echo "  loadavg=$(cat /proc/loadavg)  CPU-bound(>20%) threads on cores $CORES = $BUSY"
[ "$BUSY" -eq 0 ] && echo "  GATE OPEN (pinned cores clear)" || echo "  WARN: competitor on pinned cores (interleaved ratio still robust)"

echo "== CORRECTNESS (greedy A==B byte-identical) =="
gen(){ swap "$1"; $PIN "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" \
     --no-warmup --temp 0 --top-k 1 --seed 1 -no-cnv -st --no-display-prompt </dev/null 2>>/data/g7q41/engage.err; }
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox jumps"
: > /data/g7q41/engage.err
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue; IDX=$((IDX+1))
  gen ON  > /data/g7q41/A_$IDX.txt
  gen OFF > /data/g7q41/B_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(ON) : $(tr '\n' ' ' < /data/g7q41/A_$IDX.txt | cut -c1-90)"
  echo "   B(OFF): $(tr '\n' ' ' < /data/g7q41/B_$IDX.txt | cut -c1-90)"
  if [ ! -s /data/g7q41/A_$IDX.txt ] || [ ! -s /data/g7q41/B_$IDX.txt ]; then
    echo "   VERDICT: FAIL(empty)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q /data/g7q41/A_$IDX.txt /data/g7q41/B_$IDX.txt >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else echo "   VERDICT: MISMATCH"; FAIL=$((FAIL+1)); diff /data/g7q41/A_$IDX.txt /data/g7q41/B_$IDX.txt|head -6|sed 's/^/     /'; fi
done <<< "$PROMPTS"
echo "== CORRECTNESS SUMMARY: byte_identical=$PASS mismatch/fail=$FAIL =="
echo "== ENGAGE banner fires (proof our repack ran during e2e) =="
grep -c "TCRV G7-L3 EMITTED" /data/g7q41/engage.err

echo "== PHASE-SPLIT paired A/B (PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES => n=$((REPS*PASSES))/side) =="
run_side(){ local var="$1" lab="$2" pass="$3" fk; swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$lab variant=$var freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"; }
echo "== WARMUP (dropped) =="
swap ON;  $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
swap OFF; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="; run_side ON on "$pass"; run_side OFF off "$pass"; done
swap OFF
echo "== DONE == final loadavg=$(cat /proc/loadavg)"
