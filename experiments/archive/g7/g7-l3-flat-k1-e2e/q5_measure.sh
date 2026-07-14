#!/usr/bin/env bash
# [G7-L3 q5_x@k1] correctness (greedy A==B) + phase-split paired A/B e2e.
#   A = ON  (our net-new VLEN256 vl=16 emitted repack)
#   B = OFF (stock block-dot -- q5_x has NO stock riscv repack)
# ONE real llama ELF, ONE tree, ONE compiler (clang-18 SYMMETRIC => kernel==system).
# DVFS performance, taskset -c 0-3, -t4, interleaved, REPS/side x PASSES => n=12/side.
# Usage: q5_measure.sh q5_0|q5_1
set -u
FMT="${1:?usage: q5_measure.sh q5_0|q5_1}"
case "$FMT" in
  q5_0) N=50 ;;
  q5_1) N=51 ;;
  *) echo "unknown FMT $FMT"; exit 2 ;;
esac
BASE=/data/g7q$N
MODEL="${MODEL:-$BASE/tinyllama-$FMT.gguf}"
BUILD=/data/build-k1-$FMT; BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=$BASE/scratch
CLI=/data/k1build/bin/llama-cli
BENCH=/data/k1build/bin/llama-bench
CORES="${CORES:-0-3}"; THREADS=4; CORE0=$(echo $CORES|cut -d- -f1)
PP=128; TG=32; REPS=6; PASSES=2; NTOK=24
PIN="taskset -c $CORES"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT ($FMT@k1) =="
echo "uname=$(uname -a)"; echo "isa=$(grep -m1 -i isa /proc/cpuinfo|cut -c1-60)..."
echo "nproc=$(nproc) pinned=$CORES threads=$THREADS gov=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor)"
echo "compiler=clang-18 (both variants COMPILER-SYMMETRIC => kernel==system)"
echo "model=$MODEL sha256=$(sha256sum "$MODEL"|cut -c1-16)"
echo "OFF md5=$(md5sum "$SCR/libggml-cpu.so.OFF"|awk '{print $1}')  ON md5=$(md5sum "$SCR/libggml-cpu.so.ON"|awk '{print $1}')"

echo "== LOAD-GATE (verify NO CPU-bound competitor on pinned cores $CORES) =="
BUSY=$(for c in $(seq ${CORES%-*} ${CORES#*-}); do ps -eLo psr,pcpu --no-headers | awk -v c=$c '$1==c && $2>20{print}'; done | wc -l)
echo "  loadavg=$(cat /proc/loadavg)  CPU-bound(>20%) threads on cores $CORES = $BUSY"
[ "$BUSY" -eq 0 ] && echo "  GATE OPEN (pinned cores clear)" || echo "  WARN: competitor on pinned cores (interleaved ratio still robust)"

echo "== CORRECTNESS (greedy A==B byte-identical) =="
gen(){ swap "$1"; $PIN "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" \
     --no-warmup --temp 0 --top-k 1 --seed 1 -no-cnv -st --no-display-prompt </dev/null 2>>"$BASE/engage.err"; }
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox jumps"
: > "$BASE/engage.err"
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue; IDX=$((IDX+1))
  gen ON  > "$BASE/A_$IDX.txt"
  gen OFF > "$BASE/B_$IDX.txt"
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(ON) : $(tr '\n' ' ' < "$BASE/A_$IDX.txt" | cut -c1-90)"
  echo "   B(OFF): $(tr '\n' ' ' < "$BASE/B_$IDX.txt" | cut -c1-90)"
  if [ ! -s "$BASE/A_$IDX.txt" ] || [ ! -s "$BASE/B_$IDX.txt" ]; then
    echo "   VERDICT: FAIL(empty)"; FAIL=$((FAIL+1)); continue; fi
  if diff -q "$BASE/A_$IDX.txt" "$BASE/B_$IDX.txt" >/dev/null; then
    echo "   VERDICT: BYTE-IDENTICAL (A==B)"; PASS=$((PASS+1))
  else echo "   VERDICT: MISMATCH"; FAIL=$((FAIL+1)); diff "$BASE/A_$IDX.txt" "$BASE/B_$IDX.txt"|head -6|sed 's/^/     /'; fi
done <<< "$PROMPTS"
echo "== CORRECTNESS SUMMARY: byte_identical=$PASS mismatch/fail=$FAIL =="
echo "== ENGAGE banner fires (proof our repack ran during e2e) =="
grep -c "TCRV G7-L3 EMITTED" "$BASE/engage.err"

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
