#!/usr/bin/env bash
# [G6-B Phase-2 q2_K @ k1] THREE-way paired phase-split (prefill pp / decode tg), physical .so swap
# in ONE build tree / ONE llama-bench ELF / ONE compiler (clang-18 SYMMETRIC => kernel-acct==system-acct):
#   R = ON_ROLLED   (OUR emitted rolled-loop q2_K GEMM)   <- Phase-2 headline
#   U = ON_UNROLLED (OUR emitted full-unroll q2_K GEMM)   <- reproduces g5 baseline (.873x control)
#   S = OFF         (STOCK hand-tuned RVV q2_K 16x1 repack, case256) <- real opponent / hand-brick
# Interleaved R,U,S per pass; ratios rolled/stock, unrolled/stock, rolled/unrolled.
# Matches g5 measured config (4-thread pin, PP=64 TG=32 REPS=6 PASSES=2 => 12 samples/side) for
# apples-to-apples vs the 0.873x unrolled baseline.
set -u
BUILD=/data/build-k1-q2k-rolled
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
WK=/tmp/g6b-p2-q2k
MODEL="${MODEL:-/data/tinyllama-1.1b-Q2_K_M.gguf}"
CORES="${CORES:-0-3}"; THREADS="${THREADS:-4}"
PP="${PP:-64}"; TG="${TG:-32}"; REPS="${REPS:-6}"; PASSES="${PASSES:-2}"
BENCH="${BENCH:-/data/k1build/bin/llama-bench}"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$WK/libggml-cpu.so.$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "vlenb=$(grep -m1 -i vlenb /proc/cpuinfo || echo NA)  nproc=$(nproc)  pinned=$CORES threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor 2>/dev/null)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq 2>/dev/null)"
echo "loadavg=$(cat /proc/loadavg)"
echo "compiler=clang-18 (all variants; COMPILER-SYMMETRIC => kernel==system ledger)"
echo "model=$MODEL  sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "ROLLED   md5=$(md5sum "$WK/libggml-cpu.so.ON_ROLLED"|awk '{print $1}')"
echo "UNROLLED md5=$(md5sum "$WK/libggml-cpu.so.ON_UNROLLED"|awk '{print $1}')"
echo "OFF      md5=$(md5sum "$WK/libggml-cpu.so.OFF"|awk '{print $1}')"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES  R=rolled U=unrolled S=stock"

run_side(){ # $1=variant-file $2=label $3=pass
  local var="$1" lab="$2" pass="$3" fk
  swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$lab variant=$var freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}
echo "== WARMUP (dropped) =="
for v in ON_ROLLED ON_UNROLLED OFF; do swap $v; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1; done
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="
  run_side ON_ROLLED   rolled   "$pass"
  run_side ON_UNROLLED unrolled "$pass"
  run_side OFF         stock    "$pass"
done
swap OFF   # leave live at pristine stock hand-brick repack
echo "== DONE == final loadavg=$(cat /proc/loadavg)"
