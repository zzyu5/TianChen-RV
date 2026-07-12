#!/usr/bin/env bash
# [G5-M2 q2_K @ k1] Same-tree physical-.so-swap paired A/B phase-split (prefill pp / decode tg)
# for q2_K on k1 (SpacemiT X60 VLEN256, stock clang-18).
#   A = ON  (OUR emitted VLA q2_K repack: S6-tiled GEMM + plain GEVM, replacing stock)
#   B = OFF (STOCK hand-tuned RVV q2_K 16x1 repack ggml_gemm/gemv_q2_K_16x1_q8_K, case256)
# ONE llama-bench ELF, ONE source tree, ONE compiler (clang-18 SYMMETRIC => kernel-acct ==
# system-acct, dual-ledger same number). The ONLY A/B difference is the q2_K compute (our-emit vs stock hand-brick; q4_K Win-K1-VLEN contest).
# DVFS locked, taskset pinned, interleaved, PASSES passes, REPS/side, JSON per rep.
# ratio = ON / OFF ; >=1.0 => our emitted repack >=parity the stock hand-brick e2e (q4_K precedent).
set -u
BUILD=/data/build-k1-q2k
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
WK=/tmp/g5_q2k
MODEL="${MODEL:-/data/tinyllama-1.1b-Q2_K_M.gguf}"
CORES="${CORES:-0-7}"; THREADS="${THREADS:-8}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-12}"; PASSES="${PASSES:-2}"
BENCH="${BENCH:-/data/k1build/bin/llama-bench}"   # real ELF
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
echo "compiler=clang-18 (both variants; COMPILER-SYMMETRIC => kernel==system ledger)"
echo "model=$MODEL  sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "ON  md5=$(md5sum "$WK/libggml-cpu.so.ON"|awk '{print $1}')"
echo "OFF md5=$(md5sum "$WK/libggml-cpu.so.OFF"|awk '{print $1}')"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES  A=ON(our-emit) B=OFF(stock-hand-brick)"

run_side(){ # $1=variant $2=label $3=pass
  local var="$1" lab="$2" pass="$3" fk
  swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$lab variant=$var freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}
echo "== WARMUP (dropped) =="
swap ON;  $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
swap OFF; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="
  run_side ON  on  "$pass"
  run_side OFF off "$pass"
done
swap OFF   # leave live at pristine stock hand-brick repack
echo "== DONE == final loadavg=$(cat /proc/loadavg)"
