#!/usr/bin/env bash
# [WORK-ITEM-K1-KQUANT-E2E] Same-tree physical-.so-swap paired A/B phase-split
# (prefill pp / decode tg) for q4_K on k1 (SpacemiT X60 VLEN256, stock clang-18).
#   A = REPACK (stock as-shipped q4_K -> ggml_gemm/gemv_q4_K_16x1_q8_K repack)
#   B = VECDOT (q4_K repack disabled -> generic block-dot ggml_vec_dot_q4_K_q8_K)
# ONE llama-bench binary, ONE source tree, ONE compiler (clang-18 SYMMETRIC) =>
# the ONLY A/B difference is the q4_K dispatch (1-line). DVFS locked (perf gov 1.6GHz),
# taskset pinned, interleaved, PASSES passes, REPS/side, JSON per rep.
# ratio = REPACK / VECDOT ; >=1.0 => repack transduces on clang (gcc-death was rvv-specific).
set -u
BUILD=/data/build-k1-workitem
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
WK=/data/wk
MODEL="${MODEL:-/data/tinyllama-1.1b-Q4_K_M.gguf}"
CORES="${CORES:-0-7}"; THREADS="${THREADS:-8}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-10}"; PASSES="${PASSES:-2}"
# CRITICAL: $BIN/llama-bench is a WRAPPER SCRIPT that forces LD_LIBRARY_PATH=/data/k1build-stock/bin
# and execs /data/k1build/bin/llama-bench -> it IGNORES our swap (loads stock lib). MUST use the
# real ELF directly so LD_LIBRARY_PATH (below) resolves our swapped libggml-cpu.
BENCH="${BENCH:-/data/k1build/bin/llama-bench}"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$WK/libggml-cpu.so.$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "vlenb=$(cat /proc/cpuinfo | grep -m1 -i vlenb || echo NA)  nproc=$(nproc)  pinned=$CORES threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor 2>/dev/null)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq 2>/dev/null)"
echo "compiler=clang-18 (both variants; COMPILER-SYMMETRIC)"
echo "model=$MODEL  sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "REPACK md5=$(md5sum "$WK/libggml-cpu.so.REPACK"|awk '{print $1}')"
echo "VECDOT md5=$(md5sum "$WK/libggml-cpu.so.VECDOT"|awk '{print $1}')"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES  A=REPACK B=VECDOT"

run_side(){ # $1=variant $2=label $3=pass
  local var="$1" lab="$2" pass="$3" fk
  swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$lab variant=$var freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}
echo "== WARMUP (dropped) =="
swap REPACK; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
swap VECDOT; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="
  run_side REPACK repack "$pass"
  run_side VECDOT vecdot "$pass"
done
swap REPACK   # leave live at pristine stock
echo "== DONE =="
