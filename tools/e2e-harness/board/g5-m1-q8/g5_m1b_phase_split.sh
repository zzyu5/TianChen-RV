#!/usr/bin/env bash
# [G5 M1b] Same-tree physical-.so-swap paired A/B phase-split (prefill pp / decode
# tg) for q8_0. ours=q8ON (EMITTED vl=8 repack GEVM+GEMM, correctness GREEN),
# stock=q8OFF (block-dot vec_dot). One llama-bench binary, one source tree, one
# compiler (gcc-15.2.0) => the ONLY A/B difference is the q8_0 gate + emitted
# kernels. Interleaved ours/stock, PASSES passes, per-rep samples in JSON,
# taskset pin, DVFS freq captured per call. Emits ###AB..###END blocks.
set -u
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_q8b
MODEL="${MODEL:-$ATREE/models/tinyllama-q8_0.gguf}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-10}"; PASSES="${PASSES:-2}"
BENCH="$BIN/llama-bench"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q8$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc)  pinned_cores=$CORES  threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq 2>/dev/null || echo NA)"
echo "model=$MODEL  model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "ab_mechanism=same-tree physical .so swap (q8ON=emitted-vl8 vs q8OFF=block-dot); one llama-bench; gcc-15.2.0 symmetric"
echo "q8ON  md5=$(md5sum "$SCR/libggml-cpu.so.q8ON"|awk '{print $1}')"
echo "q8OFF md5=$(md5sum "$SCR/libggml-cpu.so.q8OFF"|awk '{print $1}')"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES"

run_side(){ # $1=variant(ON|OFF) $2=side $3=pass
  local var="$1" side="$2" pass="$3" fk
  swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$side freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}

echo "== WARMUP (dropped) =="
swap ON;  $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
swap OFF; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1

for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="
  run_side ON  ours  "$pass"
  run_side OFF stock "$pass"
done
swap OFF  # leave live at pristine OFF
echo "== DONE =="
