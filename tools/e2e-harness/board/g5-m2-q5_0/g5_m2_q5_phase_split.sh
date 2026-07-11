#!/usr/bin/env bash
# [G5 M2 q5_0] Same-tree physical-.so-swap paired A/B phase-split (prefill pp /
# decode tg) for q5_0. ours=q5ON (NET-NEW 12-piece scaffold + EMITTED vl=8 repack
# GEVM+GEMM, correctness GREEN byte-identical A==B, objdump vl=8 sealed), stock=q5OFF
# (pristine ggml q5_0 block-dot; NO scaffold). ONE llama-bench binary, ONE source
# tree, ONE compiler (gcc-15.2.0 symmetric) => the ONLY A/B difference is the q5_0
# net-new dispatch gate + emitted kernels (KERNEL-AXIS gcc-symmetric ledger).
# System ledger: board rv64gcv shipped toolchain == gcc-15 => kernel==system, no
# clang asymmetry (contrast [CASE-COMPILER-ASYMMETRY]). Interleaved ours/stock,
# PASSES passes, per-rep samples_ts in JSON, taskset pin, DVFS freq per call,
# banner-fire engage probe (deployment-during-e2e verify). Emits ###AB..###END.
set -u
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_q5
MODEL="${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q5_0.gguf}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-10}"; PASSES="${PASSES:-2}"
BENCH="$BIN/llama-bench"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q5$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc)  pinned_cores=$CORES  threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq 2>/dev/null || echo NA)"
echo "loadavg=$(cat /proc/loadavg)"
echo "model=$MODEL  model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "ab_mechanism=same-tree physical .so swap (q5ON=net-new-scaffold+emitted-vl8 vs q5OFF=pristine-generic); one llama-bench; gcc-15.2.0 symmetric"
echo "q5ON  md5=$(md5sum "$SCR/libggml-cpu.so.q5ON"|awk '{print $1}')"
echo "q5OFF md5=$(md5sum "$SCR/libggml-cpu.so.q5OFF"|awk '{print $1}')"
echo "-- deployment verify (symbols + banner) --"
echo "  ON  q5_0_tcrv_syms(expect 2)=$(nm -C "$SCR/libggml-cpu.so.q5ON" 2>/dev/null|grep -cE 'tcrv_emitc_ggml_(vec_dot|gemm)_q5_0')"
echo "  OFF q5_0_tcrv_syms(expect 0)=$(nm -C "$SCR/libggml-cpu.so.q5OFF" 2>/dev/null|grep -cE 'tcrv_emitc_ggml_(vec_dot|gemm)_q5_0')"
echo "  ON  banner gevm/gemm=$(strings "$SCR/libggml-cpu.so.q5ON"|grep -c 'TCRV G5-M2 EMITTED GEVM(q5_0_16x1 VLEN128')/$(strings "$SCR/libggml-cpu.so.q5ON"|grep -c 'TCRV G5-M2 EMITTED GEMM(q5_0_16x1 VLEN128')"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES"

# banner-fire engage probe: confirm the emitted q5_0 kernel actually executes on
# THIS model in a real prefill+decode run (not just microbench). Run ON with a tiny
# generation and count banner lines on stderr.
echo "== ENGAGE PROBE (ON, banner fires on real model run) =="
swap ON
ENG=$($PIN "$BENCH" -m "$MODEL" -p 16 -n 8 -t "$THREADS" -r 1 2>&1 >/dev/null | grep -c 'TCRV G5-M2 EMITTED' || true)
echo "  emitted-kernel banner fires (ON, p16 n8) = $ENG (expect > 0)"

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
swap OFF  # leave live at pristine OFF (A-tree restore)
echo "== LIVE after restore md5=$(md5sum "$LIVE"|awk '{print $1}') (expect==q5OFF) =="
echo "== DONE =="
