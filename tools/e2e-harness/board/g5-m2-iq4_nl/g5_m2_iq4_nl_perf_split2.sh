#!/usr/bin/env bash
# [G5-M2 iq4_nl] Robust paired A/B perf: prefill(pp128, the verdict metric) with r10 x2
# passes (n=20/side) + decode(tg32) with r4 x1 pass (slow emitted GEVM). Separate -n0 /
# -p0 llama-bench invocations so prefill timing is not contaminated by the slow decode.
# Same-tree physical .so swap (iq4ON emitted repack / iq4OFF stock block-dot). ###AB blocks
# are analyzable by analyze_phase_split.py. Fully detached-safe.
set -u
BIN=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_iq4nl
M=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-IQ4_NL.gguf
PIN="taskset -c 8-15"
source /opt/tcrv-toolchains/env.sh 2>/dev/null || true
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.iq4$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== ENV: $(uname -r) VLEN128 gcc-15.2.0 governor=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_governor 2>/dev/null) freq=$(freq) =="
echo "== model=$M sha256=$(sha256sum "$M"|cut -c1-16) =="
echo "== iq4ON md5=$(md5sum "$SCR/libggml-cpu.so.iq4ON"|cut -c1-16) syms=$(nm -C "$SCR/libggml-cpu.so.iq4ON"|grep -cE 'tcrv_emitc_ggml_repack_(gemv|gemm)_iq4_nl') =="
echo "== iq4OFF md5=$(md5sum "$SCR/libggml-cpu.so.iq4OFF"|cut -c1-16) syms=$(nm -C "$SCR/libggml-cpu.so.iq4OFF"|grep -cE 'tcrv_emitc_ggml_repack_(gemv|gemm)_iq4_nl') =="

echo "== ENGAGE PROBE (ON, banner) =="
swap ON
ENG=$($PIN "$BIN/llama-bench" -m "$M" -p 16 -n 4 -t 8 -r 1 2>&1 >/dev/null | grep -c 'TCRV G5-M2 EMITTED' || true)
echo "  emitted-kernel banner fires (ON p16 n4) = $ENG (expect >0)"

echo "== WARMUP (dropped) =="
swap ON;  $PIN "$BIN/llama-bench" -m "$M" -p 8 -n 2 -t 8 -r 1 >/dev/null 2>&1
swap OFF; $PIN "$BIN/llama-bench" -m "$M" -p 8 -n 2 -t 8 -r 1 >/dev/null 2>&1

pp_side(){ # $1 variant $2 side $3 pass
  swap "$1"; echo "###AB pass=$3 side=$2 freq_khz=$(freq)"
  $PIN "$BIN/llama-bench" -m "$M" -p 128 -n 0 -t 8 -r 10 -o json 2>/dev/null
  echo "###END"
}
tg_side(){ # $1 variant $2 side
  swap "$1"; echo "###AB pass=1 side=$2 freq_khz=$(freq)"
  $PIN "$BIN/llama-bench" -m "$M" -p 0 -n 32 -t 8 -r 4 -o json 2>/dev/null
  echo "###END"
}

echo "###PHASE=PREFILL"
for pass in 1 2; do
  echo "== PREFILL ABPASS $pass =="
  pp_side ON  ours  "$pass"
  pp_side OFF stock "$pass"
done

echo "###PHASE=DECODE"
echo "== DECODE ABPASS 1 (r4, slow emitted GEVM) =="
tg_side ON  ours
tg_side OFF stock

swap OFF
echo "== LIVE restored to OFF md5=$(md5sum "$LIVE"|cut -c1-16) =="
echo "== PERF DONE =="
