#!/usr/bin/env bash
# [G5-M2 iq4_nl] Retry-robust paired perf under a SHARED/OVERLOADED board (load~22, other
# users' VLLM+llama jobs cause intermittent mmap SIGBUS in llama-bench hitting BOTH ON and
# OFF). Smaller batches (pp64 / tg16) + up to MAXTRY retries/side until a run yields a
# parseable avg_ts. Records every attempt (ok/BUS). Same-tree .so swap, sequential.
set -u
BIN=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_iq4nl
M=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-IQ4_NL.gguf
PIN="taskset -c 8-15"
MAXTRY="${MAXTRY:-8}"
source /opt/tcrv-toolchains/env.sh 2>/dev/null || true
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.iq4$1" "$LIVE"; sync; }

echo "== retry perf: load=$(cat /proc/loadavg) freq=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq) =="

# measure_side <variant> <label> <ppN> <tgN>  -> emits "RESULT <label> pp=<ts> tg=<ts>"
measure(){ local var="$1" lbl="$2" pp="$3" tg="$4" ; local ppts="BUS" tgts="BUS" i out
  swap "$var"
  for i in $(seq 1 "$MAXTRY"); do
    out=$($PIN "$BIN/llama-bench" -m "$M" -p "$pp" -n 0 -t 8 -r 3 2>/dev/null | grep -oE "pp$pp *\| *[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
    if [ -n "$out" ]; then ppts="$out"; echo "  $lbl pp$pp try$i OK=$out"; break; else echo "  $lbl pp$pp try$i BUS/empty"; fi
  done
  for i in $(seq 1 "$MAXTRY"); do
    out=$($PIN "$BIN/llama-bench" -m "$M" -p 0 -n "$tg" -t 8 -r 3 2>/dev/null | grep -oE "tg$tg *\| *[0-9.]+" | grep -oE "[0-9.]+$" | tail -1)
    if [ -n "$out" ]; then tgts="$out"; echo "  $lbl tg$tg try$i OK=$out"; break; else echo "  $lbl tg$tg try$i BUS/empty"; fi
  done
  echo "RESULT $lbl pp=$ppts tg=$tgts"
}

echo "== WARMUP (best effort) =="
swap OFF; $PIN "$BIN/llama-bench" -m "$M" -p 16 -n 4 -t 8 -r 1 >/dev/null 2>&1 || true

for pass in 1 2; do
  echo "== PASS $pass =="
  measure ON  ours_p$pass  64 16
  measure OFF stock_p$pass 64 16
done
swap OFF
echo "== LIVE restored OFF md5=$(md5sum "$LIVE"|cut -c1-16) =="
echo "== RETRY PERF DONE =="
