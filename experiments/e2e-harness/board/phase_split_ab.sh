#!/usr/bin/env bash
# phase_split_ab.sh -- constitutional paired A/B e2e phase-split, run ON the
# board. Uses llama-bench's native prefill/decode split: `-p PP` measures
# PROMPT PROCESSING throughput (prefill phase) and `-n TG` measures TOKEN
# GENERATION throughput (decode phase), both in tokens/s.
#
# Discipline wired (实验总纲v1 §1 第2/3/5条):
#   - paired A/B interleaved in the SAME session (ours, stock, ours, stock, ...)
#   - core-pinned (taskset) + freq captured per call (DVFS guard)
#   - 2 passes => between-pass T-N noise floor
#   - `-r REPS -o json` => per-rep samples for median + IQR
#   - board identity + env fingerprint printed for instance-hash keying
# Emits ###AB ... ###END blocks consumed by aggregate_e2e.py.
#
# Env: A_BUILD B_BUILD (dirs) | MODEL | CORES (e.g. 8-11) | PP TG REPS | LDPATH
#      BENCH_BIN_A/B (default $X_BUILD/bin/llama-bench)
set -u
A_BUILD="${A_BUILD:?}"; B_BUILD="${B_BUILD:?}"; MODEL="${MODEL:?}"
CORES="${CORES:-8-11}"; PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-5}"; PASSES="${PASSES:-2}"
BENCH_A="${BENCH_BIN_A:-$A_BUILD/bin/llama-bench}"
BENCH_B="${BENCH_BIN_B:-$B_BUILD/bin/llama-bench}"
[ -n "${LDPATH:-}" ] && export LD_LIBRARY_PATH="$LDPATH:${LD_LIBRARY_PATH:-}"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')

echo "== ENV FINGERPRINT (instance-hash inputs) =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc)  pinned_cores=$CORES"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq 2>/dev/null || echo NA)"
echo "libc=$(getconf GNU_LIBC_VERSION 2>/dev/null)"
echo "model=$MODEL  model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "bench_A=$BENCH_A"
echo "bench_B=$BENCH_B"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES"

freq() { cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

run_side() {  # $1=bench  $2=label(ours|stock)  $3=pass
  local bench="$1" side="$2" pass="$3" fk
  fk=$(freq)
  echo "###AB pass=$pass side=$side freq_khz=$fk"
  # one call yields BOTH phases: pp<PP> (prefill) and tg<TG> (decode)
  $PIN "$bench" -m "$MODEL" -p "$PP" -n "$TG" -t "${THREADS:-4}" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}

# ---- warmup (dropped): one untimed pass each side to page-in mmap ----
echo "== WARMUP (dropped) =="
$PIN "$BENCH_A" -m "$MODEL" -p 8 -n 4 -t "${THREADS:-4}" -r 1 >/dev/null 2>&1
$PIN "$BENCH_B" -m "$MODEL" -p 8 -n 4 -t "${THREADS:-4}" -r 1 >/dev/null 2>&1

# ---- paired A/B, PASSES passes (pass1 vs pass2 => T-N between-run floor) ----
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass =="
  run_side "$BENCH_A" ours  "$pass"
  run_side "$BENCH_B" stock "$pass"
done
echo "== DONE =="
