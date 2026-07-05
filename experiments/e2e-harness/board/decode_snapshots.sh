#!/usr/bin/env bash
# decode_snapshots.sh -- characterize the DECODE (GEVM, memory-bound) A/B ratio
# across board memory-pressure levels, run ON the board. One invocation = ONE
# load level (light/medium/heavy); the local orchestrator calls it 3x.
#
# Decode is memory-bandwidth bound, so its ABSOLUTE throughput (and thus the
# A/B ratio) drifts with whole-board memory pressure that core-pinning cannot
# fence out. We therefore inject a controlled STREAM-triad load on cores OUTSIDE
# the measurement set and report the ratio at each level -- an honest RANGE, not
# a single point. The paired A/B interleave keeps both sides under the SAME
# instantaneous load, so the ratio stays valid at every level.
#
# Emits ###AB pass=.. side=.. blocks (aggregate_e2e.py-compatible), decode-only
# (`-p 0 -n TG`). Prints loadavg + free MiB snapshots around the measurement.
#
# Env: A_BUILD B_BUILD MODEL | MEAS_CORES(8-11) THREADS(4) TG REPS PASSES
#      LEVEL(label) LOAD_WORKERS(int) LOAD_CORES(taskset range) LDPATH
set -u
A_BUILD="${A_BUILD:?}"; B_BUILD="${B_BUILD:?}"; MODEL="${MODEL:?}"
MEAS_CORES="${MEAS_CORES:-8-11}"; THREADS="${THREADS:-4}"
TG="${TG:-32}"; REPS="${REPS:-4}"; PASSES="${PASSES:-2}"
LEVEL="${LEVEL:-light}"; LOAD_WORKERS="${LOAD_WORKERS:-0}"; LOAD_CORES="${LOAD_CORES:-16-63}"
BENCH_A="${BENCH_BIN_A:-$A_BUILD/bin/llama-bench}"
BENCH_B="${BENCH_BIN_B:-$B_BUILD/bin/llama-bench}"
[ -n "${LDPATH:-}" ] && export LD_LIBRARY_PATH="$LDPATH:${LD_LIBRARY_PATH:-}"
PIN="taskset -c $MEAS_CORES"
CORE0=$(echo "$MEAS_CORES" | grep -oE '^[0-9]+')

# ---- build a tiny STREAM-triad memory-bandwidth hog (once) ----
HOG=/tmp/tcrv_memhog
if [ ! -x "$HOG" ]; then
  cat > /tmp/tcrv_memhog.c <<'EOF'
#include <stdlib.h>
#include <string.h>
// STREAM-triad over a large buffer to saturate memory bandwidth (not just ALU).
int main(int argc, char** argv){
  size_t N = (argc>1)? (size_t)strtoull(argv[1],0,10) : (size_t)64*1024*1024; // doubles
  double *a=malloc(N*8), *b=malloc(N*8), *c=malloc(N*8);
  if(!a||!b||!c) return 1;
  for(size_t i=0;i<N;i++){a[i]=1.0;b[i]=2.0;c[i]=0.0;}
  volatile double s=0; double q=3.0;
  for(;;){ for(size_t i=0;i<N;i++) c[i]=a[i]+q*b[i]; s+=c[N-1]; }
  return (int)s;
}
EOF
  cc -O2 /tmp/tcrv_memhog.c -o "$HOG" 2>/dev/null || \
    /opt/tcrv-toolchains/gcc-15.2.0/bin/gcc -O2 /tmp/tcrv_memhog.c -o "$HOG"
fi

freq() { cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }
loadline() { echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)  free_MiB=$(free -m | awk '/Mem:/{print $4}')  freq_khz=$(freq)"; }

HOG_PIDS=""
start_load() {
  [ "$LOAD_WORKERS" -le 0 ] && return 0
  local n; for n in $(seq 1 "$LOAD_WORKERS"); do
    taskset -c "$LOAD_CORES" "$HOG" 33554432 >/dev/null 2>&1 &   # ~256MiB/buffer/worker
    HOG_PIDS="$HOG_PIDS $!"
  done
  sleep 3   # let the hogs ramp memory traffic
}
stop_load() {
  [ -z "$HOG_PIDS" ] && return 0
  kill $HOG_PIDS 2>/dev/null; wait $HOG_PIDS 2>/dev/null
  HOG_PIDS=""
}
trap stop_load EXIT

run_side() {  # $1=bench $2=label $3=pass
  local fk; fk=$(freq)
  echo "###AB pass=$3 side=$2 freq_khz=$fk"
  $PIN "$1" -m "$MODEL" -p 0 -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}

echo "== DECODE SNAPSHOT level=$LEVEL  LOAD_WORKERS=$LOAD_WORKERS on cores $LOAD_CORES =="
echo "meas_cores=$MEAS_CORES threads=$THREADS TG=$TG REPS=$REPS PASSES=$PASSES"
echo "PRE-LOAD  $(loadline)"
start_load
echo "POST-LOAD $(loadline)"
# warmup (dropped)
$PIN "$BENCH_A" -m "$MODEL" -p 0 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
$PIN "$BENCH_B" -m "$MODEL" -p 0 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass (level=$LEVEL) =="
  run_side "$BENCH_A" ours  "$pass"
  echo "MIDLOAD $(loadline)"
  run_side "$BENCH_B" stock "$pass"
done
echo "END-LOAD  $(loadline)"
stop_load
echo "== SNAPSHOT $LEVEL DONE =="
