#!/usr/bin/env bash
# G7 P0 补格: q5_K@k1 structural-axis two-point audit (DRAM-byte counter unavailable on X60 PMU).
# Counters: HW instructions + cycles (only working HW events). Two-point N=16/64 -> per decode-token.
# Swap-only deploy (no rebuild): LIVE = /data/build-k1-q5k/bin/libggml-cpu.so.0.15.1
#   OFF(stock block-dot, clang-18) = 14b6add6 ; ON(our repack GEMM/GEVM) = a408563e
set -u
BIN=/data/build-k1-q5k/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
WK=/tmp/g5_q5k
OUT=/tmp/g7q5k; mkdir -p "$OUT"
MODEL=/data/tinyllama-1.1b-Q5_K_M.gguf
BENCH=/data/k1build/bin/llama-bench
SC=/tmp/structcnt
CORES=0-3; THREADS=4; REPS=3
export LD_LIBRARY_PATH="$BIN"
swap(){ cp -f "$WK/libggml-cpu.so.$1" "$LIVE"; }

echo "== ENV =="
echo "isa=$(grep -m1 -i isa /proc/cpuinfo | cut -c1-40)..."
echo "gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
echo "loadavg_pre=$(cat /proc/loadavg)"
echo "ON  md5=$(md5sum $WK/libggml-cpu.so.ON|awk '{print $1}')"
echo "OFF md5=$(md5sum $WK/libggml-cpu.so.OFF|awk '{print $1}')"
echo "LIVE_pre md5=$(md5sum $LIVE|awk '{print $1}')"
echo "config: CORES=$CORES THREADS=$THREADS REPS=$REPS  N in {16,64}  --no-warmup -r 1"

run(){ # $1=variant $2=N $3=rep  -> emits one line: VAR N REP instr cycles ts freq
  local var=$1 N=$2 rep=$3
  local log=$OUT/run_${var}_N${N}_r${rep}.txt
  taskset -c $CORES $SC -- $BENCH -m $MODEL -p 0 -n $N -t $THREADS -r 1 --no-warmup -o json > $log 2>&1
  local instr cyc ts eng fk
  instr=$(grep STRUCTCNT $log | sed -n 's/.*instructions=\([0-9]*\).*/\1/p')
  cyc=$(grep STRUCTCNT $log | sed -n 's/.*cycles=\([0-9]*\).*/\1/p')
  ts=$(grep -m1 avg_ts $log | sed -n 's/.*: \([0-9.]*\).*/\1/p')
  eng=$(grep -c ENGAGED $log)
  fk=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
  echo "DATA var=$var N=$N rep=$rep instr=$instr cycles=$cyc ts=$ts engage=$eng freq=$fk"
}

echo "== RUNS (interleaved OFF/ON per rep) =="
for rep in $(seq 1 $REPS); do
  swap OFF; run OFF 16 $rep; run OFF 64 $rep
  swap ON;  run ON  16 $rep; run ON  64 $rep
done

echo "== RESTORE =="
swap OFF
echo "LIVE_post md5=$(md5sum $LIVE|awk '{print $1}')  (expect 14b6add6 OFF-pristine)"
echo "loadavg_post=$(cat /proc/loadavg)"
echo "== DONE =="
