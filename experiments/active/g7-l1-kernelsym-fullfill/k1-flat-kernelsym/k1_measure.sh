#!/usr/bin/env bash
# G7 L1 货架A k1-半 kernel-sym — hot/cold + nr{4,16,64} sweep. Self-driven, DONE sentinel, no watcher.
set -u
RUN=/tmp/g7_l1_k1cold
BIN=$RUN/flatcold_k1
CORE="${1:-3}"          # pin 0-3 allowed; use core 3 (L2 shared 0-3)
POOL="${2:-8}"          # P=8 tiles; each ~640KiB plain >> k1 L2 512KiB (no L3) => cold
ROUNDS="${3:-12}"       # N=12 median (>=T-N)
HITERS="${4:-6}"
FMTS="${5:-q4_1 q5_0 q5_1}"
NRS="${6:-16 4 64}"     # nr16 = anchor same-shape (FLAT立账 shape) first
K=2048; NC=512
LOG=$RUN/run.log
: > "$LOG"
echo "# MEASURE core=$CORE pool=$POOL rounds=$ROUNDS hiters=$HITERS K=$K nc=$NC" | tee -a "$LOG"
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
for FMT in $FMTS; do
  for NR in $NRS; do
    SEED=$((0x1234 + NR*7))
    echo "## $FMT nr=$NR" | tee -a "$LOG"
    taskset -c "$CORE" "$BIN" "$FMT" "$K" "$NR" "$NC" "$HITERS" "$POOL" "$ROUNDS" "$SEED" 2>>"$LOG" | tee -a "$LOG"
  done
  echo "## FMT_DONE $FMT" | tee -a "$LOG"
done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
