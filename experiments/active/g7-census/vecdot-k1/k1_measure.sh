#!/usr/bin/env bash
# k1_measure.sh — G7 L1 vec_dot@k1: load-gate + idle-core pick (0-3) + verify byte-exact +
# hot/cold census sweep (M=1 GEVM + M=8 shape point). Runs ON k1. k1 restored, no stray.
set -u
RDIR=/tmp/g7_vecdot_k1
BIN=$RDIR/vecdot_census
GGML=/data/k1build-stock/bin
LOG=$RDIR/run.log; : > "$LOG"
FMTS="q2_K q3_K q4_K q5_K q6_K q4_0 q4_1 q5_0 q5_1 q8_0"   # K-quant first (attack value), FLAT after
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-8}"; REPS="${REPS:-12}"

echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.3
BESTC=-1; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); echo "# core$c idle_pct=$pct" | tee -a "$LOG";
  if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
if [ $BESTIDLE -lt 70 ]; then echo "# LOAD_GATE_FAIL best core$BESTC idle=${BESTIDLE}% (<70) — ABORT" | tee -a "$LOG"; exit 9; fi
CORE=$BESTC
echo "# LOAD_GATE_OK pin core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"

run(){ LD_LIBRARY_PATH=$GGML taskset -c "$CORE" "$BIN" "$@" 2>>"$LOG" | tee -a "$LOG"; }

echo "=== [V] VERIFY byte-exact (M=1 & M=8, nc=64) ===" | tee -a "$LOG"
for f in $FMTS; do
  bk=256; case $f in q4_0|q4_1|q5_0|q5_1|q8_0) bk=32;; esac
  run "$f" "$K" 1 64 1 10 0xBEEF 1
  run "$f" "$K" 8 64 1 10 0xBEEF 1
done

echo "=== [S1] CENSUS M=1 GEVM (K=$K nc=$NC) ===" | tee -a "$LOG"
for f in $FMTS; do
  SEED=$(printf '0x%X' $((0x1000 + RANDOM)))
  run "$f" "$K" 1 "$NC" "$HITERS" "$REPS" "$SEED"
done
echo "# loadavg_mid=$(cat /proc/loadavg)" | tee -a "$LOG"

echo "=== [S8] CENSUS M=8 shape point (K=$K nc=$NC) ===" | tee -a "$LOG"
for f in $FMTS; do
  SEED=$(printf '0x%X' $((0x2000 + RANDOM)))
  run "$f" "$K" 8 "$NC" "$HITERS" "$REPS" "$SEED"
done

echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -f vecdot_census || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
