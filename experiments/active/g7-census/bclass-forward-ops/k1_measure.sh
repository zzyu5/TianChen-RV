#!/usr/bin/env bash
# G7 L1 B-fwd kernel-sym — hot/cold + M=8 shape sweep @k1. Load-gate + idle-core pick (0-3) + restore.
set -u
RUN=/tmp/g7_bfwd_k1
BIN=$RUN/bfwd_k1
ROUNDS="${1:-12}"; HITERS="${2:-8}"; TMB="${3:-6}"
OPS="${4:-add mul scale cpy silu gelu rms_norm softmax rope}"
SHAPES="${5:-512 1024 2048 3072 4096 5120 8192 16384}"   # M=8 shape sweep; 4096 = anchor
LOG=$RUN/run.log; : > "$LOG"

echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
# ---- idle-core pick among 0-3 (sample /proc/stat delta 300ms) ----
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
echo "# LOAD_GATE_OK pinning core=$CORE (idle=${BESTIDLE}%)" | tee -a "$LOG"
echo "# gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
echo "# rounds=$ROUNDS hiters=$HITERS target_mb=$TMB" | tee -a "$LOG"

for OP in $OPS; do
  for N in $SHAPES; do
    SEED=$((0xB0F + N*13))
    taskset -c "$CORE" "$BIN" "$OP" "$N" "$HITERS" "$ROUNDS" "$TMB" "$SEED" 2>>"$LOG" | tee -a "$LOG"
  done
  echo "## OP_DONE $OP" | tee -a "$LOG"
done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
# --- no stray procs check ---
echo "# STRAY_BFWD=$(pgrep -c -f bfwd_k1 || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
