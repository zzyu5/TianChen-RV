#!/usr/bin/env bash
# k1_measure.sh — G7 L1 iq/fp4 vec_dot + dequant@k1: load-gate + idle-core pick (0-3) +
# verify byte-exact + hot/cold census. Runs ON k1. k1 restored, no stray.
set -u
RDIR=/tmp/g7_iqfp4_dq_k1
GGML=/data/k1build-stock/bin
LOG=$RDIR/run.log; : > "$LOG"
VFMTS="iq1_s iq1_m iq4_nl nvfp4"
DFMTS="q2_K q3_K q4_K q5_K q6_K iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4 tq1_0 tq2_0"
VK="${VK:-2048}"; VNC="${VNC:-512}"; HIT="${HIT:-8}"; REPS="${REPS:-12}"
DK="${DK:-131072}"                     # dequant streaming: 131072 elems -> 512KiB out > L2

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

runv(){ LD_LIBRARY_PATH=$GGML taskset -c "$CORE" "$RDIR/iqfp4_vecdot" "$@" 2>>"$LOG" | tee -a "$LOG"; }
rund(){ LD_LIBRARY_PATH=$GGML taskset -c "$CORE" "$RDIR/dequant_census" "$@" 2>>"$LOG" | tee -a "$LOG"; }

echo "=== [V-VERIFY] iq/fp4 vec_dot byte-exact (M=1 & M=8, nc=64) ===" | tee -a "$LOG"
for f in $VFMTS; do runv "$f" "$VK" 1 64 1 10 0xBEEF 1; runv "$f" "$VK" 8 64 1 10 0xBEEF 1; done
echo "=== [V-S1] iq/fp4 vec_dot CENSUS M=1 GEVM (K=$VK nc=$VNC) ===" | tee -a "$LOG"
for f in $VFMTS; do SEED=$(printf '0x%X' $((0x1000 + RANDOM))); runv "$f" "$VK" 1 "$VNC" "$HIT" "$REPS" "$SEED"; done
echo "=== [V-S8] iq/fp4 vec_dot CENSUS M=8 shape point ===" | tee -a "$LOG"
for f in $VFMTS; do SEED=$(printf '0x%X' $((0x2000 + RANDOM))); runv "$f" "$VK" 8 "$VNC" "$HIT" "$REPS" "$SEED"; done

echo "=== [D-VERIFY] dequant byte-exact (K=$DK) ===" | tee -a "$LOG"
for f in $DFMTS; do rund "$f" "$DK" 1 10 0xD00D 1; done
echo "=== [D-S1] dequant streaming CENSUS (K=$DK) ===" | tee -a "$LOG"
for f in $DFMTS; do SEED=$(printf '0x%X' $((0x3000 + RANDOM))); rund "$f" "$DK" "$HIT" "$REPS" "$SEED"; done

echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(( $(pgrep -c -f 'iqfp4_vecdot|dequant_census' 2>/dev/null || echo 0) ))" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
