#!/usr/bin/env bash
# G.0.3 gelu f16-LUT rematch — board build + single-instance cold run.
# env: CC MARCH CORES EXTRALINK RUN TAG  ; args: [rounds] [target_mb]
set -u
RUN="${RUN:-/tmp/gelu_f16lut}"; TAG="${TAG:-board}"
ROUNDS="${1:-12}"; TMB="${2:-6}"
CC="${CC:?set CC}"; MARCH="${MARCH:?set MARCH}"; CORES="${CORES:-0 1 2 3}"; EXTRALINK="${EXTRALINK:-}"
SEAL="$RUN/seal_$TAG.txt"; LOG="$RUN/run_$TAG.log"; : > "$SEAL"; : > "$LOG"
CCV=$($CC --version 2>/dev/null | head -1)
echo "# G.0.3 gelu-f16lut $TAG  CC=$CC ($CCV)  march=$MARCH" | tee -a "$SEAL"
echo "# SRC_MD5 kernel=$(md5sum $RUN/gelu.f16lut.kernel.c|cut -d' ' -f1) opp=$(md5sum $RUN/opp_gelu.cpp|cut -d' ' -f1) drv=$(md5sum $RUN/gelu_f16lut_driver.c|cut -d' ' -f1)" | tee -a "$SEAL"

# ---- single-instance hygiene: kill any prior instance, verify none ----
pkill -f gelu_f16lut_bin 2>/dev/null; sleep 0.4
STRAY_PRE=$(pgrep -c -f gelu_f16lut_bin || echo 0)
echo "# SINGLE_INSTANCE stray_pre_kill=$STRAY_PRE (expect 0)" | tee -a "$SEAL"

# ---- build (single TU set) ----
$CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/gelu.f16lut.kernel.c" -c -o "$RUN/k.o" 2>"$RUN/cc_k.err" || { echo "KERN_FAIL"; sed -n '1,8p' "$RUN/cc_k.err"; exit 3; }
$CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/opp_gelu.cpp" -c -o "$RUN/opp.o" 2>"$RUN/cc_opp.err" || { echo "OPP_FAIL"; sed -n '1,8p' "$RUN/cc_opp.err"; exit 3; }
$CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/gelu_f16lut_driver.c" -c -o "$RUN/drv.o" 2>"$RUN/cc_drv.err" || { echo "DRV_FAIL"; sed -n '1,12p' "$RUN/cc_drv.err"; exit 4; }
$CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on "$RUN/drv.o" "$RUN/k.o" "$RUN/opp.o" $EXTRALINK -lm -o "$RUN/gelu_f16lut_bin" 2>"$RUN/cc_link.err" || { echo "LINK_FAIL"; sed -n '1,12p' "$RUN/cc_link.err"; exit 4; }
echo "# BUILT $RUN/gelu_f16lut_bin" | tee -a "$SEAL"
# objdump: ours per-element = table gather (expect vluxei/lookup; no runtime tanhf call)
echo "# OURS_OBJ tanh_calls=$(objdump -d $RUN/k.o 2>/dev/null | grep -c 'tanh') gather_hint=$(objdump -d $RUN/k.o 2>/dev/null | grep -cE 'lhu|vlux|lh ')" | tee -a "$SEAL"

# ---- load-gate: idle-core pick ----
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in $CORES; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.3
BESTC=-1; BESTIDLE=-1
for c in $CORES; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0 ? 100*di/dt : 0 ));
  echo "# core$c idle_pct=$pct" | tee -a "$SEAL"; if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
[ $BESTIDLE -lt 70 ] && { echo "# LOAD_GATE_FAIL best core$BESTC idle=${BESTIDLE}% (<70) ABORT" | tee -a "$SEAL"; exit 9; }
CORE=$BESTC
echo "# LOAD_GATE_OK pin core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$SEAL"
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$SEAL"

# ---- run (single instance, pinned) ----
taskset -c "$CORE" "$RUN/gelu_f16lut_bin" "$ROUNDS" "$TMB" 0xC0FFEE 2>>"$LOG" | tee -a "$LOG"

echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$SEAL"
echo "# STRAY_POST=$(pgrep -c -f gelu_f16lut_bin || echo 0)" | tee -a "$SEAL"
echo "# ALL_DONE $TAG" | tee -a "$SEAL"
cat "$SEAL"
