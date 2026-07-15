#!/usr/bin/env bash
# finish_k1_run.sh — runs ON k1 (kernels already compiled in RDIR). Link (hardcoded .o list, zsh-safe)
# + load-gate + verify byte-exact + 2-seed N=24 cold census. Launched via nohup to survive ssh detach.
set -uo pipefail
RDIR=/tmp/g8_a2b8_dq_k1
GGML=/data/k1build-stock/bin
MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause
CC=clang-18
DFMTS="q2_K q3_K q4_K q5_K q6_K iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4 tq1_0 tq2_0"
DK=1048576; HIT=8; REPS=24; S1=0x1357; S2=0xACE2
cd "$RDIR"
SEAL=$RDIR/build_seal.txt
OBJS="kd_q2_K.o kd_q3_K.o kd_q4_K.o kd_q5_K.o kd_q6_K.o kd_iq1_s.o kd_iq1_m.o kd_iq2_xxs.o kd_iq2_xs.o kd_iq2_s.o kd_iq3_xxs.o kd_iq3_s.o kd_iq4_nl.o kd_iq4_xs.o kd_mxfp4.o kd_nvfp4.o kd_tq1_0.o kd_tq2_0.o"

echo "=== [C2] relink (hardcoded .o list) ===" | tee -a "$SEAL"
$CC -O2 -march=$MARCH -mabi=lp64d -x c dequant_census_driver.c -c -o drv_d.o 2>ccdrv.err || { echo DDRV_FAIL; head -12 ccdrv.err; exit 4; }
$CC drv_d.o $OBJS -L$GGML -Wl,-rpath,$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o dequant_census 2>ldd.err || { echo DLINK_FAIL; head -20 ldd.err; exit 5; }
echo "# linked OK -> ./dequant_census" | tee -a "$SEAL"

LOG=$RDIR/run.log; : > "$LOG"
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in 0 1 2 3 4 5 6 7; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=-1; BESTIDLE=-1
for c in 0 1 2 3 4 5 6 7; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
if [ $BESTIDLE -lt 70 ]; then echo "# LOAD_GATE_FAIL best core$BESTC idle=${BESTIDLE}% ABORT" | tee -a "$LOG"; exit 9; fi
CORE=$BESTC
echo "# LOAD_GATE_OK pin core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
rund(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE ./dequant_census "$@" 2>>"$LOG" | tee -a "$LOG"; }

echo "=== [D-VERIFY] byte-exact ZERO-MODEL (K=$DK) ===" | tee -a "$LOG"
for f in $DFMTS; do rund $f $DK 1 10 0xD00D 1; done
echo "=== [D-S1] cold streaming seed1=$S1 (K=$DK reps=$REPS) ===" | tee -a "$LOG"
for f in $DFMTS; do rund $f $DK $HIT $REPS $S1; done
echo "=== [D-S2] cold streaming seed2=$S2 (K=$DK reps=$REPS) ===" | tee -a "$LOG"
for f in $DFMTS; do rund $f $DK $HIT $REPS $S2; done

echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# base_md5_after=$(md5sum $GGML/libggml-base.so|cut -d' ' -f1)" | tee -a "$LOG"
echo "# STRAY_x=$(pgrep -x dequant_census | wc -l)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
