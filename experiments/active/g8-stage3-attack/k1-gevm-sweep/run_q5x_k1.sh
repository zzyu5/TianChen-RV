#!/usr/bin/env bash
# G8 §六.3 q5_0/q5_1 @k1(VLEN256) deployed-core verify + repack what-if.
# ours = clang-18 -O3 VLEN256 (repack-mf2 leaf + deployed block-dot). opp = stock libggml-cpu.so.
set -uo pipefail
WD=/tmp/g8k1_q5x
CC=/usr/bin/clang
CXX=/usr/bin/clang++
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log
mkdir -p $WD; cd $WD
: > "$SEAL"; : > "$LOG"
echo "# G8 q5x@k1 CC=$($CC --version|head -1) march=$MARCH -O3 -fno-integrated-as $(date -u)" | tee -a "$SEAL"
echo "# stock_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

# --- compile ours kernels ---
declare -A KOBJ
for k in q5_0_gevm_repack_mf2 q5_1_gevm_repack_mf2 q5_0_blockdot_deployed q5_1_blockdot_deployed; do
  $CXX -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -c $WD/kernels/${k}.c -o $WD/${k}.o 2>$WD/cc_${k}.err || { echo "KERN_FAIL $k"; head -8 $WD/cc_${k}.err; exit 3; }
  VS=$(objdump -d $WD/${k}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  RV=$(objdump -d $WD/${k}.o 2>/dev/null | grep -cE '\bv[a-z]')
  SFP=$(objdump -d $WD/${k}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2' && echo SOFTFP || echo cleanfp)
  SZ=$(objdump -h $WD/${k}.o 2>/dev/null | awk '/\.text/{print $3; exit}')
  echo "# OURS_${k}: fp16=$SFP vsetvl=$VS rvv~=$RV textsize=0x$SZ" | tee -a "$SEAL"
  KOBJ[$k]=$WD/${k}.o
done
# --- opponent probe ---
for s in ggml_vec_dot_q5_0_q8_0 ggml_vec_dot_q5_1_q8_1; do
  a=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ${s}\$" | awk '{print $1}')
  echo "# OPP $s @0x${a:-?} (stock .so, native-RVV inline)" | tee -a "$SEAL"
done
# --- compile driver + link against stock .so ---
$CXX -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -c $WD/q5x_driver.cpp -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -15 $WD/cc_drv.err; exit 4; }
$CXX $WD/drv.o ${KOBJ[q5_0_gevm_repack_mf2]} ${KOBJ[q5_1_gevm_repack_mf2]} ${KOBJ[q5_0_blockdot_deployed]} ${KOBJ[q5_1_blockdot_deployed]} \
   -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/q5x_bench 2>$WD/ld.err || { echo LINK_FAIL; head -20 $WD/ld.err; exit 5; }
echo "# LINK OK -> $WD/q5x_bench" | tee -a "$SEAL"

# --- pick idle core 0-3 ---
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=0; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); echo "# core$c idle=$pct%" | tee -a "$LOG"; [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/q5x_bench "$@" 2>>"$LOG" | tee -a "$LOG"; }

# --- verify once ---
echo "=== VERIFY (nc=64) ===" | tee -a "$LOG"
run q5_0 2048 64 12 1
run q5_1 2048 64 12 1
# --- cold census: primary shape K=2048 N=512 (matches §六 CENSUS), N>=10 independent trials ---
echo "=== COLD PRIMARY K=2048 N=512 (x12 trials) ===" | tee -a "$LOG"
for t in $(seq 1 12); do run q5_0 2048 512 15; run q5_1 2048 512 15; done
# --- secondary shape K=4096 N=256 ---
echo "=== COLD SECONDARY K=4096 N=256 (x4 trials) ===" | tee -a "$LOG"
for t in $(seq 1 4); do run q5_0 4096 256 12; run q5_1 4096 256 12; done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -f q5x_bench || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
