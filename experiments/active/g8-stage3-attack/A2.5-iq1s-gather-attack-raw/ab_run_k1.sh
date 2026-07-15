#!/usr/bin/env bash
# A2.5 iq1_s deferred-scale-reduction A/B @k1(VLEN256). baseline vs attack(deferred).
# ours clang-18 -O3 VLEN256 decree march. opp = stock libggml-cpu.so vec_dot.
set -uo pipefail
WD=/tmp/g8k1_iq_a25
CC=/usr/bin/clang
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
SEAL=$WD/build_seal.txt; LOG=$WD/ab_cold_k1.log
mkdir -p $WD; cd $WD
: > "$SEAL"; : > "$LOG"
echo "# A2.5 iq1_s@k1 CC=$($CC --version|head -1) march=$MARCH $(date -u)" | tee -a "$SEAL"
echo "# stock_so_md5=$(md5sum $GGML/libggml-cpu.so 2>/dev/null|cut -d' ' -f1)" | tee -a "$SEAL"

$CC -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -x c $WD/iq_vecdot_driver.c -c -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -12 $WD/cc_drv.err; exit 4; }
$CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/iq4_nl_blockdot.kernel.c -c -o $WD/iq4nl_bd.o 2>/dev/null
$CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/nvfp4_blockdot.kernel.c -c -o $WD/nvfp4_bd.o 2>/dev/null

build_variant(){
  local V=$1
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/iq1_s.${V}.kernel.c -c -o $WD/iq1_s_${V}.o 2>$WD/cc_iq1s_${V}.err || { echo "IQ1S_$V FAIL"; head -6 $WD/cc_iq1s_${V}.err; exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/iq1_m.${V}.kernel.c -c -o $WD/iq1_m_${V}.o 2>$WD/cc_iq1m_${V}.err || { echo "IQ1M_$V FAIL"; head -6 $WD/cc_iq1m_${V}.err; exit 3; }
  VS=$(objdump -d $WD/iq1_s_${V}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  VM=$(objdump -d $WD/iq1_s_${V}.o 2>/dev/null | grep -cE '\bvmv\.x\.s\b')
  VSE=$(objdump -d $WD/iq1_s_${V}.o 2>/dev/null | grep -cE '\bvse32\b')
  echo "# OURS_iq1_s_${V}: vsetvl=$VS vmv.x.s=$VM vse32=$VSE size=$(wc -c<$WD/iq1_s_${V}.o)B" | tee -a "$SEAL"
  $CC $WD/drv.o $WD/iq1_s_${V}.o $WD/iq1_m_${V}.o $WD/iq4nl_bd.o $WD/nvfp4_bd.o -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/iq_bench_${V} 2>$WD/ld_${V}.err || { echo "LINK_$V FAIL"; head -20 $WD/ld_${V}.err; exit 5; }
  echo "# LINK OK -> iq_bench_${V}" | tee -a "$SEAL"
}
build_variant baseline
build_variant attack

# pick idle core 0-3
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=0; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
echo "# loadavg=$(cat /proc/loadavg)" | tee -a "$LOG"
run(){ local V=$1; shift; LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/iq_bench_${V} "$@" 2>>"$LOG" | tee -a "$LOG"; }

for V in baseline attack; do
  echo "======== VARIANT=$V ========" | tee -a "$LOG"
  run $V iq1_s 2048 1 64 8 20 0x1234 1     # VERIFY M=1
  run $V iq1_s 2048 8 64 8 20 0x1234 1     # VERIFY M=8
  for nc in 64 256 512; do
    for seed in 0x2001 0x5ABC; do run $V iq1_s 2048 1 $nc 8 20 $seed; done
  done
done
echo "# ALL_DONE $(date -u)" | tee -a "$LOG"
