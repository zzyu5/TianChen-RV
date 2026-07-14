#!/usr/bin/env bash
# G8 §六.3 group2: iq1_s/iq1_m gather attack @k1(VLEN256). baseline vs gather-widened (rvv lever).
# ours clang-18 -O3 VLEN256 decree march. opp = stock libggml-cpu.so vec_dot (VLEN-adaptive RVV gather).
set -uo pipefail
WD=/tmp/g8k1_iq
CC=/usr/bin/clang
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log
mkdir -p $WD; cd $WD
: > "$SEAL"; : > "$LOG"
echo "# G8 iq@k1 CC=$($CC --version|head -1) march=$MARCH -O3 -fno-integrated-as $(date -u)" | tee -a "$SEAL"
echo "# stock_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

build_variant(){ # $1 = baseline|wide
  local V=$1
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/iq1_s.${V}.kernel.c -c -o $WD/iq1_s_${V}.o 2>$WD/cc_iq1s_${V}.err || { echo "IQ1S_$V FAIL"; head -6 $WD/cc_iq1s_${V}.err; exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/iq1_m.${V}.kernel.c -c -o $WD/iq1_m_${V}.o 2>$WD/cc_iq1m_${V}.err || { echo "IQ1M_$V FAIL"; head -6 $WD/cc_iq1m_${V}.err; exit 3; }
  for f in iq1_s iq1_m; do
    VS=$(objdump -d $WD/${f}_${V}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    VG=$(objdump -d $WD/${f}_${V}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather')
    echo "# OURS_${f}_${V}: vsetvl=$VS gather=$VG size=$(wc -c<$WD/${f}_${V}.o)B" | tee -a "$SEAL"
  done
  # shared iq4_nl/nvfp4 (satisfy driver externs; unused in iq1 runs)
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/iq4_nl_blockdot.kernel.c -c -o $WD/iq4nl_bd.o 2>/dev/null
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -x c++ $WD/kernels/nvfp4_blockdot.kernel.c -c -o $WD/nvfp4_bd.o 2>/dev/null
  $CC $WD/drv.o $WD/iq1_s_${V}.o $WD/iq1_m_${V}.o $WD/iq4nl_bd.o $WD/nvfp4_bd.o -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/iq_bench_${V} 2>$WD/ld_${V}.err || { echo "LINK_$V FAIL"; head -20 $WD/ld_${V}.err; exit 5; }
  echo "# LINK OK -> iq_bench_${V}" | tee -a "$SEAL"
}
$CC -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -x c $WD/iq_vecdot_driver.c -c -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -12 $WD/cc_drv.err; exit 4; }
build_variant baseline
build_variant wide
# opponent probe
for s in ggml_vec_dot_iq1_s_q8_K ggml_vec_dot_iq1_m_q8_K; do a=$(nm -D $GGML/libggml-cpu.so|grep -E " T ${s}\$"|awk '{print $1}'); echo "# OPP $s @0x${a:-?}" | tee -a "$SEAL"; done

echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=0; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ local V=$1; shift; LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/iq_bench_${V} "$@" 2>>"$LOG" | tee -a "$LOG"; }

for V in baseline wide; do
  echo "======== VARIANT=$V ========" | tee -a "$LOG"
  for f in iq1_s iq1_m; do
    for t in 1 2 3 4 5; do
      run $V $f 2048 1 512 8 15 $(printf '0x%X' $((0x2000+t*13)))
    done
  done
done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -f iq_bench || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
