#!/usr/bin/env bash
# A2-batch5 K-quant M=1 GEVM decode @k1(VLEN256, clang-18 symmetric). ours=repack-GEVM leaf (hl8=half-util at VLEN256, what-if:
# k1 front-door DECLINES repack@decode per batch4; this is a force-constructed candidate). opp=stock .so block-dot.
set -uo pipefail
WD=/tmp/g8_a2b5_k1; CXX=/usr/bin/clang++
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log
mkdir -p $WD; cd $WD; : > "$SEAL"; : > "$LOG"
echo "# A2b5 KQUANT-GEVM-M1 @k1 CXX=$($CXX --version|head -1) march=$MARCH -O3 -fno-integrated-as -DBOARD_K1 $(date -u)" | tee -a "$SEAL"
echo "# stock_so_md5_before=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
FMTS="q2_K q3_K q5_K q6_K"
declare -A KOBJ
for f in q2 q3 q5 q6; do
  $CXX -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -c $WD/leaves/${f}_K_gevm.c -o $WD/${f}.o 2>$WD/cc_${f}.err || { echo "KERN_FAIL $f"; head -8 $WD/cc_${f}.err; exit 3; }
  VS=$(objdump -d $WD/${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  SFP=$(objdump -d $WD/${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2' && echo SOFTFP || echo cleanfp)
  SZ=$(objdump -h $WD/${f}.o 2>/dev/null | awk '/\.text/{print $3; exit}')
  echo "# OURS_${f}_K: fp16=$SFP vsetvl=$VS text=0x$SZ" | tee -a "$SEAL"; KOBJ[$f]=$WD/${f}.o
done
for s in ggml_vec_dot_q2_K_q8_K ggml_vec_dot_q3_K_q8_K ggml_vec_dot_q5_K_q8_K ggml_vec_dot_q6_K_q8_K; do
  a=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ${s}\$" | awk '{print $1}')
  echo "# OPP $s @0x${a:-?} (stock .so, native-RVV inline)" | tee -a "$SEAL"
done
$CXX -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -DBOARD_K1 -std=c++17 -c $WD/kquant_gevm_m1_driver_b5.cpp -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -20 $WD/cc_drv.err; exit 4; }
$CXX $WD/drv.o ${KOBJ[q2]} ${KOBJ[q3]} ${KOBJ[q5]} ${KOBJ[q6]} \
   -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/bench 2>$WD/ld.err || { echo LINK_FAIL; head -20 $WD/ld.err; exit 5; }
echo "# LINK OK -> $WD/bench" | tee -a "$SEAL"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4; BESTC=0; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); echo "# core$c idle=$pct%" | tee -a "$LOG"; [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/bench "$@" 2>>"$LOG" | tee -a "$LOG"; }
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "=== VERIFY (nc=64) ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 64 12 1; done
echo "=== COLD PRIMARY K=2048 N=512 reps=25 seed A=0xC0FFEE1 (x2 trials) ===" | tee -a "$LOG"
for f in $FMTS; do for t in 1 2; do run $f 2048 512 25 0 0xC0FFEE1; done; done
echo "=== COLD 2nd-seed K=2048 N=512 reps=25 seed B=0x1357ACE (x2 trials) ===" | tee -a "$LOG"
for f in $FMTS; do for t in 1 2; do run $f 2048 512 25 0 0x1357ACE; done; done
echo "# stock_so_md5_after=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -f a2b5_k1/bench || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
