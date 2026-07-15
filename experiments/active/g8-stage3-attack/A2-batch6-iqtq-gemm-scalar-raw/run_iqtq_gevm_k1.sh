#!/usr/bin/env bash
# A2-batch6 iq/tq/fp4 DECODE (M=1 GEVM) scalar-ref @k1 (clang-18 deploy). ours=repack-GEVM leaf, opp=generic per-col.
set -uo pipefail
WD=/tmp/g8_a2b6g_k1; SRC=$WD/src
CXX=/usr/bin/clang++; MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log
mkdir -p $WD; cd $WD; : > "$SEAL"; : > "$LOG"
echo "# A2b6-DECODE @k1 $($CXX --version|head -1) march=$MARCH $(date -u) stock_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
FMTS="iq4_xs iq2_xxs iq2_xs iq2_s mxfp4 tq1_0 tq2_0"
OBJS=""
for k in $FMTS; do
  $CXX -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -include math.h -c $SRC/kernels/${k}_gevm.c -o $WD/${k}g.o 2>$WD/cc_${k}.err \
    || { echo "KERN_FAIL $k"; head -12 $WD/cc_${k}.err; exit 3; }
  VS=$(objdump -d $WD/${k}g.o 2>/dev/null | grep -cE 'vsetvli|vsetivli'); echo "# OURS_gevm_${k}: vsetvl=$VS" | tee -a "$SEAL"; OBJS="$OBJS $WD/${k}g.o"
done
$CXX -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -DDECODE_ONLY -c $SRC/iqtq_gemm_scalar_driver.cpp -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -20 $WD/cc_drv.err; exit 4; }
$CXX $WD/drv.o $OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/bench 2>$WD/ld.err || { echo LINK_FAIL; head -20 $WD/ld.err; exit 5; }
echo "# LINK OK" | tee -a "$SEAL"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4; BESTC=0; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC; echo "# PIN core=$CORE idle=${BESTIDLE}%" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/bench "$@" 2>>"$LOG" | tee -a "$LOG"; }
echo "=== DECODE VERIFY (K=2048 nc=64 gevm) ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 64 1 1 0x1357 gevm; done
echo "=== DECODE COLD K=2048 nc=512 reps=25 seedA gevm ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 512 25 0 0xC0FFEE1 gevm; done
echo "=== DECODE COLD seedB gevm ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 512 25 0 0x1357ACE gevm; done
echo "# stock_after=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) STRAY=$(pgrep -c -x bench||echo 0) ALL_DONE" | tee -a "$LOG"
