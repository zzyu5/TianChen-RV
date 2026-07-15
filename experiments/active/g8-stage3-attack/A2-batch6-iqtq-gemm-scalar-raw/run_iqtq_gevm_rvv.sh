#!/usr/bin/env bash
# A2-batch6 iq/tq/fp4 DECODE (M=1 GEVM) scalar-ref @rvv. DOMAIN=gcc(deploy MAIN)|clang(micro).
set -uo pipefail
DOMAIN=${DOMAIN:-gcc}
WD=/tmp/g8_a2b6g_rvv_$DOMAIN; SRC=/tmp/g8_a2b6g_rvv/src
GCC=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++; CLANG=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
GTOOL=/opt/tcrv-toolchains/gcc-15.2.0; MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log; mkdir -p $WD; cd $WD; : > "$SEAL"; : > "$LOG"
if [ "$DOMAIN" = "gcc" ]; then CXX="$GCC"; CXXFLAGS="-O3 -march=$MARCH -mabi=lp64d"; else CXX="$CLANG"; CXXFLAGS="-O3 -march=$MARCH -mabi=lp64d --gcc-toolchain=$GTOOL"; fi
echo "# A2b6-DECODE @rvv DOMAIN=$DOMAIN $($CXX --version|head -1) $(date -u) stock_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
FMTS="iq4_xs iq2_xxs iq2_xs iq2_s mxfp4 tq1_0 tq2_0"
OBJS=""
for k in $FMTS; do
  $CXX $CXXFLAGS -include math.h -c $SRC/kernels/${k}_gevm.c -o $WD/${k}g.o 2>$WD/cc_${k}.err || { echo "KERN_FAIL $k"; head -12 $WD/cc_${k}.err; exit 3; }
  VS=$(objdump -d $WD/${k}g.o 2>/dev/null | grep -cE 'vsetvli|vsetivli'); echo "# OURS_gevm_${k}: vsetvl=$VS" | tee -a "$SEAL"; OBJS="$OBJS $WD/${k}g.o"
done
$CXX $CXXFLAGS -DDECODE_ONLY -c $SRC/iqtq_gemm_scalar_driver.cpp -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -20 $WD/cc_drv.err; exit 4; }
$GCC $WD/drv.o $OBJS -L$GTOOL/lib -Wl,-rpath,$GTOOL/lib -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/bench 2>$WD/ld.err || { echo LINK_FAIL; head -20 $WD/ld.err; exit 5; }
echo "# LINK OK" | tee -a "$SEAL"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4; BESTC=8; BESTIDLE=-1
for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC; echo "# PIN core=$CORE idle=${BESTIDLE}%" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/bench "$@" 2>>"$LOG" | tee -a "$LOG"; }
echo "=== DECODE VERIFY DOMAIN=$DOMAIN ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 64 1 1 0x1357 gevm; done
echo "=== DECODE COLD seedA gevm ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 512 25 0 0xC0FFEE1 gevm; done
echo "=== DECODE COLD seedB gevm ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 512 25 0 0x1357ACE gevm; done
echo "# stock_after=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) STRAY=$(pgrep -c -x bench||echo 0) ALL_DONE DOMAIN=$DOMAIN" | tee -a "$LOG"
