#!/usr/bin/env bash
# A2-batch6 iq/tq/fp4 GEMM scalar-ref @rvv (VLEN128). DUAL-DOMAIN:
#   DOMAIN=gcc  : ours gcc-15.2  vs opp generic in gcc-15 stock .so = deploy-clean MAIN (symmetric)
#   DOMAIN=clang: ours clang-18  vs opp generic in gcc-15 stock .so = clang-micro footnote (asym)
set -uo pipefail
DOMAIN=${DOMAIN:-gcc}
WD=/tmp/g8_a2b6_rvv_$DOMAIN; SRC=/tmp/g8_a2b6_rvv/src
GCC=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
CLANG=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
GTOOL=/opt/tcrv-toolchains/gcc-15.2.0
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log
mkdir -p $WD; cd $WD; : > "$SEAL"; : > "$LOG"
if [ "$DOMAIN" = "gcc" ]; then CXX="$GCC"; CXXFLAGS="-O3 -march=$MARCH -mabi=lp64d"; LINK="$GCC";
else CXX="$CLANG"; CXXFLAGS="-O3 -march=$MARCH -mabi=lp64d --gcc-toolchain=$GTOOL"; LINK="$GCC"; fi
echo "# A2b6 iqtq-GEMM-scalar @rvv DOMAIN=$DOMAIN CXX=$($CXX --version|head -1) march=$MARCH $(date -u)" | tee -a "$SEAL"
echo "# stock_so=$GGML md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
FMTS="iq4_xs iq2_xxs iq2_xs iq2_s mxfp4 tq1_0 tq2_0"
for s in $FMTS; do case $s in mxfp4) sym=ggml_vec_dot_mxfp4_q8_0_generic;; *) sym=ggml_vec_dot_${s}_q8_K_generic;; esac
  a=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ${sym}\$" | awk '{print $1}' | head -1)
  echo "# OPP-generic $sym @0x${a:-MISSING}" | tee -a "$SEAL"; done
OBJS=""
for k in $FMTS; do
  $CXX $CXXFLAGS -include math.h -c $SRC/kernels/${k}_gemm.c -o $WD/${k}.o 2>$WD/cc_${k}.err \
    || { echo "KERN_FAIL $k"; head -12 $WD/cc_${k}.err; exit 3; }
  VS=$(objdump -d $WD/${k}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  echo "# OURS_${k}: vsetvl=$VS comment=$(readelf -p .comment $WD/${k}.o 2>/dev/null | grep -oE '(GCC|clang)[^]]*' | head -1)" | tee -a "$SEAL"
  OBJS="$OBJS $WD/${k}.o"
done
$CXX $CXXFLAGS -c $SRC/iqtq_gemm_scalar_driver.cpp -o $WD/drv.o 2>$WD/cc_drv.err || { echo DRV_FAIL; head -20 $WD/cc_drv.err; exit 4; }
$LINK $WD/drv.o $OBJS -L$GTOOL/lib -Wl,-rpath,$GTOOL/lib -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/bench 2>$WD/ld.err || { echo LINK_FAIL; head -20 $WD/ld.err; exit 5; }
echo "# LINK OK -> $WD/bench" | tee -a "$SEAL"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4; BESTC=8; BESTIDLE=-1
for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]})); pct=$(( dt>0?100*di/dt:0 )); [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/bench "$@" 2>>"$LOG" | tee -a "$LOG"; }
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "=== VERIFY (K=2048 nr=4 nc=64) DOMAIN=$DOMAIN ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 4 64 1 1 0x1357; done
if [ "${VERIFY_ONLY:-0}" = "1" ]; then echo "# VERIFY_ONLY done"; exit 0; fi
echo "=== COLD K=2048 nr=16 nc=512 reps=25 seedA ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 16 512 25 0 0xC0FFEE1; done
echo "=== COLD 2nd-seed seedB ===" | tee -a "$LOG"
for f in $FMTS; do run $f 2048 16 512 25 0 0x1357ACE; done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -x bench || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE DOMAIN=$DOMAIN" | tee -a "$LOG"
