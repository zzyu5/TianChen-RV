#!/usr/bin/env bash
# A2-batch5 K-quant M=1 GEVM decode @rvv(VLEN128). ours=repack-GEVM leaf (hl8, front-door product).
# Dual-domain: OUR leaf built BOTH clang-18.1.8 (symmetric-micro) AND gcc-15.2 (deploy-clean, rvv ships gcc-15).
# opp = gcc-15 stock libggml-cpu.so (ggml_vec_dot_qX_K_q8_K block-dot, compiler-insensitive).
set -uo pipefail
source /opt/tcrv-toolchains/env.sh
WD=/tmp/g8_a2b5_rvv
CLANG=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
GCC=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
SEAL=$WD/build_seal.txt; LOG=$WD/run.log
mkdir -p $WD; cd $WD; : > "$SEAL"; : > "$LOG"
echo "# A2b5 KQUANT-GEVM-M1 @rvv(VLEN128) march=$MARCH -O3 $(date -u)" | tee -a "$SEAL"
echo "# CLANG=$($CLANG --version|head -1)  GCC=$($GCC --version|head -1)" | tee -a "$SEAL"
echo "# stock_so_md5_before=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
FMTS="q2_K q3_K q5_K q6_K"
build_domain(){  # $1=tag  $2=CXX  $3=extraflags
  local tag=$1 CXX=$2 XF=$3; declare -A O
  for f in q2 q3 q5 q6; do
    $CXX -O3 -march=$MARCH -mabi=lp64d $XF -include math.h -c $WD/leaves/${f}_K_gevm.c -o $WD/${tag}_${f}.o 2>$WD/cc_${tag}_${f}.err || { echo "KERN_FAIL $tag $f"; head -8 $WD/cc_${tag}_${f}.err; return 3; }
    local VS=$(objdump -d $WD/${tag}_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    local SFP=$(objdump -d $WD/${tag}_${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2' && echo SOFTFP || echo cleanfp)
    echo "# OURS[$tag]_${f}_K: fp16=$SFP vsetvl=$VS" | tee -a "$SEAL"; O[$f]=$WD/${tag}_${f}.o
  done
  $CXX -O2 -march=$MARCH -mabi=lp64d $XF -std=c++17 -c $WD/kquant_gevm_m1_driver_b5.cpp -o $WD/${tag}_drv.o 2>$WD/cc_${tag}_drv.err || { echo "DRV_FAIL $tag"; head -20 $WD/cc_${tag}_drv.err; return 4; }
  $CXX $XF -march=$MARCH -mabi=lp64d $WD/${tag}_drv.o ${O[q2]} ${O[q3]} ${O[q5]} ${O[q6]} \
     -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $WD/bench_${tag} 2>$WD/ld_${tag}.err || { echo "LINK_FAIL $tag"; head -20 $WD/ld_${tag}.err; return 5; }
  echo "# LINK OK[$tag] -> $WD/bench_${tag}" | tee -a "$SEAL"
}
build_domain clang "$CLANG" "-fno-integrated-as --gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0" || exit $?
build_domain gcc   "$GCC"   "" || exit $?
for s in ggml_vec_dot_q2_K_q8_K ggml_vec_dot_q3_K_q8_K ggml_vec_dot_q5_K_q8_K ggml_vec_dot_q6_K_q8_K; do
  a=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ${s}\$" | awk '{print $1}'); echo "# OPP $s @0x${a:-?} (gcc-15 stock .so)" | tee -a "$SEAL"; done
CORE=32
echo "# PIN core=$CORE gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
run(){ local tag=$1; shift; LD_LIBRARY_PATH=$GGML taskset -c $CORE $WD/bench_${tag} "$@" 2>>"$LOG" | sed "s/^/[$tag] /" | tee -a "$LOG"; }
echo "=== VERIFY (nc=64) ===" | tee -a "$LOG"
for tag in clang gcc; do for f in $FMTS; do run $tag $f 2048 64 12 1; done; done
echo "=== COLD PRIMARY K=2048 N=512 reps=25 seedA=0xC0FFEE1 (x2) ===" | tee -a "$LOG"
for tag in clang gcc; do for f in $FMTS; do for t in 1 2; do run $tag $f 2048 512 25 0 0xC0FFEE1; done; done; done
echo "=== COLD 2nd-seed K=2048 N=512 reps=25 seedB=0x1357ACE (x2) ===" | tee -a "$LOG"
for tag in clang gcc; do for f in $FMTS; do for t in 1 2; do run $tag $f 2048 512 25 0 0x1357ACE; done; done; done
echo "# stock_so_md5_after=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -f a2b5_rvv/bench || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
