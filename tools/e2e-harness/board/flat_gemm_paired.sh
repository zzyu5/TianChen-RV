#!/usr/bin/env bash
# flat_gemm_paired.sh — board compile+run for the FLAT repack-vs-blockdot paired A/B.
# Usage: flat_gemm_paired.sh <cc=clang|gcc> <fmt> <K> <nr> <nc> <iters> <reps> <core>
# Compiles OUR exported kernel + driver with <cc>, links the board's own libggml-cpu.so (gcc-15),
# pins to <core>, and runs the driver <reps> times (each does its own best-of-7 + numeric gate).
# Prints one FLATGEMM line per rep. Compiler identity + march are disclosed on the first line.
set -u
CC_ID="$1"; FMT="$2"; K="$3"; NR="$4"; NC="$5"; ITERS="$6"; REPS="$7"; CORE="${8:-8}"
RUN=/tmp/flat_export
GGML_DIR=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
MARCH="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs"

source /opt/tcrv-toolchains/env.sh 2>/dev/null
# KERNEL codegen compiler VARIES (the measured variable); driver+link ALWAYS gcc-15 (neutral harness,
# same as the gcc-15 that built the opponent .so) so only OUR-kernel codegen differs between accounts.
GCC=$(command -v riscv64-unknown-linux-gnu-gcc || command -v gcc)
if [ "$CC_ID" = "clang" ]; then KCC=$(command -v clang);
elif [ "$CC_ID" = "gcc" ]; then KCC=$GCC;
else echo "bad cc"; exit 2; fi
KCCV=$($KCC --version|head -1); GCCV=$($GCC --version|head -1)

BIN=$RUN/flatdrv.${CC_ID}
OBJS=""
# compile ALL 5 OUR kernels with the account compiler (driver switch references all symbols)
for ff in q4_0 q4_1 q5_0 q5_1 q8_0; do
  O=$RUN/gemm_${ff}.${CC_ID}.o
  $KCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/gemm_${ff}.kernel.c" -c -o "$O" 2>$RUN/cc_kern_${ff}_${CC_ID}.err || { echo "KERN_COMPILE_FAIL $ff $CC_ID"; sed -n '1,8p' $RUN/cc_kern_${ff}_${CC_ID}.err; exit 3; }
  OBJS="$OBJS $O"
done
$GCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on "$RUN/flat_gemm_paired_driver.c" $OBJS \
    -L"$GGML_DIR" -Wl,-rpath,"$GGML_DIR" -lggml-cpu -lggml-base -lggml -lm -o "$BIN" 2>$RUN/cc_drv_${CC_ID}.err \
    || { echo "DRV_COMPILE_FAIL $CC_ID"; sed -n '1,12p' $RUN/cc_drv_${CC_ID}.err; exit 4; }

KOBJ=$RUN/gemm_${FMT}.${CC_ID}.o
echo "# KERNEL_CC=$CC_ID ($KCCV) LINK_CC=gcc ($GCCV) march=$MARCH ggml=$GGML_DIR(gcc-15) core=$CORE ${FMT}_obj=$(wc -c <$KOBJ)B"
objdump -d "$KOBJ" 2>/dev/null | grep -Eq '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo "# OUR_${FMT} SOFT-FP16-LIBCALL (confound!)" || echo "# OUR_${FMT} fp16-libcall-free OK"
OBJ=$KOBJ
KOBJ_VSETVL=$(objdump -d "$OBJ" 2>/dev/null | grep -c vsetvl)
echo "# OUR_KERNEL vsetvl_count=$KOBJ_VSETVL"
for i in $(seq 1 "$REPS"); do
  taskset -c "$CORE" "$BIN" "$FMT" "$K" "$NR" "$NC" "$ITERS" $((0x1234+i*7))
done
