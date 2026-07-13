#!/usr/bin/env bash
# G7 L1 货架A — FLAT-5 kernel-sym hot/cold + nr-shape variant sweep @rvv/VLEN128 (gcc-15 symmetric).
# Usage: run_rvv_cold.sh <cc=gcc|clang> <core> <pool> <rounds> <hot_iters>
set -u
CC_ID="${1:-gcc}"; CORE="${2:-8}"; POOL="${3:-200}"; ROUNDS="${4:-12}"; HITERS="${5:-8}"
RUN=/tmp/g7_l1_cold
GGML_DIR=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
MARCH="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs"
source /opt/tcrv-toolchains/env.sh 2>/dev/null
GCC=$(command -v riscv64-unknown-linux-gnu-gcc || command -v gcc)
if [ "$CC_ID" = "clang" ]; then KCC=$(command -v clang); else KCC=$GCC; fi
KCCV=$($KCC --version|head -1); GCCV=$($GCC --version|head -1)
BIN=$RUN/flatcold.${CC_ID}
OBJS=""
for ff in q4_0 q4_1 q5_0 q5_1 q8_0; do
  O=$RUN/gemm_${ff}.${CC_ID}.o
  $KCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/gemm_${ff}.kernel.c" -c -o "$O" 2>$RUN/cc_${ff}.err || { echo "KERN_FAIL $ff"; sed -n '1,6p' $RUN/cc_${ff}.err; exit 3; }
  OBJS="$OBJS $O"
done
$GCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on "$RUN/flat_gemm_cold_driver.c" $OBJS \
  -L"$GGML_DIR" -Wl,-rpath,"$GGML_DIR" -lggml-cpu -lggml-base -lggml -lm -o "$BIN" 2>$RUN/cc_drv.err \
  || { echo "DRV_FAIL"; sed -n '1,12p' $RUN/cc_drv.err; exit 4; }
echo "# KERNEL_CC=$CC_ID ($KCCV) LINK_CC=gcc ($GCCV) march=$MARCH ggml=$GGML_DIR(gcc-15) core=$CORE pool=$POOL rounds=$ROUNDS"
echo "# ggml_so_md5=$(md5sum $GGML_DIR/libggml-cpu.so | cut -d' ' -f1)"
# --- machine-probe opponent symbol identity (which kernel does the linker resolve?) ---
for sy in q4_0_q8_0 q4_1_q8_1 q5_0_q8_0 q5_1_q8_1 q8_0_q8_0; do
  ADDR=$(nm -D "$GGML_DIR/libggml-cpu.so" 2>/dev/null | grep -E " (T|t|W) ggml_vec_dot_${sy}$" | head -1)
  echo "# OPP_SYM ggml_vec_dot_${sy}: ${ADDR:-ABSENT}"
done
# per-kernel our vsetvl + fp16-libcall probe (gcc-15 .o)
for ff in q4_0 q4_1 q5_0 q5_1 q8_0; do
  KOBJ=$RUN/gemm_${ff}.${CC_ID}.o
  VS=$(objdump -d "$KOBJ" 2>/dev/null | grep -c vsetvl)
  RV=$(objdump -d "$KOBJ" 2>/dev/null | grep -cE 'vle|vse|vmacc|vwmacc|vadd|vfmacc|vmv|vfcvt|vsetvl')
  LC=$(objdump -d "$KOBJ" 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2' && echo SOFTFP || echo cleanfp)
  echo "# OUR_${ff}(gcc-15 .o): vsetvl=$VS rvv_insn~=$RV fp16=$LC size=$(wc -c<$KOBJ)B"
done
echo "=== RUNS (hot/cold; nr-shape variant sweep) ==="
K=2048; NC=512
for FMT in q4_0 q4_1 q5_0 q5_1 q8_0; do
  for NR in 4 16 64; do
    SEED=$((0x1234 + NR*7))
    echo "## $FMT nr=$NR"
    taskset -c "$CORE" "$BIN" "$FMT" "$K" "$NR" "$NC" "$HITERS" "$POOL" "$ROUNDS" "$SEED"
  done
done
echo "=== DONE ==="
