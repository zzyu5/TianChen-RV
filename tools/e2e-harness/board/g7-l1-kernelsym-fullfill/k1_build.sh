#!/usr/bin/env bash
# G7 L1 货架A k1-半 kernel-sym — build q4_1/q5_0/q5_1 VLEN256 vl=16 ours (clang-18 symmetric) + probes.
set -u
RUN=/tmp/g7_l1_k1cold
GGML_DIR=/data/k1build-stock/bin           # stock clang-18 lib (block-dot opponents; ZERO qX repack)
MARCH="rv64gcv_zfh_zvfh_zicbop_zihintpause" # == stock ggml-cpu compile_commands march
KCC=/usr/bin/clang-18
OPT="${1:-O3}"                              # symmetric w/ stock -O3
KCCV=$($KCC --version | head -1)
mkdir -p "$RUN"
echo "# BUILD k1 kernel-sym FLAT q4_1/q5_0/q5_1  KCC=$KCC ($KCCV)  march=$MARCH  opt=-$OPT" | tee "$RUN/build_seal.txt"
echo "# ggml_stock=$GGML_DIR/libggml-cpu.so md5_before=$(md5sum $GGML_DIR/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$RUN/build_seal.txt"

OBJS=""
for ff in q4_1 q5_0 q5_1; do
  SRC="$RUN/weft_emitted_gemm_${ff}.inc"
  O="$RUN/gemm_${ff}.k1.o"
  # .inc is a self-contained extern "C" C++ TU (vl=16). Compile as C++.
  $KCC -$OPT -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$SRC" -c -o "$O" 2>"$RUN/cc_${ff}.err" \
    || { echo "KERN_FAIL $ff"; sed -n '1,8p' "$RUN/cc_${ff}.err"; exit 3; }
  OBJS="$OBJS $O"
  # md5 double-proof of ours kernel source (unchanged from casefile)
  echo "# OURS_SRC $ff md5=$(md5sum $SRC|cut -d' ' -f1)" | tee -a "$RUN/build_seal.txt"
done

# link driver (C) with kernels + stock ggml block-dot (opponent). link cc = clang-18 (symmetric).
$KCC -$OPT -march=$MARCH -mabi=lp64d -ffp-contract=on "$RUN/flat_gemm_cold_driver_k1.c" $OBJS \
  -L"$GGML_DIR" -Wl,-rpath,"$GGML_DIR" -lggml-cpu -lggml-base -lggml -lm -o "$RUN/flatcold_k1" 2>"$RUN/cc_drv.err" \
  || { echo "DRV_FAIL"; sed -n '1,14p' "$RUN/cc_drv.err"; exit 4; }
echo "# LINKED $RUN/flatcold_k1" | tee -a "$RUN/build_seal.txt"

# --- machine-probe opponent symbol identity (which kernel does the linker resolve?) ---
for sy in q4_1_q8_1 q5_0_q8_0 q5_1_q8_1; do
  ADDR=$(nm -D "$GGML_DIR/libggml-cpu.so" 2>/dev/null | grep -E " (T|t|W) ggml_vec_dot_${sy}$" | head -1)
  echo "# OPP_SYM ggml_vec_dot_${sy}: ${ADDR:-ABSENT}" | tee -a "$RUN/build_seal.txt"
done
# confirm opponent has NO qX repack gemm (single-implementation compromise 折中态)
echo "# OPP_REPACK_SYMS(q4_1/q5_0/q5_1 gemm; expect none): $(nm -D $GGML_DIR/libggml-cpu.so 2>/dev/null | grep -icE 'gemm_q4_1|gemm_q5_0|gemm_q5_1')" | tee -a "$RUN/build_seal.txt"

# --- objdump structural seal on ours (部署编译器 spill 轴·G1 精化②) + fp16-libcall + AVL width ---
for ff in q4_1 q5_0 q5_1; do
  KOBJ="$RUN/gemm_${ff}.k1.o"
  VS=$(objdump -d "$KOBJ" 2>/dev/null | grep -c vsetvl)
  VW=$(objdump -d "$KOBJ" 2>/dev/null | grep -cE 'vwmacc')
  SPILL=$(objdump -d "$KOBJ" 2>/dev/null | grep -cE '(sd|ld)[[:space:]].*sp\)|addi[[:space:]]+sp,sp,-')
  LC=$(objdump -d "$KOBJ" 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__extendhfsf|__trunchfsf' && echo SOFTFP || echo cleanfp-native-zfh)
  SZ=$(wc -c < "$KOBJ")
  echo "# OURS_$ff(clang-18 -$OPT .o): vsetvl=$VS vwmacc=$VW spillish=$SPILL fp16=$LC size=${SZ}B" | tee -a "$RUN/build_seal.txt"
done
echo "# BUILD_DONE" | tee -a "$RUN/build_seal.txt"
