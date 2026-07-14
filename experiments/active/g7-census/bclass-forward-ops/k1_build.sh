#!/usr/bin/env bash
# G7 L1 B-class forward-op kernel-sym — build 9 ours + ggml opponents @k1 (clang-18 symmetric).
set -u
RUN=/tmp/g7_bfwd_k1
GGML_DIR=/data/k1build-stock/bin                 # stock clang-18 lib (silu/softmax exported opponents)
MARCH="rv64gcv_zfh_zvfh_zicbop_zihintpause"      # == stock ggml-cpu compile_commands march
KCC=/usr/bin/clang-18
OPT="${1:-O3}"
KCCV=$($KCC --version | head -1)
mkdir -p "$RUN"
SEAL="$RUN/build_seal.txt"; : > "$SEAL"
echo "# BUILD k1 B-fwd kernel-sym  KCC=$KCC ($KCCV)  march=$MARCH  opt=-$OPT" | tee -a "$SEAL"
echo "# ggml_stock=$GGML_DIR/libggml-cpu.so md5_before=$(md5sum $GGML_DIR/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

OPS="add mul scale cpy silu gelu rms-norm soft-max rope"
OBJS=""
for op in $OPS; do
  SRC="$RUN/kernels/$op.kernel.c"; O="$RUN/$op.k1.o"
  $KCC -$OPT -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$SRC" -c -o "$O" 2>"$RUN/cc_$op.err" \
    || { echo "KERN_FAIL $op"; sed -n '1,8p' "$RUN/cc_$op.err"; exit 3; }
  OBJS="$OBJS $O"
  echo "# OURS_SRC $op md5=$(md5sum $SRC|cut -d' ' -f1)" | tee -a "$SEAL"
done

# opponent verbatim ggml source
$KCC -$OPT -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/opponent_ggml.cpp" -c -o "$RUN/opp.o" 2>"$RUN/cc_opp.err" \
  || { echo "OPP_FAIL"; sed -n '1,12p' "$RUN/cc_opp.err"; exit 3; }
echo "# OPP_SRC opponent_ggml.cpp md5=$(md5sum $RUN/opponent_ggml.cpp|cut -d' ' -f1)" | tee -a "$SEAL"

# driver -> object (compile as C++), then link all objects + stock ggml (silu/softmax exported opponents)
$KCC -$OPT -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$RUN/bfwd_micro_driver.c" -c -o "$RUN/drv.o" 2>"$RUN/cc_drv.err" \
  || { echo "DRV_CC_FAIL"; sed -n '1,16p' "$RUN/cc_drv.err"; exit 4; }
$KCC -$OPT -march=$MARCH -mabi=lp64d -ffp-contract=on "$RUN/drv.o" $OBJS "$RUN/opp.o" \
  -L"$GGML_DIR" -Wl,-rpath,"$GGML_DIR" -lggml-cpu -lggml-base -lggml -lm -o "$RUN/bfwd_k1" 2>"$RUN/cc_link.err" \
  || { echo "LINK_FAIL"; sed -n '1,16p' "$RUN/cc_link.err"; exit 4; }
echo "# LINKED $RUN/bfwd_k1" | tee -a "$SEAL"

# --- machine-probe opponent symbol identity (which kernel does the linker resolve for silu/softmax?) ---
for sy in ggml_vec_silu_f32 ggml_vec_soft_max_f32; do
  ADDR=$(nm -D "$GGML_DIR/libggml-cpu.so" 2>/dev/null | grep -E " (T|t|W) ${sy}$" | head -1)
  echo "# OPP_SYM_EXPORTED $sy: ${ADDR:-ABSENT}" | tee -a "$SEAL"
done
# confirm inline opponents (add/mul/scale/cpy/gelu) are NOT exported (=> compiled from ggml source, correct)
echo "# OPP_INLINE_NOT_EXPORTED(expect all ABSENT): $(for s in ggml_vec_add_f32 ggml_vec_mul_f32 ggml_vec_scale_f32 ggml_vec_cpy_f32 ggml_vec_gelu_f32; do nm -D $GGML_DIR/libggml-cpu.so 2>/dev/null | grep -qE " T ${s}$" && echo "$s=PRESENT" || echo "$s=absent"; done)" | tee -a "$SEAL"

# --- opponent-class machine-judgment: objdump the compiled opponent .o (RVV vs scalar/autovec) ---
echo "## OPP-CLASS (objdump opp.o per-symbol RVV insn count)" | tee -a "$SEAL"
for sy in opp_ggml_vec_add_f32 opp_ggml_vec_mul_f32 opp_ggml_vec_cpy_f32 opp_ggml_vec_scale_f32 opp_ggml_vec_gelu_f32 opp_ggml_rms_norm_f32 opp_ggml_rope_norm_f32; do
  BODY=$(objdump -d --disassemble="$sy" "$RUN/opp.o" 2>/dev/null)
  VL=$(echo "$BODY" | grep -cE '\bvle|\bvse|\bvfmul|\bvfmacc|\bvfadd|\bvsetvl|\bvfnmsac')
  CLASS=$( [ "$VL" -gt 0 ] && echo native-RVV || echo scalar-or-autovec )
  echo "# OPPCLASS $sy: rvv_insn=$VL => $CLASS" | tee -a "$SEAL"
done
# our-kernel RVV width probe (LMUL) + softfp
for op in add mul scale cpy silu gelu rms-norm soft-max rope; do
  KOBJ="$RUN/$op.k1.o"
  M8=$(objdump -d "$KOBJ" 2>/dev/null | grep -cE 'e32m8|f32m8'); M2=$(objdump -d "$KOBJ" 2>/dev/null | grep -cE 'e32m2|f32m2')
  VS=$(objdump -d "$KOBJ" 2>/dev/null | grep -c vsetvl)
  echo "# OURS_$op(clang-18 .o): vsetvl=$VS m8_ops=$M8 m2_ops=$M2 size=$(wc -c<$KOBJ)B" | tee -a "$SEAL"
done
echo "# BUILD_DONE" | tee -a "$SEAL"
