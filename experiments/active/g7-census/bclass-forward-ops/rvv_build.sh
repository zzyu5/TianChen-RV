#!/usr/bin/env bash
# G7 L1 B-class forward-op kernel-sym — build 9 ours + ggml opponents @rvv (gcc-15.2 symmetric).
# [CASE-COMPILER-ASYMMETRY] rvv shipped compiler = gcc-15 => use gcc-15 both sides.
set -u
RUN=/tmp/g7_bfwd_rvv
GGML_DIR=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin     # stock gcc-15 lib (silu/softmax exported opponents)
MARCH="rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause"  # == stock ggml-cpu gcc15 build march
KCC=/opt/tcrv-toolchains/gcc-15.2.0/bin/riscv64-unknown-linux-gnu-g++
TCLIB=/opt/tcrv-toolchains/gcc-15.2.0/lib                          # gcc-15 libstdc++/libgcc_s (binutils ld needs explicit path)
OBJDUMP=/opt/tcrv-toolchains/binutils-2.46.1/bin/objdump
OPT="${1:-O3}"
KCCV=$($KCC --version | head -1)
mkdir -p "$RUN"
SEAL="$RUN/build_seal.txt"; : > "$SEAL"
echo "# BUILD rvv B-fwd kernel-sym  KCC=$KCC ($KCCV)  march=$MARCH  opt=-$OPT" | tee -a "$SEAL"
echo "# ggml_stock=$GGML_DIR/libggml-cpu.so md5_before=$(md5sum $GGML_DIR/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

OPS="add mul scale cpy silu gelu rms-norm soft-max rope"
OBJS=""
for op in $OPS; do
  SRC="$RUN/kernels/$op.kernel.c"; O="$RUN/$op.rvv.o"
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
  -L"$TCLIB" -Wl,-rpath,"$TCLIB" \
  -L"$GGML_DIR" -Wl,-rpath,"$GGML_DIR" -lggml-cpu -lggml-base -lggml -lm -o "$RUN/bfwd_rvv" 2>"$RUN/cc_link.err" \
  || { echo "LINK_FAIL"; sed -n '1,16p' "$RUN/cc_link.err"; exit 4; }
echo "# LINKED $RUN/bfwd_rvv" | tee -a "$SEAL"

# --- machine-probe opponent symbol identity (which kernel does the linker resolve for silu/softmax?) ---
for sy in ggml_vec_silu_f32 ggml_vec_soft_max_f32; do
  ADDR=$(nm -D "$GGML_DIR/libggml-cpu.so" 2>/dev/null | grep -E " (T|t|W) ${sy}$" | head -1)
  echo "# OPP_SYM_EXPORTED $sy: ${ADDR:-ABSENT}" | tee -a "$SEAL"
done
# confirm inline opponents (add/mul/scale/cpy/gelu) are NOT exported (=> compiled from ggml source, correct)
echo "# OPP_INLINE_NOT_EXPORTED(expect all ABSENT): $(for s in ggml_vec_add_f32 ggml_vec_mul_f32 ggml_vec_scale_f32 ggml_vec_cpy_f32 ggml_vec_gelu_f32; do nm -D $GGML_DIR/libggml-cpu.so 2>/dev/null | grep -qE " T ${s}$" && echo "$s=PRESENT" || echo "$s=absent"; done)" | tee -a "$SEAL"

# --- opponent-class machine-judgment: objdump the compiled opponent .o (RVV vs scalar/autovec) + vtype LMUL ---
echo "## OPP-CLASS (objdump opp.o per-symbol RVV insn count + vtype LMUL)" | tee -a "$SEAL"
for sy in opp_ggml_vec_add_f32 opp_ggml_vec_mul_f32 opp_ggml_vec_cpy_f32 opp_ggml_vec_scale_f32 opp_ggml_vec_gelu_f32 opp_ggml_rms_norm_f32 opp_ggml_rope_norm_f32; do
  BODY=$($OBJDUMP -d --disassemble="$sy" "$RUN/opp.o" 2>/dev/null)
  VL=$(echo "$BODY" | grep -cE '\bvle|\bvse|\bvfmul|\bvfmacc|\bvfadd|\bvsetvl|\bvfnmsac')
  CLASS=$( [ "$VL" -gt 0 ] && echo native-RVV || echo scalar-or-autovec )
  VTYPE=$(echo "$BODY" | grep -oE 'e(8|16|32|64),m(f?[0-9]+)' | sort | uniq -c | tr '\n' ' ')
  echo "# OPPCLASS $sy: rvv_insn=$VL => $CLASS  vtype[$VTYPE]" | tee -a "$SEAL"
done
# our-kernel RVV width probe (LMUL via vtype comma-format) + vsetvl count
for op in add mul scale cpy silu gelu rms-norm soft-max rope; do
  KOBJ="$RUN/$op.rvv.o"
  M8=$($OBJDUMP -d "$KOBJ" 2>/dev/null | grep -cE 'e32,m8'); M2=$($OBJDUMP -d "$KOBJ" 2>/dev/null | grep -cE 'e32,m2'); M1=$($OBJDUMP -d "$KOBJ" 2>/dev/null | grep -cE 'e32,m1\b')
  VS=$($OBJDUMP -d "$KOBJ" 2>/dev/null | grep -c vsetvl)
  echo "# OURS_$op(gcc-15 .o): vsetvl=$VS e32m8=$M8 e32m2=$M2 e32m1=$M1 size=$(wc -c<$KOBJ)B" | tee -a "$SEAL"
done
echo "# BUILD_DONE" | tee -a "$SEAL"
