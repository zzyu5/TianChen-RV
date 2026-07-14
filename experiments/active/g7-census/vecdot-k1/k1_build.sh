#!/usr/bin/env bash
# k1_build.sh — G7 L1 vec_dot@k1 census: build ours (clang-18 symmetric) + machine-probe opponents,
# link against stock libggml-cpu.so. Runs ON k1 (invoke via ssh). No git, main tree untouched.
set -uo pipefail
RDIR=/tmp/g7_vecdot_k1
MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause      # == stock ggml-cpu compile march
GGML=/data/k1build-stock/bin
CC=clang-18
SEAL=$RDIR/build_seal.txt
cd $RDIR
: > "$SEAL"
echo "# BUILD vec_dot@k1  CC=$($CC --version|head -1)  march=$MARCH" | tee -a "$SEAL"
echo "# board=$(uname -srm) vlenb=$(cat /sys/devices/system/cpu/cpu0/... 2>/dev/null)" >> "$SEAL"
echo "# ggml_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

FMTS="q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K"
OBJS=""
echo "=== [A] compile ours kernels (clang-18 -O3 symmetric) + objdump self-probe ===" | tee -a "$SEAL"
for f in $FMTS; do
  $CC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/${f}.kernel.c -c -o k_${f}.o 2>cc_${f}.err || { echo "KERN_FAIL $f"; head -6 cc_${f}.err; exit 3; }
  SOFTFP=$(objdump -d k_${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h' && echo SOFTFP || echo cleanfp)
  VS=$(objdump -d k_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  VW=$(objdump -d k_${f}.o 2>/dev/null | grep -cE 'vwmacc|vwmul|vwredsum|vmadot')
  MV=$(objdump -d k_${f}.o 2>/dev/null | grep -oE 'v[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
  echo "# OURS_${f}(clang18 .o): fp16=$SOFTFP vsetvl=$VS widen=$VW maxVreg=v${MV:-?} size=$(wc -c<k_${f}.o)B" | tee -a "$SEAL"
  OBJS="$OBJS k_${f}.o"
done

echo "=== [B] opponent symbol machine-probe (as-shipped dispatched kernel class) ===" | tee -a "$SEAL"
declare -A OPP=( [q4_0]=q4_0_q8_0 [q4_1]=q4_1_q8_1 [q5_0]=q5_0_q8_0 [q5_1]=q5_1_q8_1 [q8_0]=q8_0_q8_0 \
  [q2_K]=q2_K_q8_K [q3_K]=q3_K_q8_K [q4_K]=q4_K_q8_K [q5_K]=q5_K_q8_K [q6_K]=q6_K_q8_K )
# dump the whole lib once for objdump-by-address
objdump -d $GGML/libggml-cpu.so > lib_disasm.txt 2>/dev/null
for f in $FMTS; do
  sym=ggml_vec_dot_${OPP[$f]}
  addr=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ${sym}\$" | head -1 | awk '{print $1}')
  # sibling specializations present?
  sibs=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -oE "${sym}(_vl128|_vl256|_generic)" | tr '\n' ',')
  # objdump the main symbol body (from its label to next label) : count rvv insn + calls
  body=$(awk -v s="<${sym}>:" 'f&&/^[0-9a-f]+ </{exit} $0 ~ s{f=1} f' lib_disasm.txt)
  nins=$(echo "$body" | grep -cE '\t[a-z]')
  nrvv=$(echo "$body" | grep -cE '\tv[a-z]')
  vl256=$(echo "$body" | grep -cE 'vl256')
  vl128=$(echo "$body" | grep -cE 'vl128')
  calls=$(echo "$body" | grep -oE 'jal|call|<[a-z_]+q8[_A-Za-z0-9]*>' | tr '\n' ',' | head -c 80)
  tailcall=$(echo "$body" | grep -oE '<ggml_vec_dot_[a-z0-9_]+>' | sort -u | tr '\n' ',')
  echo "# OPP_${f}: sym=$sym @${addr:-ABSENT} sibs=[${sibs:-none}] ins=$nins rvv=$nrvv vl256hits=$vl256 vl128hits=$vl128 tailcalls=[${tailcall:-none}]" | tee -a "$SEAL"
done

echo "=== [C] compile driver + link ours + stock libggml-cpu.so ===" | tee -a "$SEAL"
$CC -O2 -march=$MARCH -mabi=lp64d -x c vecdot_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; head -12 cc_drv.err; exit 4; }
$CC drv.o $OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vecdot_census 2>ld.err || { echo LINK_FAIL; head -12 ld.err; exit 5; }
echo "# linked OK -> ./vecdot_census" | tee -a "$SEAL"
echo "# ALL_BUILT" | tee -a "$SEAL"
