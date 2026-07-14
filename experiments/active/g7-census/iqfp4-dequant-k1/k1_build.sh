#!/usr/bin/env bash
# k1_build.sh — G7 L1 iq/fp4 vec_dot + dequant@k1 census: build ours (clang-18 symmetric),
# machine-probe opponents, link against stock libggml-{cpu,base}.so. Runs ON k1. No git, main untouched.
set -uo pipefail
RDIR=/tmp/g7_iqfp4_dq_k1
MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause      # == stock ggml-cpu compile march
GGML=/data/k1build-stock/bin
CC=clang-18
SEAL=$RDIR/build_seal.txt
cd "$RDIR"
: > "$SEAL"
echo "# BUILD iqfp4+dequant@k1  CC=$($CC --version|head -1)  march=$MARCH" | tee -a "$SEAL"
echo "# board=$(uname -srm)" | tee -a "$SEAL"
echo "# ggml_cpu_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)  ggml_base_md5=$(md5sum $GGML/libggml-base.so|cut -d' ' -f1)" | tee -a "$SEAL"

VFMTS="iq1_s iq1_m iq4_nl nvfp4"
DFMTS="q2_K q3_K q4_K q5_K q6_K iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4 tq1_0 tq2_0"

echo "=== [A] compile OURS vec_dot kernels (clang-18 -O3 symmetric) + objdump self-probe ===" | tee -a "$SEAL"
VOBJS=""
for f in $VFMTS; do
  $CC -O3 -march=$MARCH -mabi=lp64d -include math.h -x c++ kernels_vecdot/${f}.kernel.c -c -o kv_${f}.o 2>ccv_${f}.err || { echo "VKERN_FAIL $f"; head -6 ccv_${f}.err; exit 3; }
  VS=$(objdump -d kv_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  VG=$(objdump -d kv_${f}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather')
  echo "# OURS_vecdot_${f}: vsetvl=$VS gather=$VG size=$(wc -c<kv_${f}.o)B" | tee -a "$SEAL"
  VOBJS="$VOBJS kv_${f}.o"
done
echo "=== [B] compile OURS dequant kernels + self-probe ===" | tee -a "$SEAL"
DOBJS=""
for f in $DFMTS; do
  $CC -O3 -march=$MARCH -mabi=lp64d -include math.h -x c++ kernels_dequant/${f}.dq.c -c -o kd_${f}.o 2>ccd_${f}.err || { echo "DKERN_FAIL $f"; head -6 ccd_${f}.err; exit 3; }
  VS=$(objdump -d kd_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  VG=$(objdump -d kd_${f}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather')
  echo "# OURS_dequant_${f}: vsetvl=$VS gather=$VG size=$(wc -c<kd_${f}.o)B" | tee -a "$SEAL"
  DOBJS="$DOBJS kd_${f}.o"
done

echo "=== [C] opponent symbol machine-probe (objdump per sym; as-shipped dispatched class) ===" | tee -a "$SEAL"
objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
objdump -d $GGML/libggml-base.so > lib_base.txt 2>/dev/null
probe(){ # $1=symbol $2=disasm-file
  local sym=$1 f=$2
  local body; body=$(awk -v s="<${sym}>:" 'f&&/^[0-9a-f]+ </{exit} $0 ~ s{f=1} f' "$f")
  local nins nrvv gath csrr disp
  nins=$(echo "$body" | grep -cE '\t[a-z]')
  nrvv=$(echo "$body" | grep -cE '\tv[a-z]')
  gath=$(echo "$body" | grep -cE 'vlux|vloxei|vrgather')
  csrr=$(echo "$body" | grep -cE 'csrr')
  disp=$(echo "$body" | grep -oE '<(ggml_vec_dot|dequantize_row)_[a-z0-9_]+(_vl128|_vl256|_generic)>' | sort -u | tr '\n' ',')
  echo "# OPP $sym : ins=$nins rvv=$nrvv gather=$gath csrr=$csrr dispatch=[${disp:-none}]" | tee -a "$SEAL"
}
for f in $VFMTS; do case $f in nvfp4) sym=ggml_vec_dot_nvfp4_q8_0;; iq4_nl) sym=ggml_vec_dot_iq4_nl_q8_0;; *) sym=ggml_vec_dot_${f}_q8_K;; esac; probe "$sym" lib_cpu.txt; done
for f in $DFMTS; do probe "dequantize_row_${f}" lib_base.txt; done

echo "=== [D] compile drivers + link ours + stock libggml-{cpu,base}.so ===" | tee -a "$SEAL"
$CC -O2 -march=$MARCH -mabi=lp64d -x c iqfp4_vecdot_driver.c -c -o drv_v.o 2>ccdrvv.err || { echo VDRV_FAIL; head -12 ccdrvv.err; exit 4; }
$CC -O2 -march=$MARCH -mabi=lp64d -x c dequant_census_driver.c -c -o drv_d.o 2>ccdrvd.err || { echo DDRV_FAIL; head -12 ccdrvd.err; exit 4; }
$CC drv_v.o $VOBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o iqfp4_vecdot 2>ldv.err || { echo VLINK_FAIL; head -20 ldv.err; exit 5; }
$CC drv_d.o $DOBJS -L$GGML -Wl,-rpath,$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o dequant_census 2>ldd.err || { echo DLINK_FAIL; head -20 ldd.err; exit 5; }
echo "# linked OK -> ./iqfp4_vecdot ./dequant_census" | tee -a "$SEAL"
echo "# ALL_BUILT" | tee -a "$SEAL"
