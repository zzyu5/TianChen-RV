#!/usr/bin/env bash
# G8 §六.3 iq1_s/iq1_m clang-18-symmetric harness build (ours clang-18 vs sealed quants_opp.o clang-18)
set -uo pipefail
source /opt/tcrv-toolchains/env.sh 2>/dev/null
CLANG=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
CLANGXX=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
BU=/opt/tcrv-toolchains/binutils-2.46.1/bin
GCC=$(command -v gcc)
MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
OPP=/tmp/g8s3/rvv
WD=/tmp/g8s3iq/rvv
cd $WD

# --- generate fail-closed stubs for opponent ggml-internal UND (off VLEN128 timed path) ---
# stub = symbols UND in some opp object AND defined in NONE of the three (truly external ggml-internals)
OBJS3="$OPP/quants_opp.o $OPP/quants_generic_opp.o $OPP/ggmlquants_opp.o"
UNDS=$(for o in $OBJS3; do $BU/objdump -t $o 2>/dev/null | awk '/UND/{print $NF}'; done | sort -u)
DEFS=$(for o in $OBJS3; do $BU/objdump -t $o 2>/dev/null | awk '$0 !~ /UND/ && ($2=="g"||$2=="l"||$2=="w"){print $NF}'; done | sort -u)
SYMS=$(comm -23 <(echo "$UNDS") <(echo "$DEFS") \
  | grep -E '^(ggml_vec_dot_.*_generic|quantize_row_.*_ref|quantize_iq4_xs|ggml_type_size|ggml_type_name|ggml_row_size|ggml_abort)$')
{
  echo '#include <stdlib.h>'
  echo '#include <stdio.h>'
  echo 'static void _fc(const char*s){ fprintf(stderr,"FAIL-CLOSED stub called: %s\n",s); abort(); }'
  for s in $SYMS; do echo "void $s(void){ _fc(\"$s\"); }"; done
  # iq4_nl/nvfp4 weft kernels are referenced by the shared driver switch but never called in this run
  echo 'void weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(void){ _fc("iq4_nl"); }'
  echo 'void weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot(void){ _fc("nvfp4"); }'
} > opp_stubs.c
echo "# stubbed $(echo "$SYMS" | wc -w) ggml-internal symbols"
$CLANG -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -B$BU -c opp_stubs.c -o opp_stubs.o 2>st.err || { echo STUB_FAIL; cat st.err; exit 3; }

# --- compile driver clang-18 (-O2) ---
$CLANG -O2 -march=$MARCH -mabi=lp64d -fno-integrated-as -B$BU -c iqfp4_vecdot_driver.c -o drv_v.o 2>drv.err || { echo DRV_FAIL; head -20 drv.err; exit 4; }

# --- (re)compile OUR kernels clang-18 (-O3) : arg $1 = tag (baseline / attack) ---
TAG=${1:-baseline}
for f in iq1_s iq1_m; do
  $CLANGXX -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -B$BU -include math.h -c ${f}.kernel.c -o ours_${f}.o 2>ck_${f}.err || { echo KERN_FAIL $f; head -10 ck_${f}.err; exit 5; }
  VS=$($BU/objdump -d ours_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  VG=$($BU/objdump -d ours_${f}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather')
  SZ=$($BU/objdump -h ours_${f}.o 2>/dev/null | awk '/\.text/{print $3; exit}')
  echo "# OURS_${TAG}_${f}: vsetvl=$VS gather=$VG textsize=0x$SZ"
done

# --- link with gcc driver (runtime resolution; object codegen all clang-18) ---
$GCC drv_v.o ours_iq1_s.o ours_iq1_m.o $OPP/quants_opp.o $OPP/quants_generic_opp.o $OPP/ggmlquants_opp.o opp_stubs.o -lstdc++ -lm -o iq_bench 2>ld.err || { echo LINK_FAIL; head -30 ld.err; exit 6; }
echo "# LINK OK -> $WD/iq_bench ($TAG)"
