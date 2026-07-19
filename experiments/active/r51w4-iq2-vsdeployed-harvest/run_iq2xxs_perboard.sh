#!/usr/bin/env bash
# iq2_xxs PER-BOARD VLEN-adaptive test: rvv uses m2(VLEN128) leaf, k1 uses m1(VLEN256) leaf.
# Both plain-vget (no vslidedown) -- coreLmul selector is the designed VLEN-correctness mechanism.
# byte-exact ZERO-MODEL gate + 2-seed cold vs deployed. link needs all 3 leaves (driver externs).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-k1}"; RDIR=/tmp/w4_iq2xxs_perboard
REPS="${REPS:-25}"; SEEDS="${SEEDS:-0x1 0x2}"; K=2048; NC=512; FL=224
if [ "$BOARD" = k1 ]; then
  MARCH=rv64gcv_zvl256b_zfh_zvfh; GGML=/home/bianbu/tcrv-k1-llama/build-off/bin; ENVSRC=":"; GT=""; PIN=4
  XXS=iq2_xxs_m1_vlen256   # VLEN256 -> m1 native-widening leaf
else
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs; GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  ENVSRC="source /opt/tcrv-toolchains/env.sh"; GT='--gcc-toolchain=$TCRV_GCC'; PIN=40
  XXS=iq2_xxs_m2_vlen128    # VLEN128 -> m2 leaf (== committed db1bfdcd)
fi
ssh "$BOARD" "mkdir -p $RDIR/kernels"
scp -q "$HERE/iq2_vecdot_driver_ext.c" "$HERE/iq2_oracle_tables_ext.h" "$BOARD:$RDIR/"
scp -q "$HERE/kernels/${XXS}.kernel.c" "$BOARD:$RDIR/kernels/iq2_xxs_sel.kernel.c"
scp -q "$HERE/kernels/iq2_xs_new.kernel.c" "$HERE/kernels/iq2_s_new.kernel.c" "$BOARD:$RDIR/kernels/"
ssh "$BOARD" "set -uo pipefail; cd $RDIR; $ENVSRC 2>/dev/null
  CC=\$(command -v clang-18); GT=\"$GT\"; M=$MARCH; GGML=$GGML; PIN=$PIN
  echo \"# board=\$(uname -srm) march=\$M leaf=$XXS ggml_md5=\$(md5sum \$GGML/libggml-cpu.so|cut -d' ' -f1)\"
  \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/iq2_xxs_sel.kernel.c -c -o k_iq2_xxs.o 2>e1 || { echo BUILD_FAIL_xxs; cat e1; exit 3; }
  \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/iq2_xs_new.kernel.c -c -o k_iq2_xs.o 2>e2 || { echo BUILD_FAIL_xs; exit 3; }
  \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/iq2_s_new.kernel.c -c -o k_iq2_s.o 2>e3 || { echo BUILD_FAIL_s; exit 3; }
  echo \"# OURS_iq2_xxs($XXS): vslidedown=\$(objdump -d k_iq2_xxs.o|grep -c vslidedown) vredsum=\$(objdump -d k_iq2_xxs.o|grep -cE 'vwredsum|vredsum') vsetvli=\$(objdump -d k_iq2_xxs.o|grep -cE 'vsetvli|vsetivli') vlux=\$(objdump -d k_iq2_xxs.o|grep -c vlux)\"
  \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c iq2_vecdot_driver_ext.c -c -o drv.o 2>e4 || { echo DRV_FAIL; cat e4; exit 4; }
  \$CC \$GT drv.o k_iq2_xxs.o k_iq2_xs.o k_iq2_s.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o iq2_dot 2>e5 || { echo LINK_FAIL; cat e5; exit 5; }
  echo '=== byte-exact gate (iq2_xxs, 2-seed + anti-hollow) ==='
  G=1
  for sd in $SEEDS; do o=\$(taskset -c \$PIN ./iq2_dot iq2_xxs $K 1 $NC 0 \$sd 0 $FL); echo \"\$o\"|grep BYTE_EXACT; echo \"\$o\"|grep -q ALL=true || G=0; done
  for inj in 1 2 3; do taskset -c \$PIN ./iq2_dot iq2_xxs $K 1 $NC 0 0x1 \$inj $FL | grep ANTIHOLLOW; done
  [ \$G -ne 1 ] && { echo '# GATE_FAIL -> no timing'; exit 0; }
  echo '=== cold (reps='$REPS' 2-seed) ==='
  for sd in $SEEDS; do taskset -c \$PIN ./iq2_dot iq2_xxs $K 1 $NC $REPS \$sd 0 $FL | grep VECDOT_COLD; done
  echo '# DONE'
"
