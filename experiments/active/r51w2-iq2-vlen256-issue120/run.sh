#!/usr/bin/env bash
# run.sh — ISSUE-120 fix gate: iq2_xs + iq2_s vec_dot byte-exact at VLEN256 (k1) AND
# VLEN128 (rvv), old (pre-fix, broken@VLEN256) vs new (fixed). ZERO-MODEL 3-way + 3-arm
# anti-hollow (iq2_vecdot_driver_ext.c oracle_iq2_{xxs,xs,s} are libcall-free integer
# recomputes; iq2s_grid[1024] in iq2_oracle_tables_ext.h). Correctness gate only (no perf).
#   BOARD=k1  -> VLEN256, march rv64gcv_zvl256b_zfh_zvfh, ggml build-off
#   BOARD=rvv -> VLEN128, march rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs (env.sh), ggml clang18
# usage: BOARD=k1 ./run.sh   |   BOARD=rvv ./run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-k1}"
RDIR=/tmp/iq2_120_issue

if [ "$BOARD" = k1 ]; then
  MARCH=rv64gcv_zvl256b_zfh_zvfh; GGML=/home/bianbu/tcrv-k1-llama/build-off/bin
  ENVSRC=":"; GT=""; PIN="${PIN:-4}"
else
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  ENVSRC="source /opt/tcrv-toolchains/env.sh"; GT='--gcc-toolchain=$TCRV_GCC'; PIN="${PIN:-40}"
fi

ssh "$BOARD" "mkdir -p $RDIR/kernels"
scp -q "$HERE/iq2_vecdot_driver_ext.c" "$HERE/iq2_oracle_tables_ext.h" "$BOARD:$RDIR/"
scp -q "$HERE"/kernels/*.c "$BOARD:$RDIR/kernels/"

ssh "$BOARD" "set -uo pipefail; cd $RDIR; $ENVSRC 2>/dev/null
  CC=\$(command -v clang-18); GT=\"$GT\"; M=$MARCH; GGML=$GGML; PIN=$PIN
  echo \"# board=\$(uname -srm) march=\$M\"
  for k in iq2_xxs iq2_xs_old iq2_xs_new iq2_s_old iq2_s_new; do
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/\${k}.kernel.c -c -o k_\${k}.o || { echo BUILD_FAIL \$k; exit 3; }
  done
  \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c iq2_vecdot_driver_ext.c -c -o drv.o || { echo DRV_FAIL; exit 4; }
  \$CC \$GT drv.o k_iq2_xxs.o k_iq2_xs_old.o k_iq2_s_old.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o dot_old || { echo LINK_OLD_FAIL; exit 5; }
  \$CC \$GT drv.o k_iq2_xxs.o k_iq2_xs_new.o k_iq2_s_new.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o dot_new || { echo LINK_NEW_FAIL; exit 5; }
  for f in iq2_xs iq2_s; do
    echo \"===== \$f OLD (pre-fix) =====\"; taskset -c \$PIN ./dot_old \$f 2048 1 512 0 0x1 0 224 | grep BYTE_EXACT
    echo \"===== \$f NEW (fixed) =====\"
    for sd in 0x1 0x2; do taskset -c \$PIN ./dot_new \$f 2048 1 512 0 \$sd 0 224 | grep -E 'BYTE_EXACT|worst_ulp'; done
    for inj in 1 2 3; do taskset -c \$PIN ./dot_new \$f 2048 1 512 0 0x1 \$inj 224 | grep ANTIHOLLOW; done
  done
"
