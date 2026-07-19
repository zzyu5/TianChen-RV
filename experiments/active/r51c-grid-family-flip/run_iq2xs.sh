#!/usr/bin/env bash
# run_iq2xs.sh — R5.1-C iq2_xs vec_dot attack loop: DEPLOYED vs NARROW (vC) vs NARROWACC (vD).
# Board = rvv (VLEN128, clang-18, ggml build-clang18-rv64gcv _vl128 hand-tuned opponent).
# Gate: byte-exact ZERO-MODEL 3-way + 3-arm anti-hollow, THEN 2-seed cold timing.
# usage: REPS=25 ./run_iq2xs.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/r51c_iq2xs
REPS="${REPS:-25}"
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
PIN="${PIN:-40}"

ssh rvv "mkdir -p $RDIR/kernels"
scp -q "$HERE/iq2_vecdot_driver_ext.c" "$HERE/iq2_oracle_tables_ext.h" "rvv:$RDIR/"
scp -q "$HERE/tables/iq2xs_grid_tables.h" "rvv:$RDIR/kernels/"
scp -q "$HERE"/kernels/*.kernel.c "rvv:$RDIR/kernels/"

ssh rvv "set -uo pipefail; cd $RDIR; source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v clang-18); GT=\"--gcc-toolchain=\$TCRV_GCC\"; M=$MARCH; GGML=$GGML; PIN=$PIN; REPS=$REPS
  echo \"# board=\$(uname -srm) march=\$M CC=\$(\$CC --version|head -1) load=\$(cat /proc/loadavg)\"
  for k in iq2_xxs iq2_s_deployed iq2_xs_deployed iq2_xs_narrow iq2_xs_narrowacc; do
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -Ikernels -x c++ kernels/\${k}.kernel.c -c -o k_\${k}.o || { echo BUILD_FAIL \$k; exit 3; }
  done
  echo '=== structure self-probe (compiled leaf .o) ==='
  OBJ=\$(\$CC \$GT -print-prog-name=llvm-objdump 2>/dev/null); OBJ=\${OBJ:-llvm-objdump}
  for k in iq2_xs_deployed iq2_xs_narrow iq2_xs_narrowacc; do
    D=\$(\$OBJ -d k_\${k}.o 2>/dev/null)
    printf '# %-18s vluxei=%s vwredsum=%s vwmul=%s vwmacc=%s vslidedown=%s vsetvli=%s bytes=%s\n' \"\$k\" \
      \"\$(echo \"\$D\"|grep -c vlux)\" \"\$(echo \"\$D\"|grep -c vwredsum)\" \"\$(echo \"\$D\"|grep -c vwmul)\" \
      \"\$(echo \"\$D\"|grep -c vwmacc)\" \"\$(echo \"\$D\"|grep -c vslidedown)\" \"\$(echo \"\$D\"|grep -c vset)\" \"\$(stat -c%s k_\${k}.o)\"
  done
  \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c iq2_vecdot_driver_ext.c -c -o drv.o || { echo DRV_FAIL; exit 4; }
  LINK() { \$CC \$GT drv.o k_iq2_xxs.o \$1 k_iq2_s_deployed.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o \$2; }
  LINK k_iq2_xs_deployed.o   dot_dep || { echo LINK_DEP_FAIL; exit 5; }
  LINK k_iq2_xs_narrow.o     dot_nar || { echo LINK_NAR_FAIL; exit 5; }
  LINK k_iq2_xs_narrowacc.o  dot_acc || { echo LINK_ACC_FAIL; exit 5; }
  echo '=== [A] byte-exact VERIFY (reps=0) + anti-hollow ==='
  for pair in dep:DEPLOYED nar:NARROW acc:NARROWACC; do
    ex=\${pair%%:*}; nm=\${pair##*:}; echo \"--- \$nm ---\"
    taskset -c \$PIN ./dot_\$ex iq2_xs 2048 1 512 0 0x1 0 224 | grep -E 'BYTE_EXACT|worst_ulp'
    for inj in 1 2 3; do taskset -c \$PIN ./dot_\$ex iq2_xs 2048 1 512 0 0x1 \$inj 224 | grep ANTIHOLLOW; done
  done
  echo \"=== [B] 2-seed cold timing (224MiB flush, reps=\$REPS) ===\"
  for sd in 0x1 0x2; do
    echo -n 'DEP '; taskset -c \$PIN ./dot_dep iq2_xs 2048 1 512 \$REPS \$sd 0 224 | grep VECDOT_COLD
    echo -n 'NAR '; taskset -c \$PIN ./dot_nar iq2_xs 2048 1 512 \$REPS \$sd 0 224 | grep VECDOT_COLD
    echo -n 'ACC '; taskset -c \$PIN ./dot_acc iq2_xs 2048 1 512 \$REPS \$sd 0 224 | grep VECDOT_COLD
  done
  echo '# DONE'
"
