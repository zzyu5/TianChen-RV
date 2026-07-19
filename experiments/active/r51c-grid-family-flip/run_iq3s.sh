#!/usr/bin/env bash
# run_iq3s.sh — R5.1-C iq3_s dequantize_row: NARROW-per-entry owned body vs DEPLOYED scalar.
# Board = rvv (VLEN128, clang-18). Opponent = ggml dequantize_row_iq3_s (libggml-base.so,
# host-autovec codegen-lottery, 标量类档). Gate: byte-exact ZERO-MODEL 3-way + 3-arm
# anti-hollow, THEN 2-seed cold timing. usage: REPS=25 ./run_iq3s.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/r51c_iq3s
REPS="${REPS:-25}"; NB="${NB:-512}"
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
PIN="${PIN:-40}"

ssh rvv "mkdir -p $RDIR/kernels $RDIR/tables"
scp -q "$HERE/iq3s_dequant_driver.cpp" "rvv:$RDIR/"
scp -q "$HERE/tables/iq3s_grid_table.h" "rvv:$RDIR/tables/"
scp -q "$HERE/tables/iq3s_grid_table.h" "rvv:$RDIR/kernels/"
scp -q "$HERE"/kernels/iq3_s_dequant_deployed.c "$HERE"/kernels/iq3_s_dequant_narrow.c "rvv:$RDIR/kernels/"

ssh rvv "set -uo pipefail; cd $RDIR; source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v clang-18); GT=\"--gcc-toolchain=\$TCRV_GCC\"; M=$MARCH; GGML=$GGML; PIN=$PIN; REPS=$REPS; NB=$NB
  echo \"# board=\$(uname -srm) march=\$M CC=\$(\$CC --version|head -1) load=\$(cat /proc/loadavg)\"
  for k in iq3_s_dequant_deployed iq3_s_dequant_narrow; do
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -Ikernels -x c++ kernels/\${k}.c -c -o k_\${k}.o || { echo BUILD_FAIL \$k; exit 3; }
  done
  echo '=== structure self-probe (compiled leaf .o) ==='
  OBJ=\$(\$CC \$GT -print-prog-name=llvm-objdump 2>/dev/null); OBJ=\${OBJ:-llvm-objdump}
  for k in iq3_s_dequant_deployed iq3_s_dequant_narrow; do
    D=\$(\$OBJ -d k_\${k}.o 2>/dev/null)
    printf '# %-24s vlux=%s vsext=%s vfcvt=%s vfmul=%s vslide=%s vset=%s bytes=%s\n' \"\$k\" \
      \"\$(echo \"\$D\"|grep -c vlux)\" \"\$(echo \"\$D\"|grep -c vsext)\" \"\$(echo \"\$D\"|grep -c vfcvt)\" \
      \"\$(echo \"\$D\"|grep -c vfmul)\" \"\$(echo \"\$D\"|grep -c vslide)\" \"\$(echo \"\$D\"|grep -c vset)\" \"\$(stat -c%s k_\${k}.o)\"
  done
  \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c++ iq3s_dequant_driver.cpp -c -o drv.o || { echo DRV_FAIL; exit 4; }
  LINK() { \$CC \$GT drv.o \$1 -L\$GGML -Wl,-rpath,\$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o \$2; }
  LINK k_iq3_s_dequant_deployed.o dq_dep || { echo LINK_DEP_FAIL; exit 5; }
  LINK k_iq3_s_dequant_narrow.o   dq_nar || { echo LINK_NAR_FAIL; exit 5; }
  echo '=== [A] byte-exact VERIFY (reps=0) + anti-hollow ==='
  for pair in dep:DEPLOYED nar:NARROW; do
    ex=\${pair%%:*}; nm=\${pair##*:}; echo \"--- \$nm ---\"
    taskset -c \$PIN ./dq_\$ex \$NB 0 0x1 0 224 | grep -E 'BYTE-EXACT|CROSSCHECK|CORPUS'
    for inj in 1 2 3; do taskset -c \$PIN ./dq_\$ex \$NB 0 0x1 \$inj 224 | grep ANTIHOLLOW; done
  done
  echo \"=== [B] 2-seed cold timing (224MiB flush, reps=\$REPS, nb=\$NB) ===\"
  for sd in 0x1 0x2; do
    echo -n 'DEP '; taskset -c \$PIN ./dq_dep \$NB \$REPS \$sd 0 224 | grep DEQUANT_ROW
    echo -n 'NAR '; taskset -c \$PIN ./dq_nar \$NB \$REPS \$sd 0 224 | grep DEQUANT_ROW
  done
  echo '# DONE'
"
