#!/usr/bin/env bash
# run.sh — W5 grid-tail dequantize_row: OWNED narrow-per-entry body vs DEPLOYED ggml, for
# iq2_xxs / iq2_s / iq1_s (the grid-family flip tail). Board = rvv (VLEN128, clang-18).
# Opponent = ggml dequantize_row_<fmt> (libggml-base.so, host-autovec codegen-lottery,
# 标量类档 · COMPILER-SYMMETRIC: both our leaf and libggml built with clang-18). Gate:
# objdump 分拣前置 (opponent-vs-ours structure diff) -> byte-exact ZERO-MODEL 3-way + 3-arm
# anti-hollow -> 2-seed cold timing (vs-deployed >=0.8 => WIN入账·不问强弱). NO inline-asm,
# NO hand-scheduling (gather-free real-vector). usage: REPS=25 NB=2048 ./run.sh [fmt...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/r51h_grid_tail
REPS="${REPS:-25}"; NB="${NB:-2048}"
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
GGML="${GGML:-/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin}"
PIN="${PIN:-40}"
FMTS="${*:-iq2_xxs iq2_s iq1_s}"

ssh rvv "mkdir -p $RDIR/kernels $RDIR/tables"
for f in $FMTS; do
  scp -q "$HERE/${f}_dequant_driver.cpp" "rvv:$RDIR/"
  scp -q "$HERE/tables/${f}_grid_tables.h" "rvv:$RDIR/tables/"
  scp -q "$HERE/kernels/${f}_dequant.c" "rvv:$RDIR/kernels/"
done

ssh rvv "set -uo pipefail; cd $RDIR; source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v clang-18); GT=\"--gcc-toolchain=\$TCRV_GCC\"; M=$MARCH; GGML=$GGML; PIN=$PIN; REPS=$REPS; NB=$NB
  echo \"# board=\$(uname -srm) march=\$M CC=\$(\$CC --version|head -1) load=\$(cat /proc/loadavg)\"
  OBJ=\$(\$CC \$GT -print-prog-name=llvm-objdump 2>/dev/null); OBJ=\${OBJ:-llvm-objdump}
  for f in $FMTS; do
    echo \"================ \$f ================\"
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -Ikernels -x c++ kernels/\${f}_dequant.c -c -o k_\${f}.o || { echo BUILD_FAIL \$f; continue; }
    echo '=== [分拣] objdump structure: OURS (owned leaf .o) vs OPP (ggml deployed symbol) ==='
    D=\$(\$OBJ -d k_\${f}.o 2>/dev/null)
    printf '# OURS %-8s vlux=%s vrgather=%s vsext=%s vfcvt=%s vfmul=%s vfadd=%s vslide=%s vset=%s bytes=%s\n' \"\$f\" \
      \"\$(echo \"\$D\"|grep -c vlux)\" \"\$(echo \"\$D\"|grep -c vrgather)\" \"\$(echo \"\$D\"|grep -c vsext)\" \"\$(echo \"\$D\"|grep -c vfcvt)\" \
      \"\$(echo \"\$D\"|grep -c vfmul)\" \"\$(echo \"\$D\"|grep -c vfadd)\" \"\$(echo \"\$D\"|grep -c vslide)\" \"\$(echo \"\$D\"|grep -c vset)\" \"\$(stat -c%s k_\${f}.o)\"
    OD=\$(\$OBJ -d --disassemble-symbols=dequantize_row_\${f} \$GGML/libggml-base.so 2>/dev/null)
    [ -z \"\$OD\" ] && OD=\$(\$OBJ -d \$GGML/libggml-base.so 2>/dev/null | awk \"/<dequantize_row_\${f}>:/{p=1} p{print} p&&/^\$/{exit}\")
    printf '# OPP  %-8s vlux=%s vrgather=%s vsext=%s vfcvt=%s vfmul=%s vfadd=%s vslide=%s vset=%s\n' \"\$f\" \
      \"\$(echo \"\$OD\"|grep -c vlux)\" \"\$(echo \"\$OD\"|grep -c vrgather)\" \"\$(echo \"\$OD\"|grep -c vsext)\" \"\$(echo \"\$OD\"|grep -c vfcvt)\" \
      \"\$(echo \"\$OD\"|grep -c vfmul)\" \"\$(echo \"\$OD\"|grep -c vfadd)\" \"\$(echo \"\$OD\"|grep -c vslide)\" \"\$(echo \"\$OD\"|grep -c vset)\"
    \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c++ \${f}_dequant_driver.cpp -c -o drv_\${f}.o || { echo DRV_FAIL \$f; continue; }
    \$CC \$GT drv_\${f}.o k_\${f}.o -L\$GGML -Wl,-rpath,\$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o dq_\${f} || { echo LINK_FAIL \$f; continue; }
    echo '=== [A] byte-exact VERIFY (reps=0) + corpus + 3-arm anti-hollow ==='
    taskset -c \$PIN ./dq_\${f} \$NB 0 0x1 0 224 | grep -E 'BYTE-EXACT|CROSSCHECK|CORPUS'
    for inj in 1 2 3; do taskset -c \$PIN ./dq_\${f} \$NB 0 0x1 \$inj 224 | grep ANTIHOLLOW; done
    echo \"=== [B] 2-seed cold timing (224MiB flush, reps=\$REPS, nb=\$NB · vs-deployed >=0.8 => WIN) ===\"
    for sd in 0x1 0x2; do taskset -c \$PIN ./dq_\${f} \$NB \$REPS \$sd 0 224 | grep DEQUANT_ROW; done
  done
  echo '# DONE'
"
