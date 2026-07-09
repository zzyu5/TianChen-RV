#!/usr/bin/env bash
# kquant_repack_cert.sh -- [SEL-1 T4b / M1] build+run the GENERAL K-quant repacker per-tensor
# byte-exact / bounded-ULP certificate on rvv/VLEN128 against the board's OWN stock ggml integer
# paths (q4_K + q5_K). Compiles OUR two golden emitted repack-GEMM kernels (pre-generated; NO local
# tcrv-opt) + the general cert driver (kquant_repack_cert.c, over kquant_repacker.h) with clang-17
# -O2 and links the board's real libggml-cpu.so. Also builds+runs the standalone offline repacker
# tool over several shapes. Main tree + build/ + ggml A-tree UNTOUCHED; nothing committed; no git
# stash. NOT a perf probe (feasibility + numeric equivalence only).
#
# Usage: BOARD=rvv CORE=8 bash tools/e2e-harness/board/kquant_repack_cert.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
K4="${K4:-/tmp/kquant_tile_s6_ab/golden_q4K.c}"              # golden q4_K repack-GEMM, md5 b0b5beac
K5="${K5:-/tmp/kquant_tile_s6_ab/gemm_q5_K_q8_K.kernel.c}"   # golden q5_K repack-GEMM, md5 ba30ba54
RDIR="/tmp/m1_repack_cert"
SEED="${SEED:-20260709}"

echo "[m1-cert] board=$BOARD core=$CORE march=$MARCH ggml=$GGML_BIN"
echo "[m1-cert] golden kernels: q4_K=$K4  q5_K=$K5"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$HERE/kquant_repacker.h"     "$BOARD:$RDIR/"
scp -q "$HERE/kquant_repack_cert.c"  "$BOARD:$RDIR/"
scp -q "$HERE/kquant_repack_tool.c"  "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  test -f $K4 || { echo 'MISSING golden q4_K kernel $K4'; exit 40; }
  test -f $K5 || { echo 'MISSING golden q5_K kernel $K5'; exit 40; }
  cp -f $K4 ./golden_q4K.c; cp -f $K5 ./golden_q5K.c
  echo '[md5] golden kernels (guard vs stale):'; md5sum golden_q4K.c golden_q5K.c
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 41; }

  echo '[compile] two golden kernels + general cert (clang-17 -O2), link real libggml-cpu.so'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q4K.c -c -o kq4k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q5K.c -c -o kq5k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c++ kquant_repack_cert.c -c -o cert.o
  $CC cert.o kq4k.o kq5k.o -o m1cert -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm -lstdc++

  echo; echo '=== M1 general repacker per-tensor certificate vs STOCK ggml (VLEN128) ==='
  RC=0
  LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./m1cert all $SEED || RC=\$?

  echo; echo '[compile] standalone offline repacker tool (host-only, libstdc++)'
  $CC -O2 -x c++ kquant_repack_tool.c -o kquant_repack_tool -lstdc++
  echo '=== M1 standalone repacker: arbitrary shapes -> block_qX_Kx16 (geometry + determinism) ==='
  for args in 'q4_K 16 256' 'q4_K 160 2560' 'q4_K 4096 4096' 'q5_K 16 256' 'q5_K 80 1024' 'q5_K 4096 4096' 'q8_K 4 256' 'q8_K 32 2048'; do
    ./kquant_repack_tool \$args $SEED
  done
  exit \$RC
"
echo "[m1-cert] done. restore: nothing to restore (scratch under $RDIR on board; main tree + build/ + ggml A-tree untouched)."
