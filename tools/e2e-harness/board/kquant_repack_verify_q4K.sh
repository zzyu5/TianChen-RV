#!/usr/bin/env bash
# kquant_repack_verify_q4K.sh — [SEL-1 T4b / M0 tracer bullet] build+run the q4_K repack-GEMM
# byte-exact / bounded-ULP verifier on rvv/VLEN128 against the board's OWN stock ggml integer path.
# Compiles OUR golden emitted kernel (golden_q4K.c, md5 b0b5beac) + the oracle with clang-17 -O2
# and links the board's real libggml-cpu.so. Main tree + build/ UNTOUCHED; nothing committed;
# no git stash. NOT a perf probe (feasibility + numeric equivalence only).
#
# Usage: BOARD=rvv CORE=8 bash tools/e2e-harness/board/kquant_repack_verify_q4K.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
KERNEL="${KERNEL:-/tmp/kquant_tile_s6_ab/golden_q4K.c}"   # golden emitted q4_K repack-GEMM, md5 b0b5beac
RDIR="/tmp/m0_q4k_tracer"

echo "[m0-q4k] board=$BOARD core=$CORE march=$MARCH ggml=$GGML_BIN kernel=$KERNEL"
scp -q "$HERE/kquant_repack_verify_q4K.c" "$BOARD:$RDIR/" 2>/dev/null || {
  ssh "$BOARD" "mkdir -p $RDIR"; scp -q "$HERE/kquant_repack_verify_q4K.c" "$BOARD:$RDIR/"; }
ssh "$BOARD" "set -e; cd $RDIR
  test -f $KERNEL || { echo 'MISSING golden kernel $KERNEL'; exit 40; }
  cp -f $KERNEL ./golden_q4K.c; md5sum golden_q4K.c
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 41; }
  echo '[compile] golden kernel + oracle (clang-17 -O2), link real libggml-cpu.so'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q4K.c -c -o kq4k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c++ kquant_repack_verify_q4K.c -c -o orc.o
  $CC orc.o kq4k.o -o m0q4k -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm -lstdc++
  echo '=== M0 q4_K repack-GEMM vs STOCK ggml (VLEN128) ==='
  LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./m0q4k 20260709
"
echo "[m0-q4k] done. restore: nothing to restore (scratch under $RDIR on board; main tree + build/ untouched)."
