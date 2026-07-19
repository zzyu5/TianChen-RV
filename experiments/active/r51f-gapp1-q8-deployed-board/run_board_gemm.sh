#!/usr/bin/env bash
# GEMM (prefill) arm of the [GAP-P1]-loosen board check. Separate from the GEVM
# because the prefill columnsPerPass tradeoff (mf2=4 vs m1=1) is a distinct
# crossover measured on its own.
set -euo pipefail
BOARD="${BOARD:-rvv}"; REMOTE_DIR="${REMOTE_DIR:-/tmp/r51f-gapp1-q8-gemm}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/harness_gemm.cpp" "$HERE/kernels/q8_gemm_mf2.cpp" \
       "$HERE/kernels/q8_gemm_m1.cpp" "$BOARD:$REMOTE_DIR/"
ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51f-gapp1-q8-gemm
CXX=clang++; ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="; $CXX --version | head -1
$CXX -O3 $ARCH -c q8_gemm_mf2.cpp -o q8_gemm_mf2.o
$CXX -O3 $ARCH -c q8_gemm_m1.cpp  -o q8_gemm_m1.o
$CXX -O3 $ARCH -c harness_gemm.cpp -o harness_gemm.o
$CXX -O3 $ARCH q8_gemm_mf2.o q8_gemm_m1.o harness_gemm.o -o q8_gemm_bench -lrt
echo "== objdump mf2 vtypes =="; objdump -d q8_gemm_mf2.o | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
echo "== objdump m1 vtypes =="; objdump -d q8_gemm_m1.o | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
echo "== run (GEMM prefill, 2-seed cold) =="; ./q8_gemm_bench
REMOTE
