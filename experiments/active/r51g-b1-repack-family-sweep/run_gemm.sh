#!/usr/bin/env bash
set -euo pipefail
BOARD="${BOARD:-rvv}"; REMOTE_DIR="${REMOTE_DIR:-/tmp/r51g-gemm}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/harness_gemm.cpp" $HERE/kernels/q40_gemm_mf2.cpp $HERE/kernels/q40_gemm_m1.cpp \
  $HERE/kernels/q41_gemm_mf2.cpp $HERE/kernels/q41_gemm_m1.cpp \
  $HERE/kernels/q50_gemm_mf2.cpp $HERE/kernels/q50_gemm_m1.cpp \
  $HERE/kernels/q51_gemm_mf2.cpp $HERE/kernels/q51_gemm_m1.cpp "$BOARD:$REMOTE_DIR/"
ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51g-gemm
CXX=clang++; ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="; $CXX --version | head -1
for k in q40 q41 q50 q51; do for arm in mf2 m1; do $CXX -O3 $ARCH -c ${k}_gemm_${arm}.cpp -o ${k}_gemm_${arm}.o; done; done
$CXX -O3 $ARCH -c harness_gemm.cpp -o harness.o
$CXX -O3 $ARCH q40_gemm_mf2.o q40_gemm_m1.o q41_gemm_mf2.o q41_gemm_m1.o q50_gemm_mf2.o q50_gemm_m1.o q51_gemm_mf2.o q51_gemm_m1.o harness.o -o bench -lrt
for k in q40 q41 q50 q51; do
  echo "$k mf2:$(objdump -d ${k}_gemm_mf2.o | grep -oE 'e(8|16|32),m(f2|1|2|4)' | sort | uniq -c | tr '\n' ' ') | m1:$(objdump -d ${k}_gemm_m1.o | grep -oE 'e(8|16|32),m(f2|1|2|4)' | sort | uniq -c | tr '\n' ' ') | spill m1=$(objdump -d ${k}_gemm_m1.o | grep -cE 'vl[0-9]+re[0-9]+\.v|vs[0-9]+r\.v' || true) gather m1=$(objdump -d ${k}_gemm_m1.o | grep -cE 'vlux|vrgather' || true)"
done
echo "== run =="; ./bench
REMOTE
