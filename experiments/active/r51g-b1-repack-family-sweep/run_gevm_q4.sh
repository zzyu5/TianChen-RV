#!/usr/bin/env bash
# r51g family sweep: board-verify the DEPLOYED q4_0/q4_1 repack-GEVM m1-vs-mf2
# crossover (CORE==PROD front-door emit). compiler-symmetric (one system clang),
# -O3 -march=rv64gcv_zvfh, 2-seed cold sweep, objdump vtype proof + CSV.
set -euo pipefail
BOARD="${BOARD:-rvv}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/r51g-gevm-q4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/harness_gevm_q4.cpp" \
       "$HERE/kernels/q40_gevm_mf2.cpp" "$HERE/kernels/q40_gevm_m1.cpp" \
       "$HERE/kernels/q41_gevm_mf2.cpp" "$HERE/kernels/q41_gevm_m1.cpp" \
       "$BOARD:$REMOTE_DIR/"
ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51g-gevm-q4
CXX=clang++; ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="; $CXX --version | head -1
for k in q40_gevm_mf2 q40_gevm_m1 q41_gevm_mf2 q41_gevm_m1; do
  $CXX -O3 $ARCH -c $k.cpp -o $k.o
done
$CXX -O3 $ARCH -c harness_gevm_q4.cpp -o harness.o
$CXX -O3 $ARCH q40_gevm_mf2.o q40_gevm_m1.o q41_gevm_mf2.o q41_gevm_m1.o harness.o -o bench -lrt
for k in q40 q41; do
  echo "== objdump $k mf2 vtypes =="; objdump -d ${k}_gevm_mf2.o | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
  echo "== objdump $k m1  vtypes =="; objdump -d ${k}_gevm_m1.o  | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
  echo "$k mf2 gather: $(objdump -d ${k}_gevm_mf2.o | grep -cE 'vlux|vrgather' || true)  m1 gather: $(objdump -d ${k}_gevm_m1.o | grep -cE 'vlux|vrgather' || true)"
done
echo "== run (2-seed cold sweep) =="; ./bench
REMOTE
