#!/usr/bin/env bash
set -euo pipefail
BOARD="${BOARD:-rvv}"; REMOTE_DIR="${REMOTE_DIR:-/tmp/r51g-gevm-q5}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/harness_gevm_q5.cpp" \
       "$HERE/kernels/q50_gevm_mf2.cpp" "$HERE/kernels/q50_gevm_m1.cpp" \
       "$HERE/kernels/q51_gevm_mf2.cpp" "$HERE/kernels/q51_gevm_m1.cpp" "$BOARD:$REMOTE_DIR/"
ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51g-gevm-q5
CXX=clang++; ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="; $CXX --version | head -1
for k in q50_gevm_mf2 q50_gevm_m1 q51_gevm_mf2 q51_gevm_m1; do $CXX -O3 $ARCH -c $k.cpp -o $k.o; done
$CXX -O3 $ARCH -c harness_gevm_q5.cpp -o harness.o
$CXX -O3 $ARCH q50_gevm_mf2.o q50_gevm_m1.o q51_gevm_mf2.o q51_gevm_m1.o harness.o -o bench -lrt
for k in q50 q51; do
  echo "== objdump $k mf2 =="; objdump -d ${k}_gevm_mf2.o | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
  echo "== objdump $k m1  =="; objdump -d ${k}_gevm_m1.o  | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
  echo "$k spill mf2=$(objdump -d ${k}_gevm_mf2.o | grep -cE 'vl[0-9]+re[0-9]+\.v|vs[0-9]+r\.v' || true) m1=$(objdump -d ${k}_gevm_m1.o | grep -cE 'vl[0-9]+re[0-9]+\.v|vs[0-9]+r\.v' || true)  gather m1=$(objdump -d ${k}_gevm_m1.o | grep -cE 'vlux|vrgather' || true)"
done
echo "== run =="; ./bench
REMOTE
