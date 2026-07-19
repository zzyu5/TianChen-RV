#!/usr/bin/env bash
# [GAP-P1]-loosen rollout: board-verify the DEPLOYED q8_0 repack-GEVM m1-vs-mf2
# crossover (CORE==PROD emit). scp the deployed kernels + harness to `ssh rvv`,
# compile both chains with the SAME system clang-17 (compiler-symmetric ratio) at
# -O3 -march=rv64gcv_zvfh, run the 2-seed cold sweep, print objdump vtype proof +
# CSV. NO plugin code compiled here; kernels are the front-door emit (byte-exact
# sealed md5 in FINDING.md).
set -euo pipefail
BOARD="${BOARD:-rvv}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/r51f-gapp1-q8-deployed}"
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/harness.cpp" "$HERE/kernels/q8_gevm_mf2.cpp" \
       "$HERE/kernels/q8_gevm_m1.cpp" "$BOARD:$REMOTE_DIR/"

ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51f-gapp1-q8-deployed
CXX=clang++
ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="
$CXX --version | head -1
echo "== compile (-O3 $ARCH) =="
$CXX -O3 $ARCH -c q8_gevm_mf2.cpp -o q8_gevm_mf2.o
$CXX -O3 $ARCH -c q8_gevm_m1.cpp  -o q8_gevm_m1.o
$CXX -O3 $ARCH -c harness.cpp     -o harness.o
$CXX -O3 $ARCH q8_gevm_mf2.o q8_gevm_m1.o harness.o -o q8_bench -lrt
echo "== objdump: DEPLOYED mf2 chain vtypes (expect e8,mf2 + i16m1 + i32m2) =="
objdump -d q8_gevm_mf2.o | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
echo "== objdump: DEPLOYED m1 chain vtypes (expect e8,m1 + i16m2 + i32m4) =="
objdump -d q8_gevm_m1.o  | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
echo "== objdump: no gather (vlux/vrgather) in either clean int8 chain =="
echo "mf2 gather: $(objdump -d q8_gevm_mf2.o | grep -cE 'vlux|vrgather' || true)"
echo "m1  gather: $(objdump -d q8_gevm_m1.o  | grep -cE 'vlux|vrgather' || true)"
echo "== run (2-seed cold sweep) =="
./q8_bench
REMOTE
