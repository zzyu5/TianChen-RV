#!/usr/bin/env bash
set -euo pipefail
BOARD="${BOARD:-rvv}"; RD="${RD:-/tmp/r51g-vsopp}"; HERE="$(cd "$(dirname "$0")" && pwd)"
ssh "$BOARD" "mkdir -p $RD"
scp -q "$HERE/harness_vs_opponent.cpp" "$HERE/ours_q8_gevm_m1.cpp" "$BOARD:$RD/"
ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51g-vsopp
CXX=clang++; ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="; $CXX --version | head -1
$CXX -O3 $ARCH -c ours_q8_gevm_m1.cpp -o ours.o
$CXX -O3 $ARCH -c harness_vs_opponent.cpp -o h.o
$CXX -O3 $ARCH ours.o h.o -o bench -lrt
echo "== VLMAX probe (e8mf2 / e32m2 at this VLEN) =="
echo "== run (ours m1 vs ggml_gemv_q8_0_16x1_q8_0) =="; ./bench
REMOTE
