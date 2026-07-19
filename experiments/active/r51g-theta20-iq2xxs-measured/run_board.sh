#!/usr/bin/env bash
# theta20 measured-table board check: iq2_xxs GRID-of-8 decode m2-vs-m1 crossover on
# `ssh k1` (VLEN256 -- the ONLY board where m1 is legal: e8m1 VLMAX 32 == the 32-element
# grid-codebook sub-block; at VLEN128 m1 is verifier-rejected). scp the two CORE EmitC
# kernels + harness, compile with the SAME system clang-18 (compiler-symmetric ratio) at
# -O3 -march=rv64gcv_zvfh, run the 2-seed cold sweep, print objdump vtype + gather proof
# + CSV. NO plugin code compiled here; kernels are the front-door emit (sealed md5 in
# FINDING.md).
set -euo pipefail
BOARD="${BOARD:-k1}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/r51g-theta20-iq2xxs}"
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/harness.cpp" "$HERE/oracle_tables.h" \
       "$HERE/kernels/iq2xxs_m2.cpp" "$HERE/kernels/iq2xxs_m1.cpp" \
       "$BOARD:$REMOTE_DIR/"

ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/r51g-theta20-iq2xxs
CXX=clang++
ARCH="-march=rv64gcv_zvfh"
echo "== toolchain =="
$CXX --version | head -1
echo "== VLEN =="
cat > vlen.c <<EOF
#include <stdio.h>
int main(){ unsigned long v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); printf("VLENB=%lu VLEN=%lu\n", v, v*8); return 0;}
EOF
clang -O2 -march=rv64gcv vlen.c -o vlen && ./vlen
echo "== compile (-O3 $ARCH) =="
$CXX -O3 $ARCH -c iq2xxs_m2.cpp -o iq2xxs_m2.o
$CXX -O3 $ARCH -c iq2xxs_m1.cpp -o iq2xxs_m1.o
$CXX -O3 $ARCH -c harness.cpp   -o harness.o
$CXX -O3 $ARCH iq2xxs_m2.o iq2xxs_m1.o harness.o -o iq2xxs_bench -lrt
echo "== objdump: m2 chain vtypes (expect e8,m2 + e16,m4 + e64,m4 wide gather) =="
objdump -d iq2xxs_m2.o | grep -oE "e(8|16|32|64),m(f2|1|2|4)" | sort | uniq -c
echo "== objdump: m1 chain vtypes (expect e8,m1 + e16,m2 + e64,m2 wide gather) =="
objdump -d iq2xxs_m1.o | grep -oE "e(8|16|32|64),m(f2|1|2|4)" | sort | uniq -c
echo "== objdump: gather (vlux) count -- iq2_xxs IS a codebook-gather leaf (nonzero) =="
echo "m2 vlux: $(objdump -d iq2xxs_m2.o | grep -cE 'vlux' || true)"
echo "m1 vlux: $(objdump -d iq2xxs_m1.o | grep -cE 'vlux' || true)"
echo "== objdump: no spill (no stack vector store/reload in the hot chain) =="
echo "m2 vs*/vl* stack: $(objdump -d iq2xxs_m2.o | grep -cE '(vs|vl)[0-9]*r' || true)"
echo "m1 vs*/vl* stack: $(objdump -d iq2xxs_m1.o | grep -cE '(vs|vl)[0-9]*r' || true)"
echo "== run (2-seed cold sweep, median of 7, 64MB flush) =="
./iq2xxs_bench
REMOTE
