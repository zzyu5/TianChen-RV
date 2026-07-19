#!/usr/bin/env bash
# W2 §③ widen-to-m1 board runner. Copies the standalone kernels to `ssh rvv`,
# compiles both chains at -O3 -march=rv64gcv, runs the 2-seed cold GEVM sweep,
# and prints the CSV. No plugin code, no selector change -- board data only.
set -euo pipefail

BOARD="${BOARD:-rvv}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/w2-widen-to-m1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

ssh "$BOARD" "mkdir -p $REMOTE_DIR"
scp -q "$HERE/widen_kernels.c" "$HERE/harness.c" "$BOARD:$REMOTE_DIR/"

ssh "$BOARD" bash -s <<'REMOTE'
set -euo pipefail
cd /tmp/w2-widen-to-m1
# System clang-17 (integrated with the system gcc runtime so it links). Both
# chains use the SAME compiler, so the m1-vs-mf2 ratio is compiler-symmetric.
# (The pinned llvm-18 clang compiles objects but lacks a gcc sysroot to link.)
CC=clang
echo "== toolchain =="
$CC --version | head -1
echo "== compile (-O3 -march=rv64gcv) =="
$CC -O3 -march=rv64gcv -c widen_kernels.c -o widen_kernels.o
$CC -O3 -march=rv64gcv -c harness.c -o harness.o
$CC -O3 -march=rv64gcv widen_kernels.o harness.o -o widen_bench -lrt
echo "== objdump: confirm the two chains emit distinct vtypes (e8mf2 vs e8m1) =="
objdump -d widen_kernels.o | grep -oE "e(8|16|32),m(f2|1|2|4)" | sort | uniq -c
echo "== run =="
./widen_bench
REMOTE
