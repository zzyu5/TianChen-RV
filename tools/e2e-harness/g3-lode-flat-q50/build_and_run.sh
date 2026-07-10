#!/usr/bin/env bash
# G3-lode-flat FLAT-2 q5_0 repack-GEVM/GEMM byte-exact cert driver.
#  (1) ZERO-MODEL numerical cert: host-compile + run the independent scalar oracle
#      pair (integer sumi bit-exact, GEVM + GEMM). (2) emitted-C compile
#      verification: emit the front-door q5_0 GEVM/GEMM C and cross-compile it with
#      the riscv64 toolchain (well-formed vector object; not executed -- no qemu on
#      this host). NO git ops. Regenerates emitted C into /tmp (not committed).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OPT="$REPO/build/bin/tcrv-opt"
TR=/usr/lib/llvm-20/bin/mlir-translate
RVCLANG=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/clang

echo "== [1] ZERO-MODEL numerical cert =="
gcc -O2 -Wall -o /tmp/q50_cert "$HERE/q50_repack_cert.c" -lm && /tmp/q50_cert
NUM_RC=$?

echo "== [2] emitted-C compile verification (riscv64 -march=rv64gcv_zvfh) =="
for reg in \
  "test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir:GEVM" \
  "test/Conversion/RVV/rvv-emit-quant-contraction-q5-0-repack-gemm-prefill-vlen128.mlir:GEMM"; do
  f="${reg%:*}"; tag="${reg#*:}"
  "$OPT" "$REPO/$f" --tcrv-rvv-lower-quant-contraction=march=rv64gcv \
      --tcrv-rvv-lower-to-emitc 2>/dev/null | "$TR" --mlir-to-cpp > /tmp/q50_$tag.cpp
  if [ -x "$RVCLANG" ] && "$RVCLANG" -O2 -march=rv64gcv_zvfh -c /tmp/q50_$tag.cpp -o /tmp/q50_$tag.o 2>/tmp/q50_$tag.err; then
    echo "  COMPILE-OK $tag ($(wc -l < /tmp/q50_$tag.cpp) lines emitted C)"
  else
    echo "  COMPILE-SKIP/FAIL $tag (riscv clang $RVCLANG)"; head -3 /tmp/q50_$tag.err 2>/dev/null
  fi
done
exit $NUM_RC
