#!/usr/bin/env bash
# G5-M3 IME q8_0 forward-bridge INTEGRATION UT runner (k1 - IME harts 0-3).
#
# Builds + runs the q8_0 int8-DIRECT bridge UT on the K1 board, linking the REAL
# ggml quantizer/dequantizer from libggml-base.so so the oracle is genuine ggml,
# not a re-implementation:
#   g5m3_q8_bridge_ut.c -> #4 weight repack (int8-DIRECT) + #3 activation quant/pack
#                          byte-exact vs ggml native q8_0; FULL single-tensor mul_mat
#                          A==B vs ZERO-MODEL q8_0 x q8_0 double block-dot.
# Uses the emitter-verbatim `vmadot` asm leaf -> silicon seal of the integrated
# flat int8-direct path. Does NOT touch the vendor ggml tree. Scratch: /tmp/g5m3q8.
set -euo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
LIB="$DIR/build-off/bin"          # RVV stock (no IME) -> ggml-base symbols
MARCH="rv64gcv_xsmtvdotii1p0"
BD="/tmp/g5m3q8"

ssh "$BOARD" "mkdir -p $BD"
scp tools/e2e-harness/board/g5-m3-ime-q8_0/g5m3_q8_bridge_ut.c "$BOARD:$BD/" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -e
cd "$BD"
echo "# toolchain: \$(gcc --version | head -1)"
gcc -O2 -std=c11 -march=$MARCH -mabi=lp64d g5m3_q8_bridge_ut.c \
    -L "$LIB" -lggml-base -lm -Wl,-rpath,"$LIB" -o utq8
echo "## objdump vmadot engage (expect 0xe210312b count>0):"
objdump -d utq8 | grep -c e210312b | sed 's/^/vmadot_count=/'
echo "## q8_0 bridge UT (random matrices, real ggml native oracle), pinned harts 0-3:"
taskset -c 0-3 ./utq8
EOF
