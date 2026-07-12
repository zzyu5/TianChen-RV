#!/usr/bin/env bash
# G5-M3 IME q4_K forward-bridge INTEGRATION UT runner (k1 - IME harts 0-3).
#
# Builds + runs the q4_K super-block bridge UT on the K1 board, linking the REAL
# ggml quantizers/dequantizer from libggml-base.so AND the REAL stock RVV kernel
# ggml_vec_dot_q4_K_q8_K from libggml-cpu.so so the oracle is genuine ggml, not a
# re-implementation:
#   g5m3_q4k_bridge_ut.c -> #4 weight repack + #3 activation quant/pack byte-exact
#                           vs ggml native q4_K/q8_K; FULL single-tensor mul_mat
#                           A==B vs ZERO-MODEL q4_K x q8_K double + real ggml kernel.
# Uses the emitter-verbatim `vmadot` asm leaf -> silicon seal of the integrated
# super-block path. Does NOT touch the vendor ggml tree. Board scratch: /tmp/g5m3q4k.
set -euo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
INC="$DIR/ggml/include"
LIB="$DIR/build-off/bin"          # RVV stock (no IME) -> ggml-base + ggml-cpu symbols
MARCH="rv64gcv_xsmtvdotii1p0"
BD="/tmp/g5m3q4k"

ssh "$BOARD" "mkdir -p $BD"
scp tools/e2e-harness/board/g5-m3-ime-q4_K/g5m3_q4k_bridge_ut.c "$BOARD:$BD/" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -e
cd "$BD"
echo "# toolchain: \$(gcc --version | head -1)"
gcc -O2 -std=c11 -march=$MARCH -mabi=lp64d g5m3_q4k_bridge_ut.c \
    -L "$LIB" -lggml-base -lggml-cpu -lm -Wl,-rpath,"$LIB" -o utq4k
echo "## objdump vmadot engage (expect 0xe210312b count>0):"
objdump -d utq4k | grep -c e210312b | sed 's/^/vmadot_count=/'
echo "## q4_K bridge UT (random matrices, real ggml native oracle), pinned harts 0-3:"
taskset -c 0-3 ./utq4k
EOF
