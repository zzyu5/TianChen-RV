#!/usr/bin/env bash
# G5-M3 session-2 IME q4_0 forward-bridge INTEGRATION UT runner (k1 · IME harts 0-3).
#
# Builds + runs the two integration-layer bridge UTs on the K1 board, linking the
# REAL ggml quantizers/dequantizer from libggml-base.so so the oracle is genuine
# ggml native q4_0/q8_0, not a re-implementation:
#   g5m3_bridge_ut.c      -> #4 weight repack + #3 activation quant/pack byte-exact
#                            vs ggml native; FULL single-tensor mul_mat A==B vs the
#                            stock q4_0xq8_0 block-dot (bit-exact float order).
#   g5m3_realtensor_ab.c  -> same FULL A==B on a REAL tinyllama-q4_0.gguf weight
#                            tensor (token_embd.weight sub-tile).
# Both use the emitter-verbatim `vmadot` asm leaf -> silicon seal of the integrated
# path. Does NOT touch the vendor ggml tree. Board scratch: /tmp/g5m3.
set -euo pipefail
BOARD="${BOARD:-k1}"
DIR="/home/bianbu/tcrv-k1-llama"
INC="$DIR/ggml/include"
LIB="$DIR/build-off/bin"          # RVV stock (no IME) -> pure ggml-base symbols
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
MARCH="rv64gcv_xsmtvdotii1p0"
BD="/tmp/g5m3"

ssh "$BOARD" "mkdir -p $BD"
scp tools/e2e-harness/board/g5-m3-ime-q4_0/g5m3_bridge_ut.c \
    tools/e2e-harness/board/g5-m3-ime-q4_0/g5m3_realtensor_ab.c "$BOARD:$BD/" >/dev/null

ssh "$BOARD" bash -s <<EOF
set -e
cd "$BD"
echo "# toolchain: \$(gcc --version | head -1)"
gcc -O2 -std=c11 -march=$MARCH -mabi=lp64d g5m3_bridge_ut.c \
    -L "$LIB" -lggml-base -lm -Wl,-rpath,"$LIB" -o ut
gcc -O2 -std=c11 -march=$MARCH -mabi=lp64d g5m3_realtensor_ab.c \
    -I "$INC" -L "$LIB" -lggml-base -lm -Wl,-rpath,"$LIB" -o rt
echo "## objdump vmadot engage (expect 0xe210312b count>0):"
objdump -d ut | grep -c e210312b | sed 's/^/vmadot_count=/'
echo "## bridge UT (random matrices, real ggml native oracle), pinned harts 0-3:"
taskset -c 0-3 ./ut
echo "## real-model-tensor A==B (tinyllama-q4_0.gguf), pinned harts 0-3:"
taskset -c 0-3 ./rt "$MODEL"
EOF
