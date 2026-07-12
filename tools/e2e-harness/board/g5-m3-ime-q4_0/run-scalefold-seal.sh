#!/usr/bin/env bash
# G5-M3 q4_0@ime forward-bridge SCALE-FOLD seal runner (k1 · IME harts 0-3).
#
# Builds the emitter-verbatim scale-fold seal on the K1 board with the vmadot-
# unlocking march token, verifies the `vmadot` (0xe210312b) engages via objdump,
# and runs it pinned to the IME harts (X60: IME present on harts 0-3). The seal
# validates the int32 MAC core bit-exact on real silicon AND the deferred per-block
# d_a*d_w f32 scale fold (bridge #1) at runtime shape (bridge #2) against a
# canonical q4_0 x q8_0 ZERO-MODEL reference.
#
# This does NOT touch the vendor ggml tree; it is a standalone kernel+oracle seal
# (the forward hook wiring is a later G5-M3 session). Board scratch: /tmp/g5m3.
#
# Usage:  ./run-scalefold-seal.sh            # copies seal from repo + builds + runs
#         SEAL_ONLY_RUN=1 ./run-scalefold-seal.sh   # re-run an already-built binary
set -euo pipefail

BOARD="${BOARD:-k1}"
SEAL_SRC="test/Target/IME/q4-0-matmul-tile-scalefold-k1seal.c"
BOARD_DIR="/tmp/g5m3"
MARCH="rv64gcv_xsmtvdotii1p0"

if [[ "${SEAL_ONLY_RUN:-0}" != "1" ]]; then
  scp "$SEAL_SRC" "$BOARD:$BOARD_DIR/" >/dev/null
fi

ssh "$BOARD" bash -s <<EOF
set -e
cd "$BOARD_DIR"
echo "# toolchain: \$(gcc --version | head -1)"
gcc -O2 -std=c11 -march=$MARCH -mabi=lp64d q4-0-matmul-tile-scalefold-k1seal.c -lm -o q40sfseal
echo "## objdump vmadot engage (expect 0xe210312b, count>0):"
objdump -d q40sfseal | grep -c e210312b | sed 's/^/vmadot_count=/'
echo "## run pinned to IME harts 0-3:"
taskset -c 0-3 ./q40sfseal
echo "seal_exit=\$?"
EOF
