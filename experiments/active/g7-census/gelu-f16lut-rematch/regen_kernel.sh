#!/usr/bin/env bash
# Deterministically regenerate the f16-LUT gelu variant kernel from the working-tree
# emitter (front-door). OURS = weft-emitted, gelu_precision="f16lut".
set -eu
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
WOPT="$ROOT/build-weft/bin/weft-opt"
XLATE=/usr/bin/mlir-translate-20
SELF="$(cd "$(dirname "$0")" && pwd)"
FIX="$ROOT/test/Conversion/RVV/rvv-to-emitc-ggml-forward-elementwise-gelu.mlir"
# 1) materialize front-door, 2) inject the gelu_precision="f16lut" numeric-tier attr
"$WOPT" "$FIX" --weft-rvv-materialize-forward-elementwise-stream-front-door \
 | sed 's/kind = "elementwise_gelu_map"}/gelu_precision = "f16lut", kind = "elementwise_gelu_map"}/' \
 > "$SELF/gelu_f16lut.materialized.mlir"
# 3) lower to emitc + translate to C
"$WOPT" "$SELF/gelu_f16lut.materialized.mlir" --weft-rvv-lower-to-emitc \
 | "$XLATE" --mlir-to-cpp > "$SELF/kernels/gelu.f16lut.kernel.c"
echo "regenerated: kernels/gelu.f16lut.kernel.c md5=$(md5sum "$SELF/kernels/gelu.f16lut.kernel.c"|cut -d' ' -f1)"
