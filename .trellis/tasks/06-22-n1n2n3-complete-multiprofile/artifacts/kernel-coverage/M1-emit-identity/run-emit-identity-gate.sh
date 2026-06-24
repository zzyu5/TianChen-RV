#!/usr/bin/env bash
# OPTION-2 M1 EMIT-IDENTITY GATE (host-only, no hardware).
# Kernel A = the AUTO-SELECTED repack emit (abstract op -> --tcrv-rvv-lower-quant-contraction
#            =march=rv64gcv [RVV1.0 VLEN128] -> --tcrv-rvv-lower-to-emitc -> mlir-translate).
# Kernel B = the DIRECT repack-op emit (same ABI wrapper, repack_gemv_q4_0_q8_0 with the
#            SAME x16 facts but WITHOUT the 4 audit attrs) -> --tcrv-rvv-lower-to-emitc -> mlir-translate.
# GATE: diff A B must be EMPTY (byte-identical) => audit attrs emitter-inert AND auto-select
#       emits EXACTLY what hand-authoring the repack op produces.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OPT=build/bin/tcrv-opt
TR=mlir-translate-20
A_SRC=test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir
ART=.trellis/tasks/06-22-n1n2n3-complete-multiprofile/artifacts/kernel-coverage/M1-emit-identity
B_SRC=$ART/kernelB.mlir

$OPT "$A_SRC" --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
  | $TR --mlir-to-cpp > "$ART/kernelA.cpp"
$OPT "$B_SRC" --tcrv-rvv-lower-to-emitc \
  | $TR --mlir-to-cpp > "$ART/kernelB.cpp"

if diff "$ART/kernelA.cpp" "$ART/kernelB.cpp"; then
  echo "GATE PASS: byte-identical"
  sha256sum "$ART/kernelA.cpp" "$ART/kernelB.cpp"
else
  echo "GATE FAIL: residual above"
  exit 1
fi
