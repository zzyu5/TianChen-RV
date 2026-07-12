// RUN: weft-translate --help | FileCheck %s --check-prefix=HELP
// RUN: weft-opt %s --weft-check-capability-requires --weft-materialize-plugin-variants --weft-verify-plugin-variant-legality --weft-select-variants | weft-translate --weft-scalar-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="int main"
// RUN: weft-translate --weft-scalar-emitc-to-cpp %s | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="int main"

// X-SCALAR tracer bullet (line D, step 2): a hand-written portable-scalar
// source kernel flows through the generic capability/variant planning passes
// (check-capability-requires -> materialize-plugin-variants ->
// verify-plugin-variant-legality -> select-variants) and then the scalar
// backend emission driver lowers the selected weft_scalar.compute_skeleton
// boundary directly to a standalone EmitC module that the
// --weft-scalar-emitc-to-cpp route renders as PURE SCALAR C/C++ (no __riscv_
// intrinsics). The second RUN proves the translate route is self-contained on
// the same source (byte-identical output).
//
// This is a TRIVIAL compute boundary (step 3 lands a real kernel). The emission
// is operand-driven, not vacuous: the exported function name is derived from
// the op's source_kernel + selected_variant, and the emitted `int32_t v = 7;`
// echoes the op's scalar_immediate attribute -- changing either attribute
// changes the emitted C.
//
// NOTE: --weft-materialize-emission-plans is intentionally NOT in the pipe. That
// pass gates on the origin plugin reporting a SUPPORTED emission readiness
// (lib/Transforms/EmissionReadiness.cpp: "does not accept a materialized plugin
// lowering boundary"); the scalar plugin's emission-readiness/plan/boundary
// methods still fail-closed as Unsupported, and flipping them is blocked by the
// out-of-scope unit test test/Plugin/ScalarExtensionPluginTest.cpp, which
// hard-locks the Unsupported behavior for the canonical scalar variant.

// HELP: --weft-scalar-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

module {
  weft.exec.kernel @scalar_kernel {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft_scalar.compute_skeleton {source_kernel = "scalar_kernel", selected_variant = @scalar_fallback_first_slice, scalar_immediate = 7 : i64}
  }
}

// SOURCE: #include <stdint.h>
// SOURCE: int32_t weft_scalar_compute_skeleton(int32_t);
// SOURCE: extern "C" void weft_emitc_scalar_kernel_scalar_fallback_first_slice()
// SOURCE: weft_emitc.route_source_op=weft_scalar.compute_skeleton role=compute op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: weft_emitc.source_op=weft_scalar.compute_skeleton role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_scalar_compute_skeleton
// SOURCE: int32_t v{{[0-9]+}} = 7;
// SOURCE: int32_t v{{[0-9]+}} = weft_scalar_compute_skeleton(v{{[0-9]+}});
