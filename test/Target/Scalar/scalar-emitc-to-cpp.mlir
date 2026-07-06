// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %s --tcrv-check-capability-requires --tcrv-materialize-plugin-variants --tcrv-verify-plugin-variant-legality --tcrv-select-variants | tcrv-translate --tcrv-scalar-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_template" --implicit-check-not="int main"
// RUN: tcrv-translate --tcrv-scalar-emitc-to-cpp %s | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_template" --implicit-check-not="int main"

// X-SCALAR tracer bullet (line D, step 2): a hand-written portable-scalar
// source kernel flows through the generic capability/variant planning passes
// (check-capability-requires -> materialize-plugin-variants ->
// verify-plugin-variant-legality -> select-variants) and then the scalar
// backend emission driver lowers the selected tcrv_scalar.compute_skeleton
// boundary directly to a standalone EmitC module that the
// --tcrv-scalar-emitc-to-cpp route renders as PURE SCALAR C/C++ (no __riscv_
// intrinsics). The second RUN proves the translate route is self-contained on
// the same source (byte-identical output).
//
// This is a TRIVIAL compute boundary (step 3 lands a real kernel). The emission
// is operand-driven, not vacuous: the exported function name is derived from
// the op's source_kernel + selected_variant, and the emitted `int32_t v = 7;`
// echoes the op's scalar_immediate attribute -- changing either attribute
// changes the emitted C.
//
// NOTE: --tcrv-materialize-emission-plans is intentionally NOT in the pipe. That
// pass gates on the origin plugin reporting a SUPPORTED emission readiness
// (lib/Transforms/EmissionReadiness.cpp: "does not accept a materialized plugin
// lowering boundary"); the scalar plugin's emission-readiness/plan/boundary
// methods still fail-closed as Unsupported, and flipping them is blocked by the
// out-of-scope unit test test/Plugin/ScalarExtensionPluginTest.cpp, which
// hard-locks the Unsupported behavior for the canonical scalar variant.

// HELP: --tcrv-scalar-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

module {
  tcrv.exec.kernel @scalar_kernel {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv_scalar.compute_skeleton {source_kernel = "scalar_kernel", selected_variant = @scalar_fallback_first_slice, scalar_immediate = 7 : i64}
  }
}

// SOURCE: #include <stdint.h>
// SOURCE: int32_t tcrv_scalar_compute_skeleton(int32_t);
// SOURCE: extern "C" void tcrv_emitc_scalar_kernel_scalar_fallback_first_slice()
// SOURCE: tcrv_emitc.route_source_op=tcrv_scalar.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: tcrv_emitc.source_op=tcrv_scalar.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_compute_skeleton
// SOURCE: int32_t v{{[0-9]+}} = 7;
// SOURCE: int32_t v{{[0-9]+}} = tcrv_scalar_compute_skeleton(v{{[0-9]+}});
