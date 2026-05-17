// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-translate --tcrv-template-emitc-to-cpp %S/template-target-artifact-object.mlir | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="int main"

// This route is a low-level Template EmitC-to-C++ translator. The positive
// test consumes the materialized/planned Template fixture directly instead of
// accepting a manual execution-planning-pipeline pipe.

// HELP: --tcrv-template-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stdint.h>
// SOURCE: int32_t tcrv_template_compute_skeleton();
// SOURCE: extern "C" void tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice()
// SOURCE: tcrv_emitc.route_source_op=tcrv_template.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: tcrv_emitc.source_op=tcrv_template.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_template_compute_skeleton
// SOURCE: int32_t v{{[0-9]+}} = tcrv_template_compute_skeleton();
