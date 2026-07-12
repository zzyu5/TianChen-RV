// RUN: weft-translate --help | FileCheck %s --check-prefix=HELP
// RUN: weft-translate --weft-template-emitc-to-cpp %S/template-target-artifact-object.mlir | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="int main"

// This route is a low-level Template EmitC-to-C++ translator. The positive
// test consumes the materialized/planned Template fixture directly instead of
// accepting a manual execution-planning-pipeline pipe.

// HELP: --weft-template-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stdint.h>
// SOURCE: int32_t weft_template_compute_skeleton();
// SOURCE: extern "C" void weft_emitc_template_emitc_kernel_template_zero_core_first_slice()
// SOURCE: weft_emitc.route_source_op=weft_template.compute_skeleton role=compute op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: weft_emitc.source_op=weft_template.compute_skeleton role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_template_compute_skeleton
// SOURCE: int32_t v{{[0-9]+}} = weft_template_compute_skeleton();
