// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_template.lowering_boundary" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_tensorext_lite"

module {
  tcrv.exec.kernel @template_emitc_kernel {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}

// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.func private @tcrv_template_compute_skeleton() -> !emitc.opaque<"int32_t">
// EMITC: emitc.func @tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice
// EMITC: tcrv_emitc.route_source_op=tcrv_template.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_template.compute_skeleton role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_template_compute_skeleton
// EMITC: call_opaque "tcrv_template_compute_skeleton"() : () -> !emitc.opaque<"int32_t">
