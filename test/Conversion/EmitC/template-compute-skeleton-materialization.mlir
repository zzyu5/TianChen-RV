// RUN: weft-opt %s --weft-execution-planning-pipeline --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="weft_template.lowering_boundary" --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_tensorext_lite"

module {
  weft.exec.kernel @template_emitc_kernel attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}

// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.func private @weft_template_compute_skeleton() -> !emitc.opaque<"int32_t">
// EMITC: emitc.func @weft_emitc_template_emitc_kernel_template_zero_core_first_slice
// EMITC: weft_emitc.route_source_op=weft_template.compute_skeleton role=compute op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_template.compute_skeleton role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_template_compute_skeleton
// EMITC: call_opaque "weft_template_compute_skeleton"() : () -> !emitc.opaque<"int32_t">
