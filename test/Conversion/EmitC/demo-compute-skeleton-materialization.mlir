// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries | sed 's/weft_demo.handoff_kind = "demo-extension-lowering-boundary"/weft_demo.emitc_route_mapping = "legacy-stale-route", weft_demo.handoff_kind = "demo-extension-lowering-boundary"/' | weft-opt --weft-materialize-emitc-lowerable-routes | FileCheck %s

// The Demo extension is a registered production target route.  Its final
// compute body is constructed and qualified by the Demo family owner before
// the shared backend emitter consumes it; no construction-template fallback is
// allowed to manufacture this module.
module {
  weft.exec.capability @demo_extension {
      id = "demo.extension",
      kind = "future-extension-demo",
      status = "available",
      integration_contract = "demo-zero-core-handoff.v1",
      handoff_kind = "demo-extension-lowering-boundary"
  }
  weft.exec.target @demo_profile {id = "demo.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@demo_extension]}
  weft.exec.kernel @demo_direct_emitc attributes {target = @demo_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}

// CHECK: emitc.include <"stdint.h">
// CHECK: emitc.func private @weft_demo_compute_skeleton() -> !emitc.opaque<"int32_t">
// CHECK: emitc.func @weft_emitc_demo_direct_emitc_demo_zero_core_first_slice
// CHECK: weft_emitc.route_source_op=weft_demo.compute_skeleton role=compute
// CHECK: call_opaque "weft_demo_compute_skeleton"
