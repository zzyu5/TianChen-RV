// RUN: not weft-translate --weft-rvv-emitc-to-cpp %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  weft.exec.kernel @rvv_missing_selected_boundary_attrs_kernel {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !weft_rvv.vl
    }
    weft.exec.diagnostic {
      reason = "variant-selected",
      message = "selected stale RVV route",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add,
      selection_kind = "static-variant"
    }
    weft.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "weft_rvv.with_vl",
      lowering_pipeline = "rvv-generic-typed-body-emitc-route-family",
      message = "stale RVV plan missing selected-boundary attrs",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-exact-typed-body-callable-c-abi.v2",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2",
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      severity = "info",
      status = "supported",
      target = @rvv_i32_add
    }
  }
}

// CHECK: Weft RVV exact-body artifact bridge failed
// CHECK-SAME: candidate must carry the exact body's non-empty typed runtime ABI
