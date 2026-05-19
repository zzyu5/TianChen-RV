// RUN: not tcrv-translate --tcrv-rvv-emitc-to-cpp %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.kernel @rvv_missing_with_vl_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected stale RVV route",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "rvv-generic-typed-body-emitc-route-family",
      message = "stale RVV plan missing the selected with_vl boundary",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      severity = "info",
      status = "supported",
      target = @rvv_i32_add
    }
  }
}

// CHECK: construction-template artifact adapter failed
// CHECK-SAME: requires one selected materialized tcrv_rvv.with_vl before artifact export
