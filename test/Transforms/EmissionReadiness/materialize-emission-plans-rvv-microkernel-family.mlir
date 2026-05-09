// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_family_plan_input {
  tcrv.exec.kernel @rvv_microkernel_i32_vsub_direct {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.variant @rvv_sub_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_sub_slice
    }
  }
}

// CHECK-LABEL: tcrv.exec.kernel @rvv_microkernel_i32_vsub_direct
// CHECK: tcrv_rvv.lowering_boundary
// CHECK-SAME: selected_variant = @rvv_sub_slice
// CHECK-SAME: status = "unsupported"
// CHECK: tcrv_rvv.i32_vsub_microkernel
// CHECK-SAME: element_count = 16 : i64
// CHECK-SAME: selected_mabi = "lp64d"
// CHECK-SAME: selected_variant = @rvv_sub_slice
// CHECK: %[[DIFF:.*]] = tcrv_rvv.i32_sub
// CHECK: tcrv_rvv.i32_store %[[DIFF]]
// CHECK: tcrv.exec.diagnostic
// CHECK-SAME: artifact_kind = "runtime-callable-c-source"
// CHECK-SAME: emission_kind = "rvv-explicit-i32-vsub-microkernel-c-source"
// CHECK-SAME: lowering_pipeline = "tcrv-export-rvv-i32-vsub-microkernel-c"
// CHECK-SAME: reason = "emission_plan"
// CHECK-SAME: runtime_abi = "rvv-i32-vsub-runtime-callable-c-abi.v1"
// CHECK-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// CHECK-SAME: runtime_abi_name = "rvv-i32-vsub-runtime-callable-c-function.v1"
// CHECK-SAME: runtime_glue_role = "runtime-callable-i32-vsub-function"
// CHECK-SAME: status = "supported"
// CHECK-SAME: target = @rvv_sub_slice
