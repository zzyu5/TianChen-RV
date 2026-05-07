// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_plan_input {
  tcrv.exec.kernel @rvv_microkernel_direct {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
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
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
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
      target = @rvv_first_slice
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_microkernel_direct"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        tcrv_rvv.i32_vadd_dataflow {lhs = "lhs", out = "out", rhs = "rhs", runtime_n = "n"}
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-LABEL: tcrv.exec.kernel @rvv_microkernel_direct
// CHECK: tcrv_rvv.lowering_boundary
// CHECK-SAME: selected_variant = @rvv_first_slice
// CHECK-SAME: status = "unsupported"
// CHECK: tcrv.exec.diagnostic
// CHECK-SAME: artifact_kind = "runtime-callable-c-source"
// CHECK-SAME: emission_kind = "rvv-explicit-i32-vadd-microkernel-c-source"
// CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
// CHECK-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// CHECK-SAME: message = "explicit RVV i32 vector-add microkernel C source export provides a library-style runtime-callable C ABI function for this selected path; any self-check main is an explicit harness export and is not the default artifact contract; this is not generic RVV lowering, runtime integration, arbitrary kernel emission, correctness, or performance evidence"
// CHECK-SAME: origin = "rvv-plugin"
// CHECK-SAME: reason = "emission_plan"
// CHECK-SAME: required_capabilities = [@rvv]
// CHECK-SAME: role = "direct variant"
// CHECK-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
// CHECK-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// CHECK-SAME: runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1"
// CHECK-SAME: runtime_abi_parameters = [{{.*}}role = "lhs-input-buffer"{{.*}}ownership = "target-export-abi-owned"{{.*}}role = "runtime-element-count"
// CHECK-SAME: runtime_glue_role = "runtime-callable-i32-vadd-function"
// CHECK-SAME: severity = "info"
// CHECK-SAME: status = "supported"
// CHECK-SAME: target = @rvv_first_slice
