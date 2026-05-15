// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=GENERIC-HDR-DELETED --implicit-check-not="__riscv" --implicit-check-not="out[index]"

module {
  tcrv.exec.kernel @conflict_planned_dispatch {
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
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @host_policy_no_vector {
      id = "host.policy.no-vector",
      kind = "toolchain",
      conflicts = ["rvv"],
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_sew_capability = "rvv.i32_m1.sew32",
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_vector_lmul_capability = "rvv.i32_m1.lmul_m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_tail_policy_capability = "rvv.i32_m1.tail_policy.agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_mask_policy_capability = "rvv.i32_m1.mask_policy.agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
  }
}

// IR: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// IR-SAME: abi_role = "dispatch-availability-guard"
// IR-SAME: c_name = "dispatch_available"
// IR-SAME: c_type = "int"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: policy = "metadata_only_first_slice"
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR-SAME: runtime_guard_required = true
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv_rvv.lowering_boundary {
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-NOT: tcrv_scalar.lowering_boundary
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "rvv-unsupported-metadata-boundary"
// IR-SAME: message = "RVV first slice has no materialized EmitC lowering, runtime ABI, artifact contract, or executable emission path"
// IR-SAME: role = "dispatch case"
// IR-SAME: runtime_abi_kind = "unsupported-plugin-runtime-abi"
// IR-SAME: status = "unsupported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "scalar-fallback-unsupported-emission"
// IR-SAME: message = "scalar fallback first slice has no materialized extension-family body, EmitC lowering, runtime ABI, target artifact route, or metadata-only emission route"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: runtime_abi_kind = "unsupported-plugin-runtime-abi"
// IR-SAME: status = "unsupported"
// IR-SAME: target = @scalar_fallback_first_slice

// GENERIC-HDR-DELETED: requires exactly one supported header artifact emission-plan route; found none
