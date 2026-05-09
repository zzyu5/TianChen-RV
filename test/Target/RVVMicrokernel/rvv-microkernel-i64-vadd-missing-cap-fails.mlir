// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --implicit-check-not=tcrv_rvv.i64_vadd_microkernel --implicit-check-not="emission_plan" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_i64_vadd_missing_cap_input {
  tcrv.exec.kernel @export_i64_vadd_missing_i64_caps {
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
    tcrv.exec.variant @rvv_i64_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.element_count = 8 : i64,
      tcrv_rvv.lowering_descriptor = "i64-vadd-microkernel.v1",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_setvl_suffix = "e64m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_vector_sew = 64 : i64,
      tcrv_rvv.selected_vector_shape = "i64m1",
      tcrv_rvv.selected_vector_suffix = "i64m1",
      tcrv_rvv.selected_vector_type = "vint64m1_t"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV i64 vadd microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_i64_slice
    }
  }
}

// CHECK: requires finite i64m1 vector-shape config capability ids
