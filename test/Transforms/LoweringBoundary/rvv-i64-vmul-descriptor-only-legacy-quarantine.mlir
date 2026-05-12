// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_descriptor_only_i64_vmul {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic"],
      sew_bits = 64 : i64,
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
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.element_count = 8 : i64,
      tcrv_rvv.lowering_descriptor = "i64-vmul-microkernel.v1",
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
      message = "legacy descriptor-only RVV i64-vmul fixture must be quarantined",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
  }
}

// CHECK: descriptor-only finite RVV binary legality metadata 'i64-vmul-microkernel.v1'
// CHECK-SAME: descriptor metadata is non-authoritative mirror metadata
// CHECK-SAME: typed RVV family/body or selected-source authority
