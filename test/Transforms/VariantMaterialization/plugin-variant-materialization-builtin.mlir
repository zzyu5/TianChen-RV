// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants | FileCheck %s --check-prefix=MAT
// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants --weft-check-capability-requires --weft-verify-plugin-variant-legality --weft-select-variants | FileCheck %s --check-prefix=PIPE
// RUN: weft-opt %s --split-input-file --weft-materialize-plugin-variants --weft-materialize-plugin-variants | FileCheck %s --check-prefix=RERUN

module {
  weft.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }
  weft.exec.target @scalar_only_profile {id = "scalar.only.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@scalar_fallback]}
  // MAT-LABEL: weft.exec.kernel @scalar_only
  weft.exec.kernel @scalar_only attributes {target = @scalar_only_profile, problem = @problem} {
    weft.exec.dequantize_row_q4_0_problem @problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    // MAT: weft.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: weft.exec.kernel @scalar_only
    // PIPE: weft.exec.variant @scalar_fallback_first_slice
    // PIPE: weft.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // RERUN-LABEL: weft.exec.kernel @scalar_only
    // RERUN-COUNT-1: weft.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: weft.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  weft.exec.capability @rvv {
    id = "rvv",
    kind = "isa-vector",
    architecture = "riscv64",
    isa_vector_hints = "rv64gcv_zvl128b",
    status = "available"
  }
  weft.exec.capability @rvv_hart_count {
    id = "rvv.hart_count",
    kind = "uarch",
    count = 64 : i64,
    status = "available"
  }
  weft.exec.capability @rvv_probe_compile_run {
    id = "rvv.probe.compile_run",
    kind = "toolchain",
    selected_march = "rv64gcv",
    status = "available"
  }
  weft.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }
  weft.exec.target @rvv_scalar_profile {id = "rvv.scalar.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@rvv, @rvv_hart_count, @rvv_probe_compile_run, @scalar_fallback]}
  // MAT-LABEL: weft.exec.kernel @rvv_capability_with_scalar_fallback
  weft.exec.kernel @rvv_capability_with_scalar_fallback attributes {target = @rvv_scalar_profile, problem = @problem} {
    weft.exec.dequantize_row_q4_0_problem @problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    // MAT-NOT: origin = "rvv-plugin"
    // MAT: weft.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: weft.exec.kernel @rvv_capability_with_scalar_fallback
    // PIPE-NOT: origin = "rvv-plugin"
    // PIPE: weft.exec.variant @scalar_fallback_first_slice
    // PIPE: weft.exec.diagnostic
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // RERUN-LABEL: weft.exec.kernel @rvv_capability_with_scalar_fallback
    // RERUN-NOT: origin = "rvv-plugin"
    // RERUN-COUNT-1: weft.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: weft.exec.variant @scalar_fallback_first_slice
  }
}
