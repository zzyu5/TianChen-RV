// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants | FileCheck %s --check-prefix=MAT
// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality --tcrv-select-variants | FileCheck %s --check-prefix=PIPE
// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants --tcrv-materialize-plugin-variants | FileCheck %s --check-prefix=RERUN

module {
  // MAT-LABEL: tcrv.exec.kernel @scalar_only
  tcrv.exec.kernel @scalar_only {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @scalar_only
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @scalar_only
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @deterministic_builtin_order
  tcrv.exec.kernel @deterministic_builtin_order {
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      selected_vector_shape = "i32m1",
      selected_vector_sew = 32 : i64,
      selected_vector_lmul = "m1",
      selected_tail_policy = "agnostic",
      selected_mask_policy = "agnostic",
      selected_vector_type = "vint32m1_t",
      selected_vector_suffix = "i32m1",
      selected_setvl_suffix = "e32m1",
      source_kernel = "deterministic_builtin_order"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    // MAT: tcrv.exec.variant @rvv_first_slice
    // MAT-SAME: condition = "rvv_capability_properties_available"
    // MAT-SAME: guard = "plugin_local_rvv_property_evidence"
    // MAT-SAME: origin = "rvv-plugin"
    // MAT-SAME: policy = "metadata_only_first_slice"
    // MAT-SAME: requires = [@rvv]
    // MAT-SAME: tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // MAT-SAME: tcrv_rvv.required_march = "rv64gcv"
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @deterministic_builtin_order
    // PIPE: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-NOT: tcrv.exec.dispatch
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "static-variant"
    // PIPE-SAME: target = @rvv_first_slice
    // PIPE-NOT: runtime_guard

    // RERUN-LABEL: tcrv.exec.kernel @deterministic_builtin_order
    // RERUN-COUNT-1: tcrv.exec.variant @rvv_first_slice
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @rvv_profile_provides_rvv
  tcrv.exec.kernel @rvv_profile_provides_rvv {
    tcrv.exec.target @rvv_profile {
      id = "rvv.profile.rv64gcv",
      kind = "profile",
      provides = ["rvv", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
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
      requires = [@rvv_profile],
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv_profile],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      selected_vector_shape = "i32m1",
      selected_vector_sew = 32 : i64,
      selected_vector_lmul = "m1",
      selected_tail_policy = "agnostic",
      selected_mask_policy = "agnostic",
      selected_vector_type = "vint32m1_t",
      selected_vector_suffix = "i32m1",
      selected_setvl_suffix = "e32m1",
      source_kernel = "rvv_profile_provides_rvv"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    // MAT: tcrv.exec.target @rvv_profile
    // MAT-SAME: provides = ["rvv", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"]
    // MAT: tcrv.exec.variant @rvv_first_slice
    // MAT-SAME: origin = "rvv-plugin"
    // MAT-SAME: requires = [@rvv_profile]
    // MAT-SAME: tcrv_rvv.required_march = "rv64gcv"
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @rvv_profile_provides_rvv
    // PIPE: tcrv.exec.variant @rvv_first_slice
    // PIPE-SAME: requires = [@rvv_profile]
    // PIPE-NOT: tcrv.exec.dispatch
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "static-variant"
    // PIPE-SAME: target = @rvv_first_slice
    // PIPE-NOT: runtime_guard

    // RERUN-LABEL: tcrv.exec.kernel @rvv_profile_provides_rvv
    // RERUN-COUNT-1: tcrv.exec.variant @rvv_first_slice
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  tcrv.exec.target @module_rvv_profile {
    id = "rvv.profile.module",
    kind = "profile",
    provides = ["rvv", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    architecture = "riscv64",
    isa_vector_hints = "rv64gcv_zvl128b",
    status = "available"
  }

  // MAT-LABEL: tcrv.exec.kernel @module_rvv_profile_provides_rvv
  // MAT-SAME: target = @module_rvv_profile
  tcrv.exec.kernel @module_rvv_profile_provides_rvv attributes {target = @module_rvv_profile} {
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
      requires = [@module_rvv_profile],
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@module_rvv_profile],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      selected_vector_shape = "i32m1",
      selected_vector_sew = 32 : i64,
      selected_vector_lmul = "m1",
      selected_tail_policy = "agnostic",
      selected_mask_policy = "agnostic",
      selected_vector_type = "vint32m1_t",
      selected_vector_suffix = "i32m1",
      selected_setvl_suffix = "e32m1",
      source_kernel = "module_rvv_profile_provides_rvv"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    // MAT: tcrv.exec.variant @rvv_first_slice
    // MAT-SAME: origin = "rvv-plugin"
    // MAT-SAME: requires = [@module_rvv_profile]
    // MAT-SAME: tcrv_rvv.required_march = "rv64gcv"
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @module_rvv_profile_provides_rvv
    // PIPE-SAME: target = @module_rvv_profile
    // PIPE: tcrv.exec.variant @rvv_first_slice
    // PIPE-SAME: requires = [@module_rvv_profile]
    // PIPE-NOT: tcrv.exec.dispatch
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "static-variant"
    // PIPE-SAME: target = @rvv_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @module_rvv_profile_provides_rvv
    // RERUN-COUNT-1: tcrv.exec.variant @rvv_first_slice
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @scalar_independent_from_unavailable_rvv
  tcrv.exec.kernel @scalar_independent_from_unavailable_rvv {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "missing"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT-NOT: tcrv.exec.variant @rvv_first_slice
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @scalar_independent_from_unavailable_rvv
    // PIPE-NOT: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @scalar_independent_from_unavailable_rvv
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties
  tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT-NOT: tcrv.exec.variant @rvv_first_slice
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties
    // PIPE-NOT: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}
