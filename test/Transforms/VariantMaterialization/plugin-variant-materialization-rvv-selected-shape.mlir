// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module {
  // CHECK-LABEL: tcrv.exec.kernel @selector_i32m2_with_both_shapes
  tcrv.exec.kernel @selector_i32m2_with_both_shapes {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_binary_selected_shape {
      id = "rvv.i32_binary.selected_vector_shape",
      kind = "isa-vector-config",
      shape = "i32m2",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_sew32 {
      id = "rvv.i32_m1.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lmul_m1 {
      id = "rvv.i32_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_tail_agnostic {
      id = "rvv.i32_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_mask_agnostic {
      id = "rvv.i32_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_sew32 {
      id = "rvv.i32_m2.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_lmul_m2 {
      id = "rvv.i32_m2.lmul_m2",
      kind = "isa-vector-config",
      lmul = "m2",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_tail_agnostic {
      id = "rvv.i32_m2.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_mask_agnostic {
      id = "rvv.i32_m2.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
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
      value = "rv64gcv",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv, @rvv_i32_m2_sew32, @rvv_i32_m2_lmul_m2, @rvv_i32_m2_tail_agnostic, @rvv_i32_m2_mask_agnostic],
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_binary_dtype = "i32",
      tcrv_rvv.selected_binary_family = "i32-vadd",
      tcrv_rvv.selected_binary_operator = "add",
      tcrv_rvv.selected_vector_shape = "i32m2",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m2",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m2_t",
      tcrv_rvv.selected_vector_suffix = "i32m2",
      tcrv_rvv.selected_setvl_suffix = "e32m2"
    } {
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv, @rvv_i32_m2_sew32, @rvv_i32_m2_lmul_m2, @rvv_i32_m2_tail_agnostic, @rvv_i32_m2_mask_agnostic],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      selected_vector_shape = "i32m2",
      selected_vector_sew = 32 : i64,
      selected_vector_lmul = "m2",
      selected_tail_policy = "agnostic",
      selected_mask_policy = "agnostic",
      selected_vector_type = "vint32m2_t",
      selected_vector_suffix = "i32m2",
      selected_setvl_suffix = "e32m2",
      source_kernel = "selector_i32m2_with_both_shapes"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m2, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }

  // CHECK: tcrv.exec.variant @rvv_first_slice
  // CHECK-SAME: origin = "rvv-plugin"
  // CHECK-SAME: requires = [@rvv, @rvv_i32_m2_sew32, @rvv_i32_m2_lmul_m2, @rvv_i32_m2_tail_agnostic, @rvv_i32_m2_mask_agnostic]
  // CHECK-SAME: tcrv_rvv.selected_setvl_suffix = "e32m2"
  // CHECK-SAME: tcrv_rvv.selected_vector_lmul = "m2"
  // CHECK-SAME: tcrv_rvv.selected_vector_shape = "i32m2"
  // CHECK-SAME: tcrv_rvv.selected_vector_suffix = "i32m2"
  // CHECK-SAME: tcrv_rvv.selected_vector_type = "vint32m2_t"
  // CHECK: tcrv.exec.diagnostic
  // CHECK-SAME: reason = "variant-selected"
  // CHECK-SAME: selection_kind = "static-variant"
  // CHECK-SAME: target = @rvv_first_slice
  // CHECK: tcrv_rvv.lowering_boundary
  // CHECK-SAME: required_capabilities = [@rvv, @rvv_i32_m2_sew32, @rvv_i32_m2_lmul_m2, @rvv_i32_m2_tail_agnostic, @rvv_i32_m2_mask_agnostic]
  // CHECK-SAME: selected_variant = @rvv_first_slice
  // CHECK-SAME: selected_vector_lmul = "m2"
  // CHECK-SAME: selected_vector_shape = "i32m2"
  // CHECK: tcrv.exec.diagnostic
  // CHECK-SAME: emission_kind = "rvv-unsupported-metadata-boundary"
  // CHECK-SAME: required_capabilities = [@rvv, @rvv_i32_m2_sew32, @rvv_i32_m2_lmul_m2, @rvv_i32_m2_tail_agnostic, @rvv_i32_m2_mask_agnostic]
  // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
  // CHECK-SAME: status = "unsupported"

  // CHECK-LABEL: tcrv.exec.kernel @default_i32m1_without_selector
  tcrv.exec.kernel @default_i32m1_without_selector {
    tcrv.exec.capability @default_rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @default_rvv_i32_m1_sew32 {
      id = "rvv.i32_m1.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @default_rvv_i32_m1_lmul_m1 {
      id = "rvv.i32_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @default_rvv_i32_m1_tail_agnostic {
      id = "rvv.i32_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @default_rvv_i32_m1_mask_agnostic {
      id = "rvv.i32_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @default_rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @default_rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @default_rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      value = "rv64gcv",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@default_rvv, @default_rvv_i32_m1_sew32, @default_rvv_i32_m1_lmul_m1, @default_rvv_i32_m1_tail_agnostic, @default_rvv_i32_m1_mask_agnostic],
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_binary_dtype = "i32",
      tcrv_rvv.selected_binary_family = "i32-vadd",
      tcrv_rvv.selected_binary_operator = "add",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@default_rvv, @default_rvv_i32_m1_sew32, @default_rvv_i32_m1_lmul_m1, @default_rvv_i32_m1_tail_agnostic, @default_rvv_i32_m1_mask_agnostic],
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
      source_kernel = "default_i32m1_without_selector"
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
  }

  // CHECK: tcrv.exec.variant @rvv_first_slice
  // CHECK-SAME: requires = [@default_rvv, @default_rvv_i32_m1_sew32, @default_rvv_i32_m1_lmul_m1, @default_rvv_i32_m1_tail_agnostic, @default_rvv_i32_m1_mask_agnostic]
  // CHECK-SAME: tcrv_rvv.selected_setvl_suffix = "e32m1"
  // CHECK-SAME: tcrv_rvv.selected_vector_lmul = "m1"
  // CHECK-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
  // CHECK-SAME: tcrv_rvv.selected_vector_type = "vint32m1_t"
  // CHECK: tcrv_rvv.lowering_boundary
  // CHECK-SAME: selected_variant = @rvv_first_slice
  // CHECK-SAME: selected_vector_lmul = "m1"
  // CHECK-SAME: selected_vector_shape = "i32m1"
  // CHECK: tcrv.exec.diagnostic
  // CHECK-SAME: emission_kind = "rvv-unsupported-metadata-boundary"
  // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
  // CHECK-SAME: status = "unsupported"
}
