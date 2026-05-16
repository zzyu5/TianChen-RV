// RUN: not tcrv-translate --tcrv-export-target-artifact %s 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_ambiguous_selected_dispatch {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add_a attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @rvv_i32_add_b attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_i32_add_a
      tcrv.exec.case @rvv_i32_add_b
      tcrv.exec.fallback @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "tcrv_rvv.config_contract", value = "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"},
        {key = "tcrv_rvv.sew", value = "32"},
        {key = "tcrv_rvv.lmul", value = "m1"},
        {key = "tcrv_rvv.tail_policy", value = "agnostic"},
        {key = "tcrv_rvv.mask_policy", value = "agnostic"},
        {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-setvl-with-vl-same-vl.v1"},
        {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"},
        {key = "tcrv_rvv.vl_def", value = "tcrv_rvv.setvl"},
        {key = "tcrv_rvv.vl_scope", value = "tcrv_rvv.with_vl"},
        {key = "tcrv_rvv.vl_uses", value = "with_vl,i32_load,i32_load,i32_arithmetic,i32_store"},
        {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
      ],
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "tcrv-rvv-i32m1-add-riscv-elf-object",
      message = "first supported dispatch case",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "dispatch case",
      runtime_abi = "rvv-i32m1-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"},
        {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"},
        {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"},
        {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}
      ],
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      status = "supported",
      target = @rvv_i32_add_a
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "tcrv_rvv.config_contract", value = "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"},
        {key = "tcrv_rvv.sew", value = "32"},
        {key = "tcrv_rvv.lmul", value = "m1"},
        {key = "tcrv_rvv.tail_policy", value = "agnostic"},
        {key = "tcrv_rvv.mask_policy", value = "agnostic"},
        {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-setvl-with-vl-same-vl.v1"},
        {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"},
        {key = "tcrv_rvv.vl_def", value = "tcrv_rvv.setvl"},
        {key = "tcrv_rvv.vl_scope", value = "tcrv_rvv.with_vl"},
        {key = "tcrv_rvv.vl_uses", value = "with_vl,i32_load,i32_load,i32_arithmetic,i32_store"},
        {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,out,n"}
      ],
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "tcrv-rvv-i32m1-add-riscv-elf-object",
      message = "second supported dispatch case",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "dispatch case",
      runtime_abi = "rvv-i32m1-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"},
        {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"},
        {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"},
        {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}
      ],
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      status = "supported",
      target = @rvv_i32_add_b
    }
    tcrv.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic", emission_kind = "scalar-fallback-unsupported-emission", lowering_pipeline = "scalar-fallback-no-materialized-emitc-route", message = "fallback unsupported", origin = "scalar-plugin", plan_kind = "plugin-emission-plan", reason = "emission_plan", required_capabilities = [@scalar_fallback], role = "dispatch fallback", runtime_abi = "scalar-fallback-no-runtime-abi", runtime_abi_kind = "unsupported-plugin-runtime-abi", runtime_abi_name = "unsupported-emission-runtime-abi", runtime_glue_role = "no-runtime-glue-unsupported", severity = "error", status = "unsupported", target = @scalar_fallback_first_slice}
  }
}

// CHECK: requires at most one supported target artifact emission-plan route
// CHECK-SAME: found multiple ambiguous supported artifacts
