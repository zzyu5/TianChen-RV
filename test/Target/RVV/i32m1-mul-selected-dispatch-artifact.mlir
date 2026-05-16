// RUN: tcrv-translate --tcrv-export-target-artifact %s > %t.dispatch.o
// RUN: file %t.dispatch.o | FileCheck %s --check-prefix=OBJECT
// RUN: tcrv-translate --tcrv-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="scalar_fallback_first_slice"
// RUN: tcrv-translate --tcrv-rvv-i32m1-mul-object %s > %t.direct.o
// RUN: file %t.direct.o | FileCheck %s --check-prefix=OBJECT
// RUN: tcrv-translate --tcrv-rvv-i32m1-mul-header %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle %s | FileCheck %s --check-prefix=BUNDLE-STATUS
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

module {
  tcrv.exec.kernel @rvv_selected_dispatch_mul_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_i32_mul attributes {
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
        %product = tcrv_rvv.i32_mul %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_i32_mul
      tcrv.exec.fallback @scalar_fallback_first_slice
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "tcrv-rvv-i32m1-mul-riscv-elf-object",
      message = "selected dispatch RVV i32m1 mul object route",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "dispatch case",
      runtime_abi = "rvv-i32m1-mul-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-i32m1-mul-callable-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"},
        {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"},
        {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"},
        {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}
      ],
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
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      status = "supported",
      target = @rvv_i32_mul
    }
    tcrv.exec.diagnostic {
      artifact_kind = "unsupported-emission-diagnostic",
      emission_kind = "scalar-fallback-unsupported-emission",
      lowering_pipeline = "scalar-fallback-no-materialized-emitc-route",
      message = "selected dispatch scalar fallback remains unsupported",
      origin = "scalar-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      runtime_abi = "scalar-fallback-no-runtime-abi",
      runtime_abi_kind = "unsupported-plugin-runtime-abi",
      runtime_abi_name = "unsupported-emission-runtime-abi",
      runtime_glue_role = "no-runtime-glue-unsupported",
      severity = "error",
      status = "unsupported",
      target = @scalar_fallback_first_slice
    }
  }
}

// OBJECT: ELF 64-bit LSB relocatable
// OBJECT-SAME: RISC-V
// HEADER: tianchenrv.runtime_abi_name: rvv-i32m1-mul-callable-c-abi.v1
// HEADER: tianchenrv.object_route: tcrv-rvv-i32m1-mul-riscv-elf-object
// HEADER: tianchenrv.header_route: tcrv-rvv-i32m1-mul-callable-c-header
// HEADER: void tcrv_emitc_rvv_selected_dispatch_mul_kernel_rvv_i32_mul
// BUNDLE-STATUS: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: selected_variant: @rvv_i32_mul
// BUNDLE-INDEX-NOT: scalar_fallback_first_slice
// BUNDLE-INDEX: route: "tcrv-rvv-i32m1-mul-riscv-elf-object"
// BUNDLE-INDEX: route: "tcrv-rvv-i32m1-mul-callable-c-header"
