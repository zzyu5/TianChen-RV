// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_runtime_scalar_splat_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_runtime_scalar_splat_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-scalar-splat-store:rhs-scalar", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-scalar-splat-store:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-runtime-scalar-splat-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_scalar_splat_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_runtime_scalar_splat_store {origin = "rvv-plugin", policy = "pre-realized-runtime-scalar-splat-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-runtime-scalar-splat-store-fallback"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_runtime_scalar_splat_store
// REALIZED: weft_rvv.splat
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.load
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_runtime_scalar_splat_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_runtime_scalar_splat_store
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: weft.rvv.runtime_abi_parameter[0]: int32_t rhs_scalar role=rhs-scalar-value ownership=target-export-abi-owned
// HEADER: weft.rvv.runtime_abi_parameter[1]: int32_t *out role=output-buffer ownership=target-export-abi-owned
// HEADER: void weft_emitc_pre_realized_body_runtime_scalar_splat_store_kernel_pre_realized_body_rvv_runtime_scalar_splat_store(int32_t rhs_scalar, int32_t *out, size_t n);
