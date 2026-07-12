// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-selected-lowering-boundaries

module {
  weft.exec.kernel @pre_realized_runtime_splat_reject_operation_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_runtime_splat_reject_operation attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only op_kind "runtime_scalar_splat_store"}}
      weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "scalar_broadcast_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_runtime_splat_reject_operation_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_runtime_splat_reject_operation {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_runtime_splat_reject_operation_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_runtime_splat_reject_authority_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_runtime_splat_reject_authority attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute}}
      weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_scalar_splat_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", sew = 32 : i64} : (i32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_runtime_splat_reject_authority_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_runtime_splat_reject_authority {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_runtime_splat_reject_authority_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  weft.exec.kernel @pre_realized_runtime_splat_reject_output_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_runtime_splat_reject_output attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires out operand to bind runtime ABI role 'output-buffer'}}
      weft_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_scalar_splat_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_runtime_splat_reject_output_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_runtime_splat_reject_output {origin = "rvv-plugin"}
      weft.exec.fallback @pre_realized_runtime_splat_reject_output_scalar {origin = "scalar-plugin"}
    }
  }
}
