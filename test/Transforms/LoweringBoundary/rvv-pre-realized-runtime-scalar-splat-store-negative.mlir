// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries

module {
  tcrv.exec.kernel @pre_realized_runtime_splat_reject_operation_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_runtime_splat_reject_operation attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only op_kind "runtime_i32_splat_store"}}
      tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "scalar_broadcast_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_runtime_splat_reject_operation_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_runtime_splat_reject_operation {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_runtime_splat_reject_operation_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_runtime_splat_reject_authority_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_runtime_splat_reject_authority attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute}}
      tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_i32_splat_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", sew = 32 : i64} : (i32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_runtime_splat_reject_authority_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_runtime_splat_reject_authority {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_runtime_splat_reject_authority_scalar {origin = "scalar-plugin"}
    }
  }
}

// -----

module {
  tcrv.exec.kernel @pre_realized_runtime_splat_reject_output_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_runtime_splat_reject_output attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires out operand to bind runtime ABI role 'output-buffer'}}
      tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body %rhs_scalar, %out, %n {lmul = "m1", memory_form = "runtime-scalar-splat-store", op_kind = "runtime_i32_splat_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (i32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_runtime_splat_reject_output_scalar attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_runtime_splat_reject_output {origin = "rvv-plugin"}
      tcrv.exec.fallback @pre_realized_runtime_splat_reject_output_scalar {origin = "scalar-plugin"}
    }
  }
}
