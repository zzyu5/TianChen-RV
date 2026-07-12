// RUN: weft-opt %s --split-input-file --verify-diagnostics

module {
  weft.exec.kernel @rvv_pre_realized_strided_store_memory_reject_stride_unit {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_strided_store_memory_bad_stride_unit attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      // expected-error@+1 {{currently supports only stride_unit "byte" for the bounded selected-body strided-store memory movement hook}}
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "element"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_strided_store_memory_reject_stride_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_strided_store_memory_bad_stride_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      // expected-error@+1 {{requires destination byte stride operand to bind runtime ABI role 'destination-byte-stride'}}
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_strided_store_memory_reject_old_output_stride_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_strided_store_memory_old_output_stride_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride = weft_rvv.runtime_abi_value {c_name = "dst_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "output-stride"} : index
      // expected-error@+1 {{requires destination byte stride operand to bind runtime ABI role 'destination-byte-stride'}}
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_strided_store_memory_reject_stride_c_type {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_strided_store_memory_bad_stride_c_type attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires runtime ABI role 'destination-byte-stride' to use C type 'size_t'}}
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "int32_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_strided_store_memory_reject_destination_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_strided_store_memory_bad_destination_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      // expected-error@+1 {{requires destination operand to bind runtime ABI role 'output-buffer'}}
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_strided_store_memory_reject_authority_metadata {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_strided_store_memory_authority_metadata attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      weft_rvv.typed_strided_store_memory_pre_realized_body %src, %dst, %n, %dst_stride_bytes {lmul = "m1", memory_form = "unit-load-strided-store", op_kind = "unit_load_strided_store", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "rvv-i32m1-legacy", sew = 32 : i64, stride_unit = "byte"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}
