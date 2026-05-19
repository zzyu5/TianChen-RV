// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_memory_reject_stride_unit {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_memory_bad_stride_unit attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %src_stride = tcrv_rvv.runtime_abi_value {c_name = "src_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      // expected-error@+1 {{currently supports only stride_unit "element" for the bounded selected-body strided memory movement hook}}
      tcrv_rvv.typed_strided_memory_pre_realized_body %src, %out, %n, %src_stride {lmul = "m1", memory_form = "strided-load-unit-store", op_kind = "strided_load_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "byte"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_memory_reject_stride_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_memory_bad_stride_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %src_stride = tcrv_rvv.runtime_abi_value {c_name = "src_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      // expected-error@+1 {{requires source stride operand to bind runtime ABI role 'lhs-input-stride'}}
      tcrv_rvv.typed_strided_memory_pre_realized_body %src, %out, %n, %src_stride {lmul = "m1", memory_form = "strided-load-unit-store", op_kind = "strided_load_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, stride_unit = "element"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index) -> ()
    }
  }
}
