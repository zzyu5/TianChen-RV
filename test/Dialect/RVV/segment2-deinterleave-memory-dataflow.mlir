// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_segment2_deinterleave_memory_dataflow_valid
  tcrv.exec.kernel @rvv_segment2_deinterleave_memory_dataflow_valid {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
    %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: tcrv_rvv.segment2_load
      %field0, %field1 = tcrv_rvv.segment2_load %src, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.move
      %moved0 = tcrv_rvv.move %field0, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.move
      %moved1 = tcrv_rvv.move %field1, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.store
      tcrv_rvv.store %out0, %moved0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      // CHECK: tcrv_rvv.store
      tcrv_rvv.store %out1, %moved1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_segment2_reject_segment_count {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_segment2_bad_segment_count attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires segment_count 2 for the bounded segment2 deinterleave memory hook}}
      tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 3 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_segment2_reject_duplicate_field_roles {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_segment2_bad_duplicate_fields attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires field1_role "segment-field1-output-buffer"}}
      tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field0-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_segment2_reject_source_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_segment2_bad_source_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires source operand to bind runtime ABI role 'lhs-input-buffer'}}
      tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_segment2_reject_destination_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_segment2_bad_destination_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires field0 destination operand to bind runtime ABI role 'segment-field0-output-buffer'}}
      tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_segment2_load_reject_bad_memory_form {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{currently supports only source_memory_form "segment2-interleaved-unit-stride-load" for tcrv_rvv.segment2_load}}
      %field0, %field1 = tcrv_rvv.segment2_load %src, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", segment_count = 2 : i64, source_memory_form = "unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_segment2_reject_authority_attr {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_segment2_bad_authority_attr attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "rvv-i32m1", segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
