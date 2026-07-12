// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_segment2_interleave_memory_dataflow_valid
  weft.exec.kernel @rvv_segment2_interleave_memory_dataflow_valid {
    %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
    %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
    %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
    %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: %[[FIELD0:.*]] = weft_rvv.load
      %field0 = weft_rvv.load %src0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: %[[FIELD1:.*]] = weft_rvv.load
      %field1 = weft_rvv.load %src1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: weft_rvv.segment2_store
      // CHECK-SAME: destination_memory_form = "segment2-interleaved-unit-stride-store"
      // CHECK-SAME: field0_role = "segment-field0-input-buffer"
      // CHECK-SAME: field1_role = "segment-field1-input-buffer"
      // CHECK-SAME: segment_count = 2 : i64
      weft_rvv.segment2_store %dst, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_segment_count {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_segment_count attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires segment_count 2 for the bounded segment2 interleave memory hook}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 3 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_duplicate_field_roles {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_duplicate_fields attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires field1_role "segment-field1-input-buffer"}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field0-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_swapped_field_order {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_swapped_fields attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires field0_role "segment-field0-input-buffer"}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field1-input-buffer", field1_role = "segment-field0-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_source_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_source_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires field0 source operand to bind runtime ABI role 'segment-field0-input-buffer'}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_destination_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_destination_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires interleaved destination operand to bind runtime ABI role 'segment-interleaved-output-buffer'}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_source_memory_form {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_source_memory_form attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only source0_memory_form "unit-stride-load"}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "strided-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_segment2_store_reject_bad_memory_form {
    %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
    %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
    %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
    %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      %field0 = weft_rvv.load %src0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      %field1 = weft_rvv.load %src1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // expected-error@+1 {{currently supports only destination_memory_form "segment2-interleaved-unit-stride-store" for weft_rvv.segment2_store}}
      weft_rvv.segment2_store %dst, %field0, %field1, %vl {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_missing_avl_role {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_avl attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %bad_n = weft_rvv.runtime_abi_value {c_name = "bad_n", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      // expected-error@+1 {{requires runtime n/AVL operand to have index type}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %bad_n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value) -> ()
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_segment2_store_reject_mismatched_field_config {
    %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
    %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
    %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
    %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      %field0 = weft_rvv.load %src0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      %field1 = builtin.unrealized_conversion_cast %field0 : !weft_rvv.vector<i32, "m1"> to !weft_rvv.vector<i64, "m1">
      // expected-error@+1 {{requires field0 and field1 operands to have matching generic RVV vector types}}
      weft_rvv.segment2_store %dst, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_segment2_interleave_reject_authority_attr {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_pre_realized_segment2_interleave_bad_authority_attr attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      weft_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, route_id = "rvv-i32m1", segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}
