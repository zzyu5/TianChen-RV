// RUN: not tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_segment2_interleave_reject_field_source_swap {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_segment2_interleave_swapped_fields attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src0 = tcrv_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src1 = tcrv_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_segment2_interleave_swapped_fields, sew = 32 : i64, source_kernel = "rvv_segment2_interleave_reject_field_source_swap", status = "selected-lowering-boundary"} {
        %field0 = tcrv_rvv.load %src0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %field1 = tcrv_rvv.load %src1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.segment2_store %dst, %field1, %field0, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: bounded generic RVV segment2 interleave route requires segment2_store to consume matching field0 and field1 load results
