// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-rvv-materialize-gearbox-schedules | FileCheck %s

// CHECK-LABEL: weft.exec.kernel @rvv_gearbox_dequantize_i32_to_f32_kernel
// CHECK: weft_rvv.with_vl
// CHECK-SAME: weft_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]"
// CHECK-SAME: weft_rvv.gearbox.dest_lmul = "m1"
// CHECK-SAME: weft_rvv.gearbox.dest_sew = 32 : i64
// CHECK-SAME: weft_rvv.gearbox.legality_scope = "typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl"
// CHECK-SAME: weft_rvv.gearbox.operation = "dequantize_i32_to_f32"
// CHECK-SAME: weft_rvv.gearbox.runtime_avl_source = "runtime_abi:n"
// CHECK-SAME: weft_rvv.gearbox.schedule_id = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: weft_rvv.gearbox.selected_candidate = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: weft_rvv.gearbox.selection_reason = "select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl"
// CHECK-SAME: weft_rvv.gearbox.selector = "static-dequantize-i32-to-f32-e32-m1-u2"
// CHECK-SAME: weft_rvv.gearbox.source = "rvv-gearbox-static-pass.v1"
// CHECK-SAME: weft_rvv.gearbox.source_lmul = "m1"
// CHECK-SAME: weft_rvv.gearbox.source_sew = 32 : i64
// CHECK-SAME: weft_rvv.gearbox.unroll = 2 : i64
// CHECK-SAME: weft_rvv.gearbox.vl_policy = "runtime-avl-two-slice-setvl"
// CHECK: weft_rvv.dequantize
// CHECK-SAME: weft_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]"
// CHECK-SAME: weft_rvv.gearbox.dest_lmul = "m1"
// CHECK-SAME: weft_rvv.gearbox.dest_sew = 32 : i64
// CHECK-SAME: weft_rvv.gearbox.legality_scope = "typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl"
// CHECK-SAME: weft_rvv.gearbox.operation = "dequantize_i32_to_f32"
// CHECK-SAME: weft_rvv.gearbox.runtime_avl_source = "runtime_abi:n"
// CHECK-SAME: weft_rvv.gearbox.schedule_id = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: weft_rvv.gearbox.selected_candidate = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: weft_rvv.gearbox.selection_reason = "select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl"
// CHECK-SAME: weft_rvv.gearbox.selector = "static-dequantize-i32-to-f32-e32-m1-u2"
// CHECK-SAME: weft_rvv.gearbox.source = "rvv-gearbox-static-pass.v1"
// CHECK-SAME: weft_rvv.gearbox.source_lmul = "m1"
// CHECK-SAME: weft_rvv.gearbox.source_sew = 32 : i64
// CHECK-SAME: weft_rvv.gearbox.unroll = 2 : i64
// CHECK-SAME: weft_rvv.gearbox.vl_policy = "runtime-avl-two-slice-setvl"

module {
  weft.exec.kernel @rvv_gearbox_dequantize_i32_to_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_gearbox_dequantize_i32_to_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_gearbox_dequantize_i32_to_f32, sew = 32 : i64, source_kernel = "rvv_gearbox_dequantize_i32_to_f32_kernel", status = "selected-lowering-boundary"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}
