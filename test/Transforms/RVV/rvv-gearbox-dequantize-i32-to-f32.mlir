// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-rvv-materialize-gearbox-schedules | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @rvv_gearbox_dequantize_i32_to_f32_kernel
// CHECK: tcrv_rvv.with_vl
// CHECK-SAME: tcrv_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]"
// CHECK-SAME: tcrv_rvv.gearbox.dest_lmul = "m1"
// CHECK-SAME: tcrv_rvv.gearbox.dest_sew = 32 : i64
// CHECK-SAME: tcrv_rvv.gearbox.legality_scope = "typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl"
// CHECK-SAME: tcrv_rvv.gearbox.operation = "dequantize_i32_to_f32"
// CHECK-SAME: tcrv_rvv.gearbox.runtime_avl_source = "runtime_abi:n"
// CHECK-SAME: tcrv_rvv.gearbox.schedule_id = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: tcrv_rvv.gearbox.selected_candidate = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: tcrv_rvv.gearbox.selection_reason = "select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl"
// CHECK-SAME: tcrv_rvv.gearbox.selector = "static-dequantize-i32-to-f32-e32-m1-u2"
// CHECK-SAME: tcrv_rvv.gearbox.source = "rvv-gearbox-static-pass.v1"
// CHECK-SAME: tcrv_rvv.gearbox.source_lmul = "m1"
// CHECK-SAME: tcrv_rvv.gearbox.source_sew = 32 : i64
// CHECK-SAME: tcrv_rvv.gearbox.unroll = 2 : i64
// CHECK-SAME: tcrv_rvv.gearbox.vl_policy = "runtime-avl-two-slice-setvl"
// CHECK: tcrv_rvv.dequantize
// CHECK-SAME: tcrv_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]"
// CHECK-SAME: tcrv_rvv.gearbox.dest_lmul = "m1"
// CHECK-SAME: tcrv_rvv.gearbox.dest_sew = 32 : i64
// CHECK-SAME: tcrv_rvv.gearbox.legality_scope = "typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl"
// CHECK-SAME: tcrv_rvv.gearbox.operation = "dequantize_i32_to_f32"
// CHECK-SAME: tcrv_rvv.gearbox.runtime_avl_source = "runtime_abi:n"
// CHECK-SAME: tcrv_rvv.gearbox.schedule_id = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: tcrv_rvv.gearbox.selected_candidate = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"
// CHECK-SAME: tcrv_rvv.gearbox.selection_reason = "select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl"
// CHECK-SAME: tcrv_rvv.gearbox.selector = "static-dequantize-i32-to-f32-e32-m1-u2"
// CHECK-SAME: tcrv_rvv.gearbox.source = "rvv-gearbox-static-pass.v1"
// CHECK-SAME: tcrv_rvv.gearbox.source_lmul = "m1"
// CHECK-SAME: tcrv_rvv.gearbox.source_sew = 32 : i64
// CHECK-SAME: tcrv_rvv.gearbox.unroll = 2 : i64
// CHECK-SAME: tcrv_rvv.gearbox.vl_policy = "runtime-avl-two-slice-setvl"

module {
  tcrv.exec.kernel @rvv_gearbox_dequantize_i32_to_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_gearbox_dequantize_i32_to_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-dequantize-i32-to-f32:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_gearbox_dequantize_i32_to_f32, sew = 32 : i64, source_kernel = "rvv_gearbox_dequantize_i32_to_f32_kernel", status = "selected-lowering-boundary"} {
        %source = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %result = tcrv_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
