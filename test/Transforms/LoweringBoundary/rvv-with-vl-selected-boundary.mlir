// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s

module {
  tcrv.exec.kernel @rvv_i32m1_add_selected_boundary {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-i32m1-arithmetic-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_i32_add,
        sew = 32 : i64,
        source_kernel = "rvv_i32m1_add_selected_boundary",
        status = "selected-lowering-boundary"
      } {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {message = "selected add", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_add}
  }

  tcrv.exec.kernel @rvv_i32m1_sub_selected_boundary {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_sub attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-i32m1-arithmetic-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_i32_sub,
        sew = 32 : i64,
        source_kernel = "rvv_i32m1_sub_selected_boundary",
        status = "selected-lowering-boundary"
      } {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %diff = tcrv_rvv.i32_sub %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %diff, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {message = "selected sub", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_sub}
  }

  tcrv.exec.kernel @rvv_i32m1_mul_selected_boundary {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_mul attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-i32m1-arithmetic-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_i32_mul,
        sew = 32 : i64,
        source_kernel = "rvv_i32m1_mul_selected_boundary",
        status = "selected-lowering-boundary"
      } {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %product = tcrv_rvv.i32_mul %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %product, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {message = "selected mul", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_mul}
  }
}

// CHECK-LABEL: tcrv.exec.kernel @rvv_i32m1_add_selected_boundary
// CHECK: tcrv_rvv.with_vl
// CHECK-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
// CHECK-SAME: selected_variant = @rvv_i32_add
// CHECK-SAME: source_kernel = "rvv_i32m1_add_selected_boundary"
// CHECK-SAME: status = "selected-lowering-boundary"
// CHECK: tcrv_rvv.i32_add
// CHECK-LABEL: tcrv.exec.kernel @rvv_i32m1_sub_selected_boundary
// CHECK: tcrv_rvv.with_vl
// CHECK-SAME: selected_variant = @rvv_i32_sub
// CHECK-SAME: source_kernel = "rvv_i32m1_sub_selected_boundary"
// CHECK: tcrv_rvv.i32_sub
// CHECK-LABEL: tcrv.exec.kernel @rvv_i32m1_mul_selected_boundary
// CHECK: tcrv_rvv.with_vl
// CHECK-SAME: selected_variant = @rvv_i32_mul
// CHECK-SAME: source_kernel = "rvv_i32m1_mul_selected_boundary"
// CHECK: tcrv_rvv.i32_mul
