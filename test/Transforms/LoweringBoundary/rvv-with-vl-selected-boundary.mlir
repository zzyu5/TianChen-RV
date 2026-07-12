// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s

module {
  weft.exec.kernel @rvv_i32m1_add_selected_boundary {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_i32_add,
        sew = 32 : i64,
        source_kernel = "rvv_i32m1_add_selected_boundary",
        status = "selected-lowering-boundary"
      } {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.diagnostic {message = "selected add", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_add}
  }

  weft.exec.kernel @rvv_i32m1_sub_selected_boundary {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_i32_sub attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_i32_sub,
        sew = 32 : i64,
        source_kernel = "rvv_i32m1_sub_selected_boundary",
        status = "selected-lowering-boundary"
      } {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %diff = weft_rvv.binary %lhs, %rhs, %vl {kind = "sub"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %diff, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.diagnostic {message = "selected sub", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_sub}
  }

  weft.exec.kernel @rvv_i32m1_mul_selected_boundary {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_i32_mul attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_i32_mul,
        sew = 32 : i64,
        source_kernel = "rvv_i32m1_mul_selected_boundary",
        status = "selected-lowering-boundary"
      } {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %product = weft_rvv.binary %lhs, %rhs, %vl {kind = "mul"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.diagnostic {message = "selected mul", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_i32_mul}
  }
}

// CHECK-LABEL: weft.exec.kernel @rvv_i32m1_add_selected_boundary
// CHECK: weft_rvv.with_vl
// CHECK-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
// CHECK-SAME: selected_variant = @rvv_i32_add
// CHECK-SAME: source_kernel = "rvv_i32m1_add_selected_boundary"
// CHECK-SAME: status = "selected-lowering-boundary"
// CHECK: weft_rvv.binary
// CHECK-LABEL: weft.exec.kernel @rvv_i32m1_sub_selected_boundary
// CHECK: weft_rvv.with_vl
// CHECK-SAME: selected_variant = @rvv_i32_sub
// CHECK-SAME: source_kernel = "rvv_i32m1_sub_selected_boundary"
// CHECK: weft_rvv.binary
// CHECK-LABEL: weft.exec.kernel @rvv_i32m1_mul_selected_boundary
// CHECK: weft_rvv.with_vl
// CHECK-SAME: selected_variant = @rvv_i32_mul
// CHECK-SAME: source_kernel = "rvv_i32m1_mul_selected_boundary"
// CHECK: weft_rvv.binary
