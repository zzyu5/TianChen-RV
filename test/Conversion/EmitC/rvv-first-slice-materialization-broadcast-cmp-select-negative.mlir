// RUN: not weft-opt %s --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --implicit-check-not="emitc.func"

module {
  weft.exec.kernel @rvv_cmp_select_broadcast_rejected {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_i32_cmp_select_broadcast attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_cmp_select_broadcast, sew = 32 : i64, source_kernel = "rvv_cmp_select_broadcast_rejected", status = "selected-lowering-boundary"} {
        %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
        %rhs = weft_rvv.i32_broadcast_load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
        %mask = weft_rvv.i32_cmp_eq %lhs, %rhs, %vl : !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1_mask
        %selected = weft_rvv.i32_select %mask, %lhs, %rhs, %vl : !weft_rvv.i32m1_mask, !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
        weft_rvv.i32_store %out_ptr, %selected, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.i32m1, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: no registered backend emission driver fully legalizes the selected variant @rvv_i32_cmp_select_broadcast body to EmitC
