// RUN: not tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_cmp_select_broadcast_rejected {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_cmp_select_broadcast attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_i32_cmp_select_broadcast, sew = 32 : i64, source_kernel = "rvv_cmp_select_broadcast_rejected", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_broadcast_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %mask = tcrv_rvv.i32_cmp_eq %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1_mask
        %selected = tcrv_rvv.i32_select %mask, %lhs, %rhs, %vl : !tcrv_rvv.i32m1_mask, !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %out_ptr, %selected, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: legacy selected-body op 'tcrv_rvv.i32_load' is fail-closed
// CHECK-SAME: generic tcrv_rvv.load, tcrv_rvv.broadcast_load, tcrv_rvv.splat, tcrv_rvv.strided_load, tcrv_rvv.binary, tcrv_rvv.index_load, tcrv_rvv.indexed_load, tcrv_rvv.segment2_load, tcrv_rvv.segment2_store, tcrv_rvv.indexed_store, tcrv_rvv.mask_load, tcrv_rvv.compare, tcrv_rvv.masked_binary, tcrv_rvv.select, tcrv_rvv.reduce, tcrv_rvv.standalone_reduce, tcrv_rvv.masked_standalone_reduce, tcrv_rvv.macc, tcrv_rvv.widening_convert, tcrv_rvv.move, tcrv_rvv.widening_dot_reduce, tcrv_rvv.masked_widening_dot_reduce, tcrv_rvv.masked_move, tcrv_rvv.masked_load, tcrv_rvv.masked_strided_load, tcrv_rvv.masked_store, tcrv_rvv.masked_strided_store, tcrv_rvv.store, and tcrv_rvv.strided_store
