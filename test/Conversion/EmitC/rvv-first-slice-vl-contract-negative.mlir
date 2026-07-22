// RUN: not weft-opt %s --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --implicit-check-not="emitc.func"

module {
  weft.exec.kernel @rvv_i32m1_with_vl_not_setvl_result {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      %other_vl = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vl
      weft_rvv.with_vl %other_vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = weft_rvv.i32_load %lhs_ptr, %other_vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
        %rhs = weft_rvv.i32_load %rhs_ptr, %other_vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
        %sum = weft_rvv.i32_add %lhs, %rhs, %other_vl : !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
        weft_rvv.i32_store %out_ptr, %sum, %other_vl : !weft_rvv.runtime_abi_value, !weft_rvv.i32m1, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: selected RVV body config/VL structure requires weft_rvv.with_vl to consume the visible weft_rvv.setvl result
