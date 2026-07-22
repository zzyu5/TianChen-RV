// RUN: weft-opt %s --split-input-file --weft-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: both sections used to assert legacy string-route
// structural scope-limits — "strided lhs load requires lhs-input-stride runtime
// ABI value" (a role-vs-operand convention) and "cannot mix strided memory ops
// with unit-stride load/store" (a memory-form-mixing scope-limit). With the
// string-plan owner retired, the real RVV->emitc DialectConversion lowers the
// actual typed dataflow: it binds each strided load's stride by SSA Value (role
// labels are metadata, not lowering inputs), and it freely lowers a strided-gather
// + unit-store kernel. Both bodies are well-formed typed IR producing type-correct
// vlse/vse intrinsics, so they MATERIALIZE rather than hitting the deleted legacy
// checks.

module {
  weft.exec.kernel @rvv_generic_strided_add_reject_stride_pair {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_generic_strided_add_bad_stride_pair attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      %out_stride = weft_rvv.runtime_abi_value {c_name = "out_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = weft_rvv.strided_load %lhs_ptr, %rhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.strided_load %rhs_ptr, %lhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.strided_store %out_ptr, %sum, %out_stride, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_generic_strided_add_reject_stride_pair_rvv_generic_strided_add_bad_stride_pair
// CHECK: callee=__riscv_vlse32_v_i32m1
// CHECK: callee=__riscv_vsse32_v_i32m1

// -----

module {
  weft.exec.kernel @rvv_generic_strided_add_reject_mixed_memory_form {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_generic_strided_add_mixed_memory attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = weft_rvv.strided_load %lhs_ptr, %lhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.strided_load %rhs_ptr, %rhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_generic_strided_add_reject_mixed_memory_form_rvv_generic_strided_add_mixed_memory
// CHECK: callee=__riscv_vlse32_v_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1
