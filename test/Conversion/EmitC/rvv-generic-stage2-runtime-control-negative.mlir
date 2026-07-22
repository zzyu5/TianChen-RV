// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: the legacy string route rejected an AVL runtime ABI
// parameter not literally spelled 'n' ("AVL runtime ABI parameter must be named
// 'n'") — a string-route NAMING convention. With the string-plan owner retired,
// the real RVV->emitc DialectConversion binds the AVL by SSA Value (the operand
// of weft_rvv.setvl), not by the parameter's c_name, so a correctly-roled
// runtime-element-count value named "runtime_count" materializes faithfully. The
// generated control loop is identical regardless of the parameter spelling.
module {
  weft.exec.kernel @rvv_scalar_broadcast_wrong_runtime_n_name_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_scalar_broadcast_wrong_runtime_n_name attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "runtime_count", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_scalar_broadcast_wrong_runtime_n_name_rejected_rvv_scalar_broadcast_wrong_runtime_n_name
// CHECK: callee=__riscv_vsetvl_e32m1
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vadd_vv_i32m1
