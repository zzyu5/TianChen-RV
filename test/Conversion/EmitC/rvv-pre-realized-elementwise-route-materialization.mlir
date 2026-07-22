// RUN: weft-opt %s --split-input-file --weft-materialize-emitc-lowerable-routes | FileCheck %s

// A pre-realized RVV source body is an input to family construction.  The
// family realizes its final setvl/with_vl body before the construction-blind
// EmitC backend consumes it; no route/provider fallback participates.

module {
  weft.exec.kernel @pre_route_add_kernel {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_pre_route_add attributes { origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic> } {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      weft_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// CHECK-LABEL: emitc.func @weft_emitc_pre_route_add_kernel_rvv_pre_route_add
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// CHECK: call_opaque "__riscv_vadd_vv_i32m1"
// CHECK: call_opaque "__riscv_vse32_v_i32m1"
// CHECK-NOT: typed_binary_pre_realized_body
