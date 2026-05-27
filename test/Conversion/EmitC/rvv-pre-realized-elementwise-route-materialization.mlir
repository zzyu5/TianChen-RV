// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// The RVV provider route entry should still invoke plugin-owned selected-body
// realization for bounded direct route-entry arithmetic bodies. Plain
// compare/select is intentionally not covered here; it must pass through the
// selected lowering-boundary producer before EmitC route materialization.

module {
  tcrv.exec.kernel @pre_route_add_kernel {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_pre_route_add attributes { origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic> } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// CHECK-LABEL: emitc.func @tcrv_emitc_pre_route_add_kernel_rvv_pre_route_add
// CHECK: tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK: tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK: tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK: tcrv_emitc.source_op=tcrv_rvv.binary role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// CHECK: tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
