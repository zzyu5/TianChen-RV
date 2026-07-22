// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s

module {
  weft.exec.kernel @rvv_runtime_scalar_splat_store_kernel {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_runtime_scalar_splat_store attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %broadcast = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %broadcast, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_runtime_scalar_splat_store_kernel_rvv_runtime_scalar_splat_store
// CHECK-SAME: !emitc.opaque<"int32_t">
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK-SAME: !emitc.opaque<"size_t">
// CHECK: weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK: weft_emitc.source_op=weft_rvv.splat role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
// CHECK-NOT: weft_emitc.source_op=weft_rvv.load
// CHECK-NOT: weft_emitc.source_op=weft_rvv.binary
// CHECK: weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
