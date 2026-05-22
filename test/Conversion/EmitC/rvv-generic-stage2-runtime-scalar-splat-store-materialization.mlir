// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

module {
  tcrv.exec.kernel @rvv_runtime_i32_splat_store_kernel {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_runtime_i32_splat_store attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_i32_splat_store, sew = 32 : i64, source_kernel = "rvv_runtime_i32_splat_store_kernel", status = "selected-lowering-boundary"} {
        %broadcast = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %broadcast, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_runtime_i32_splat_store_kernel_rvv_runtime_i32_splat_store
// CHECK-SAME: !emitc.opaque<"int32_t">
// CHECK-SAME: !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK-SAME: !emitc.opaque<"size_t">
// CHECK: tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK: tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_i32m1
// CHECK-NOT: tcrv_emitc.source_op=tcrv_rvv.load
// CHECK-NOT: tcrv_emitc.source_op=tcrv_rvv.binary
// CHECK: tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
