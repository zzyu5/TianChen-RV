// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

module {
  tcrv.exec.kernel @rvv_i32_add_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.include <"stddef.h">
// CHECK: emitc.include <"stdint.h">
// CHECK: emitc.include <"riscv_vector.h">
// CHECK: emitc.func @tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add
// CHECK: tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
// CHECK: tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// CHECK: tcrv_emitc.source_op=tcrv_rvv.i32_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK: call_opaque "__riscv_vle32_v_i32m1"
// CHECK: tcrv_emitc.source_op=tcrv_rvv.i32_add role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// CHECK: call_opaque "__riscv_vadd_vv_i32m1"
// CHECK: tcrv_emitc.source_op=tcrv_rvv.i32_store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
// CHECK: call_opaque "__riscv_vse32_v_i32m1"
