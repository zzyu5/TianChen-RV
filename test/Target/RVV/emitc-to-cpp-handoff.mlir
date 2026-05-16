// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | tcrv-translate --tcrv-rvv-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE

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

// HELP: --tcrv-rvv-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stddef.h>
// SOURCE: #include <stdint.h>
// SOURCE: #include <riscv_vector.h>
// SOURCE: void tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(
// SOURCE-SAME: const int32_t*
// SOURCE-SAME: const int32_t*
// SOURCE-SAME: int32_t*
// SOURCE-SAME: size_t
// SOURCE: tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: __riscv_vle32_v_i32m1
// SOURCE: tcrv_emitc.source_op=tcrv_rvv.i32_add role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: __riscv_vse32_v_i32m1
