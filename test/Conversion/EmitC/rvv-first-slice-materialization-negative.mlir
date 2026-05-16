// RUN: not tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_i32m2_rejected {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32m2_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m2",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m2",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m2, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: bounded RVV EmitC route supports only SEW32 LMUL m1 i32 add
