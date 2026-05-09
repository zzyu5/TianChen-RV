// RUN: not tcrv-opt %s --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_missing_property_evidence {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
  }
}

// CHECK: error: TianChen-RV plugin variant materialization for kernel @rvv_missing_property_evidence collected no viable plugin proposals; recoverable plugin declines in registration order: rvv-plugin:
// CHECK-SAME: capability id 'rvv' requires preserved property 'architecture'
