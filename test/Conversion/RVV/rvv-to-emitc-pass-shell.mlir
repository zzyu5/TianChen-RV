// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass currently stands up the DialectConversion
// harness (TypeConverter + ConversionTarget + applyPartialConversion) with zero
// conversion patterns, so it is an additive structural no-op: the beachhead
// generic typed tcrv_rvv body must round-trip UNCHANGED.

module {
  tcrv.exec.kernel @rvv_generic_add_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_generic_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
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
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// The pass adds no emitc and converts nothing.
// CHECK-NOT: emitc.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: tcrv.exec.kernel @rvv_generic_add_kernel
// CHECK: tcrv.exec.variant @rvv_generic_add
// CHECK: %[[LHS_PTR:.*]] = tcrv_rvv.runtime_abi_value {{.*}}role = "lhs-input-buffer"{{.*}} : !tcrv_rvv.runtime_abi_value
// CHECK: %[[RHS_PTR:.*]] = tcrv_rvv.runtime_abi_value {{.*}}role = "rhs-input-buffer"{{.*}} : !tcrv_rvv.runtime_abi_value
// CHECK: %[[OUT_PTR:.*]] = tcrv_rvv.runtime_abi_value {{.*}}role = "output-buffer"{{.*}} : !tcrv_rvv.runtime_abi_value
// CHECK: %[[N:.*]] = tcrv_rvv.runtime_abi_value {{.*}}role = "runtime-element-count"{{.*}} : index
// CHECK: %[[VL:.*]] = tcrv_rvv.setvl %[[N]] {{.*}} : index -> !tcrv_rvv.vl
// CHECK: tcrv_rvv.with_vl %[[VL]]
// CHECK: %[[LHS:.*]] = tcrv_rvv.load %[[LHS_PTR]], %[[VL]] : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// CHECK: %[[RHS:.*]] = tcrv_rvv.load %[[RHS_PTR]], %[[VL]] : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// CHECK: %[[SUM:.*]] = tcrv_rvv.binary %[[LHS]], %[[RHS]], %[[VL]] {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// CHECK: tcrv_rvv.store %[[OUT_PTR]], %[[SUM]], %[[VL]] : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
