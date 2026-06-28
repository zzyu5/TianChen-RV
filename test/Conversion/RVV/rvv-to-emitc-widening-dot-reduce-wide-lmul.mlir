// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
//
// The 2nd kernel family (signed i16 widening dot-reduce) N3 resource-aware
// max-legal-LMUL DEFERRED-WIDE chain -- the measured ssh-rvv winner
// (dot_wide_deferred). The deferred-wide algorithm is modeled FIRST-CLASS in the
// typed body via tcrv_rvv.deferred_accumulate (the i32m8 NON-widening vadd.vv
// deferred vector accumulate); emission follows that op's identity, NOT metadata
// (I5).
//
// The typed body (the measured winner's structure):
//   load i16/m4 -> vwmul i32/m8 product (SINGLE widening) ->
//   tcrv_rvv.deferred_accumulate (DEFERRED vadd.vv into an i32/m8 vector
//   accumulator) -> ONE trailing tcrv_rvv.standalone_reduce (i32m8 -> i32m1
//   vredsum) + scalar acc[0] add -> i32 lane-0 store. The enclosing with_vl is
//   the strip config SEW16/m4; the i32m8 accumulator is zero-seeded (NOT
//   splatted from acc[0]); acc[0] is added as a SCALAR after the single vredsum.
//   There is NO per-iteration vredsum (the narrow dot-reduce default).

module {
  tcrv.exec.kernel @wide_dot_reduce_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @wide_dot_reduce attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m4", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], selected_path_role = "dispatch case", selected_variant = @wide_dot_reduce, sew = 16 : i64, source_kernel = "wide_dot_reduce_kernel", status = "selected-lowering-boundary", unroll_factor = 1 : i64} {
        %l0 = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %r0 = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %p0 = tcrv_rvv.widening_product %l0, %r0, %vl {kind = "signed_widening_product", product_relation = "signed-i16m4xi16m4-to-i32m8"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
        %a0 = tcrv_rvv.deferred_accumulate %p0, %vl {accumulate_relation = "signed-i32m8-into-i32m8-deferred-add", kind = "signed_deferred_accumulate_add"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
        %red0 = tcrv_rvv.standalone_reduce %a0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %red0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @tcrv_emitc_wide_dot_reduce_kernel_wide_dot_reduce
// The i32 accumulator is a function-scoped wide (m8) vector, seeded to ZERO at
// the accumulator's own VLMAX (NOT splatted from acc[0]).
// CHECK: %[[ACC:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m8_t">>
// CHECK: call_opaque "__riscv_vsetvlmax_e32m8"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m8"
// CHECK: assign {{.*}} to %[[ACC]] : <!emitc.opaque<"vint32m8_t">>
// The single (unroll=1) main loop: wide i16/m4 loads, a SINGLE-step i32/m8
// widening product (vwmul), and a DEFERRED NON-widening vadd.vv accumulate into
// the m8 vector accumulator -- there is NO per-iteration vredsum.
// CHECK: for
// CHECK: call_opaque "__riscv_vle16_v_i16m4"
// CHECK: call_opaque "__riscv_vle16_v_i16m4"
// CHECK: call_opaque "__riscv_vwmul_vv_i32m8"
// CHECK: call_opaque "__riscv_vadd_vv_i32m8"
// CHECK: assign {{.*}} to %[[ACC]]
// CHECK-NOT: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// After the loop: ONE trailing vredsum folds the i32m8 accumulator, then acc[0]
// is added as a SCALAR, then the i32 lane-0 store (NO dequant / scale).
// CHECK: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: add
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vse32_v_i32m1"
// CHECK-NOT: __riscv_vfmv_v_f
