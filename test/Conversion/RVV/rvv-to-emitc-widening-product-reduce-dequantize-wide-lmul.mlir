// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
//
// The N3 resource-aware max-legal-LMUL low-precision contraction -- the DEFERRED
// wide-LMUL chain that the measured ssh-rvv winner (var_v_m2_a1.c) realizes and
// that the resource-aware LMUL selection (RVVGearboxSchedule.h enumerate/prune/
// select) is designed to pick. The deferred-wide algorithm is modeled FIRST-
// CLASS in the typed body via tcrv_rvv.widening_accumulate (the i32m8 deferred
// vector accumulate); emission follows that op's identity, NOT metadata (I5).
//
// The typed body (the measured winner's structure):
//   load i8/m2  ->  vwmul i16/m4 product  ->  tcrv_rvv.widening_accumulate
//   (DEFERRED vwadd.wv into an i32/m8 vector accumulator)  ->  ONE trailing
//   tcrv_rvv.standalone_reduce (i32m8 -> i32m1 vredsum) + scalar acc[0] add ->
//   i32->f32 dequant. The enclosing with_vl is the strip config SEW8/m2; the
//   i32m8 vector accumulator is zero-seeded (NOT splatted from acc[0]); acc[0]
//   is added as a SCALAR after the single vredsum. There is NO per-iteration
//   vwredsum -- a per-iteration vwredsum at wide LMUL was NOT the measured
//   winner.

module {
  tcrv.exec.kernel @wide_dequantize_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @wide_dequantize attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], selected_path_role = "dispatch case", selected_variant = @wide_dequantize, sew = 8 : i64, source_kernel = "wide_dequantize_kernel", status = "selected-lowering-boundary", unroll_factor = 1 : i64} {
        %l0 = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %r0 = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %p0 = tcrv_rvv.widening_product %l0, %r0, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %a0 = tcrv_rvv.widening_accumulate %p0, %vl {accumulate_relation = "signed-i16m4-into-i32m8-deferred-add", kind = "signed_widening_accumulate_add"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
        %red0 = tcrv_rvv.standalone_reduce %a0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %deq = tcrv_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %deq, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @tcrv_emitc_wide_dequantize_kernel_wide_dequantize
// The i32 accumulator is a function-scoped wide (m8) vector, seeded to ZERO at
// the accumulator's own VLMAX (NOT splatted from acc[0]).
// CHECK: %[[ACC:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m8_t">>
// CHECK: call_opaque "__riscv_vsetvlmax_e32m8"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m8"
// CHECK: assign {{.*}} to %[[ACC]] : <!emitc.opaque<"vint32m8_t">>
// The single (unroll=1) main loop: wide i8/m2 loads, i16/m4 widening product,
// and a DEFERRED vwadd.wv accumulate into the m8 vector accumulator -- there is
// NO per-iteration vwredsum.
// CHECK: for
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m8"
// CHECK: assign {{.*}} to %[[ACC]]
// CHECK-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// The epilogue folds the wide accumulator with ONE trailing vredsum, extracts
// the lane-0 scalar, then adds the acc[0] seed as a SCALAR.
// CHECK: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: add
// Scalar-extract dequant epilogue (vfmv -> *scale -> store).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m1"
