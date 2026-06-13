// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The Stage-3 Gearbox grouped widening-product-reduce-dequantize-f32 selected
// body converts through the real DialectConversion. This is the typed target
// structure the selected-body realization emits (region markers + cross-region
// handoff retired; the unroll-2 main loop + scalar tail made first-class as
// op-intrinsic structure): a SINGLE with_vl scope carrying
//   unroll_factor = 2  (main loop steps by vlmax*2; the body carries exactly
//                        ONE typed product/reduce slice -- the conversion expands
//                        it unroll_factor times in the main loop and synthesizes
//                        the scalar tail loop over the remainder, both from this
//                        structural attr)
// followed by the i32->f32 dequant epilogue (run once after the loops).
//
// The i32 accumulator dot_acc_vec crosses the loop/epilogue boundary as a
// function-scoped emitc.variable (seeded from acc[0], reassigned per slice,
// scalar-extracted in the epilogue). The conversion reads ONLY op-intrinsic
// structural attrs (with_vl.unroll_factor, standalone_reduce.accumulator_layout/
// result_layout, dequantize relation/kind); it never reads operand_form /
// unpack_intent / realized_unroll_factor candidate mirrors.

module {
  tcrv.exec.kernel @grouped_dequantize_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @grouped_dequantize attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], selected_path_role = "dispatch case", selected_variant = @grouped_dequantize, sew = 32 : i64, source_kernel = "grouped_dequantize_kernel", status = "selected-lowering-boundary", unroll_factor = 2 : i64} {
        %l0 = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %r0 = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %p0 = tcrv_rvv.widening_product %l0, %r0, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %red0 = tcrv_rvv.standalone_reduce %p0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %deq = tcrv_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %deq, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @tcrv_emitc_grouped_dequantize_kernel_grouped_dequantize
// The i32 accumulator is a function-scoped mutable variable seeded from acc[0].
// CHECK: %[[ACC:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: assign {{.*}} to %[[ACC]] : <!emitc.opaque<"vint32m1_t">>
// The main loop steps by vlmax*2 (unroll-2) and carries two reduce slices.
// CHECK: %[[STEP:.*]] = mul {{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: for {{.*}} step %[[STEP]]
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK: assign {{.*}} to %[[ACC]]
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK: assign {{.*}} to %[[ACC]]
// The scalar tail loop steps by vlmax (one slice over the remainder).
// CHECK: for
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// Scalar-extract dequant epilogue (NOT the vector vfcvt path).
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m1"
