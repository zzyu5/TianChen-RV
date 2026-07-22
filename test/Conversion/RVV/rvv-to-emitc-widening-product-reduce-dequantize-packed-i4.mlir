// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The Stage-3 Gearbox packed-i4 widening-product-reduce-dequantize-f32 selected
// body converts through the real DialectConversion. The signed-i4 nibble unpack
// is now a typed op (weft_rvv.packed_i4_nibble_unpack_product) whose lowering is
// the FIXED vsll/vsra/vwmul/vsra/vwmacc intrinsic chain -- the conversion never
// reads operand_form / unpack_intent candidate-mirror strings to choose it.
// packed-i4 is the single-loop (unroll=1) candidate: one product/reduce slice
// over the runtime VL chunk loop, the i32 accumulator carried as a
// function-scoped emitc.variable, then the scalar dequant epilogue.

module {
  weft.exec.kernel @packed_i4_dequantize_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @packed_i4_dequantize attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %l0 = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %r0 = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %p0 = weft_rvv.packed_i4_nibble_unpack_product %l0, %r0, %vl {kind = "signed_packed_i4_nibble_unpack_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %red0 = weft_rvv.standalone_reduce %p0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %deq = weft_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %deq, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @weft_emitc_packed_i4_dequantize_kernel_packed_i4_dequantize
// CHECK: %[[ACC:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: assign {{.*}} to %[[ACC]]
// Single runtime-VL chunk loop (unroll=1: no vlmax*2 step, no tail loop).
// CHECK: for
// The fixed signed-i4 nibble-unpack widening-product intrinsic chain.
// CHECK: call_opaque "__riscv_vsll_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf4"
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i16mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf4"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// CHECK: assign {{.*}} to %[[ACC]]
// Scalar-extract dequant epilogue.
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m1"
