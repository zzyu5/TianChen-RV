// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The Stage-3 Gearbox grouped widening-product-reduce-dequant-CLAMP-f32 selected
// body converts through the real DialectConversion. Same unroll-2 main + scalar
// tail accumulator structure as the plain dequantize, but the epilogue clamps
// the scaled f32 lane-0 result to [lower, upper] via splat/compare(vmflt)/
// select(vmerge) at VL=1 before storing. The body carries exactly ONE typed
// product/reduce slice; the conversion expands it unroll_factor (=2) times in
// the main loop and synthesizes the scalar tail loop from the structural attr.

module {
  weft.exec.kernel @grouped_clamp_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @grouped_clamp attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower", c_type = "float", ownership = "target-export-abi-owned", purpose = "lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper", c_type = "float", ownership = "target-export-abi-owned", purpose = "upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64, unroll_factor = 2 : i64} {
        %l0 = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %r0 = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %p0 = weft_rvv.widening_product %l0, %r0, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %red0 = weft_rvv.standalone_reduce %p0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %deq = weft_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        %ls = weft_rvv.splat %lower, %vl : f32, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        %us = weft_rvv.splat %upper, %vl : f32, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        %lcmp = weft_rvv.compare %deq, %ls, %vl {kind = "slt"} : !weft_rvv.vector<f32, "m1">, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<f32, "m1">
        %lsel = weft_rvv.select %lcmp, %ls, %deq, %vl : !weft_rvv.mask<f32, "m1">, !weft_rvv.vector<f32, "m1">, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        %ucmp = weft_rvv.compare %us, %lsel, %vl {kind = "slt"} : !weft_rvv.vector<f32, "m1">, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<f32, "m1">
        %usel = weft_rvv.select %ucmp, %us, %lsel, %vl : !weft_rvv.mask<f32, "m1">, !weft_rvv.vector<f32, "m1">, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %usel, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @weft_emitc_grouped_clamp_kernel_grouped_clamp
// CHECK: %[[ACC:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
// Unroll-2 main loop (step is vlmax*unroll) + scalar tail (two slices then one).
// CHECK: mul
// CHECK: for {{.*}} step
// CHECK: assign {{.*}} to %[[ACC]]
// CHECK: assign {{.*}} to %[[ACC]]
// CHECK: for
// CHECK: assign {{.*}} to %[[ACC]]
// Scalar dequant then VL=1 f32 clamp (splat + vmflt compare + vmerge select).
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m1"
// CHECK: call_opaque "__riscv_vmflt_vv_f32m1_b32"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m1"
// CHECK: call_opaque "__riscv_vmflt_vv_f32m1_b32"
// CHECK: call_opaque "__riscv_vmerge_vvm_f32m1"
// CHECK: call_opaque "__riscv_vse32_v_f32m1"
