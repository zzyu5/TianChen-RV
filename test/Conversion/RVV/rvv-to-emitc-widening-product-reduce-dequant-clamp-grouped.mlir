// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The Stage-3 Gearbox grouped widening-product-reduce-dequant-CLAMP-f32 selected
// body converts through the real DialectConversion. Same unroll-2 main + scalar
// tail accumulator structure as the plain dequantize, but the epilogue clamps
// the scaled f32 lane-0 result to [lower, upper] via splat/compare(vmflt)/
// select(vmerge) at VL=1 before storing. The body carries exactly ONE typed
// product/reduce slice; the conversion expands it unroll_factor (=2) times in
// the main loop and synthesizes the scalar tail loop from the structural attr.

module {
  tcrv.exec.kernel @grouped_clamp_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @grouped_clamp attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower", c_type = "float", ownership = "target-export-abi-owned", purpose = "lower", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper", c_type = "float", ownership = "target-export-abi-owned", purpose = "upper", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], selected_path_role = "dispatch case", selected_variant = @grouped_clamp, sew = 32 : i64, source_kernel = "grouped_clamp_kernel", status = "selected-lowering-boundary", unroll_factor = 2 : i64} {
        %l0 = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %r0 = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %p0 = tcrv_rvv.widening_product %l0, %r0, %vl {kind = "signed_widening_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %red0 = tcrv_rvv.standalone_reduce %p0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %deq = tcrv_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %ls = tcrv_rvv.splat %lower, %vl : f32, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %us = tcrv_rvv.splat %upper, %vl : f32, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %lcmp = tcrv_rvv.compare %deq, %ls, %vl {kind = "slt"} : !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<f32, "m1">
        %lsel = tcrv_rvv.select %lcmp, %ls, %deq, %vl : !tcrv_rvv.mask<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %ucmp = tcrv_rvv.compare %us, %lsel, %vl {kind = "slt"} : !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<f32, "m1">
        %usel = tcrv_rvv.select %ucmp, %us, %lsel, %vl : !tcrv_rvv.mask<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %usel, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @tcrv_emitc_grouped_clamp_kernel_grouped_clamp
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
