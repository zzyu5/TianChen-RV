// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — signed widening dot-product reduction (the scalar-carry
// contraction). The single tcrv_rvv.widening_dot_reduce op fuses a widened
// product (i16/mf2 x i16/mf2 -> i32/m1 via __riscv_vwmul_vv_i32m1) with a plain
// horizontal __riscv_vredsum_vs_i32m1_i32m1, accumulating one i32 scalar through
// the output cell out[0] across runtime VL chunks: a pre-loop seed out[0]=acc[0]
// (vmv_v_x splat + vse32 to base VL=1), and each chunk a running-seed read back
// from out[0] + reduce + lane-0 store to base. Reuses the shared scalar-carry
// standalone-reduction machinery. Structure-level CHECKs; byte identity is
// pinned by the Target/RVV fixture.

module {
  tcrv.exec.kernel @rvv_widening_dot_reduce_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_widening_dot_reduce attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_widening_dot_reduce, sew = 32 : i64, source_kernel = "rvv_widening_dot_reduce_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %sum = tcrv_rvv.widening_dot_reduce %a, %b, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_widening_dot_reduce_add", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_kernel_rvv_widening_dot_reduce(
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop scalar seed: out[0] = acc[0] (acc is the i32 result-width accumulator).
// CHECK: %[[SEEDSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"(%[[SEEDSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg3,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// i16/mf2 dot-product input loads.
// CHECK: %[[LHS:.*]] = call_opaque "__riscv_vle16_v_i16mf2"
// CHECK: %[[RHS:.*]] = call_opaque "__riscv_vle16_v_i16mf2"
// Widened product (result type i32/m1), running seed from out[0], plain reduce.
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i32m1"(%[[LHS]], %[[RHS]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint16mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vredsum_vs_i32m1_i32m1"(%[[PROD]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg3, %[[RED]],
// CHECK: return
