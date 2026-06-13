// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — widening standalone reduction (the hard mf2 fractional-LMUL
// rung). The source is a typed i16/mf2 vector (loaded via vle16_v_i16mf2 ->
// vint16mf2_t) reduced into an i32/m1 scalar accumulator via the widening
// __riscv_vwredsum_vs_i16mf2_i32m1. The scalar-carry structure is identical to
// the non-widening standalone reduction (out[0] seeded from acc[0] pre-loop,
// read-modify-written each chunk), but the reduce intrinsic crosses the
// i16mf2 -> i32m1 width boundary. This exercises the fractional-LMUL
// TypeConverter rung (i16/mf2 -> vint16mf2_t). Structure-level CHECKs; byte
// identity to the legacy oracle is pinned by the Target/RVV artifact fixture.

module {
  tcrv.exec.kernel @rvv_i16_i32_wred_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_i16_i32_wred attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-standalone-reduce-add:input", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-standalone-reduce-add:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-standalone-reduce-add:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-standalone-reduce-add:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_i16_i32_wred, sew = 32 : i64, source_kernel = "rvv_i16_i32_wred_kernel", status = "selected-lowering-boundary"} {
        %input = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %reduced = tcrv_rvv.standalone_reduce %input, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_i16_i32_wred_kernel_rvv_i16_i32_wred(
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop i32 seed: out[0] = acc[0] (acc is int32_t*, the result-width accumulator).
// CHECK: %[[ACCSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg2,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The i16/mf2 source load lands a vint16mf2_t (fractional-LMUL TypeConverter rung).
// CHECK: %[[INPUT:.*]] = call_opaque "__riscv_vle16_v_i16mf2"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int16_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// In-loop i32 running seed.
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// The widening reduce: i16mf2 source -> i32m1 accumulator/result.
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"(%[[INPUT]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg2, %[[RED]],
// CHECK: return
