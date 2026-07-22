// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — standalone (scalar-carry-through-memory) reduction family
// conversion. Unlike the per-chunk `reduce` family, the standalone reduction
// seeds the scalar accumulator into out[0] BEFORE the loop (out[0] = acc[0]),
// then read-modify-writes out[0] each chunk: read out[0] -> splat into a lane-0
// m1 seed -> __riscv_vredsum_vs_i32m1_i32m1(input, seed, vl) -> store lane 0
// back to the output BASE (no `+ i`) with VL=1. Structure-level CHECKs (token
// regex over the structured emitc), not byte-exact C; byte-identity to the
// hardware-validated legacy oracle is pinned by the Target/RVV artifact fixture.

module {
  weft.exec.kernel @explicit_selected_body_standalone_reduce_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_standalone_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:scalar-output", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-standalone-reduce-add:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %input = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %reduced = weft_rvv.standalone_reduce %input, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_explicit_selected_body_standalone_reduce_add_kernel_explicit_selected_body_rvv_standalone_reduce_add(
// CHECK: %[[VLMAX:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop seed: out[0] = acc[0] (read acc[0], splat into lane-0 m1, store VL=1).
// CHECK: %[[ACC0:.*]] = subscript %arg1
// CHECK: %[[ACCSCALAR:.*]] = load %[[ACC0]]
// CHECK: %[[ONE:.*]] = literal "1"
// CHECK: %[[SEED0:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]], %[[ONE]])
// CHECK: %[[ONESTORE0:.*]] = literal "1"
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg2, %[[SEED0]], %[[ONESTORE0]])
// for-loop over the runtime chunks.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// CHECK: %[[INPUT:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// In-loop running seed: out[0] -> splat into lane-0 m1.
// CHECK: %[[OUT0:.*]] = subscript %arg2
// CHECK: %[[OUTSCALAR:.*]] = load %[[OUT0]]
// CHECK: %[[ONE2:.*]] = literal "1"
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"(%[[OUTSCALAR]], %[[ONE2]])
// The horizontal reduction over the chunk + lane-0 store to the output base.
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vredsum_vs_i32m1_i32m1"(%[[INPUT]], %[[SEED]], %[[BODYVL]])
// CHECK: %[[ONESTORE:.*]] = literal "1"
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg2, %[[RED]], %[[ONESTORE]])
// CHECK: return
