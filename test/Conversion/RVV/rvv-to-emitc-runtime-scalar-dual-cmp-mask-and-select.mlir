// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — runtime-scalar DUAL compare + mask_and + select. Two
// weft_rvv.compare ops produce two predicate masks (each lane compared against a
// runtime-scalar splat via vmsle), weft_rvv.mask_and composes them with
// __riscv_vmand_mm_b32, and weft_rvv.select merges the true/false vectors under
// the composed mask via __riscv_vmerge_vvm_i32m1. Every op in this body already
// has a conversion handler, so the whole body lowers with no new emit code; this
// test pins the two-compare composition structure. Byte identity to the legacy
// oracle is pinned by the Target/RVV runtime-scalar-dual-cmp-mask-and-select
// artifact fixtures.

module {
  weft.exec.kernel @rvv_dual_cmp_mask_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_dual_cmp_mask_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs_a = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "dual-cmp:cmp_lhs_a", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_a = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "dual-cmp:rhs_scalar_a", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = weft_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "dual-cmp:cmp_lhs_b", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar_b = weft_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "dual-cmp:rhs_scalar_b", role = "rhs-secondary-scalar-value"} : i32
      %true_value = weft_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "dual-cmp:true_value", role = "true-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %false_value = weft_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "dual-cmp:false_value", role = "false-value-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "dual-cmp:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "dual-cmp:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_dual_cmp_mask_select, sew = 32 : i64, source_kernel = "rvv_dual_cmp_mask_select_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_a_vec = weft_rvv.load %cmp_lhs_a, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_a_vec = weft_rvv.splat %rhs_scalar_a, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %cmp_lhs_b_vec = weft_rvv.load %cmp_lhs_b, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_b_vec = weft_rvv.splat %rhs_scalar_b, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %true_vec = weft_rvv.load %true_value, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %false_vec = weft_rvv.load %false_value, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask_a = weft_rvv.compare %cmp_lhs_a_vec, %rhs_a_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %mask_b = weft_rvv.compare %cmp_lhs_b_vec, %rhs_b_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %composed = weft_rvv.mask_and %mask_a, %mask_b, %vl {kind = "and"} : !weft_rvv.mask<i32, "m1">, !weft_rvv.mask<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %selected = weft_rvv.select %composed, %true_vec, %false_vec, %vl : !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %selected, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_dual_cmp_mask_select_kernel_rvv_dual_cmp_mask_select(
// Two runtime-scalar splats feed the two compares.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// First predicate mask from the first compare (vmsle, b32 predicate).
// CHECK: %[[MASKA:.*]] = call_opaque "__riscv_vmsle_vv_i32m1_b32"({{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">
// Second predicate mask from the second compare.
// CHECK: %[[MASKB:.*]] = call_opaque "__riscv_vmsle_vv_i32m1_b32"({{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">
// The two masks compose with vmand_mm_b32.
// CHECK: %[[COMP:.*]] = call_opaque "__riscv_vmand_mm_b32"(%[[MASKA]], %[[MASKB]], %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.opaque<"vbool32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">
// Select merges true/false under the composed mask.
// CHECK: call_opaque "__riscv_vmerge_vvm_i32m1"({{.*}}, %[[COMP]], %{{.*}})
// CHECK: call_opaque "__riscv_vse32_v_i32m1"
// CHECK: return
