// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Stage 3 换心 — computed-mask widening dot-product reduction. A
// tcrv_rvv.compare produces a predicate mask (vmslt -> vbool32_t) that gates the
// tcrv_rvv.masked_widening_dot_reduce. The in-loop dataflow zeroes the inactive
// product lanes before the horizontal reduce:
//   vmv_v_x_i32m1(0, vl)                          // zero background (running vl)
//   vwmul_vv_i32m1_m(mask, lhs, rhs, vl)          // masked widened product
//   vmerge_vvm_i32m1(zero, masked_product, mask, vl)  // 0 on inactive lanes
//   running-seed read out[0]; vredsum_vs_i32m1_i32m1
// The converter REORDERS the compare to immediately follow its two i32 compare-
// input loads (BEFORE the two i16 dot-input loads) to keep the legacy
// mask-early emit order. Structure-level CHECKs; byte identity is pinned by the
// Target/RVV fixture.

module {
  tcrv.exec.kernel @rvv_cm_dot_reduce_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_cm_dot_reduce attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:lhs", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:rhs", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "cm-dot-reduce:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_cm_dot_reduce, sew = 32 : i64, source_kernel = "rvv_cm_dot_reduce_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %sum = tcrv_rvv.masked_widening_dot_reduce %mask, %a, %b, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_masked_widening_dot_reduce_add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_cm_dot_reduce_kernel_rvv_cm_dot_reduce(
// Pre-loop scalar seed out[0] = acc[0].
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vse32_v_i32m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The two i32 compare-input loads come first.
// CHECK: %[[CMPL:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[CMPR:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// The compare is REORDERED to immediately follow its compare-input loads,
// BEFORE the i16 dot-product input loads.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"(%[[CMPL]], %[[CMPR]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">
// CHECK: %[[DOTL:.*]] = call_opaque "__riscv_vle16_v_i16mf2"
// CHECK: %[[DOTR:.*]] = call_opaque "__riscv_vle16_v_i16mf2"
// Zero background (running vl), masked widened product, merge inactive lanes -> 0.
// CHECK: %[[ZERO:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"(%{{.*}}, %[[BODYVL]])
// CHECK: %[[MPROD:.*]] = call_opaque "__riscv_vwmul_vv_i32m1_m"(%[[MASK]], %[[DOTL]], %[[DOTR]], %[[BODYVL]]) : (!emitc.opaque<"vbool32_t">, !emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint16mf2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: %[[MERGED:.*]] = call_opaque "__riscv_vmerge_vvm_i32m1"(%[[ZERO]], %[[MPROD]], %[[MASK]], %[[BODYVL]])
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vredsum_vs_i32m1_i32m1"(%[[MERGED]], %[[SEED]], %[[BODYVL]])
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg5, %[[RED]],
// CHECK: return
