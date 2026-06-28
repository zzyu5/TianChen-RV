// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// computed-mask multiply-accumulate beachhead body via a real MLIR
// DialectConversion. tcrv_rvv.compare lowers to the predicate intrinsic
// __riscv_vmslt_vv_i32m1_b32 producing a vbool32_t mask; tcrv_rvv.masked_macc
// lowers to the unmasked fused __riscv_vmacc_vv_i32m1(accumulator, lhs, rhs, vl)
// followed by a __riscv_vmerge_vvm_i32m1(accumulator, active, mask, vl) keeping
// the accumulator passthrough vector on inactive lanes. Byte-equivalence to the
// legacy materializer C is pinned by the e2e diff; this test asserts the emitc
// STRUCTURE and the passthrough = accumulator merge contract.

module {
  tcrv.exec.kernel @explicit_selected_body_computed_mask_macc_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_computed_mask_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:lhs-payload", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:rhs-payload", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-macc:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_computed_mask_macc_add, sew = 32 : i64, source_kernel = "explicit_selected_body_computed_mask_macc_add_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %sum = tcrv_rvv.masked_macc %mask, %lhs_vec, %rhs_vec, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_computed_mask_macc_add_kernel_explicit_selected_body_rvv_computed_mask_macc_add(
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"

// compare lhs / rhs and payload lhs / rhs and accumulator loads.
// CHECK: %[[CMPLHS:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[CMPRHS:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[LHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[RHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[ACCVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// compare: vmslt producing the vbool32_t predicate mask.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"(%[[CMPLHS]], %[[CMPRHS]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// masked_macc: unmasked fused vmacc (accumulator first), then merge keeping the
// accumulator passthrough vector on inactive lanes.
// CHECK: %[[ACTIVE:.*]] = call_opaque "__riscv_vmacc_vv_i32m1"(%[[ACCVEC]], %[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: %[[MERGED:.*]] = call_opaque "__riscv_vmerge_vvm_i32m1"(%[[ACCVEC]], %[[ACTIVE]], %[[MASK]], %[[BODYVL]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"vbool32_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[MERGED]], %[[BODYVL]])
// CHECK: return
