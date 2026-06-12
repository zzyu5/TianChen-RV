// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers a typed generic tcrv_rvv masked
// elementwise sub body at LMUL=m2 (SEW=32) via the same real MLIR
// DialectConversion. This pins the m2 extension of the TypeConverter: the
// !tcrv_rvv.vector<i32, "m2"> maps to emitc.opaque<"vint32m2_t"> and the
// !tcrv_rvv.mask<i32, "m2"> maps to emitc.opaque<"vbool16_t"> (maskbits =
// SEW/LMUL = 32/2 = 16). tcrv_rvv.compare lowers to __riscv_vmseq_vv_i32m2_b16;
// tcrv_rvv.masked_binary lowers to an unmasked __riscv_vsub_vv_i32m2 then a
// __riscv_vmerge_vvm_i32m2 keeping the passthrough vector on inactive lanes.
// Byte-equivalence to the legacy oracle is pinned by the e2e harness
// (PASS op=masked_lmul_m2_sub); this test asserts the emitc STRUCTURE.

module {
  tcrv.exec.kernel @explicit_selected_body_masked_lmul_m2_sub_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_masked_lmul_m2_sub attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_masked_lmul_m2_sub, sew = 32 : i64, source_kernel = "explicit_selected_body_masked_lmul_m2_sub_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        %mask = tcrv_rvv.compare %a, %b, %vl {kind = "eq"} : !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m2">
        %res = tcrv_rvv.masked_binary %mask, %a, %a, %b, %vl {kind = "sub"} : !tcrv_rvv.mask<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        tcrv_rvv.store %out, %res, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_explicit_selected_body_masked_lmul_m2_sub_kernel_explicit_selected_body_rvv_masked_lmul_m2_sub(
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step

// LMUL=m2 setvl mangles to e32m2.
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m2"

// m2 unit-stride loads produce vint32m2_t.
// CHECK: %[[LHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m2"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
// CHECK: %[[RHSVEC:.*]] = call_opaque "__riscv_vle32_v_i32m2"(%{{.*}}, %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">

// compare: vmseq with the m2-derived b16 predicate width producing vbool16_t.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmseq_vv_i32m2_b16"(%[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]]) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool16_t">

// masked_binary: unmasked sub, then merge with the passthrough vector (lhs).
// CHECK: %[[ACTIVE:.*]] = call_opaque "__riscv_vsub_vv_i32m2"(%[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]]) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint32m2_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">
// CHECK: %[[MERGED:.*]] = call_opaque "__riscv_vmerge_vvm_i32m2"(%[[LHSVEC]], %[[ACTIVE]], %[[MASK]], %[[BODYVL]]) : (!emitc.opaque<"vint32m2_t">, !emitc.opaque<"vint32m2_t">, !emitc.opaque<"vbool16_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m2_t">

// CHECK: call_opaque "__riscv_vse32_v_i32m2"(%{{.*}}, %[[MERGED]], %[[BODYVL]])
// CHECK: return
