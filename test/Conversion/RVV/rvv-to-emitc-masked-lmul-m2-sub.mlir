// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers a typed generic weft_rvv masked
// elementwise sub body at LMUL=m2 (SEW=32) via the same real MLIR
// DialectConversion. This pins the m2 extension of the TypeConverter: the
// !weft_rvv.vector<i32, "m2"> maps to emitc.opaque<"vint32m2_t"> and the
// !weft_rvv.mask<i32, "m2"> maps to emitc.opaque<"vbool16_t"> (maskbits =
// SEW/LMUL = 32/2 = 16). weft_rvv.compare lowers to __riscv_vmseq_vv_i32m2_b16;
// weft_rvv.masked_binary lowers to an unmasked __riscv_vsub_vv_i32m2 then a
// __riscv_vmerge_vvm_i32m2 keeping the passthrough vector on inactive lanes.
// Byte-equivalence to the legacy oracle is pinned by the e2e harness
// (PASS op=masked_lmul_m2_sub); this test asserts the emitc STRUCTURE.

module {
  weft.exec.kernel @explicit_selected_body_masked_lmul_m2_sub_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_masked_lmul_m2_sub attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-masked-lmul-m2-sub:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
        %mask = weft_rvv.compare %a, %b, %vl {kind = "eq"} : !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m2">
        %res = weft_rvv.masked_binary %mask, %a, %a, %b, %vl {kind = "sub"} : !weft_rvv.mask<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
        weft_rvv.store %out, %res, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_explicit_selected_body_masked_lmul_m2_sub_kernel_explicit_selected_body_rvv_masked_lmul_m2_sub(
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
