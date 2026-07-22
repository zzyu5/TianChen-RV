// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// COMPUTED-MASK segment2 STORE body (Stage 3 换心) via a real MLIR
// DialectConversion. A compare in the same VL scope produces the predicate, then
// the two payload fields are packed into a tuple and masked-stored to the
// interleaved destination:
//   weft_rvv.load x4 (cmp_lhs, cmp_rhs, field0, field1) -> __riscv_vle32_v_i32m1
//   weft_rvv.compare -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   weft_rvv.masked_segment2_store ->
//       __riscv_vcreate_v_i32m1x2(field0, field1)          (payload pack)
//       __riscv_vsseg2e32_v_i32m1x2_m(mask, dst+i*2, tuple, vl)
// The masked store consumes the COMPARE mask. Asserts STRUCTURE.

module {
  weft.exec.kernel @cmseg2_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_cmseg2_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %field0 = weft_rvv.load %src0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %field1 = weft_rvv.load %src1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        weft_rvv.masked_segment2_store %dst, %mask, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", inactive_lane_policy = "preserve-output-on-false-lanes", segment_count = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_cmseg2_store_kernel_rvv_cmseg2_store(

// compare produces the predicate mask.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"(%{{.*}}, %{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// pack the two payload fields, then masked interleaved store (_m).
// CHECK: %[[TUPLE:.*]] = call_opaque "__riscv_vcreate_v_i32m1x2"(%{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"vint32m1x2_t">
// CHECK: call_opaque "__riscv_vsseg2e32_v_i32m1x2_m"(%[[MASK]], %{{.*}}, %[[TUPLE]], %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vint32m1x2_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
